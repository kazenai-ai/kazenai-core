"""Where the SDK's keys may go, and what an unknown cost looks like.

- Requests that carry a key never follow a redirect (urllib would copy the
  ``Authorization`` or ``X-API-Key`` header to the redirect target).
- In staging and production the ingest URL must be public HTTPS, including after
  DNS resolution.
- A model call whose cost is unknown is emitted with ``cost_usd=None``, never $0.
"""

from __future__ import annotations

import http.server
import threading
import urllib.error
import urllib.request
from typing import Any, Iterator

import pytest

from kazenai import outbound_http
from kazenai.ingest_url import split_ingest_url
from kazenai.sinks import HttpSink, HttpSinkConfig, MemorySink, _check_ingest_url_ssrf


class _Server:
    """A local HTTP server that records requests and answers with a fixed status."""

    def __init__(self, status: int, location: str = "") -> None:
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _answer(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                outer.requests.append({"path": self.path, "method": self.command, "headers": dict(self.headers)})
                self.send_response(status)
                if location:
                    self.send_header("Location", location)
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_GET = do_POST = _answer

            def log_message(self, *args: Any) -> None:  # noqa: ARG002
                return

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def redirect_pair() -> Iterator[tuple[_Server, _Server]]:
    target = _Server(200)
    origin = _Server(302, location=f"{target.url}/collect")
    try:
        yield origin, target
    finally:
        origin.close()
        target.close()


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, method="GET", headers={"Authorization": "Bearer kz_test_secret", "X-API-Key": "kz_test_secret"})


def test_plain_urllib_would_forward_the_key_to_the_redirect_target(redirect_pair) -> None:
    # Why the helper exists: the standard opener follows and copies the key.
    origin, target = redirect_pair
    with urllib.request.urlopen(_request(f"{origin.url}/v1/pricing/catalog"), timeout=5) as resp:
        assert resp.status == 200
    assert target.requests and target.requests[0]["headers"].get("Authorization") == "Bearer kz_test_secret"


def test_outbound_http_refuses_the_redirect(redirect_pair) -> None:
    origin, target = redirect_pair
    with pytest.raises(urllib.error.HTTPError) as caught:
        outbound_http.urlopen(_request(f"{origin.url}/v1/pricing/catalog"), timeout=5)
    assert caught.value.code == 302
    assert target.requests == []


def test_http_sink_does_not_follow_a_redirect_with_the_key(redirect_pair, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.delenv("KAZENAI_ENV", raising=False)
    origin, target = redirect_pair
    sink = HttpSink(
        HttpSinkConfig(
            ingest_url=origin.url,
            api_key="kz_test_secret",
            offline_queue_path=str(tmp_path / "spool.jsonl"),
            start_workers=False,
        )
    )
    assert sink._post_batch([{"event_type": "model.call", "event_id": "e1"}]) is False
    assert [r["path"] for r in origin.requests] == ["/v1/events"]
    assert target.requests == []


@pytest.fixture()
def production(monkeypatch) -> None:
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "production")
    for name in ("KAZENAI_ENV", "KAZENAI_FINOPS_ENV", "KAZENAI_AGENTLENS_ENV"):
        monkeypatch.delenv(name, raising=False)


def _resolves_to(monkeypatch, *addresses: str) -> None:
    import kazenai.sinks as sinks

    def fake(host, port, type=0):  # noqa: A002, ARG001
        return [(2, 1, 6, "", (a, port)) for a in addresses]

    monkeypatch.setattr(sinks.socket, "getaddrinfo", fake)


@pytest.mark.parametrize(
    "url",
    [
        "http://finops.example.com",
        "https://10.0.0.5",
        "https://127.0.0.1:8090",
        "https://[::1]",
        "https://[::ffff:10.0.0.1]",
        "https://100.64.0.10",
        "https://169.254.169.254",
        "https://localhost:8090",
        "https://api.localhost",
        "https://user:pw@finops.example.com",
    ],
)
def test_production_refuses_non_public_or_plain_http(production, monkeypatch, url: str) -> None:
    _resolves_to(monkeypatch, "93.184.216.34")
    with pytest.raises(ValueError):
        _check_ingest_url_ssrf(url)


def test_production_refuses_a_name_that_resolves_to_a_private_address(production, monkeypatch) -> None:
    _resolves_to(monkeypatch, "93.184.216.34", "10.1.2.3")
    with pytest.raises(ValueError, match="private/loopback"):
        _check_ingest_url_ssrf("https://finops.example.com")


def test_production_accepts_a_public_https_name(production, monkeypatch) -> None:
    _resolves_to(monkeypatch, "93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946")
    _check_ingest_url_ssrf("https://finops.staging.kazenai.com")


def test_unanswered_dns_does_not_block_startup(production, monkeypatch) -> None:
    import socket

    import kazenai.sinks as sinks

    def fail(*args, **kwargs):  # noqa: ARG001
        raise socket.gaierror("no answer")

    monkeypatch.setattr(sinks.socket, "getaddrinfo", fail)
    _check_ingest_url_ssrf("https://finops.example.com")


def test_development_keeps_local_urls(monkeypatch) -> None:
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.delenv("KAZENAI_ENV", raising=False)
    _check_ingest_url_ssrf("http://localhost:8090")
    _check_ingest_url_ssrf("http://10.0.0.5:8090")


def test_split_ingest_url_returns_a_pair() -> None:
    assert split_ingest_url("") == ("", "")
    assert split_ingest_url("   ") == ("", "")
    assert split_ingest_url("https://finops.example.com/v1/events") == (
        "https://finops.example.com",
        "https://finops.example.com/v1/events",
    )


@pytest.mark.parametrize("cost, expected", [(None, None), (0.0, 0.0), (0.000125, 0.000125)])
def test_model_event_keeps_an_unknown_cost_unknown(cost, expected) -> None:
    from kazenai.spine.guard import _guard_model_event

    sink = MemorySink()
    _guard_model_event(
        ids=("org_1", "ws_1", "run_1", "agent_1", None, "step_1"),
        parent_step_id=None,
        agent_role="worker",
        surface="openai.chat",
        model="gpt-6-luna",
        feature="default",
        tokens_used=None if cost is None else 42,
        cost_usd=cost,
        elapsed_ms=12.5,
        event_sink=sink,
    )
    (event,) = sink.events
    assert event.event_type == "model.call"
    assert event.cost_usd == expected
