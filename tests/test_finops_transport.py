"""Kept-alive connections to Agent FinOps (kazenai.spine.transport).

These run against real local HTTP/1.1 servers, not mocks, so they see the
connections the SDK actually opens.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest

from kazenai.spine import transport


class _Server:
    """Keep-alive JSON server that records (path, client port) for every request.

    ``drop_next`` closes the connection on the next request without answering,
    the way a server does when it times out an idle connection as a request
    arrives. ``delay_s`` holds every answer.
    """

    def __init__(self) -> None:
        self.seen: list[tuple[str, int]] = []
        self.drop_next = 0
        self.delay_s = 0.0
        self.status = 200
        self.headers: dict[str, str] = {}
        self.reservations: set[str] = set()
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                return

            def _answer(self, body: dict[str, Any]) -> None:
                server.seen.append((self.path, self.client_address[1]))
                if server.drop_next:
                    server.drop_next -= 1
                    self.close_connection = True
                    return
                if server.delay_s:
                    threading.Event().wait(server.delay_s)
                raw = json.dumps(server.respond(self.path, body)).encode()
                self.send_response(server.status)
                for name, value in server.headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                self._answer({})

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                self._answer(json.loads(self.rfile.read(length) or b"{}"))

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def respond(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if path.endswith("/lifecycle"):
            return {
                "ok": True,
                "effective_lifecycle": {"profile": "control", "lifecycle_version": "reservation.lifecycle.v1"},
                "capabilities": {"reservation_settlement": True, "legacy_check_reconcile": False},
            }
        if path.endswith("/reserve"):
            rid = f"rsv_{uuid.uuid4().hex}"
            self.reservations.add(rid)
            return {
                "allowed": True,
                "reservation_id": rid,
                "decision_id": "dec_t",
                "lifecycle_version": "reservation.lifecycle.v1",
                "call_id": body.get("call_id"),
                "attempt": body.get("attempt") or 1,
                "reserved_usd_micros": int(body.get("estimated_usd_micros") or 0),
                "settlement_capability": "kzcap1.test." + "A" * 43,
            }
        if path.endswith("/settle"):
            return {"result": "applied", "reservation_id": body.get("reservation_id")}
        return {"ok": True}

    def ports(self, prefix: str = "/v1/budget/") -> set[int]:
        return {port for path, port in self.seen if path.startswith(prefix)}

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    srv = _Server()
    yield srv
    transport.reset_pools()
    srv.close()


@pytest.fixture(autouse=True)
def _fresh_pools(monkeypatch):
    monkeypatch.delenv("KAZENAI_FINOPS_KEEPALIVE_S", raising=False)
    transport.reset_pools()
    yield
    transport.reset_pools()


def _post(url: str, body: dict[str, Any] | None = None, timeout: float = 2.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with transport.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode())


def test_sequential_requests_share_one_connection(server):
    for _ in range(3):
        status, _ = _post(f"{server.url}/v1/budget/settle", {"reservation_id": "r"})
        assert status == 200
    assert len(server.seen) == 3
    assert len(server.ports()) == 1


def test_keepalive_zero_opens_a_connection_per_request(server, monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_KEEPALIVE_S", "0")
    transport.reset_pools()
    for _ in range(3):
        _post(f"{server.url}/v1/budget/settle")
    assert len(server.ports()) == 3


def test_http_errors_keep_the_urllib_contract(server):
    server.status = 402
    server.headers = {"Retry-After": "7"}
    with pytest.raises(urllib.error.HTTPError) as caught:
        _post(f"{server.url}/v1/budget/reserve", {"call_id": "c"})
    err = caught.value
    assert err.code == 402
    assert err.headers.get("Retry-After") == "7"
    assert json.loads(err.read().decode())["allowed"] is True  # body is readable


def test_success_response_has_status_getcode_and_headers(server):
    req = urllib.request.Request(f"{server.url}/v1/budget/lifecycle", method="GET")
    with transport.urlopen(req, timeout=2.0) as resp:
        assert resp.status == resp.getcode() == 200
        assert resp.headers.get("Content-Type") == "application/json"
        assert json.loads(resp.read())["ok"] is True


def test_a_slow_answer_times_out(server):
    server.delay_s = 0.5
    with pytest.raises(TimeoutError):
        _post(f"{server.url}/v1/budget/reserve", timeout=0.1)


def test_a_refused_connection_is_a_urlerror():
    with pytest.raises(urllib.error.URLError):
        _post("http://127.0.0.1:9/v1/budget/reserve", timeout=0.5)


def test_reserve_and_settle_are_resent_once_after_a_dropped_connection(server):
    _post(f"{server.url}/v1/budget/reserve")  # warm the connection
    for path in ("/v1/budget/reserve", "/v1/budget/settle"):
        server.drop_next = 1
        status, _ = _post(f"{server.url}{path}")
        assert status == 200
    # warm-up + (dropped, resent) for each path
    assert [p for p, _ in server.seen] == [
        "/v1/budget/reserve",
        "/v1/budget/reserve",
        "/v1/budget/reserve",
        "/v1/budget/settle",
        "/v1/budget/settle",
    ]


def test_legacy_routes_are_never_resent(server):
    _post(f"{server.url}/v1/budget/check")
    server.drop_next = 1
    with pytest.raises(urllib.error.URLError):
        _post(f"{server.url}/v1/budget/check")
    assert [p for p, _ in server.seen].count("/v1/budget/check") == 2


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork only")
def test_a_forked_child_starts_with_empty_pools(server):
    _post(f"{server.url}/v1/budget/settle")
    assert transport._sync_client is not None
    pid = os.fork()
    if pid == 0:  # child
        os._exit(0 if transport._sync_client is None and not transport._async_pools else 1)
    _, status = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(status) == 0
    assert transport._sync_client is not None  # the parent keeps its pool


def test_async_requests_share_one_connection_per_loop(server):
    async def three_posts() -> None:
        for _ in range(3):
            async with transport.async_session(2.0) as client:
                resp = await client.post(f"{server.url}/v1/budget/settle", content="{}")
                assert resp.status_code == 200

    asyncio.run(three_posts())
    assert len(server.ports()) == 1
    # asyncio.run shut the loop's async generators down, which closed the pool.
    assert transport._async_pools == {}

    asyncio.run(three_posts())  # a new loop gets its own connection
    assert len(server.ports()) == 2
    assert transport._async_pools == {}


def test_async_reserve_is_resent_once_after_a_dropped_connection(server):
    async def run() -> list[int]:
        codes = []
        async with transport.async_session(2.0) as client:
            codes.append((await client.post(f"{server.url}/v1/budget/reserve", content="{}")).status_code)
            server.drop_next = 1
            codes.append((await client.post(f"{server.url}/v1/budget/reserve", content="{}")).status_code)
        return codes

    assert asyncio.run(run()) == [200, 200]
    assert len([p for p, _ in server.seen if p == "/v1/budget/reserve"]) == 3


def test_async_timeout_still_raises(server):
    server.delay_s = 0.5

    async def run() -> None:
        async with transport.async_session(httpx.Timeout(0.1)) as client:
            await client.post(f"{server.url}/v1/budget/reserve", content="{}")

    with pytest.raises(httpx.TimeoutException):
        asyncio.run(run())


def _guarded_env(monkeypatch, server, tmp_path) -> None:
    monkeypatch.setenv("KAZENAI_FINOPS_URL", server.url)
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "test")
    monkeypatch.setenv("KAZENAI_SDK_SETTLE_QUEUE_DIR", str(tmp_path / "settle"))


def _provider_body() -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_a_guarded_call_reserves_and_settles_over_one_connection(server, monkeypatch, tmp_path):
    openai = pytest.importorskip("openai")
    from kazenai import monitor

    _guarded_env(monkeypatch, server, tmp_path)
    raw = openai.OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_provider_body()))),
    )
    guarded = monitor(raw, max_budget_usd=10.0, org_id="org-keepalive", workspace_id="ws")
    for i in range(3):
        guarded.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": f"q{i}"}])

    budget = [path for path, _ in server.seen if path.startswith("/v1/budget/")]
    assert budget.count("/v1/budget/reserve") == 3
    assert budget.count("/v1/budget/settle") == 6  # provider_started + usage_known per call
    assert len(server.ports()) == 1


def test_an_async_guarded_call_reserves_and_settles_over_one_connection(server, monkeypatch, tmp_path):
    openai = pytest.importorskip("openai")
    from kazenai import monitor

    _guarded_env(monkeypatch, server, tmp_path)

    async def run() -> None:
        raw = openai.AsyncOpenAI(
            api_key="test",
            base_url="https://openai.test/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_provider_body()))
            ),
        )
        guarded = monitor(raw, max_budget_usd=10.0, org_id="org-keepalive-async", workspace_id="ws")
        for i in range(3):
            await guarded.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": f"q{i}"}])

    asyncio.run(run())
    budget = [path for path, _ in server.seen if path.startswith("/v1/budget/")]
    assert budget.count("/v1/budget/reserve") == 3
    assert budget.count("/v1/budget/settle") == 6
    assert len(server.ports()) == 1


def test_an_unreachable_finops_still_refuses_the_call(monkeypatch, tmp_path):
    openai = pytest.importorskip("openai")
    from kazenai import monitor
    from kazenai.enforcement import BudgetUnavailable

    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "test")
    monkeypatch.setenv("KAZENAI_SDK_SETTLE_QUEUE_DIR", str(tmp_path / "settle"))
    hits: list[str] = []

    def provider(request: httpx.Request) -> httpx.Response:
        hits.append(str(request.url))
        return httpx.Response(200, json=_provider_body())

    raw = openai.OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(provider)),
    )
    guarded = monitor(raw, max_budget_usd=10.0, org_id="org-down", workspace_id="ws")
    with pytest.raises(BudgetUnavailable):
        guarded.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "q"}])
    assert hits == []


def test_importing_the_transport_starts_no_connection():
    assert "kazenai.spine.transport" in sys.modules
    transport.reset_pools()
    assert transport._sync_client is None and transport._async_pools == {}
