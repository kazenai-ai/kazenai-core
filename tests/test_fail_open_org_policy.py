"""An organization that chose fail-open keeps working while Agent FinOps is unreachable.

The setting comes from Agent FinOps (``org_policy.when_unreachable`` in the
capabilities document) and is remembered in memory and on disk. It applies only
when FinOps does not answer: no connection, a timeout, or HTTP 408/429/5xx.
An answer (402 deny, tier breaker, 401 bad key, a malformed body) still refuses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import stat
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest

from kazenai import monitor
from kazenai.enforcement import BudgetExceeded, BudgetUnavailable
from kazenai.spine import guard, org_policy, transport

openai = pytest.importorskip("openai")

_CHAT = {
    "id": "c1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}
_CHUNK = '{"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":%s%s}'
_SSE = "".join(
    f"data: {line}\n\n"
    for line in (
        _CHUNK % ('[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]', ""),
        _CHUNK % ('[{"index":0,"delta":{},"finish_reason":"stop"}]', ""),
        _CHUNK % ("[]", ',"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}'),
        "[DONE]",
    )
)


class FinOps:
    """Local Agent FinOps stand-in. ``policy`` is the org's setting; ``reserve_status``
    makes reserve answer with that status; ``stop()`` makes it unreachable."""

    def __init__(self, policy: str | None = "fail_open", org_id: str = "org-fo") -> None:
        self.policy = policy
        self.org_id = org_id
        self.reserve_status = 200
        self.reserve_body: Any = None
        self.reserves = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return

            def _send(self, status: int, payload: Any) -> None:
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                body: dict[str, Any] = {
                    "ok": True,
                    "effective_lifecycle": {"profile": "control", "lifecycle_version": "reservation.lifecycle.v1"},
                    "capabilities": {"reservation_settlement": True, "legacy_check_reconcile": False},
                }
                if server.policy is not None:
                    body["org_policy"] = {"org_id": server.org_id, "when_unreachable": server.policy}
                self._send(200, body)

            def do_POST(self) -> None:  # noqa: N802
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if self.path.endswith("/reserve"):
                    server.reserves += 1
                    if server.reserve_body is not None:
                        return self._send(server.reserve_status, server.reserve_body)
                    if server.reserve_status != 200:
                        return self._send(server.reserve_status, {"detail": {"error_code": "x", "reason": "x"}})
                    return self._send(200, {
                        "allowed": True,
                        "reservation_id": f"rsv_{uuid.uuid4().hex}",
                        "decision_id": "dec_1",
                        "lifecycle_version": "reservation.lifecycle.v1",
                        "call_id": req.get("call_id"),
                        "attempt": req.get("attempt") or 1,
                        "reserved_usd_micros": int(req.get("estimated_usd_micros") or 0),
                        "settlement_capability": "kzcap1.test." + "A" * 43,
                    })
                if self.path.endswith("/settle"):
                    return self._send(200, {"result": "applied"})
                return self._send(200, {"ok": True})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        transport.reset_pools()


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setenv("KAZENAI_SDK_SETTLE_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "test")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVE_TIMEOUT_S", "0.5")
    org_policy.reset_for_tests()
    guard._LIFECYCLE_CAP_CACHE.clear()
    transport.reset_pools()
    yield
    org_policy.reset_for_tests()
    guard._LIFECYCLE_CAP_CACHE.clear()
    transport.reset_pools()


def _client(monkeypatch, finops: FinOps, hits: list, *, org: str = "org-fo", budget: float = 5.0, is_async: bool = False):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", finops.url)

    def provider(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.path)
        if json.loads(request.content or b"{}").get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_SSE.encode())
        return httpx.Response(200, json=_CHAT)

    if is_async:
        raw = openai.AsyncOpenAI(api_key="t", base_url="https://openai.test/v1", max_retries=0,
                                 http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)))
    else:
        raw = openai.OpenAI(api_key="t", base_url="https://openai.test/v1", max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(provider)))
    return monitor(raw, org_id=org, workspace_id="ws", agent_id=f"fo-{uuid.uuid4().hex[:6]}", max_budget_usd=budget)


def _ask(client, text: str) -> str:
    return client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": text}]).choices[0].message.content


def _learn_then_go_down(monkeypatch, finops: FinOps, hits: list, **kw):
    client = _client(monkeypatch, finops, hits, **kw)
    assert _ask(client, f"learn {uuid.uuid4().hex}") == "ok"  # reachable: normal reserve, setting learned
    assert finops.reserves == 1
    finops.stop()
    guard._LIFECYCLE_CAP_CACHE.clear()
    return client


def test_fail_open_org_keeps_working_while_finops_is_down(monkeypatch, caplog):
    finops, hits = FinOps("fail_open"), []
    client = _learn_then_go_down(monkeypatch, finops, hits)
    with caplog.at_level(logging.WARNING, logger="kazenai.spine"):
        assert _ask(client, f"down {uuid.uuid4().hex}") == "ok"
        assert _ask(client, f"down again {uuid.uuid4().hex}") == "ok"
    assert len(hits) == 3
    assert "fail-open" in caplog.text and "not charged" in caplog.text
    assert caplog.text.count("fail-open setting") == 1  # throttled to one line per minute


def test_fail_closed_org_is_refused_while_finops_is_down(monkeypatch):
    finops, hits = FinOps("fail_closed"), []
    client = _learn_then_go_down(monkeypatch, finops, hits)
    with pytest.raises(BudgetUnavailable):
        _ask(client, f"down {uuid.uuid4().hex}")
    assert len(hits) == 1


def test_an_org_whose_setting_was_never_seen_fails_closed(monkeypatch):
    finops, hits = FinOps(None), []  # older server: no org_policy block
    client = _learn_then_go_down(monkeypatch, finops, hits)
    with pytest.raises(BudgetUnavailable):
        _ask(client, f"down {uuid.uuid4().hex}")
    assert len(hits) == 1


def test_the_setting_survives_a_restart_on_disk(monkeypatch, tmp_path):
    finops, hits = FinOps("fail_open"), []
    client = _learn_then_go_down(monkeypatch, finops, hits)
    org_policy.reset_for_tests()  # a new process: memory is empty, the disk copy remains
    files = list((tmp_path / "queue" / "org-policy").glob("*.json"))
    assert len(files) == 1
    assert stat.S_IMODE(os.stat(files[0]).st_mode) == 0o600
    assert _ask(client, f"after restart {uuid.uuid4().hex}") == "ok"
    assert len(hits) == 2


def test_switching_back_to_fail_closed_takes_effect(monkeypatch):
    finops, hits = FinOps("fail_open"), []
    client = _client(monkeypatch, finops, hits)
    assert _ask(client, f"open {uuid.uuid4().hex}") == "ok"
    finops.policy = "fail_closed"
    guard._LIFECYCLE_CAP_CACHE.clear()
    assert _ask(client, f"closed {uuid.uuid4().hex}") == "ok"  # learns the new setting
    finops.stop()
    guard._LIFECYCLE_CAP_CACHE.clear()
    with pytest.raises(BudgetUnavailable):
        _ask(client, f"down {uuid.uuid4().hex}")
    assert len(hits) == 2


@pytest.mark.parametrize("status", [503, 500, 429, 408])
def test_availability_statuses_fail_open(monkeypatch, status):
    finops, hits = FinOps("fail_open"), []
    client = _client(monkeypatch, finops, hits)
    assert _ask(client, f"learn {uuid.uuid4().hex}") == "ok"
    finops.reserve_status = status
    assert _ask(client, f"status {status} {uuid.uuid4().hex}") == "ok"
    assert len(hits) == 2
    finops.stop()


@pytest.mark.parametrize("status", [401, 403, 400, 422])
def test_answers_that_are_not_outages_still_refuse(monkeypatch, status):
    finops, hits = FinOps("fail_open"), []
    client = _client(monkeypatch, finops, hits)
    assert _ask(client, f"learn {uuid.uuid4().hex}") == "ok"
    finops.reserve_status = status
    with pytest.raises(BudgetUnavailable):
        _ask(client, f"status {status} {uuid.uuid4().hex}")
    assert len(hits) == 1
    finops.stop()


def test_a_budget_denial_still_denies_a_fail_open_org(monkeypatch):
    finops, hits = FinOps("fail_open"), []
    client = _client(monkeypatch, finops, hits)
    assert _ask(client, f"learn {uuid.uuid4().hex}") == "ok"
    finops.reserve_status, finops.reserve_body = 402, {"detail": {"allowed": False, "reason": "over limit"}}
    with pytest.raises(BudgetExceeded):
        _ask(client, f"over {uuid.uuid4().hex}")
    finops.reserve_status, finops.reserve_body = 409, {"detail": {"error_code": "tier_breaker", "reason": "open"}}
    with pytest.raises(BudgetExceeded):
        _ask(client, f"breaker {uuid.uuid4().hex}")
    # Over the per-minute reservation limit Agent FinOps refuses with 402, not 429:
    # a 429 would read as an outage here and the call would run unchecked.
    finops.reserve_status, finops.reserve_body = 402, {
        "detail": {
            "allowed": False,
            "error_code": "rate_limited",
            "scope": "org",
            "reason": "org reservation rate limit exceeded (>3000 reservations/min)",
            "retry_after_s": 12,
        }
    }
    with pytest.raises(BudgetExceeded, match="reservation rate limit exceeded"):
        _ask(client, f"rate limited {uuid.uuid4().hex}")
    assert len(hits) == 1
    finops.stop()


def test_a_malformed_answer_is_not_an_outage(monkeypatch):
    finops, hits = FinOps("fail_open"), []
    client = _client(monkeypatch, finops, hits)
    assert _ask(client, f"learn {uuid.uuid4().hex}") == "ok"
    finops.reserve_status, finops.reserve_body = 200, b"not json"
    with pytest.raises(BudgetUnavailable):
        _ask(client, f"garbled {uuid.uuid4().hex}")
    assert len(hits) == 1
    finops.stop()


def test_the_local_cap_still_applies_under_fail_open(monkeypatch):
    finops, hits = FinOps("fail_open"), []
    client = _learn_then_go_down(monkeypatch, finops, hits, budget=0.015)
    with pytest.raises(BudgetExceeded):
        # About 120K input tokens: a projected $0.028 against the $0.015 local cap.
        _ask(client, f"over the local cap {uuid.uuid4().hex} " + "x" * 120_000)
    assert len(hits) == 1


def test_one_orgs_setting_does_not_apply_to_another(monkeypatch):
    finops, hits = FinOps("fail_open"), []
    _learn_then_go_down(monkeypatch, finops, hits, org="org-fo")
    other = _client(monkeypatch, finops, hits, org="org-other")
    with pytest.raises(BudgetUnavailable):
        _ask(other, f"other org {uuid.uuid4().hex}")
    assert len(hits) == 1


def test_async_fail_open_while_finops_is_down(monkeypatch):
    finops, hits = FinOps("fail_open"), []
    _learn_then_go_down(monkeypatch, finops, hits)

    async def run() -> str:
        client = _client(monkeypatch, finops, hits, is_async=True)
        out = await client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": f"async down {uuid.uuid4().hex}"}]
        )
        return out.choices[0].message.content

    assert asyncio.run(run()) == "ok"
    assert len(hits) == 2


def test_async_fail_closed_while_finops_is_down(monkeypatch):
    finops, hits = FinOps("fail_closed"), []
    _learn_then_go_down(monkeypatch, finops, hits)

    async def run() -> None:
        client = _client(monkeypatch, finops, hits, is_async=True)
        await client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": f"async down {uuid.uuid4().hex}"}]
        )

    with pytest.raises(BudgetUnavailable):
        asyncio.run(run())
    assert len(hits) == 1


def test_another_api_key_on_the_same_machine_does_not_share_the_setting(monkeypatch, tmp_path):
    # Two API keys on one machine declaring the same org_id (for example a misconfigured app):
    # the setting a process acts on is always the one Agent FinOps sent for its own key.
    finops, hits = FinOps("fail_open", org_id="org-a"), []
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "key-of-org-a")
    _learn_then_go_down(monkeypatch, finops, hits, org="local")
    org_policy.reset_for_tests()  # a second process: only the disk copy is shared
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "key-of-org-b")
    other = _client(monkeypatch, finops, hits, org="local")
    with pytest.raises(BudgetUnavailable):
        _ask(other, f"org b while down {uuid.uuid4().hex}")
    assert len(hits) == 1
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "key-of-org-a")
    assert _ask(other, f"org a while down {uuid.uuid4().hex}") == "ok"
    assert len(hits) == 2
    stored = [json.loads(f.read_text()) for f in (tmp_path / "queue" / "org-policy").glob("*.json")]
    assert stored == [{"when_unreachable": "fail_open", "org_id": "org-a", "seen_at": stored[0]["seen_at"]}]
    assert not any("key-of-org" in f.read_text() for f in (tmp_path / "queue" / "org-policy").glob("*"))


def test_the_warning_names_the_org_agent_finops_reported(monkeypatch, caplog):
    finops, hits = FinOps("fail_open", org_id="org_7f3a"), []
    client = _learn_then_go_down(monkeypatch, finops, hits, org="local")
    with caplog.at_level(logging.WARNING, logger="kazenai.spine"):
        assert _ask(client, f"down {uuid.uuid4().hex}") == "ok"
    assert "for org org_7f3a ran under" in caplog.text


def test_a_stream_runs_under_fail_open_and_sends_nothing_to_settle(monkeypatch):
    finops, hits = FinOps("fail_open"), []
    client = _learn_then_go_down(monkeypatch, finops, hits)
    stream = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": f"stream down {uuid.uuid4().hex}"}], stream=True
    )
    text = "".join((c.choices[0].delta.content or "") for c in stream if c.choices)
    assert text == "ok"
    assert len(hits) == 2
    assert _queued(os.environ["KAZENAI_SDK_SETTLE_QUEUE_DIR"]) == []


def test_an_async_stream_runs_under_fail_open(monkeypatch):
    finops, hits = FinOps("fail_open"), []
    _learn_then_go_down(monkeypatch, finops, hits)

    async def run() -> str:
        client = _client(monkeypatch, finops, hits, is_async=True)
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"async stream down {uuid.uuid4().hex}"}],
            stream=True,
        )
        return "".join([(c.choices[0].delta.content or "") async for c in stream if c.choices])

    assert asyncio.run(run()) == "ok"
    assert len(hits) == 2
    assert _queued(os.environ["KAZENAI_SDK_SETTLE_QUEUE_DIR"]) == []


def _queued(root: str) -> list[str]:
    """Settle intents waiting on disk (the org-policy folder is not one)."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "org-policy"]
        found += [os.path.join(dirpath, f) for f in filenames if not f.endswith(".lock")]
    return found
