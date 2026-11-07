"""SDK 1.2.4: the settle after a call is sent in the background (the default).

FinOps holds the reserved amount until usage_known lands, so sending it after the
response reaches the app never weakens the limit. The intent is journaled before the
call returns; a failed send stays journaled for the flusher; ``drain_settlements``
waits for the backlog; ``KAZENAI_SETTLE_IN_BACKGROUND=0`` keeps the old inline send.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import httpx
import pytest

from fast_finops import FastFinOps
from kazenai import monitor
from kazenai.spine import guard, org_policy, settle_queue, transport

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


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.delenv("KAZENAI_SETTLE_IN_BACKGROUND", raising=False)  # the default
    monkeypatch.setenv("KAZENAI_SDK_SETTLE_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "test")
    monkeypatch.setenv("KAZENAI_FINOPS_CONTROL_PROFILE", "1")
    settle_queue.reset_settle_queue_for_tests(str(tmp_path / "queue"))
    org_policy.reset_for_tests()
    guard._LIFECYCLE_CAP_CACHE.clear()
    transport.reset_pools()
    yield
    guard.drain_settlements(5.0)
    org_policy.reset_for_tests()
    guard._LIFECYCLE_CAP_CACHE.clear()
    transport.reset_pools()


def _client(monkeypatch, finops: FastFinOps, *, is_async: bool = False):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", finops.url)

    def provider(request: httpx.Request) -> httpx.Response:
        import json

        if json.loads(request.content or b"{}").get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_SSE.encode())
        return httpx.Response(200, json=_CHAT)

    if is_async:
        raw = openai.AsyncOpenAI(api_key="t", base_url="https://openai.test/v1", max_retries=0,
                                 http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)))
    else:
        raw = openai.OpenAI(api_key="t", base_url="https://openai.test/v1", max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(provider)))
    return monitor(raw, org_id="org-bg", workspace_id="ws", agent_id=f"bg-{uuid.uuid4().hex[:6]}", max_budget_usd=5.0)


def _ask(client, **kw):
    return client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": f"q {uuid.uuid4().hex}"}], **kw
    )


def _pending() -> int:
    return int(settle_queue.queue_stats().get("depth") or 0)


def test_the_call_returns_before_its_settle_is_sent(monkeypatch):
    finops = FastFinOps()
    finops.slow["usage_known"] = 1.0
    try:
        client = _client(monkeypatch, finops)
        started = time.monotonic()
        assert _ask(client).choices[0].message.content == "ok"
        assert time.monotonic() - started < 0.6
        assert _pending() == 1  # journaled before the call returned
        assert guard.drain_settlements(5.0)
        assert len(finops.settles("usage_known")) == 1
        assert _pending() == 0  # acknowledged, so removed
    finally:
        finops.stop()


def test_the_call_returns_before_its_settle_is_sent_async(monkeypatch):
    finops = FastFinOps()
    finops.slow["usage_known"] = 1.0
    try:
        client = _client(monkeypatch, finops, is_async=True)

        async def one():
            started = time.monotonic()
            resp = await _ask(client)
            return resp.choices[0].message.content, time.monotonic() - started

        content, took = asyncio.run(one())
        assert content == "ok" and took < 0.6
        assert guard.drain_settlements(5.0)
        assert len(finops.settles("usage_known")) == 1
        assert _pending() == 0
    finally:
        finops.stop()


def test_a_streamed_call_settles_in_the_background_too(monkeypatch):
    finops = FastFinOps()
    finops.slow["usage_known"] = 1.0
    try:
        client = _client(monkeypatch, finops)
        started = time.monotonic()
        text = "".join(
            (chunk.choices[0].delta.content or "") if chunk.choices else ""
            for chunk in _ask(client, stream=True, stream_options={"include_usage": True})
        )
        assert text == "ok" and time.monotonic() - started < 0.6
        assert guard.drain_settlements(5.0)
        assert len(finops.settles("usage_known")) == 1
    finally:
        finops.stop()


def test_a_failed_background_send_stays_journaled_for_the_flusher(monkeypatch):
    finops = FastFinOps()
    finops.settle_status = 503
    try:
        client = _client(monkeypatch, finops)
        assert _ask(client).choices[0].message.content == "ok"
        assert guard.drain_settlements(5.0)
        assert _pending() == 1  # due now for the flusher
        finops.settle_status = 200
        assert settle_queue.flush_pending_settlements(default_base_url=finops.url)["applied"] == 1
        assert _pending() == 0
    finally:
        finops.stop()


def test_turning_it_off_settles_inline(monkeypatch):
    monkeypatch.setenv("KAZENAI_SETTLE_IN_BACKGROUND", "0")
    finops = FastFinOps()
    finops.slow["usage_known"] = 0.8
    try:
        client = _client(monkeypatch, finops)
        started = time.monotonic()
        assert _ask(client).choices[0].message.content == "ok"
        assert time.monotonic() - started >= 0.8
        assert len(finops.settles("usage_known")) == 1
    finally:
        finops.stop()


def test_drain_reports_a_backlog_it_could_not_finish(monkeypatch):
    finops = FastFinOps()
    finops.slow["usage_known"] = 1.5
    try:
        client = _client(monkeypatch, finops)
        _ask(client)
        assert guard.drain_settlements(0.2) is False
        assert guard.drain_settlements(5.0) is True
    finally:
        finops.stop()
