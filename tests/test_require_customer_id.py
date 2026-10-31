"""monitor(..., require_customer_id=True): a call without a customer ID is refused
before it reserves or reaches the provider (kazenai 1.2.1)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import json
from typing import Any

import httpx
import pytest

import kazenai
from kazenai import CustomerIdRequired, KazenBlock, monitor, use_attribution

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")

OPENAI_JSON = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

OPENAI_SSE = """data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":25,"completion_tokens":15,"total_tokens":40}}

data: [DONE]

"""

ANTHROPIC_JSON = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello"}],
    "model": "claude-haiku-4-5",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}

ANTHROPIC_SSE = """event: message_start
data: {"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-haiku-4-5","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":15}}

event: message_stop
data: {"type":"message_stop"}

"""

MESSAGES = [{"role": "user", "content": "Hello"}]


@pytest.fixture(autouse=True)
def _standalone(monkeypatch):
    for key in (
        "KAZENAI_FINOPS_URL",
        "KAZENAI_FINOPS_INGEST_URL",
        "KAZENAI_INGEST_URL",
        "KAZENAI_DENY_STREAMING",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_CONTROL_PROFILE", "1")


def _handler(seen: list[Any], *, sse: str | None = None, body: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        seen.append(payload)
        if payload.get("stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(sse or "").encode(),
                request=request,
            )
        return httpx.Response(200, json=body, request=request)

    return handler


def _openai(seen: list[Any], **kw: Any):
    return monitor(
        openai.OpenAI(
            api_key="test",
            base_url="https://openai.test/v1",
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(_handler(seen, sse=OPENAI_SSE, body=OPENAI_JSON))
            ),
        ),
        max_budget_usd=5.0,
        **kw,
    )


def _async_openai(seen: list[Any], **kw: Any):
    return monitor(
        openai.AsyncOpenAI(
            api_key="test",
            base_url="https://openai.test/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(_handler(seen, sse=OPENAI_SSE, body=OPENAI_JSON))
            ),
        ),
        max_budget_usd=5.0,
        **kw,
    )


def _anthropic(seen: list[Any], **kw: Any):
    return monitor(
        anthropic.Anthropic(
            api_key="test",
            base_url="https://anthropic.test",
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(_handler(seen, sse=ANTHROPIC_SSE, body=ANTHROPIC_JSON))
            ),
        ),
        max_budget_usd=5.0,
        **kw,
    )


def _async_anthropic(seen: list[Any], **kw: Any):
    return monitor(
        anthropic.AsyncAnthropic(
            api_key="test",
            base_url="https://anthropic.test",
            max_retries=0,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(_handler(seen, sse=ANTHROPIC_SSE, body=ANTHROPIC_JSON))
            ),
        ),
        max_budget_usd=5.0,
        **kw,
    )


def _chat(client, text: str = "Hello"):
    return client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": text}]
    )


def test_error_is_a_public_kazen_block():
    assert issubclass(CustomerIdRequired, KazenBlock)
    assert "CustomerIdRequired" in kazenai.__all__
    assert CustomerIdRequired.error_code == "customer_id_required"


def test_off_by_default_a_call_without_a_customer_still_runs():
    seen: list[Any] = []
    resp = _chat(_openai(seen))
    assert resp.usage.total_tokens == 15
    assert len(seen) == 1


def test_no_customer_is_refused_before_the_provider():
    seen: list[Any] = []
    client = _openai(seen, require_customer_id=True)
    with pytest.raises(CustomerIdRequired) as exc:
        _chat(client)
    assert "use_attribution" in str(exc.value)
    assert seen == []


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_customer_counts_as_missing(blank):
    seen: list[Any] = []
    client = _openai(seen, require_customer_id=True)
    with use_attribution(business_subject_ref=blank):
        with pytest.raises(CustomerIdRequired):
            _chat(client)
    assert seen == []


def test_use_attribution_names_the_customer():
    seen: list[Any] = []
    client = _openai(seen, require_customer_id=True)
    with use_attribution(business_subject_ref="cust_123"):
        assert _chat(client).usage.total_tokens == 15
    assert len(seen) == 1


def test_monitor_kwarg_names_the_customer():
    seen: list[Any] = []
    client = _openai(seen, require_customer_id=True, business_subject_ref="cust_123")
    assert _chat(client).usage.total_tokens == 15
    assert len(seen) == 1


def test_thread_pool_worker_without_context_is_refused():
    seen: list[Any] = []
    client = _openai(seen, require_customer_id=True)
    with use_attribution(business_subject_ref="cust_123"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_chat, client)
            with pytest.raises(CustomerIdRequired):
                future.result()
    assert seen == []


def test_thread_pool_worker_with_copied_context_runs():
    seen: list[Any] = []
    client = _openai(seen, require_customer_id=True)
    with use_attribution(business_subject_ref="cust_123"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            # A fresh copy per task: one Context can't be entered by two threads at once.
            # Distinct prompts keep the SDK's loop guard out of the way.
            futures = [
                pool.submit(contextvars.copy_context().run, _chat, client, f"Ticket {i}")
                for i in range(2)
            ]
            for future in futures:
                assert future.result().usage.total_tokens == 15
    assert len(seen) == 2


def test_openai_stream_true_without_customer_is_refused():
    seen: list[Any] = []
    client = _openai(seen, require_customer_id=True)
    with pytest.raises(CustomerIdRequired):
        client.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES, stream=True)
    assert seen == []


def test_openai_stream_helper_checks_at_enter():
    seen: list[Any] = []
    client = _openai(seen, require_customer_id=True)
    if not hasattr(client.chat.completions, "stream"):
        pytest.skip("installed openai lacks chat.completions.stream")
    with pytest.raises(CustomerIdRequired):
        with client.chat.completions.stream(model="gpt-4o-mini", messages=MESSAGES):
            pass
    assert seen == []
    with use_attribution(business_subject_ref="cust_123"):
        with client.chat.completions.stream(model="gpt-4o-mini", messages=MESSAGES) as stream:
            for _ in stream:
                pass
    assert len(seen) == 1


def test_async_openai_without_customer_is_refused_and_tasks_inherit():
    seen: list[Any] = []

    async def _run():
        client = _async_openai(seen, require_customer_id=True)
        with pytest.raises(CustomerIdRequired):
            await client.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES)
        assert seen == []
        with use_attribution(business_subject_ref="cust_123"):
            task = asyncio.create_task(
                client.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES)
            )
        resp = await task
        assert resp.usage.total_tokens == 15
        await client.close()

    asyncio.run(_run())
    assert len(seen) == 1


def test_async_openai_stream_helper_checks_at_enter():
    seen: list[Any] = []

    async def _run():
        client = _async_openai(seen, require_customer_id=True)
        if not hasattr(client.chat.completions, "stream"):
            pytest.skip("installed openai lacks chat.completions.stream")
        with pytest.raises(CustomerIdRequired):
            async with client.chat.completions.stream(model="gpt-4o-mini", messages=MESSAGES):
                pass
        await client.close()

    asyncio.run(_run())
    assert seen == []


def test_anthropic_create_and_stream_helper():
    seen: list[Any] = []
    client = _anthropic(seen, require_customer_id=True)
    with pytest.raises(CustomerIdRequired):
        client.messages.create(model="claude-haiku-4-5", max_tokens=64, messages=MESSAGES)
    with pytest.raises(CustomerIdRequired):
        with client.messages.stream(model="claude-haiku-4-5", max_tokens=64, messages=MESSAGES):
            pass
    assert seen == []
    with use_attribution(business_subject_ref="cust_123"):
        resp = client.messages.create(model="claude-haiku-4-5", max_tokens=64, messages=MESSAGES)
        assert resp.usage.output_tokens == 5
        with client.messages.stream(
            model="claude-haiku-4-5", max_tokens=64, messages=MESSAGES
        ) as stream:
            assert "".join(stream.text_stream) == "Hello"
    assert len(seen) == 2


def test_async_anthropic_create_and_stream_helper():
    seen: list[Any] = []

    async def _run():
        client = _async_anthropic(seen, require_customer_id=True)
        with pytest.raises(CustomerIdRequired):
            await client.messages.create(model="claude-haiku-4-5", max_tokens=64, messages=MESSAGES)
        with pytest.raises(CustomerIdRequired):
            async with client.messages.stream(
                model="claude-haiku-4-5", max_tokens=64, messages=MESSAGES
            ):
                pass
        assert seen == []
        with use_attribution(business_subject_ref="cust_123"):
            resp = await client.messages.create(
                model="claude-haiku-4-5", max_tokens=64, messages=MESSAGES
            )
            assert resp.usage.output_tokens == 5
        await client.close()

    asyncio.run(_run())
    assert len(seen) == 1
