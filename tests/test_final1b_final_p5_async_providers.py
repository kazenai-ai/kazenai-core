"""FINAL_1_b_final P5 — eight-cell provider matrix via official SDK + fake HTTP/SSE."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from kazenai import monitor
from kazenai.streaming.lifecycle import StreamOutcome

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")


OPENAI_SSE = """data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":25,"completion_tokens":15,"total_tokens":40}}

data: [DONE]

"""

OPENAI_JSON = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
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

ANTHROPIC_JSON = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello"}],
    "model": "claude-haiku-4-5",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


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


def _openai_handler(stream: bool, bodies: list[dict[str, Any]]):
    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode()))
        if stream:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=OPENAI_SSE.encode(),
                request=request,
            )
        return httpx.Response(200, json=OPENAI_JSON, request=request)

    return handler


def _anthropic_handler(stream: bool):
    def handler(request: httpx.Request) -> httpx.Response:
        if stream:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=ANTHROPIC_SSE.encode(),
                request=request,
            )
        return httpx.Response(200, json=ANTHROPIC_JSON, request=request)

    return handler


def test_sync_openai_nonstream_official_sdk():
    bodies: list[dict[str, Any]] = []
    client = monitor(
        openai.OpenAI(
            api_key="test",
            base_url="https://openai.test/v1",
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(_openai_handler(False, bodies))
            ),
        ),
        max_budget_usd=5.0,
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert resp.choices[0].message.content == "Hello"
    assert resp.usage.total_tokens == 15


def test_sync_openai_stream_official_sdk():
    bodies: list[dict[str, Any]] = []
    client = monitor(
        openai.OpenAI(
            api_key="test",
            base_url="https://openai.test/v1",
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(_openai_handler(True, bodies))
            ),
        ),
        max_budget_usd=5.0,
    )
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )
    text = "".join(
        chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices
    )
    assert text == "Hello"
    assert stream._attempt.terminal_outcome == StreamOutcome.COMPLETE
    assert stream._attempt.observed_usage.total_tokens == 40


def test_async_openai_nonstream_official_sdk():
    bodies: list[dict[str, Any]] = []

    async def _run():
        client = monitor(
            openai.AsyncOpenAI(
                api_key="test",
                base_url="https://openai.test/v1",
                max_retries=0,
                http_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(_openai_handler(False, bodies))
                ),
            ),
            max_budget_usd=5.0,
        )
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert resp.choices[0].message.content == "Hello"
        await client.close()

    asyncio.run(_run())


def test_async_openai_stream_official_sdk():
    bodies: list[dict[str, Any]] = []

    async def _run():
        client = monitor(
            openai.AsyncOpenAI(
                api_key="test",
                base_url="https://openai.test/v1",
                max_retries=0,
                http_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(_openai_handler(True, bodies))
                ),
            ),
            max_budget_usd=5.0,
        )
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        parts: list[str] = []
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
        assert "".join(parts) == "Hello"
        assert stream._attempt.terminal_outcome == StreamOutcome.COMPLETE
        assert stream._attempt.observed_usage.total_tokens == 40
        assert bodies[-1]["stream_options"]["include_usage"] is True
        await client.close()

    asyncio.run(_run())


def test_async_openai_stream_early_aclose_pending():
    bodies: list[dict[str, Any]] = []

    async def _run():
        client = monitor(
            openai.AsyncOpenAI(
                api_key="test",
                base_url="https://openai.test/v1",
                max_retries=0,
                http_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(_openai_handler(True, bodies))
                ),
            ),
            max_budget_usd=5.0,
        )
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        await stream.__anext__()
        await stream.aclose()
        assert stream._attempt.terminal_outcome == StreamOutcome.CLIENT_CANCELLED
        assert stream._attempt.financial_pending is True
        await client.close()

    asyncio.run(_run())


def test_sync_anthropic_nonstream_official_sdk():
    client = monitor(
        anthropic.Anthropic(
            api_key="test",
            base_url="https://anthropic.test",
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(_anthropic_handler(False))
            ),
        ),
        max_budget_usd=5.0,
    )
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=64,
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert msg.content[0].text == "Hello"


def test_sync_anthropic_stream_official_sdk():
    client = monitor(
        anthropic.Anthropic(
            api_key="test",
            base_url="https://anthropic.test",
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(_anthropic_handler(True))
            ),
        ),
        max_budget_usd=5.0,
    )
    stream = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=64,
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )
    texts = []
    for event in stream:
        if getattr(event, "type", None) == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta is not None and getattr(delta, "text", None):
                texts.append(delta.text)
    assert "".join(texts) == "Hello"
    assert stream._attempt.terminal_outcome == StreamOutcome.COMPLETE


def test_async_anthropic_nonstream_official_sdk():
    async def _run():
        client = monitor(
            anthropic.AsyncAnthropic(
                api_key="test",
                base_url="https://anthropic.test",
                max_retries=0,
                http_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(_anthropic_handler(False))
                ),
            ),
            max_budget_usd=5.0,
        )
        msg = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=64,
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert msg.content[0].text == "Hello"
        await client.close()

    asyncio.run(_run())


def test_async_anthropic_stream_official_sdk():
    async def _run():
        client = monitor(
            anthropic.AsyncAnthropic(
                api_key="test",
                base_url="https://anthropic.test",
                max_retries=0,
                http_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(_anthropic_handler(True))
                ),
            ),
            max_budget_usd=5.0,
        )
        stream = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=64,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        texts: list[str] = []
        async for event in stream:
            if getattr(event, "type", None) == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta is not None and getattr(delta, "text", None):
                    texts.append(delta.text)
        assert "".join(texts) == "Hello"
        assert stream._attempt.terminal_outcome == StreamOutcome.COMPLETE
        await client.close()

    asyncio.run(_run())


def test_async_openai_never_entered_stream_manager_no_reserve():
    """Construction of .stream() must not reserve; only __aenter__ does."""
    bodies: list[dict[str, Any]] = []

    async def _run():
        client = monitor(
            openai.AsyncOpenAI(
                api_key="test",
                base_url="https://openai.test/v1",
                max_retries=0,
                http_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(_openai_handler(True, bodies))
                ),
            ),
            max_budget_usd=5.0,
        )
        if not hasattr(client.chat.completions, "stream"):
            pytest.skip("installed openai lacks chat.completions.stream")
        manager = client.chat.completions.stream(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )
        # Never enter — no HTTP yet.
        assert bodies == []
        close = getattr(manager, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        await client.close()

    asyncio.run(_run())


def test_openai_sdk_version_recorded():
    ver = getattr(openai, "__version__", "") or ""
    assert ver
    # Baseline 1.x or certified 2.x both acceptable in their envs.
    major = int(ver.split(".")[0])
    assert major in (1, 2)
