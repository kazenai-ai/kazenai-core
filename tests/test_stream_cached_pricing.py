"""A streamed call bills cached tokens at the cached rate, as a plain call does (kazenai 1.2.2).

The official provider SDKs run on an ``httpx.MockTransport``. The usage reaches pricing
through the same streaming observers a real provider stream does.
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Any

import httpx
import pytest

from kazenai import monitor
from kazenai.cost_engine import TokenCostEngine

openai = pytest.importorskip("openai")

MODEL = "gpt-4o-mini"
CHAT_USAGE = {
    "prompt_tokens": 1_000,
    "completion_tokens": 100,
    "total_tokens": 1_100,
    "prompt_tokens_details": {"cached_tokens": 900},
}
RESPONSES_USAGE = {
    "input_tokens": 1_000,
    "output_tokens": 100,
    "total_tokens": 1_100,
    "input_tokens_details": {"cached_tokens": 900},
    "output_tokens_details": {"reasoning_tokens": 0},
}
_COMPLETION = {"id": "chatcmpl-test", "created": 1, "model": MODEL}


def _sse(*events: dict[str, Any]) -> bytes:
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def _chat_stream() -> bytes:
    chunk = {**_COMPLETION, "object": "chat.completion.chunk"}
    return _sse(
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": None}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {**chunk, "choices": [], "usage": CHAT_USAGE},
    )


def _response(status: str) -> dict[str, Any]:
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "model": MODEL,
        "status": status,
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": RESPONSES_USAGE if status == "completed" else None,
    }


def _responses_stream() -> bytes:
    return _sse(
        {"type": "response.created", "sequence_number": 0, "response": _response("in_progress")},
        {"type": "response.completed", "sequence_number": 1, "response": _response("completed")},
    )


def _provider(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if request.url.path.endswith("/responses"):
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_responses_stream())
        return httpx.Response(200, json=_response("completed"))
    if body.get("stream"):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_chat_stream())
    return httpx.Response(
        200,
        json={
            **_COMPLETION,
            "object": "chat.completion",
            "usage": CHAT_USAGE,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
        },
    )


@pytest.fixture(autouse=True)
def _local_caps_only(monkeypatch, tmp_path):
    for key in ("KAZENAI_FINOPS_URL", "KAZENAI_FINOPS_API_KEY", "KAZENAI_CONTROL_PROFILE", "KAZENAI_ENFORCEMENT_OWNER"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_TIER_MISMATCH_MULTIPLIER", "1")
    monkeypatch.setenv("KAZENAI_SDK_SETTLE_QUEUE_DIR", str(tmp_path))


def _expected_usd() -> float:
    rates = TokenCostEngine()._row(MODEL).rates
    micros = Decimal(100) * rates.input + Decimal(900) * rates.cached_input + Decimal(100) * rates.output
    return float(micros) / 1_000_000


def _recorded(client: Any) -> float:
    return float(client._kazenai_enforcement.snapshot()["cumulative_cost_usd"])


def _client() -> Any:
    raw = openai.OpenAI(api_key="test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(_provider)))
    return monitor(raw, max_budget_usd=1.0, org_id="org-test")


def _async_client() -> Any:
    raw = openai.AsyncOpenAI(
        api_key="test", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(_provider))
    )
    return monitor(raw, max_budget_usd=1.0, org_id="org-test")


def test_a_plain_chat_call_bills_the_cached_rate():
    client = _client()
    client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=100)
    assert _recorded(client) == pytest.approx(_expected_usd(), abs=1e-6)


def test_a_streamed_chat_call_bills_the_cached_rate():
    client = _client()
    stream = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=100, stream=True
    )
    list(stream)
    assert _recorded(client) == pytest.approx(_expected_usd(), abs=1e-6)


def test_an_async_streamed_chat_call_bills_the_cached_rate():
    client = _async_client()

    async def run() -> None:
        stream = await client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=100, stream=True
        )
        async for _chunk in stream:
            pass

    asyncio.run(run())
    assert _recorded(client) == pytest.approx(_expected_usd(), abs=1e-6)


def test_a_streamed_responses_call_bills_the_cached_rate():
    client = _client()
    list(client.responses.create(model=MODEL, input="hi", max_output_tokens=100, stream=True))
    assert _recorded(client) == pytest.approx(_expected_usd(), abs=1e-6)


def test_a_streamed_anthropic_call_bills_cache_tokens_at_their_rates():
    anthropic = pytest.importorskip("anthropic")
    model = "claude-haiku-4-5"
    usage = {"input_tokens": 100, "cache_read_input_tokens": 400, "cache_creation_input_tokens": 1_000}
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {**usage, "output_tokens": 1},
                },
            },
        ),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 50}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()

    def provider(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    raw = anthropic.Anthropic(api_key="test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(provider)))
    client = monitor(raw, max_budget_usd=1.0, org_id="org-test")
    list(client.messages.create(model=model, max_tokens=100, messages=[{"role": "user", "content": "hi"}], stream=True))

    rates = TokenCostEngine()._row(model).rates
    expected = (
        Decimal(100) * rates.input
        + Decimal(400) * rates.cached_input
        + Decimal(1_000) * rates.cache_write_5m
        + Decimal(50) * rates.output
    )
    assert _recorded(client) == pytest.approx(float(expected) / 1_000_000, abs=1e-6)


def test_a_streamed_anthropic_call_bills_one_hour_writes_at_their_rate():
    anthropic = pytest.importorskip("anthropic")
    model = "claude-haiku-4-5"
    usage = {
        "input_tokens": 100,
        "cache_read_input_tokens": 400,
        "cache_creation_input_tokens": 1_000,
        "cache_creation": {"ephemeral_5m_input_tokens": 600, "ephemeral_1h_input_tokens": 400},
    }
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {**usage, "output_tokens": 1},
                },
            },
        ),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 50}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()

    def provider(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    raw = anthropic.Anthropic(api_key="test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(provider)))
    client = monitor(raw, max_budget_usd=1.0, org_id="org-test")
    list(client.messages.create(model=model, max_tokens=100, messages=[{"role": "user", "content": "hi"}], stream=True))

    rates = TokenCostEngine()._row(model).rates
    expected = (
        Decimal(100) * rates.input
        + Decimal(400) * rates.cached_input
        + Decimal(600) * rates.cache_write_5m
        + Decimal(400) * rates.cache_write_1h
        + Decimal(50) * rates.output
    )
    assert _recorded(client) == pytest.approx(float(expected) / 1_000_000, abs=1e-6)


def test_a_plain_responses_call_bills_the_cached_rate():
    client = _client()
    client.responses.create(model=MODEL, input="hi", max_output_tokens=100)
    assert _recorded(client) == pytest.approx(_expected_usd(), abs=1e-6)


def test_an_async_plain_responses_call_bills_the_cached_rate():
    client = _async_client()
    asyncio.run(client.responses.create(model=MODEL, input="hi", max_output_tokens=100))
    assert _recorded(client) == pytest.approx(_expected_usd(), abs=1e-6)
