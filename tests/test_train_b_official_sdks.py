"""Hermetic provider-contract tests using the official synchronous SDK objects."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from kazenai import StreamCutoffError, monitor, patch_anthropic, patch_openai
from kazenai.enforcement import Enforcement
from kazenai.sinks import MemorySink
from kazenai.streaming.lifecycle import StreamOutcome

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")


OPENAI_SSE = """data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":25,"completion_tokens":15,"total_tokens":40}}

data: [DONE]

"""


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


@pytest.fixture(autouse=True)
def _standalone(monkeypatch):
    for key in (
        "KAZENAI_FINOPS_URL",
        "KAZENAI_FINOPS_INGEST_URL",
        "KAZENAI_INGEST_URL",
        "KAZENAI_DENY_STREAMING",
        "KAZENAI_UNKNOWN_MODEL_POLICY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_CONTROL_PROFILE", "1")


def _openai_client(request_bodies: list[dict[str, Any]]):
    def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=OPENAI_SSE.encode(),
            request=request,
        )

    return openai.OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _anthropic_client():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=ANTHROPIC_SSE.encode(),
            request=request,
        )

    return anthropic.Anthropic(
        api_key="test",
        base_url="https://anthropic.test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_official_openai_stream_merges_usage_and_settles_exact() -> None:
    bodies: list[dict[str, Any]] = []
    client = monitor(_openai_client(bodies), max_budget_usd=5.0)
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )
    assert "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices) == "Hello"
    assert bodies[-1]["stream_options"]["include_usage"] is True
    assert stream._attempt.terminal_outcome == StreamOutcome.COMPLETE
    assert stream._attempt.cost_confidence == "exact"
    assert stream._attempt.observed_usage.total_tokens == 40


def test_official_openai_local_cutoff_and_early_context_exit() -> None:
    bodies: list[dict[str, Any]] = []
    cutoff_client = monitor(
        _openai_client(bodies),
        max_budget_usd=5.0,
        stream_cutoff_usd=0.0000001,
    )
    cutoff_stream = cutoff_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )
    with pytest.raises(StreamCutoffError):
        list(cutoff_stream)
    assert cutoff_stream._attempt.terminal_outcome == StreamOutcome.CUTOFF

    context_client = monitor(_openai_client([]), max_budget_usd=5.0)
    context_stream = context_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )
    with context_stream as entered:
        next(entered)
    assert context_stream._attempt.terminal_outcome == StreamOutcome.CLIENT_CANCELLED
    assert context_stream._attempt.financial_pending is True


def test_official_openai_stream_helper_is_lazy_and_settles_exact() -> None:
    bodies: list[dict[str, Any]] = []
    sink = MemorySink()
    client = _openai_client(bodies)
    patch_openai(
        client,
        org_id="o",
        project_id="p",
        agent_id="a",
        enforcement=Enforcement(max_cost_usd=5.0),
        event_sink=sink,
        certified_surface=True,
    )
    manager = client.chat.completions.stream(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert bodies == []
    with manager as stream:
        deltas = [
            event.delta
            for event in stream
            if event.type == "content.delta"
        ]
    assert "".join(deltas) == "Hello"
    assert bodies[-1]["stream_options"]["include_usage"] is True
    assert stream._attempt.terminal_outcome == StreamOutcome.COMPLETE
    assert stream._attempt.cost_confidence == "exact"
    assert stream._attempt.observed_usage.total_tokens == 40
    events = [event for event in sink.snapshot() if event["event_type"] == "model.call"]
    assert len(events) == 1
    assert events[0]["payload"]["method"] == "openai.chat.completions.stream"


@pytest.mark.parametrize("consumer", ["until_done", "get_final_completion"])
def test_official_openai_stream_helper_completion_methods(consumer: str) -> None:
    client = monitor(_openai_client([]), max_budget_usd=5.0)
    with client.chat.completions.stream(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
    ) as stream:
        getattr(stream, consumer)()
    assert stream._attempt.terminal_outcome == StreamOutcome.COMPLETE
    assert stream._attempt.financial_pending is False
    assert stream._attempt.observed_usage.total_tokens == 40


def test_official_anthropic_create_stream_merges_partial_usage() -> None:
    client = monitor(_anthropic_client(), max_budget_usd=5.0)
    stream = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=64,
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )
    list(stream)
    assert stream._attempt.terminal_outcome == StreamOutcome.COMPLETE
    assert stream._attempt.cost_confidence == "exact"
    assert stream._attempt.observed_usage.input_tokens == 25
    assert stream._attempt.observed_usage.output_tokens == 15
    assert stream._attempt.observed_usage.total_tokens == 40


@pytest.mark.parametrize("consumer", ["text_stream", "get_final_text", "until_done"])
def test_official_anthropic_manager_helpers_complete_exact(consumer: str) -> None:
    sink = MemorySink()
    client = _anthropic_client()
    patch_anthropic(
        client,
        org_id="o",
        project_id="p",
        agent_id="a",
        enforcement=Enforcement(max_cost_usd=5.0),
        event_sink=sink,
        certified_surface=True,
    )
    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=64,
        messages=[{"role": "user", "content": "Hello"}],
    ) as stream:
        if consumer == "text_stream":
            assert "".join(stream.text_stream) == "Hello"
        else:
            getattr(stream, consumer)()
    assert stream._attempt.terminal_outcome == StreamOutcome.COMPLETE
    assert stream._attempt.financial_pending is False
    assert stream._attempt.observed_usage.input_tokens == 25
    assert stream._attempt.observed_usage.output_tokens == 15
    events = [event for event in sink.snapshot() if event["event_type"] == "model.call"]
    assert len(events) == 1
    assert events[0]["payload"]["method"] == "anthropic.messages.stream"
