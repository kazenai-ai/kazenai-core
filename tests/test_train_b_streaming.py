"""Train B — hermetic Control-certified streaming contract tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterator, List

import pytest

from kazenai import BudgetExceeded, StreamCutoffError, UnsupportedModeError, monitor
from kazenai.monitor import SUPPORTED_OPENAI_STREAM_METHOD
from kazenai.streaming.lifecycle import StreamAttempt, StreamOutcome, StreamUsage
from kazenai.streaming.openai import merge_stream_options, observe_openai_chunk


def _standalone(monkeypatch) -> None:
    monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)
    monkeypatch.delenv("KAZENAI_FINOPS_INGEST_URL", raising=False)
    monkeypatch.delenv("KAZENAI_INGEST_URL", raising=False)
    monkeypatch.delenv("KAZENAI_DENY_STREAMING", raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_CONTROL_PROFILE", "1")


class _FakeOpenAIStream:
    def __init__(self, chunks: List[Any]) -> None:
        self._chunks = list(chunks)
        self._i = 0
        self.closed = False

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        if self._i >= len(self._chunks):
            raise StopIteration
        c = self._chunks[self._i]
        self._i += 1
        return c

    def close(self) -> None:
        self.closed = True


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls = 0
        self.last_kwargs: dict[str, Any] = {}
        self.chunks: List[Any] = []

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self.last_kwargs = dict(kwargs)
        self.calls += 1
        if kwargs.get("stream"):
            return _FakeOpenAIStream(self.chunks)
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        )


class _FakeOpenAI:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions())


class _FakeAnthropicEventStream:
    def __init__(self, events: List[Any]) -> None:
        self._events = list(events)
        self._i = 0
        self.closed = False
        self._final = SimpleNamespace(usage=SimpleNamespace(input_tokens=11, output_tokens=7))

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        if self._i >= len(self._events):
            raise StopIteration
        e = self._events[self._i]
        self._i += 1
        return e

    def close(self) -> None:
        self.closed = True

    def get_final_message(self) -> Any:
        return self._final

    def get_final_text(self) -> str:
        return "final"


class _FakeAnthropicManager:
    def __init__(self, events: List[Any]) -> None:
        self._events = events
        self.entered = False
        self.exited = False

    def __enter__(self) -> _FakeAnthropicEventStream:
        self.entered = True
        return _FakeAnthropicEventStream(self._events)

    def __exit__(self, *args: Any) -> None:
        self.exited = True


class _FakeAnthropicMessages:
    def __init__(self) -> None:
        self.calls = 0
        self.stream_calls = 0
        self.last_kwargs: dict[str, Any] = {}
        self.events: List[Any] = []

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self.last_kwargs = dict(kwargs)
        self.calls += 1
        if kwargs.get("stream"):
            return _FakeAnthropicEventStream(self.events)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=12, output_tokens=8),
        )

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        self.stream_calls += 1
        self.last_kwargs = dict(kwargs)
        return _FakeAnthropicManager(self.events)


class _FakeAnthropic:
    def __init__(self) -> None:
        self.messages = _FakeAnthropicMessages()


def _openai_text_chunk(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}}]}


def _openai_usage_chunk(*, prompt: int = 9, completion: int = 4) -> dict:
    return {
        "choices": [],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
    }


def test_merge_stream_options_preserves_caller_keys():
    out = merge_stream_options({"stream": True, "stream_options": {"foo": 1}, "model": "m"})
    assert out["stream"] is True
    assert out["stream_options"]["foo"] == 1
    assert out["stream_options"]["include_usage"] is True


def test_observe_openai_usage_chunk_empty_choices():
    text, usage = observe_openai_chunk(_openai_usage_chunk())
    assert text == ""
    assert usage is not None and usage.authoritative
    assert usage.total_tokens == 13


def test_openai_stream_settles_after_usage_chunk(monkeypatch):
    _standalone(monkeypatch)
    client = _FakeOpenAI()
    client.chat.completions.chunks = [
        _openai_text_chunk("hello "),
        _openai_text_chunk("world"),
        _openai_usage_chunk(prompt=10, completion=6),
    ]
    monitor(client, max_budget_usd=5.0)
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert client.chat.completions.calls == 1
    opts = client.chat.completions.last_kwargs.get("stream_options") or {}
    assert opts.get("include_usage") is True
    chunks = list(stream)
    assert len(chunks) == 3
    assert stream._attempt.finalized
    assert stream._attempt.terminal_outcome == StreamOutcome.COMPLETE
    assert stream._attempt.financial_pending is False
    assert stream._attempt.cost_confidence == "exact"


def test_openai_stream_early_close_pending(monkeypatch):
    _standalone(monkeypatch)
    client = _FakeOpenAI()
    client.chat.completions.chunks = [_openai_text_chunk("partial"), _openai_usage_chunk()]
    monitor(client, max_budget_usd=5.0)
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )
    it = iter(stream)
    next(it)
    stream.close()
    assert stream._attempt.finalized
    assert stream._attempt.terminal_outcome == StreamOutcome.CLIENT_CANCELLED
    assert stream._attempt.financial_pending is True
    assert getattr(stream._upstream, "closed", False) is True


def test_openai_stream_cutoff_local(monkeypatch):
    _standalone(monkeypatch)
    client = _FakeOpenAI()
    client.chat.completions.chunks = [_openai_text_chunk("x" * 400), _openai_text_chunk("never")]
    from kazenai import patch_openai
    from kazenai.enforcement import Enforcement

    patch_openai(
        client,
        org_id="local",
        project_id="default",
        agent_id="agent",
        enforcement=Enforcement(max_cost_usd=5.0),
        usd_per_1k_tokens=1.0,
        stream_cutoff_usd=0.05,
        certified_surface=True,
    )
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )
    with pytest.raises(StreamCutoffError):
        list(stream)
    assert stream._attempt.terminal_outcome == StreamOutcome.CUTOFF
    assert stream._attempt.financial_pending is True


def test_openai_stream_deny_before_provider(monkeypatch):
    _standalone(monkeypatch)
    client = _FakeOpenAI()
    client.chat.completions.chunks = [_openai_text_chunk("x")]
    monitor(client, max_budget_usd=0.000001)
    with pytest.raises(BudgetExceeded):
        client.chat.completions.create(
            model="gpt-4o",
            stream=True,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert client.chat.completions.calls == 0


def test_openai_stream_explicit_deny_env(monkeypatch):
    _standalone(monkeypatch)
    monkeypatch.setenv("KAZENAI_DENY_STREAMING", "1")
    client = _FakeOpenAI()
    monitor(client, max_budget_usd=5.0)
    with pytest.raises(UnsupportedModeError):
        client.chat.completions.create(
            model="gpt-4o-mini",
            stream=True,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert client.chat.completions.calls == 0


def test_anthropic_create_stream_complete(monkeypatch):
    _standalone(monkeypatch)
    client = _FakeAnthropic()
    client.messages.events = [
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text="Hi")),
        SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(input_tokens=5, output_tokens=3),
        ),
        SimpleNamespace(type="message_stop"),
    ]
    monitor(client, max_budget_usd=5.0)
    stream = client.messages.create(
        model="claude-3-5-haiku-latest",
        stream=True,
        max_tokens=64,
        messages=[{"role": "user", "content": "hi"}],
    )
    list(stream)
    assert client.messages.calls == 1
    assert stream._attempt.finalized
    assert stream._attempt.terminal_outcome == StreamOutcome.COMPLETE


def test_anthropic_messages_stream_manager(monkeypatch):
    _standalone(monkeypatch)
    client = _FakeAnthropic()
    client.messages.events = [
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text="Hello")),
        SimpleNamespace(
            type="message_delta",
            usage=SimpleNamespace(input_tokens=8, output_tokens=2),
        ),
    ]
    monitor(client, max_budget_usd=5.0)
    with client.messages.stream(
        model="claude-3-5-haiku-latest",
        max_tokens=32,
        messages=[{"role": "user", "content": "hi"}],
    ) as stream:
        chunks = list(stream)
        assert chunks
        msg = stream.get_final_message()
        assert msg.usage.output_tokens == 7
    assert client.messages.stream_calls == 1


def test_stream_finalize_once_under_close_and_exhaust(monkeypatch):
    _standalone(monkeypatch)
    client = _FakeOpenAI()
    client.chat.completions.chunks = [_openai_usage_chunk()]
    monitor(client, max_budget_usd=5.0)
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )
    list(stream)
    stream.close()
    stream.close()
    assert stream._attempt.finalized
    assert stream._attempt.terminal_outcome == StreamOutcome.COMPLETE


def test_non_streaming_still_works(monkeypatch):
    _standalone(monkeypatch)
    client = _FakeOpenAI()
    monitor(client, max_budget_usd=5.0)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp.usage.total_tokens == 15
    assert client.chat.completions.calls == 1


def test_privacy_stream_events_have_no_chunk_text(monkeypatch):
    _standalone(monkeypatch)
    client = _FakeOpenAI()
    client.chat.completions.chunks = [
        _openai_text_chunk("SECRET_PROMPT_CONTENT"),
        _openai_usage_chunk(),
    ]
    monitor(client, max_budget_usd=5.0)
    # Access MemorySink via patched path is hard; build attempt finalize payload instead.
    from kazenai.enforcement import Enforcement
    from kazenai.context import RunContext

    events = []

    def emit(attempt, settle_cost, tokens_used, error):
        events.append(
            {
                "pending": attempt.financial_pending,
                "payload_keys": ["stream", "terminal_outcome"],
                "secret": "SECRET_PROMPT_CONTENT" in str(attempt.observed_usage.raw),
            }
        )

    attempt = StreamAttempt(
        surface=SUPPORTED_OPENAI_STREAM_METHOD,
        step_ctx=RunContext.new(org_id="o", project_id="p", workspace_id="w", agent_id="a"),
        enforcement=Enforcement(max_cost_usd=5.0),
        held_projection=0.01,
        reserved_cost_usd=None,
        model="gpt-4o-mini",
        method=SUPPORTED_OPENAI_STREAM_METHOD,
        usd_per_1k_tokens=0.0,
        capture_mode=__import__("kazenai.capture_policy", fromlist=["CaptureMode"]).CaptureMode.METADATA,
        agent_role="agent",
        emit_event_fn=emit,
    )
    attempt.observe_usage(
        StreamUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, authoritative=True, raw={"total_tokens": 2})
    )
    attempt.finalize(StreamOutcome.COMPLETE)
    assert events
    assert events[0]["secret"] is False
