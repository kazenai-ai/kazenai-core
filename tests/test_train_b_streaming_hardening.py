"""Release-blocking Train B lifecycle, accounting, and privacy regressions."""

from __future__ import annotations

import gc
import importlib
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from kazenai import (
    BudgetExceeded,
    BudgetUnavailable,
    KazenCircuitBreaker,
    StreamCutoffError,
    UnknownModelError,
    monitor,
    patch_openai,
)
from kazenai.enforcement import Enforcement
from kazenai.monitor import _precall_projection_usd
from kazenai.sinks import MemorySink
from kazenai.streaming.anthropic import observe_anthropic_event
from kazenai.streaming.lifecycle import StreamOutcome, StreamUsage
from kazenai.streaming.openai import observe_openai_chunk
from kazenai.spine.guard import ReservationHandle


def _standalone(monkeypatch) -> None:
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


def _oa_text(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": text}}]}


def _oa_tool(arguments: str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"function": {"name": "lookup", "arguments": arguments}}
                    ]
                }
            }
        ]
    }


def _oa_usage(prompt: int = 20, completion: int = 10) -> dict[str, Any]:
    return {
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


class _Stream:
    def __init__(self, chunks: list[Any], *, fail_at: int | None = None) -> None:
        self.chunks = chunks
        self.index = 0
        self.fail_at = fail_at
        self.closed = False

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        if self.fail_at is not None and self.index == self.fail_at:
            raise RuntimeError("provider secret=sk-do-not-log")
        if self.index >= len(self.chunks):
            raise StopIteration
        value = self.chunks[self.index]
        self.index += 1
        return value

    def close(self) -> None:
        self.closed = True


class _OpenAICompletions:
    def __init__(self, chunks: list[Any], *, open_error: BaseException | None = None) -> None:
        self.chunks = chunks
        self.open_error = open_error
        self.calls = 0
        self.last_stream: _Stream | None = None

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.open_error is not None:
            raise self.open_error
        self.last_stream = _Stream(self.chunks)
        return self.last_stream


class _OpenAI:
    def __init__(self, chunks: list[Any], *, open_error: BaseException | None = None) -> None:
        self.chat = SimpleNamespace(
            completions=_OpenAICompletions(chunks, open_error=open_error)
        )


class _AnthropicStream(_Stream):
    def __init__(self, chunks: list[Any]) -> None:
        super().__init__(chunks)
        self._final = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=11, output_tokens=7)
        )

    def get_final_message(self) -> Any:
        return self._final

    def get_final_text(self) -> str:
        return "complete"

    def until_done(self) -> None:
        self.index = len(self.chunks)


class _AnthropicManager:
    def __init__(self, chunks: list[Any], *, enter_error: BaseException | None = None) -> None:
        self.chunks = chunks
        self.enter_error = enter_error
        self.entered = False
        self.exited = False

    def __enter__(self) -> _AnthropicStream:
        self.entered = True
        if self.enter_error is not None:
            raise self.enter_error
        return _AnthropicStream(self.chunks)

    def __exit__(self, *args: Any) -> None:
        self.exited = True


class _AnthropicMessages:
    def __init__(self, chunks: list[Any], *, enter_error: BaseException | None = None) -> None:
        self.chunks = chunks
        self.enter_error = enter_error
        self.manager_calls = 0

    def create(self, *args: Any, **kwargs: Any) -> _AnthropicStream:
        return _AnthropicStream(self.chunks)

    def stream(self, *args: Any, **kwargs: Any) -> _AnthropicManager:
        self.manager_calls += 1
        return _AnthropicManager(self.chunks, enter_error=self.enter_error)


class _Anthropic:
    def __init__(self, chunks: list[Any], *, enter_error: BaseException | None = None) -> None:
        self.messages = _AnthropicMessages(chunks, enter_error=enter_error)


def test_projection_prices_estimated_input_plus_enforced_output() -> None:
    projected = _precall_projection_usd(
        {
            "model": "gpt-4o-mini",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
        },
        None,
    )
    # 2,000 conservative input tokens + the actual 100-token output cap.
    assert projected == pytest.approx(
        (2_000 * 0.00015 + 100 * 0.00060) / 1_000.0
    )


def test_unknown_model_block_policy_reaches_caller_before_dispatch(monkeypatch) -> None:
    _standalone(monkeypatch)
    monkeypatch.setenv("KAZENAI_UNKNOWN_MODEL_POLICY", "block")
    client = _OpenAI([])
    monitor(client, max_budget_usd=5.0)
    with pytest.raises(UnknownModelError):
        client.chat.completions.create(
            model="unpriced-provider/model",
            stream=True,
            messages=[{"role": "user", "content": "hello"}],
        )
    assert client.chat.completions.calls == 0


def test_shared_deny_releases_local_projection_without_counting_call(monkeypatch) -> None:
    _standalone(monkeypatch)
    monitor_module = importlib.import_module("kazenai.monitor")

    enforcement = Enforcement(max_cost_usd=5.0, calls_per_minute=5)
    client = _OpenAI([])
    monkeypatch.setattr(
        monitor_module,
        "_try_reserve_budget",
        lambda **kwargs: (_ for _ in ()).throw(BudgetExceeded("shared deny")),
    )
    patch_openai(
        client,
        org_id="o",
        project_id="p",
        agent_id="a",
        enforcement=enforcement,
        certified_surface=True,
    )
    with pytest.raises(BudgetExceeded, match="shared deny"):
        client.chat.completions.create(
            model="gpt-4o-mini",
            stream=True,
            messages=[{"role": "user", "content": "hello"}],
        )
    assert enforcement.snapshot()["pending_projected_usd"] == 0
    assert enforcement.snapshot()["calls_in_window"] == 0
    assert client.chat.completions.calls == 0


def test_provider_open_error_releases_local_hold_and_sanitizes_event(monkeypatch) -> None:
    _standalone(monkeypatch)
    sink = MemorySink()
    enforcement = Enforcement(max_cost_usd=5.0)
    client = _OpenAI([], open_error=RuntimeError("secret=sk-do-not-log"))
    patch_openai(
        client,
        org_id="o",
        project_id="p",
        agent_id="a",
        enforcement=enforcement,
        event_sink=sink,
        certified_surface=True,
    )
    with pytest.raises(RuntimeError, match="do-not-log"):
        client.chat.completions.create(
            model="gpt-4o-mini",
            stream=True,
            messages=[{"role": "user", "content": "hello"}],
        )
    snap = enforcement.snapshot()
    assert snap["pending_projected_usd"] == 0
    assert snap["calls_in_window"] == 1
    event = sink.snapshot()[-1]
    assert event["payload"]["terminal_outcome"] == "PROVIDER_ERROR"
    assert event["payload"]["financial_pending"] is True
    assert event["payload"]["error"] == {"type": "RuntimeError"}
    assert "sk-do-not-log" not in str(event)


def test_public_monitor_cutoff_uses_model_output_price(monkeypatch) -> None:
    _standalone(monkeypatch)
    client = _OpenAI([_oa_text("x" * 4_000)])
    monitor(client, max_budget_usd=5.0, stream_cutoff_usd=0.0001)
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        stream=True,
        messages=[{"role": "user", "content": "hello"}],
    )
    with pytest.raises(StreamCutoffError):
        list(stream)
    assert stream._attempt.estimated_spend_usd > 0
    assert stream._attempt.terminal_outcome == StreamOutcome.CUTOFF


def test_openai_tool_arguments_count_toward_local_cutoff() -> None:
    text, usage = observe_openai_chunk(_oa_tool("x" * 100))
    assert "lookup" in text
    assert len(text) >= 100
    assert usage is None


def test_openai_context_exit_before_eof_is_cancelled(monkeypatch) -> None:
    _standalone(monkeypatch)
    client = _OpenAI([_oa_text("partial"), _oa_usage()])
    monitor(client, max_budget_usd=5.0)
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        stream=True,
        messages=[{"role": "user", "content": "hello"}],
    )
    with stream as entered:
        next(entered)
    assert stream._attempt.terminal_outcome == StreamOutcome.CLIENT_CANCELLED
    assert stream._attempt.financial_pending is True


@pytest.mark.parametrize("terminal", ["exact", "cancelled"])
def test_stream_reservation_lifecycle_follows_provider_outcome(
    monkeypatch, terminal: str
) -> None:
    _standalone(monkeypatch)
    monitor_module = importlib.import_module("kazenai.monitor")
    lifecycle_events: list[str] = []
    handle = ReservationHandle(
        reserved_cost_usd=0.01,
        reservation_id="res_stream_1",
        call_id="call_stream_1",
        attempt=1,
        lifecycle=True,
    )
    monkeypatch.setattr(
        monitor_module,
        "_try_reserve_budget",
        lambda **kwargs: handle,
    )
    monkeypatch.setattr(
        monitor_module,
        "_try_start_stream_reservation",
        lambda **kwargs: lifecycle_events.append("provider_started") or True,
    )
    monkeypatch.setattr(
        monitor_module,
        "_try_settle_stream_budget",
        lambda **kwargs: lifecycle_events.append("usage_known") or True,
    )
    monkeypatch.setattr(
        monitor_module,
        "_try_mark_stream_pending",
        lambda **kwargs: lifecycle_events.append("outcome_unknown") or True,
    )

    client = _OpenAI([_oa_text("hello"), _oa_usage()])
    monitor(client, max_budget_usd=5.0)
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        stream=True,
        messages=[{"role": "user", "content": "hello"}],
    )
    if terminal == "exact":
        list(stream)
        assert lifecycle_events == ["provider_started", "usage_known"]
    else:
        with stream as entered:
            next(entered)
        assert lifecycle_events == ["provider_started", "outcome_unknown"]


def test_stream_provider_start_failure_blocks_dispatch_and_releases_hold(monkeypatch) -> None:
    _standalone(monkeypatch)
    monitor_module = importlib.import_module("kazenai.monitor")
    enforcement = Enforcement(max_cost_usd=5.0)
    handle = ReservationHandle(
        reserved_cost_usd=0.01,
        reservation_id="res_stream_2",
        call_id="call_stream_2",
        attempt=1,
        lifecycle=True,
    )
    monkeypatch.setattr(
        monitor_module,
        "_try_reserve_budget",
        lambda **kwargs: handle,
    )
    monkeypatch.setattr(
        monitor_module,
        "_try_start_stream_reservation",
        lambda **kwargs: (_ for _ in ()).throw(
            BudgetUnavailable("provider-start lifecycle unavailable")
        ),
    )

    client = _OpenAI([_oa_usage()])
    patch_openai(
        client,
        org_id="o",
        project_id="p",
        agent_id="a",
        enforcement=enforcement,
        certified_surface=True,
    )
    with pytest.raises(BudgetUnavailable, match="provider-start"):
        client.chat.completions.create(
            model="gpt-4o-mini",
            stream=True,
            messages=[{"role": "user", "content": "hello"}],
        )
    assert client.chat.completions.calls == 0
    assert enforcement.snapshot()["pending_projected_usd"] == 0


def test_exact_usage_with_failed_shared_settlement_stays_pending(monkeypatch) -> None:
    _standalone(monkeypatch)
    monitor_module = importlib.import_module("kazenai.monitor")
    lifecycle_events: list[str] = []
    handle = ReservationHandle(
        reserved_cost_usd=0.01,
        reservation_id="res_stream_3",
        call_id="call_stream_3",
        attempt=1,
        lifecycle=True,
    )
    monkeypatch.setattr(
        monitor_module,
        "_try_reserve_budget",
        lambda **kwargs: handle,
    )
    monkeypatch.setattr(
        monitor_module,
        "_try_start_stream_reservation",
        lambda **kwargs: lifecycle_events.append("provider_started") or True,
    )
    monkeypatch.setattr(
        monitor_module,
        "_try_settle_stream_budget",
        lambda **kwargs: lifecycle_events.append("usage_known_failed") or False,
    )
    monkeypatch.setattr(
        monitor_module,
        "_try_mark_stream_pending",
        lambda **kwargs: lifecycle_events.append("outcome_unknown") or True,
    )

    enforcement = Enforcement(max_cost_usd=5.0)
    client = _OpenAI([_oa_usage()])
    patch_openai(
        client,
        org_id="o",
        project_id="p",
        agent_id="a",
        enforcement=enforcement,
        certified_surface=True,
    )
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        stream=True,
        messages=[{"role": "user", "content": "hello"}],
    )
    list(stream)

    assert lifecycle_events == [
        "provider_started",
        "usage_known_failed",
        "outcome_unknown",
    ]
    assert stream._attempt.financial_pending is True
    assert stream._attempt.cost_confidence == "exact_usage_pending_settlement"
    assert enforcement.snapshot()["cumulative_cost_usd"] > 0


def test_openai_plain_for_break_finalizes_cancelled(monkeypatch) -> None:
    _standalone(monkeypatch)
    enforcement = Enforcement(max_cost_usd=5.0)
    client = _OpenAI([_oa_text("partial"), _oa_usage()])
    patch_openai(
        client,
        org_id="o",
        project_id="p",
        agent_id="a",
        enforcement=enforcement,
        certified_surface=True,
    )
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        stream=True,
        messages=[{"role": "user", "content": "hello"}],
    )
    for _chunk in stream:
        break
    gc.collect()
    assert stream._attempt.terminal_outcome == StreamOutcome.CLIENT_CANCELLED
    assert enforcement.snapshot()["pending_projected_usd"] == 0


def test_complete_without_final_usage_stays_financially_pending(monkeypatch) -> None:
    _standalone(monkeypatch)
    client = _OpenAI([_oa_text("complete but no usage")])
    monitor(client, max_budget_usd=5.0)
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        stream=True,
        messages=[{"role": "user", "content": "hello"}],
    )
    list(stream)
    assert stream._attempt.terminal_outcome == StreamOutcome.COMPLETE
    assert stream._attempt.financial_pending is True
    assert stream._attempt.cost_confidence == "pending"


def test_anthropic_partial_usage_merges_start_and_delta() -> None:
    _, start = observe_anthropic_event(
        {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 25}},
        }
    )
    _, delta = observe_anthropic_event(
        {"type": "message_delta", "usage": {"output_tokens": 15}}
    )
    assert start is not None and delta is not None
    merged = StreamUsage().merged(start).merged(delta)
    assert merged.authoritative is True
    assert merged.as_openai_style() == {
        "prompt_tokens": 25,
        "input_tokens": 25,
        "completion_tokens": 15,
        "output_tokens": 15,
        "total_tokens": 40,
    }


def test_anthropic_start_usage_alone_is_not_exact() -> None:
    _, start = observe_anthropic_event(
        {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 25, "output_tokens": 1}},
        }
    )
    assert start is not None
    merged = StreamUsage().merged(start)
    assert merged.authoritative is False
    assert merged.total_tokens == 26


def test_anthropic_non_text_deltas_count_for_cutoff() -> None:
    text, usage = observe_anthropic_event(
        {
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": "{\"q\":\"x\"}"},
        }
    )
    assert text == '{"q":"x"}'
    assert usage is None


def test_anthropic_manager_construction_does_not_reserve(monkeypatch) -> None:
    _standalone(monkeypatch)
    enforcement = Enforcement(max_cost_usd=5.0)
    client = _Anthropic([])
    from kazenai import patch_anthropic

    patch_anthropic(
        client,
        org_id="o",
        project_id="p",
        agent_id="a",
        enforcement=enforcement,
        certified_surface=True,
    )
    manager = client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=32,
        messages=[{"role": "user", "content": "hello"}],
    )
    assert manager._attempt is None
    assert enforcement.snapshot()["pending_projected_usd"] == 0


def test_anthropic_manager_enter_error_releases_hold(monkeypatch) -> None:
    _standalone(monkeypatch)
    enforcement = Enforcement(max_cost_usd=5.0)
    sink = MemorySink()
    client = _Anthropic([], enter_error=RuntimeError("provider enter failed"))
    from kazenai import patch_anthropic

    patch_anthropic(
        client,
        org_id="o",
        project_id="p",
        agent_id="a",
        enforcement=enforcement,
        event_sink=sink,
        certified_surface=True,
    )
    manager = client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=32,
        messages=[{"role": "user", "content": "hello"}],
    )
    with pytest.raises(RuntimeError, match="enter failed"):
        with manager:
            pass
    assert enforcement.snapshot()["pending_projected_usd"] == 0
    assert manager._attempt.terminal_outcome == StreamOutcome.PROVIDER_ERROR
    assert sink.snapshot()[-1]["payload"]["financial_pending"] is True


@pytest.mark.parametrize("helper", ["get_final_text", "until_done"])
def test_anthropic_completion_helpers_finalize_exact(monkeypatch, helper: str) -> None:
    _standalone(monkeypatch)
    client = _Anthropic([])
    monitor(client, max_budget_usd=5.0)
    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=32,
        messages=[{"role": "user", "content": "hello"}],
    ) as stream:
        getattr(stream, helper)()
    assert stream._attempt.terminal_outcome == StreamOutcome.COMPLETE
    assert stream._attempt.financial_pending is False
    assert stream._attempt.observed_usage.total_tokens == 18


def test_post_call_control_exception_is_not_swallowed(monkeypatch) -> None:
    _standalone(monkeypatch)

    class _FinOps:
        def handle_llm_call(self, event: Any) -> dict[str, Any]:
            return {"_circuit_breaker_exc": KazenCircuitBreaker("pause")}

    client = _OpenAI([_oa_usage()])
    patch_openai(
        client,
        org_id="o",
        project_id="p",
        agent_id="a",
        enforcement=Enforcement(max_cost_usd=5.0),
        finops=_FinOps(),
        certified_surface=True,
    )
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        stream=True,
        messages=[{"role": "user", "content": "hello"}],
    )
    with pytest.raises(KazenCircuitBreaker, match="pause"):
        list(stream)
    assert stream._attempt.finalized is True
    assert stream._attempt.terminal_outcome == StreamOutcome.COMPLETE
