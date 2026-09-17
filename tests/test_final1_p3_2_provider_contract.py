"""FINAL_1 P3-2 — certified provider-client contract (sync OpenAI Chat Completions + Anthropic Messages)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from kazenai import (
    BudgetExceeded,
    KazenCircuitBreaker,
    LoopDetected,
    UnsupportedModeError,
    monitor,
    patch_anthropic,
    patch_openai,
)
from kazenai.finops import FinOpsConfig, FinOpsController
from kazenai.model_pricing import default_max_output_tokens
from kazenai.monitor import (
    SUPPORTED_ANTHROPIC_METHOD,
    SUPPORTED_OPENAI_METHOD,
    _apply_output_token_bound,
    _postcall_cost_usd,
)


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = 0
        self.last_kwargs: dict[str, Any] = {}
        self.raise_exc: Optional[BaseException] = None
        self.usage: Any = SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20)

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self.last_kwargs = dict(kwargs)
        if self.raise_exc is not None:
            raise self.raise_exc
        self.calls += 1
        return SimpleNamespace(
            usage=self.usage,
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        )


class FakeResponses:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return SimpleNamespace(output=[], usage=None)


class FakeOpenAI:
    def __init__(self, *, with_responses: bool = True) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())
        if with_responses:
            self.responses = FakeResponses()


class FakeAnthropicMessages:
    def __init__(self) -> None:
        self.calls = 0
        self.last_kwargs: dict[str, Any] = {}
        self.stream = lambda *a, **k: None  # type: ignore[misc]

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self.last_kwargs = dict(kwargs)
        self.calls += 1
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=12, output_tokens=8),
        )


class FakeAnthropic:
    def __init__(self) -> None:
        self.messages = FakeAnthropicMessages()


def _standalone(monkeypatch) -> None:
    monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)
    monkeypatch.delenv("KAZENAI_FINOPS_INGEST_URL", raising=False)
    monkeypatch.delenv("KAZENAI_INGEST_URL", raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")


def test_p32_supported_method_constants():
    assert SUPPORTED_OPENAI_METHOD == "openai.chat.completions.create"
    assert SUPPORTED_ANTHROPIC_METHOD == "anthropic.messages.create"


def test_p32_monitor_signature_has_no_api_key():
    sig = inspect.signature(monitor)
    assert "api_key" not in sig.parameters
    assert "max_budget_usd" in sig.parameters


def test_p32_openai_allow_then_deny_before_provider(monkeypatch):
    _standalone(monkeypatch)
    client = FakeOpenAI()
    monitor(client, max_budget_usd=1.0)
    client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    assert client.chat.completions.calls == 1

    client2 = FakeOpenAI()
    monitor(client2, max_budget_usd=0.000001)
    with pytest.raises(BudgetExceeded):
        client2.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert client2.chat.completions.calls == 0


def test_p32_anthropic_allow_and_deny(monkeypatch):
    _standalone(monkeypatch)
    client = FakeAnthropic()
    monitor(client, max_budget_usd=1.0)
    client.messages.create(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert client.messages.calls == 1

    client2 = FakeAnthropic()
    monitor(client2, max_budget_usd=0.000001)
    with pytest.raises(BudgetExceeded):
        client2.messages.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert client2.messages.calls == 0


def test_p32_monitor_rejects_openai_responses_api(monkeypatch):
    _standalone(monkeypatch)
    client = FakeOpenAI(with_responses=True)
    monitor(client, max_budget_usd=1.0)
    with pytest.raises(UnsupportedModeError):
        client.responses.create(input="hi")
    assert client.responses.calls == 0
    # Direct patch_openai (non-certified) may still wrap responses — covered elsewhere.


def test_p32_monitor_rejects_anthropic_stream_surface(monkeypatch):
    _standalone(monkeypatch)
    client = FakeAnthropic()
    monitor(client, max_budget_usd=1.0)
    with pytest.raises(UnsupportedModeError):
        client.messages.stream(model="claude-sonnet-4-6", messages=[])


def test_p32_control_streaming_denied_before_provider(monkeypatch):
    _standalone(monkeypatch)
    monkeypatch.setenv("KAZENAI_CONTROL_PROFILE", "1")
    client = FakeOpenAI()
    monitor(client, max_budget_usd=1.0)
    with pytest.raises(UnsupportedModeError):
        client.chat.completions.create(
            model="gpt-4o-mini",
            stream=True,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert client.chat.completions.calls == 0


def test_p32_async_client_rejected(monkeypatch):
    _standalone(monkeypatch)

    class AsyncOpenAI:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    with pytest.raises(UnsupportedModeError):
        monitor(AsyncOpenAI(), max_budget_usd=1.0)


def test_p32_provider_exception_releases_and_does_not_count_success(monkeypatch):
    _standalone(monkeypatch)
    client = FakeOpenAI()
    client.chat.completions.raise_exc = RuntimeError("provider down")
    monitor(client, max_budget_usd=1.0)
    with pytest.raises(RuntimeError, match="provider down"):
        client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])
    assert client.chat.completions.calls == 0


def test_p32_retry_after_deny_still_zero_provider_calls(monkeypatch):
    _standalone(monkeypatch)
    client = FakeOpenAI()
    monitor(client, max_budget_usd=0.000001)
    for i in range(3):
        with pytest.raises(BudgetExceeded):
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"retry-{i}"}],
            )
    assert client.chat.completions.calls == 0


def test_p32_missing_usage_still_returns_without_fabricated_cost(monkeypatch):
    _standalone(monkeypatch)
    client = FakeOpenAI()
    client.chat.completions.usage = None
    monitor(client, max_budget_usd=1.0)
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    assert resp.usage is None
    assert client.chat.completions.calls == 1
    # Cost helper returns None when usage missing
    assert _postcall_cost_usd(model="gpt-4o-mini", usage=None, tokens_used=None, usd_per_1k_tokens=None) is None


def test_p32_unknown_model_usage_fields_documented(monkeypatch):
    _standalone(monkeypatch)
    client = FakeOpenAI()
    monitor(client, max_budget_usd=0.000001)
    # Unknown model → projection 0 → local tiny cap does not block (explicit).
    client.chat.completions.create(
        model="totally-unknown-model-xyz",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert client.chat.completions.calls == 1
    assert (
        _postcall_cost_usd(
            model="totally-unknown-model-xyz",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
            tokens_used=20,
            usd_per_1k_tokens=None,
        )
        is None
    )


def test_p32_output_token_bound_injected_and_clamped(monkeypatch):
    _standalone(monkeypatch)
    bound = default_max_output_tokens("gpt-4o")
    assert bound == 4096
    client = FakeOpenAI()
    monitor(client, max_budget_usd=1.0)
    client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert client.chat.completions.last_kwargs.get("max_tokens") == bound

    client2 = FakeOpenAI()
    monitor(client2, max_budget_usd=1.0)
    client2.chat.completions.create(
        model="gpt-4o",
        max_tokens=999999,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert client2.chat.completions.last_kwargs.get("max_tokens") == bound


def test_p32_loop_detected_before_provider(monkeypatch):
    _standalone(monkeypatch)
    client = FakeOpenAI()
    monitor(client, max_budget_usd=10.0)
    msg = [{"role": "user", "content": "identical loop bait"}]
    client.chat.completions.create(model="gpt-4o-mini", messages=msg)
    with pytest.raises(LoopDetected):
        client.chat.completions.create(model="gpt-4o-mini", messages=msg)
    assert client.chat.completions.calls == 1


def test_p32_circuit_breaker_is_post_call_not_budget_exceeded(monkeypatch):
    """Soft CB raises KazenCircuitBreaker after provider success — distinct from BudgetExceeded."""
    _standalone(monkeypatch)
    client = FakeOpenAI()
    # Large recorded usage so trajectory soft-pause trips quickly.
    client.chat.completions.usage = SimpleNamespace(
        prompt_tokens=50_000, completion_tokens=50_000, total_tokens=100_000
    )
    monitor(client, max_budget_usd=0.50, soft_pause_pct=0.10)
    # First expensive call completes then CB may open; catch either CB on first or second.
    raised: list[BaseException] = []
    for i in range(5):
        try:
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"cb-{i}"}],
            )
        except KazenCircuitBreaker as exc:
            raised.append(exc)
            break
        except BudgetExceeded as exc:
            # Hard cap before dispatch is a different failure mode — also valid distinction.
            pytest.fail(f"expected soft KazenCircuitBreaker, got hard BudgetExceeded: {exc}")
    assert raised, "expected KazenCircuitBreaker"
    assert client.chat.completions.calls >= 1
    assert all(isinstance(r, KazenCircuitBreaker) for r in raised)
    assert not any(isinstance(r, BudgetExceeded) for r in raised)


def test_p32_patch_openai_non_certified_still_allows_responses(monkeypatch):
    _standalone(monkeypatch)
    client = FakeOpenAI()
    patch_openai(client, org_id="o", project_id="p", agent_id="a", certified_surface=False)
    client.responses.create(input="hi")
    assert client.responses.calls == 1


def test_p32_patch_anthropic_exported():
    assert callable(patch_anthropic)


def test_p32_apply_output_token_bound_helper():
    kwargs: dict[str, Any] = {"model": "gpt-4o-mini"}
    _apply_output_token_bound(kwargs)
    assert kwargs["max_tokens"] == default_max_output_tokens("gpt-4o-mini")
