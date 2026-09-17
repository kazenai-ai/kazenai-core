"""FINAL_1 P0-2 / P3-1 — durable regression for G06 public monitor tiny-cap.

``monitor(..., max_budget_usd=...)`` must deny before the provider is invoked when the
projected call exceeds the local cap. Closed by P3-1 (Enforcement wired + pending hold).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from kazenai import BudgetExceeded, monitor
from kazenai.monitor import monitor as monitor_fn


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=1000, total_tokens=2000),
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        )


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeOpenAI:
    def __init__(self) -> None:
        self.chat = FakeChat()


def test_final1_g06_monitor_signature_has_no_api_key_parameter():
    """Documented README snippets pass api_key=...; the real signature does not."""
    sig = inspect.signature(monitor_fn)
    assert "api_key" not in sig.parameters
    assert "max_budget_usd" in sig.parameters
    # api_key is env-only today (KAZENAI_FINOPS_API_KEY / KAZENAI_API_KEY).


def test_final1_g06_tiny_cap_denies_before_provider_invocation(monkeypatch):
    """$0.000001 local cap + known higher-priced fake call → provider invocations == 0.

    Isolates the *local* public-monitor contract: no FinOps URL, fail-open reservation so
    shared authority is not the deny path. max_budget_usd must still pre-call block via
    Enforcement (or equivalent).
    """
    monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)
    monkeypatch.delenv("KAZENAI_FINOPS_INGEST_URL", raising=False)
    monkeypatch.delenv("KAZENAI_INGEST_URL", raising=False)
    # fail_open is honored only when KAZENAI_ENV == "dev" (exact token).
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")

    client = FakeOpenAI()
    monitor(
        client,
        org_id="org-final1-p02",
        project_id="proj-final1-p02",
        workspace_id="ws-final1-p02",
        agent_id="agent-tiny-cap",
        max_budget_usd=0.000001,
    )

    try:
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
        )
        raised: Optional[BaseException] = None
    except BaseException as exc:  # noqa: BLE001 — capture exact deny path
        raised = exc

    assert client.chat.completions.calls == 0, (
        f"G06 tiny-cap bypass: expected 0 provider invocations before denial, "
        f"observed {client.chat.completions.calls}; raised={type(raised).__name__ if raised else None}. "
        f"max_budget_usd is not wired into Enforcement on the public monitor() path."
    )
    assert isinstance(raised, BudgetExceeded), (
        f"G06 expected BudgetExceeded before provider call, got {type(raised).__name__ if raised else None}"
    )


def test_final1_g06_budget_exceeded_is_exported():
    from kazenai import __all__ as public_all

    assert "BudgetExceeded" in public_all
    assert issubclass(BudgetExceeded, Exception)
