"""FINAL_1 P3-1 — public monitor max_budget_usd enforcement (extends G06)."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any, List, Optional

import pytest

from kazenai import BudgetExceeded, monitor
from kazenai.enforcement import Enforcement
from kazenai.monitor import _precall_projection_usd


class FakeCompletions:
    def __init__(self, *, hold: Optional[threading.Event] = None) -> None:
        self.calls = 0
        self.lock = threading.Lock()
        self._hold = hold
        self.in_provider = threading.Event()

    def create(self, *args: Any, **kwargs: Any) -> Any:
        with self.lock:
            self.calls += 1
        self.in_provider.set()
        if self._hold is not None:
            self._hold.wait(timeout=5)
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        )


class FakeOpenAI:
    def __init__(self, *, hold: Optional[threading.Event] = None) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(hold=hold))


def _standalone_env(monkeypatch) -> None:
    monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)
    monkeypatch.delenv("KAZENAI_FINOPS_INGEST_URL", raising=False)
    monkeypatch.delenv("KAZENAI_INGEST_URL", raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")


def test_p31_enforcement_pending_blocks_concurrent_projection():
    """Direct unit proof: in-flight projected hold consumes shared headroom."""
    projected = 1.0
    enf = Enforcement(max_cost_usd=projected * 1.1)
    held = enf.check_local(projected_cost_usd=projected)
    assert held == projected
    with pytest.raises(BudgetExceeded):
        enf.check_local(projected_cost_usd=projected)
    enf.record_call(cost_usd=0.05, released_projection=held)
    snap = enf.snapshot()
    assert snap["pending_projected_usd"] == 0.0
    assert snap["cumulative_cost_usd"] == pytest.approx(0.05)


def test_p31_cap_boundary_allows_when_projection_fits(monkeypatch):
    """Cap at exactly the projected floor must allow (projected_total > limit denies)."""
    _standalone_env(monkeypatch)
    projected = _precall_projection_usd({"model": "gpt-4o"}, None)
    assert projected > 0
    client = FakeOpenAI()
    monitor(client, max_budget_usd=projected)  # equal → allowed (check uses >)
    client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert client.chat.completions.calls == 1


def test_p31_sequential_calls_exhaust_local_cap(monkeypatch):
    _standalone_env(monkeypatch)
    client = FakeOpenAI()
    first_proj = _precall_projection_usd({"model": "gpt-4o-mini"}, None)
    assert first_proj > 0
    # Cap just above one projected mini call so a subsequent gpt-4o projection exceeds.
    cap = first_proj * 1.5
    monitor(client, max_budget_usd=cap)

    client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "a"}])
    assert client.chat.completions.calls == 1

    try:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "b"}])
        raised = None
    except BudgetExceeded as exc:
        raised = exc
    assert isinstance(raised, BudgetExceeded)
    assert client.chat.completions.calls == 1


def test_p31_concurrent_calls_respect_shared_local_enforcement(monkeypatch):
    """While one call holds projected spend, siblings must deny before provider."""
    _standalone_env(monkeypatch)
    projected = _precall_projection_usd({"model": "gpt-4o"}, None)
    hold = threading.Event()
    client = FakeOpenAI(hold=hold)
    monitor(client, max_budget_usd=projected * 1.1)

    results: List[Optional[BaseException]] = []
    results_lock = threading.Lock()

    def _one(idx: int) -> None:
        try:
            client.chat.completions.create(
                model="gpt-4o",
                # Distinct text avoids LoopDetected before the budget check.
                messages=[{"role": "user", "content": f"concurrent-{idx}-{time.time_ns()}"}],
            )
            with results_lock:
                results.append(None)
        except BaseException as exc:  # noqa: BLE001
            with results_lock:
                results.append(exc)

    # Admit one call into the provider, then race the rest while pending is held.
    first = threading.Thread(target=_one, args=(0,))
    first.start()
    assert client.chat.completions.in_provider.wait(timeout=5)

    rest = [threading.Thread(target=_one, args=(i,)) for i in range(1, 8)]
    for t in rest:
        t.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with results_lock:
            # All siblings must finish (deny) before we release the first call's hold.
            if len(results) >= 7:
                break
        time.sleep(0.01)

    with results_lock:
        assert len(results) >= 7
        assert all(isinstance(r, BudgetExceeded) for r in results)
    assert client.chat.completions.calls == 1

    hold.set()
    first.join(timeout=5)
    for t in rest:
        t.join(timeout=5)

    allowed = sum(1 for r in results if r is None)
    denied = sum(1 for r in results if isinstance(r, BudgetExceeded))
    assert allowed + denied == 8
    assert allowed == 1
    assert denied == 7
    assert client.chat.completions.calls == 1


def test_p31_unknown_model_projection_is_zero_without_usd_hint(monkeypatch):
    """Unknown model → projection 0 → local cap alone does not block (documented)."""
    _standalone_env(monkeypatch)
    assert _precall_projection_usd({"model": "totally-unknown-model-xyz"}, None) == 0.0
    client = FakeOpenAI()
    monitor(client, max_budget_usd=0.000001)
    client.chat.completions.create(
        model="totally-unknown-model-xyz",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert client.chat.completions.calls == 1


def test_p31_finops_unavailable_still_honors_local_cap(monkeypatch):
    """Shared authority unreachable + fail_open reservation must not bypass local cap."""
    _standalone_env(monkeypatch)
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://127.0.0.1:1")  # nothing listening
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "k")
    client = FakeOpenAI()
    monitor(client, max_budget_usd=0.000001)
    with pytest.raises(BudgetExceeded):
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert client.chat.completions.calls == 0
