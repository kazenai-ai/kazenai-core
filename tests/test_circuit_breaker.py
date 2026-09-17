"""Circuit breaker and KazenCircuitBreaker tests."""

from __future__ import annotations

import pytest

from kazenai.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerDecision,
    KazenCircuitBreaker,
    StateSerializer,
)


class _FakeSerializer(StateSerializer):
    def snapshot(self, *, run_ctx, opaque_state):
        return f"token-{run_ctx.get('run_id', 'x')}"

    def resume(self, *, resume_token: str):
        return {"resumed": resume_token}


def test_no_budget_never_opens():
    cb = CircuitBreaker(budget_usd=None)
    dec = cb.evaluate(
        projected_total_cost_usd=100.0,
        current_spend_usd=50.0,
        run_ctx={"run_id": "r1"},
    )
    assert dec.opened is False
    assert dec.reason == ""


def test_below_soft_threshold_stays_closed():
    cb = CircuitBreaker(budget_usd=10.0, soft_pause_pct=0.9)
    dec = cb.evaluate(
        projected_total_cost_usd=8.0,
        current_spend_usd=5.0,
        run_ctx={"run_id": "r1"},
    )
    assert dec.opened is False


def test_exceeds_soft_threshold_opens_with_token():
    cb = CircuitBreaker(budget_usd=10.0, soft_pause_pct=0.9, serializer=_FakeSerializer())
    dec = cb.evaluate(
        projected_total_cost_usd=9.5,
        current_spend_usd=8.0,
        run_ctx={"run_id": "run-abc"},
    )
    assert dec.opened is True
    assert dec.reason == "projected_cost_exceeds_soft_threshold"
    assert dec.resume_token == "token-run-abc"


def test_already_open_stays_open():
    cb = CircuitBreaker(budget_usd=1.0, soft_pause_pct=0.5)
    cb.evaluate(projected_total_cost_usd=1.0, current_spend_usd=0.5, run_ctx={})
    dec = cb.evaluate(projected_total_cost_usd=0.1, current_spend_usd=0.1, run_ctx={})
    assert dec.opened is True
    assert dec.reason == "already_open"


def test_reset_allows_reopen():
    cb = CircuitBreaker(budget_usd=1.0, soft_pause_pct=0.5)
    cb.evaluate(projected_total_cost_usd=1.0, current_spend_usd=0.5, run_ctx={})
    cb.reset()
    dec = cb.evaluate(projected_total_cost_usd=0.1, current_spend_usd=0.1, run_ctx={})
    assert dec.opened is False


def test_serializer_failure_still_opens():
    class _BrokenSerializer(StateSerializer):
        def snapshot(self, *, run_ctx, opaque_state):
            raise RuntimeError("snap failed")

        def resume(self, *, resume_token: str):
            return None

    cb = CircuitBreaker(budget_usd=1.0, soft_pause_pct=0.5, serializer=_BrokenSerializer())
    dec = cb.evaluate(projected_total_cost_usd=1.0, current_spend_usd=0.5, run_ctx={})
    assert dec.opened is True
    assert dec.resume_token == ""


def test_kazen_circuit_breaker_carries_context():
    err = KazenCircuitBreaker("paused", resume_token="tok", context={"run_id": "r1"})
    assert err.resume_token == "tok"
    assert err.context["run_id"] == "r1"
    with pytest.raises(KazenCircuitBreaker):
        raise err


def test_circuit_breaker_decision_frozen():
    dec = CircuitBreakerDecision(opened=True, reason="x", resume_token="t")
    assert dec.opened is True
