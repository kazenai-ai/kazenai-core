"""Local enforcement (budget / rate limit) tests."""

from __future__ import annotations

import time

import pytest

from kazenai.enforcement import BudgetExceeded, Enforcement, RateLimitExceeded


def test_record_call_accumulates_cost():
    e = Enforcement(max_cost_usd=10.0)
    e.record_call(cost_usd=0.5)
    e.record_call(cost_usd=0.25)
    snap = e.snapshot()
    assert snap["cumulative_cost_usd"] == pytest.approx(0.75)


def test_check_local_blocks_over_budget():
    e = Enforcement(max_cost_usd=1.0)
    e.record_call(cost_usd=0.9)
    with pytest.raises(BudgetExceeded):
        e.check_local(projected_cost_usd=0.2)


def test_rate_limit_blocks_burst():
    e = Enforcement(calls_per_minute=2, window_seconds=60.0)
    e.record_call(cost_usd=None)
    e.record_call(cost_usd=None)
    with pytest.raises(RateLimitExceeded):
        e.check_local()


def test_apply_server_decision_updates_limits():
    e = Enforcement(max_cost_usd=1.0, calls_per_minute=10)

    async def _run():
        await e.apply_server_decision({"max_cost_usd": 5.0, "calls_per_minute": 3})

    import asyncio

    asyncio.run(_run())
    snap = e.snapshot()
    assert snap["max_cost_usd"] == 5.0
    assert snap["calls_per_minute"] == 3


def test_prune_old_calls_from_window():
    e = Enforcement(calls_per_minute=1, window_seconds=0.05)
    e.record_call(cost_usd=None)
    time.sleep(0.06)
    e.check_local()  # should not raise after prune


def test_snapshot_fail_open_returns_dict():
    e = Enforcement()
    assert isinstance(e.snapshot(), dict)


def test_rate_limit_exceeded_message():
    e = Enforcement(calls_per_minute=1, window_seconds=60.0)
    e.record_call(cost_usd=None)
    with pytest.raises(RateLimitExceeded) as exc:
        e.check_local()
    assert "calls_per_minute" in str(exc.value)


def test_budget_exceeded_message():
    e = Enforcement(max_cost_usd=0.5)
    e.record_call(cost_usd=0.4)
    with pytest.raises(BudgetExceeded) as exc:
        e.check_local(projected_cost_usd=0.2)
    assert "cumulative_cost" in str(exc.value)
