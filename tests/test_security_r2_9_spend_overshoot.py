"""SEC-R2-9 abuse regression (core): spend cap must not overshoot on one expensive run.

Local-first enforcement previously checked the budget pre-call with ``projected_cost_usd=0.0``
and only recorded the real cost afterwards, so a single very-expensive run at the budget edge
slipped past the gate and the cap was only enforced on the *next* call — overshooting once.

Two layers close this:
  1. The pre-call projection is now derived from the real per-model price table
     (`monitor._precall_projection_usd` → `cost_engine`), so an expensive-model call with no
     headroom is denied BEFORE it runs.
  2. `Enforcement.record_call(enforce=True)` halts a streamed/multi-turn run the instant its
     metered cost crosses the cap, bounding overshoot to a single turn, and
     `remaining_budget_usd()` exposes headroom so a caller can cap a run mid-flight.
"""

from __future__ import annotations

import pytest

from kazenai.enforcement import BudgetExceeded, Enforcement, StreamCutoffError
from kazenai.monitor import _precall_projection_usd


# ---------------------------------------------------------------------------
# Layer 1 — price-table-derived pre-call floor blocks the expensive run up front.
# ---------------------------------------------------------------------------

def test_precall_floor_matches_price_table_and_ranks_models():
    cheap = _precall_projection_usd({"model": "gpt-4o-mini"}, None)
    pricey = _precall_projection_usd({"model": "claude-opus-4-8"}, None)
    assert pricey > cheap > 0
    # An unknown-but-priced model falls back to usd_per_1k_tokens; no model + no price -> 0.
    assert _precall_projection_usd({}, 0.03) == pytest.approx(0.03 * (2000 / 1000))
    assert _precall_projection_usd({}, None) == 0.0


def test_expensive_run_blocked_before_it_starts_at_budget_edge():
    # Cap $1.00, already spent $0.95 -> $0.05 headroom.
    e = Enforcement(max_cost_usd=1.00)
    e.record_call(cost_usd=0.95)
    # A cheap model whose floor fits the headroom is allowed through the pre-call gate.
    e.check_local(projected_cost_usd=_precall_projection_usd({"model": "gpt-4o-mini"}, None))
    # An expensive Opus run, projected from the price table, exceeds headroom -> denied
    # BEFORE the call executes (no overshoot).
    with pytest.raises(BudgetExceeded):
        e.check_local(projected_cost_usd=_precall_projection_usd({"model": "claude-opus-4-8"}, None))


# ---------------------------------------------------------------------------
# Layer 2 — streamed/per-turn metering halts mid-run and headroom is exposed.
# ---------------------------------------------------------------------------

def test_remaining_budget_usd_reports_headroom():
    e = Enforcement(max_cost_usd=2.0)
    assert e.remaining_budget_usd() == pytest.approx(2.0)
    e.record_call(cost_usd=0.5)
    assert e.remaining_budget_usd() == pytest.approx(1.5)
    assert Enforcement().remaining_budget_usd() is None  # uncapped


def test_streamed_run_halts_mid_run_when_cost_crosses_cap():
    e = Enforcement(max_cost_usd=1.0)
    # Simulate a run metering cost per turn; enforce=True hard-stops the moment it crosses.
    turns = [0.3, 0.3, 0.3, 0.3, 0.3]  # would reach $1.50 unchecked
    completed_turns = 0
    with pytest.raises(StreamCutoffError):
        for c in turns:
            e.record_call(cost_usd=c, enforce=True)  # raises on the 4th turn ($1.20)
            completed_turns += 1
    # Three turns completed without raising; the 4th turn's cost was metered then the run was
    # halted — overshoot bounded to a single turn ($1.20), not the full $1.50 run.
    assert completed_turns == 3
    assert e.snapshot()["cumulative_cost_usd"] == pytest.approx(1.2)


def test_enforce_false_preserves_legacy_non_raising_behavior():
    e = Enforcement(max_cost_usd=0.5)
    # Default enforce=False never raises even far over budget (existing callers rely on this).
    for _ in range(5):
        e.record_call(cost_usd=0.5)
    assert e.snapshot()["cumulative_cost_usd"] == pytest.approx(2.5)
