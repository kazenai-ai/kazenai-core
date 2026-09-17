"""Cost trajectory / spend projection tests."""

from __future__ import annotations

import pytest

from kazenai.trajectory import CostTrajectoryPredictor


def test_first_step_sets_spend_and_projection():
    pred = CostTrajectoryPredictor(budget_usd=10.0, soft_pause_pct=0.9)
    point = pred.update(step_cost_usd=0.5, depth=0)
    assert point.step_index == 1
    assert point.current_spend_usd == pytest.approx(0.5)
    assert point.budget_usd == 10.0
    assert point.soft_pause_threshold_usd == pytest.approx(9.0)


def test_accumulates_spend_across_steps():
    pred = CostTrajectoryPredictor(budget_usd=5.0)
    pred.update(step_cost_usd=1.0)
    point = pred.update(step_cost_usd=2.0)
    assert point.current_spend_usd == pytest.approx(3.0)


def test_depth_increases_projection():
    pred = CostTrajectoryPredictor(budget_usd=20.0)
    low = pred.update(step_cost_usd=1.0, depth=0)
    pred2 = CostTrajectoryPredictor(budget_usd=20.0)
    high = pred2.update(step_cost_usd=1.0, depth=5)
    assert high.projected_total_cost_usd >= low.projected_total_cost_usd


def test_no_budget_soft_threshold_none():
    pred = CostTrajectoryPredictor(budget_usd=None)
    point = pred.update(step_cost_usd=1.0)
    assert point.soft_pause_threshold_usd is None
    assert point.projected_total_cost_usd == pytest.approx(1.0)


def test_zero_step_cost():
    pred = CostTrajectoryPredictor(budget_usd=1.0)
    point = pred.update(step_cost_usd=0.0)
    assert point.current_spend_usd == 0.0
