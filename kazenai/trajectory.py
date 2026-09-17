from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrajectoryPoint:
    step_index: int
    current_spend_usd: float
    projected_total_cost_usd: float
    budget_usd: Optional[float]
    soft_pause_threshold_usd: Optional[float]


class CostTrajectoryPredictor:
    """
    O(1) per-step predictor using exponential smoothing of cost-per-step, with a
    simple fan-out/depth multiplier hook.

    This is intentionally lightweight and framework-agnostic.
    """

    def __init__(
        self,
        *,
        budget_usd: Optional[float],
        soft_pause_pct: float = 0.90,
        alpha: float = 0.3,
    ) -> None:
        self._budget = float(budget_usd) if budget_usd is not None else None
        self._soft_pause_pct = float(soft_pause_pct)
        self._alpha = float(alpha)
        self._step = 0
        self._ema_cost_per_step: float = 0.0
        self._current_spend: float = 0.0
        self._max_depth_seen: int = 0

    def update(
        self,
        *,
        step_cost_usd: float,
        depth: int = 0,
    ) -> TrajectoryPoint:
        self._step += 1
        self._current_spend += float(step_cost_usd or 0.0)
        if depth > self._max_depth_seen:
            self._max_depth_seen = int(depth)

        # EMA of per-step cost.
        c = float(step_cost_usd or 0.0)
        if self._step == 1:
            self._ema_cost_per_step = c
        else:
            self._ema_cost_per_step = (self._alpha * c) + ((1.0 - self._alpha) * self._ema_cost_per_step)

        projected = self._current_spend
        if self._budget is not None:
            # Predict remaining steps as a function of depth growth (fan-out proxy).
            # This is deliberately simple; it produces a "heads up" line, not a guarantee.
            depth_multiplier = 1.0 + max(0.0, float(self._max_depth_seen)) * 0.10
            projected = self._current_spend + (self._ema_cost_per_step * depth_multiplier * 5.0)

        soft_pause = (self._budget * self._soft_pause_pct) if self._budget is not None else None
        return TrajectoryPoint(
            step_index=self._step,
            current_spend_usd=float(self._current_spend),
            projected_total_cost_usd=float(projected),
            budget_usd=self._budget,
            soft_pause_threshold_usd=soft_pause,
        )

