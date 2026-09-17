from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


class KazenCircuitBreaker(RuntimeError):
    """Raised when Agent FinOps pauses execution before budget exhaustion."""

    def __init__(self, message: str, *, resume_token: str = "", context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.resume_token = str(resume_token or "")
        self.context = context or {}


@dataclass(frozen=True)
class CircuitBreakerDecision:
    opened: bool
    reason: str
    resume_token: str = ""


class StateSerializer:
    """
    User-supplied interface to snapshot/resume a running agent.

    SDK stays framework-agnostic: adapters can implement this for specific frameworks.
    """

    def snapshot(self, *, run_ctx: Dict[str, Any], opaque_state: Any) -> str:  # pragma: no cover
        raise NotImplementedError

    def resume(self, *, resume_token: str) -> Any:  # pragma: no cover
        raise NotImplementedError


class CircuitBreaker:
    def __init__(
        self,
        *,
        budget_usd: Optional[float],
        soft_pause_pct: float = 0.90,
        serializer: Optional[StateSerializer] = None,
    ) -> None:
        self._budget = float(budget_usd) if budget_usd is not None else None
        self._soft_pct = float(soft_pause_pct)
        self._serializer = serializer
        self._opened = False

    def evaluate(
        self,
        *,
        projected_total_cost_usd: float,
        current_spend_usd: float,
        run_ctx: Dict[str, Any],
        opaque_state: Any = None,
    ) -> CircuitBreakerDecision:
        if self._budget is None:
            return CircuitBreakerDecision(opened=False, reason="")
        if self._opened:
            return CircuitBreakerDecision(opened=True, reason="already_open")
        soft = self._budget * self._soft_pct
        if float(projected_total_cost_usd) <= float(soft):
            return CircuitBreakerDecision(opened=False, reason="")

        self._opened = True
        token = ""
        if self._serializer is not None:
            try:
                token = str(self._serializer.snapshot(run_ctx=run_ctx, opaque_state=opaque_state) or "")
            except Exception:
                token = ""
        return CircuitBreakerDecision(
            opened=True,
            reason="projected_cost_exceeds_soft_threshold",
            resume_token=token,
        )

    def reset(self) -> None:
        self._opened = False

