"""One local spending cap shared by several monitor() clients.

Each ``monitor(..., max_budget_usd=...)`` client enforces its own cap. A
workflow whose steps use different models or providers usually has several
clients, so per-client caps add up: three clients at $1.00 can spend $3.00.
Pass one :class:`SharedBudget` to all of them instead::

    budget = SharedBudget(max_usd=1.00)
    openai_client = monitor(OpenAI(), budget=budget)
    anthropic_client = monitor(Anthropic(), budget=budget)

Every call, on any of those clients, checks and charges the same budget
before dispatch (thread- and asyncio-safe, in-flight calls included).

Scope: one Python process. Budgets shared across processes or machines need
the shared Agent FinOps authority (``workflow_id`` + a workflow limit).
"""

from __future__ import annotations

from typing import Any, Optional

from .enforcement import Enforcement


class SharedBudget:
    """A process-local USD cap shared across monitor() clients."""

    def __init__(self, max_usd: float, *, name: str = "") -> None:
        if max_usd is None or float(max_usd) < 0:
            raise ValueError("SharedBudget max_usd must be a non-negative number")
        self.max_usd = float(max_usd)
        self.name = name
        self._enforcement = Enforcement(max_cost_usd=self.max_usd)

    @property
    def enforcement(self) -> Enforcement:
        return self._enforcement

    @property
    def spent_usd(self) -> float:
        return float(self._enforcement.snapshot().get("cumulative_cost_usd", 0.0))

    @property
    def remaining_usd(self) -> Optional[float]:
        return self._enforcement.remaining_budget_usd()

    def snapshot(self) -> dict[str, Any]:
        return {"name": self.name, **self._enforcement.snapshot()}

    def __repr__(self) -> str:
        label = f" {self.name!r}" if self.name else ""
        return f"<SharedBudget{label} max_usd={self.max_usd:.6f} spent_usd={self.spent_usd:.6f}>"
