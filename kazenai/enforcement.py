from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Mapping, Optional

try:
    import structlog  # type: ignore
except Exception:  # pragma: no cover
    structlog = None
    import logging


class KazenBlock(RuntimeError):
    """Raised when local-first enforcement blocks an LLM call."""


class BudgetExceeded(KazenBlock):
    pass


class BudgetUnavailable(KazenBlock):
    """Raised when FinOps is unreachable and enforcement mode is fail_closed."""


class RateLimitExceeded(KazenBlock):
    pass


class LoopDetected(KazenBlock):
    pass


class StreamCutoffError(KazenBlock):
    """Raised when mid-stream FinOps budget enforcement severs an SSE response."""

    def __init__(
        self,
        message: str = "stream cutoff: budget exceeded",
        *,
        blocked_usd: float = 0.0,
        total_tokens: int = 0,
    ) -> None:
        super().__init__(message)
        self.blocked_usd = blocked_usd
        self.total_tokens = total_tokens


class UnsupportedModeError(KazenBlock):
    """Raised when a Control-unsupported call mode is requested (e.g. streaming)."""

    def __init__(
        self,
        message: str = "unsupported_mode",
        *,
        error_code: str = "unsupported_mode",
    ) -> None:
        super().__init__(message)
        self.error_code = error_code


class Enforcement:
    """
    Fast local-first enforcement for:
      - cumulative_cost (USD)
      - calls_per_minute (sliding window)

    Fail-open: internal errors are caught and logged (except explicit blocks).
    """

    def __init__(
        self,
        *,
        max_cost_usd: Optional[float] = None,
        calls_per_minute: Optional[int] = None,
        window_seconds: float = 60.0,
        logger: Optional[Any] = None,
    ) -> None:
        self._max_cost_usd = float(max_cost_usd) if max_cost_usd is not None else None
        self._calls_per_minute = int(calls_per_minute) if calls_per_minute is not None else None
        self._window_seconds = float(window_seconds)

        self._lock = threading.Lock()
        self._cumulative_cost_usd = 0.0
        self._pending_projected_usd = 0.0
        self._call_timestamps: deque[float] = deque()
        if logger is not None:
            self._log = logger
        elif structlog is not None:
            self._log = structlog.get_logger(__name__)
        else:
            self._log = logging.getLogger(__name__)

    def snapshot(self) -> dict[str, Any]:
        try:
            now = time.monotonic()
            with self._lock:
                self._prune_locked(now)
                return {
                    "cumulative_cost_usd": self._cumulative_cost_usd,
                    "pending_projected_usd": self._pending_projected_usd,
                    "calls_in_window": len(self._call_timestamps),
                    "max_cost_usd": self._max_cost_usd,
                    "calls_per_minute": self._calls_per_minute,
                    "window_seconds": self._window_seconds,
                }
        except Exception:
            self._log.exception("kazenai.enforcement.snapshot_failed")
            return {}

    def check_local(self, *, projected_cost_usd: float = 0.0) -> float:
        """
        Synchronous hot-path check. Raises on block.

        When capped, reserves ``projected_cost_usd`` against pending exposure so concurrent
        callers cannot all slip under the same remaining headroom. Returns the reserved
        projection so ``record_call(..., released_projection=...)`` can release it.
        """

        try:
            now = time.monotonic()
            with self._lock:
                self._prune_locked(now)

                if self._calls_per_minute is not None and len(self._call_timestamps) >= self._calls_per_minute:
                    raise RateLimitExceeded(
                        f"calls_per_minute exceeded: limit={self._calls_per_minute} window={self._window_seconds}s"
                    )

                proj = float(projected_cost_usd or 0.0)
                if self._max_cost_usd is not None:
                    projected_total = (
                        self._cumulative_cost_usd + self._pending_projected_usd + proj
                    )
                    if projected_total > self._max_cost_usd:
                        raise BudgetExceeded(
                            f"cumulative_cost exceeded: total={projected_total:.6f} limit={self._max_cost_usd:.6f}"
                        )
                    if proj > 0:
                        self._pending_projected_usd += proj
                return proj
        except (BudgetExceeded, RateLimitExceeded):
            raise
        except Exception:
            self._log.exception("kazenai.enforcement.check_local_failed")
            return 0.0

    def remaining_budget_usd(self) -> Optional[float]:
        """Headroom (USD) left under the cost cap, or None when uncapped.

        Lets a long-running caller cap a single run to the remaining budget and halt
        mid-run instead of overshooting once on an expensive call (SEC-R2-9).
        """
        if self._max_cost_usd is None:
            return None
        with self._lock:
            used = self._cumulative_cost_usd + self._pending_projected_usd
            return max(0.0, self._max_cost_usd - used)

    def record_call(
        self,
        *,
        cost_usd: Optional[float],
        enforce: bool = False,
        released_projection: float = 0.0,
    ) -> None:
        """Record a call's cost. With ``enforce=True``, raise once cumulative crosses the cap.

        ``enforce`` is for per-turn / per-chunk streamed metering: a run that calls
        ``record_call(cost_usd=..., enforce=True)`` after each turn halts the instant its
        streamed cost crosses budget (StreamCutoffError), bounding overshoot to a single
        turn rather than letting one expensive run blow past the cap unchecked. Existing
        callers (default ``enforce=False``) keep the original never-raises behavior.

        ``released_projection`` releases a prior ``check_local`` hold so pending exposure
        does not double-count with the recorded actual.
        """
        try:
            now = time.monotonic()
            with self._lock:
                self._prune_locked(now)
                self._call_timestamps.append(now)
                released = float(released_projection or 0.0)
                if released > 0:
                    self._pending_projected_usd = max(0.0, self._pending_projected_usd - released)
                if cost_usd is not None:
                    self._cumulative_cost_usd += float(cost_usd)
                over_budget = (
                    enforce
                    and self._max_cost_usd is not None
                    and self._cumulative_cost_usd > self._max_cost_usd
                )
                cumulative = self._cumulative_cost_usd
                limit = self._max_cost_usd
        except Exception:
            self._log.exception("kazenai.enforcement.record_call_failed")
            return
        if over_budget:
            raise StreamCutoffError(
                f"stream cutoff: budget exceeded mid-run total={cumulative:.6f} limit={limit:.6f}",
                blocked_usd=float(cost_usd or 0.0),
            )

    async def apply_server_decision(self, decision: Mapping[str, Any]) -> None:
        """
        Async hook to update local limits from the backend.

        Expected keys (optional): max_cost_usd, calls_per_minute, window_seconds
        """

        try:
            if "window_seconds" in decision and decision["window_seconds"] is not None:
                self._window_seconds = float(decision["window_seconds"])

            if "max_cost_usd" in decision:
                val = decision["max_cost_usd"]
                self._max_cost_usd = None if val is None else float(val)

            if "calls_per_minute" in decision:
                val = decision["calls_per_minute"]
                self._calls_per_minute = None if val is None else int(val)
        except Exception:
            self._log.exception("kazenai.enforcement.apply_server_decision_failed")

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self._window_seconds
        dq = self._call_timestamps
        while dq and dq[0] < cutoff:
            dq.popleft()
