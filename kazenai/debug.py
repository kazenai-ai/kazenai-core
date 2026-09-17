from __future__ import annotations

import sys
from typing import Any, Optional, TextIO

from .loop_detector import LoopScores


class _C:
    RESET = "\x1b[0m"
    DIM = "\x1b[2m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    CYAN = "\x1b[36m"


class DebugPrinter:
    def __init__(self, *, enabled: bool, stream: Optional[TextIO] = None) -> None:
        self._enabled = bool(enabled)
        self._out = stream or sys.stdout

    def pre_call(
        self,
        *,
        step_id: str,
        parent_step_id: Optional[str],
        projected_cost_usd: Optional[float],
        cumulative_cost_usd: Optional[float],
        calls_in_window: Optional[int],
        loop_scores: Optional[LoopScores],
    ) -> None:
        if not self._enabled:
            return
        try:
            loop = loop_scores.combined if loop_scores else 0.0
            loop_color = _C.GREEN if loop < 0.6 else (_C.YELLOW if loop < 0.85 else _C.RED)
            parent = parent_step_id or "-"
            msg = (
                f"{_C.CYAN}kazenai{_C.RESET} "
                f"{_C.DIM}pre{_C.RESET} "
                f"step={step_id} parent={parent} "
                f"proj=${(projected_cost_usd or 0.0):.6f} "
                f"total=${(cumulative_cost_usd or 0.0):.6f} "
                f"cpm={calls_in_window if calls_in_window is not None else '-'} "
                f"loop={loop_color}{loop:.3f}{_C.RESET}"
            )
            print(msg, file=self._out, flush=True)
        except Exception:
            return

    def post_call(
        self,
        *,
        step_id: str,
        step_cost_usd: Optional[float],
        tokens_used: Optional[int],
        cumulative_cost_usd: Optional[float],
        loop_scores: Optional[LoopScores],
    ) -> None:
        if not self._enabled:
            return
        try:
            loop = loop_scores.combined if loop_scores else 0.0
            msg = (
                f"{_C.CYAN}kazenai{_C.RESET} "
                f"{_C.DIM}post{_C.RESET} "
                f"step={step_id} "
                f"cost=${(step_cost_usd or 0.0):.6f} "
                f"tokens={tokens_used if tokens_used is not None else '-'} "
                f"total=${(cumulative_cost_usd or 0.0):.6f} "
                f"loop={loop:.3f}"
            )
            print(msg, file=self._out, flush=True)
        except Exception:
            return
