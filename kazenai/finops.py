from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .alerts import AlertDispatcher
from .circuit_breaker import CircuitBreaker, KazenCircuitBreaker
from .cost_engine import TokenCostEngine
from .schema import KazenEvent, new_id, now_ms
from .trajectory import CostTrajectoryPredictor


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _tokenize(text: str) -> set[str]:
    out: set[str] = set()
    for raw in (text or "").lower().split():
        t = "".join(ch for ch in raw if ch.isalnum())
        if t:
            out.add(t)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return (inter / union) if union else 0.0


@dataclass
class FinOpsConfig:
    budget_usd: Optional[float] = None
    soft_pause_pct: float = 0.90
    loop_anomaly_threshold: float = 0.90


class FinOpsController:
    """
    Consumes model.call events (llm.call accepted as alias), emits derived events:
    - finops.trajectory
    - finops.circuit_breaker.opened
    - finops.loop.anomaly
    """

    def __init__(
        self,
        *,
        cfg: FinOpsConfig,
        cost_engine: Optional[TokenCostEngine] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        alerting: Optional[AlertDispatcher] = None,
    ) -> None:
        self._cfg = cfg
        self._cost_engine = cost_engine or TokenCostEngine()
        self._trajectory = CostTrajectoryPredictor(budget_usd=cfg.budget_usd, soft_pause_pct=cfg.soft_pause_pct)
        self._cb = circuit_breaker or CircuitBreaker(budget_usd=cfg.budget_usd, soft_pause_pct=cfg.soft_pause_pct)
        self._alert = alerting or AlertDispatcher()

        self._last_output_tokens: Optional[set[str]] = None
        self._last_cost_usd: float = 0.0

    def handle_llm_call(self, event: KazenEvent) -> Dict[str, KazenEvent]:
        # Ensure event has cost; if missing but we have usage-ish fields in payload, compute.
        cost_usd = event.cost_usd
        tokens_used = event.tokens_used
        payload = dict(event.payload or {})

        model = str(payload.get("model") or payload.get("resolved_model") or payload.get("method") or "")
        usage = payload.get("usage")
        if cost_usd is None and usage is not None and model:
            _, _, computed = self._cost_engine.from_usage(model=model, usage=usage)
            cost_usd = computed

        if tokens_used is None:
            try:
                tokens_used = int(payload.get("input_tokens") or 0) + int(payload.get("output_tokens") or 0)
            except Exception:
                tokens_used = None

        step_cost = float(cost_usd or 0.0)
        depth = 0
        try:
            depth = int(payload.get("depth") or 0)
        except Exception:
            depth = 0

        point = self._trajectory.update(step_cost_usd=step_cost, depth=depth)

        derived: Dict[str, KazenEvent] = {}

        remaining_budget = None
        if point.budget_usd is not None:
            remaining_budget = max(0.0, float(point.budget_usd) - float(point.current_spend_usd))
        cost_quality_score = None
        if point.projected_total_cost_usd > 0:
            # Until an eval model is wired in, use budget survival as the
            # conservative quality proxy: lower projected spend per budget dollar
            # receives a higher score.
            budget = float(point.budget_usd or point.projected_total_cost_usd)
            expected_success_probability = max(
                0.0,
                min(1.0, 1.0 - (float(point.projected_total_cost_usd) / max(budget, 0.000001))),
            )
            cost_quality_score = expected_success_probability / float(point.projected_total_cost_usd)
        else:
            expected_success_probability = None

        traj = KazenEvent(
            schema_version="1.2",
            ts_ms=now_ms(),
            event_id=new_id(),
            org_id=event.org_id,
            project_id=event.project_id,
            surface=event.surface,
            agent_id=event.agent_id,
            agent_role=event.agent_role,
            run_id=event.run_id,
            step_id=new_id(),
            parent_step_id=event.step_id,
            event_type="finops.trajectory",
            tokens_used=tokens_used,
            cost_usd=step_cost,
            stage_budget_usd=point.budget_usd,
            remaining_budget_usd=remaining_budget,
            projected_total_cost_usd=point.projected_total_cost_usd,
            expected_success_probability=expected_success_probability,
            cost_quality_score=cost_quality_score,
            payload={
                "step_index": point.step_index,
                "current_spend_usd": point.current_spend_usd,
                "projected_total_cost_usd": point.projected_total_cost_usd,
                "budget_usd": point.budget_usd,
                "soft_pause_threshold_usd": point.soft_pause_threshold_usd,
                "expected_success_probability": expected_success_probability,
                "cost_quality_score": cost_quality_score,
                "model": model,
            },
        )
        derived["trajectory"] = traj

        # Loop anomaly: rising cost with diminishing novelty.
        novelty = None
        try:
            out_text = str(payload.get("outputs_ref", {}).get("value") or payload.get("output") or "")
            toks = _tokenize(out_text)
            if self._last_output_tokens is not None:
                novelty = 1.0 - _jaccard(toks, self._last_output_tokens)
            self._last_output_tokens = toks
        except Exception:
            novelty = None

        slope = step_cost - float(self._last_cost_usd or 0.0)
        self._last_cost_usd = step_cost
        score = 0.0
        if novelty is not None:
            # High score when cost is rising and novelty is low.
            score = max(0.0, min(1.0, (_safe_float(slope, 0.0) * 10.0) + (1.0 - float(novelty))))
        if score >= float(self._cfg.loop_anomaly_threshold):
            an = KazenEvent(
                schema_version="1.2",
                ts_ms=now_ms(),
                event_id=new_id(),
                org_id=event.org_id,
                project_id=event.project_id,
                surface=event.surface,
                agent_id=event.agent_id,
                agent_role=event.agent_role,
                run_id=event.run_id,
                step_id=new_id(),
                parent_step_id=event.step_id,
                event_type="finops.loop.anomaly",
                tokens_used=tokens_used,
                cost_usd=step_cost,
                stage_budget_usd=point.budget_usd,
                remaining_budget_usd=remaining_budget,
                projected_total_cost_usd=point.projected_total_cost_usd,
                payload={
                    "score": float(score),
                    "novelty": float(novelty) if novelty is not None else None,
                    "cost_delta_usd": float(slope),
                    "reason": "rising_cost_low_novelty",
                },
            )
            derived["loop_anomaly"] = an
            self._alert.notify(kind="finops.loop.anomaly", payload={"run_id": event.run_id, "score": float(score)})

        # Circuit breaker open (soft threshold).
        dec = self._cb.evaluate(
            projected_total_cost_usd=point.projected_total_cost_usd,
            current_spend_usd=point.current_spend_usd,
            run_ctx={"run_id": event.run_id, "agent_id": event.agent_id, "project_id": event.project_id},
            opaque_state=None,
        )
        if dec.opened:
            avoided_cost = max(0.0, float(point.projected_total_cost_usd or 0.0) - float(point.current_spend_usd or 0.0))
            cb_ev = KazenEvent(
                schema_version="1.2",
                ts_ms=now_ms(),
                event_id=new_id(),
                org_id=event.org_id,
                project_id=event.project_id,
                surface=event.surface,
                agent_id=event.agent_id,
                agent_role=event.agent_role,
                run_id=event.run_id,
                step_id=new_id(),
                parent_step_id=event.step_id,
                event_type="finops.circuit_breaker.opened",
                tokens_used=tokens_used,
                cost_usd=step_cost,
                stage_budget_usd=point.budget_usd,
                remaining_budget_usd=remaining_budget,
                projected_total_cost_usd=point.projected_total_cost_usd,
                avoided_cost_usd=avoided_cost,
                payload={
                    "reason": dec.reason,
                    "resume_token": dec.resume_token,
                    "projected_total_cost_usd": point.projected_total_cost_usd,
                    "soft_pause_threshold_usd": point.soft_pause_threshold_usd,
                    "budget_usd": point.budget_usd,
                    "avoided_cost_usd": avoided_cost,
                },
            )
            derived["circuit_breaker_opened"] = cb_ev
            self._alert.notify(kind="finops.circuit_breaker.opened", payload={"run_id": event.run_id})
            derived["_circuit_breaker_exc"] = KazenCircuitBreaker(
                f"circuit breaker opened: projected=${point.projected_total_cost_usd:.4f} threshold=${(point.soft_pause_threshold_usd or 0.0):.4f}",
                resume_token=dec.resume_token,
                context={"run_id": event.run_id},
            )

        return derived

    @staticmethod
    def raise_if_blocked(derived: Dict[str, Any]) -> None:
        """Raise deferred circuit-breaker after derived events are emitted to sinks."""
        exc = derived.pop("_circuit_breaker_exc", None)
        if exc is not None:
            raise exc
