from __future__ import annotations

import pytest

from kazenai.circuit_breaker import KazenCircuitBreaker
from kazenai.finops import FinOpsConfig, FinOpsController
from kazenai.schema import KazenEvent, new_id, now_ms


def _event(*, cost_usd: float, run_id: str = "r1") -> KazenEvent:
    return KazenEvent(
        schema_version="1.0",
        ts_ms=now_ms(),
        event_id=new_id(),
        org_id="org",
        project_id="proj",
        surface="kazenai-core",
        agent_id="agent",
        agent_role="agent",
        run_id=run_id,
        step_id=new_id(),
        parent_step_id=None,
        event_type="model.call",
        tokens_used=100,
        cost_usd=cost_usd,
        payload={"model": "gpt-4o-mini", "usage": {"prompt_tokens": 50, "completion_tokens": 50}},
    )


def test_finops_emits_trajectory() -> None:
    c = FinOpsController(cfg=FinOpsConfig(budget_usd=10.0))
    derived = c.handle_llm_call(_event(cost_usd=0.01))
    assert "trajectory" in derived
    assert derived["trajectory"].event_type == "finops.trajectory"
    assert derived["trajectory"].schema_version == "1.2"
    assert derived["trajectory"].projected_total_cost_usd is not None
    assert derived["trajectory"].cost_quality_score is not None


def test_finops_circuit_breaker_opens() -> None:
    c = FinOpsController(cfg=FinOpsConfig(budget_usd=0.01, soft_pause_pct=0.9))
    derived = c.handle_llm_call(_event(cost_usd=0.01))
    assert "circuit_breaker_opened" in derived
    with pytest.raises(KazenCircuitBreaker):
        FinOpsController.raise_if_blocked(derived)


def test_finops_computes_cost_from_usage() -> None:
    c = FinOpsController(cfg=FinOpsConfig(budget_usd=100.0))
    ev = _event(cost_usd=None)
    ev.payload = {
        "model": "gpt-4o-mini",
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
    }
    derived = c.handle_llm_call(ev)
    assert derived["trajectory"].cost_usd is not None
    assert derived["trajectory"].cost_usd > 0


def test_finops_loop_anomaly_on_repeated_output() -> None:
    c = FinOpsController(cfg=FinOpsConfig(budget_usd=50.0, loop_anomaly_threshold=0.01))
    ev1 = _event(cost_usd=0.01)
    ev1.payload = {"outputs_ref": {"value": "same answer every time"}, "model": "gpt-4o-mini"}
    c.handle_llm_call(ev1)
    ev2 = _event(cost_usd=0.50)
    ev2.payload = {"outputs_ref": {"value": "same answer every time"}, "model": "gpt-4o-mini"}
    derived = c.handle_llm_call(ev2)
    assert "loop_anomaly" in derived
    assert derived["loop_anomaly"].event_type == "finops.loop.anomaly"


def test_finops_no_budget_skips_circuit_breaker() -> None:
    c = FinOpsController(cfg=FinOpsConfig(budget_usd=None))
    derived = c.handle_llm_call(_event(cost_usd=100.0))
    assert "circuit_breaker_opened" not in derived


def test_finops_token_count_from_payload_fields() -> None:
    c = FinOpsController(cfg=FinOpsConfig(budget_usd=10.0))
    ev = _event(cost_usd=None)
    ev.tokens_used = None
    ev.payload = {"input_tokens": 10, "output_tokens": 5, "model": "gpt-4o-mini"}
    derived = c.handle_llm_call(ev)
    assert derived["trajectory"].tokens_used == 15


def test_finops_trajectory_remaining_budget() -> None:
    c = FinOpsController(cfg=FinOpsConfig(budget_usd=10.0))
    derived = c.handle_llm_call(_event(cost_usd=1.0))
    traj = derived["trajectory"]
    assert traj.remaining_budget_usd is not None
    assert traj.remaining_budget_usd <= 10.0
