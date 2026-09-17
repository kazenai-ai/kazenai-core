"""FinOps internal helper coverage."""

from __future__ import annotations

from kazenai.finops import FinOpsController, FinOpsConfig, _jaccard, _safe_float, _tokenize


def test_safe_float_invalid():
    assert _safe_float("nope", default=1.5) == 1.5


def test_tokenize_and_jaccard():
    a = _tokenize("Hello World")
    b = _tokenize("hello world!")
    assert _jaccard(a, b) == 1.0


def test_finops_handles_invalid_depth():
    from kazenai.schema import KazenEvent, new_id, now_ms

    c = FinOpsController(cfg=FinOpsConfig(budget_usd=5.0))
    ev = KazenEvent(
        schema_version="1.0",
        ts_ms=now_ms(),
        event_id=new_id(),
        org_id="org",
        project_id="proj",
        surface="kazenai-core",
        agent_id="agent",
        agent_role="agent",
        run_id="r1",
        step_id=new_id(),
        event_type="model.call",
        cost_usd=0.01,
        payload={},
    )
    ev.payload = {"depth": "not-a-number", "model": "gpt-4o-mini"}
    derived = c.handle_llm_call(ev)
    assert "trajectory" in derived
