from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kazenai.schema import KazenEvent, new_id, now_ms


def test_kazenevent_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        KazenEvent(
            schema_version="1.0",
            ts_ms=now_ms(),
            event_id=new_id(),
            org_id="org",
            project_id="proj",
            surface="kazenai-core",
            agent_id="agent",
            agent_role="sdk",
            run_id="run",
            step_id="step",
            parent_step_id=None,
            event_type="model.call",
            payload={},
            extra_field_not_allowed=True,
        )


def test_kazenevent_v12_score5_fields() -> None:
    ev = KazenEvent(
        ts_ms=now_ms(),
        event_id=new_id(),
        org_id="org",
        workspace_id="workspace",
        project_id="proj",
        surface="kazenai-core",
        agent_id="agent",
        agent_role="sdk",
        run_id="run",
        step_id="step",
        event_type="replay.completed",
        cost_usd=0.12,
        latency_ms=42,
        stage_budget_usd=1.0,
        remaining_budget_usd=0.88,
        projected_total_cost_usd=0.3,
        avoided_cost_usd=0.7,
        expected_success_probability=0.9,
        cost_quality_score=3.0,
        replay_group_id="rg1",
        frozen_trace_ref="runs/run/frozen_trace.jsonl",
        drift_baseline_id="baseline-1",
        sandbox_backend="aws_cloud",
        artifact_refs={"run_evidence": "run_evidence.json"},
        payload={"ok": True},
    )

    assert ev.schema_version == "1.2"
    assert ev.projected_total_cost_usd == 0.3
    assert ev.frozen_trace_ref == "runs/run/frozen_trace.jsonl"


def test_kazenevent_schema_snapshot_matches_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    contract_path = repo_root / "contracts" / "kazenevent.schema.v1.json"
    expected = json.loads(contract_path.read_text(encoding="utf-8") or "{}")
    generated = KazenEvent.model_json_schema()
    assert generated["title"] == expected["title"]
    for field in (
        "schema_version",
        "event_id",
        "org_id",
        "step_id",
        "event_type",
        "projected_total_cost_usd",
        "cost_quality_score",
    ):
        assert field in generated["properties"]
