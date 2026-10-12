"""FINAL_1_b_final P2 — canonical event round-trips via core re-export."""

from __future__ import annotations

from kazenai.schema import KazenEvent, canonicalize_event_dict, new_id, now_ms, validate_payload


def test_core_reexports_helpers() -> None:
    assert callable(now_ms)
    assert callable(new_id)
    assert callable(canonicalize_event_dict)
    assert callable(validate_payload)
    assert now_ms() > 0
    assert len(new_id()) >= 8


def test_tokens_17_round_trip_through_core() -> None:
    raw = {
        "schema_version": "1.2",
        "ts_ms": 1_700_000_000_000,
        "event_id": "evt_p2_core_roundtrip",
        "org_id": "org_p2",
        "workspace_id": "ws_p2",
        "project_id": "ws_p2",
        "surface": "kazenai-sdk",
        "agent_id": "agent_p2",
        "agent_role": "worker",
        "run_id": "run_p2",
        "step_id": "step_p2",
        "event_type": "model.call",
        "tokens": 17,
        "cost_usd": 0.0,
        "payload": {"model": "fake"},
    }
    dumped = canonicalize_event_dict(raw)
    again = KazenEvent.model_validate(dumped)
    assert again.tokens_used == 17
    assert again.tokens == 17
    assert again.root_run_id == "run_p2"
