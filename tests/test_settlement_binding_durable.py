"""Immutable settlement binding + durable gateway store."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from kazenai.integrations.litellm_control import (
    AttemptIdentity,
    AttemptReservation,
    DurableGatewayStore,
    GatewayFinOpsClient,
    estimate_cost_usd,
    estimate_reserve_usd,
)
from kazenai.lifecycle_config import map_accounting_owner_to_enforcement
from kazenai.spine.guard import ReservationHandle, _capture_settlement_binding


def test_accounting_owner_maps_to_enforcement_gateway():
    assert map_accounting_owner_to_enforcement("server") == "gateway"
    assert map_accounting_owner_to_enforcement("gateway") == "gateway"
    assert map_accounting_owner_to_enforcement("sdk") == "sdk"


def test_reservation_handle_settlement_binding_survives_env_mutation(monkeypatch, tmp_path):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://admit.example:8090")
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "admit-key")
    monkeypatch.setenv("KAZENAI_FINOPS_CONTROL_PROFILE", "control")
    bind = _capture_settlement_binding(base="http://admit.example:8090", org_id="org-a", workspace_id="ws-a")
    handle = ReservationHandle(
        reserved_cost_usd=0.01,
        reservation_id="rsv_1",
        call_id="call_1",
        attempt=1,
        lifecycle=True,
        **bind,
    )
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://mutated.example:9999")
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "mutated-key")
    assert handle.settlement_base_url() == "http://admit.example:8090"
    assert handle.settlement_api_key() == "admit-key"
    assert handle.settlement_org_id() == "org-a"
    assert handle.failure_policy in {"fail_closed", "fail_open"}


def test_estimate_cost_unavailable_returns_none():
    with patch("kazenai.cost_engine.TokenCostEngine") as eng:
        eng.return_value.cost_usd.side_effect = RuntimeError("no price table")
        assert estimate_cost_usd(model="x", prompt_tokens=10, completion_tokens=5) is None
    # Reserve helper always returns a positive hold bound even without a price.
    with patch("kazenai.cost_engine.TokenCostEngine") as eng:
        eng.return_value.cost_usd.side_effect = RuntimeError("no price table")
        assert estimate_reserve_usd(model="x", prompt_tokens=10, completion_tokens=5) > 0


def test_durable_gateway_store_roundtrip(tmp_path):
    store = DurableGatewayStore(root=str(tmp_path / "gw"))
    identity = AttemptIdentity(
        operation_id="op1",
        call_id="c1",
        attempt_id="att1",
        attempt=1,
        idempotency_key="idem1",
    )
    rsv = AttemptReservation(
        identity=identity,
        reservation_id="rsv_x",
        decision_id="dec_x",
        estimated_usd_micros=1000,
        status="started",
        meta={"model": "gpt-4o-mini"},
    )
    store.put_open(rsv)
    restored = store.get_open("att1")
    assert restored is not None
    assert restored.reservation_id == "rsv_x"
    assert restored.status == "started"

    body = {
        "lifecycle_version": "reservation.lifecycle.v1",
        "reservation_id": "rsv_x",
        "event": "outcome_unknown",
        "idempotency_key": "idem1:outcome_unknown_signal",
        "org_id": "org-a",
    }
    store.enqueue_pending(body=body, error="finops_down")
    pending = store.list_pending()
    assert len(pending) == 1
    assert pending[0]["body"]["reservation_id"] == "rsv_x"
    store.ack_pending(pending[0]["_path"])
    assert store.list_pending() == []
    store.drop_open("att1")
    assert store.get_open("att1") is None


def test_settle_success_none_usd_goes_pending(tmp_path, monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://unused")
    client = GatewayFinOpsClient(
        base_url="http://unused",
        api_key="k",
        org_id="org",
        durable_store=DurableGatewayStore(root=str(tmp_path / "gw2")),
    )
    identity = AttemptIdentity(
        operation_id="op",
        call_id="c",
        attempt_id="a",
        attempt=1,
        idempotency_key="idem",
    )
    rsv = AttemptReservation(identity=identity, reservation_id="rsv", estimated_usd_micros=100)
    with patch.object(client, "settle_cancel_or_pending", return_value={"result": "pending"}) as mocked:
        out = client.settle_success(rsv, actual_usd=None)
    assert out["result"] == "pending"
    mocked.assert_called_once()
