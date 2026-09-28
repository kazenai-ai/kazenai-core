"""FINAL_1_b P3-2 — single enforcement owner; no double reservation (hermetic)."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError
from io import BytesIO

import pytest

from kazenai.enforcement_owner import (
    EnforcementOwnershipError,
    bind_decision_refs,
    claim_enforcement_owner,
    get_enforcement_owner,
)
from kazenai.spine.guard import ReservationHandle, reserve_budget


class _FakeResp:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self.status = status
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def test_nested_same_owner_single_reservation(monkeypatch):
    monkeypatch.delenv("KAZENAI_ENFORCEMENT_OWNER", raising=False)
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops.test")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    monkeypatch.delenv("KAZENAI_FINOPS_CONTROL_PROFILE", raising=False)
    monkeypatch.delenv("KAZENAI_CONTROL_PROFILE", raising=False)
    monkeypatch.delenv("KAZENAI_PROFILE", raising=False)
    monkeypatch.delenv("KAZENAI_FINOPS_STRICT_BUDGET", raising=False)

    calls: list[dict[str, Any]] = []

    def fake_urlopen(req: Any, timeout: float = 0):  # noqa: ARG001
        body = json.loads(req.data.decode("utf-8"))
        calls.append(body)
        return _FakeResp({"estimated_cost_usd": 0.02, "reservation_id": "res-1", "decision_id": "dec-1"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with claim_enforcement_owner("sdk"):
        first = reserve_budget(org_id="o", workspace_id="w", run_id="r1")
        assert first == pytest.approx(0.02)
        owner = get_enforcement_owner()
        assert owner is not None
        assert owner.reservation_id == "res-1"
        # Nested same-owner scope reuses reservation — no second HTTP reserve.
        with claim_enforcement_owner("sdk"):
            second = reserve_budget(org_id="o", workspace_id="w", run_id="r1")
            assert isinstance(second, ReservationHandle) or second == 0.0
            assert len(calls) == 1


def test_ambiguous_owner_env_raises(monkeypatch):
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_OWNER", "gateway")
    with pytest.raises(EnforcementOwnershipError):
        with claim_enforcement_owner("sdk"):
            pass


def test_gateway_without_reservation_blocks_sdk_reserve(monkeypatch):
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_OWNER", "gateway")
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops.test")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_closed")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_closed")

    def boom(*args: Any, **kwargs: Any):  # noqa: ARG001
        raise AssertionError("must not call FinOps when ownership is ambiguous")

    monkeypatch.setattr("urllib.request.urlopen", boom)

    with pytest.raises(EnforcementOwnershipError):
        # claim gateway first to set context, then SDK should not independently reserve
        with claim_enforcement_owner("gateway"):
            reserve_budget(org_id="o", workspace_id="w", run_id="r")


def test_propagated_reservation_skips_second_charge(monkeypatch):
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_OWNER", "gateway")
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops.test")

    def boom(*args: Any, **kwargs: Any):  # noqa: ARG001
        raise AssertionError("must not re-reserve")

    monkeypatch.setattr("urllib.request.urlopen", boom)

    with claim_enforcement_owner("gateway", reservation_id="res-gateway", call_id="c1", attempt=1):
        bind_decision_refs(decision_id="dec-g")
        handle = reserve_budget(org_id="o", workspace_id="w", run_id="r")
        assert isinstance(handle, ReservationHandle)
        assert handle.reservation_id == "res-gateway"
