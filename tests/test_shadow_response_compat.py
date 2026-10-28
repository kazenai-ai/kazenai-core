"""Shadow mode (Agent FinOps 1.2.0) compatibility for the SDK.

Shadow mode is server-side: an allowed reservation may carry a receipt
``enforcement`` block and a top-level ``shadow`` summary saying a shadow-mode
limit would have denied the call. The SDK must treat it as an ordinary allow.
No SDK code changes are needed; this test pins that behaviour.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx

from kazenai.spine.guard import ReservationHandle, reserve_budget_async

_SHADOW_ALLOW = {
    "lifecycle_version": "reservation.lifecycle.v1",
    "allowed": True,
    "reservation_id": "rsv_shadow",
    "settlement_capability": "kzcap1.test." + "A" * 43,
    "state": "reserved",
    "reserved_usd_micros": 250_000,
    "admission_mode": "postgresql_atomic",
    "duplicate": False,
    "call_id": "call-shadow",
    "attempt": 1,
    "decision_id": "dec_shadow",
    "shadow": {
        "would_decision": "deny",
        "enforcement_mode": "shadow",
        "policy_id": "budget.business_subject.cust:acme.daily",
        "policy_version": "2",
    },
    "decision_receipt": {
        "receipt_version": "economic-decision-receipt.v1",
        "decision_id": "dec_shadow",
        "decision": "allow",
        "reason_code": "allowed_with_reservation",
        "authority": {"org_id": "o", "workspace_id": "w"},
        "economics": {"attribution_state": "attributed", "business_subject_ref": "cust:acme"},
        "execution": {"call_id": "call-shadow", "attempt": 1},
        "evaluated_policies": [
            {
                "policy_id": "budget.business_subject.cust:acme.daily",
                "policy_version": "2",
                "scope": "business_subject",
                "result": "deny",
                "mode": "shadow",
            }
        ],
        "determining_constraint": {"scope": "business_subject", "outcome": "allow"},
        "requested_estimate_usd_micros": 250_000,
        "reserved_usd_micros": 250_000,
        "remaining_capacity_usd_micros": None,
        "currency": "USD",
        "precision": "micro",
        "enforcement": {
            "mode": "shadow",
            "shadow": {
                "would_decision": "deny",
                "reason_code": "insufficient_capacity",
                "virtual_remaining_capacity_usd_micros": 0,
            },
        },
    },
}


def test_shadow_would_deny_response_is_an_ordinary_allow(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")
    monkeypatch.setenv("KAZENAI_FINOPS_CONTROL_PROFILE", "control")
    monkeypatch.setenv("KAZENAI_ENV", "development")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_SHADOW_ALLOW)

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    async def _run():
        with patch(
            "kazenai.spine.guard._ensure_server_lifecycle_compatible_async",
            return_value=None,
        ):
            with patch("httpx.AsyncClient", side_effect=client_factory):
                return await reserve_budget_async(
                    org_id="o",
                    workspace_id="w",
                    run_id="r",
                    model="gpt-4o-mini",
                    estimated_cost_usd=0.25,
                    call_id="call-shadow",
                    attempt=1,
                    idempotency_key="reserve:call-shadow:1:fixed",
                    business_subject_ref="cust:acme",
                )

    handle = asyncio.run(_run())
    assert isinstance(handle, ReservationHandle)
    assert handle.reservation_id == "rsv_shadow"
    assert handle.call_id == "call-shadow"
