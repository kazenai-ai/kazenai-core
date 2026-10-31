"""The LiteLLM gateway path names why Agent FinOps refused an attempt (kazenai 1.2.1).

``GatewayFinOpsClient.reserve_attempt`` raises ``GatewayAdmissionDenied`` on a 402.
A paused organization gets the closed reserve shape with no ``reason`` or
``error_code`` (Agent FinOps ``budget/admit_pg.py``, the ``OrgPausedError`` branch),
so the message comes from the receipt's ``determining_constraint.detail``, the same
rule as the SDK path (``kazenai.spine.guard._denial_reason``). FinOps is reached
through the SDK's own transport on an ``httpx.MockTransport``.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from kazenai.integrations.litellm_control import (
    DurableGatewayStore,
    GatewayAdmissionDenied,
    GatewayFinOpsClient,
    new_attempt_identity,
)
from kazenai.spine import transport

PAUSED = "organization is paused for new admissions"
PLAN_SENTENCE = "Free plan limit reached: 1,000 calls this month. Contact founder@kazenai.com to upgrade."
CUSTOMER_LIMIT = "insufficient remaining capacity on customer"


def _receipt(constraint: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": "dec_test",
        "decision": "deny",
        "reason_code": "policy_denied",
        "determining_constraint": constraint,
    }


# The body FinOps sends for a paused org: ``HTTPException(402, detail=deny_body)``.
PAUSE_BODY = {
    "detail": {
        "lifecycle_version": "reservation.lifecycle.v1",
        "allowed": False,
        "decision_id": "dec_test",
        "decision_receipt": _receipt(
            {
                "scope": "org",
                "outcome": "deny",
                "policy_id": "org_pause:pause_test",
                "policy_version": "1",
                "detail": PAUSED,
            }
        ),
        "http_status": 402,
    }
}

# A plan cap: ``AdmitError("plan_cap_reached", <plan sentence>, http_status=402, receipt=...)``.
PLAN_CAP_BODY = {
    "detail": {
        "lifecycle_version": "reservation.lifecycle.v1",
        "allowed": False,
        "error_code": "plan_cap_reached",
        "reason": PLAN_SENTENCE,
        "decision_receipt": _receipt(
            {"scope": "org", "outcome": "deny", "policy_id": "plan:free:calls", "detail": PLAN_SENTENCE}
        ),
        "decision_id": "dec_test",
    }
}

# A customer limit: ``AdmitError("invalid_amount", "insufficient remaining capacity on customer", ...)``.
CUSTOMER_LIMIT_BODY = {
    "detail": {
        "lifecycle_version": "reservation.lifecycle.v1",
        "allowed": False,
        "error_code": "invalid_amount",
        "reason": CUSTOMER_LIMIT,
        "decision_receipt": _receipt({"scope": "customer", "outcome": "deny", "detail": CUSTOMER_LIMIT}),
        "decision_id": "dec_test",
    }
}


class FinOps:
    """Agent FinOps on a mock transport: every request gets ``status`` and ``body``."""

    def __init__(self, body: Any, status: int = 402) -> None:
        self.body = body
        self.status = status
        self.paths: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        return httpx.Response(self.status, json=self.body)


@pytest.fixture(autouse=True)
def _clean_transport():
    transport.reset_pools()
    yield
    transport.reset_pools()


def _deny(monkeypatch, tmp_path, body: Any, status: int = 402) -> tuple[GatewayAdmissionDenied, FinOps, GatewayFinOpsClient]:
    finops = FinOps(body, status)
    monkeypatch.setattr(transport, "_sync_client", httpx.Client(transport=httpx.MockTransport(finops)))
    client = GatewayFinOpsClient(
        base_url="http://finops.test",
        api_key="test",
        org_id="org-paused",
        durable_store=DurableGatewayStore(str(tmp_path / "gateway")),
    )
    with pytest.raises(GatewayAdmissionDenied) as caught:
        client.reserve_attempt(identity=new_attempt_identity(), model="gpt-4o-mini", estimated_usd=0.01)
    return caught.value, finops, client


def test_a_paused_org_is_told_it_is_paused(monkeypatch, tmp_path):
    denied, finops, client = _deny(monkeypatch, tmp_path, PAUSE_BODY)
    assert str(denied) == f"gateway admission denied: {PAUSED}"
    assert "budget exhausted" not in str(denied)
    assert denied.code == "budget_denied"
    assert denied.http_status == 402
    assert finops.paths == ["/v1/budget/reserve"]
    assert client._open == {}


def test_a_plan_cap_keeps_its_plan_sentence(monkeypatch, tmp_path):
    denied, _, _ = _deny(monkeypatch, tmp_path, PLAN_CAP_BODY)
    assert PLAN_CAP_BODY["detail"]["error_code"] == "plan_cap_reached"
    assert str(denied) == f"gateway admission denied: {PLAN_SENTENCE}"
    assert (denied.code, denied.http_status) == ("budget_denied", 402)


def test_a_customer_limit_message_is_unchanged(monkeypatch, tmp_path):
    denied, _, _ = _deny(monkeypatch, tmp_path, CUSTOMER_LIMIT_BODY)
    assert str(denied) == f"gateway admission denied: {CUSTOMER_LIMIT}"


def test_a_flat_body_keeps_reason_over_error_code(monkeypatch, tmp_path):
    body = {"allowed": False, "reason": "capacity", "error_code": "budget_exhausted"}
    denied, _, _ = _deny(monkeypatch, tmp_path, body)
    assert str(denied) == "gateway admission denied: capacity"


def test_an_error_code_alone_is_the_reason(monkeypatch, tmp_path):
    denied, _, _ = _deny(monkeypatch, tmp_path, {"detail": {"allowed": False, "error_code": "tier_breaker"}})
    assert str(denied) == "gateway admission denied: tier_breaker"


@pytest.mark.parametrize(
    "body",
    [
        {"detail": {"allowed": False}},
        {"detail": {"allowed": False, "decision_receipt": {"determining_constraint": {"scope": "org"}}}},
        {"detail": {"allowed": False, "decision_receipt": "not a receipt"}},
    ],
    ids=["no-receipt", "no-constraint-detail", "receipt-not-a-dict"],
)
def test_nothing_to_say_falls_back_to_budget_exhausted(monkeypatch, tmp_path, body):
    denied, _, _ = _deny(monkeypatch, tmp_path, body)
    assert str(denied) == "gateway admission denied: budget exhausted"


def test_a_200_with_allowed_false_reads_the_receipt_too(monkeypatch, tmp_path):
    denied, _, _ = _deny(monkeypatch, tmp_path, PAUSE_BODY["detail"], status=200)
    assert str(denied) == f"gateway admission denied: {PAUSED}"
    assert (denied.code, denied.http_status) == ("budget_denied", 402)
