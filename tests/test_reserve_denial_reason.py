"""A 402 from Agent FinOps names why the call was refused (kazenai 1.2.1).

A paused organization gets the closed reserve-response shape with no ``reason``
or ``error_code`` (Agent FinOps ``budget/admit_pg.py``, the ``OrgPausedError``
branch of ``admit_reservation``). The message then comes from the receipt's
``determining_constraint.detail``, not "budget exhausted". FinOps is reached
through the SDK's own transport on an ``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from typing import Any

import httpx
import pytest

from kazenai import monitor
from kazenai.enforcement import BudgetExceeded
from kazenai.spine import guard, org_policy, transport

openai = pytest.importorskip("openai")

_CHAT = {
    "id": "c1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}

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
    """Agent FinOps on a mock transport: capabilities, then ``reserve_body`` with 402."""

    def __init__(self, reserve_body: Any, reserve_status: int = 402) -> None:
        self.reserve_body = reserve_body
        self.reserve_status = reserve_status
        self.paths: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "effective_lifecycle": {"profile": "control", "lifecycle_version": "reservation.lifecycle.v1"},
                    "capabilities": {"reservation_settlement": True, "legacy_check_reconcile": False},
                },
            )
        if request.url.path.endswith(("/reserve", "/check")):
            return httpx.Response(self.reserve_status, json=self.reserve_body)
        return httpx.Response(200, json={"ok": True})


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    for key in ("KAZENAI_FINOPS_INGEST_URL", "KAZENAI_INGEST_URL", "KAZENAI_DENY_STREAMING"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops.test")
    monkeypatch.setenv("KAZENAI_SDK_SETTLE_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "test")
    org_policy.reset_for_tests()
    guard._LIFECYCLE_CAP_CACHE.clear()
    transport.reset_pools()
    yield
    org_policy.reset_for_tests()
    guard._LIFECYCLE_CAP_CACHE.clear()
    transport.reset_pools()


def _route_finops(monkeypatch, finops: FinOps) -> None:
    """Send the SDK's sync and async FinOps requests to ``finops``."""
    monkeypatch.setattr(transport, "_sync_client", httpx.Client(transport=httpx.MockTransport(finops)))

    @contextlib.asynccontextmanager
    async def async_session(timeout: Any):
        async with httpx.AsyncClient(transport=httpx.MockTransport(finops)) as client:
            yield transport._Session(client, timeout)

    monkeypatch.setattr(transport, "async_session", async_session)


def _legacy(monkeypatch) -> None:
    for key in ("KAZENAI_FINOPS_CONTROL_PROFILE", "KAZENAI_CONTROL_PROFILE", "KAZENAI_PROFILE"):
        monkeypatch.setenv(key, "legacy")
    monkeypatch.setenv("KAZENAI_ENV", "dev")


def _client(finops: FinOps, hits: list, *, is_async: bool = False):
    def provider(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.path)
        return httpx.Response(200, json=_CHAT)

    if is_async:
        raw = openai.AsyncOpenAI(api_key="t", base_url="https://openai.test/v1", max_retries=0,
                                 http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)))
    else:
        raw = openai.OpenAI(api_key="t", base_url="https://openai.test/v1", max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(provider)))
    return monitor(raw, org_id="org-paused", workspace_id="ws", agent_id=f"deny-{uuid.uuid4().hex[:6]}",
                   max_budget_usd=5.0)


def _ask(client) -> Any:
    return client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])


def _refusal(monkeypatch, body: Any, *, is_async: bool) -> tuple[str, FinOps, list]:
    finops, hits = FinOps(body), []
    _route_finops(monkeypatch, finops)
    client = _client(finops, hits, is_async=is_async)
    with pytest.raises(BudgetExceeded) as caught:
        if is_async:
            asyncio.run(_ask(client))
        else:
            _ask(client)
    return str(caught.value), finops, hits


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_a_paused_org_is_told_it_is_paused(monkeypatch, is_async):
    message, finops, hits = _refusal(monkeypatch, PAUSE_BODY, is_async=is_async)
    assert message == f"Pre-call reservation denied: {PAUSED}"
    assert "budget exhausted" not in message
    assert finops.paths.count("/v1/budget/reserve") == 1
    assert hits == []


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_a_plan_cap_keeps_its_plan_sentence(monkeypatch, is_async):
    message, _, hits = _refusal(monkeypatch, PLAN_CAP_BODY, is_async=is_async)
    assert message == f"Pre-call reservation denied: {PLAN_SENTENCE}"
    assert hits == []


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_a_customer_limit_message_is_unchanged(monkeypatch, is_async):
    message, _, hits = _refusal(monkeypatch, CUSTOMER_LIMIT_BODY, is_async=is_async)
    assert message == f"Pre-call reservation denied: {CUSTOMER_LIMIT}"
    assert hits == []


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_legacy_check_reads_the_receipt_too(monkeypatch, is_async):
    _legacy(monkeypatch)
    finops = FinOps(PAUSE_BODY)
    _route_finops(monkeypatch, finops)
    with pytest.raises(BudgetExceeded) as caught:
        if is_async:
            asyncio.run(guard.reserve_budget_async(org_id="o", workspace_id="w", run_id="r"))
        else:
            guard.reserve_budget(org_id="o", workspace_id="w", run_id="r")
    assert str(caught.value) == f"Pre-call reservation denied: {PAUSED}"
    assert finops.paths == ["/v1/budget/check"]


@pytest.mark.parametrize(
    ("body", "fields", "expected"),
    [
        ({"reason": "daily budget exhausted"}, ("reason", "detail"), "daily budget exhausted"),
        ({"detail": "Payment Required"}, ("reason", "detail"), "Payment Required"),
        ({"error_code": "tier_breaker"}, ("reason", "error_code"), "tier_breaker"),
        ({"detail": {"reason": "over limit"}}, ("reason", "error_code"), "over limit"),
        (PAUSE_BODY, ("reason", "error_code"), PAUSED),
        (PAUSE_BODY["detail"], ("reason", "error_code"), PAUSED),
        ({"allowed": False}, ("reason", "error_code"), "budget exhausted"),
        ({"decision_receipt": {"determining_constraint": {"scope": "org"}}}, ("reason",), "budget exhausted"),
        ({"decision_receipt": "not a receipt"}, ("reason",), "budget exhausted"),
        (["not", "a", "dict"], ("reason",), "budget exhausted"),
    ],
)
def test_denial_reason_order(body, fields, expected):
    assert guard._denial_reason(body, *fields) == expected
