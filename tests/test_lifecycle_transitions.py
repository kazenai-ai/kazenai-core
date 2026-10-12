"""Shared lifecycle transitions + async admission."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import httpx

from kazenai.capture_policy import CaptureMode
from kazenai.context import RunContext
from kazenai.enforcement import Enforcement
from kazenai.lifecycle_transitions import (
    AttemptIdentities,
    build_settle_payload,
    classify_settlement_delivery,
    create_attempt_identities,
    settlement_evidence_matches,
    settle_idempotency_key,
)
from kazenai.spine import guard as guard_mod
from kazenai.spine.guard import (
    ReservationHandle,
    reconcile_budget,
    reserve_budget_async,
    signal_reservation_event,
)
from kazenai.monitor import SUPPORTED_OPENAI_STREAM_METHOD
from kazenai.streaming.lifecycle import StreamAttempt, StreamOutcome, StreamUsage


def test_create_attempt_identities_stable_ordering():
    first = create_attempt_identities(
        operation_id="op-1",
        call_id="call-1",
        attempt=2,
        attempt_id="call-1:attempt:2",
        idempotency_key="reserve:call-1:2:k",
    )
    assert isinstance(first, AttemptIdentities)
    assert first.call_id == "call-1"
    assert first.attempt == 2
    assert first.idempotency_key == "reserve:call-1:2:k"
    # Regenerating without a new key would be a bug — reuse must be explicit.
    again = create_attempt_identities(
        call_id=first.call_id,
        attempt=first.attempt,
        idempotency_key=first.idempotency_key,
    )
    assert again.idempotency_key == first.idempotency_key
    assert again.call_id == first.call_id


def test_build_settle_payload_and_keys():
    payload = build_settle_payload(
        reservation_id="res-1",
        call_id="call-1",
        attempt=1,
        org_id="org",
        event="usage_known",
        actual_usd_micros=820_000,
    )
    assert payload["idempotency_key"] == settle_idempotency_key("res-1", "usage_known")
    assert payload["actual_usd_micros"] == 820_000
    assert payload["lifecycle_version"] == "reservation.lifecycle.v1"


def test_settlement_evidence_requires_identity_and_amount():
    reservation = {
        "reservation_id": "res-1",
        "call_id": "call-1",
        "attempt": 1,
        "state": "settled",
        "settle_result": "applied",
        "actual_usd_micros": 820_000,
        "last_settle_json": {"actual_usd_micros": 820_000},
    }
    assert settlement_evidence_matches(
        reservation=reservation,
        call_id="call-1",
        attempt=1,
        reservation_id="res-1",
        actual_usd_micros=820_000,
    )
    # Message-level "settled" is irrelevant — wrong amount must not match.
    assert not settlement_evidence_matches(
        reservation=reservation,
        call_id="call-1",
        attempt=1,
        reservation_id="res-1",
        actual_usd_micros=900_000,
    )
    assert not settlement_evidence_matches(
        reservation={**reservation, "state": "in_flight"},
        call_id="call-1",
        attempt=1,
        reservation_id="res-1",
        actual_usd_micros=820_000,
    )


def test_classify_separates_local_finalize_from_financial_ack():
    assert (
        classify_settlement_delivery(
            local_finalized=True, financial_ack=True, authoritative_usage=True
        )
        == "settled"
    )
    assert (
        classify_settlement_delivery(
            local_finalized=True, financial_ack=False, authoritative_usage=True
        )
        == "exact_usage_pending_settlement"
    )
    assert (
        classify_settlement_delivery(
            local_finalized=True, financial_ack=False, authoritative_usage=False
        )
        == "pending_reconciliation"
    )


def test_stream_finalize_marks_pending_when_reconcile_returns_none():
    attempt = StreamAttempt(
        surface=SUPPORTED_OPENAI_STREAM_METHOD,
        step_ctx=RunContext.new(org_id="o", project_id="p", workspace_id="w", agent_id="a"),
        enforcement=Enforcement(max_cost_usd=5.0),
        held_projection=0.01,
        reserved_cost_usd=ReservationHandle(
            reserved_cost_usd=1.0,
            reservation_id="res-x",
            call_id="call-x",
            attempt=1,
            lifecycle=True,
        ),
        model="gpt-test",
        method=SUPPORTED_OPENAI_STREAM_METHOD,
        usd_per_1k_tokens=0.0,
        capture_mode=CaptureMode.METADATA,
        agent_role="agent",
        reconcile_fn=lambda **_kwargs: None,
        pending_fn=lambda **_kwargs: True,
        postcall_cost_fn=lambda **_kwargs: 0.82,
    )
    attempt.finalize(
        StreamOutcome.COMPLETE,
        usage=StreamUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            authoritative=True,
            raw={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        ),
    )
    assert attempt.finalized is True
    assert attempt.financial_pending is True
    assert attempt.cost_confidence == "exact_usage_pending_settlement"


def _ack_resp(body: dict | None = None):
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: mock_resp
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = json.dumps(body or {"result": "applied"}).encode()
    return mock_resp


def test_reconcile_budget_returns_bool_synchronously(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")
    monkeypatch.setenv("KAZENAI_FINOPS_CONTROL_PROFILE", "control")
    handle = ReservationHandle(
        reserved_cost_usd=1.0,
        reservation_id="res-bool",
        call_id="call-bool",
        attempt=1,
        lifecycle=True,
        finops_base_url="http://finops:8090",
    )

    with patch("kazenai.spine.guard.urllib.request.urlopen", return_value=_ack_resp()):
        ok = reconcile_budget(
            org_id="o",
            workspace_id="w",
            run_id="r",
            reserved_cost_usd=handle,
            actual_cost_usd=0.82,
        )
    assert ok is True

    def boom(*_a, **_k):
        raise TimeoutError("down")

    with patch("kazenai.spine.guard.urllib.request.urlopen", side_effect=boom):
        ok = reconcile_budget(
            org_id="o",
            workspace_id="w",
            run_id="r",
            reserved_cost_usd=handle,
            actual_cost_usd=0.82,
        )
    assert ok is False


def test_reserve_budget_async_uses_httpx_not_to_thread(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")
    monkeypatch.setenv("KAZENAI_FINOPS_CONTROL_PROFILE", "control")
    monkeypatch.setenv("KAZENAI_ENV", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if "/lifecycle" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "effective_lifecycle": {
                        "profile": "control",
                        "lifecycle_version": "reservation.lifecycle.v1",
                    },
                    "capabilities": {"reservation_settlement": True},
                },
            )
        return httpx.Response(
            200,
            json={
                "reservation_id": "res-async",
                "call_id": "call-async",
                "attempt": 1,
                "reserved_usd_micros": 1_000_000,
            },
        )

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    async def _run():
        with patch("asyncio.to_thread") as to_thread:
            with patch(
                "kazenai.spine.guard._ensure_server_lifecycle_compatible_async",
                return_value=None,
            ):
                with patch("httpx.AsyncClient", side_effect=client_factory):
                    handle = await reserve_budget_async(
                        org_id="o",
                        workspace_id="w",
                        run_id="r",
                        model="gpt-4o-mini",
                        estimated_cost_usd=1.0,
                        call_id="call-async",
                        attempt=1,
                        idempotency_key="reserve:call-async:1:fixed",
                    )
            to_thread.assert_not_called()
        return handle

    handle = asyncio.run(_run())
    assert isinstance(handle, ReservationHandle)
    assert handle.reservation_id == "res-async"
    reserve_reqs = [r for r in captured if "/v1/budget/reserve" in str(r.url)]
    assert reserve_reqs, f"expected reserve POST, got {captured!r}"
    body = json.loads(reserve_reqs[0].content.decode("utf-8"))
    assert body["idempotency_key"] == "reserve:call-async:1:fixed"
    assert body["call_id"] == "call-async"


def test_ensure_server_lifecycle_compatible_async_ok(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_CONTROL_PROFILE", "control")
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_closed")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_closed")
    guard_mod._LIFECYCLE_CAP_CACHE.clear()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "capabilities": {"reservation_settlement": True},
                "effective_lifecycle": {
                    "profile": "control",
                    "accounting_mode": "reservation_settlement",
                    "failure_policy": "fail_closed",
                },
            },
        )

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    async def _run():
        with patch("httpx.AsyncClient", side_effect=client_factory):
            await guard_mod._ensure_server_lifecycle_compatible_async(
                "http://finops:8090", org_id="org-x", api_key="k"
            )

    asyncio.run(_run())
    guard_mod._LIFECYCLE_CAP_CACHE.clear()


def test_async_lifecycle_capability_fetch_and_cache_identity(monkeypatch):
    """Async GET /v1/budget/lifecycle + cache keyed by (url, org, api_key_fp)."""
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")
    guard_mod._LIFECYCLE_CAP_CACHE.clear()

    hits: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "capabilities": {"reservation_settlement": True},
                "effective_lifecycle": {
                    "profile": "control",
                    "accounting_mode": "reservation_settlement",
                    "failure_policy": "fail_closed",
                },
            },
        )

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    key_a = guard_mod._lifecycle_cap_cache_key(
        "http://finops:8090", org_id="org-a", api_key="key-a"
    )
    key_b = guard_mod._lifecycle_cap_cache_key(
        "http://finops:8090", org_id="org-a", api_key="key-b"
    )
    assert key_a[0] == "http://finops:8090"
    assert key_a[1] == "org-a"
    assert key_a[2] != key_b[2]

    async def _run():
        with patch("httpx.AsyncClient", side_effect=client_factory):
            first = await guard_mod._fetch_server_lifecycle_capabilities_async(
                "http://finops:8090", org_id="org-a", api_key="key-a"
            )
            second = await guard_mod._fetch_server_lifecycle_capabilities_async(
                "http://finops:8090", org_id="org-a", api_key="key-a"
            )
            other = await guard_mod._fetch_server_lifecycle_capabilities_async(
                "http://finops:8090", org_id="org-a", api_key="key-b"
            )
        return first, second, other

    first, second, other = asyncio.run(_run())
    assert first.get("capabilities", {}).get("reservation_settlement") is True
    assert second == first
    assert other.get("capabilities", {}).get("reservation_settlement") is True
    # Cache hit for same identity; second identity does its own GET.
    assert len(hits) == 2
    assert all("/v1/budget/lifecycle" in u for u in hits)
    guard_mod._LIFECYCLE_CAP_CACHE.clear()


def test_signal_uses_shared_settle_payload(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")
    requests: list[dict] = []

    def fake_urlopen(req, timeout=None):
        requests.append(json.loads(req.data.decode("utf-8")))
        return _ack_resp()

    handle = ReservationHandle(
        reserved_cost_usd=0.5,
        reservation_id="res_sig",
        call_id="call_sig",
        attempt=3,
        lifecycle=True,
        finops_base_url="http://finops:8090",
    )
    with patch("kazenai.spine.guard.urllib.request.urlopen", side_effect=fake_urlopen):
        assert signal_reservation_event(
            org_id="o",
            run_id="r",
            reserved_cost_usd=handle,
            event="provider_started",
            required=True,
        )
    assert requests[0]["idempotency_key"] == "settle:res_sig:started"
    assert requests[0]["attempt"] == 3
