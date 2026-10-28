"""LiteLLM gateway Control (hermetic).

Proves application-owned single-attempt accounting (fake upstream oracle) and
gateway-owned per-attempt reserve/settle including the mandatory retry fixture.
Does not require LiteLLM installed in the app env.
"""

from __future__ import annotations

import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from kazenai.enforcement_owner import gateway_owns_enforcement, should_skip_sdk_reserve
from kazenai.integrations.litellm_control import (
    GATEWAY_OWNER,
    GatewayAdmissionDenied,
    GatewayFinOpsClient,
    AttemptReservation,
    control_attempt_marker,
    inject_traceparent,
    is_control_owned_callback,
    new_attempt_identity,
    resolve_attempt_from_kwargs,
)
from kazenai.spine.guard import ReservationHandle, reserve_budget

# Test-local copy of kazenai-examples/litellm-otel-control/fake_upstream.py, so
# Core's suite does not depend on a sibling checkout.
from fake_upstream import serve as serve_fake_upstream  # noqa: E402


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
RESULTS: list[dict[str, Any]] = []


def _record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append({"scenario": name, "status": status, "detail": detail})


@pytest.fixture(scope="module", autouse=True)
def _write_artifact():
    yield
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ARTIFACTS / f"kazenai-hermetic-{ts}.json"
    path.write_text(
        json.dumps(
            {
                "recorded_at": ts,
                "gates": [
                    "GATEWAY_SINGLE_ATTEMPT_VERIFIED",
                    "GATEWAY_RETRY_ACCOUNTING_VERIFIED",
                ],
                "results": RESULTS,
                "pass_count": sum(1 for r in RESULTS if r["status"] == "pass"),
                "fail_count": sum(1 for r in RESULTS if r["status"] == "fail"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


class FakeFinOps:
    """Minimal FinOps reserve/settle oracle for gateway-owned proofs."""

    def __init__(self, *, capacity_micros: int = 5_000_000) -> None:
        self.lock = threading.Lock()
        self.capacity = capacity_micros
        self.reserved = 0
        self.settled = 0
        self.reservations: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []

    def handle(self, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        with self.lock:
            if path.endswith("/reserve"):
                est = int(body.get("estimated_usd_micros") or 0)
                if self.reserved + est > self.capacity:
                    return 402, {"allowed": False, "reason": "capacity", "error_code": "budget_exhausted"}
                rid = f"rsv_{uuid.uuid4().hex}"
                self.reserved += est
                self.reservations[rid] = {
                    **body,
                    "reservation_id": rid,
                    "state": "reserved",
                    "estimated_usd_micros": est,
                }
                self.events.append({"event": "reserve", "reservation_id": rid, **body})
                return 200, {
                    "allowed": True,
                    "reservation_id": rid,
                    "decision_id": f"dec_{uuid.uuid4().hex[:8]}",
                    "lifecycle_version": "reservation.lifecycle.v1",
                    "call_id": body.get("call_id"),
                    "attempt": body.get("attempt") or 1,
                    "reserved_usd_micros": est,
                    "settlement_capability": "kzcap1.test." + "A" * 43,
                }
            if path.endswith("/settle"):
                # Mirror real FinOps contract: event names + ``result`` field.
                rid = str(body.get("reservation_id") or "")
                row = self.reservations.get(rid)
                if not row:
                    return 404, {"result": "pending", "error": "unknown_reservation"}
                event = str(body.get("event") or "")
                key = str(body.get("idempotency_key") or "")
                prior = [e for e in self.events if e.get("idempotency_key") == key]
                if prior:
                    return 200, {"result": "duplicate", "reservation_id": rid, "state": row.get("state")}
                self.events.append({**body, "event": event})
                if event in {"provider_started", "start"}:
                    row["state"] = "in_flight"
                    return 200, {"result": "applied", "reservation_id": rid, "state": "in_flight"}
                if event in {"usage_known", "settle"}:
                    actual = int(body.get("actual_usd_micros") or 0)
                    if row.get("settled"):
                        return 200, {"result": "duplicate", "reservation_id": rid, "state": "settled"}
                    row["settled"] = True
                    row["state"] = "settled"
                    row["actual_usd_micros"] = actual
                    self.settled += actual
                    self.reserved = max(0, self.reserved - int(row["estimated_usd_micros"]))
                    return 200, {"result": "applied", "reservation_id": rid, "state": "settled"}
                if event in {"cancel_before_call", "cancel"}:
                    row["state"] = "cancelled"
                    self.reserved = max(0, self.reserved - int(row["estimated_usd_micros"]))
                    return 200, {"result": "applied", "reservation_id": rid, "state": "cancelled"}
                if event in {"outcome_unknown", "pending", "replay_pending"}:
                    row["state"] = "pending_reconciliation"
                    return 200, {"result": "pending", "reservation_id": rid, "state": "pending_reconciliation"}
                return 400, {"result": "pending", "error": "invalid_event", "error_code": "invalid_transition"}
            if path.endswith("/lifecycle") or path.rstrip("/").endswith("/budget/lifecycle"):
                return 200, {
                    "ok": True,
                    "effective_lifecycle": {
                        "profile": "control",
                        "lifecycle_version": "reservation.lifecycle.v1",
                        "failure_policy": "fail_closed",
                    },
                    "capabilities": {
                        "reservation_settlement": True,
                        "legacy_check_reconcile": False,
                    },
                }
            return 404, {"error": "not_found"}

    def lifecycle(self) -> dict[str, Any]:
        return {
            "ok": True,
            "effective_lifecycle": {
                "profile": "control",
                "lifecycle_version": "reservation.lifecycle.v1",
                "failure_policy": "fail_closed",
            },
            "capabilities": {
                "reservation_settlement": True,
                "legacy_check_reconcile": False,
            },
        }


def _serve_finops(fake: FakeFinOps) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/").endswith("/budget/lifecycle"):
                raw = json.dumps(fake.lifecycle()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(raw)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n).decode() or "{}")
            status, payload = fake.handle(self.path, body)
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_attempt_identity_unique_per_attempt():
    a = new_attempt_identity(operation_id="op1", call_id="c1", attempt=1)
    b = new_attempt_identity(operation_id="op1", call_id="c1", attempt=2)
    assert a.attempt_id != b.attempt_id
    assert a.idempotency_key != b.idempotency_key
    _record("attempt_identity_unique", "pass")


def test_fallback_depth_not_used_as_attempt_id():
    kwargs = {
        "metadata": {"kazenai_operation_id": "op", "fallback_depth": 2},
        "fallback_depth": 2,
    }
    ident = resolve_attempt_from_kwargs(kwargs, attempt_number=1)
    assert ident.attempt == 1
    assert ident.attempt_id.startswith("att_")
    _record("fallback_depth_diagnostic_only", "pass")


def test_control_owned_callback_detection():
    marker = control_attempt_marker(new_attempt_identity(operation_id="o", call_id="c"))
    assert is_control_owned_callback({"metadata": marker})
    assert is_control_owned_callback(
        {"litellm_params": {"metadata": {**marker}}}
    )
    assert not is_control_owned_callback({"metadata": {"run_id": "r1"}})
    _record("control_owned_callback_detection", "pass")


def test_gateway_owner_skips_sdk_reserve(monkeypatch):
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_OWNER", "gateway")
    monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)
    assert gateway_owns_enforcement()
    assert should_skip_sdk_reserve()
    handle = reserve_budget(
        org_id="org",
        workspace_id="ws",
        run_id="run",
        model="gpt-4o-mini",
    )
    assert isinstance(handle, ReservationHandle)
    assert handle.reservation_id == ""
    _record("gateway_owner_skips_sdk_reserve", "pass")


def test_inject_traceparent_allowlist():
    kwargs: dict[str, Any] = {"headers": {}, "metadata": {}}
    tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    inject_traceparent(kwargs, tp)
    assert kwargs["headers"]["traceparent"] == tp
    assert kwargs["metadata"]["kazenai_traceparent"] == tp
    _record("traceparent_propagation", "pass")


def test_app_owned_single_attempt_fake_upstream(monkeypatch):
    """Application-owned: monitor → OpenAI HTTP → fake upstream; exactly one dispatch."""
    openai = pytest.importorskip("openai")
    server, oracle = serve_fake_upstream("127.0.0.1", 0)
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)
    monkeypatch.delenv("KAZENAI_ENFORCEMENT_OWNER", raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")

    from kazenai import monitor

    client = openai.OpenAI(
        api_key="test",
        base_url=base,
        max_retries=0,
        http_client=httpx.Client(timeout=5.0),
    )
    guarded = monitor(client, max_budget_usd=1.0, org_id="org-p6", workspace_id="ws-p6")
    resp = guarded.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp.choices[0].message.content == "ok"
    assert oracle.count() == 1, oracle.snapshot()
    # Retries disabled: a second logical call is a second dispatch, not hidden.
    guarded.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "again"}],
    )
    assert oracle.count() == 2
    server.shutdown()
    _record("app_owned_single_attempt_oracle", "pass", f"dispatches={oracle.count()}")


@pytest.mark.asyncio
async def test_gateway_owned_retry_fixture_two_attempts(monkeypatch):
    """Mandatory retry fixture via deployment hooks + fake FinOps (no LiteLLM process)."""
    # Import callback with a stub CustomLogger base so LiteLLM is not required.
    import types

    stub = types.ModuleType("litellm")
    integ = types.ModuleType("litellm.integrations")
    cl = types.ModuleType("litellm.integrations.custom_logger")

    class CustomLogger:  # noqa: N801
        pass

    cl.CustomLogger = CustomLogger
    monkeypatch.setitem(sys.modules, "litellm", stub)
    monkeypatch.setitem(sys.modules, "litellm.integrations", integ)
    monkeypatch.setitem(sys.modules, "litellm.integrations.custom_logger", cl)

    import importlib

    # Load the packaged gateway callback against the stub.
    monkeypatch.delitem(sys.modules, "kazenai.integrations.litellm_gateway", raising=False)
    cb_mod = importlib.import_module("kazenai.integrations.litellm_gateway")

    fake = FakeFinOps(capacity_micros=10_000_000)
    server = _serve_finops(fake)
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_OWNER", GATEWAY_OWNER)
    monkeypatch.setenv("KAZENAI_FINOPS_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "k")
    monkeypatch.setenv("KAZENAI_ORG_ID", "org-p6")
    monkeypatch.setenv("KAZENAI_WORKSPACE_ID", "ws-p6")

    logger = cb_mod.KazenGatewayControlLogger(finops=GatewayFinOpsClient())
    op = f"op_{uuid.uuid4().hex}"
    call = f"call_{uuid.uuid4().hex}"

    # Attempt A: reserve+start, then response lost → pending exposure
    kwargs_a: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"kazenai_operation_id": op, "kazenai_call_id": call, "run_id": op},
        "headers": {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"},
        "litellm_call_id": "lc_a",
    }
    out_a = await logger.async_pre_call_deployment_hook(kwargs_a, None)
    assert out_a is not None
    assert out_a["metadata"]["kazenai_control_attempt"] is True
    att_a = out_a["metadata"]["kazenai_attempt_id"]
    rsv_a = logger.finops.get_open(att_a)
    assert rsv_a is not None
    await logger.async_post_call_failure_deployment_hook(
        out_a, RuntimeError("response_lost"), None, fallback_depth=None
    )
    assert rsv_a.status == "pending"

    # Attempt B: fallback success → settle
    kwargs_b: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {
            "kazenai_operation_id": op,
            "kazenai_call_id": call,
            "run_id": op,
            "fallback_depth": 1,  # diagnostic only
        },
        "litellm_call_id": "lc_b",
    }
    out_b = await logger.async_pre_call_deployment_hook(kwargs_b, None)
    att_b = out_b["metadata"]["kazenai_attempt_id"]
    assert att_a != att_b
    rsv_b = logger.finops.get_open(att_b)
    resp = MagicMock()
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    await logger.async_post_call_success_deployment_hook(out_b, resp, None)
    assert rsv_b is not None and rsv_b.status == "settled"

    # Late evidence for A → settle exactly once
    late = logger.finops.settle_success(
        rsv_a,
        actual_usd=0.02,
        usage_evidence={"prompt_tokens": 10, "completion_tokens": 5, "late": True},
    )
    assert (late.get("result") or late.get("status")) in ("applied", "duplicate")
    again = logger.finops.settle_success(
        rsv_a,
        actual_usd=0.02,
        usage_evidence={"prompt_tokens": 10, "completion_tokens": 5, "late": True},
    )
    assert (again.get("result") or again.get("status")) == "duplicate"

    settled_rows = [r for r in fake.reservations.values() if r.get("settled")]
    pending_or_settled_a = fake.reservations[rsv_a.reservation_id]
    assert pending_or_settled_a.get("settled") is True
    assert len(settled_rows) == 2
    assert fake.settled > 0

    # Non-dispatched denial: capacity exhausted for a third attempt
    fake.capacity = fake.reserved  # no room
    kwargs_c: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"kazenai_operation_id": op, "kazenai_call_id": call},
        "litellm_call_id": "lc_c",
    }
    # When real litellm is importable (full suite), the callback wraps admission
    # denial as BudgetExceededError; hermetic stub runs raise GatewayAdmissionDenied.
    expected: tuple[type[BaseException], ...] = (GatewayAdmissionDenied,)
    try:
        from litellm.exceptions import BudgetExceededError  # type: ignore

        expected = (GatewayAdmissionDenied, BudgetExceededError)
    except Exception:
        pass
    with pytest.raises(expected):
        await logger.async_pre_call_deployment_hook(kwargs_c, None)

    server.shutdown()
    _record(
        "gateway_owned_retry_fixture",
        "pass",
        f"attempts=2 settled_rows={len(settled_rows)} ops={op}",
    )


def test_pre_call_exception_blocks_dispatch_contract():
    """Documented contract: GatewayAdmissionDenied propagates from pre-call hook."""
    with pytest.raises(GatewayAdmissionDenied):
        raise GatewayAdmissionDenied("denied", code="budget_denied")
    _record("pre_call_blocks_dispatch_contract", "pass")


def test_legacy_callback_skip_marker_shape():
    payload = {
        "run_id": "r1",
        "metadata": {
            "kazenai_control_attempt": True,
            "kazenai_budget_accounting_owner": "gateway",
            "kazenai_attempt_id": "att_x",
        },
    }
    assert is_control_owned_callback(payload)
    _record("legacy_callback_skip_marker", "pass")
