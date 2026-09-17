"""Single canonical FinOps guard for every LLM / embedding call."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar, Union

from ..context import RunContext, get_current_context
from ..cost_engine import TokenCostEngine
from ..deployment import enforcement_fail_closed, finops_reservation_fail_closed
from ..enforcement import BudgetExceeded, BudgetUnavailable
from ..schema import KazenEvent, new_id, now_ms, validate_payload
from ..sinks import EventSink, HttpSink, HttpSinkConfig, MemorySink, MultiSink

_log = logging.getLogger("kazenai.spine")
_T = TypeVar("_T")
_COST_ENGINE = TokenCostEngine()

_default_sink_lock = threading.Lock()
_default_sink_instance: EventSink | None = None
_default_sink_key: tuple[str, str, str, str, str] | None = None


def reset_default_sink() -> None:
    """Test/helper: close and drop the process-cached default sink."""
    global _default_sink_instance, _default_sink_key
    with _default_sink_lock:
        sink = _default_sink_instance
        _default_sink_instance = None
        _default_sink_key = None
    if sink is not None:
        try:
            sink.close(deadline_s=1.0)
        except Exception:
            pass


def _finops_url() -> str:
    return (
        os.getenv("KAZENAI_FINOPS_URL", "").strip()
        or os.getenv("KAZENAI_FINOPS_INGEST_URL", "").strip()
    ).rstrip("/")


def _api_key() -> str:
    return (os.getenv("KAZENAI_FINOPS_API_KEY") or os.getenv("KAZENAI_API_KEY") or "").strip()


def _finops_headers(org_id: str = "") -> dict[str, str]:
    """Headers for FinOps budget calls.

    The spine guard runs inside services that authenticate to FinOps with a shared
    service credential. FinOps org binding (Loop 25 H4) requires a service credential
    to assert the tenant org it operates on via ``X-Kazen-Act-As-Org``; without it,
    cross-org reserve/reconcile calls are rejected with 403 and enforcement fails closed.
    """
    headers = {"Content-Type": "application/json", "X-API-Key": _api_key()}
    tenant_org = (org_id or "").strip()
    if tenant_org:
        headers["X-Kazen-Act-As-Org"] = tenant_org
    return headers


def _agentlens_mirror_enabled() -> bool:
    return os.getenv("KAZENAI_AGENTLENS_MIRROR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _default_sink() -> EventSink:
    """Process-cached MultiSink — one HttpSink worker set per ingest config (P3-4)."""
    global _default_sink_instance, _default_sink_key
    ingest = (
        os.getenv("KAZENAI_FINOPS_INGEST_URL")
        or os.getenv("KAZENAI_FINOPS_URL")
        or os.getenv("KAZENAI_INGEST_URL")
        or ""
    ).strip()
    api_key = _api_key()
    lens = (
        os.getenv("KAZENAI_AGENTLENS_INGEST_URL")
        or os.getenv("KAZENAI_AGENTLENS_URL")
        or os.getenv("KAZENAI_LENS_INGEST_URL")
        or ""
    ).strip()
    lens_key = (
        os.getenv("KAZENAI_AGENTLENS_API_KEY")
        or os.getenv("KAZENAI_LENS_API_KEY")
        or ""
    ).strip()
    mirror = "1" if _agentlens_mirror_enabled() else "0"
    key = (ingest, api_key, lens, lens_key, mirror)
    with _default_sink_lock:
        if _default_sink_instance is not None and _default_sink_key == key:
            return _default_sink_instance
        # Config changed — close previous.
        prev = _default_sink_instance
        _default_sink_instance = None
        _default_sink_key = None
    if prev is not None:
        try:
            prev.close(deadline_s=1.0)
        except Exception:
            pass

    sinks: list[EventSink] = [MemorySink(max_events=2000)]
    org_id = (os.getenv("KAZENAI_ORG_ID") or "").strip()
    from ..ingest_url import normalize_ingest_base_url

    if ingest:
        base = normalize_ingest_base_url(ingest)
        sinks.append(
            HttpSink(
                HttpSinkConfig(
                    ingest_url=base,
                    api_key=api_key,
                    org_id=org_id,
                    extra_headers={"X-Kazen-Source": "sdk-finops"},
                )
            )
        )
    if _agentlens_mirror_enabled() and lens and lens_key:
        base = normalize_ingest_base_url(lens)
        sinks.append(
            HttpSink(
                HttpSinkConfig(
                    ingest_url=base,
                    api_key=lens_key,
                    org_id=org_id,
                    # Metadata mirror only — FinOps must not settle if these are mis-routed.
                    extra_headers={"X-Kazen-Source": "sdk-lens-mirror"},
                )
            )
        )
    sink: EventSink = MultiSink(sinks)
    with _default_sink_lock:
        _default_sink_instance = sink
        _default_sink_key = key
        return sink


def _emit_event(
    event: KazenEvent,
    *,
    event_sink: EventSink | None,
) -> None:
    try:
        validate_payload(event.event_type, event.payload)
    except Exception:
        _log.debug("spine event payload validation skipped", exc_info=True)
    sink = event_sink or _default_sink()
    try:
        sink.emit(event)
    except Exception:
        _log.debug("spine event emit failed", exc_info=True)


def _budget_event(
    *,
    event_type: str,
    org_id: str,
    workspace_id: str,
    run_id: str,
    agent_id: str,
    agent_role: str,
    surface: str,
    project_id: str,
    step_id: str | None,
    parent_step_id: str | None,
    model: str,
    feature: str,
    allowed: bool,
    reason: str = "",
    reserved_cost_usd: float | None = None,
    projected_cost_usd: float | None = None,
    cost_usd: float | None = None,
) -> KazenEvent:
    payload: dict[str, Any] = {
        "allowed": allowed,
        "model": model,
        "feature": feature,
    }
    if reason:
        payload["reason"] = reason
    if reserved_cost_usd is not None:
        payload["reserved_cost_usd"] = reserved_cost_usd
    if projected_cost_usd is not None:
        payload["projected_cost_usd"] = projected_cost_usd
    if cost_usd is not None:
        payload["actual_cost_usd"] = cost_usd
    return KazenEvent(
        schema_version="1.2",
        ts_ms=now_ms(),
        event_id=new_id(),
        org_id=org_id,
        workspace_id=workspace_id,
        project_id=project_id,
        surface=surface,
        agent_id=agent_id,
        agent_role=agent_role,
        run_id=run_id,
        step_id=step_id,
        parent_step_id=parent_step_id,
        event_type=event_type,
        cost_usd=cost_usd,
        payload=payload,
    )


def _extract_usage_cost(model: str, response: Any) -> tuple[int | None, float]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None, 0.0
    try:
        import litellm  # type: ignore

        cost = float(litellm.completion_cost(completion_response=response, model=model) or 0.0)
        if cost > 0:
            total = None
            if isinstance(usage, dict):
                total = usage.get("total_tokens")
            else:
                total = getattr(usage, "total_tokens", None)
            return (int(total) if total is not None else None), cost
    except Exception:
        pass
    _in, _out, cost = _COST_ENGINE.from_usage(model=model, usage=usage)
    total_tokens = (_in or 0) + (_out or 0)
    return (total_tokens or None), float(cost or 0.0)


def _embedding_cost(model: str, response: Any, texts: list[str]) -> float:
    try:
        import litellm  # type: ignore

        return float(litellm.embedding_cost(model=model, response=response) or 0.0)
    except Exception:
        pass
    usage = getattr(response, "usage", None) or (response.get("usage") if isinstance(response, dict) else None)
    if usage is not None:
        _in, _out, cost = _COST_ENGINE.from_usage(model=model, usage=usage)
        if cost > 0:
            return float(cost)
    # Conservative fallback: ~4 chars per token
    est_tokens = sum(max(1, len(t) // 4) for t in texts)
    return float(_COST_ENGINE.cost_usd(model=model, input_tokens=est_tokens, output_tokens=0))


def _apply_feature(body: dict[str, Any], feature: str | None) -> None:
    feat = (feature or "").strip()
    if feat:
        body["feature"] = feat


def _lifecycle_budget_enabled() -> bool:
    """Control / strict mode: use /reserve+/settle with call/attempt/reservation IDs."""
    for key in (
        "KAZENAI_FINOPS_CONTROL_PROFILE",
        "KAZENAI_CONTROL_PROFILE",
        "KAZENAI_PROFILE",
        "KAZENAI_FINOPS_STRICT_BUDGET",
    ):
        val = os.getenv(key, "").strip().lower()
        if val in ("1", "true", "yes", "control", "strict"):
            return True
    return False


@dataclass
class ReservationHandle:
    """Pre-call reservation result. Float-compatible for legacy callers."""

    reserved_cost_usd: float
    reservation_id: str = ""
    call_id: str = ""
    attempt: int = 1
    reserved_usd_micros: int = 0
    lifecycle: bool = False
    idempotency_key: str = ""

    def __float__(self) -> float:
        return float(self.reserved_cost_usd)

    def __bool__(self) -> bool:
        return bool(self.reservation_id) or float(self.reserved_cost_usd) > 0.0

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            return float(self.reserved_cost_usd) == float(other)
        if isinstance(other, ReservationHandle):
            return (
                self.reservation_id == other.reservation_id
                and float(self.reserved_cost_usd) == float(other.reserved_cost_usd)
            )
        return NotImplemented


def _usd_to_micros_ceil(usd: float) -> int:
    from decimal import Decimal, ROUND_CEILING

    micros = int(
        (Decimal(str(usd)) * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING)
    )
    return max(0, micros)


def _micros_to_usd(micros: int) -> float:
    return float(micros) / 1_000_000.0


def reserve_budget(
    *,
    org_id: str,
    workspace_id: str,
    run_id: str,
    model: str = "",
    input_tokens: int = 1000,
    estimated_cost_usd: float | None = None,
    increment_step: bool = True,
    feature: str | None = None,
    call_id: str | None = None,
    attempt: int = 1,
    idempotency_key: str | None = None,
) -> Union[ReservationHandle, float]:
    """Atomic pre-call FinOps reserve. Raises BudgetExceeded (402) or BudgetUnavailable.

    Under Control/strict lifecycle mode, calls ``POST /v1/budget/reserve`` and returns
    a ``ReservationHandle`` carrying reservation_id/call_id/attempt. Legacy mode keeps
    ``POST /v1/budget/check`` and returns a float (tests may also receive a Handle that
    compares equal to floats).
    """
    base = _finops_url()
    if not base:
        if enforcement_fail_closed():
            raise BudgetUnavailable("FinOps URL not configured")
        return ReservationHandle(reserved_cost_usd=0.0) if _lifecycle_budget_enabled() else 0.0

    if _lifecycle_budget_enabled():
        return _reserve_lifecycle(
            base=base,
            org_id=org_id,
            workspace_id=workspace_id,
            run_id=run_id,
            model=model,
            input_tokens=input_tokens,
            estimated_cost_usd=estimated_cost_usd,
            feature=feature,
            call_id=call_id,
            attempt=attempt,
            idempotency_key=idempotency_key,
        )

    body: dict[str, Any] = {
        "org_id": org_id,
        "workspace_id": workspace_id,
        "run_id": run_id,
        "reserve": True,
        "increment_step": increment_step,
        "model": model or None,
        "input_tokens": int(input_tokens),
    }
    if estimated_cost_usd is not None:
        body["estimated_cost_usd"] = float(estimated_cost_usd)
        body["projected_spend_usd"] = float(estimated_cost_usd)
    _apply_feature(body, feature)

    data = json.dumps({k: v for k, v in body.items() if v is not None}).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/v1/budget/check",
        data=data,
        headers=_finops_headers(org_id),
        method="POST",
    )
    try:
        timeout_s = float(os.getenv("KAZENAI_FINOPS_RESERVE_TIMEOUT_S", "0.8") or "0.8")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return float(payload.get("estimated_cost_usd") or 0.0)
    except urllib.error.HTTPError as exc:
        if exc.code == 402:
            try:
                detail = json.loads(exc.read().decode("utf-8"))
                reason = detail.get("reason") or detail.get("detail") or "budget exhausted"
            except Exception:
                reason = "budget exhausted"
            raise BudgetExceeded(f"Pre-call reservation denied: {reason}") from exc
        if finops_reservation_fail_closed():
            raise BudgetUnavailable(f"Pre-call reservation HTTP {exc.code}") from exc
        return 0.0
    except (BudgetExceeded, BudgetUnavailable):
        raise
    except Exception as exc:
        if finops_reservation_fail_closed() or enforcement_fail_closed():
            raise BudgetUnavailable(f"Pre-call reservation unavailable: {exc}") from exc
        return 0.0


def _reserve_lifecycle(
    *,
    base: str,
    org_id: str,
    workspace_id: str,
    run_id: str,
    model: str,
    input_tokens: int,
    estimated_cost_usd: float | None,
    feature: str | None,
    call_id: str | None,
    attempt: int,
    idempotency_key: str | None,
) -> ReservationHandle:
    cid = (call_id or "").strip() or f"call_{new_id()}"
    att = max(1, int(attempt or 1))
    idem = (idempotency_key or "").strip() or f"reserve:{cid}:{att}:{new_id()}"
    if estimated_cost_usd is not None:
        est_usd = float(estimated_cost_usd)
    else:
        try:
            est_usd = float(
                _COST_ENGINE.cost_usd(
                    model=model or "unknown",
                    input_tokens=int(input_tokens),
                    output_tokens=0,
                )
                or 0.01
            )
        except Exception:
            est_usd = 0.01
    if est_usd <= 0:
        est_usd = 0.01
    micros = _usd_to_micros_ceil(est_usd)
    body: dict[str, Any] = {
        "lifecycle_version": "reservation.lifecycle.v1",
        "call_id": cid,
        "attempt": att,
        "idempotency_key": idem,
        "org_id": org_id,
        "workspace_id": workspace_id,
        "estimated_usd_micros": micros,
        "run_id": run_id or "",
    }
    _apply_feature(body, feature)
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/v1/budget/reserve",
        data=data,
        headers=_finops_headers(org_id),
        method="POST",
    )
    try:
        timeout_s = float(os.getenv("KAZENAI_FINOPS_RESERVE_TIMEOUT_S", "0.8") or "0.8")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            reserved_micros = int(payload.get("reserved_usd_micros") or micros)
            return ReservationHandle(
                reserved_cost_usd=_micros_to_usd(reserved_micros),
                reservation_id=str(payload.get("reservation_id") or ""),
                call_id=str(payload.get("call_id") or cid),
                attempt=int(payload.get("attempt") or att),
                reserved_usd_micros=reserved_micros,
                lifecycle=True,
                idempotency_key=idem,
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8") if hasattr(exc, "read") else ""
        detail: Any = {}
        try:
            detail = json.loads(raw) if raw else {}
            if isinstance(detail.get("detail"), dict):
                detail = detail["detail"]
        except Exception:
            detail = {"reason": raw}
        if exc.code == 402:
            reason = detail.get("reason") or detail.get("error_code") or "budget exhausted"
            raise BudgetExceeded(f"Pre-call reservation denied: {reason}") from exc
        if finops_reservation_fail_closed():
            raise BudgetUnavailable(f"Pre-call reservation HTTP {exc.code}") from exc
        return ReservationHandle(reserved_cost_usd=0.0, call_id=cid, attempt=att, lifecycle=True)
    except (BudgetExceeded, BudgetUnavailable):
        raise
    except Exception as exc:
        if finops_reservation_fail_closed() or enforcement_fail_closed():
            raise BudgetUnavailable(f"Pre-call reservation unavailable: {exc}") from exc
        return ReservationHandle(reserved_cost_usd=0.0, call_id=cid, attempt=att, lifecycle=True)


async def reserve_budget_async(
    *,
    org_id: str,
    workspace_id: str,
    run_id: str,
    model: str = "",
    input_tokens: int = 1000,
    estimated_cost_usd: float | None = None,
    increment_step: bool = True,
    feature: str | None = None,
    call_id: str | None = None,
    attempt: int = 1,
    idempotency_key: str | None = None,
) -> Union[ReservationHandle, float]:
    return await asyncio.to_thread(
        reserve_budget,
        org_id=org_id,
        workspace_id=workspace_id,
        run_id=run_id,
        model=model,
        input_tokens=input_tokens,
        estimated_cost_usd=estimated_cost_usd,
        increment_step=increment_step,
        feature=feature,
        call_id=call_id,
        attempt=attempt,
        idempotency_key=idempotency_key,
    )


def reconcile_budget(
    *,
    org_id: str,
    workspace_id: str,
    run_id: str,
    reserved_cost_usd: float | ReservationHandle = 0.0,
    actual_cost_usd: float,
    feature: str | None = None,
    reservation: ReservationHandle | None = None,
) -> None:
    """Post-call settle. Lifecycle/Control uses ``/v1/budget/settle``; legacy uses ``/reconcile``."""
    base = _finops_url()
    handle = reservation if isinstance(reservation, ReservationHandle) else None
    if isinstance(reserved_cost_usd, ReservationHandle):
        handle = reserved_cost_usd
        reserved_f = float(handle.reserved_cost_usd)
    else:
        reserved_f = float(reserved_cost_usd)

    use_lifecycle = bool(
        (handle and handle.lifecycle and handle.reservation_id)
        or _lifecycle_budget_enabled()
    )
    if not base:
        return
    if use_lifecycle and not (handle and handle.reservation_id):
        # Strict mode without a reservation handle: refuse silent legacy float reconcile.
        _log.warning(
            "lifecycle settle skipped: missing reservation_id (refusing legacy reconcile in Control)"
        )
        return
    if not use_lifecycle and not run_id:
        return

    def _do() -> None:
        if use_lifecycle and handle is not None and handle.reservation_id:
            actual_micros = _usd_to_micros_ceil(float(actual_cost_usd))
            # provider_started then usage_known (idempotent if already in_flight)
            for event, key_suffix, extra in (
                ("provider_started", "started", {}),
                (
                    "usage_known",
                    "usage",
                    {"actual_usd_micros": actual_micros},
                ),
            ):
                payload = {
                    "lifecycle_version": "reservation.lifecycle.v1",
                    "reservation_id": handle.reservation_id,
                    "call_id": handle.call_id or f"call_{run_id}",
                    "attempt": int(handle.attempt or 1),
                    "idempotency_key": f"settle:{handle.reservation_id}:{key_suffix}",
                    "event": event,
                    "org_id": org_id,
                    **extra,
                }
                body = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    f"{base}/v1/budget/settle",
                    data=body,
                    headers=_finops_headers(org_id),
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=2.0):
                        pass
                except Exception:
                    _log.debug("lifecycle settle %s failed", event, exc_info=True)
            return

        payload = {
            "org_id": org_id,
            "workspace_id": workspace_id,
            "run_id": run_id,
            "reserved_cost_usd": reserved_f,
            "actual_cost_usd": float(actual_cost_usd),
        }
        _apply_feature(payload, feature)
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/v1/budget/reconcile",
            data=body,
            headers=_finops_headers(org_id),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=2.0):
                pass
        except Exception:
            pass

    threading.Thread(target=_do, daemon=True).start()


def _resolve_context(
    *,
    org_id: str,
    workspace_id: str,
    run_id: str,
    agent_id: str,
    project_id: str | None,
) -> tuple[str, str, str, str, str | None, str | None]:
    ctx = get_current_context()
    if ctx is not None:
        return (
            ctx.org_id,
            ctx.workspace_id,
            ctx.run_id,
            ctx.agent_id,
            ctx.project_id,
            ctx.step_id,
        )
    return org_id, workspace_id, run_id, agent_id, project_id or "default", None


def guarded_llm_call(
    fn: Callable[[], _T],
    *,
    model: str,
    org_id: str,
    workspace_id: str = "default",
    run_id: str,
    agent_id: str = "agent",
    agent_role: str = "agent",
    surface: str = "kazenai-core",
    project_id: str | None = None,
    feature: str = "",
    projected_cost_usd: float | None = None,
    input_tokens: int = 1000,
    event_sink: EventSink | None = None,
    increment_step: bool = True,
) -> _T:
    """Reserve budget, invoke provider, reconcile actual cost, emit typed events."""
    oid, wid, rid, aid, pid, step_id = _resolve_context(
        org_id=org_id,
        workspace_id=workspace_id,
        run_id=run_id,
        agent_id=agent_id,
        project_id=project_id,
    )
    parent_step_id = get_current_context().parent_step_id if get_current_context() else None

    reserved = 0.0
    try:
        reserved = reserve_budget(
            org_id=oid,
            workspace_id=wid,
            run_id=rid,
            model=model,
            input_tokens=input_tokens,
            estimated_cost_usd=projected_cost_usd,
            increment_step=increment_step,
            feature=feature,
        )
        _emit_event(
            _budget_event(
                event_type="finops.budget.reserve",
                org_id=oid,
                workspace_id=wid,
                run_id=rid,
                agent_id=aid,
                agent_role=agent_role,
                surface=surface,
                project_id=pid or "default",
                step_id=step_id,
                parent_step_id=parent_step_id,
                model=model,
                feature=feature,
                allowed=True,
                reserved_cost_usd=float(reserved),
                projected_cost_usd=projected_cost_usd,
            ),
            event_sink=event_sink,
        )
    except BudgetExceeded as exc:
        _emit_event(
            _budget_event(
                event_type="finops.budget.denied",
                org_id=oid,
                workspace_id=wid,
                run_id=rid,
                agent_id=aid,
                agent_role=agent_role,
                surface=surface,
                project_id=pid or "default",
                step_id=step_id,
                parent_step_id=parent_step_id,
                model=model,
                feature=feature,
                allowed=False,
                reason=str(exc),
                projected_cost_usd=projected_cost_usd,
            ),
            event_sink=event_sink,
        )
        raise
    except BudgetUnavailable as exc:
        _emit_event(
            _budget_event(
                event_type="finops.budget.denied",
                org_id=oid,
                workspace_id=wid,
                run_id=rid,
                agent_id=aid,
                agent_role=agent_role,
                surface=surface,
                project_id=pid or "default",
                step_id=step_id,
                parent_step_id=parent_step_id,
                model=model,
                feature=feature,
                allowed=False,
                reason=str(exc),
                projected_cost_usd=projected_cost_usd,
            ),
            event_sink=event_sink,
        )
        raise

    started = time.perf_counter()
    resp = fn()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    tokens_used, actual_cost = _extract_usage_cost(model, resp)

    reconcile_budget(
        org_id=oid,
        workspace_id=wid,
        run_id=rid,
        reserved_cost_usd=reserved,
        actual_cost_usd=actual_cost,
        feature=feature,
    )

    model_event = KazenEvent(
        schema_version="1.2",
        ts_ms=now_ms(),
        event_id=new_id(),
        org_id=oid,
        workspace_id=wid,
        project_id=pid or "default",
        surface=surface,
        agent_id=aid,
        agent_role=agent_role,
        run_id=rid,
        step_id=step_id,
        parent_step_id=parent_step_id,
        event_type="model.call",
        tokens_used=tokens_used,
        cost_usd=actual_cost,
        latency_ms=int(elapsed_ms),
        payload={"model": model, "feature": feature, "elapsed_ms": elapsed_ms},
    )
    _emit_event(model_event, event_sink=event_sink)
    return resp


async def aguarded_llm_call(
    fn: Callable[[], Any],
    *,
    model: str,
    org_id: str,
    workspace_id: str = "default",
    run_id: str,
    agent_id: str = "agent",
    agent_role: str = "agent",
    surface: str = "kazenai-core",
    project_id: str | None = None,
    feature: str = "",
    projected_cost_usd: float | None = None,
    input_tokens: int = 1000,
    event_sink: EventSink | None = None,
    increment_step: bool = True,
) -> Any:
    """Async wrapper — awaits async ``fn`` or runs sync ``fn`` in a thread."""
    oid, wid, rid, aid, pid, step_id = _resolve_context(
        org_id=org_id,
        workspace_id=workspace_id,
        run_id=run_id,
        agent_id=agent_id,
        project_id=project_id,
    )
    parent_step_id = get_current_context().parent_step_id if get_current_context() else None

    reserved = 0.0
    try:
        reserved = await reserve_budget_async(
            org_id=oid,
            workspace_id=wid,
            run_id=rid,
            model=model,
            input_tokens=input_tokens,
            estimated_cost_usd=projected_cost_usd,
            increment_step=increment_step,
            feature=feature,
        )
        _emit_event(
            _budget_event(
                event_type="finops.budget.reserve",
                org_id=oid,
                workspace_id=wid,
                run_id=rid,
                agent_id=aid,
                agent_role=agent_role,
                surface=surface,
                project_id=pid or "default",
                step_id=step_id,
                parent_step_id=parent_step_id,
                model=model,
                feature=feature,
                allowed=True,
                reserved_cost_usd=float(reserved),
                projected_cost_usd=projected_cost_usd,
            ),
            event_sink=event_sink,
        )
    except BudgetExceeded as exc:
        _emit_event(
            _budget_event(
                event_type="finops.budget.denied",
                org_id=oid,
                workspace_id=wid,
                run_id=rid,
                agent_id=aid,
                agent_role=agent_role,
                surface=surface,
                project_id=pid or "default",
                step_id=step_id,
                parent_step_id=parent_step_id,
                model=model,
                feature=feature,
                allowed=False,
                reason=str(exc),
                projected_cost_usd=projected_cost_usd,
            ),
            event_sink=event_sink,
        )
        raise
    except BudgetUnavailable as exc:
        _emit_event(
            _budget_event(
                event_type="finops.budget.denied",
                org_id=oid,
                workspace_id=wid,
                run_id=rid,
                agent_id=aid,
                agent_role=agent_role,
                surface=surface,
                project_id=pid or "default",
                step_id=step_id,
                parent_step_id=parent_step_id,
                model=model,
                feature=feature,
                allowed=False,
                reason=str(exc),
                projected_cost_usd=projected_cost_usd,
            ),
            event_sink=event_sink,
        )
        raise

    started = time.perf_counter()
    if asyncio.iscoroutinefunction(fn):
        resp = await fn()
    else:
        resp = await asyncio.to_thread(fn)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    tokens_used, actual_cost = _extract_usage_cost(model, resp)

    reconcile_budget(
        org_id=oid,
        workspace_id=wid,
        run_id=rid,
        reserved_cost_usd=reserved,
        actual_cost_usd=actual_cost,
        feature=feature,
    )
    _emit_event(
        KazenEvent(
            schema_version="1.2",
            ts_ms=now_ms(),
            event_id=new_id(),
            org_id=oid,
            workspace_id=wid,
            project_id=pid or "default",
            surface=surface,
            agent_id=aid,
            agent_role=agent_role,
            run_id=rid,
            step_id=step_id,
            parent_step_id=parent_step_id,
            event_type="model.call",
            tokens_used=tokens_used,
            cost_usd=actual_cost,
            latency_ms=int(elapsed_ms),
            payload={"model": model, "feature": feature, "elapsed_ms": elapsed_ms},
        ),
        event_sink=event_sink,
    )
    return resp


def guarded_embedding_call(
    fn: Callable[[], Any],
    *,
    model: str,
    texts: list[str],
    org_id: str,
    workspace_id: str = "default",
    run_id: str,
    agent_id: str = "agent",
    agent_role: str = "agent",
    surface: str = "kazenai-core",
    project_id: str | None = None,
    feature: str = "embedding",
    event_sink: EventSink | None = None,
) -> Any:
    est_tokens = sum(max(1, len(t) // 4) for t in texts)
    resp = guarded_llm_call(
        fn,
        model=model,
        org_id=org_id,
        workspace_id=workspace_id,
        run_id=run_id,
        agent_id=agent_id,
        agent_role=agent_role,
        surface=surface,
        project_id=project_id,
        feature=feature,
        input_tokens=est_tokens,
        event_sink=event_sink,
        increment_step=True,
    )
    # Reconcile with embedding-specific cost if the generic path under-estimated.
    actual = _embedding_cost(model, resp, texts)
    oid, wid, rid, _, pid, _ = _resolve_context(
        org_id=org_id,
        workspace_id=workspace_id,
        run_id=run_id,
        agent_id=agent_id,
        project_id=project_id,
    )
    reconcile_budget(
        org_id=oid,
        workspace_id=wid,
        run_id=rid,
        reserved_cost_usd=0.0,
        actual_cost_usd=actual,
        feature=feature,
    )
    return resp
