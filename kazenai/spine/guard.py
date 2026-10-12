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
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, TypeVar, Union

from ..context import get_current_context
from ..cost_engine import TokenCostEngine
from ..deployment import enforcement_fail_closed, finops_reservation_fail_closed
from ..enforcement import BudgetExceeded, BudgetUnavailable
from ..lifecycle_transitions import (
    build_settle_payload,
    create_attempt_identities,
)
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


def _finops_headers(org_id: str = "", *, api_key: str | None = None) -> dict[str, str]:
    """Headers for FinOps budget calls.

    The spine guard runs inside services that authenticate to FinOps with a shared
    service credential. FinOps org binding requires a service credential
    to assert the tenant org it operates on via ``X-Kazen-Act-As-Org``; without it,
    cross-org reserve/reconcile calls are rejected with 403 and enforcement fails closed.

    When ``api_key`` is supplied (frozen on a ReservationHandle), use it instead of
    the current process environment (immutable settlement binding).
    """
    key = (api_key if api_key is not None else _api_key()).strip()
    headers = {"Content-Type": "application/json", "X-API-Key": key}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    tenant_org = (org_id or "").strip()
    if tenant_org:
        headers["X-Kazen-Act-As-Org"] = tenant_org
    try:
        from kazenai.otel import inject_w3c_trace_headers

        inject_w3c_trace_headers(headers)
    except Exception:
        pass
    return headers


def _agentlens_mirror_enabled() -> bool:
    return os.getenv("KAZENAI_AGENTLENS_MIRROR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _default_sink() -> EventSink:
    """Process-cached MultiSink — one HttpSink worker set per ingest config."""
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
    """Price from usage. LiteLLM is optional and never authorizes a false zero.

    If LiteLLM is absent, raises, or returns a non-positive cost, fall through to
    the Kazen cost engine. Unknown LiteLLM models must not settle as $0 when the
    engine can still price the model.
    """
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
        # cost <= 0: treat as "LiteLLM could not price" — never as settled zero.
    except Exception:
        pass
    _in, _out, cost = _COST_ENGINE.from_usage(model=model, usage=usage)
    total_tokens = (_in or 0) + (_out or 0)
    return (total_tokens or None), float(cost or 0.0)


def _embedding_cost(model: str, response: Any, texts: list[str]) -> float:
    """Optional LiteLLM helper; never prefer a false zero over engine pricing."""
    try:
        import litellm  # type: ignore

        cost = float(litellm.embedding_cost(model=model, response=response) or 0.0)
        if cost > 0:
            return cost
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


def _apply_attribution(
    body: dict[str, Any],
    *,
    feature: str | None = None,
    business_subject_ref: str | None = None,
    feature_id: str | None = None,
    workflow_id: str | None = None,
    operation_id: str | None = None,
    attempt_id: str | None = None,
) -> None:
    """Attach business attribution metadata to a FinOps reserve/check body."""
    from ..attribution import attribution_for_reserve_body

    attrs = attribution_for_reserve_body(
        business_subject_ref=business_subject_ref,
        feature_id=feature_id,
        workflow_id=workflow_id,
        operation_id=operation_id,
        attempt_id=attempt_id,
        feature=feature,
    )
    # Prefer explicit feature kwarg for legacy ``feature`` when no feature_id resolved.
    if feature and not attrs.get("feature"):
        _apply_feature(body, feature)
    body.update(attrs)


def _lifecycle_budget_enabled() -> bool:
    """Control profile: use /reserve+/settle with call/attempt/reservation IDs.

    ``KAZENAI_FINOPS_STRICT_BUDGET`` alone does not enable the transactional
    lifecycle — see ``kazenai.lifecycle_config``.
    """
    from kazenai.lifecycle_config import lifecycle_budget_enabled

    return lifecycle_budget_enabled()


@dataclass
class ReservationHandle:
    """Pre-call reservation result. Float-compatible for legacy callers.

    When ``lifecycle`` is True, settlement uses the frozen FinOps binding
    captured at admission (URL/credential/org/workspace/failure policy), not the
    live process environment — so mid-flight env mutation cannot redirect cleanup.
    """

    reserved_cost_usd: float
    reservation_id: str = ""
    call_id: str = ""
    attempt: int = 1
    reserved_usd_micros: int = 0
    lifecycle: bool = False
    idempotency_key: str = ""
    decision_id: str = ""
    business_subject_ref: str = ""
    feature_id: str = ""
    workflow_id: str = ""
    # Immutable settlement destination / policy (populated at admit).
    finops_base_url: str = ""
    finops_api_key: str = field(default="", repr=False)  # never in logs/tracebacks
    org_id: str = ""
    workspace_id: str = ""
    failure_policy: str = ""
    accounting_owner: str = ""
    enforcement_owner: str = ""
    lifecycle_version: str = ""
    config_version: str = ""

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

    def settlement_base_url(self) -> str:
        return (self.finops_base_url or _finops_url()).rstrip("/")

    def settlement_api_key(self) -> str:
        return (self.finops_api_key or _api_key()).strip()

    def settlement_org_id(self, fallback: str = "") -> str:
        return (self.org_id or fallback or "").strip()


def _usd_to_micros_ceil(usd: float) -> int:
    """Round a USD amount up to whole micro-dollars.

    Costs are float sums (tokens x rate), so 0.000102 can arrive as
    0.00010200000000000001; ceiling that directly charges 103 micros for a
    102-micro call. Snap to nano-dollars first: float noise is far below 1e-9,
    while a genuine sub-micro fraction (3.3 micros) still rounds up.
    """
    from decimal import Decimal, ROUND_CEILING, ROUND_HALF_EVEN

    nanos = Decimal(str(usd)).quantize(Decimal("1e-9"), rounding=ROUND_HALF_EVEN)
    micros = int((nanos * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING))
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
    business_subject_ref: str | None = None,
    feature_id: str | None = None,
    workflow_id: str | None = None,
    operation_id: str | None = None,
    attempt_id: str | None = None,
) -> Union[ReservationHandle, float]:
    """Atomic pre-call FinOps reserve. Raises BudgetExceeded (402) or BudgetUnavailable.

    Under Control/strict lifecycle mode, calls ``POST /v1/budget/reserve`` and returns
    a ``ReservationHandle`` carrying reservation_id/call_id/attempt. Legacy mode keeps
    ``POST /v1/budget/check`` and returns a float (tests may also receive a Handle that
    compares equal to floats).

    Attribution fields prefer ambient ``AttributionContext`` then these kwargs.
    """
    from ..enforcement_owner import (
        assert_sdk_may_reserve,
        bind_decision_refs,
        gateway_owns_enforcement,
        get_enforcement_owner,
        should_skip_sdk_reserve,
    )

    if should_skip_sdk_reserve():
        owner = get_enforcement_owner()
        # Gateway-owned attempts: SDK returns a noop handle (no second debit).
        return ReservationHandle(
            reserved_cost_usd=0.0,
            reservation_id=(owner.reservation_id if owner else "") or "",
            call_id=(owner.call_id if owner else None) or (call_id or ""),
            attempt=int((owner.attempt if owner and owner.attempt is not None else None) or attempt or 1),
            lifecycle=_lifecycle_budget_enabled() or gateway_owns_enforcement(),
            decision_id=(owner.decision_id if owner else "") or "",
        )

    assert_sdk_may_reserve()

    base = _finops_url()
    if not base:
        if enforcement_fail_closed():
            raise BudgetUnavailable("FinOps URL not configured")
        return ReservationHandle(reserved_cost_usd=0.0) if _lifecycle_budget_enabled() else 0.0

    if _lifecycle_budget_enabled():
        handle = _reserve_lifecycle(
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
            business_subject_ref=business_subject_ref,
            feature_id=feature_id,
            workflow_id=workflow_id,
            operation_id=operation_id,
            attempt_id=attempt_id,
        )
        if getattr(handle, "reservation_id", None) or getattr(handle, "decision_id", None):
            bind_decision_refs(
                decision_id=str(getattr(handle, "decision_id", "") or "") or None,
                reservation_id=str(handle.reservation_id or "") or None,
                call_id=str(getattr(handle, "call_id", "") or call_id or ""),
                attempt=int(getattr(handle, "attempt", None) or attempt or 1),
            )
        return handle

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
    _apply_attribution(
        body,
        feature=feature,
        business_subject_ref=business_subject_ref,
        feature_id=feature_id,
        workflow_id=workflow_id,
        operation_id=operation_id,
        attempt_id=attempt_id,
    )

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
            reserved = float(payload.get("estimated_cost_usd") or 0.0)
            decision_id = str(payload.get("decision_id") or "") or None
            reservation_id = str(payload.get("reservation_id") or "") or None
            if decision_id or reservation_id:
                bind_decision_refs(
                    decision_id=decision_id,
                    reservation_id=reservation_id,
                    call_id=call_id,
                    attempt=attempt,
                )
            return reserved
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


def _reserve_timeout_s() -> float:
    return float(os.getenv("KAZENAI_FINOPS_RESERVE_TIMEOUT_S", "0.8") or "0.8")


def _parse_finops_http_error(exc: Exception) -> tuple[int | None, Any]:
    """Normalize urllib / httpx HTTP errors to (status_code, detail)."""
    status: int | None = None
    raw = ""
    if isinstance(exc, urllib.error.HTTPError):
        status = int(exc.code)
        try:
            raw = exc.read().decode("utf-8") if hasattr(exc, "read") else ""
        except Exception:
            raw = ""
    else:
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                status = int(getattr(response, "status_code", None) or 0) or None
            except Exception:
                status = None
            try:
                raw = response.text or ""
            except Exception:
                raw = ""
    detail: Any = {}
    try:
        detail = json.loads(raw) if raw else {}
        if isinstance(detail.get("detail"), dict):
            detail = detail["detail"]
    except Exception:
        detail = {"reason": raw}
    return status, detail


def _capture_settlement_binding(
    *,
    base: str,
    org_id: str,
    workspace_id: str,
) -> dict[str, str]:
    """Freeze admit-time FinOps destination + effective lifecycle policy."""
    from kazenai.lifecycle_config import (
        map_accounting_owner_to_enforcement,
        resolve_effective_lifecycle,
    )

    cfg = resolve_effective_lifecycle()
    return {
        "finops_base_url": (base or "").rstrip("/"),
        "finops_api_key": _api_key(),
        "org_id": (org_id or "").strip(),
        "workspace_id": (workspace_id or "").strip() or "default",
        "failure_policy": str(cfg.failure_policy or ""),
        "accounting_owner": str(cfg.accounting_owner or ""),
        "enforcement_owner": map_accounting_owner_to_enforcement(cfg.accounting_owner),
        "lifecycle_version": str(cfg.lifecycle_version or ""),
        "config_version": str(cfg.config_version or ""),
    }


# Cache key: (base_url, org_id, api_key_fingerprint) — identity-scoped.
_LIFECYCLE_CAP_CACHE: dict[tuple[str, str, str], tuple[float, dict[str, Any]]] = {}


def _lifecycle_cap_cache_key(
    base: str,
    *,
    org_id: str = "",
    api_key: str | None = None,
) -> tuple[str, str, str]:
    import hashlib

    url = (base or "").rstrip("/")
    org = (org_id or "").strip()
    key = (api_key if api_key is not None else _api_key()).strip()
    fp = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16] if key else ""
    return (url, org, fp)


def _fetch_server_lifecycle_capabilities(
    base: str,
    *,
    org_id: str = "",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Authenticated GET /v1/budget/lifecycle with a short process cache."""
    import time

    cache_key = _lifecycle_cap_cache_key(base, org_id=org_id, api_key=api_key)
    url = cache_key[0]
    if not url:
        return {}
    now = time.time()
    cached = _LIFECYCLE_CAP_CACHE.get(cache_key)
    if cached and (now - cached[0]) < 30.0:
        return cached[1]
    headers = _finops_headers(org_id, api_key=api_key)
    req = urllib.request.Request(
        f"{url}/v1/budget/lifecycle",
        headers=headers,
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=_reserve_timeout_s()) as resp:
            payload = json.loads(resp.read().decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                payload = {}
            _LIFECYCLE_CAP_CACHE[cache_key] = (now, payload)
            return payload
    except Exception:
        _log.debug("server lifecycle capability fetch failed", exc_info=True)
        return {}


async def _fetch_server_lifecycle_capabilities_async(
    base: str,
    *,
    org_id: str = "",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Native async GET /v1/budget/lifecycle (does not block the event loop)."""
    import time

    import httpx

    cache_key = _lifecycle_cap_cache_key(base, org_id=org_id, api_key=api_key)
    url = cache_key[0]
    if not url:
        return {}
    now = time.time()
    cached = _LIFECYCLE_CAP_CACHE.get(cache_key)
    if cached and (now - cached[0]) < 30.0:
        return cached[1]
    headers = _finops_headers(org_id, api_key=api_key)
    try:
        timeout = httpx.Timeout(_reserve_timeout_s())
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{url}/v1/budget/lifecycle", headers=headers)
            if resp.status_code != 200:
                return {}
            payload = resp.json()
            if not isinstance(payload, dict):
                payload = {}
            _LIFECYCLE_CAP_CACHE[cache_key] = (now, payload)
            return payload
    except Exception:
        _log.debug("async server lifecycle capability fetch failed", exc_info=True)
        return {}


def _looks_like_lifecycle_capabilities(payload: Mapping[str, Any]) -> bool:
    """True when payload is a lifecycle diagnostics / capabilities document."""
    if not payload:
        return False
    caps = payload.get("capabilities")
    if isinstance(caps, Mapping) and "reservation_settlement" in caps:
        return True
    if "reservation_settlement" in payload and "reservation_id" not in payload:
        return True
    return isinstance(payload.get("effective_lifecycle"), Mapping)


def _apply_lifecycle_capability_gate(caps: Mapping[str, Any]) -> None:
    """Shared Control gate after capabilities are fetched (sync or async)."""
    from kazenai.lifecycle_config import (
        ClientServerLifecycleMismatch,
        assert_compatible_with_server,
    )

    # Hermetic mocks often return reserve bodies for every urlopen; those must
    # not be interpreted as a negative capability advertisement.
    if not _looks_like_lifecycle_capabilities(caps):
        if enforcement_fail_closed() or finops_reservation_fail_closed():
            raise BudgetUnavailable(
                "FinOps lifecycle capabilities unavailable before Control reserve"
            )
        return
    try:
        assert_compatible_with_server(caps)
    except ClientServerLifecycleMismatch as exc:
        raise BudgetUnavailable(str(exc)) from exc


def _ensure_server_lifecycle_compatible(
    base: str,
    *,
    org_id: str = "",
    api_key: str | None = None,
) -> None:
    """Pre-dispatch Control gate: client/server lifecycle must agree."""
    from kazenai.lifecycle_config import resolve_effective_lifecycle

    if not resolve_effective_lifecycle().uses_reservation_settlement:
        return
    caps = _fetch_server_lifecycle_capabilities(base, org_id=org_id, api_key=api_key)
    _apply_lifecycle_capability_gate(caps)


async def _ensure_server_lifecycle_compatible_async(
    base: str,
    *,
    org_id: str = "",
    api_key: str | None = None,
) -> None:
    """Async twin of ``_ensure_server_lifecycle_compatible``."""
    from kazenai.lifecycle_config import resolve_effective_lifecycle

    if not resolve_effective_lifecycle().uses_reservation_settlement:
        return
    caps = await _fetch_server_lifecycle_capabilities_async(
        base, org_id=org_id, api_key=api_key
    )
    _apply_lifecycle_capability_gate(caps)


def _handle_to_reservation(
    payload: dict[str, Any],
    *,
    cid: str,
    att: int,
    micros: int,
    idem: str,
    business_subject_ref: str | None,
    feature_id: str | None,
    workflow_id: str | None,
    binding: Mapping[str, str] | None = None,
) -> ReservationHandle:
    reserved_micros = int(payload.get("reserved_usd_micros") or micros)
    decision_id = str(payload.get("decision_id") or "")
    receipt = payload.get("decision_receipt") if isinstance(payload.get("decision_receipt"), dict) else {}
    if not decision_id and receipt:
        decision_id = str(receipt.get("decision_id") or "")
    economics = (
        receipt.get("economics")
        if isinstance(receipt.get("economics"), dict)
        else {}
    )
    if not economics and isinstance(receipt.get("attribution"), dict):
        legacy = receipt["attribution"]
        economics = (
            legacy.get("economics")
            if isinstance(legacy.get("economics"), dict)
            else {}
        )
    bind = dict(binding or {})
    return ReservationHandle(
        reserved_cost_usd=_micros_to_usd(reserved_micros),
        reservation_id=str(payload.get("reservation_id") or ""),
        call_id=str(payload.get("call_id") or cid),
        attempt=int(payload.get("attempt") or att),
        reserved_usd_micros=reserved_micros,
        lifecycle=True,
        idempotency_key=idem,
        decision_id=decision_id,
        business_subject_ref=str(
            payload.get("business_subject_ref")
            or economics.get("business_subject_ref")
            or business_subject_ref
            or ""
        ),
        feature_id=str(
            payload.get("feature_id")
            or economics.get("feature_id")
            or feature_id
            or ""
        ),
        workflow_id=str(
            payload.get("workflow_id")
            or economics.get("workflow_id")
            or workflow_id
            or ""
        ),
        finops_base_url=str(bind.get("finops_base_url") or ""),
        finops_api_key=str(bind.get("finops_api_key") or ""),
        org_id=str(bind.get("org_id") or ""),
        workspace_id=str(bind.get("workspace_id") or ""),
        failure_policy=str(bind.get("failure_policy") or ""),
        accounting_owner=str(bind.get("accounting_owner") or ""),
        enforcement_owner=str(bind.get("enforcement_owner") or ""),
        lifecycle_version=str(bind.get("lifecycle_version") or ""),
        config_version=str(bind.get("config_version") or ""),
    )


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
    business_subject_ref: str | None = None,
    feature_id: str | None = None,
    workflow_id: str | None = None,
    operation_id: str | None = None,
    attempt_id: str | None = None,
) -> ReservationHandle:
    identities = create_attempt_identities(
        operation_id=operation_id,
        call_id=call_id,
        attempt=attempt,
        attempt_id=attempt_id,
        idempotency_key=idempotency_key,
    )
    cid = identities.call_id
    att = identities.attempt
    idem = identities.idempotency_key
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
    binding = _capture_settlement_binding(base=base, org_id=org_id, workspace_id=workspace_id)
    _ensure_server_lifecycle_compatible(
        base,
        org_id=org_id,
        api_key=binding.get("finops_api_key"),
    )
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
    _apply_attribution(
        body,
        feature=feature,
        business_subject_ref=business_subject_ref,
        feature_id=feature_id,
        workflow_id=workflow_id,
        operation_id=operation_id,
        attempt_id=attempt_id or identities.attempt_id,
    )
    data = json.dumps(body).encode("utf-8")
    from kazenai.otel import build_control_attributes, control_span, record_span_event

    with control_span(
        "kazenai.admission",
        kind="client",
        attributes=build_control_attributes(
            call_id=cid,
            attempt=att,
            attempt_id=identities.attempt_id,
            operation_id=operation_id,
            workflow_id=workflow_id,
            feature_id=feature_id or feature,
            business_subject_ref=business_subject_ref,
            model=model,
            reserved_usd_micros=micros,
            economic_state="reserving",
        ),
    ) as span:
        # Inject W3C headers inside the admission span.
        req = urllib.request.Request(
            f"{base}/v1/budget/reserve",
            data=data,
            headers=_finops_headers(org_id, api_key=binding.get("finops_api_key")),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_reserve_timeout_s()) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                handle = _handle_to_reservation(
                    payload,
                    cid=cid,
                    att=att,
                    micros=micros,
                    idem=idem,
                    business_subject_ref=business_subject_ref,
                    feature_id=feature_id or feature,
                    workflow_id=workflow_id,
                    binding=binding,
                )
                record_span_event(
                    span,
                    "reservation.lifecycle.reserved",
                    {
                        "kazenai.economic.reservation_id": handle.reservation_id or "",
                        "kazenai.decision.id": handle.decision_id or "",
                    },
                )
                return handle
        except urllib.error.HTTPError as exc:
            status, detail = _parse_finops_http_error(exc)
            if status == 402:
                reason = detail.get("reason") or detail.get("error_code") or "budget exhausted"
                raise BudgetExceeded(f"Pre-call reservation denied: {reason}") from exc
            if finops_reservation_fail_closed():
                raise BudgetUnavailable(f"Pre-call reservation HTTP {status}") from exc
            return ReservationHandle(reserved_cost_usd=0.0, call_id=cid, attempt=att, lifecycle=True)
        except (BudgetExceeded, BudgetUnavailable):
            raise
        except Exception as exc:
            if finops_reservation_fail_closed() or enforcement_fail_closed():
                raise BudgetUnavailable(f"Pre-call reservation unavailable: {exc}") from exc
            return ReservationHandle(reserved_cost_usd=0.0, call_id=cid, attempt=att, lifecycle=True)


def _build_lifecycle_reserve_body(
    *,
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
    business_subject_ref: str | None,
    feature_id: str | None,
    workflow_id: str | None,
    operation_id: str | None,
    attempt_id: str | None,
) -> tuple[dict[str, Any], Any, int]:
    """Shared reserve request body + identities + estimated micros."""
    identities = create_attempt_identities(
        operation_id=operation_id,
        call_id=call_id,
        attempt=attempt,
        attempt_id=attempt_id,
        idempotency_key=idempotency_key,
    )
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
        "call_id": identities.call_id,
        "attempt": identities.attempt,
        "idempotency_key": identities.idempotency_key,
        "org_id": org_id,
        "workspace_id": workspace_id,
        "estimated_usd_micros": micros,
        "run_id": run_id or "",
    }
    _apply_attribution(
        body,
        feature=feature,
        business_subject_ref=business_subject_ref,
        feature_id=feature_id,
        workflow_id=workflow_id,
        operation_id=operation_id,
        attempt_id=attempt_id or identities.attempt_id,
    )
    return body, identities, micros


async def _reserve_lifecycle_async(
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
    business_subject_ref: str | None = None,
    feature_id: str | None = None,
    workflow_id: str | None = None,
    operation_id: str | None = None,
    attempt_id: str | None = None,
) -> ReservationHandle:
    """Native async lifecycle admission (httpx — does not block the event loop)."""
    import httpx

    # Identities before awaiting admission — crash-safe recovery uses the same key.
    body, identities, micros = _build_lifecycle_reserve_body(
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
        business_subject_ref=business_subject_ref,
        feature_id=feature_id,
        workflow_id=workflow_id,
        operation_id=operation_id,
        attempt_id=attempt_id,
    )
    cid = identities.call_id
    att = identities.attempt
    idem = identities.idempotency_key
    binding = _capture_settlement_binding(base=base, org_id=org_id, workspace_id=workspace_id)
    await _ensure_server_lifecycle_compatible_async(
        base,
        org_id=org_id,
        api_key=binding.get("finops_api_key"),
    )
    timeout = httpx.Timeout(_reserve_timeout_s())
    from kazenai.otel import build_control_attributes, control_span, record_span_event

    with control_span(
        "kazenai.admission",
        kind="client",
        attributes=build_control_attributes(
            call_id=cid,
            attempt=att,
            attempt_id=identities.attempt_id,
            operation_id=operation_id,
            workflow_id=workflow_id,
            feature_id=feature_id or feature,
            business_subject_ref=business_subject_ref,
            model=model,
            reserved_usd_micros=micros,
            economic_state="reserving",
        ),
    ) as span:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{base}/v1/budget/reserve",
                    content=json.dumps(body),
                    headers=_finops_headers(org_id, api_key=binding.get("finops_api_key")),
                )
                if resp.status_code == 402:
                    detail: Any = {}
                    try:
                        detail = resp.json()
                        if isinstance(detail.get("detail"), dict):
                            detail = detail["detail"]
                    except Exception:
                        detail = {"reason": resp.text}
                    reason = detail.get("reason") or detail.get("error_code") or "budget exhausted"
                    raise BudgetExceeded(f"Pre-call reservation denied: {reason}")
                if resp.status_code >= 400:
                    if finops_reservation_fail_closed():
                        raise BudgetUnavailable(
                            f"Pre-call reservation HTTP {resp.status_code}"
                        )
                    return ReservationHandle(
                        reserved_cost_usd=0.0, call_id=cid, attempt=att, lifecycle=True
                    )
                payload = resp.json()
                handle = _handle_to_reservation(
                    payload if isinstance(payload, dict) else {},
                    cid=cid,
                    att=att,
                    micros=micros,
                    idem=idem,
                    business_subject_ref=business_subject_ref,
                    feature_id=feature_id or feature,
                    workflow_id=workflow_id,
                    binding=binding,
                )
                record_span_event(
                    span,
                    "reservation.lifecycle.reserved",
                    {
                        "kazenai.economic.reservation_id": handle.reservation_id or "",
                        "kazenai.decision.id": handle.decision_id or "",
                    },
                )
                return handle
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
    business_subject_ref: str | None = None,
    feature_id: str | None = None,
    workflow_id: str | None = None,
    operation_id: str | None = None,
    attempt_id: str | None = None,
) -> Union[ReservationHandle, float]:
    """Async admission using native httpx I/O (no asyncio.to_thread + urlopen)."""
    import httpx

    from ..enforcement_owner import bind_decision_refs

    base = _finops_url()
    if not base:
        if enforcement_fail_closed():
            raise BudgetUnavailable("FinOps URL not configured")
        return ReservationHandle(reserved_cost_usd=0.0) if _lifecycle_budget_enabled() else 0.0

    if _lifecycle_budget_enabled():
        handle = await _reserve_lifecycle_async(
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
            business_subject_ref=business_subject_ref,
            feature_id=feature_id,
            workflow_id=workflow_id,
            operation_id=operation_id,
            attempt_id=attempt_id,
        )
        if getattr(handle, "reservation_id", None) or getattr(handle, "decision_id", None):
            bind_decision_refs(
                decision_id=str(getattr(handle, "decision_id", "") or "") or None,
                reservation_id=str(handle.reservation_id or "") or None,
                call_id=str(getattr(handle, "call_id", "") or call_id or ""),
                attempt=int(getattr(handle, "attempt", None) or attempt or 1),
            )
        return handle

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
    _apply_attribution(
        body,
        feature=feature,
        business_subject_ref=business_subject_ref,
        feature_id=feature_id,
        workflow_id=workflow_id,
        operation_id=operation_id,
        attempt_id=attempt_id,
    )
    timeout = httpx.Timeout(_reserve_timeout_s())
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{base}/v1/budget/check",
                content=json.dumps({k: v for k, v in body.items() if v is not None}),
                headers=_finops_headers(org_id),
            )
            if resp.status_code == 402:
                try:
                    detail = resp.json()
                    reason = detail.get("reason") or detail.get("detail") or "budget exhausted"
                except Exception:
                    reason = "budget exhausted"
                raise BudgetExceeded(f"Pre-call reservation denied: {reason}")
            if resp.status_code >= 400:
                if finops_reservation_fail_closed():
                    raise BudgetUnavailable(f"Pre-call reservation HTTP {resp.status_code}")
                return 0.0
            payload = resp.json() if resp.content else {}
            reserved = float(payload.get("estimated_cost_usd") or 0.0)
            decision_id = str(payload.get("decision_id") or "") or None
            reservation_id = str(payload.get("reservation_id") or "") or None
            if decision_id or reservation_id:
                bind_decision_refs(
                    decision_id=decision_id,
                    reservation_id=reservation_id,
                    call_id=call_id,
                    attempt=attempt,
                )
            return reserved
    except (BudgetExceeded, BudgetUnavailable):
        raise
    except Exception as exc:
        if finops_reservation_fail_closed() or enforcement_fail_closed():
            raise BudgetUnavailable(f"Pre-call reservation unavailable: {exc}") from exc
        return 0.0


def _finops_result_ack(payload: dict[str, Any] | None, *, event: str) -> bool:
    """True only when FinOps acknowledges the intended lifecycle transition.

    HTTP 200 with ``result=pending`` is *not* a settlement acknowledgment.
    Accepts both ``result`` (canonical) and legacy ``status`` during migration.
    """
    if not isinstance(payload, dict):
        return False
    code = str(payload.get("result") or payload.get("status") or "").strip().lower()
    if event in {"usage_known"}:
        return code in {"applied", "duplicate"}
    if event in {"provider_started", "cancel_before_call", "outcome_unknown", "replay_pending"}:
        # pending is a valid durable mark for outcome_unknown / replay_pending.
        if event in {"outcome_unknown", "replay_pending"}:
            return code in {"applied", "duplicate", "pending"}
        return code in {"applied", "duplicate"}
    return code in {"applied", "duplicate"}


def _journal_settle_intent(
    payload: dict[str, Any],
    *,
    base: str,
    org_id: str,
    api_key: str | None,
    error: str = "pre_send",
) -> tuple[str, str]:
    """Journal exact settle intent before delivery. Returns (path, fingerprint)."""
    from kazenai.spine.settle_queue import intent_fingerprint, journal_settle_intent

    fp = intent_fingerprint(payload, org_id=org_id, base_url=base)
    path = journal_settle_intent(
        payload,
        error=error,
        base_url=base,
        org_id=org_id,
        api_key=None,  # never persist credentials in the spool
        persist_api_key=False,
    )
    return path, fp


def _ack_settle_intent(path: str | None, fingerprint: str | None) -> None:
    if not path or not fingerprint:
        return
    try:
        from kazenai.spine.settle_queue import get_settle_queue

        get_settle_queue().ack(path, fingerprint=fingerprint)
    except Exception:
        _log.debug("settle intent ack failed", exc_info=True)


def _enqueue_settle_intent(
    payload: dict[str, Any],
    *,
    base: str,
    org_id: str,
    api_key: str | None,
    error: str,
) -> None:
    """Fallback enqueue when pre-send journal was skipped."""
    try:
        _journal_settle_intent(
            payload, base=base, org_id=org_id, api_key=api_key, error=error
        )
    except Exception:
        _log.debug("settle intent enqueue failed", exc_info=True)


def _post_settle_sync(
    base: str,
    org_id: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
    api_key: str | None = None,
    enqueue_on_failure: bool = True,
    journal_before_send: bool = True,
) -> bool:
    event = str(payload.get("event") or "")
    from kazenai.otel import build_control_attributes, control_span, link_recovery_span

    attrs = build_control_attributes(
        call_id=str(payload.get("call_id") or "") or None,
        attempt=int(payload.get("attempt") or 0) or None,
        reservation_id=str(payload.get("reservation_id") or "") or None,
        settled_usd_micros=(
            int(payload["actual_usd_micros"])
            if payload.get("actual_usd_micros") is not None
            else None
        ),
        economic_state=event or "settle",
        decision_result=event or None,
    )
    span_cm: Any
    if event in {"outcome_unknown", "replay_pending"}:
        # Prefer linked recovery when the handle carried original span IDs.
        linked_trace = str(payload.get("trace_id") or "") or None
        linked_span = str(payload.get("span_id") or "") or None
        span_cm = link_recovery_span(
            "kazenai.recovery",
            linked_trace_id=linked_trace,
            linked_span_id=linked_span,
            attributes=attrs,
        )
    else:
        span_cm = control_span("kazenai.settlement", kind="client", attributes=attrs)

    journal_path: str | None = None
    journal_fp: str | None = None
    if journal_before_send and enqueue_on_failure:
        # Exact intent must survive process death during the send window.
        try:
            journal_path, journal_fp = _journal_settle_intent(
                payload, base=base, org_id=org_id, api_key=api_key, error="pre_send"
            )
        except Exception as exc:
            _log.error("settle intent journal failed before send: %s", exc)
            raise

    try:
        # Build/inject headers *inside* the settlement span so traceparent
        # parentage matches the active client span.
        with span_cm:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{base}/v1/budget/settle",
                data=body,
                headers=_finops_headers(org_id, api_key=api_key),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read() if hasattr(resp, "read") else b"{}"
                try:
                    parsed = json.loads(raw.decode("utf-8") or "{}")
                except Exception:
                    parsed = {}
                if not isinstance(parsed, dict):
                    parsed = {}
                ack = _finops_result_ack(parsed, event=event)
                if ack:
                    _ack_settle_intent(journal_path, journal_fp)
                elif enqueue_on_failure and not journal_path:
                    code = str(parsed.get("result") or parsed.get("status") or "unacked")
                    _enqueue_settle_intent(
                        payload,
                        base=base,
                        org_id=org_id,
                        api_key=api_key,
                        error=f"result_{code}",
                    )
                return ack
    except Exception:
        _log.debug("lifecycle settle %s failed", event, exc_info=True)
        if enqueue_on_failure and not journal_path:
            _enqueue_settle_intent(
                payload,
                base=base,
                org_id=org_id,
                api_key=api_key,
                error="transport_error",
            )
        return False


def reconcile_budget(
    *,
    org_id: str,
    workspace_id: str,
    run_id: str,
    reserved_cost_usd: float | ReservationHandle = 0.0,
    actual_cost_usd: float,
    feature: str | None = None,
    reservation: ReservationHandle | None = None,
) -> bool:
    """Post-call settle. Returns True only when FinOps acknowledges settlement.

    Lifecycle/Control uses ``/v1/budget/settle`` synchronously so stream finalize
    can distinguish local completion from durable financial ack. Legacy uses
    ``/reconcile``.
    """
    handle = reservation if isinstance(reservation, ReservationHandle) else None
    if isinstance(reserved_cost_usd, ReservationHandle):
        handle = reserved_cost_usd
        reserved_f = float(handle.reserved_cost_usd)
    else:
        reserved_f = float(reserved_cost_usd)

    # Prefer frozen admit-time binding over live env.
    base = handle.settlement_base_url() if handle is not None else _finops_url()
    settle_org = handle.settlement_org_id(org_id) if handle is not None else org_id
    settle_key = handle.settlement_api_key() if handle is not None else None

    use_lifecycle = bool(
        (handle and handle.lifecycle and handle.reservation_id)
        or _lifecycle_budget_enabled()
    )
    if not base:
        return False
    if use_lifecycle and not (handle and handle.reservation_id):
        _log.warning(
            "lifecycle settle skipped: missing reservation_id (refusing legacy reconcile in Control)"
        )
        return False
    if not use_lifecycle and not run_id:
        return False

    if use_lifecycle and handle is not None and handle.reservation_id:
        # provider_started must already have been recorded before dispatch.
        # Reconcile only settles known usage; never invents a second start.
        actual_micros = _usd_to_micros_ceil(float(actual_cost_usd))
        payload = build_settle_payload(
            reservation_id=handle.reservation_id,
            call_id=handle.call_id or f"call_{run_id}",
            attempt=int(handle.attempt or 1),
            org_id=settle_org,
            event="usage_known",
            actual_usd_micros=actual_micros,
        )
        return _post_settle_sync(
            base, settle_org, payload, timeout_s=2.0, api_key=settle_key
        )

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
            return True
    except Exception:
        _log.debug("legacy reconcile failed", exc_info=True)
        return False


async def reconcile_budget_async(
    *,
    org_id: str,
    workspace_id: str,
    run_id: str,
    reserved_cost_usd: float | ReservationHandle = 0.0,
    actual_cost_usd: float,
    feature: str | None = None,
    reservation: ReservationHandle | None = None,
) -> bool:
    """Native async post-call settle (httpx)."""
    import httpx

    handle = reservation if isinstance(reservation, ReservationHandle) else None
    if isinstance(reserved_cost_usd, ReservationHandle):
        handle = reserved_cost_usd
        reserved_f = float(handle.reserved_cost_usd)
    else:
        reserved_f = float(reserved_cost_usd)

    base = handle.settlement_base_url() if handle is not None else _finops_url()
    settle_org = handle.settlement_org_id(org_id) if handle is not None else org_id
    settle_key = handle.settlement_api_key() if handle is not None else None

    use_lifecycle = bool(
        (handle and handle.lifecycle and handle.reservation_id)
        or _lifecycle_budget_enabled()
    )
    if not base:
        return False
    if use_lifecycle and not (handle and handle.reservation_id):
        _log.warning(
            "lifecycle settle skipped: missing reservation_id (refusing legacy reconcile in Control)"
        )
        return False
    if not use_lifecycle and not run_id:
        return False

    timeout = httpx.Timeout(2.0)
    journal_path: str | None = None
    journal_fp: str | None = None
    lifecycle_payload: dict[str, Any] | None = None
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if use_lifecycle and handle is not None and handle.reservation_id:
                # provider_started must already have been recorded before dispatch.
                actual_micros = _usd_to_micros_ceil(float(actual_cost_usd))
                payload = build_settle_payload(
                    reservation_id=handle.reservation_id,
                    call_id=handle.call_id or f"call_{run_id}",
                    attempt=int(handle.attempt or 1),
                    org_id=settle_org,
                    event="usage_known",
                    actual_usd_micros=actual_micros,
                )
                lifecycle_payload = payload
                # Journal before await so CancelledError / ConnectError keep intent.
                journal_path, journal_fp = _journal_settle_intent(
                    payload,
                    base=base,
                    org_id=settle_org,
                    api_key=settle_key,
                    error="pre_send",
                )
                from kazenai.otel import build_control_attributes, control_span

                with control_span(
                    "kazenai.settlement",
                    kind="client",
                    attributes=build_control_attributes(
                        call_id=payload.get("call_id"),
                        attempt=int(payload.get("attempt") or 0) or None,
                        reservation_id=payload.get("reservation_id"),
                        settled_usd_micros=actual_micros,
                        economic_state="usage_known",
                    ),
                ):
                    resp = await client.post(
                        f"{base}/v1/budget/settle",
                        content=json.dumps(payload),
                        headers=_finops_headers(settle_org, api_key=settle_key),
                    )
                if resp.status_code >= 400:
                    _log.debug(
                        "lifecycle settle usage_known failed HTTP %s",
                        resp.status_code,
                    )
                    return False
                try:
                    parsed = resp.json()
                except Exception:
                    parsed = {}
                if not isinstance(parsed, dict):
                    parsed = {}
                ack = _finops_result_ack(parsed, event="usage_known")
                if ack:
                    _ack_settle_intent(journal_path, journal_fp)
                return ack

            payload = {
                "org_id": org_id,
                "workspace_id": workspace_id,
                "run_id": run_id,
                "reserved_cost_usd": reserved_f,
                "actual_cost_usd": float(actual_cost_usd),
            }
            _apply_feature(payload, feature)
            resp = await client.post(
                f"{base}/v1/budget/reconcile",
                content=json.dumps(payload),
                headers=_finops_headers(org_id),
            )
            return resp.status_code < 400
    except BaseException as exc:
        # Preserve CancelledError after journaling; intent must survive.
        import asyncio as _asyncio

        if lifecycle_payload is not None and journal_path is None:
            try:
                _journal_settle_intent(
                    lifecycle_payload,
                    base=base,
                    org_id=settle_org,
                    api_key=settle_key,
                    error="transport_or_cancel",
                )
            except Exception:
                _log.debug("async settle journal on failure failed", exc_info=True)
        if isinstance(exc, (_asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            raise
        _log.debug("async reconcile failed", exc_info=True)
        return False


def signal_reservation_event(
    *,
    org_id: str,
    run_id: str,
    reserved_cost_usd: float | ReservationHandle,
    event: str,
    actual_cost_usd: float | None = None,
    required: bool = False,
) -> bool:
    """Synchronously advance one lifecycle reservation.

    Streaming calls use this at the provider-dispatch boundary so an abandoned
    or failed stream cannot later expire as though the provider was never
    called.  Legacy float reservations have no lifecycle to advance and are a
    successful no-op.
    """
    if not isinstance(reserved_cost_usd, ReservationHandle):
        return True
    handle = reserved_cost_usd
    if not (handle.lifecycle and handle.reservation_id):
        return True

    allowed_events = {
        "provider_started",
        "usage_known",
        "outcome_unknown",
        "cancel_before_call",
        "replay_pending",
    }
    if event not in allowed_events:
        raise ValueError(f"unsupported reservation lifecycle event: {event}")
    if event == "usage_known" and actual_cost_usd is None:
        raise ValueError("actual_cost_usd is required for usage_known")

    base = handle.settlement_base_url()
    settle_org = handle.settlement_org_id(org_id)
    settle_key = handle.settlement_api_key()
    if not base:
        if required:
            raise BudgetUnavailable("FinOps URL not configured for reservation lifecycle")
        return False

    actual_micros = (
        _usd_to_micros_ceil(float(actual_cost_usd or 0.0))
        if event == "usage_known"
        else None
    )
    payload = build_settle_payload(
        reservation_id=handle.reservation_id,
        call_id=handle.call_id or f"call_{run_id}",
        attempt=int(handle.attempt or 1),
        org_id=settle_org,
        event=event,
        actual_usd_micros=actual_micros,
    )
    if not _post_settle_sync(
        base, settle_org, payload, timeout_s=_reserve_timeout_s(), api_key=settle_key
    ):
        if required:
            raise BudgetUnavailable(
                f"reservation lifecycle {event} unavailable"
            )
        return False
    return True


async def signal_reservation_event_async(
    *,
    org_id: str,
    run_id: str,
    reserved_cost_usd: float | ReservationHandle,
    event: str,
    actual_cost_usd: float | None = None,
    required: bool = False,
) -> bool:
    """Native async lifecycle advance (httpx)."""
    import httpx

    if not isinstance(reserved_cost_usd, ReservationHandle):
        return True
    handle = reserved_cost_usd
    if not (handle.lifecycle and handle.reservation_id):
        return True

    allowed_events = {
        "provider_started",
        "usage_known",
        "outcome_unknown",
        "cancel_before_call",
        "replay_pending",
    }
    if event not in allowed_events:
        raise ValueError(f"unsupported reservation lifecycle event: {event}")
    if event == "usage_known" and actual_cost_usd is None:
        raise ValueError("actual_cost_usd is required for usage_known")

    base = handle.settlement_base_url()
    settle_org = handle.settlement_org_id(org_id)
    settle_key = handle.settlement_api_key()
    if not base:
        if required:
            raise BudgetUnavailable("FinOps URL not configured for reservation lifecycle")
        return False

    actual_micros = (
        _usd_to_micros_ceil(float(actual_cost_usd or 0.0))
        if event == "usage_known"
        else None
    )
    payload = build_settle_payload(
        reservation_id=handle.reservation_id,
        call_id=handle.call_id or f"call_{run_id}",
        attempt=int(handle.attempt or 1),
        org_id=settle_org,
        event=event,
        actual_usd_micros=actual_micros,
    )
    journal_path: str | None = None
    journal_fp: str | None = None
    try:
        # Journal before await so CancelledError / ConnectError keep intent.
        journal_path, journal_fp = _journal_settle_intent(
            payload,
            base=base,
            org_id=settle_org,
            api_key=settle_key,
            error="pre_send",
        )
        async with httpx.AsyncClient(timeout=httpx.Timeout(_reserve_timeout_s())) as client:
            resp = await client.post(
                f"{base}/v1/budget/settle",
                content=json.dumps(payload),
                headers=_finops_headers(settle_org, api_key=settle_key),
            )
            if resp.status_code >= 400:
                if required:
                    raise BudgetUnavailable(
                        f"reservation lifecycle {event} failed with HTTP {resp.status_code}"
                    )
                _log.debug(
                    "reservation lifecycle %s failed HTTP %s", event, resp.status_code
                )
                return False
            try:
                parsed = resp.json()
            except Exception:
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            ack = _finops_result_ack(parsed, event=event)
            if ack:
                _ack_settle_intent(journal_path, journal_fp)
            if not ack and required:
                raise BudgetUnavailable(
                    f"reservation lifecycle {event} not acknowledged "
                    f"(result={parsed.get('result') or parsed.get('status')!r})"
                )
            return ack
    except BudgetUnavailable:
        raise
    except BaseException as exc:
        import asyncio as _asyncio

        if journal_path is None:
            try:
                _journal_settle_intent(
                    payload,
                    base=base,
                    org_id=settle_org,
                    api_key=settle_key,
                    error="transport_or_cancel",
                )
            except Exception:
                _log.debug("async lifecycle journal on failure failed", exc_info=True)
        if isinstance(exc, (_asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            raise
        if required:
            raise BudgetUnavailable(
                f"reservation lifecycle {event} unavailable"
            ) from exc
        _log.debug("reservation lifecycle %s failed", event, exc_info=True)
        return False


def flush_pending_settlements(
    *,
    default_base_url: str = "",
    default_api_key: str | None = None,
    timeout_s: float = 2.0,
) -> dict[str, int]:
    """Public SDK entry to redeliver durable settle intents after restart."""
    from kazenai.spine.settle_queue import flush_pending_settlements as _flush

    return _flush(
        default_base_url=default_base_url or _finops_url(),
        default_api_key=default_api_key,
        timeout_s=timeout_s,
    )


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

    await reconcile_budget_async(
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
