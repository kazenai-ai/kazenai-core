from __future__ import annotations

import dataclasses

from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar

import asyncio
import inspect
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import aiohttp  # type: ignore
except Exception:  # pragma: no cover
    aiohttp = None

try:
    import structlog  # type: ignore
except Exception:  # pragma: no cover
    structlog = None

from .context import RunContext, get_current_context, use_context
from .provider_errors import is_definitive_provider_rejection as _definitive_provider_rejection
from .debug import DebugPrinter
from ._dispatch_gate import install_dispatch_gate
from .enforcement import BudgetExceeded, BudgetUnavailable, Enforcement, LoopDetected, RateLimitExceeded, StreamCutoffError, UnsupportedModeError
from .capture_policy import CaptureMode, build_content_ref, capture_disclosure, event_log_fields, resolve_capture_mode
from .loop_detector import LoopDetector, LoopScores
from .finops import FinOpsConfig, FinOpsController
from .schema import KazenEvent, new_id, now_ms
from .sinks import EventSink, HttpSink, HttpSinkConfig, JsonlSink, MemorySink, MultiSink
from .pricing_policy import UnknownModelError



def _log_event_safe(log: Any, event: Any) -> None:
    """INFO log without dumping bodies; compatible with stdlib and structlog."""
    fields = event_log_fields(event)
    try:
        log.info("kazenai.event", **fields)
    except TypeError:
        log.info("kazenai.event %s", fields)


def _monitor_log():
    if structlog is not None:
        return structlog.get_logger("kazenai.monitor")
    return logging.getLogger("kazenai.monitor")


_EXPECTED_FAIL_OPEN = (
    BudgetExceeded,
    BudgetUnavailable,
    LoopDetected,
    RateLimitExceeded,
    StreamCutoffError,
    UnsupportedModeError,
)


def _streaming_explicitly_denied() -> bool:
    """Hard kill-switch only. Control-certified sync streaming is allowed by default."""
    val = os.getenv("KAZENAI_DENY_STREAMING", "").strip().lower()
    return val in ("1", "true", "yes")


def _control_profile_active() -> bool:
    from kazenai.lifecycle_config import control_profile_active

    return control_profile_active()


def _control_profile_denies_streaming() -> bool:
    """Deprecated name: only the explicit deny switch blocks certified streaming now."""
    return _streaming_explicitly_denied()


def _reject_streaming_if_unsupported(kwargs: Mapping[str, Any]) -> None:
    if not kwargs.get("stream"):
        return
    if _streaming_explicitly_denied():
        raise UnsupportedModeError(
            "Streaming is disabled by KAZENAI_DENY_STREAMING",
            error_code="unsupported_mode",
        )


# Certified Control sync Chat Completions + Messages (stream + non-stream).
SUPPORTED_OPENAI_METHOD = "openai.chat.completions.create"
SUPPORTED_ANTHROPIC_METHOD = "anthropic.messages.create"
SUPPORTED_OPENAI_STREAM_METHOD = "openai.chat.completions.create.stream"
SUPPORTED_OPENAI_STREAM_MANAGER = "openai.chat.completions.stream"
SUPPORTED_ANTHROPIC_STREAM_METHOD = "anthropic.messages.create.stream"
SUPPORTED_ANTHROPIC_STREAM_MANAGER = "anthropic.messages.stream"
SUPPORTED_OPENAI_RESPONSES_METHOD = "openai.responses.create"
SUPPORTED_OPENAI_RESPONSES_PARSE_METHOD = "openai.responses.parse"
SUPPORTED_OPENAI_RESPONSES_STREAM_METHOD = "openai.responses.create.stream"
SUPPORTED_OPENAI_RESPONSES_STREAM_MANAGER = "openai.responses.stream"

_HELPER_INNER_CREATE_BYPASS: ContextVar[bool] = ContextVar(
    "kazenai_helper_inner_create_bypass", default=False
)


@contextmanager
def _helper_inner_create_bypass():
    """Let an outer official SDK helper own the one Control lifecycle.

    Both official ``.stream()`` managers dispatch through their provider's
    ``create(stream=True)`` method during ``__enter__``.  The outer helper
    wrapper already reserved and owns finalization, so the nested patched
    create must call the original SDK method without creating a second hold,
    reservation, or event.
    """
    token = _HELPER_INNER_CREATE_BYPASS.set(True)
    try:
        yield
    finally:
        _HELPER_INNER_CREATE_BYPASS.reset(token)


def _is_async_callable(fn: Any) -> bool:
    if inspect.iscoroutinefunction(fn):
        return True
    underlying = getattr(fn, "__func__", None)
    if underlying is not None and inspect.iscoroutinefunction(underlying):
        return True
    return False


def _client_is_async(client: Any) -> bool:
    """True for AsyncOpenAI / AsyncAnthropic (and duck-typed async create)."""
    name = type(client).__name__
    if "Async" in name:
        return True
    for attr_path in (
        ("chat", "completions", "create"),
        ("messages", "create"),
    ):
        obj: Any = client
        try:
            for attr in attr_path:
                obj = getattr(obj, attr)
        except Exception:
            continue
        if _is_async_callable(obj):
            return True
    return False


def _provider_retries_value(client: Any) -> Optional[int]:
    retries = getattr(client, "max_retries", None)
    if retries is None:
        return None
    try:
        return int(retries)
    except Exception:
        return None


def _require_zero_provider_retries() -> bool:
    return os.getenv("KAZENAI_REQUIRE_ZERO_PROVIDER_RETRIES", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _allow_provider_retries_escape() -> bool:
    return os.getenv("KAZENAI_ALLOW_PROVIDER_RETRIES", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _check_hidden_provider_retries(client: Any) -> None:
    """Direct-provider Control contract: prefer ``max_retries=0``.

    Does not mutate the caller's client. Nonzero retries are rejected when the
    Control profile is active or ``KAZENAI_REQUIRE_ZERO_PROVIDER_RETRIES=1``
    (unless ``KAZENAI_ALLOW_PROVIDER_RETRIES=1``). Observe/dev paths warn only.
    """
    retries = _provider_retries_value(client)
    if retries is None or retries == 0:
        return
    strict = _require_zero_provider_retries() or _control_profile_active()
    if strict and not _allow_provider_retries_escape():
        raise UnsupportedModeError(
            f"provider max_retries={retries} is unsupported under Control; "
            "construct the client with max_retries=0 (or clone) so each Kazen "
            "retry is a new billable attempt "
            "(set KAZENAI_ALLOW_PROVIDER_RETRIES=1 only for explicit escape)",
            error_code="unsupported_provider_retries",
        )
    _monitor_log().warning(
        "kazenai.provider_retries: client max_retries=%s; prefer max_retries=0 "
        "for strict direct-provider Control (each Kazen retry is a new attempt)",
        retries,
    )


def configure_direct_provider_client(client: Any, *, max_retries: int = 0) -> Any:
    """Return a monitored-safe client copy with explicit retry ownership.

    Prefer this (or constructing with ``max_retries=0``) over mutating a shared
    client in place. Uses SDK ``copy`` / ``with_options`` when available.
    """
    with_options = getattr(client, "with_options", None)
    if callable(with_options):
        try:
            return with_options(max_retries=int(max_retries))
        except Exception:
            pass
    copy_fn = getattr(client, "copy", None)
    if callable(copy_fn):
        try:
            return copy_fn(max_retries=int(max_retries))
        except TypeError:
            try:
                return copy_fn()
            except Exception:
                pass
        except Exception:
            pass
    # Return original; caller remains responsible for max_retries=0 construction.
    return client


def _unsupported_mode_stub(message: str) -> Callable[..., Any]:
    def _stub(*args: Any, **kwargs: Any) -> Any:
        raise UnsupportedModeError(message, error_code="unsupported_mode")

    return _stub


def _apply_output_token_bound(kwargs: Dict[str, Any], *, responses: bool = False) -> None:
    """Inject/clamp the output limit using model_pricing.default_max_output_tokens.

    Chat Completions: injects ``max_completion_tokens`` for OpenAI reasoning
    families (which reject ``max_tokens``) and ``max_tokens`` otherwise.
    Responses: ``max_output_tokens``, which also bounds reasoning tokens.
    """
    try:
        from .model_pricing import default_max_output_tokens, output_limit_param
    except Exception:
        return
    model = str(kwargs.get("model") or "").strip()
    if not model:
        return
    bound = default_max_output_tokens(model)
    if bound is None:
        return
    if responses:
        requested = kwargs.get("max_output_tokens")
        if requested is None:
            kwargs["max_output_tokens"] = int(bound)
            return
        try:
            if int(requested) > int(bound):
                kwargs["max_output_tokens"] = int(bound)
        except (TypeError, ValueError):
            return
        return
    for key in ("max_tokens", "max_completion_tokens"):
        if key in kwargs and kwargs[key] is not None:
            try:
                requested = int(kwargs[key])
            except Exception:
                return
            if requested > int(bound):
                kwargs[key] = int(bound)
            return
    kwargs[output_limit_param(model)] = int(bound)


def _log_fail_open(context: str, exc: BaseException) -> None:
    """Log unexpected exceptions on intentional fail-open paths without blocking calls."""
    if isinstance(exc, _EXPECTED_FAIL_OPEN):
        return
    _monitor_log().warning("kazenai.monitor fail-open: %s", context, exc_info=True)


# ---------------------------------------------------------------------------
# Pre-call cost projection
# ---------------------------------------------------------------------------

_COST_ENGINE: Any = None
# Conservative assumed token count for one call when flooring its cost pre-flight. This is a
# lower bound, not a real estimate — enough that a known-expensive model with no budget
# headroom is blocked BEFORE it runs rather than overshooting the cap once.
_PRECALL_MIN_TOKENS = 2000


def _estimate_input_tokens(kwargs: Mapping[str, Any]) -> int:
    """Return a conservative, content-free upper estimate for request input.

    Provider tokenizers are intentionally not a Core dependency.  A byte-count
    upper estimate avoids the unsafe ``len(text) // 4`` undercount for JSON,
    tools and non-Latin text while keeping request content in memory only.
    """

    excluded = {
        "model",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
    }

    def _size(value: Any, *, depth: int = 0) -> int:
        if depth > 32:
            return 64
        if value is None:
            return 0
        if isinstance(value, bytes):
            return len(value)
        if isinstance(value, str):
            return len(value.encode("utf-8", errors="replace"))
        if isinstance(value, (bool, int, float)):
            return len(str(value))
        if isinstance(value, Mapping):
            total = 2
            for key, item in value.items():
                total += _size(str(key), depth=depth + 1)
                total += _size(item, depth=depth + 1)
                total += 2
            return total
        if isinstance(value, (list, tuple, set)):
            return 2 + sum(_size(item, depth=depth + 1) + 1 for item in value)
        try:
            return len(repr(value).encode("utf-8", errors="replace"))
        except Exception:
            return 64

    request_body = {key: value for key, value in (kwargs or {}).items() if key not in excluded}
    # Keep the historical floor for request framing/provider overhead.  Above
    # that floor one UTF-8 byte is treated as at most one token.
    return max(_PRECALL_MIN_TOKENS, _size(request_body) + 64)


def _max_output_tokens(kwargs: Mapping[str, Any]) -> int:
    for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
        value = (kwargs or {}).get(key)
        if value is None:
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            break
    try:
        from .model_pricing import default_max_output_tokens

        return int(default_max_output_tokens(str((kwargs or {}).get("model") or "")) or 0)
    except Exception:
        return 0


def _stream_output_rate_usd_per_1k(model: str, explicit_rate: Optional[float]) -> float:
    """Resolve the local observable-output cutoff rate from canonical pricing."""
    if explicit_rate is not None and float(explicit_rate) > 0:
        return float(explicit_rate)
    if not model:
        return 0.0
    global _COST_ENGINE
    if _COST_ENGINE is None:
        from .cost_engine import TokenCostEngine

        _COST_ENGINE = TokenCostEngine()
    return float(_COST_ENGINE.price(model).output_per_1k)



def _postcall_cost_usd(
    *,
    model: str,
    usage: Any,
    tokens_used: Optional[int],
    usd_per_1k_tokens: Optional[float],
    pricing: Any = None,
) -> Optional[float]:
    """Actual call cost using the same price table as pre-call projection when possible."""
    if tokens_used is not None and usd_per_1k_tokens:
        return (float(tokens_used) * float(usd_per_1k_tokens)) / 1000.0
    model_name = str(model or "").strip()
    if not model_name or usage is None:
        return None
    try:
        global _COST_ENGINE
        if _COST_ENGINE is None:
            from .cost_engine import TokenCostEngine

            _COST_ENGINE = TokenCostEngine()

        def _field(name: str, *alts: str) -> int:
            for key in (name, *alts):
                val = getattr(usage, key, None)
                if val is None and isinstance(usage, dict):
                    val = usage.get(key)
                if val is not None:
                    return int(val)
            return 0

        # OpenAI-style vs Anthropic-style usage objects
        in_tok = _field("prompt_tokens", "input_tokens")
        out_tok = _field("completion_tokens", "output_tokens")
        if in_tok == 0 and out_tok == 0 and tokens_used:
            # Fall back to splitting total when only total is known.
            in_tok = int(tokens_used)
        from .pricing_context import current_pricing, scope_org
        from .request_pricing import reported_service_tier, settle_usage

        bound = pricing if pricing is not None else current_pricing()
        if bound is not None and usage is not None:
            micros = settle_usage(
                bound,
                usage,
                reported_tier=reported_service_tier(usage),
                org_id=scope_org(),
            )
            from .price_catalog import micro_usd_to_decimal

            cost = float(micro_usd_to_decimal(micros))
            return cost
        _in, _out, cost = _COST_ENGINE.from_usage(model=model_name, usage=usage)
        if cost and cost > 0:
            return float(cost)
    except Exception as exc:
        # The provider call already happened and was paid for. Never raise here:
        # an unpriceable result returns None and the caller holds the
        # reservation for reconciliation instead of discarding the response.
        logging.getLogger("kazenai.monitor").warning(
            "post-call cost unavailable (%s); settlement held for reconciliation",
            type(exc).__name__,
        )
    return None


def _bound_postcall():
    """Capture this call's frozen basis for a later stream finalize."""
    from .pricing_context import current_pricing

    bound = current_pricing()

    def _fn(**kw: Any) -> Optional[float]:
        return _postcall_cost_usd(**kw, pricing=bound)

    return _fn


def _arm_price_scope(
    *,
    provider: str,
    surface: str,
    client: Any,
    org_id: str,
    observe_unbounded: tuple = (),
    kwargs: Any = None,
) -> None:
    """Clear the previous call's basis and install this call's scope for ``kwargs``."""
    from .pricing_context import bind_call, set_current_pricing

    set_current_pricing(None)
    bind_call(
        provider=provider,
        surface=surface,
        client=client,
        org_id=org_id,
        observe_unbounded=tuple(observe_unbounded or ()),
        kwargs=kwargs,
    )


def _pricing_holds_reconciliation() -> bool:
    from .pricing_context import current_pricing

    pricing = current_pricing()
    return bool(pricing is not None and getattr(pricing, "hold_for_reconciliation", False))


def _precall_projection_usd(kwargs: Mapping[str, Any], usd_per_1k_tokens: Optional[float]) -> float:
    """Bounded pre-call exposure from estimated input plus enforced output cap.

    Passed as ``projected_cost_usd`` to ``Enforcement.check_local`` so a single expensive run
    at the budget edge is denied before execution. An unpriced model raises
    ``UnknownModelError`` under the default ``block`` policy; only an explicit
    ``usd_per_1k_tokens`` or the dev-only ``warn`` opt-in reaches the fallbacks below.
    """
    model = str((kwargs or {}).get("model") or "").strip()
    input_tokens = _estimate_input_tokens(kwargs)
    output_tokens = _max_output_tokens(kwargs)
    from .pricing_context import (
        current_pricing,
        pricing_scope_active,
        scope_client,
        scope_observe_unbounded,
        scope_org,
        scope_provider,
        scope_surface,
    )

    # The armed scope applies only to the request it was armed for.
    scoped = isinstance(kwargs, dict) and pricing_scope_active(kwargs)
    existing = current_pricing() if scoped else None
    if scoped and existing is None:
        from .request_pricing import prepare_provider_call

        try:
            existing = prepare_provider_call(
                kwargs,
                provider=scope_provider(),
                surface=scope_surface(),
                client=scope_client(),
                org_id=scope_org(),
                observe_unbounded=scope_observe_unbounded(),
            )
        except UnknownModelError:
            # Only an unpriced model may fall through to a caller-supplied rate.
            # Shape refusals (tools, n, tier, images) always propagate.
            if not usd_per_1k_tokens:
                raise
            existing = None
    # The certified catalog is the ceiling: a known model is never re-priced by a
    # caller-supplied flat rate (that could under-reserve). The rate below applies
    # only to a model the catalog does not price.
    if existing is not None and existing.reserved_usd > 0:
        return float(existing.reserved_usd)
    try:
        if model:
            global _COST_ENGINE
            if _COST_ENGINE is None:
                from .cost_engine import TokenCostEngine

                _COST_ENGINE = TokenCostEngine()
            cost = _COST_ENGINE.cost_usd(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            if cost and cost > 0:
                return float(cost)
    except UnknownModelError:
        # The block policy is part of the public financial-control contract,
        # not an internal pricing failure that may fail open. A caller-supplied
        # rate is an explicit price, so it is the one way past it.
        if not usd_per_1k_tokens:
            raise
    except Exception as exc:  # pragma: no cover - unexpected pricing internals fail open
        _log_fail_open("precall_projection", exc)
    if usd_per_1k_tokens:
        return float(usd_per_1k_tokens) * ((input_tokens + output_tokens) / 1000.0)
    return 0.0


# ---------------------------------------------------------------------------
# FinOps service pre-call reservation helpers
# ---------------------------------------------------------------------------


def _reservation_payload_fields(reserved) -> dict:
    """Carry call/attempt/reservation/decision IDs on model.call for FinOps/Lens join."""
    if reserved is None:
        return {}
    from .spine.guard import ReservationHandle

    if isinstance(reserved, ReservationHandle) and reserved.lifecycle:
        out = {
            "reservation_id": reserved.reservation_id,
            "call_id": reserved.call_id,
            "attempt": int(reserved.attempt or 1),
            "reserved_cost_usd": float(reserved.reserved_cost_usd),
            "reserved_usd_micros": int(reserved.reserved_usd_micros or 0),
            "decision_id": reserved.decision_id or None,
            "business_subject_ref": reserved.business_subject_ref or None,
            "feature_id": reserved.feature_id or None,
            "workflow_id": reserved.workflow_id or None,
        }
        if reserved.reserved_usd_micros:
            out["actual_usd_micros"] = None  # filled after usage when known
        pricing = getattr(reserved, "pricing_call", None)
        if pricing is not None:
            from .request_pricing import pricing_evidence

            out.update(pricing_evidence(pricing))
        return {k: v for k, v in out.items() if v is not None}
    try:
        return {"reserved_cost_usd": float(reserved)}
    except Exception:
        return {}


def _attribution_payload_fields(explicit: Optional[Mapping[str, Any]] = None) -> dict:
    """Emit commercial attribution on every model.call (local + shared-reserve paths).

    Prefer an explicit mapping (stream attempts capture defaults at start). Fall back
    to the ambient AttributionContext installed by patch wrappers.
    """
    src: Dict[str, Any] = {}
    if explicit:
        src.update({k: explicit.get(k) for k in (
            "business_subject_ref",
            "feature_id",
            "workflow_id",
            "operation_id",
            "attempt_id",
        )})
    else:
        try:
            from .attribution import get_attribution

            attr = get_attribution()
        except Exception:
            attr = None
        if attr is not None:
            for key in (
                "business_subject_ref",
                "feature_id",
                "workflow_id",
                "operation_id",
                "attempt_id",
            ):
                src[key] = getattr(attr, key, None)
    return {k: v for k, v in src.items() if v is not None and v != ""}

def _try_reserve_budget(
    *,
    org_id: str,
    workspace_id: str,
    run_id: str,
    call_id: Optional[str] = None,
    estimated_cost_usd: Optional[float] = None,
    business_subject_ref: Optional[str] = None,
    feature_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    operation_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
    feature: Optional[str] = None,
    model: str = "",
    input_tokens: int = 0,
):
    """Pre-call reservation via canonical spine (raises on fail-closed deny).

    Returns a float (legacy) or ReservationHandle (Control lifecycle).
    Attribution prefers ambient AttributionContext, then these kwargs.
    """
    from .spine.guard import reserve_budget

    try:
        return reserve_budget(
            org_id=org_id,
            workspace_id=workspace_id,
            run_id=run_id,
            increment_step=False,
            call_id=call_id,
            model=model,
            input_tokens=input_tokens,
            estimated_cost_usd=estimated_cost_usd,
            business_subject_ref=business_subject_ref,
            feature_id=feature_id,
            workflow_id=workflow_id,
            operation_id=operation_id,
            attempt_id=attempt_id,
            feature=feature,
        )
    except BudgetUnavailable:
        raise
    except BudgetExceeded:
        raise


def _reserve_with_local_hold(
    *,
    enforcement: Enforcement,
    projected_cost_usd: float,
    reserve_kwargs: Mapping[str, Any],
) -> Tuple[float, Any]:
    """Acquire local + shared admission and unwind local state on denial."""
    held_projection = enforcement.check_local(projected_cost_usd=projected_cost_usd)
    try:
        reserved = _try_reserve_budget(**dict(reserve_kwargs))
    except Exception:
        enforcement.release_projection(held_projection)
        raise
    from .pricing_context import current_pricing
    from .request_pricing import attach_pricing

    attach_pricing(reserved, current_pricing())
    return held_projection, reserved


async def _try_reserve_budget_async(
    *,
    org_id: str,
    workspace_id: str,
    run_id: str,
    call_id: Optional[str] = None,
    estimated_cost_usd: Optional[float] = None,
    business_subject_ref: Optional[str] = None,
    feature_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    operation_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
    feature: Optional[str] = None,
    model: str = "",
    input_tokens: int = 0,
):
    from .spine.guard import reserve_budget_async

    return await reserve_budget_async(
        org_id=org_id,
        workspace_id=workspace_id,
        run_id=run_id,
        increment_step=False,
        call_id=call_id,
        model=model,
        input_tokens=input_tokens,
        estimated_cost_usd=estimated_cost_usd,
        business_subject_ref=business_subject_ref,
        feature_id=feature_id,
        workflow_id=workflow_id,
        operation_id=operation_id,
        attempt_id=attempt_id,
        feature=feature,
    )


async def _reserve_with_local_hold_async(
    *,
    enforcement: Enforcement,
    projected_cost_usd: float,
    reserve_kwargs: Mapping[str, Any],
) -> Tuple[float, Any]:
    held_projection = enforcement.check_local(projected_cost_usd=projected_cost_usd)
    try:
        reserved = await _try_reserve_budget_async(**dict(reserve_kwargs))
    except Exception:
        enforcement.release_projection(held_projection)
        raise
    from .pricing_context import current_pricing
    from .request_pricing import attach_pricing

    attach_pricing(reserved, current_pricing())
    return held_projection, reserved


async def _try_reconcile_budget_async(
    *,
    org_id: str,
    workspace_id: str,
    run_id: str,
    reserved_cost_usd,
    actual_cost_usd: float,
) -> bool:
    from .spine.guard import reconcile_budget_async

    return bool(
        await reconcile_budget_async(
            org_id=org_id,
            workspace_id=workspace_id,
            run_id=run_id,
            reserved_cost_usd=reserved_cost_usd,
            actual_cost_usd=actual_cost_usd,
        )
    )


async def _try_start_stream_reservation_async(
    *,
    org_id: str,
    workspace_id: str,
    run_id: str,
    reserved_cost_usd: Any,
) -> bool:
    del workspace_id
    from .deployment import finops_reservation_fail_closed
    from .spine.guard import signal_reservation_event_async

    return await signal_reservation_event_async(
        org_id=org_id,
        run_id=run_id,
        reserved_cost_usd=reserved_cost_usd,
        event="provider_started",
        required=finops_reservation_fail_closed(),
    )


async def _try_settle_stream_budget_async(
    *,
    org_id: str,
    workspace_id: str,
    run_id: str,
    reserved_cost_usd: Any,
    actual_cost_usd: float,
) -> bool:
    del workspace_id
    from .spine.guard import signal_reservation_event_async

    return await signal_reservation_event_async(
        org_id=org_id,
        run_id=run_id,
        reserved_cost_usd=reserved_cost_usd,
        event="usage_known",
        actual_cost_usd=actual_cost_usd,
    )


async def _try_mark_stream_pending_async(
    *,
    org_id: str,
    workspace_id: str,
    run_id: str,
    reserved_cost_usd: Any,
    actual_cost_usd: float | None = None,
) -> bool:
    """Async twin of ``_try_mark_stream_pending`` (exact-amount pending)."""
    del workspace_id
    from .spine.guard import signal_reservation_event_async

    if actual_cost_usd is not None:
        return await signal_reservation_event_async(
            org_id=org_id,
            run_id=run_id,
            reserved_cost_usd=reserved_cost_usd,
            event="usage_known",
            actual_cost_usd=float(actual_cost_usd),
        )
    return await signal_reservation_event_async(
        org_id=org_id,
        run_id=run_id,
        reserved_cost_usd=reserved_cost_usd,
        event="outcome_unknown",
    )


async def _mark_stream_started_or_release_async(
    *,
    enforcement: Enforcement,
    held_projection: float,
    org_id: str,
    workspace_id: str,
    run_id: str,
    reserved_cost_usd: Any,
) -> None:
    """Fail before provider dispatch if strict lifecycle state cannot be recorded."""
    try:
        await _try_start_stream_reservation_async(
            org_id=org_id,
            workspace_id=workspace_id,
            run_id=run_id,
            reserved_cost_usd=reserved_cost_usd,
        )
    except Exception:
        enforcement.release_projection(held_projection)
        raise


def _try_reconcile_budget(
    *,
    org_id: str,
    workspace_id: str,
    run_id: str,
    reserved_cost_usd,
    actual_cost_usd: float,
) -> bool:
    """Reconcile/settle via canonical spine; False when FinOps does not ack."""
    from .spine.guard import reconcile_budget

    return bool(
        reconcile_budget(
            org_id=org_id,
            workspace_id=workspace_id,
            run_id=run_id,
            reserved_cost_usd=reserved_cost_usd,
            actual_cost_usd=actual_cost_usd,
        )
    )


def _try_start_stream_reservation(
    *,
    org_id: str,
    workspace_id: str,
    run_id: str,
    reserved_cost_usd: Any,
) -> bool:
    """Mark a lifecycle reservation in-flight at the provider dispatch boundary."""
    del workspace_id
    from .deployment import finops_reservation_fail_closed
    from .spine.guard import signal_reservation_event

    return signal_reservation_event(
        org_id=org_id,
        run_id=run_id,
        reserved_cost_usd=reserved_cost_usd,
        event="provider_started",
        required=finops_reservation_fail_closed(),
    )


def _try_settle_stream_budget(
    *,
    org_id: str,
    workspace_id: str,
    run_id: str,
    reserved_cost_usd: Any,
    actual_cost_usd: float,
) -> bool:
    """Settle exact terminal stream usage after provider_started was recorded."""
    del workspace_id
    from .spine.guard import signal_reservation_event

    return signal_reservation_event(
        org_id=org_id,
        run_id=run_id,
        reserved_cost_usd=reserved_cost_usd,
        event="usage_known",
        actual_cost_usd=actual_cost_usd,
    )


def _try_mark_stream_pending(
    *,
    org_id: str,
    workspace_id: str,
    run_id: str,
    reserved_cost_usd: Any,
    actual_cost_usd: float | None = None,
) -> bool:
    """Keep shared exposure pending when settlement is not financially acknowledged.

    When ``actual_cost_usd`` is known, re-signal ``usage_known`` so the exact amount
    remains the lifecycle intent. Only fall back to ``outcome_unknown`` when
    usage/pricing is unavailable.
    """
    del workspace_id
    from .spine.guard import signal_reservation_event

    if actual_cost_usd is not None:
        return signal_reservation_event(
            org_id=org_id,
            run_id=run_id,
            reserved_cost_usd=reserved_cost_usd,
            event="usage_known",
            actual_cost_usd=float(actual_cost_usd),
        )
    return signal_reservation_event(
        org_id=org_id,
        run_id=run_id,
        reserved_cost_usd=reserved_cost_usd,
        event="outcome_unknown",
    )


def _mark_stream_started_or_release(
    *,
    enforcement: Enforcement,
    held_projection: float,
    org_id: str,
    workspace_id: str,
    run_id: str,
    reserved_cost_usd: Any,
) -> None:
    """Fail before provider dispatch if strict lifecycle state cannot be recorded."""
    try:
        _try_start_stream_reservation(
            org_id=org_id,
            workspace_id=workspace_id,
            run_id=run_id,
            reserved_cost_usd=reserved_cost_usd,
        )
    except Exception:
        enforcement.release_projection(held_projection)
        raise


def _extract_openai_input(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    # Best-effort extraction; stays fail-open.
    if "messages" in kwargs:
        return {"messages": kwargs.get("messages"), "model": kwargs.get("model")}
    if "input" in kwargs:
        return {
            "input": kwargs.get("input"),
            "instructions": kwargs.get("instructions"),
            "model": kwargs.get("model"),
        }
    if args:
        return {"args": args, "model": kwargs.get("model")}
    return {"model": kwargs.get("model")}


def _extract_tool_names(kwargs: Mapping[str, Any]) -> Optional[List[str]]:
    tools = kwargs.get("tools")
    if not tools:
        return None
    out: list[str] = []
    try:
        for t in tools:
            if isinstance(t, dict):
                fn = t.get("function") or {}
                # Chat nests the name under ``function``; Responses tools are flat.
                name = (fn.get("name") if isinstance(fn, dict) else None) or t.get("name")
                if name:
                    out.append(str(name))
            else:
                out.append(str(t))
    except Exception as exc:
        _log_fail_open("extract_tool_names", exc)
        return None
    return out or None


def _extract_input_text(payload: Any) -> str:
    # Loop heuristic needs a single string. Keep it cheap.
    try:
        if isinstance(payload, dict) and "messages" in payload and isinstance(payload["messages"], list):
            parts: list[str] = []
            for m in payload["messages"]:
                if isinstance(m, dict):
                    c = m.get("content")
                    if isinstance(c, str):
                        parts.append(c)
            return "\n".join(parts)
        if isinstance(payload, dict) and "input" in payload:
            return _responses_input_text(payload.get("instructions"), payload["input"])
        return str(payload)
    except Exception as exc:
        _log_fail_open("extract_input_text", exc)
        return ""


def _responses_input_text(instructions: Any, items: Any) -> str:
    """Flatten Responses ``instructions`` + ``input`` (string or item list) to text."""
    parts: list[str] = []
    if isinstance(instructions, str) and instructions:
        parts.append(instructions)
    if isinstance(items, str):
        parts.append(items)
    elif isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for piece in content:
                    if isinstance(piece, dict) and isinstance(piece.get("text"), str):
                        parts.append(piece["text"])
            for key in ("output", "arguments"):
                if isinstance(item.get(key), str):
                    parts.append(item[key])
    elif items is not None:
        parts.append(str(items))
    return "\n".join(parts)


def _unwrap_raw_response(resp: Any) -> Any:
    """Return the parsed body of an SDK raw-response wrapper, else ``resp``.

    ``client.chat.completions.with_raw_response.create(...)`` (used by
    langchain-openai for every non-streaming call) goes through the monitored
    ``create`` but returns ``LegacyAPIResponse``, which has no ``.usage``.
    ``parse()`` is cached by the SDK, so the caller's own ``parse()`` gets the
    same object and no second network read happens.
    """
    if resp is None or getattr(resp, "usage", None) is not None:
        return resp
    parse = getattr(resp, "parse", None)
    if callable(parse) and hasattr(resp, "http_response"):
        try:
            return parse()
        except Exception as exc:
            _log_fail_open("unwrap_raw_response", exc)
    return resp


_DEFAULT_ORG_ID = "local"
_DEFAULT_WORKSPACE_ID = "default"


def _call_context(client_ctx: Any) -> Any:
    """Run/step lineage from the ambient context; tenant from the client when the ambient one is a default.

    Framework adapters (``wrap_graph`` / ``wrap_langchain_runnable``) install a
    RunContext whose org/workspace default to ``local`` / ``default``. Before
    this, that default replaced the tenant the monitored client was configured
    with, so a service credential reserved under org ``local``: wrong tenant,
    no limits, fail-closed denial. An explicitly set ambient tenant still wins.
    """
    ambient = get_current_context()
    if ambient is None:
        return client_ctx
    if ambient.org_id == _DEFAULT_ORG_ID and client_ctx.org_id not in ("", _DEFAULT_ORG_ID):
        workspace = ambient.workspace_id
        if workspace == _DEFAULT_WORKSPACE_ID:
            workspace = client_ctx.workspace_id
        return dataclasses.replace(ambient, org_id=client_ctx.org_id, workspace_id=workspace)
    return ambient


class _RawStreamResponse:
    """Raw-response wrapper whose ``parse()`` yields the metered stream.

    ``with_raw_response.create(stream=True)`` (langchain-anthropic uses it for
    every stream) returns a raw wrapper; the caller gets the event stream from
    ``parse()``. Wrapping the raw object itself would hand back the unmetered
    stream there, so its cost would never be recorded or settled.
    """

    def __init__(self, raw: Any, wrap: Callable[[Any], Any]) -> None:
        self._kazenai_raw = raw
        self._kazenai_wrap = wrap
        self._kazenai_parsed: Any = None

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        if self._kazenai_parsed is None:
            self._kazenai_parsed = self._kazenai_wrap(self._kazenai_raw.parse(*args, **kwargs))
        return self._kazenai_parsed

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kazenai_raw, name)


def _wrap_stream_result(resp: Any, attempt: Any, wrap: Callable[[Any, Any], Any]) -> Any:
    """Meter a streamed result, including one returned inside a raw-response wrapper."""
    if callable(getattr(resp, "parse", None)) and hasattr(resp, "http_response"):
        return _RawStreamResponse(resp, lambda stream: wrap(stream, attempt))
    return wrap(resp, attempt)


def _extract_usage(resp: Any) -> Tuple[Optional[int], Any]:
    try:
        resp = _unwrap_raw_response(resp)
        usage = getattr(resp, "usage", None) or (resp.get("usage") if isinstance(resp, dict) else None)
        if not usage:
            return None, None
        total = getattr(usage, "total_tokens", None)
        if total is None and isinstance(usage, dict):
            total = usage.get("total_tokens")
        return (int(total) if total is not None else None), usage
    except Exception as exc:
        _log_fail_open("extract_usage", exc)
        return None, None


async def _post_event(
    *,
    endpoint_url: str,
    event: KazenEvent,
    headers: Optional[Mapping[str, str]] = None,
    timeout_s: float = 2.0,
) -> None:
    if aiohttp is None:
        return
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s)) as session:
        async with session.post(endpoint_url, json=event.model_dump(), headers=dict(headers or {})) as r:
            await r.release()



def _safe_output(resp: Any) -> Any:
    try:
        if isinstance(resp, (dict, list, str, int, float, bool)) or resp is None:
            return resp
        model_dump = getattr(resp, "model_dump", None)
        if callable(model_dump):
            return model_dump()
        return str(resp)
    except Exception:
        return "<unserializable>"


def _wrap_stream_iterator(
    stream: Any,
    *,
    step_ctx: RunContext,
    model: str,
    stream_enforcement: bool,
    usd_per_1k_tokens: float,
    stream_cutoff_usd: Optional[float] = None,
) -> Any:
    """Legacy OpenAI-style wrapper.

    Control-certified streaming uses ``kazenai.streaming`` (finalize-once lifecycle).
    This helper remains for non-Control ``stream_enforcement`` compatibility tests and
    maps to *local* cutoff only — it does **not** call ``/v1/budget/stream-tick``.
    """
    if not stream_enforcement and stream_cutoff_usd is None:
        return stream

    from .streaming.lifecycle import StreamAttempt
    from .streaming.openai import wrap_openai_stream

    cutoff = stream_cutoff_usd
    if cutoff is None and stream_enforcement and usd_per_1k_tokens:
        # Legacy flag: approximate a local cutoff after modest estimated spend.
        cutoff = max(0.01, float(usd_per_1k_tokens) * 0.05)

    attempt = StreamAttempt(
        surface=SUPPORTED_OPENAI_STREAM_METHOD,
        step_ctx=step_ctx,
        enforcement=Enforcement(),  # local-only; caller already reserved via main path when used there
        held_projection=0.0,
        reserved_cost_usd=None,
        model=model or "",
        method=SUPPORTED_OPENAI_STREAM_METHOD,
        usd_per_1k_tokens=float(usd_per_1k_tokens or 0.0),
        capture_mode=CaptureMode.METADATA,
        agent_role="agent",
        stream_cutoff_usd=cutoff,
    )
    return wrap_openai_stream(stream, attempt)


def _emit_stream_model_call(
    attempt: Any,
    *,
    settle_cost: float,
    tokens_used: Optional[int],
    error: Optional[BaseException],
    org_id: str,
    project_id: str,
    workspace_id: str,
    agent_id: str,
    agent_role: str,
    capture_mode: CaptureMode,
    event_sink: Optional[EventSink],
    finops: Optional[FinOpsController],
    endpoint_url: Optional[str],
    endpoint_headers: Optional[Mapping[str, str]],
    log: Any,
) -> None:
    step_ctx = attempt.step_ctx
    elapsed_ms = (time.perf_counter() - attempt.started_perf) * 1000.0
    first_byte_ms = None
    if attempt.first_byte_perf is not None:
        first_byte_ms = (attempt.first_byte_perf - attempt.started_perf) * 1000.0
    usage_dict = attempt.observed_usage.as_openai_style()
    def _bounded_identifier(value: Any) -> Optional[str]:
        if value is None:
            return None
        raw = str(value)[:128]
        safe = "".join(ch for ch in raw if ch.isalnum() or ch in "-_.:/")
        return safe or None

    error_meta = None
    if error is not None:
        status = getattr(error, "status_code", None)
        try:
            status = int(status) if status is not None else None
        except (TypeError, ValueError):
            status = None
        error_meta = {
            "type": type(error).__name__,
            "code": _bounded_identifier(
                getattr(error, "error_code", None) or getattr(error, "code", None)
            ),
            "status_code": status,
            "request_id": _bounded_identifier(getattr(error, "request_id", None)),
        }
        error_meta = {key: value for key, value in error_meta.items() if value is not None}

    try:
        event = KazenEvent(
            schema_version="1.2",
            ts_ms=now_ms(),
            event_id=new_id(),
            org_id=org_id,
            project_id=project_id,
            workspace_id=workspace_id,
            surface="kazenai-core",
            agent_id=agent_id,
            agent_role=agent_role,
            run_id=step_ctx.run_id,
            root_run_id=step_ctx.run_id,
            step_id=step_ctx.step_id,
            parent_step_id=step_ctx.parent_step_id,
            event_type="model.call",
            tokens_used=tokens_used,
            cost_usd=(
                settle_cost
                if str(attempt.cost_confidence).startswith("exact")
                else None
            ),
            payload={
                **_reservation_payload_fields(attempt.reserved_cost_usd),
                **_attribution_payload_fields(getattr(attempt, "attribution", None)),
                "method": attempt.method,
                "model": attempt.model,
                "stream": True,
                "stream_surface": attempt.surface,
                "terminal_outcome": (
                    attempt.terminal_outcome.value if attempt.terminal_outcome is not None else None
                ),
                "financial_pending": attempt.financial_pending,
                "cost_confidence": attempt.cost_confidence,
                "upstream_close_attempted": attempt.upstream_close_attempted,
                "upstream_close_ok": attempt.upstream_close_ok,
                "elapsed_ms": elapsed_ms,
                "first_byte_ms": first_byte_ms,
                "usage": usage_dict,
                "pricing_version": attempt.pricing_version,
                "capture": capture_disclosure(capture_mode),
                "error": error_meta,
            },
        )
    except Exception:
        log.exception("kazenai.stream_event_build_failed")
        return

    if event_sink is not None:
        try:
            event_sink.emit(event)
        except Exception as exc:
            _log_fail_open("event_sink_emit", exc)
    if finops is not None and not attempt.financial_pending:
        try:
            derived = finops.handle_llm_call(event)
        except Exception as exc:
            if isinstance(exc, _EXPECTED_FAIL_OPEN):
                raise
            _log_fail_open("finops_stream_handle", exc)
        else:
            for key, ev in derived.items():
                if key.startswith("_"):
                    continue
                try:
                    if event_sink is not None:
                        event_sink.emit(ev)
                except Exception as exc:
                    _log_fail_open("event_sink_emit_derived", exc)
            # This is a deliberate product-control decision, not a telemetry
            # failure. Keep it outside the fail-open handler above.
            FinOpsController.raise_if_blocked(derived)
    if endpoint_url:
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None and not loop.is_closed():
                loop.create_task(
                    _post_event(
                        endpoint_url=endpoint_url,
                        event=event,
                        headers=endpoint_headers,
                    )
                )
            else:
                def _post_in_thread() -> None:
                    try:
                        asyncio.run(
                            _post_event(
                                endpoint_url=endpoint_url,
                                event=event,
                                headers=endpoint_headers,
                            )
                        )
                    except Exception as exc:
                        _log_fail_open("stream_endpoint_emit", exc)

                threading.Thread(target=_post_in_thread, daemon=True).start()
        except Exception as exc:
            _log_fail_open("stream_endpoint_dispatch", exc)
    _log_event_safe(log, event)


def patch_openai(
    client: Any,
    *,
    org_id: str,
    project_id: str,
    workspace_id: str = "default",
    agent_id: str,
    agent_role: str = "agent",
    enforcement: Optional[Enforcement] = None,
    loop_detector: Optional[LoopDetector] = None,
    loop_block_threshold: float = 0.85,
    usd_per_1k_tokens: float = 0.0,
    endpoint_url: Optional[str] = None,
    endpoint_headers: Optional[Mapping[str, str]] = None,
    debug: bool = False,
    event_sink: Optional[EventSink] = None,
    finops: Optional[FinOpsController] = None,
    stream_enforcement: bool = False,
    stream_cutoff_usd: Optional[float] = None,
    certified_surface: bool = False,
    capture_mode: CaptureMode = CaptureMode.METADATA,
    business_subject_ref: Optional[str] = None,
    feature_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    feature: Optional[str] = None,
    operation_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
    allow_unmetered: Sequence[str] = (),
    observe_unbounded: Sequence[str] = (),
    require_customer_id: bool = False,
) -> Callable[[], None]:
    """
    Monkey-patch an OpenAI-style client to enforce budgets/loops *before* each LLM call.

    Guards Chat Completions (``create`` / ``parse`` / ``stream``) and Responses
    (``create`` / ``parse`` / ``stream``). Unmetered spend methods (Realtime, legacy
    Completions, Responses ``compact`` / ``connect``) are stubbed to raise
    ``UnsupportedModeError``.

    Returns an `undo()` function that restores the original methods.
    """

    if structlog is not None:
        log = structlog.get_logger(__name__)
    else:
        log = logging.getLogger(__name__)
    enforcement = enforcement or Enforcement()
    loop_detector = loop_detector or LoopDetector()
    dbg = DebugPrinter(enabled=debug)

    from .attribution import assert_customer_id, use_attribution
    from .enforcement_owner import claim_configured_enforcement_owner as claim_enforcement_owner

    attr_defaults = dict(
        business_subject_ref=business_subject_ref,
        feature_id=feature_id or feature,
        workflow_id=workflow_id,
        operation_id=operation_id,
        attempt_id=attempt_id,
    )

    ctx0 = get_current_context()
    if ctx0 is None:
        ctx0 = RunContext.new(
            org_id=org_id,
            project_id=project_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
        )

    patches: List[Tuple[Any, str, Any]] = []
    # Backstop for references taken before monitor(): see _dispatch_gate.
    gate = install_dispatch_gate(client, patches, allow_unmetered=allow_unmetered)
    gate_scope = gate.authorized if gate is not None else nullcontext

    @contextmanager
    def _helper_scope():
        # A stream helper dispatches through the SDK during __enter__, after it reserved.
        with _helper_inner_create_bypass(), gate_scope():
            yield

    def _wrap(fn: Callable[..., Any], *, method: str) -> Callable[..., Any]:
        is_responses = method.startswith("openai.responses")
        stream_surface = (
            SUPPORTED_OPENAI_RESPONSES_STREAM_METHOD if is_responses else SUPPORTED_OPENAI_STREAM_METHOD
        )

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _HELPER_INNER_CREATE_BYPASS.get():
                return fn(*args, **kwargs)
            if is_responses:
                # Requests whose cost cannot be bounded before dispatch fail here.
                _validate_responses_request(kwargs)
            ctx = _call_context(ctx0)
            step_ctx = ctx.child_step()
            with ExitStack() as stack:
                stack.enter_context(use_context(step_ctx))
                stack.enter_context(claim_enforcement_owner())
                stack.enter_context(use_attribution(**attr_defaults))
                assert_customer_id(require_customer_id)
                payload = _extract_openai_input(args, kwargs)
                tool_names = _extract_tool_names(kwargs)

                loop_scores: Optional[LoopScores] = None
                try:
                    loop_scores = loop_detector.score(
                        input_text=_extract_input_text(payload),
                        tool_names=tool_names,
                    )
                    if loop_scores.combined >= float(loop_block_threshold) and loop_detector.is_loop(loop_scores):
                        raise LoopDetected(
                            f"loop detected: score={loop_scores.combined:.3f} h1={loop_scores.h1_jaccard:.3f} h2={loop_scores.h2_tools:.3f}"
                        )
                except LoopDetected:
                    raise
                except Exception:
                    log.exception("kazenai.loop_detector_failed")
                    loop_scores = None

                snap = enforcement.snapshot()
                dbg.pre_call(
                    step_id=step_ctx.step_id,
                    parent_step_id=step_ctx.parent_step_id,
                    projected_cost_usd=0.0,
                    cumulative_cost_usd=snap.get("cumulative_cost_usd"),
                    calls_in_window=snap.get("calls_in_window"),
                    loop_scores=loop_scores,
                )

                # Reject streaming before any network/provider work.
                _reject_streaming_if_unsupported(kwargs)
                # Enforceable output token bound before dispatch.
                _apply_output_token_bound(kwargs, responses=is_responses)

                call_kwargs = dict(kwargs)
                is_stream = bool(call_kwargs.get("stream"))
                if is_stream:
                    from .streaming.lifecycle import StreamAttempt, StreamOutcome
                    from .streaming.openai import merge_stream_options, wrap_openai_stream
                    from .streaming.openai_responses import wrap_responses_stream

                    # Responses report usage on the terminal event; they reject
                    # stream_options.include_usage.
                    if not is_responses:
                        call_kwargs = merge_stream_options(call_kwargs)

                model_name = str(call_kwargs.get("model") or "")
                cutoff = stream_cutoff_usd
                output_rate = 0.0
                if is_stream:
                    output_rate = _stream_output_rate_usd_per_1k(
                        model_name,
                        usd_per_1k_tokens,
                    )
                    if cutoff is None and stream_enforcement:
                        import warnings

                        warnings.warn(
                            "stream_enforcement is deprecated; use stream_cutoff_usd for local "
                            "observable-output cutoff. Control streaming finalizes at stream end.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
                        if output_rate > 0:
                            cutoff = max(0.01, output_rate * 0.05)
                    if cutoff is not None and output_rate <= 0:
                        raise UnsupportedModeError(
                            "stream_cutoff_usd requires known model pricing or an explicit "
                            "usd_per_1k_tokens rate",
                            error_code="stream_cutoff_pricing_unavailable",
                        )

                # Local cap first (fail closed), then shared FinOps reserve.
                _arm_price_scope(
                    provider="openai",
                    surface="responses" if is_responses else "chat",
                    client=client,
                    org_id=step_ctx.org_id,
                    observe_unbounded=observe_unbounded,
                    kwargs=call_kwargs,
                )
                projected = _precall_projection_usd(call_kwargs, usd_per_1k_tokens)
                held_projection, reserved_cost_usd = _reserve_with_local_hold(
                    enforcement=enforcement,
                    projected_cost_usd=projected,
                    reserve_kwargs={
                        "org_id": step_ctx.org_id,
                        "workspace_id": step_ctx.workspace_id,
                        "run_id": step_ctx.run_id,
                        "call_id": str(getattr(step_ctx, "step_id", None) or "") or None,
                        "estimated_cost_usd": projected,
                        "business_subject_ref": attr_defaults.get("business_subject_ref"),
                        "feature_id": attr_defaults.get("feature_id"),
                        "workflow_id": attr_defaults.get("workflow_id"),
                        "operation_id": attr_defaults.get("operation_id"),
                        "attempt_id": attr_defaults.get("attempt_id"),
                        "model": model_name,
                        "input_tokens": _estimate_input_tokens(call_kwargs),
                    },
                )
                # Durable provider_started before every potential dispatch (stream + non-stream).
                _mark_stream_started_or_release(
                    enforcement=enforcement,
                    held_projection=held_projection,
                    org_id=step_ctx.org_id,
                    workspace_id=step_ctx.workspace_id,
                    run_id=step_ctx.run_id,
                    reserved_cost_usd=reserved_cost_usd,
                )

                started = time.perf_counter()
                attempt = None
                if is_stream:
                    def _emit(attempt, settle_cost, tokens_used, error):
                        _emit_stream_model_call(
                            attempt,
                            settle_cost=settle_cost,
                            tokens_used=tokens_used,
                            error=error,
                            org_id=step_ctx.org_id,
                            project_id=step_ctx.project_id,
                            workspace_id=step_ctx.workspace_id,
                            agent_id=step_ctx.agent_id,
                            agent_role=agent_role,
                            capture_mode=capture_mode,
                            event_sink=event_sink,
                            finops=finops,
                            endpoint_url=endpoint_url,
                            endpoint_headers=endpoint_headers,
                            log=log,
                        )

                    attempt = StreamAttempt(
                        surface=stream_surface,
                        step_ctx=step_ctx,
                        enforcement=enforcement,
                        held_projection=held_projection,
                        reserved_cost_usd=reserved_cost_usd,
                        model=model_name,
                        method=stream_surface,
                        usd_per_1k_tokens=output_rate,
                        settle_usd_per_1k_tokens=usd_per_1k_tokens,
                        capture_mode=capture_mode,
                        agent_role=agent_role,
                        event_sink=event_sink,
                        finops=finops,
                        stream_cutoff_usd=cutoff,
                        reconcile_fn=_try_settle_stream_budget,
                        pending_fn=_try_mark_stream_pending,
                        postcall_cost_fn=_bound_postcall(),
                        reservation_payload_fn=_reservation_payload_fields,
                        emit_event_fn=_emit,
                        attribution=attr_defaults,
                        started_perf=started,
                    )
                try:
                    # Duration span around the real provider attempt (no-op when OTel off).
                    from .otel import build_control_attributes, control_span

                    if attempt is not None:
                        rid = getattr(reserved_cost_usd, "reservation_id", None)
                        did = getattr(reserved_cost_usd, "decision_id", None)
                        attempt.begin_otel_span(
                            provider="openai",
                            operation_id=attr_defaults.get("operation_id"),
                            attempt_id=attr_defaults.get("attempt_id"),
                            workflow_id=attr_defaults.get("workflow_id"),
                            feature_id=attr_defaults.get("feature_id"),
                            business_subject_ref=attr_defaults.get("business_subject_ref"),
                            reservation_id=rid,
                            decision_id=did,
                        )
                    with control_span(
                        "kazenai.provider.dispatch",
                        kind="client",
                        attributes=build_control_attributes(
                            model=model_name,
                            provider="openai",
                            operation_id=attr_defaults.get("operation_id"),
                            attempt_id=attr_defaults.get("attempt_id"),
                            workflow_id=attr_defaults.get("workflow_id"),
                            feature_id=attr_defaults.get("feature_id"),
                            business_subject_ref=attr_defaults.get("business_subject_ref"),
                            reservation_id=getattr(reserved_cost_usd, "reservation_id", None),
                            decision_id=getattr(reserved_cost_usd, "decision_id", None),
                        ),
                    ), gate_scope():
                        resp = fn(*args, **call_kwargs)
                except Exception as exc:
                    rejected = _definitive_provider_rejection(exc)
                    if attempt is not None:
                        attempt.finalize(
                            StreamOutcome.REJECTED if rejected else StreamOutcome.PROVIDER_ERROR,
                            error=exc,
                        )
                    else:
                        # Dispatched (or dispatch attempted): keep unresolved exposure, never invent zero.
                        # A definitive 4xx refusal settles at exactly $0 and releases the hold.
                        if reserved_cost_usd is not None:
                            _try_mark_stream_pending(
                                org_id=step_ctx.org_id,
                                workspace_id=step_ctx.workspace_id,
                                run_id=step_ctx.run_id,
                                reserved_cost_usd=reserved_cost_usd,
                                actual_cost_usd=0.0 if rejected else None,
                            )
                        enforcement.record_call(
                            cost_usd=0.0,
                            released_projection=held_projection,
                        )
                    raise

                if is_stream:
                    assert attempt is not None
                    return _wrap_stream_result(
                        resp, attempt, wrap_responses_stream if is_responses else wrap_openai_stream
                    )

                elapsed_ms = (time.perf_counter() - started) * 1000.0

                tokens_used, usage = _extract_usage(resp)
                cost_usd = _postcall_cost_usd(
                    model=str(kwargs.get("model") or ""),
                    usage=usage,
                    tokens_used=tokens_used,
                    usd_per_1k_tokens=usd_per_1k_tokens,
                )

                # Missing usage/pricing → pending; false settle ack → pending
                # with exact known amount (never discard financial delivery failure).
                if reserved_cost_usd is not None:
                    if cost_usd is None or _pricing_holds_reconciliation():
                        _try_mark_stream_pending(
                            org_id=step_ctx.org_id,
                            workspace_id=step_ctx.workspace_id,
                            run_id=step_ctx.run_id,
                            reserved_cost_usd=reserved_cost_usd,
                        )
                    else:
                        settled = _try_reconcile_budget(
                            org_id=step_ctx.org_id,
                            workspace_id=step_ctx.workspace_id,
                            run_id=step_ctx.run_id,
                            reserved_cost_usd=reserved_cost_usd,
                            actual_cost_usd=float(cost_usd),
                        )
                        if settled is not True:
                            _try_mark_stream_pending(
                                org_id=step_ctx.org_id,
                                workspace_id=step_ctx.workspace_id,
                                run_id=step_ctx.run_id,
                                reserved_cost_usd=reserved_cost_usd,
                                actual_cost_usd=float(cost_usd),
                            )

                enforcement.record_call(cost_usd=cost_usd, released_projection=held_projection)
                snap2 = enforcement.snapshot()

                dbg.post_call(
                    step_id=step_ctx.step_id,
                    step_cost_usd=cost_usd,
                    tokens_used=tokens_used,
                    cumulative_cost_usd=snap2.get("cumulative_cost_usd"),
                    loop_scores=loop_scores,
                )

                try:
                    event = KazenEvent(
                        schema_version="1.2",
                        ts_ms=now_ms(),
                        event_id=new_id(),
                        org_id=step_ctx.org_id,
                        project_id=step_ctx.project_id,
                        workspace_id=step_ctx.workspace_id,
                        surface="kazenai-core",
                        agent_id=step_ctx.agent_id,
                        agent_role=agent_role,
                        run_id=step_ctx.run_id,
                        root_run_id=step_ctx.run_id,
                        step_id=step_ctx.step_id,
                        parent_step_id=step_ctx.parent_step_id,
                        event_type="model.call",
                        tokens_used=tokens_used,
                        cost_usd=cost_usd,
                        payload={
                            **_reservation_payload_fields(reserved_cost_usd),
                            **_attribution_payload_fields(),
                            "method": method,
                            "model": (kwargs.get("model") if isinstance(kwargs, dict) else None)
                            or (payload.get("model") if isinstance(payload, dict) else None),
                            "inputs_ref": build_content_ref(payload, mode=capture_mode, role="inputs"),
                            "outputs_ref": build_content_ref(_safe_output(resp), mode=capture_mode, role="outputs"),
                            "capture": capture_disclosure(capture_mode),
                            "elapsed_ms": elapsed_ms,
                            "usage": usage if isinstance(usage, dict) else None,
                            "loop_scores": (
                                {"h1_jaccard": loop_scores.h1_jaccard, "h2_tools": loop_scores.h2_tools, "combined": loop_scores.combined}
                                if loop_scores is not None else None
                            ),
                        },
                    )
                except Exception:
                    log.exception("kazenai.event_build_failed")
                    return resp

                # Emit to sinks first (fail-open).
                if event_sink is not None:
                    try:
                        event_sink.emit(event)
                    except Exception as exc:
                        _log_fail_open("event_sink_emit", exc)

                # Derived FinOps events (trajectory, anomaly, circuit breaker).
                if finops is not None:
                    derived = finops.handle_llm_call(event)
                    for key, ev in derived.items():
                        if key.startswith("_"):
                            continue
                        try:
                            if event_sink is not None:
                                event_sink.emit(ev)
                        except Exception as exc:
                            _log_fail_open("event_sink_emit_derived", exc)
                    FinOpsController.raise_if_blocked(derived)

                if endpoint_url:
                    try:
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:
                            loop = None
                        if loop and not loop.is_closed():
                            loop.create_task(
                                _post_event(
                                    endpoint_url=endpoint_url,
                                    event=event,
                                    headers=endpoint_headers,
                                )
                            )
                        else:
                            # Avoid blocking sync hot path without a running loop.
                            _log_event_safe(log, event)
                    except Exception:
                        log.exception("kazenai.event_dispatch_failed")
                else:
                    _log_event_safe(log, event)

                return resp

        return wrapped

    # Best-effort: patch common client surfaces (OpenAI python v1 style).
    targets: List[Tuple[Any, str, str]] = []
    try:
        if hasattr(client, "chat") and hasattr(client.chat, "completions") and hasattr(client.chat.completions, "create"):
            targets.append((client.chat.completions, "create", SUPPORTED_OPENAI_METHOD))
    except Exception:
        pass
    targets.extend(_openai_parse_targets(client))
    targets.extend(_openai_responses_targets(client))

    if not targets:
        raise TypeError(
            "Unsupported client: expected `.chat.completions.create` and/or `.responses.create`"
        )

    for obj, name, event_type in targets:
        orig = getattr(obj, name)
        if not callable(orig):
            continue
        patches.append((obj, name, orig))
        wrapped_fn = _wrap(orig, method=event_type)
        try:
            setattr(wrapped_fn, "__kazenai_wrapped__", True)
        except Exception:
            pass
        setattr(obj, name, wrapped_fn)

    # The official SDK exposes lazy stream managers on ``chat.completions.stream``
    # and ``responses.stream``. They reserve on ``__enter__``; the helper's inner
    # ``create(stream=True)`` runs under the bypass so it is not reserved twice.
    def _make_stream_manager(orig_stream: Callable[..., Any], *, responses: bool) -> Callable[..., Any]:
        surface = SUPPORTED_OPENAI_RESPONSES_STREAM_MANAGER if responses else SUPPORTED_OPENAI_STREAM_MANAGER

        def _stream_manager(*s_args: Any, **s_kwargs: Any) -> Any:
            from .streaming.lifecycle import StreamAttempt
            from .streaming.openai import merge_stream_options, wrap_openai_manager
            from .streaming.openai_responses import wrap_responses_manager

            _reject_streaming_if_unsupported({"stream": True})
            ctx = _call_context(ctx0)
            step_ctx = ctx.child_step()
            call_kwargs = dict(s_kwargs)
            if responses:
                _validate_responses_request(call_kwargs, stream_helper=True)
            _apply_output_token_bound(call_kwargs, responses=responses)
            if not responses:
                call_kwargs = merge_stream_options(call_kwargs)
            # The helper itself sets stream=True and does not accept a
            # public ``stream`` keyword.
            call_kwargs.pop("stream", None)
            model_name = str(call_kwargs.get("model") or "")
            output_rate = _stream_output_rate_usd_per_1k(
                model_name,
                usd_per_1k_tokens,
            )
            cutoff = stream_cutoff_usd
            if cutoff is None and stream_enforcement and output_rate > 0:
                cutoff = max(0.01, output_rate * 0.05)
            if cutoff is not None and output_rate <= 0:
                raise UnsupportedModeError(
                    "stream_cutoff_usd requires known model pricing or an explicit "
                    "usd_per_1k_tokens rate",
                    error_code="stream_cutoff_pricing_unavailable",
                )
            _arm_price_scope(
                provider="openai",
                surface="responses" if responses else "chat",
                client=client,
                org_id=step_ctx.org_id,
                observe_unbounded=observe_unbounded,
                kwargs=call_kwargs,
            )
            projected = _precall_projection_usd(call_kwargs, usd_per_1k_tokens)
            # Construction is lazy; reserve only when __enter__ is called.
            manager = orig_stream(*s_args, **call_kwargs)

            def _attempt_factory() -> StreamAttempt:
                with ExitStack() as stack:
                    stack.enter_context(use_context(step_ctx))
                    stack.enter_context(claim_enforcement_owner())
                    stack.enter_context(use_attribution(**attr_defaults))
                    assert_customer_id(require_customer_id)
                    held_projection, reserved_cost_usd = _reserve_with_local_hold(
                        enforcement=enforcement,
                        projected_cost_usd=projected,
                        reserve_kwargs={
                            "org_id": step_ctx.org_id,
                            "workspace_id": step_ctx.workspace_id,
                            "run_id": step_ctx.run_id,
                            "call_id": str(getattr(step_ctx, "step_id", None) or "") or None,
                            "estimated_cost_usd": projected,
                            "business_subject_ref": attr_defaults.get("business_subject_ref"),
                            "feature_id": attr_defaults.get("feature_id"),
                            "workflow_id": attr_defaults.get("workflow_id"),
                            "operation_id": attr_defaults.get("operation_id"),
                            "attempt_id": attr_defaults.get("attempt_id"),
                            "model": model_name,
                            "input_tokens": _estimate_input_tokens(call_kwargs),
                        },
                    )
                    _mark_stream_started_or_release(
                        enforcement=enforcement,
                        held_projection=held_projection,
                        org_id=step_ctx.org_id,
                        workspace_id=step_ctx.workspace_id,
                        run_id=step_ctx.run_id,
                        reserved_cost_usd=reserved_cost_usd,
                    )
                started = time.perf_counter()

                def _emit(attempt, settle_cost, tokens_used, error):
                    _emit_stream_model_call(
                        attempt,
                        settle_cost=settle_cost,
                        tokens_used=tokens_used,
                        error=error,
                        org_id=step_ctx.org_id,
                        project_id=step_ctx.project_id,
                        workspace_id=step_ctx.workspace_id,
                        agent_id=step_ctx.agent_id,
                        agent_role=agent_role,
                        capture_mode=capture_mode,
                        event_sink=event_sink,
                        finops=finops,
                        endpoint_url=endpoint_url,
                        endpoint_headers=endpoint_headers,
                        log=log,
                    )

                return StreamAttempt(
                    surface=surface,
                    step_ctx=step_ctx,
                    enforcement=enforcement,
                    held_projection=held_projection,
                    reserved_cost_usd=reserved_cost_usd,
                    model=model_name,
                    method=surface,
                    usd_per_1k_tokens=output_rate,
                    settle_usd_per_1k_tokens=usd_per_1k_tokens,
                    capture_mode=capture_mode,
                    agent_role=agent_role,
                    event_sink=event_sink,
                    finops=finops,
                    stream_cutoff_usd=cutoff,
                    reconcile_fn=_try_settle_stream_budget,
                    pending_fn=_try_mark_stream_pending,
                    postcall_cost_fn=_bound_postcall(),
                    reservation_payload_fn=_reservation_payload_fields,
                    emit_event_fn=_emit,
                    attribution=attr_defaults,
                    started_perf=started,
                )

            wrap_manager = wrap_responses_manager if responses else wrap_openai_manager
            return wrap_manager(
                manager,
                attempt_factory=_attempt_factory,
                enter_scope=_helper_scope,
            )

        return _stream_manager

    for resource, is_responses in (
        (getattr(getattr(client, "chat", None), "completions", None), False),
        (getattr(client, "responses", None), True),
    ):
        try:
            orig_stream = getattr(resource, "stream", None) if resource is not None else None
            if callable(orig_stream):
                patches.append((resource, "stream", orig_stream))
                setattr(resource, "stream", _make_stream_manager(orig_stream, responses=is_responses))
        except Exception:
            pass

    # No unmetered spend path may stay reachable on a monitored client.
    _block_unsupported_openai_surfaces(
        client, patches, keep={(id(obj), name) for obj, name, _ in patches}
    )
    _reset_sdk_response_wrappers(obj for obj, _, _ in patches)

    def undo() -> None:
        for obj, name, orig in patches:
            try:
                setattr(obj, name, orig)
            except Exception:
                log.exception("kazenai.patch_undo_failed", target=str(obj), name=name)
        _reset_sdk_response_wrappers(obj for obj, _, _ in patches)

    return undo


def _extract_anthropic_usage(resp: Any) -> Tuple[Optional[int], Any]:
    """Extract token counts from an Anthropic `messages.create` response."""
    try:
        resp = _unwrap_raw_response(resp)
        usage = getattr(resp, "usage", None)
        if not usage:
            return None, None
        input_t = getattr(usage, "input_tokens", None)
        output_t = getattr(usage, "output_tokens", None)
        if input_t is None and output_t is None:
            return None, usage
        total = (int(input_t or 0)) + (int(output_t or 0))
        return total, usage
    except Exception:
        return None, None


def patch_anthropic(
    client: Any,
    *,
    org_id: str,
    project_id: str,
    workspace_id: str = "default",
    agent_id: str,
    agent_role: str = "agent",
    enforcement: Optional[Enforcement] = None,
    loop_detector: Optional[LoopDetector] = None,
    loop_block_threshold: float = 0.85,
    usd_per_1k_tokens: float = 0.0,
    endpoint_url: Optional[str] = None,
    endpoint_headers: Optional[Mapping[str, str]] = None,
    debug: bool = False,
    event_sink: Optional[EventSink] = None,
    finops: Optional[FinOpsController] = None,
    stream_cutoff_usd: Optional[float] = None,
    certified_surface: bool = False,
    capture_mode: CaptureMode = CaptureMode.METADATA,
    business_subject_ref: Optional[str] = None,
    feature_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    feature: Optional[str] = None,
    operation_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
    allow_unmetered: Sequence[str] = (),
    observe_unbounded: Sequence[str] = (),
    require_customer_id: bool = False,
) -> Callable[[], None]:
    """
    Monkey-patch an Anthropic client to emit KazenEvents on every messages.create call.

    Mirrors patch_openai() but handles the Anthropic response shape
    (msg.usage.input_tokens / output_tokens instead of usage.total_tokens).

    Returns an `undo()` function that restores the original methods.
    """
    if structlog is not None:
        log = structlog.get_logger(__name__)
    else:
        log = logging.getLogger(__name__)
    enforcement = enforcement or Enforcement()
    loop_detector = loop_detector or LoopDetector()

    from .attribution import assert_customer_id, use_attribution
    from .enforcement_owner import claim_configured_enforcement_owner as claim_enforcement_owner

    attr_defaults = dict(
        business_subject_ref=business_subject_ref,
        feature_id=feature_id or feature,
        workflow_id=workflow_id,
        operation_id=operation_id,
        attempt_id=attempt_id,
    )

    ctx0 = get_current_context()
    if ctx0 is None:
        ctx0 = RunContext.new(
            org_id=org_id,
            project_id=project_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
        )

    patches: List[Tuple[Any, str, Any]] = []
    # Backstop for references taken before monitor(): see _dispatch_gate.
    gate = install_dispatch_gate(client, patches, allow_unmetered=allow_unmetered)
    gate_scope = gate.authorized if gate is not None else nullcontext

    @contextmanager
    def _helper_scope():
        # A stream helper dispatches through the SDK during __enter__, after it reserved.
        with _helper_inner_create_bypass(), gate_scope():
            yield

    def _wrap(fn: Callable[..., Any], *, method: str) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _HELPER_INNER_CREATE_BYPASS.get():
                return fn(*args, **kwargs)
            ctx = _call_context(ctx0)
            step_ctx = ctx.child_step()
            with ExitStack() as stack:
                stack.enter_context(use_context(step_ctx))
                stack.enter_context(claim_enforcement_owner())
                stack.enter_context(use_attribution(**attr_defaults))
                assert_customer_id(require_customer_id)
                payload = _extract_openai_input(args, kwargs)  # same shape works for Anthropic
                tool_names = _extract_tool_names(kwargs)

                loop_scores = None
                try:
                    loop_scores = loop_detector.score(
                        input_text=_extract_input_text(payload),
                        tool_names=tool_names,
                    )
                    if loop_scores.combined >= float(loop_block_threshold) and loop_detector.is_loop(loop_scores):
                        raise LoopDetected(
                            f"loop detected: score={loop_scores.combined:.3f}"
                        )
                except LoopDetected:
                    raise
                except Exception:
                    loop_scores = None

                # Reject streaming before any network/provider work.
                _reject_streaming_if_unsupported(kwargs)
                # Enforceable output token bound before dispatch.
                _apply_output_token_bound(kwargs)

                call_kwargs = dict(kwargs)
                is_stream = bool(call_kwargs.get("stream"))
                model_name = str(call_kwargs.get("model") or "")
                output_rate = 0.0
                if is_stream:
                    output_rate = _stream_output_rate_usd_per_1k(
                        model_name,
                        usd_per_1k_tokens,
                    )
                    if stream_cutoff_usd is not None and output_rate <= 0:
                        raise UnsupportedModeError(
                            "stream_cutoff_usd requires known model pricing or an explicit "
                            "usd_per_1k_tokens rate",
                            error_code="stream_cutoff_pricing_unavailable",
                        )

                # Local cap first (fail closed), then shared FinOps reserve.
                _arm_price_scope(
                    provider="anthropic",
                    surface="messages",
                    client=client,
                    org_id=step_ctx.org_id,
                    observe_unbounded=observe_unbounded,
                    kwargs=call_kwargs,
                )
                projected = _precall_projection_usd(call_kwargs, usd_per_1k_tokens)
                held_projection, reserved_cost_usd = _reserve_with_local_hold(
                    enforcement=enforcement,
                    projected_cost_usd=projected,
                    reserve_kwargs={
                        "org_id": step_ctx.org_id,
                        "workspace_id": step_ctx.workspace_id,
                        "run_id": step_ctx.run_id,
                        "call_id": str(getattr(step_ctx, "step_id", None) or "") or None,
                        "estimated_cost_usd": projected,
                        "business_subject_ref": attr_defaults.get("business_subject_ref"),
                        "feature_id": attr_defaults.get("feature_id"),
                        "workflow_id": attr_defaults.get("workflow_id"),
                        "operation_id": attr_defaults.get("operation_id"),
                        "attempt_id": attr_defaults.get("attempt_id"),
                        "model": model_name,
                        "input_tokens": _estimate_input_tokens(call_kwargs),
                    },
                )
                _mark_stream_started_or_release(
                    enforcement=enforcement,
                    held_projection=held_projection,
                    org_id=step_ctx.org_id,
                    workspace_id=step_ctx.workspace_id,
                    run_id=step_ctx.run_id,
                    reserved_cost_usd=reserved_cost_usd,
                )
                started = time.perf_counter()
                attempt = None
                if is_stream:
                    from .streaming.anthropic import wrap_anthropic_event_stream
                    from .streaming.lifecycle import StreamAttempt, StreamOutcome

                    def _emit(attempt, settle_cost, tokens_used, error):
                        _emit_stream_model_call(
                            attempt,
                            settle_cost=settle_cost,
                            tokens_used=tokens_used,
                            error=error,
                            org_id=step_ctx.org_id,
                            project_id=step_ctx.project_id,
                            workspace_id=step_ctx.workspace_id,
                            agent_id=step_ctx.agent_id,
                            agent_role=agent_role,
                            capture_mode=capture_mode,
                            event_sink=event_sink,
                            finops=finops,
                            endpoint_url=endpoint_url,
                            endpoint_headers=endpoint_headers,
                            log=log,
                        )

                    attempt = StreamAttempt(
                        surface=SUPPORTED_ANTHROPIC_STREAM_METHOD,
                        step_ctx=step_ctx,
                        enforcement=enforcement,
                        held_projection=held_projection,
                        reserved_cost_usd=reserved_cost_usd,
                        model=model_name,
                        method=SUPPORTED_ANTHROPIC_STREAM_METHOD,
                        usd_per_1k_tokens=output_rate,
                        settle_usd_per_1k_tokens=usd_per_1k_tokens,
                        capture_mode=capture_mode,
                        agent_role=agent_role,
                        event_sink=event_sink,
                        finops=finops,
                        stream_cutoff_usd=stream_cutoff_usd,
                        reconcile_fn=_try_settle_stream_budget,
                        pending_fn=_try_mark_stream_pending,
                        postcall_cost_fn=_bound_postcall(),
                        reservation_payload_fn=_reservation_payload_fields,
                        emit_event_fn=_emit,
                        attribution=attr_defaults,
                        started_perf=started,
                    )
                try:
                    from .otel import build_control_attributes, control_span

                    if attempt is not None:
                        attempt.begin_otel_span(
                            provider="anthropic",
                            operation_id=attr_defaults.get("operation_id"),
                            attempt_id=attr_defaults.get("attempt_id"),
                            workflow_id=attr_defaults.get("workflow_id"),
                            feature_id=attr_defaults.get("feature_id"),
                            business_subject_ref=attr_defaults.get("business_subject_ref"),
                            reservation_id=getattr(reserved_cost_usd, "reservation_id", None),
                            decision_id=getattr(reserved_cost_usd, "decision_id", None),
                        )
                    with control_span(
                        "kazenai.provider.dispatch",
                        kind="client",
                        attributes=build_control_attributes(
                            model=str(kwargs.get("model") or ""),
                            provider="anthropic",
                            operation_id=attr_defaults.get("operation_id"),
                            attempt_id=attr_defaults.get("attempt_id"),
                            workflow_id=attr_defaults.get("workflow_id"),
                            feature_id=attr_defaults.get("feature_id"),
                            business_subject_ref=attr_defaults.get("business_subject_ref"),
                            reservation_id=getattr(reserved_cost_usd, "reservation_id", None),
                            decision_id=getattr(reserved_cost_usd, "decision_id", None),
                        ),
                    ), gate_scope():
                        resp = fn(*args, **call_kwargs)
                except Exception as exc:
                    rejected = _definitive_provider_rejection(exc)
                    if attempt is not None:
                        attempt.finalize(
                            StreamOutcome.REJECTED if rejected else StreamOutcome.PROVIDER_ERROR,
                            error=exc,
                        )
                    else:
                        # A definitive 4xx refusal settles at exactly $0 and releases the hold.
                        if reserved_cost_usd is not None:
                            _try_mark_stream_pending(
                                org_id=step_ctx.org_id,
                                workspace_id=step_ctx.workspace_id,
                                run_id=step_ctx.run_id,
                                reserved_cost_usd=reserved_cost_usd,
                                actual_cost_usd=0.0 if rejected else None,
                            )
                        enforcement.record_call(
                            cost_usd=0.0,
                            released_projection=held_projection,
                        )
                    raise

                if is_stream:
                    assert attempt is not None
                    return _wrap_stream_result(resp, attempt, wrap_anthropic_event_stream)

                elapsed_ms = (time.perf_counter() - started) * 1000.0

                tokens_used, usage = _extract_anthropic_usage(resp)
                cost_usd = _postcall_cost_usd(
                    model=str(kwargs.get("model") or ""),
                    usage=usage,
                    tokens_used=tokens_used,
                    usd_per_1k_tokens=usd_per_1k_tokens,
                )

                if reserved_cost_usd is not None:
                    if cost_usd is None or _pricing_holds_reconciliation():
                        _try_mark_stream_pending(
                            org_id=step_ctx.org_id,
                            workspace_id=step_ctx.workspace_id,
                            run_id=step_ctx.run_id,
                            reserved_cost_usd=reserved_cost_usd,
                        )
                    else:
                        settled = _try_reconcile_budget(
                            org_id=step_ctx.org_id,
                            workspace_id=step_ctx.workspace_id,
                            run_id=step_ctx.run_id,
                            reserved_cost_usd=reserved_cost_usd,
                            actual_cost_usd=float(cost_usd),
                        )
                        if settled is not True:
                            _try_mark_stream_pending(
                                org_id=step_ctx.org_id,
                                workspace_id=step_ctx.workspace_id,
                                run_id=step_ctx.run_id,
                                reserved_cost_usd=reserved_cost_usd,
                                actual_cost_usd=float(cost_usd),
                            )

                enforcement.record_call(cost_usd=cost_usd, released_projection=held_projection)

                try:
                    event = KazenEvent(
                        schema_version="1.2",
                        ts_ms=now_ms(),
                        event_id=new_id(),
                        org_id=step_ctx.org_id,
                        project_id=step_ctx.project_id,
                        workspace_id=step_ctx.workspace_id,
                        surface="kazenai-core",
                        agent_id=step_ctx.agent_id,
                        agent_role=agent_role,
                        run_id=step_ctx.run_id,
                        root_run_id=step_ctx.run_id,
                        step_id=step_ctx.step_id,
                        parent_step_id=step_ctx.parent_step_id,
                        event_type="model.call",
                        tokens_used=tokens_used,
                        cost_usd=cost_usd,
                        payload={
                            **_reservation_payload_fields(reserved_cost_usd),
                            **_attribution_payload_fields(),
                            "method": method,
                            "model": kwargs.get("model") or (
                                payload.get("model") if isinstance(payload, dict) else None
                            ),
                            "inputs_ref": build_content_ref(payload, mode=capture_mode, role="inputs"),
                            "outputs_ref": build_content_ref(_safe_output(resp), mode=capture_mode, role="outputs"),
                            "capture": capture_disclosure(capture_mode),
                            "elapsed_ms": elapsed_ms,
                            "usage": (
                                {
                                    "input_tokens": getattr(usage, "input_tokens", None),
                                    "output_tokens": getattr(usage, "output_tokens", None),
                                }
                                if usage is not None else None
                            ),
                            "loop_scores": (
                                {"h1_jaccard": loop_scores.h1_jaccard, "h2_tools": loop_scores.h2_tools, "combined": loop_scores.combined}
                                if loop_scores is not None else None
                            ),
                        },
                    )
                except Exception:
                    return resp

                if event_sink is not None:
                    try:
                        event_sink.emit(event)
                    except Exception as exc:
                        _log_fail_open("event_sink_emit", exc)

                if finops is not None:
                    derived = finops.handle_llm_call(event)
                    for key, ev in derived.items():
                        if key.startswith("_"):
                            continue
                        try:
                            if event_sink is not None:
                                event_sink.emit(ev)
                        except Exception as exc:
                            _log_fail_open("event_sink_emit_derived", exc)
                    FinOpsController.raise_if_blocked(derived)

                if endpoint_url:
                    try:
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:
                            loop = None
                        if loop and not loop.is_closed():
                            loop.create_task(
                                _post_event(endpoint_url=endpoint_url, event=event, headers=endpoint_headers)
                            )
                        else:
                            _log_event_safe(log, event)
                    except Exception:
                        pass
                else:
                    _log_event_safe(log, event)

                return resp

        return wrapped

    # Patch client.messages.create (sync) and client.messages.stream if present.
    targets: List[Tuple[Any, str, str]] = []
    try:
        if hasattr(client, "messages") and hasattr(client.messages, "create"):
            targets.append((client.messages, "create", SUPPORTED_ANTHROPIC_METHOD))
    except Exception:
        pass

    if not targets:
        raise TypeError("Unsupported client: expected `.messages.create`")

    for obj, name, event_type in targets:
        orig = getattr(obj, name)
        if not callable(orig):
            continue
        patches.append((obj, name, orig))
        wrapped_fn = _wrap(orig, method=event_type)
        try:
            setattr(wrapped_fn, "__kazenai_wrapped__", True)
        except Exception:
            pass
        setattr(obj, name, wrapped_fn)

    # Wrap messages.stream manager with finalize-once lifecycle.
    try:
        if hasattr(client, "messages") and hasattr(client.messages, "stream"):
            orig_stream = client.messages.stream
            if callable(orig_stream):

                def _stream_manager(*s_args: Any, **s_kwargs: Any) -> Any:
                    from .streaming.lifecycle import StreamAttempt
                    from .streaming.anthropic import wrap_anthropic_manager

                    ctx = _call_context(ctx0)
                    step_ctx = ctx.child_step()
                    call_kwargs = dict(s_kwargs)
                    _apply_output_token_bound(call_kwargs)
                    model_name = str(call_kwargs.get("model") or "")
                    output_rate = _stream_output_rate_usd_per_1k(
                        model_name,
                        usd_per_1k_tokens,
                    )
                    if stream_cutoff_usd is not None and output_rate <= 0:
                        raise UnsupportedModeError(
                            "stream_cutoff_usd requires known model pricing or an explicit "
                            "usd_per_1k_tokens rate",
                            error_code="stream_cutoff_pricing_unavailable",
                        )
                    _arm_price_scope(
                        provider="anthropic",
                        surface="messages",
                        client=client,
                        org_id=step_ctx.org_id,
                        observe_unbounded=observe_unbounded,
                        kwargs=call_kwargs,
                    )
                    projected = _precall_projection_usd(call_kwargs, usd_per_1k_tokens)
                    # Anthropic's helper returns a lazy manager.  Constructing it
                    # is not provider dispatch, so do not reserve until __enter__.
                    manager = orig_stream(*s_args, **call_kwargs)

                    def _attempt_factory() -> StreamAttempt:
                        with ExitStack() as stack:
                            stack.enter_context(use_context(step_ctx))
                            stack.enter_context(claim_enforcement_owner())
                            stack.enter_context(use_attribution(**attr_defaults))
                            assert_customer_id(require_customer_id)
                            held_projection, reserved_cost_usd = _reserve_with_local_hold(
                                enforcement=enforcement,
                                projected_cost_usd=projected,
                                reserve_kwargs={
                                    "org_id": step_ctx.org_id,
                                    "workspace_id": step_ctx.workspace_id,
                                    "run_id": step_ctx.run_id,
                                    "call_id": str(getattr(step_ctx, "step_id", None) or "") or None,
                                    "estimated_cost_usd": projected,
                                    "business_subject_ref": attr_defaults.get("business_subject_ref"),
                                    "feature_id": attr_defaults.get("feature_id"),
                                    "workflow_id": attr_defaults.get("workflow_id"),
                                    "operation_id": attr_defaults.get("operation_id"),
                                    "attempt_id": attr_defaults.get("attempt_id"),
                                    "model": model_name,
                                    "input_tokens": _estimate_input_tokens(call_kwargs),
                                },
                            )
                            _mark_stream_started_or_release(
                                enforcement=enforcement,
                                held_projection=held_projection,
                                org_id=step_ctx.org_id,
                                workspace_id=step_ctx.workspace_id,
                                run_id=step_ctx.run_id,
                                reserved_cost_usd=reserved_cost_usd,
                            )
                        started = time.perf_counter()

                        def _emit(attempt, settle_cost, tokens_used, error):
                            _emit_stream_model_call(
                                attempt,
                                settle_cost=settle_cost,
                                tokens_used=tokens_used,
                                error=error,
                                org_id=step_ctx.org_id,
                                project_id=step_ctx.project_id,
                                workspace_id=step_ctx.workspace_id,
                                agent_id=step_ctx.agent_id,
                                agent_role=agent_role,
                                capture_mode=capture_mode,
                                event_sink=event_sink,
                                finops=finops,
                                endpoint_url=endpoint_url,
                                endpoint_headers=endpoint_headers,
                                log=log,
                            )

                        return StreamAttempt(
                            surface=SUPPORTED_ANTHROPIC_STREAM_MANAGER,
                            step_ctx=step_ctx,
                            enforcement=enforcement,
                            held_projection=held_projection,
                            reserved_cost_usd=reserved_cost_usd,
                            model=model_name,
                            method=SUPPORTED_ANTHROPIC_STREAM_MANAGER,
                            usd_per_1k_tokens=output_rate,
                            settle_usd_per_1k_tokens=usd_per_1k_tokens,
                            capture_mode=capture_mode,
                            agent_role=agent_role,
                            event_sink=event_sink,
                            finops=finops,
                            stream_cutoff_usd=stream_cutoff_usd,
                            reconcile_fn=_try_settle_stream_budget,
                            pending_fn=_try_mark_stream_pending,
                            postcall_cost_fn=_bound_postcall(),
                            reservation_payload_fn=_reservation_payload_fields,
                            emit_event_fn=_emit,
                            attribution=attr_defaults,
                            started_perf=started,
                        )

                    return wrap_anthropic_manager(
                        manager,
                        attempt_factory=_attempt_factory,
                        enter_scope=_helper_scope,
                    )

                patches.append((client.messages, "stream", orig_stream))
                setattr(client.messages, "stream", _stream_manager)
    except Exception:
        pass

    _reset_sdk_response_wrappers(obj for obj, _, _ in patches)

    def undo() -> None:
        for obj, name, orig in patches:
            try:
                setattr(obj, name, orig)
            except Exception:
                pass
        _reset_sdk_response_wrappers(obj for obj, _, _ in patches)

    return undo


SUPPORTED_OPENAI_PARSE_METHOD = "openai.chat.completions.parse"
_OPENAI_BETA_STREAM_UNSUPPORTED = (
    "OpenAI beta.chat.completions.stream (structured-output streaming) is not guarded by "
    "KazenAI Control. Use chat.completions.create(stream=True) or a non-streaming parse()."
)


def _openai_responses_targets(client: Any) -> List[Tuple[Any, str, str]]:
    """``responses.create`` and ``responses.parse``; ``parse`` posts directly, not via create."""
    resource = getattr(client, "responses", None)
    if resource is None:
        return []
    found: List[Tuple[Any, str, str]] = []
    if callable(getattr(resource, "create", None)):
        found.append((resource, "create", SUPPORTED_OPENAI_RESPONSES_METHOD))
    if callable(getattr(resource, "parse", None)):
        found.append((resource, "parse", SUPPORTED_OPENAI_RESPONSES_PARSE_METHOD))
    return found


def _openai_parse_targets(client: Any) -> List[Tuple[Any, str, str]]:
    """Structured-output ``parse`` endpoints, guarded exactly like ``create``.

    ``chat.completions.parse`` (and the ``beta.chat.completions`` alias) post to
    the provider directly rather than through ``create``; unpatched they skip
    the budget check and record no cost. LangChain ``with_structured_output`` and
    CrewAI structured output use them.
    """
    found: List[Tuple[Any, str, str]] = []
    owners = [getattr(getattr(client, "chat", None), "completions", None)]
    beta_chat = getattr(getattr(client, "beta", None), "chat", None)
    owners.append(getattr(beta_chat, "completions", None))
    seen: set[int] = set()
    for owner in owners:
        if owner is None or id(owner) in seen:
            continue
        seen.add(id(owner))
        if callable(getattr(owner, "parse", None)):
            found.append((owner, "parse", SUPPORTED_OPENAI_PARSE_METHOD))
    return found


_OPENAI_RESPONSES_UNSUPPORTED = (
    "This OpenAI Responses method (compact / WebSocket connect) is not guarded by "
    "KazenAI Control: the call would skip the pre-call budget check. Use "
    "responses.create, responses.parse or responses.stream."
)
_OPENAI_LEGACY_COMPLETIONS_UNSUPPORTED = (
    "Legacy OpenAI Completions (client.completions) is not guarded by KazenAI Control: "
    "the call would skip the pre-call budget check. Use chat.completions.create."
)
_OPENAI_REALTIME_UNSUPPORTED = (
    "OpenAI Realtime is not supported by KazenAI Control: session spend cannot be "
    "checked before dispatch."
)

# Responses tool types billed as model tokens. Built-in tools (web search, file
# search, code interpreter, image generation, computer use, MCP, shell) are billed
# per call or per session outside ``usage``, so their cost cannot be bounded first.
_RESPONSES_TOKEN_BILLED_TOOLS = frozenset({"function", "custom"})


def _validate_responses_request(kwargs: Mapping[str, Any], *, stream_helper: bool = False) -> None:
    """Refuse Responses requests whose spend cannot be bounded before dispatch.

    - ``background=True``: returns before usage exists; settlement needs polling.
    - ``previous_response_id`` / ``conversation`` / stored ``prompt``: the input
      lives on OpenAI's side, so the pre-call estimate would undercount it.
    - built-in tools: billed outside token usage.
    - ``responses.stream(response_id=...)``: resumes a background response.

    Checked on the request as sent: ``extra_body`` fields count like top-level ones.
    """
    from .request_pricing import _effective_request

    kwargs = _effective_request(kwargs)
    if kwargs.get("background"):
        raise UnsupportedModeError(
            "OpenAI Responses background mode is not supported by KazenAI Control: "
            "its cost cannot be settled when the call returns. Remove background=True.",
            error_code="responses_background_unsupported",
        )
    for key in ("previous_response_id", "conversation", "prompt"):
        if kwargs.get(key):
            raise UnsupportedModeError(
                f"OpenAI Responses `{key}` is not supported by KazenAI Control yet: the "
                "input stored on OpenAI's side cannot be priced before the call. Send the "
                "conversation as `input` items instead.",
                error_code="responses_server_state_unsupported",
            )
    if stream_helper and kwargs.get("response_id"):
        raise UnsupportedModeError(
            "Resuming a Responses stream by response_id is not supported by KazenAI Control.",
            error_code="responses_background_unsupported",
        )
    for tool in kwargs.get("tools") or ():
        tool_type = tool.get("type") if isinstance(tool, Mapping) else getattr(tool, "type", None)
        if tool_type not in _RESPONSES_TOKEN_BILLED_TOOLS:
            raise UnsupportedModeError(
                f"OpenAI Responses built-in tool {tool_type!r} is not supported by KazenAI "
                "Control: it is billed outside token usage, so its cost cannot be checked "
                "before the call. Use function tools.",
                error_code="responses_builtin_tool_unsupported",
            )


# OpenAI spend surfaces a monitored client does not meter. Each present method is
# replaced by a stub that raises UnsupportedModeError, so it fails loudly instead
# of spending outside the budget.
_OPENAI_UNSUPPORTED_SURFACES: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("responses", ("compact", "connect"), _OPENAI_RESPONSES_UNSUPPORTED),
    ("completions", ("create",), _OPENAI_LEGACY_COMPLETIONS_UNSUPPORTED),
    ("realtime", ("connect",), _OPENAI_REALTIME_UNSUPPORTED),
    ("beta.realtime", ("connect",), _OPENAI_REALTIME_UNSUPPORTED),
)

# SDK resource attributes built lazily (functools.cached_property) from the bound
# methods at first access. A copy cached before patching keeps the unpatched method.
_SDK_RESPONSE_WRAPPER_ATTRS = ("with_raw_response", "with_streaming_response")


def _resolve_path(root: Any, dotted: str) -> Any:
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _block_unsupported_openai_surfaces(
    client: Any,
    patches: List[Tuple[Any, str, Any]],
    *,
    keep: Optional[set] = None,
) -> None:
    """Stub every unmetered OpenAI spend method present on ``client``.

    ``keep`` holds ``(id(resource), method)`` pairs already guarded by a real
    wrapper; those are left alone. Originals go into ``patches`` for ``undo()``.
    """
    keep = keep or set()
    for path, names, message in _OPENAI_UNSUPPORTED_SURFACES:
        resource = _resolve_path(client, path)
        if resource is None:
            continue
        for name in names:
            if (id(resource), name) in keep or not callable(getattr(resource, name, None)):
                continue
            try:
                orig = getattr(resource, name)
                setattr(resource, name, _unsupported_mode_stub(message))
                patches.append((resource, name, orig))
            except Exception:
                pass


def _reset_sdk_response_wrappers(resources: Any) -> None:
    """Drop cached ``with_raw_response`` / ``with_streaming_response`` objects.

    The SDK builds them on first access from the methods bound at that moment.
    One cached before ``monitor()`` (or before ``undo()``) would keep calling the
    method it captured, so the budget check is skipped. Dropping the cache makes
    the next access rebuild it from the current methods. A reference the caller
    stored themselves cannot be reached.
    """
    seen: set[int] = set()
    for resource in resources:
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))
        state = getattr(resource, "__dict__", None)
        if not isinstance(state, dict):
            continue
        for attr in _SDK_RESPONSE_WRAPPER_ATTRS:
            state.pop(attr, None)


def _block_openai_beta_chat_stream(client: Any) -> None:
    completions = getattr(getattr(getattr(client, "beta", None), "chat", None), "completions", None)
    if completions is not None and hasattr(completions, "stream"):
        try:
            setattr(completions, "stream", _unsupported_mode_stub(_OPENAI_BETA_STREAM_UNSUPPORTED))
        except Exception:
            pass


_ANTHROPIC_BETA_UNSUPPORTED = (
    "Anthropic beta Messages (client.beta.messages) is not guarded by KazenAI Control: "
    "calls there would skip the pre-call budget check. Use client.messages, or drop "
    "beta-only options (for LangChain ChatAnthropic, do not pass betas=)."
)


def _block_anthropic_beta_messages(client: Any) -> None:
    """Close the unguarded ``client.beta.messages`` path on a monitored Anthropic client.

    ``monitor()`` guards ``client.messages``; the beta resource dispatches the same
    paid calls through a separate object. Like the OpenAI Responses API, it fails
    loudly instead of spending outside the budget.
    """
    beta = getattr(client, "beta", None)
    messages = getattr(beta, "messages", None) if beta is not None else None
    if messages is None:
        return
    for name in ("create", "stream", "parse"):
        if hasattr(messages, name):
            try:
                setattr(messages, name, _unsupported_mode_stub(_ANTHROPIC_BETA_UNSUPPORTED))
            except Exception:
                pass
    _reset_sdk_response_wrappers([messages])


def _detect_client_kind(client: Any) -> str:
    """Return ``openai`` or ``anthropic`` based on client surface methods."""
    has_openai = False
    has_anthropic = False
    try:
        if (
            hasattr(client, "chat")
            and hasattr(client.chat, "completions")
            and hasattr(client.chat.completions, "create")
        ):
            has_openai = True
    except Exception:
        pass
    try:
        if hasattr(client, "messages") and hasattr(client.messages, "create"):
            has_anthropic = True
    except Exception:
        pass
    if has_openai:
        return "openai"
    if has_anthropic:
        return "anthropic"
    raise TypeError(
        "Unsupported client: expected OpenAI (`.chat.completions.create`) "
        "or Anthropic (`.messages.create`)"
    )


def monitor(
    client: Any,
    *,
    org_id: str = "local",
    project_id: str = "default",
    workspace_id: str = "default",
    agent_id: str = "agent",
    agent_role: str = "agent",
    max_budget_usd: Optional[float] = None,
    soft_pause_pct: float = 0.90,
    loop_anomaly_threshold: float = 0.90,
    stream_enforcement: bool = False,
    stream_cutoff_usd: Optional[float] = None,
    debug: bool = False,
    timeline_path: Optional[str] = None,
    capture_mode: Optional[str] = None,
    capture_consent: Optional[bool] = None,
    business_subject_ref: Optional[str] = None,
    feature_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    feature: Optional[str] = None,
    operation_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
    budget: Optional[Any] = None,
    allow_unmetered: Sequence[str] = (),
    observe_unbounded: Sequence[str] = (),
    require_customer_id: bool = False,
) -> Any:
    """
    High-level entrypoint for Agent FinOps on raw OpenAI-style or Anthropic clients.

    ``budget=SharedBudget(max_usd=...)`` makes several clients (for example one
    per provider in a multi-model workflow) draw from one local cap instead of
    one ``max_budget_usd`` each. Pass either ``budget`` or ``max_budget_usd``.

    Certified Control contract:
    - Sync and async OpenAI ``chat.completions.create`` (stream=False|True) and
      ``chat.completions.stream``
    - Sync and async OpenAI ``responses.create`` (stream=False|True), ``responses.parse``
      and ``responses.stream``; ``background``, server-side conversation state and
      built-in tools raise ``UnsupportedModeError`` before dispatch
    - Sync and async Anthropic ``messages.create`` (stream=False|True) and
      ``messages.stream``
    - Hard local cap raises ``BudgetExceeded`` before provider dispatch
    - Shared authority deny raises ``BudgetExceeded`` / unreachable raises ``BudgetUnavailable``
    - Loop guard raises ``LoopDetected`` before provider dispatch
    - Soft trajectory pause raises ``KazenCircuitBreaker`` after a completed call
    - OpenAI Realtime, legacy Completions and Responses ``compact`` / ``connect``
      raise ``UnsupportedModeError``
    - Streaming records provider start before dispatch, then settles at stream end
      or remains pending on cancel/error/missing usage; never at create-return
    - Any other paid request on the client (a method or ``with_raw_response``
      wrapper saved before ``monitor()``, embeddings, images, audio, batches...)
      raises ``UnsupportedModeError`` before it is sent. ``allow_unmetered=("/embeddings",)``
      sends the named endpoints without a budget check
    - Prefer ``max_retries=0`` on direct-provider clients so SDK auto-retries cannot
      hide billable attempts (monitor does not mutate the caller's client)

    API keys are env-only (``KAZENAI_FINOPS_API_KEY`` / ``KAZENAI_API_KEY``), not kwargs.

    ``require_customer_id=True`` refuses any call that names no customer ID
    (``business_subject_ref`` from ``use_attribution`` or this function): it raises
    ``CustomerIdRequired`` before the call reserves or reaches the provider. Without
    it, such a call meets only organization and workspace limits.

    Also enables:
    - model.call KazenEvent emission
    - finops.trajectory / finops.loop.anomaly derived events
    - optional local observable-output cutoff with ``stream_cutoff_usd``
      (raises ``StreamCutoffError``; final provider billing may still be pending)
    - optional HttpSink when FinOps ingest env is set
    - optional JsonlSink when ``timeline_path`` or ``KAZENAI_TIMELINE_PATH`` is set
    - private-by-default capture (``capture_mode=metadata``); bodies require explicit opt-in
    """

    from kazenai.lifecycle_config import resolve_effective_lifecycle

    lifecycle = resolve_effective_lifecycle()
    if lifecycle.is_control and not os.getenv("KAZENAI_TIER_MISMATCH_MULTIPLIER", "").strip():
        _monitor_log().warning(
            "KazenAI Control requires KAZENAI_TIER_MISMATCH_MULTIPLIER before "
            "priced calls; configure it as documented at "
            "https://docs.kazenai.com/reference/configuration"
        )
    if lifecycle.profile == "legacy" and lifecycle.source != "unset":
        import warnings

        warnings.warn(
            "The explicit KazenAI legacy lifecycle is deprecated and will be removed "
            "in KazenAI 2.0. Use the Control reserve/settle lifecycle.",
            FutureWarning,
            stacklevel=2,
        )

    if stream_enforcement:
        import warnings

        warnings.warn(
            "stream_enforcement is deprecated; Control-certified streaming uses automatic "
            "finalize-once accounting. Prefer stream_cutoff_usd for optional local cutoff.",
            DeprecationWarning,
            stacklevel=2,
        )
    if _streaming_explicitly_denied() and stream_enforcement:
        raise UnsupportedModeError(
            "Streaming is disabled by KAZENAI_DENY_STREAMING",
            error_code="unsupported_mode",
        )

    _check_hidden_provider_retries(client)

    resolved_capture = resolve_capture_mode(explicit=capture_mode, consent=capture_consent)
    _monitor_log().info("kazenai.capture_disclosure: %s", capture_disclosure(resolved_capture))

    sinks: list[EventSink] = [MemorySink(max_events=2000)]
    timeline = (timeline_path or os.getenv("KAZENAI_TIMELINE_PATH") or "").strip()
    if timeline:
        sinks.append(JsonlSink(timeline))
    ingest = (
        os.getenv("KAZENAI_FINOPS_INGEST_URL")
        or os.getenv("KAZENAI_FINOPS_URL")
        or os.getenv("KAZENAI_INGEST_URL")
        or ""
    ).strip()
    api_key = (os.getenv("KAZENAI_FINOPS_API_KEY") or os.getenv("KAZENAI_API_KEY") or "").strip()
    if ingest:
        sinks.append(HttpSink(HttpSinkConfig(ingest_url=ingest, api_key=api_key)))
    sink: EventSink = MultiSink(sinks)

    if budget is not None:
        from .budget import SharedBudget

        if not isinstance(budget, SharedBudget):
            raise TypeError("budget must be a kazenai.SharedBudget")
        if max_budget_usd is not None:
            raise ValueError("pass either budget=SharedBudget(...) or max_budget_usd, not both")
    soft_budget = budget.max_usd if budget is not None else max_budget_usd
    finops = FinOpsController(cfg=FinOpsConfig(budget_usd=soft_budget, soft_pause_pct=soft_pause_pct, loop_anomaly_threshold=loop_anomaly_threshold))
    # Local max_budget_usd must feed the same pre-call Enforcement
    # used on the patched provider path (not only FinOpsController post-call CB).
    if budget is not None:
        enforcement = budget.enforcement
    else:
        enforcement = Enforcement(max_cost_usd=max_budget_usd) if max_budget_usd is not None else Enforcement()
    patch_kwargs = dict(
        org_id=org_id,
        project_id=project_id,
        workspace_id=workspace_id,
        agent_id=agent_id,
        agent_role=agent_role,
        debug=debug,
        event_sink=sink,
        finops=finops,
        enforcement=enforcement,
        capture_mode=resolved_capture,
        business_subject_ref=business_subject_ref,
        feature_id=feature_id,
        workflow_id=workflow_id,
        feature=feature,
        operation_id=operation_id,
        attempt_id=attempt_id,
        allow_unmetered=tuple(allow_unmetered or ()),
        observe_unbounded=tuple(observe_unbounded or ()),
        require_customer_id=bool(require_customer_id),
    )
    is_async = _client_is_async(client)
    if _detect_client_kind(client) == "anthropic":
        if is_async:
            from ._async_provider import patch_anthropic_async

            patch_anthropic_async(
                client,
                **patch_kwargs,
                stream_cutoff_usd=stream_cutoff_usd,
                certified_surface=True,
            )
        else:
            patch_anthropic(
                client,
                **patch_kwargs,
                stream_cutoff_usd=stream_cutoff_usd,
                certified_surface=True,
            )
        _block_anthropic_beta_messages(client)
    else:
        if is_async:
            from ._async_provider import patch_openai_async

            patch_openai_async(
                client,
                **patch_kwargs,
                stream_enforcement=stream_enforcement,
                stream_cutoff_usd=stream_cutoff_usd,
                certified_surface=True,
            )
        else:
            patch_openai(
                client,
                **patch_kwargs,
                stream_enforcement=stream_enforcement,
                stream_cutoff_usd=stream_cutoff_usd,
                certified_surface=True,
            )
        _block_openai_beta_chat_stream(client)
    try:
        setattr(client, "_kazenai_monitored", True)
    except Exception:
        pass

    # Clones must share the same Enforcement counter (not a fresh budget).
    try:
        setattr(client, "_kazenai_enforcement", enforcement)
        setattr(client, "_kazenai_finops_controller", finops)
    except Exception:
        pass

    def _rewrap_clone(clone: Any) -> Any:
        if clone is None or clone is client:
            return clone
        if getattr(clone, "_kazenai_monitored", False):
            return clone
        shared_enforcement = getattr(client, "_kazenai_enforcement", enforcement)
        shared_finops = getattr(client, "_kazenai_finops_controller", finops)
        clone_kwargs = dict(
            org_id=org_id,
            project_id=project_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
            agent_role=agent_role,
            debug=debug,
            event_sink=sink,
            finops=shared_finops,
            enforcement=shared_enforcement,
            capture_mode=resolved_capture,
            business_subject_ref=business_subject_ref,
            feature_id=feature_id,
            workflow_id=workflow_id,
            feature=feature,
            operation_id=operation_id,
            attempt_id=attempt_id,
            allow_unmetered=tuple(allow_unmetered or ()),
            observe_unbounded=tuple(observe_unbounded or ()),
            require_customer_id=bool(require_customer_id),
        )
        clone_async = _client_is_async(clone)
        if _detect_client_kind(clone) == "anthropic":
            if clone_async:
                from ._async_provider import patch_anthropic_async

                patch_anthropic_async(
                    clone,
                    **clone_kwargs,
                    stream_cutoff_usd=stream_cutoff_usd,
                    certified_surface=True,
                )
            else:
                patch_anthropic(
                    clone,
                    **clone_kwargs,
                    stream_cutoff_usd=stream_cutoff_usd,
                    certified_surface=True,
                )
            _block_anthropic_beta_messages(clone)
        else:
            if clone_async:
                from ._async_provider import patch_openai_async

                patch_openai_async(
                    clone,
                    **clone_kwargs,
                    stream_enforcement=stream_enforcement,
                    stream_cutoff_usd=stream_cutoff_usd,
                    certified_surface=True,
                )
            else:
                patch_openai(
                    clone,
                    **clone_kwargs,
                    stream_enforcement=stream_enforcement,
                    stream_cutoff_usd=stream_cutoff_usd,
                    certified_surface=True,
                )
            _block_openai_beta_chat_stream(clone)
        try:
            setattr(clone, "_kazenai_monitored", True)
            setattr(clone, "_kazenai_enforcement", shared_enforcement)
            setattr(clone, "_kazenai_finops_controller", shared_finops)
        except Exception:
            pass
        for _clone_name2 in ("with_options", "copy"):
            _orig2 = getattr(clone, _clone_name2, None)
            if not callable(_orig2):
                continue
            try:
                setattr(clone, _clone_name2, _make_clone_wrapper(_orig2))
            except Exception:
                pass
        return clone

    def _make_clone_wrapper(orig_fn: Any):
        def _wrapped_clone(*a: Any, **k: Any) -> Any:
            clone = orig_fn(*a, **k)
            try:
                return _rewrap_clone(clone)
            except UnsupportedModeError:
                raise
            except Exception as exc:
                raise UnsupportedModeError(
                    "monitored client clone could not retain Control instrumentation: "
                    f"{exc}",
                    error_code="unguarded_client_clone",
                ) from exc

        return _wrapped_clone

    for _clone_name in ("with_options", "copy"):
        _orig_clone = getattr(client, _clone_name, None)
        if not callable(_orig_clone):
            continue
        try:
            setattr(client, _clone_name, _make_clone_wrapper(_orig_clone))
        except Exception:
            pass
    # Monitor a sibling provider client (for example the async twin a framework
    # builds next to this one) on this client's budget, tenant and attribution.
    try:
        setattr(client, "_kazenai_monitor_sibling", _rewrap_clone)
    except Exception:
        pass
    return client
