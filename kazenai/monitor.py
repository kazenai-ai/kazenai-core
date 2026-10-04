from __future__ import annotations

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar

import asyncio
import inspect
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

try:
    import aiohttp  # type: ignore
except Exception:  # pragma: no cover
    aiohttp = None

try:
    import structlog  # type: ignore
except Exception:  # pragma: no cover
    structlog = None

from .context import RunContext, get_current_context, use_context
from .debug import DebugPrinter
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
    """Hard kill-switch only. Control-certified sync streaming is allowed by default (Train B)."""
    val = os.getenv("KAZENAI_DENY_STREAMING", "").strip().lower()
    return val in ("1", "true", "yes")


def _control_profile_active() -> bool:
    for key in (
        "KAZENAI_CONTROL_PROFILE",
        "KAZENAI_FINOPS_CONTROL_PROFILE",
        "KAZENAI_PROFILE",
    ):
        val = os.getenv(key, "").strip().lower()
        if val in ("1", "true", "yes", "control"):
            return True
    return False


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


# FINAL_1 / Train B: certified Control sync Chat Completions + Messages (stream + non-stream).
SUPPORTED_OPENAI_METHOD = "openai.chat.completions.create"
SUPPORTED_ANTHROPIC_METHOD = "anthropic.messages.create"
SUPPORTED_OPENAI_STREAM_METHOD = "openai.chat.completions.create.stream"
SUPPORTED_OPENAI_STREAM_MANAGER = "openai.chat.completions.stream"
SUPPORTED_ANTHROPIC_STREAM_METHOD = "anthropic.messages.create.stream"
SUPPORTED_ANTHROPIC_STREAM_MANAGER = "anthropic.messages.stream"

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


def _reject_async_client(client: Any) -> None:
    """Async OpenAI/Anthropic clients are FINAL_2 — fail closed on the public monitor path."""
    name = type(client).__name__
    if "Async" in name:
        raise UnsupportedModeError(
            f"Async client {name!r} is unsupported on the certified monitor path; use sync OpenAI/Anthropic",
            error_code="unsupported_mode",
        )
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
        fn = obj
        if inspect.iscoroutinefunction(fn):
            raise UnsupportedModeError(
                "Async create() is unsupported on the certified monitor path",
                error_code="unsupported_mode",
            )
        # Unwrap bound method
        underlying = getattr(fn, "__func__", None)
        if underlying is not None and inspect.iscoroutinefunction(underlying):
            raise UnsupportedModeError(
                "Async create() is unsupported on the certified monitor path",
                error_code="unsupported_mode",
            )


def _unsupported_mode_stub(message: str) -> Callable[..., Any]:
    def _stub(*args: Any, **kwargs: Any) -> Any:
        raise UnsupportedModeError(message, error_code="unsupported_mode")

    return _stub


def _apply_output_token_bound(kwargs: Dict[str, Any]) -> None:
    """Inject/clamp max_tokens using model_pricing.default_max_output_tokens."""
    try:
        from .model_pricing import default_max_output_tokens
    except Exception:
        return
    model = str(kwargs.get("model") or "").strip()
    if not model:
        return
    bound = default_max_output_tokens(model)
    if bound is None:
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
    kwargs["max_tokens"] = int(bound)


def _log_fail_open(context: str, exc: BaseException) -> None:
    """Log unexpected exceptions on intentional fail-open paths without blocking calls."""
    if isinstance(exc, _EXPECTED_FAIL_OPEN):
        return
    _monitor_log().warning("kazenai.monitor fail-open: %s", context, exc_info=True)


# ---------------------------------------------------------------------------
# Pre-call cost projection (SEC-R2-9)
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
    for key in ("max_completion_tokens", "max_tokens"):
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
        cost = _COST_ENGINE.cost_usd(model=model_name, input_tokens=in_tok, output_tokens=out_tok)
        if cost and cost > 0:
            return float(cost)
    except Exception as exc:  # pragma: no cover
        _log_fail_open("postcall_cost", exc)
    return None


def _precall_projection_usd(kwargs: Mapping[str, Any], usd_per_1k_tokens: Optional[float]) -> float:
    """Bounded pre-call exposure from estimated input plus enforced output cap.

    Passed as ``projected_cost_usd`` to ``Enforcement.check_local`` so a single expensive run
    at the budget edge is denied before execution. Falls back to ``usd_per_1k_tokens`` and
    finally to 0.0 (the prior behavior) when the model/price is unknown, so uncapped or
    price-less callers are unaffected.
    """
    model = str((kwargs or {}).get("model") or "").strip()
    input_tokens = _estimate_input_tokens(kwargs)
    output_tokens = _max_output_tokens(kwargs)
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
        # A configured block policy is part of the public financial-control
        # contract, not an internal pricing failure that may fail open.
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
    return held_projection, reserved


def _try_reconcile_budget(
    *,
    org_id: str,
    workspace_id: str,
    run_id: str,
    reserved_cost_usd,
    actual_cost_usd: float,
) -> None:
    """Fire-and-forget reconcile/settle via canonical spine."""
    from .spine.guard import reconcile_budget

    reconcile_budget(
        org_id=org_id,
        workspace_id=workspace_id,
        run_id=run_id,
        reserved_cost_usd=reserved_cost_usd,
        actual_cost_usd=actual_cost_usd,
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
) -> bool:
    """Keep shared exposure pending when terminal provider usage is unknown."""
    del workspace_id
    from .spine.guard import signal_reservation_event

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
        return {"input": kwargs.get("input"), "model": kwargs.get("model")}
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
                name = fn.get("name") if isinstance(fn, dict) else None
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
            return str(payload["input"])
        return str(payload)
    except Exception as exc:
        _log_fail_open("extract_input_text", exc)
        return ""


def _extract_usage(resp: Any) -> Tuple[Optional[int], Any]:
    try:
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
) -> Callable[[], None]:
    """
    Monkey-patch an OpenAI-style client to enforce budgets/loops *before* each LLM call.

    When ``certified_surface=True`` (public ``monitor()`` path), only Chat Completions
    is enforced; ``responses.create`` is stubbed to ``UnsupportedModeError``.

    Returns an `undo()` function that restores the original methods.
    """

    if structlog is not None:
        log = structlog.get_logger(__name__)
    else:
        log = logging.getLogger(__name__)
    enforcement = enforcement or Enforcement()
    loop_detector = loop_detector or LoopDetector()
    dbg = DebugPrinter(enabled=debug)

    from .attribution import use_attribution
    from .enforcement_owner import claim_enforcement_owner

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

    def _wrap(fn: Callable[..., Any], *, method: str) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _HELPER_INNER_CREATE_BYPASS.get():
                return fn(*args, **kwargs)
            ctx = get_current_context() or ctx0
            step_ctx = ctx.child_step()
            with ExitStack() as stack:
                stack.enter_context(use_context(step_ctx))
                stack.enter_context(claim_enforcement_owner("sdk"))
                stack.enter_context(use_attribution(**attr_defaults))
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

                # FINAL_1 P2-5: reject streaming before any network/provider work.
                _reject_streaming_if_unsupported(kwargs)
                # FINAL_1 P3-2: enforceable output token bound before dispatch.
                _apply_output_token_bound(kwargs)

                call_kwargs = dict(kwargs)
                is_stream = bool(call_kwargs.get("stream"))
                if is_stream:
                    from .streaming.lifecycle import StreamAttempt, StreamOutcome
                    from .streaming.openai import merge_stream_options, wrap_openai_stream

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

                # FINAL_1 P3-1: local cap first (fail closed), then shared FinOps reserve.
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
                if is_stream:
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
                        surface=SUPPORTED_OPENAI_STREAM_METHOD,
                        step_ctx=step_ctx,
                        enforcement=enforcement,
                        held_projection=held_projection,
                        reserved_cost_usd=reserved_cost_usd,
                        model=model_name,
                        method=SUPPORTED_OPENAI_STREAM_METHOD,
                        usd_per_1k_tokens=output_rate,
                        capture_mode=capture_mode,
                        agent_role=agent_role,
                        event_sink=event_sink,
                        finops=finops,
                        stream_cutoff_usd=cutoff,
                        reconcile_fn=_try_settle_stream_budget,
                        pending_fn=_try_mark_stream_pending,
                        postcall_cost_fn=_postcall_cost_usd,
                        reservation_payload_fn=_reservation_payload_fields,
                        emit_event_fn=_emit,
                        attribution=attr_defaults,
                        started_perf=started,
                    )
                try:
                    resp = fn(*args, **call_kwargs)
                except Exception as exc:
                    if attempt is not None:
                        attempt.finalize(StreamOutcome.PROVIDER_ERROR, error=exc)
                    else:
                        enforcement.record_call(
                            cost_usd=0.0,
                            released_projection=held_projection,
                        )
                    raise

                if is_stream:
                    assert attempt is not None
                    return wrap_openai_stream(resp, attempt)

                elapsed_ms = (time.perf_counter() - started) * 1000.0

                tokens_used, usage = _extract_usage(resp)
                cost_usd = _postcall_cost_usd(
                    model=str(kwargs.get("model") or ""),
                    usage=usage,
                    tokens_used=tokens_used,
                    usd_per_1k_tokens=usd_per_1k_tokens,
                )

                # Post-call reconcile (fire-and-forget daemon thread)
                if reserved_cost_usd is not None:
                    _try_reconcile_budget(
                        org_id=step_ctx.org_id,
                        workspace_id=step_ctx.workspace_id,
                        run_id=step_ctx.run_id,
                        reserved_cost_usd=reserved_cost_usd,
                        actual_cost_usd=cost_usd or 0.0,
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
    if not certified_surface:
        try:
            if hasattr(client, "responses") and hasattr(client.responses, "create"):
                targets.append((client.responses, "create", "openai.responses.create"))
        except Exception:
            pass

    if not targets:
        raise TypeError(
            "Unsupported client: expected `.chat.completions.create`"
            + ("" if certified_surface else " and/or `.responses.create`")
        )

    for obj, name, event_type in targets:
        orig = getattr(obj, name)
        if not callable(orig):
            continue
        patches.append((obj, name, orig))
        setattr(obj, name, _wrap(orig, method=event_type))

    # Train B: the official SDK exposes a lazy ChatCompletionStreamManager on
    # ``chat.completions.stream``. It bypasses ``create``, so guard it directly.
    try:
        completions = client.chat.completions
        if hasattr(completions, "stream"):
            orig_stream = completions.stream
            if callable(orig_stream):

                def _stream_manager(*s_args: Any, **s_kwargs: Any) -> Any:
                    from .streaming.lifecycle import StreamAttempt
                    from .streaming.openai import merge_stream_options, wrap_openai_manager

                    _reject_streaming_if_unsupported({"stream": True})
                    ctx = get_current_context() or ctx0
                    step_ctx = ctx.child_step()
                    call_kwargs = dict(s_kwargs)
                    _apply_output_token_bound(call_kwargs)
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
                    projected = _precall_projection_usd(call_kwargs, usd_per_1k_tokens)
                    # Construction is lazy; reserve only when __enter__ is called.
                    manager = orig_stream(*s_args, **call_kwargs)

                    def _attempt_factory() -> StreamAttempt:
                        with ExitStack() as stack:
                            stack.enter_context(use_context(step_ctx))
                            stack.enter_context(claim_enforcement_owner("sdk"))
                            stack.enter_context(use_attribution(**attr_defaults))
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
                            surface=SUPPORTED_OPENAI_STREAM_MANAGER,
                            step_ctx=step_ctx,
                            enforcement=enforcement,
                            held_projection=held_projection,
                            reserved_cost_usd=reserved_cost_usd,
                            model=model_name,
                            method=SUPPORTED_OPENAI_STREAM_MANAGER,
                            usd_per_1k_tokens=output_rate,
                            capture_mode=capture_mode,
                            agent_role=agent_role,
                            event_sink=event_sink,
                            finops=finops,
                            stream_cutoff_usd=cutoff,
                            reconcile_fn=_try_settle_stream_budget,
                            pending_fn=_try_mark_stream_pending,
                            postcall_cost_fn=_postcall_cost_usd,
                            reservation_payload_fn=_reservation_payload_fields,
                            emit_event_fn=_emit,
                            attribution=attr_defaults,
                            started_perf=started,
                        )

                    return wrap_openai_manager(
                        manager,
                        attempt_factory=_attempt_factory,
                        enter_scope=_helper_inner_create_bypass,
                    )

                patches.append((completions, "stream", orig_stream))
                setattr(completions, "stream", _stream_manager)
    except Exception:
        pass

    # FINAL_1 P3-2: certified monitor must not silently leave Responses unguarded.
    if certified_surface:
        try:
            if hasattr(client, "responses") and hasattr(client.responses, "create"):
                orig_resp = client.responses.create
                patches.append((client.responses, "create", orig_resp))
                setattr(
                    client.responses,
                    "create",
                    _unsupported_mode_stub(
                        "OpenAI Responses API is unsupported on the certified Control path; "
                        "use sync non-streaming chat.completions.create"
                    ),
                )
        except Exception:
            pass

    def undo() -> None:
        for obj, name, orig in patches:
            try:
                setattr(obj, name, orig)
            except Exception:
                log.exception("kazenai.patch_undo_failed", target=str(obj), name=name)

    return undo


def _extract_anthropic_usage(resp: Any) -> Tuple[Optional[int], Any]:
    """Extract token counts from an Anthropic `messages.create` response."""
    try:
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

    from .attribution import use_attribution
    from .enforcement_owner import claim_enforcement_owner

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

    def _wrap(fn: Callable[..., Any], *, method: str) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _HELPER_INNER_CREATE_BYPASS.get():
                return fn(*args, **kwargs)
            ctx = get_current_context() or ctx0
            step_ctx = ctx.child_step()
            with ExitStack() as stack:
                stack.enter_context(use_context(step_ctx))
                stack.enter_context(claim_enforcement_owner("sdk"))
                stack.enter_context(use_attribution(**attr_defaults))
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

                # FINAL_1 P2-5: reject streaming before any network/provider work.
                _reject_streaming_if_unsupported(kwargs)
                # FINAL_1 P3-2: enforceable output token bound before dispatch.
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

                # FINAL_1 P3-1: local cap first (fail closed), then shared FinOps reserve.
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
                if is_stream:
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
                        capture_mode=capture_mode,
                        agent_role=agent_role,
                        event_sink=event_sink,
                        finops=finops,
                        stream_cutoff_usd=stream_cutoff_usd,
                        reconcile_fn=_try_settle_stream_budget,
                        pending_fn=_try_mark_stream_pending,
                        postcall_cost_fn=_postcall_cost_usd,
                        reservation_payload_fn=_reservation_payload_fields,
                        emit_event_fn=_emit,
                        attribution=attr_defaults,
                        started_perf=started,
                    )
                try:
                    resp = fn(*args, **call_kwargs)
                except Exception as exc:
                    if attempt is not None:
                        attempt.finalize(StreamOutcome.PROVIDER_ERROR, error=exc)
                    else:
                        enforcement.record_call(
                            cost_usd=0.0,
                            released_projection=held_projection,
                        )
                    raise

                if is_stream:
                    assert attempt is not None
                    return wrap_anthropic_event_stream(resp, attempt)

                elapsed_ms = (time.perf_counter() - started) * 1000.0

                tokens_used, usage = _extract_anthropic_usage(resp)
                cost_usd = _postcall_cost_usd(
                    model=str(kwargs.get("model") or ""),
                    usage=usage,
                    tokens_used=tokens_used,
                    usd_per_1k_tokens=usd_per_1k_tokens,
                )

                # Post-call reconcile (fire-and-forget daemon thread)
                if reserved_cost_usd is not None:
                    _try_reconcile_budget(
                        org_id=step_ctx.org_id,
                        workspace_id=step_ctx.workspace_id,
                        run_id=step_ctx.run_id,
                        reserved_cost_usd=reserved_cost_usd,
                        actual_cost_usd=cost_usd or 0.0,
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
        setattr(obj, name, _wrap(orig, method=event_type))

    # Train B: wrap messages.stream manager with finalize-once lifecycle.
    try:
        if hasattr(client, "messages") and hasattr(client.messages, "stream"):
            orig_stream = client.messages.stream
            if callable(orig_stream):

                def _stream_manager(*s_args: Any, **s_kwargs: Any) -> Any:
                    from .streaming.lifecycle import StreamAttempt
                    from .streaming.anthropic import wrap_anthropic_manager

                    ctx = get_current_context() or ctx0
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
                    projected = _precall_projection_usd(call_kwargs, usd_per_1k_tokens)
                    # Anthropic's helper returns a lazy manager.  Constructing it
                    # is not provider dispatch, so do not reserve until __enter__.
                    manager = orig_stream(*s_args, **call_kwargs)

                    def _attempt_factory() -> StreamAttempt:
                        with ExitStack() as stack:
                            stack.enter_context(use_context(step_ctx))
                            stack.enter_context(claim_enforcement_owner("sdk"))
                            stack.enter_context(use_attribution(**attr_defaults))
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
                            capture_mode=capture_mode,
                            agent_role=agent_role,
                            event_sink=event_sink,
                            finops=finops,
                            stream_cutoff_usd=stream_cutoff_usd,
                            reconcile_fn=_try_settle_stream_budget,
                            pending_fn=_try_mark_stream_pending,
                            postcall_cost_fn=_postcall_cost_usd,
                            reservation_payload_fn=_reservation_payload_fields,
                            emit_event_fn=_emit,
                            attribution=attr_defaults,
                            started_perf=started,
                        )

                    return wrap_anthropic_manager(
                        manager,
                        attempt_factory=_attempt_factory,
                        enter_scope=_helper_inner_create_bypass,
                    )

                patches.append((client.messages, "stream", orig_stream))
                setattr(client.messages, "stream", _stream_manager)
    except Exception:
        pass

    def undo() -> None:
        for obj, name, orig in patches:
            try:
                setattr(obj, name, orig)
            except Exception:
                pass

    return undo


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
) -> Any:
    """
    High-level entrypoint for Agent FinOps on raw OpenAI-style or Anthropic clients.

    Certified Control contract (FINAL_1 + Train B):
    - Sync OpenAI ``chat.completions.create`` (stream=False|True) and
      ``chat.completions.stream``
    - Sync Anthropic ``messages.create`` (stream=False|True) and ``messages.stream``
    - Hard local cap raises ``BudgetExceeded`` before provider dispatch
    - Shared authority deny raises ``BudgetExceeded`` / unreachable raises ``BudgetUnavailable``
    - Loop guard raises ``LoopDetected`` before provider dispatch
    - Soft trajectory pause raises ``KazenCircuitBreaker`` after a completed call
    - Async clients and OpenAI Responses API raise ``UnsupportedModeError``
    - Streaming records provider start before dispatch, then settles at stream end
      or remains pending on cancel/error/missing usage; never at create-return

    API keys are env-only (``KAZENAI_FINOPS_API_KEY`` / ``KAZENAI_API_KEY``), not kwargs.

    Also enables:
    - model.call KazenEvent emission
    - finops.trajectory / finops.loop.anomaly derived events
    - optional local observable-output cutoff with ``stream_cutoff_usd``
      (raises ``StreamCutoffError``; final provider billing may still be pending)
    - optional HttpSink when FinOps ingest env is set
    - optional JsonlSink when ``timeline_path`` or ``KAZENAI_TIMELINE_PATH`` is set
    - private-by-default capture (``capture_mode=metadata``); bodies require explicit opt-in
    """

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

    # FINAL_1 P3-2: public monitor certifies sync OpenAI Chat Completions + Anthropic Messages only.
    _reject_async_client(client)

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

    finops = FinOpsController(cfg=FinOpsConfig(budget_usd=max_budget_usd, soft_pause_pct=soft_pause_pct, loop_anomaly_threshold=loop_anomaly_threshold))
    # FINAL_1 P3-1 / G06: local max_budget_usd must feed the same pre-call Enforcement
    # used on the patched provider path (not only FinOpsController post-call CB).
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
    )
    if _detect_client_kind(client) == "anthropic":
        patch_anthropic(
            client,
            **patch_kwargs,
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
    return client
