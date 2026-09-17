from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import threading
import time
import urllib.request
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


def _control_profile_denies_streaming() -> bool:
    """FINAL_1 Control: streaming ticks/extensions are unsupported at the caller boundary."""
    for key in (
        "KAZENAI_CONTROL_PROFILE",
        "KAZENAI_FINOPS_CONTROL_PROFILE",
        "KAZENAI_PROFILE",
        "KAZENAI_DENY_STREAMING",
    ):
        val = os.getenv(key, "").strip().lower()
        if val in ("1", "true", "yes", "control"):
            return True
    return False


def _reject_streaming_if_unsupported(kwargs: Mapping[str, Any]) -> None:
    if not kwargs.get("stream"):
        return
    if _control_profile_denies_streaming():
        raise UnsupportedModeError(
            "Streaming is unsupported in the Control profile; use non-streaming Chat Completions / Messages",
            error_code="unsupported_mode",
        )



# FINAL_1 P3-2: certified Control provider-client contract (sync non-streaming).
SUPPORTED_OPENAI_METHOD = "openai.chat.completions.create"
SUPPORTED_ANTHROPIC_METHOD = "anthropic.messages.create"


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
    """Conservative pre-call cost floor (USD), derived from the real per-model price table.

    Passed as ``projected_cost_usd`` to ``Enforcement.check_local`` so a single expensive run
    at the budget edge is denied before execution. Falls back to ``usd_per_1k_tokens`` and
    finally to 0.0 (the prior behavior) when the model/price is unknown, so uncapped or
    price-less callers are unaffected.
    """
    model = str((kwargs or {}).get("model") or "").strip()
    try:
        if model:
            global _COST_ENGINE
            if _COST_ENGINE is None:
                from .cost_engine import TokenCostEngine

                _COST_ENGINE = TokenCostEngine()
            cost = _COST_ENGINE.cost_usd(
                model=model, input_tokens=_PRECALL_MIN_TOKENS, output_tokens=_PRECALL_MIN_TOKENS
            )
            if cost and cost > 0:
                return float(cost)
    except Exception as exc:  # pragma: no cover - pricing must never break the call path
        _log_fail_open("precall_projection", exc)
    if usd_per_1k_tokens:
        return float(usd_per_1k_tokens) * (_PRECALL_MIN_TOKENS / 1000.0)
    return 0.0


# ---------------------------------------------------------------------------
# FinOps service pre-call reservation helpers
# ---------------------------------------------------------------------------


def _reservation_payload_fields(reserved) -> dict:
    """Carry call/attempt/reservation IDs on model.call for FinOps ingest settle."""
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
        }
        if reserved.reserved_usd_micros:
            out["actual_usd_micros"] = None  # filled after usage when known
        return {k: v for k, v in out.items() if v is not None}
    try:
        return {"reserved_cost_usd": float(reserved)}
    except Exception:
        return {}

def _try_reserve_budget(
    *,
    org_id: str,
    workspace_id: str,
    run_id: str,
    call_id: Optional[str] = None,
    estimated_cost_usd: Optional[float] = None,
):
    """Pre-call reservation via canonical spine (raises on fail-closed deny).

    Returns a float (legacy) or ReservationHandle (Control lifecycle).
    """
    from .spine.guard import reserve_budget

    try:
        return reserve_budget(
            org_id=org_id,
            workspace_id=workspace_id,
            run_id=run_id,
            increment_step=False,
            call_id=call_id,
            estimated_cost_usd=estimated_cost_usd,
        )
    except BudgetUnavailable:
        raise
    except BudgetExceeded:
        raise


def _try_stream_tick(
    *,
    org_id: str,
    workspace_id: str,
    run_id: str,
    incremental_cost_usd: float,
    model: str = "",
) -> bool:
    """Mid-stream budget tick; returns False when FinOps returns HTTP 402."""
    finops_url = os.getenv("KAZENAI_FINOPS_URL", "").rstrip("/")
    if not finops_url or incremental_cost_usd <= 0:
        return True

    api_key = os.getenv("KAZENAI_FINOPS_API_KEY", "")
    body = json.dumps(
        {
            "org_id": org_id,
            "workspace_id": workspace_id,
            "run_id": run_id,
            "incremental_cost_usd": incremental_cost_usd,
            "model": model,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{finops_url}/v1/budget/stream-tick",
        data=body,
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        method="POST",
    )
    try:
        timeout_s = float(os.getenv("KAZENAI_FINOPS_RESERVE_TIMEOUT_S", "0.8") or "0.8")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status < 400
    except urllib.error.HTTPError as exc:
        if exc.code == 402:
            return False
        from .deployment import finops_reservation_fail_closed

        if finops_reservation_fail_closed():
            raise StreamCutoffError(f"stream tick HTTP {exc.code}") from exc
        return True
    except StreamCutoffError:
        raise
    except Exception as exc:
        from .deployment import finops_reservation_fail_closed

        if finops_reservation_fail_closed():
            raise StreamCutoffError(f"stream tick unavailable: {exc}") from exc
        return True


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
) -> Any:
    """Wrap OpenAI-style stream chunks with mid-flight FinOps ticks."""
    if not stream_enforcement:
        return stream

    usd_per_token = float(usd_per_1k_tokens) / 1000.0 if usd_per_1k_tokens else 0.000002
    pending_tokens = 0
    pending_usd = 0.0

    def _chunk_text(chunk: Any) -> str:
        try:
            choices = getattr(chunk, "choices", None) or (chunk.get("choices") if isinstance(chunk, dict) else None)
            if choices:
                delta = getattr(choices[0], "delta", None) or (
                    choices[0].get("delta") if isinstance(choices[0], dict) else None
                )
                if delta:
                    content = getattr(delta, "content", None)
                    if content is None and isinstance(delta, dict):
                        content = delta.get("content")
                    return str(content or "")
        except Exception:
            pass
        return ""

    def _flush() -> None:
        nonlocal pending_tokens, pending_usd
        if pending_usd <= 0:
            return
        usd = pending_usd
        tokens = pending_tokens
        pending_usd = 0.0
        pending_tokens = 0
        allowed = _try_stream_tick(
            org_id=step_ctx.org_id,
            workspace_id=step_ctx.workspace_id,
            run_id=step_ctx.run_id,
            incremental_cost_usd=usd,
            model=model or "",
        )
        if not allowed:
            raise StreamCutoffError(
                "stream cutoff: budget exceeded mid-flight",
                blocked_usd=usd,
                total_tokens=tokens,
            )

    class _GuardedStream:
        def __iter__(self) -> Any:
            nonlocal pending_tokens, pending_usd
            for chunk in stream:
                text = _chunk_text(chunk)
                if text:
                    tokens = max(1, len(text) // 4)
                    pending_tokens += tokens
                    pending_usd += tokens * usd_per_token
                    if pending_tokens >= 16:
                        _flush()
                yield chunk
            if pending_usd > 0:
                _flush()

    return _GuardedStream()


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
    certified_surface: bool = False,
    capture_mode: CaptureMode = CaptureMode.METADATA,
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
            ctx = get_current_context() or ctx0
            step_ctx = ctx.child_step()
            with use_context(step_ctx):
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

                # FINAL_1 P3-1: local cap first (fail closed), then shared FinOps reserve.
                projected = _precall_projection_usd(kwargs, usd_per_1k_tokens)
                held_projection = enforcement.check_local(projected_cost_usd=projected)

                # Pre-call server-side budget reservation (fails open if service unavailable)
                reserved_cost_usd = _try_reserve_budget(
                    org_id=step_ctx.org_id,
                    workspace_id=step_ctx.workspace_id,
                    run_id=step_ctx.run_id,
                    call_id=str(getattr(step_ctx, "step_id", None) or "") or None,
                    estimated_cost_usd=projected,
                )
                started = time.perf_counter()
                try:
                    resp = fn(*args, **kwargs)
                    if kwargs.get("stream") and stream_enforcement:
                        model_name = str(kwargs.get("model") or "")
                        resp = _wrap_stream_iterator(
                            resp,
                            step_ctx=step_ctx,
                            model=model_name,
                            stream_enforcement=True,
                            usd_per_1k_tokens=usd_per_1k_tokens,
                        )
                except Exception:
                    enforcement.record_call(cost_usd=0.0, released_projection=held_projection)
                    raise
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
    certified_surface: bool = False,
    capture_mode: CaptureMode = CaptureMode.METADATA,
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
    dbg = DebugPrinter(enabled=debug)

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
            ctx = get_current_context() or ctx0
            step_ctx = ctx.child_step()
            with use_context(step_ctx):
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

                # FINAL_1 P3-1: local cap first (fail closed), then shared FinOps reserve.
                projected = _precall_projection_usd(kwargs, usd_per_1k_tokens)
                held_projection = enforcement.check_local(projected_cost_usd=projected)

                # Pre-call server-side budget reservation (fails open if service unavailable)
                reserved_cost_usd = _try_reserve_budget(
                    org_id=step_ctx.org_id,
                    workspace_id=step_ctx.workspace_id,
                    run_id=step_ctx.run_id,
                    call_id=str(getattr(step_ctx, "step_id", None) or "") or None,
                    estimated_cost_usd=projected,
                )
                started = time.perf_counter()
                try:
                    resp = fn(*args, **kwargs)
                except Exception:
                    enforcement.record_call(cost_usd=0.0, released_projection=held_projection)
                    raise
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

    if certified_surface:
        try:
            if hasattr(client, "messages") and hasattr(client.messages, "stream"):
                orig_stream = client.messages.stream
                patches.append((client.messages, "stream", orig_stream))
                setattr(
                    client.messages,
                    "stream",
                    _unsupported_mode_stub(
                        "Anthropic streaming is unsupported on the certified Control path; "
                        "use sync non-streaming messages.create"
                    ),
                )
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
    debug: bool = False,
    timeline_path: Optional[str] = None,
    capture_mode: Optional[str] = None,
    capture_consent: Optional[bool] = None,
) -> Any:
    """
    High-level entrypoint for Agent FinOps on raw OpenAI-style or Anthropic clients.

    Certified Control contract (FINAL_1):
    - Sync non-streaming OpenAI ``chat.completions.create``
    - Sync non-streaming Anthropic ``messages.create``
    - Hard local cap raises ``BudgetExceeded`` before provider dispatch
    - Shared authority deny raises ``BudgetExceeded`` / unreachable raises ``BudgetUnavailable``
    - Loop guard raises ``LoopDetected`` before provider dispatch
    - Soft trajectory pause raises ``KazenCircuitBreaker`` after a completed call
    - Async clients, OpenAI Responses API, and Control streaming raise ``UnsupportedModeError``

    API keys are env-only (``KAZENAI_FINOPS_API_KEY`` / ``KAZENAI_API_KEY``), not kwargs.

    Also enables:
    - model.call KazenEvent emission
    - finops.trajectory / finops.loop.anomaly derived events
    - optional mid-stream cutoff when ``stream_enforcement=True`` outside Control
      (raises StreamCutoffError; unsupported under Control profile)
    - optional HttpSink when FinOps ingest env is set
    - optional JsonlSink when ``timeline_path`` or ``KAZENAI_TIMELINE_PATH`` is set
    - private-by-default capture (``capture_mode=metadata``); bodies require explicit opt-in
    """

    if stream_enforcement and _control_profile_denies_streaming():
        raise UnsupportedModeError(
            "stream_enforcement is unsupported in the Control profile",
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
    )
    if _detect_client_kind(client) == "anthropic":
        patch_anthropic(client, **patch_kwargs, certified_surface=True)
    else:
        patch_openai(
            client,
            **patch_kwargs,
            stream_enforcement=stream_enforcement,
            certified_surface=True,
        )
    return client
