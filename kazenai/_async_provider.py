"""Async OpenAI / Anthropic monitor patches.

Shares admission/settlement helpers with ``monitor.py``; transport is native
awaitable provider SDK calls plus httpx FinOps I/O.
"""

from __future__ import annotations

import importlib
import logging
import time
from contextlib import ExitStack, asynccontextmanager, nullcontext
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .context import RunContext, get_current_context, use_context
from ._dispatch_gate import install_dispatch_gate
from .enforcement import Enforcement, LoopDetected, UnsupportedModeError
from .schema import KazenEvent, new_id, now_ms

try:
    import structlog  # type: ignore
except Exception:  # pragma: no cover
    structlog = None  # type: ignore

_log = logging.getLogger("kazenai.async_provider")


def _monitor_mod():
    """Load the monitor submodule (not the shadowed ``kazenai.monitor`` function)."""
    return importlib.import_module("kazenai.monitor")


async def _priced_projection(
    m,
    call_kwargs,
    usd_per_1k_tokens,
    *,
    provider: str,
    surface: str,
    client,
    org_id: str,
    observe_unbounded=(),
):
    """Arm the price scope, count long context, then reuse the sync projection."""
    m._arm_price_scope(
        provider=provider,
        surface=surface,
        client=client,
        org_id=org_id,
        observe_unbounded=tuple(observe_unbounded or ()),
        kwargs=call_kwargs,
    )
    from .request_pricing import prepare_provider_call_async

    await prepare_provider_call_async(
        call_kwargs,
        provider=provider,
        surface=surface,
        client=client,
        org_id=org_id,
        observe_unbounded=tuple(observe_unbounded or ()),
    )
    return m._precall_projection_usd(call_kwargs, usd_per_1k_tokens)




@asynccontextmanager
async def _helper_inner_create_bypass_async(token_var, gate_scope=nullcontext):
    token = token_var.set(True)
    try:
        # The stream helper dispatches during __aenter__, after it reserved.
        with gate_scope():
            yield
    finally:
        token_var.reset(token)


def patch_openai_async(
    client: Any,
    *,
    org_id: str,
    project_id: str,
    workspace_id: str,
    agent_id: str,
    agent_role: str,
    debug: bool,
    event_sink: Any,
    finops: Any,
    enforcement: Enforcement,
    capture_mode: Any,
    business_subject_ref: Optional[str],
    feature_id: Optional[str],
    workflow_id: Optional[str],
    feature: Optional[str],
    operation_id: Optional[str],
    attempt_id: Optional[str],
    stream_enforcement: bool = False,
    stream_cutoff_usd: Optional[float] = None,
    usd_per_1k_tokens: Optional[float] = None,
    loop_block_threshold: float = 0.90,
    loop_detector: Any = None,
    endpoint_url: Optional[str] = None,
    endpoint_headers: Optional[Dict[str, str]] = None,
    certified_surface: bool = True,
    allow_unmetered: Sequence[str] = (),
    observe_unbounded: Sequence[str] = (),
    require_customer_id: bool = False,
) -> Callable[[], None]:
    m = _monitor_mod()
    from .attribution import assert_customer_id, use_attribution
    from .capture_policy import build_content_ref, capture_disclosure
    from .debug import DebugPrinter
    from .enforcement_owner import claim_configured_enforcement_owner as claim_enforcement_owner
    from .finops import FinOpsController
    from .loop_detector import LoopDetector, LoopScores

    if structlog is not None:
        log = structlog.get_logger(__name__)
    else:
        log = _log
    loop_detector = loop_detector or LoopDetector()
    dbg = DebugPrinter(enabled=debug)
    del dbg  # parity with the sync path's optional pre-call printer
    attr_defaults = dict(
        business_subject_ref=business_subject_ref,
        feature_id=feature_id or feature,
        workflow_id=workflow_id,
        operation_id=operation_id,
        attempt_id=attempt_id,
    )
    ctx0 = get_current_context() or RunContext.new(
        org_id=org_id,
        project_id=project_id,
        workspace_id=workspace_id,
        agent_id=agent_id,
    )
    patches: List[Tuple[Any, str, Any]] = []
    # Backstop for references taken before monitor(): see _dispatch_gate.
    gate = install_dispatch_gate(client, patches, allow_unmetered=allow_unmetered, is_async=True)
    gate_scope = gate.authorized if gate is not None else nullcontext

    def _wrap(fn: Callable[..., Any], *, method: str) -> Callable[..., Any]:
        is_responses = method.startswith("openai.responses")
        stream_surface = (
            m.SUPPORTED_OPENAI_RESPONSES_STREAM_METHOD if is_responses else m.SUPPORTED_OPENAI_STREAM_METHOD
        )

        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            if m._HELPER_INNER_CREATE_BYPASS.get():
                return await fn(*args, **kwargs)
            if is_responses:
                # Requests whose cost cannot be bounded before dispatch fail here.
                m._validate_responses_request(kwargs)
            ctx = m._call_context(ctx0)
            step_ctx = ctx.child_step()
            with ExitStack() as stack:
                stack.enter_context(use_context(step_ctx))
                stack.enter_context(claim_enforcement_owner())
                stack.enter_context(use_attribution(**attr_defaults))
                assert_customer_id(require_customer_id)
                payload = m._extract_openai_input(args, kwargs)
                tool_names = m._extract_tool_names(kwargs)
                loop_scores: Optional[LoopScores] = None
                try:
                    loop_scores = loop_detector.score(
                        input_text=m._extract_input_text(payload),
                        tool_names=tool_names,
                    )
                    if (
                        loop_scores.combined >= float(loop_block_threshold)
                        and loop_detector.is_loop(loop_scores)
                    ):
                        raise LoopDetected(
                            f"loop detected: score={loop_scores.combined:.3f}"
                        )
                except LoopDetected:
                    raise
                except Exception:
                    log.exception("kazenai.loop_detector_failed")
                    loop_scores = None

                m._reject_streaming_if_unsupported(kwargs)
                m._apply_output_token_bound(kwargs, responses=is_responses)
                call_kwargs = dict(kwargs)
                is_stream = bool(call_kwargs.get("stream"))
                if is_stream:
                    from .streaming.lifecycle import StreamAttempt, StreamOutcome
                    from .streaming.openai import (
                        merge_stream_options,
                        wrap_openai_async_stream,
                    )
                    from .streaming.openai_responses import wrap_responses_async_stream

                    # Responses report usage on the terminal event; they reject
                    # stream_options.include_usage.
                    if not is_responses:
                        call_kwargs = merge_stream_options(call_kwargs)

                model_name = str(call_kwargs.get("model") or "")
                cutoff = stream_cutoff_usd
                output_rate = 0.0
                if is_stream:
                    output_rate = m._stream_output_rate_usd_per_1k(
                        model_name, usd_per_1k_tokens
                    )
                    if cutoff is None and stream_enforcement and output_rate > 0:
                        cutoff = max(0.01, output_rate * 0.05)
                    if cutoff is not None and output_rate <= 0:
                        raise UnsupportedModeError(
                            "stream_cutoff_usd requires known model pricing or an explicit "
                            "usd_per_1k_tokens rate",
                            error_code="stream_cutoff_pricing_unavailable",
                        )

                projected = await _priced_projection(
                    m,
                    call_kwargs,
                    usd_per_1k_tokens,
                    provider='openai',
                    surface="responses" if is_responses else "chat",
                    client=client,
                    org_id=step_ctx.org_id,
                    observe_unbounded=observe_unbounded,
                )
                held_projection, reserved_cost_usd = await m._reserve_with_local_hold_async(
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
                        "input_tokens": m._estimate_input_tokens(call_kwargs),
                    },
                )
                # Provider_started before every potential dispatch.
                await m._mark_stream_started_or_release_async(
                    enforcement=enforcement,
                    held_projection=held_projection,
                    org_id=step_ctx.org_id,
                    workspace_id=step_ctx.workspace_id,
                    run_id=step_ctx.run_id,
                    reserved_cost_usd=reserved_cost_usd,
                )

                attempt = None
                if is_stream:

                    def _emit(attempt, settle_cost, tokens_used, error):
                        m._emit_stream_model_call(
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
                        reconcile_async_fn=m._try_settle_stream_budget_async,
                        pending_async_fn=m._try_mark_stream_pending_async,
                        postcall_cost_fn=m._bound_postcall(),
                        reservation_payload_fn=m._reservation_payload_fields,
                        emit_event_fn=_emit,
                        attribution=attr_defaults,
                        started_perf=time.perf_counter(),
                    )
                started = time.perf_counter()
                try:
                    from .otel import build_control_attributes, control_span

                    if attempt is not None:
                        attempt.begin_otel_span(
                            provider="openai",
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
                        resp = await fn(*args, **call_kwargs)
                except Exception as exc:
                    rejected = m._definitive_provider_rejection(exc)
                    if attempt is not None:
                        await attempt.finalize_async(
                            StreamOutcome.REJECTED if rejected else StreamOutcome.PROVIDER_ERROR,
                            error=exc,
                        )
                    else:
                        # A definitive 4xx refusal settles at exactly $0 and releases the hold.
                        if reserved_cost_usd is not None:
                            await m._try_mark_stream_pending_async(
                                org_id=step_ctx.org_id,
                                workspace_id=step_ctx.workspace_id,
                                run_id=step_ctx.run_id,
                                reserved_cost_usd=reserved_cost_usd,
                                actual_cost_usd=0.0 if rejected else None,
                            )
                        enforcement.record_call(
                            cost_usd=0.0, released_projection=held_projection
                        )
                    raise
                except BaseException as cancel_exc:
                    # CancelledError after dispatch — finalize stream attempt or
                    # mark pending; never leave a reservation without cleanup evidence.
                    if attempt is not None:
                        try:
                            from .streaming.lifecycle import StreamOutcome as _SO

                            await attempt.finalize_async(
                                _SO.CLIENT_CANCELLED, error=cancel_exc
                            )
                        except Exception:
                            pass
                    elif reserved_cost_usd is not None:
                        try:
                            await m._try_mark_stream_pending_async(
                                org_id=step_ctx.org_id,
                                workspace_id=step_ctx.workspace_id,
                                run_id=step_ctx.run_id,
                                reserved_cost_usd=reserved_cost_usd,
                            )
                        except Exception:
                            pass
                    raise

                if is_stream:
                    assert attempt is not None
                    return m._wrap_stream_result(
                        resp,
                        attempt,
                        wrap_responses_async_stream if is_responses else wrap_openai_async_stream,
                    )

                elapsed_ms = (time.perf_counter() - started) * 1000.0
                tokens_used, usage = m._extract_usage(resp)
                cost_usd = m._postcall_cost_usd(
                    model=str(kwargs.get("model") or ""),
                    usage=usage,
                    tokens_used=tokens_used,
                    usd_per_1k_tokens=usd_per_1k_tokens,
                )
                if reserved_cost_usd is not None:
                    if cost_usd is None or m._pricing_holds_reconciliation():
                        await m._try_mark_stream_pending_async(
                            org_id=step_ctx.org_id,
                            workspace_id=step_ctx.workspace_id,
                            run_id=step_ctx.run_id,
                            reserved_cost_usd=reserved_cost_usd,
                        )
                    else:
                        try:
                            settled = await m._try_reconcile_budget_async(
                                org_id=step_ctx.org_id,
                                workspace_id=step_ctx.workspace_id,
                                run_id=step_ctx.run_id,
                                reserved_cost_usd=reserved_cost_usd,
                                actual_cost_usd=float(cost_usd),
                            )
                        except BaseException as settle_exc:
                            try:
                                await m._try_mark_stream_pending_async(
                                    org_id=step_ctx.org_id,
                                    workspace_id=step_ctx.workspace_id,
                                    run_id=step_ctx.run_id,
                                    reserved_cost_usd=reserved_cost_usd,
                                )
                            except Exception:
                                pass
                            if isinstance(settle_exc, Exception):
                                settled = False
                            else:
                                raise
                        if settled is not True:
                            try:
                                await m._try_mark_stream_pending_async(
                                    org_id=step_ctx.org_id,
                                    workspace_id=step_ctx.workspace_id,
                                    run_id=step_ctx.run_id,
                                    reserved_cost_usd=reserved_cost_usd,
                                    actual_cost_usd=float(cost_usd),
                                )
                            except Exception:
                                pass
                enforcement.record_call(
                    cost_usd=cost_usd, released_projection=held_projection
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
                            **m._reservation_payload_fields(reserved_cost_usd),
                            **m._attribution_payload_fields(),
                            "method": method,
                            "model": kwargs.get("model")
                            if isinstance(kwargs, dict)
                            else None,
                            "inputs_ref": build_content_ref(
                                payload, mode=capture_mode, role="inputs"
                            ),
                            "outputs_ref": build_content_ref(
                                m._safe_output(resp), mode=capture_mode, role="outputs"
                            ),
                            "capture": capture_disclosure(capture_mode),
                            "elapsed_ms": elapsed_ms,
                            "usage": usage if isinstance(usage, dict) else None,
                            "async": True,
                        },
                    )
                except Exception:
                    log.exception("kazenai.event_build_failed")
                    return resp
                if event_sink is not None:
                    try:
                        event_sink.emit(event)
                    except Exception as exc:
                        m._log_fail_open("event_sink_emit", exc)
                if finops is not None:
                    derived = finops.handle_llm_call(event)
                    for key, ev in derived.items():
                        if key.startswith("_"):
                            continue
                        try:
                            if event_sink is not None:
                                event_sink.emit(ev)
                        except Exception as exc:
                            m._log_fail_open("event_sink_emit_derived", exc)
                    FinOpsController.raise_if_blocked(derived)
                return resp

        return wrapped

    targets: List[Tuple[Any, str, str]] = []
    if hasattr(client, "chat") and hasattr(client.chat, "completions") and hasattr(
        client.chat.completions, "create"
    ):
        targets.append(
            (client.chat.completions, "create", m.SUPPORTED_OPENAI_METHOD)
        )
    targets.extend(m._openai_parse_targets(client))
    targets.extend(m._openai_responses_targets(client))
    if not targets:
        raise TypeError(
            "Unsupported async client: expected `.chat.completions.create` and/or `.responses.create`"
        )

    for obj, name, event_type in targets:
        orig = getattr(obj, name)
        patches.append((obj, name, orig))
        wrapped_fn = _wrap(orig, method=event_type)
        try:
            setattr(wrapped_fn, "__kazenai_wrapped__", True)
        except Exception:
            pass
        setattr(obj, name, wrapped_fn)

    # Async stream helper managers (``chat.completions.stream`` and ``responses.stream``).
    def _make_stream_manager(orig_stream: Callable[..., Any], *, responses: bool) -> Callable[..., Any]:
        surface = (
            m.SUPPORTED_OPENAI_RESPONSES_STREAM_MANAGER if responses else m.SUPPORTED_OPENAI_STREAM_MANAGER
        )

        def _stream_manager(*s_args: Any, **s_kwargs: Any) -> Any:
            from .streaming.lifecycle import StreamAttempt
            from .streaming.openai import (
                merge_stream_options,
                wrap_openai_async_manager,
            )
            from .streaming.openai_responses import wrap_responses_async_manager

            m._reject_streaming_if_unsupported({"stream": True})
            ctx = m._call_context(ctx0)
            step_ctx = ctx.child_step()
            call_kwargs = dict(s_kwargs)
            if responses:
                m._validate_responses_request(call_kwargs, stream_helper=True)
            m._apply_output_token_bound(call_kwargs, responses=responses)
            if not responses:
                call_kwargs = merge_stream_options(call_kwargs)
            call_kwargs.pop("stream", None)
            model_name = str(call_kwargs.get("model") or "")
            output_rate = m._stream_output_rate_usd_per_1k(
                model_name, usd_per_1k_tokens
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
            manager = orig_stream(*s_args, **call_kwargs)

            async def _attempt_factory() -> StreamAttempt:
                projected = await _priced_projection(
                    m,
                    call_kwargs,
                    usd_per_1k_tokens,
                    provider="openai",
                    surface="responses" if responses else "chat",
                    client=client,
                    org_id=step_ctx.org_id,
                    observe_unbounded=observe_unbounded,
                )
                with ExitStack() as stack:
                    stack.enter_context(use_context(step_ctx))
                    stack.enter_context(claim_enforcement_owner())
                    stack.enter_context(use_attribution(**attr_defaults))
                    assert_customer_id(require_customer_id)
                    held_projection, reserved_cost_usd = await m._reserve_with_local_hold_async(
                        enforcement=enforcement,
                        projected_cost_usd=projected,
                        reserve_kwargs={
                            "org_id": step_ctx.org_id,
                            "workspace_id": step_ctx.workspace_id,
                            "run_id": step_ctx.run_id,
                            "call_id": str(getattr(step_ctx, "step_id", None) or "")
                            or None,
                            "estimated_cost_usd": projected,
                            "business_subject_ref": attr_defaults.get(
                                "business_subject_ref"
                            ),
                            "feature_id": attr_defaults.get("feature_id"),
                            "workflow_id": attr_defaults.get("workflow_id"),
                            "operation_id": attr_defaults.get("operation_id"),
                            "attempt_id": attr_defaults.get("attempt_id"),
                            "model": model_name,
                            "input_tokens": m._estimate_input_tokens(call_kwargs),
                        },
                    )
                    await m._mark_stream_started_or_release_async(
                        enforcement=enforcement,
                        held_projection=held_projection,
                        org_id=step_ctx.org_id,
                        workspace_id=step_ctx.workspace_id,
                        run_id=step_ctx.run_id,
                        reserved_cost_usd=reserved_cost_usd,
                    )

                def _emit(attempt, settle_cost, tokens_used, error):
                    m._emit_stream_model_call(
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
                    reconcile_async_fn=m._try_settle_stream_budget_async,
                    pending_async_fn=m._try_mark_stream_pending_async,
                    postcall_cost_fn=m._bound_postcall(),
                    reservation_payload_fn=m._reservation_payload_fields,
                    emit_event_fn=_emit,
                    attribution=attr_defaults,
                    started_perf=time.perf_counter(),
                )

            wrap_manager = wrap_responses_async_manager if responses else wrap_openai_async_manager
            return wrap_manager(
                manager,
                attempt_factory=_attempt_factory,
                enter_scope=lambda: _helper_inner_create_bypass_async(
                    m._HELPER_INNER_CREATE_BYPASS, gate_scope
                ),
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
    m._block_unsupported_openai_surfaces(
        client, patches, keep={(id(obj), name) for obj, name, _ in patches}
    )
    m._reset_sdk_response_wrappers(obj for obj, _, _ in patches)

    def undo() -> None:
        for obj, name, orig in patches:
            try:
                setattr(obj, name, orig)
            except Exception:
                log.exception("kazenai.patch_undo_failed", target=str(obj), name=name)
        m._reset_sdk_response_wrappers(obj for obj, _, _ in patches)

    return undo


def patch_anthropic_async(
    client: Any,
    *,
    org_id: str,
    project_id: str,
    workspace_id: str,
    agent_id: str,
    agent_role: str,
    debug: bool,
    event_sink: Any,
    finops: Any,
    enforcement: Enforcement,
    capture_mode: Any,
    business_subject_ref: Optional[str],
    feature_id: Optional[str],
    workflow_id: Optional[str],
    feature: Optional[str],
    operation_id: Optional[str],
    attempt_id: Optional[str],
    stream_cutoff_usd: Optional[float] = None,
    usd_per_1k_tokens: Optional[float] = None,
    loop_block_threshold: float = 0.90,
    loop_detector: Any = None,
    endpoint_url: Optional[str] = None,
    endpoint_headers: Optional[Dict[str, str]] = None,
    certified_surface: bool = True,
    allow_unmetered: Sequence[str] = (),
    observe_unbounded: Sequence[str] = (),
    require_customer_id: bool = False,
) -> Callable[[], None]:
    m = _monitor_mod()
    from .attribution import assert_customer_id, use_attribution
    from .capture_policy import build_content_ref, capture_disclosure
    from .debug import DebugPrinter
    from .enforcement_owner import claim_configured_enforcement_owner as claim_enforcement_owner
    from .finops import FinOpsController
    from .loop_detector import LoopDetector

    if structlog is not None:
        log = structlog.get_logger(__name__)
    else:
        log = _log
    loop_detector = loop_detector or LoopDetector()
    dbg = DebugPrinter(enabled=debug)
    del dbg  # parity with sync path pre_call optional
    attr_defaults = dict(
        business_subject_ref=business_subject_ref,
        feature_id=feature_id or feature,
        workflow_id=workflow_id,
        operation_id=operation_id,
        attempt_id=attempt_id,
    )
    ctx0 = get_current_context() or RunContext.new(
        org_id=org_id,
        project_id=project_id,
        workspace_id=workspace_id,
        agent_id=agent_id,
    )
    patches: List[Tuple[Any, str, Any]] = []
    # Backstop for references taken before monitor(): see _dispatch_gate.
    gate = install_dispatch_gate(client, patches, allow_unmetered=allow_unmetered, is_async=True)
    gate_scope = gate.authorized if gate is not None else nullcontext

    def _wrap(fn: Callable[..., Any], *, method: str) -> Callable[..., Any]:
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            if m._HELPER_INNER_CREATE_BYPASS.get():
                return await fn(*args, **kwargs)
            ctx = m._call_context(ctx0)
            step_ctx = ctx.child_step()
            with ExitStack() as stack:
                stack.enter_context(use_context(step_ctx))
                stack.enter_context(claim_enforcement_owner())
                stack.enter_context(use_attribution(**attr_defaults))
                assert_customer_id(require_customer_id)
                payload = m._extract_openai_input(args, kwargs)
                m._reject_streaming_if_unsupported(kwargs)
                m._apply_output_token_bound(kwargs)
                call_kwargs = dict(kwargs)
                is_stream = bool(call_kwargs.get("stream"))
                model_name = str(call_kwargs.get("model") or "")
                cutoff = stream_cutoff_usd
                output_rate = 0.0
                if is_stream:
                    from .streaming.lifecycle import StreamAttempt, StreamOutcome
                    from .streaming.anthropic import wrap_anthropic_async_event_stream

                    output_rate = m._stream_output_rate_usd_per_1k(
                        model_name, usd_per_1k_tokens
                    )
                    if cutoff is not None and output_rate <= 0:
                        raise UnsupportedModeError(
                            "stream_cutoff_usd requires known model pricing or an explicit "
                            "usd_per_1k_tokens rate",
                            error_code="stream_cutoff_pricing_unavailable",
                        )

                projected = await _priced_projection(
                    m,
                    call_kwargs,
                    usd_per_1k_tokens,
                    provider='anthropic',
                    surface="messages",
                    client=client,
                    org_id=step_ctx.org_id,
                    observe_unbounded=observe_unbounded,
                )
                held_projection, reserved_cost_usd = await m._reserve_with_local_hold_async(
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
                        "input_tokens": m._estimate_input_tokens(call_kwargs),
                    },
                )
                await m._mark_stream_started_or_release_async(
                    enforcement=enforcement,
                    held_projection=held_projection,
                    org_id=step_ctx.org_id,
                    workspace_id=step_ctx.workspace_id,
                    run_id=step_ctx.run_id,
                    reserved_cost_usd=reserved_cost_usd,
                )
                attempt = None
                if is_stream:

                    def _emit(attempt, settle_cost, tokens_used, error):
                        m._emit_stream_model_call(
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
                        surface=m.SUPPORTED_ANTHROPIC_STREAM_METHOD,
                        step_ctx=step_ctx,
                        enforcement=enforcement,
                        held_projection=held_projection,
                        reserved_cost_usd=reserved_cost_usd,
                        model=model_name,
                        method=m.SUPPORTED_ANTHROPIC_STREAM_METHOD,
                        usd_per_1k_tokens=output_rate,
                        settle_usd_per_1k_tokens=usd_per_1k_tokens,
                        capture_mode=capture_mode,
                        agent_role=agent_role,
                        event_sink=event_sink,
                        finops=finops,
                        stream_cutoff_usd=cutoff,
                        reconcile_async_fn=m._try_settle_stream_budget_async,
                        pending_async_fn=m._try_mark_stream_pending_async,
                        postcall_cost_fn=m._bound_postcall(),
                        reservation_payload_fn=m._reservation_payload_fields,
                        emit_event_fn=_emit,
                        attribution=attr_defaults,
                        started_perf=time.perf_counter(),
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
                            model=model_name,
                            operation_id=attr_defaults.get("operation_id"),
                            attempt_id=attr_defaults.get("attempt_id"),
                            workflow_id=attr_defaults.get("workflow_id"),
                            feature_id=attr_defaults.get("feature_id"),
                            business_subject_ref=attr_defaults.get("business_subject_ref"),
                            provider="anthropic",
                            reservation_id=getattr(reserved_cost_usd, "reservation_id", None),
                            decision_id=getattr(reserved_cost_usd, "decision_id", None),
                        ),
                    ), gate_scope():
                        resp = await fn(*args, **call_kwargs)
                except Exception as exc:
                    rejected = m._definitive_provider_rejection(exc)
                    if attempt is not None:
                        await attempt.finalize_async(
                            StreamOutcome.REJECTED if rejected else StreamOutcome.PROVIDER_ERROR,
                            error=exc,
                        )
                    else:
                        # A definitive 4xx refusal settles at exactly $0 and releases the hold.
                        if reserved_cost_usd is not None:
                            await m._try_mark_stream_pending_async(
                                org_id=step_ctx.org_id,
                                workspace_id=step_ctx.workspace_id,
                                run_id=step_ctx.run_id,
                                reserved_cost_usd=reserved_cost_usd,
                                actual_cost_usd=0.0 if rejected else None,
                            )
                        enforcement.record_call(
                            cost_usd=0.0, released_projection=held_projection
                        )
                    raise
                except BaseException as cancel_exc:
                    # CancelledError after dispatch — finalize or mark pending.
                    if attempt is not None:
                        try:
                            from .streaming.lifecycle import StreamOutcome as _SO

                            await attempt.finalize_async(
                                _SO.CLIENT_CANCELLED, error=cancel_exc
                            )
                        except Exception:
                            pass
                    elif reserved_cost_usd is not None:
                        try:
                            await m._try_mark_stream_pending_async(
                                org_id=step_ctx.org_id,
                                workspace_id=step_ctx.workspace_id,
                                run_id=step_ctx.run_id,
                                reserved_cost_usd=reserved_cost_usd,
                            )
                        except Exception:
                            pass
                    raise

                if is_stream:
                    assert attempt is not None
                    return m._wrap_stream_result(resp, attempt, wrap_anthropic_async_event_stream)

                tokens_used, usage = m._extract_anthropic_usage(resp)
                cost_usd = m._postcall_cost_usd(
                    model=model_name,
                    usage=usage,
                    tokens_used=tokens_used,
                    usd_per_1k_tokens=usd_per_1k_tokens,
                )
                if reserved_cost_usd is not None:
                    if cost_usd is None or m._pricing_holds_reconciliation():
                        await m._try_mark_stream_pending_async(
                            org_id=step_ctx.org_id,
                            workspace_id=step_ctx.workspace_id,
                            run_id=step_ctx.run_id,
                            reserved_cost_usd=reserved_cost_usd,
                        )
                    else:
                        try:
                            settled = await m._try_reconcile_budget_async(
                                org_id=step_ctx.org_id,
                                workspace_id=step_ctx.workspace_id,
                                run_id=step_ctx.run_id,
                                reserved_cost_usd=reserved_cost_usd,
                                actual_cost_usd=float(cost_usd),
                            )
                        except BaseException as settle_exc:
                            try:
                                await m._try_mark_stream_pending_async(
                                    org_id=step_ctx.org_id,
                                    workspace_id=step_ctx.workspace_id,
                                    run_id=step_ctx.run_id,
                                    reserved_cost_usd=reserved_cost_usd,
                                )
                            except Exception:
                                pass
                            if isinstance(settle_exc, Exception):
                                settled = False
                            else:
                                raise
                        if settled is not True:
                            try:
                                await m._try_mark_stream_pending_async(
                                    org_id=step_ctx.org_id,
                                    workspace_id=step_ctx.workspace_id,
                                    run_id=step_ctx.run_id,
                                    reserved_cost_usd=reserved_cost_usd,
                                    actual_cost_usd=float(cost_usd),
                                )
                            except Exception:
                                pass
                enforcement.record_call(
                    cost_usd=cost_usd, released_projection=held_projection
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
                            **m._reservation_payload_fields(reserved_cost_usd),
                            **m._attribution_payload_fields(),
                            "method": method,
                            "model": model_name,
                            "inputs_ref": build_content_ref(
                                payload, mode=capture_mode, role="inputs"
                            ),
                            "outputs_ref": build_content_ref(
                                m._safe_output(resp), mode=capture_mode, role="outputs"
                            ),
                            "capture": capture_disclosure(capture_mode),
                            "usage": usage if isinstance(usage, dict) else None,
                            "async": True,
                        },
                    )
                except Exception:
                    log.exception("kazenai.event_build_failed")
                    return resp
                if event_sink is not None:
                    try:
                        event_sink.emit(event)
                    except Exception as exc:
                        m._log_fail_open("event_sink_emit", exc)
                if finops is not None:
                    derived = finops.handle_llm_call(event)
                    for key, ev in derived.items():
                        if key.startswith("_"):
                            continue
                        try:
                            if event_sink is not None:
                                event_sink.emit(ev)
                        except Exception as exc:
                            m._log_fail_open("event_sink_emit_derived", exc)
                    FinOpsController.raise_if_blocked(derived)
                return resp

        return wrapped

    if not (
        hasattr(client, "messages") and hasattr(client.messages, "create")
    ):
        raise TypeError("Unsupported async client: expected `.messages.create`")
    orig = client.messages.create
    patches.append((client.messages, "create", orig))
    wrapped_fn = _wrap(orig, method=m.SUPPORTED_ANTHROPIC_METHOD)
    try:
        setattr(wrapped_fn, "__kazenai_wrapped__", True)
    except Exception:
        pass
    setattr(client.messages, "create", wrapped_fn)

    try:
        if hasattr(client.messages, "stream") and callable(client.messages.stream):
            orig_stream = client.messages.stream

            def _stream_manager(*s_args: Any, **s_kwargs: Any) -> Any:
                from .streaming.lifecycle import StreamAttempt
                from .streaming.anthropic import wrap_anthropic_async_manager

                m._reject_streaming_if_unsupported({"stream": True})
                ctx = m._call_context(ctx0)
                step_ctx = ctx.child_step()
                call_kwargs = dict(s_kwargs)
                m._apply_output_token_bound(call_kwargs)
                model_name = str(call_kwargs.get("model") or "")
                output_rate = m._stream_output_rate_usd_per_1k(
                    model_name, usd_per_1k_tokens
                )
                cutoff = stream_cutoff_usd
                if cutoff is not None and output_rate <= 0:
                    raise UnsupportedModeError(
                        "stream_cutoff_usd requires known model pricing or an explicit "
                        "usd_per_1k_tokens rate",
                        error_code="stream_cutoff_pricing_unavailable",
                    )
                manager = orig_stream(*s_args, **call_kwargs)

                async def _attempt_factory() -> StreamAttempt:
                    projected = await _priced_projection(
                        m,
                        call_kwargs,
                        usd_per_1k_tokens,
                        provider="anthropic",
                        surface="messages",
                        client=client,
                        org_id=step_ctx.org_id,
                        observe_unbounded=observe_unbounded,
                    )
                    with ExitStack() as stack:
                        stack.enter_context(use_context(step_ctx))
                        stack.enter_context(claim_enforcement_owner())
                        stack.enter_context(use_attribution(**attr_defaults))
                        assert_customer_id(require_customer_id)
                        held_projection, reserved_cost_usd = await m._reserve_with_local_hold_async(
                            enforcement=enforcement,
                            projected_cost_usd=projected,
                            reserve_kwargs={
                                "org_id": step_ctx.org_id,
                                "workspace_id": step_ctx.workspace_id,
                                "run_id": step_ctx.run_id,
                                "call_id": str(getattr(step_ctx, "step_id", None) or "")
                                or None,
                                "estimated_cost_usd": projected,
                                "business_subject_ref": attr_defaults.get(
                                    "business_subject_ref"
                                ),
                                "feature_id": attr_defaults.get("feature_id"),
                                "workflow_id": attr_defaults.get("workflow_id"),
                                "operation_id": attr_defaults.get("operation_id"),
                                "attempt_id": attr_defaults.get("attempt_id"),
                                "model": model_name,
                                "input_tokens": m._estimate_input_tokens(call_kwargs),
                            },
                        )
                        await m._mark_stream_started_or_release_async(
                            enforcement=enforcement,
                            held_projection=held_projection,
                            org_id=step_ctx.org_id,
                            workspace_id=step_ctx.workspace_id,
                            run_id=step_ctx.run_id,
                            reserved_cost_usd=reserved_cost_usd,
                        )

                    def _emit(attempt, settle_cost, tokens_used, error):
                        m._emit_stream_model_call(
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
                        surface="anthropic.messages.stream",
                        step_ctx=step_ctx,
                        enforcement=enforcement,
                        held_projection=held_projection,
                        reserved_cost_usd=reserved_cost_usd,
                        model=model_name,
                        method="anthropic.messages.stream",
                        usd_per_1k_tokens=output_rate,
                        settle_usd_per_1k_tokens=usd_per_1k_tokens,
                        capture_mode=capture_mode,
                        agent_role=agent_role,
                        event_sink=event_sink,
                        finops=finops,
                        stream_cutoff_usd=cutoff,
                        reconcile_async_fn=m._try_settle_stream_budget_async,
                        pending_async_fn=m._try_mark_stream_pending_async,
                        postcall_cost_fn=m._bound_postcall(),
                        reservation_payload_fn=m._reservation_payload_fields,
                        emit_event_fn=_emit,
                        attribution=attr_defaults,
                        started_perf=time.perf_counter(),
                    )

                return wrap_anthropic_async_manager(
                    manager,
                    attempt_factory=_attempt_factory,
                    enter_scope=lambda: _helper_inner_create_bypass_async(
                        m._HELPER_INNER_CREATE_BYPASS, gate_scope
                    ),
                )

            patches.append((client.messages, "stream", orig_stream))
            setattr(client.messages, "stream", _stream_manager)
    except Exception:
        pass

    m._reset_sdk_response_wrappers(obj for obj, _, _ in patches)

    def undo() -> None:
        for obj, name, orig in patches:
            try:
                setattr(obj, name, orig)
            except Exception:
                log.exception("kazenai.patch_undo_failed", target=str(obj), name=name)
        m._reset_sdk_response_wrappers(obj for obj, _, _ in patches)

    return undo
