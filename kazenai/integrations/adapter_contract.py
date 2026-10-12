"""Shared framework adapter contract (FINAL_1_b_final P8).

Framework layers supply execution context and Observe spans. Economic admission
stays at the guarded provider client or LiteLLM gateway hook — middleware must
not reserve again around an already guarded call.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Iterator, Optional

from ..context import RunContext, get_current_context, use_context
from ..enforcement_owner import configured_enforcement_owner, get_enforcement_owner
from ..schema import KazenEvent, new_id, now_ms

WIRE_VERSION = "1.2"
CORRELATION_ENVELOPE_VERSION = "kazenai.framework.correlation.v1"
EVENT_SCHEMA_VERSION = "1.2"


class CapabilityLevel(str, Enum):
    OBSERVE = "observe"
    CONTROL = "control"


class UnsupportedFrameworkPath(RuntimeError):
    """Control rejects an unverified framework/provider entry point before dispatch."""

    def __init__(self, message: str, *, code: str = "unsupported_framework_path") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class FrameworkOperation:
    org_id: str
    workspace_id: str
    project_id: str
    agent_id: str
    operation_id: str
    run_id: str
    framework: str
    capability: CapabilityLevel = CapabilityLevel.OBSERVE
    call_id: Optional[str] = None
    attempt_id: Optional[str] = None
    parent_step_id: Optional[str] = None
    accounting_owner: str = "sdk"
    metadata: dict[str, Any] = field(default_factory=dict)

    def child_call(self, *, call_site: str = "") -> "FrameworkOperation":
        return replace(
            self,
            call_id=f"call_{uuid.uuid4().hex}",
            attempt_id=f"att_{uuid.uuid4().hex}",
            parent_step_id=self.call_id or self.run_id,
            metadata={**self.metadata, "call_site": call_site},
        )


_OP: contextvars.ContextVar[Optional[FrameworkOperation]] = contextvars.ContextVar(
    "kazenai_framework_operation", default=None
)
_ATTACH_LOCK = threading.Lock()
_ATTACHED: set[int] = set()


def get_framework_operation() -> Optional[FrameworkOperation]:
    return _OP.get()


def new_operation(
    *,
    framework: str,
    org_id: str = "local",
    workspace_id: str = "default",
    project_id: str = "default",
    agent_id: str = "agent",
    operation_id: str | None = None,
    run_id: str | None = None,
    capability: CapabilityLevel | str = CapabilityLevel.OBSERVE,
    metadata: dict[str, Any] | None = None,
) -> FrameworkOperation:
    cap = capability if isinstance(capability, CapabilityLevel) else CapabilityLevel(str(capability))
    owner = configured_enforcement_owner() or "sdk"
    return FrameworkOperation(
        org_id=org_id,
        workspace_id=workspace_id or "default",
        project_id=project_id or "default",
        agent_id=agent_id,
        operation_id=(operation_id or "").strip() or f"op_{uuid.uuid4().hex}",
        run_id=(run_id or "").strip() or uuid.uuid4().hex,
        framework=framework,
        capability=cap,
        accounting_owner=owner,
        metadata=dict(metadata or {}),
    )


@contextlib.contextmanager
def enter_operation(op: FrameworkOperation) -> Iterator[FrameworkOperation]:
    """Bind framework operation + RunContext + AttributionContext; restore on exit."""
    from ..attribution import use_attribution

    token = _OP.set(op)
    ctx = RunContext.new(
        org_id=op.org_id,
        project_id=op.project_id,
        workspace_id=op.workspace_id,
        agent_id=op.agent_id,
        run_id=op.run_id,
        metadata={
            "operation_id": op.operation_id,
            "framework": op.framework,
            "capability": op.capability.value,
            "accounting_owner": op.accounting_owner,
            **op.metadata,
        },
    )
    try:
        with use_context(ctx):
            with use_attribution(operation_id=op.operation_id, attempt_id=op.attempt_id or None):
                yield op
    finally:
        _OP.reset(token)


def correlation_envelope(op: FrameworkOperation) -> dict[str, Any]:
    """Minimal versioned checkpoint-safe correlation (no secrets/clients/spans)."""
    return {
        "v": CORRELATION_ENVELOPE_VERSION,
        "org_id": op.org_id,
        "workspace_id": op.workspace_id,
        "project_id": op.project_id,
        "operation_id": op.operation_id,
        "run_id": op.run_id,
        "framework": op.framework,
        "accounting_owner": op.accounting_owner,
        "capability": op.capability.value,
    }


def restore_operation_from_envelope(envelope: dict[str, Any] | None) -> Optional[FrameworkOperation]:
    if not isinstance(envelope, dict):
        return None
    if envelope.get("v") != CORRELATION_ENVELOPE_VERSION:
        return None
    # Checkpoint metadata is untrusted for tenancy — caller must re-bind auth.
    return FrameworkOperation(
        org_id=str(envelope.get("org_id") or "local"),
        workspace_id=str(envelope.get("workspace_id") or "default"),
        project_id=str(envelope.get("project_id") or "default"),
        agent_id=str(envelope.get("agent_id") or "agent"),
        operation_id=str(envelope.get("operation_id") or f"op_{uuid.uuid4().hex}"),
        run_id=str(envelope.get("run_id") or uuid.uuid4().hex),
        framework=str(envelope.get("framework") or "unknown"),
        capability=CapabilityLevel(str(envelope.get("capability") or "observe")),
        accounting_owner=str(envelope.get("accounting_owner") or "sdk"),
        metadata={"restored_from_checkpoint": True},
    )


def emit_framework_event(
    *,
    op: FrameworkOperation,
    event_type: str,
    tokens_used: int | None = None,
    cost_usd: float | None = None,
    payload: dict[str, Any] | None = None,
    on_event: Callable[[KazenEvent], Any] | None = None,
) -> KazenEvent:
    body = {
        "framework": op.framework,
        "capability": op.capability.value,
        "operation_id": op.operation_id,
        "accounting_owner": op.accounting_owner,
        "wire_version": WIRE_VERSION,
        **(payload or {}),
    }
    # Never export prompts/tool args/secrets by default.
    for banned in ("messages", "prompt", "tool_arguments", "authorization", "api_key"):
        body.pop(banned, None)
    ev = KazenEvent(
        schema_version=EVENT_SCHEMA_VERSION,
        ts_ms=now_ms(),
        event_id=new_id(),
        org_id=op.org_id,
        project_id=op.project_id,
        workspace_id=op.workspace_id or "default",
        surface="kazenai-core",
        agent_id=op.agent_id,
        agent_role=op.framework,
        run_id=op.run_id,
        step_id=op.attempt_id or new_id(),
        parent_step_id=op.parent_step_id,
        event_type=event_type,
        tokens_used=tokens_used,
        cost_usd=cost_usd,
        payload=body,
    )
    try:
        from ..otel import build_control_attributes, control_span

        span_cm = control_span(
            f"kazenai.{event_type}",
            attributes=build_control_attributes(
                operation_id=op.operation_id,
                workflow_id=str((payload or {}).get("workflow_id") or "") or None,
                feature_id=str((payload or {}).get("feature_id") or "") or None,
            ),
        )
    except Exception:
        from contextlib import nullcontext

        span_cm = nullcontext(None)

    with span_cm:
        if on_event is not None:
            try:
                on_event(ev)
            except Exception:
                pass
        else:
            # U12: default delivery through spine sinks when no callback is supplied.
            try:
                from ..spine.guard import _emit_event

                _emit_event(ev, event_sink=None)
            except Exception:
                pass
    return ev


def mark_attached(obj: Any) -> bool:
    """Return True if newly attached; False if already attached (idempotent)."""
    key = id(obj)
    with _ATTACH_LOCK:
        if key in _ATTACHED:
            return False
        _ATTACHED.add(key)
        return True


def mark_detached(obj: Any) -> None:
    with _ATTACH_LOCK:
        _ATTACHED.discard(id(obj))


def is_monitored_client(client: Any) -> bool:
    """True only when ``monitor()`` marked the client or patched create."""
    if getattr(client, "_kazenai_monitored", False) is True:
        return True
    if getattr(client, "__kazenai_monitored__", False) is True:
        return True
    create = getattr(getattr(getattr(client, "chat", None), "completions", None), "create", None)
    if create is not None and getattr(create, "__kazenai_wrapped__", False) is True:
        return True
    messages_create = getattr(getattr(client, "messages", None), "create", None)
    if messages_create is not None and getattr(messages_create, "__kazenai_wrapped__", False) is True:
        return True
    return False


def require_control_client(client: Any, *, label: str = "provider client") -> None:
    """Control mode requires a monitored client or gateway-owned enforcement."""
    owner = configured_enforcement_owner() or (get_enforcement_owner().owner if get_enforcement_owner() else "sdk")
    if owner == "gateway":
        return
    if is_monitored_client(client):
        return
    raise UnsupportedFrameworkPath(
        f"Control requires a kazenai.monitor()-guarded {label} (or KAZENAI_ENFORCEMENT_OWNER=gateway)",
        code="unguarded_control_boundary",
    )


def reject_openai_responses_api(model_or_client: Any) -> None:
    """This release's Control envelope is Chat Completions / Anthropic Messages only."""
    name = type(model_or_client).__name__
    module = type(model_or_client).__module__ or ""
    text = f"{module}.{name}".lower()
    if "responses" in text and "openai" in text:
        raise UnsupportedFrameworkPath(
            "OpenAI Responses API is not Control-supported in this release; use Chat Completions",
            code="unsupported_provider_api",
        )
    # langchain-openai 1.7+ may infer Responses routing on ChatOpenAI; reject explicit opt-in.
    if getattr(model_or_client, "use_responses_api", False) is True:
        cls_name = type(model_or_client).__name__
        fields_set = getattr(model_or_client, "model_fields_set", None)
        if cls_name != "ChatOpenAI" or (fields_set is not None and "use_responses_api" in fields_set):
            raise UnsupportedFrameworkPath(
                "OpenAI Responses API binding rejected before dispatch",
                code="unsupported_provider_api",
            )
    if getattr(model_or_client, "_use_responses_api", False) is True:
        raise UnsupportedFrameworkPath(
            "OpenAI Responses API binding rejected before dispatch",
            code="unsupported_provider_api",
        )


def extract_chat_usage(response: Any) -> tuple[int | None, dict[str, Any]]:
    """Best-effort usage extraction; missing usage is unknown (None), not zero."""
    meta: dict[str, Any] = {"usage_provenance": "unknown"}
    usage = None
    for attr in ("usage_metadata", "usage", "response_metadata"):
        cand = getattr(response, attr, None)
        if cand is not None:
            usage = cand
            break
    if usage is None and isinstance(response, dict):
        usage = response.get("usage") or (response.get("llm_output") or {}).get("token_usage")
    if usage is None:
        return None, meta
    if isinstance(usage, dict):
        total = usage.get("total_tokens")
        if total is None:
            prompt = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            completion = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
            total = prompt + completion if (prompt or completion) else None
        meta["usage_provenance"] = "framework_or_provider"
        return (int(total) if total is not None else None), meta
    total = getattr(usage, "total_tokens", None)
    if total is None:
        prompt = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0)
        total = prompt + completion if (prompt or completion) else None
    meta["usage_provenance"] = "framework_or_provider"
    return (int(total) if total is not None else None), meta


def active_or_new_operation(**kwargs: Any) -> FrameworkOperation:
    current = get_framework_operation()
    if current is not None and kwargs.get("reuse_active", True):
        return current
    kwargs.pop("reuse_active", None)
    return new_operation(**kwargs)


def bind_run_context_from_op(op: FrameworkOperation) -> RunContext:
    existing = get_current_context()
    if existing and existing.run_id == op.run_id:
        return existing
    return RunContext.new(
        org_id=op.org_id,
        project_id=op.project_id,
        workspace_id=op.workspace_id,
        agent_id=op.agent_id,
        run_id=op.run_id,
        metadata={"operation_id": op.operation_id, "framework": op.framework},
    )
