"""Optional OpenTelemetry integration for KazenAI Control.

Ordinary ``import kazenai`` works without the OTel SDK. Install the optional
extra: ``pip install 'kazenai[otel]'``.

Library rules:
  - Never replace an application-owned TracerProvider / MeterProvider.
  - Never configure logging or create exporters on import.
  - Runtime export failure must not change admission/settle results.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, MutableMapping, Optional

_log = logging.getLogger(__name__)

# Pinned GenAI semantic-convention mapping for this release (not "latest experimental").
GENAI_CONVENTION_VERSION = "1.36.0"
GENAI_COMPAT_MAP = {
    # older inbound → current
    "gen_ai.system": "gen_ai.system",
    "llm.system": "gen_ai.system",
    "llm.vendor": "gen_ai.system",
    "gen_ai.request.model": "gen_ai.request.model",
    "llm.request.model": "gen_ai.request.model",
    "gen_ai.usage.prompt_tokens": "gen_ai.usage.input_tokens",
    "llm.usage.prompt_tokens": "gen_ai.usage.input_tokens",
    "gen_ai.usage.completion_tokens": "gen_ai.usage.output_tokens",
    "llm.usage.completion_tokens": "gen_ai.usage.output_tokens",
}

KAZENAI_ATTRS = (
    "kazenai.workflow.id",
    "kazenai.operation.id",
    "kazenai.call.id",
    "kazenai.attempt.id",
    "kazenai.attempt.number",
    "kazenai.budget.profile",
    "kazenai.budget.accounting_owner",
    "kazenai.budget.lifecycle_version",
    "kazenai.policy.version",
    "kazenai.decision.id",
    "kazenai.decision.result",
    "kazenai.economic.reservation_id",
    "kazenai.economic.state",
    "kazenai.economic.reserved_usd_micros",
    "kazenai.economic.settled_usd_micros",
    "kazenai.economic.cost_confidence",
    "kazenai.business.subject_ref",
    "kazenai.business.feature_id",
)


@dataclass
class TelemetryCounters:
    export_dropped: int = 0
    export_failed: int = 0
    spans_started: int = 0


_COUNTERS = TelemetryCounters()


def telemetry_counters() -> TelemetryCounters:
    return _COUNTERS


def otel_available() -> bool:
    try:
        import opentelemetry.trace  # noqa: F401

        return True
    except ImportError:
        return False


def otel_enabled() -> bool:
    if os.getenv("OTEL_SDK_DISABLED", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if os.getenv("KAZENAI_OTEL_ENABLED", "1").strip().lower() in {"0", "false", "no"}:
        return False
    return True


def _get_tracer(name: str = "kazenai"):
    if not otel_enabled() or not otel_available():
        return None
    from opentelemetry import trace

    return trace.get_tracer(name, "1.1.1")


def sanitize_attributes(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Drop secrets and unbounded prompt/tool content before export."""
    banned_sub = ("authorization", "api_key", "apikey", "password", "cookie", "bearer", "jwt")
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        lk = str(key).lower()
        if any(b in lk for b in banned_sub):
            continue
        if lk.endswith((".prompt", ".completion", ".messages", ".tool_arguments", ".input", ".output")):
            # Default message capture OFF.
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[str(key)] = value
        else:
            out[str(key)] = str(value)[:256]
    return out


def build_control_attributes(
    *,
    operation_id: str | None = None,
    call_id: str | None = None,
    attempt_id: str | None = None,
    attempt: int | None = None,
    workflow_id: str | None = None,
    reservation_id: str | None = None,
    decision_id: str | None = None,
    decision_result: str | None = None,
    economic_state: str | None = None,
    reserved_usd_micros: int | None = None,
    settled_usd_micros: int | None = None,
    cost_confidence: str | None = None,
    accounting_owner: str | None = None,
    lifecycle_version: str | None = None,
    business_subject_ref: str | None = None,
    feature_id: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "kazenai.genai_convention_version": GENAI_CONVENTION_VERSION,
    }
    if operation_id:
        attrs["kazenai.operation.id"] = operation_id
    if call_id:
        attrs["kazenai.call.id"] = call_id
    if attempt_id:
        attrs["kazenai.attempt.id"] = attempt_id
    if attempt is not None:
        attrs["kazenai.attempt.number"] = int(attempt)
    if workflow_id:
        attrs["kazenai.workflow.id"] = workflow_id
    if reservation_id:
        attrs["kazenai.economic.reservation_id"] = reservation_id
    if decision_id:
        attrs["kazenai.decision.id"] = decision_id
    if decision_result:
        attrs["kazenai.decision.result"] = decision_result
    if economic_state:
        attrs["kazenai.economic.state"] = economic_state
    if reserved_usd_micros is not None:
        attrs["kazenai.economic.reserved_usd_micros"] = int(reserved_usd_micros)
    if settled_usd_micros is not None:
        attrs["kazenai.economic.settled_usd_micros"] = int(settled_usd_micros)
    if cost_confidence:
        attrs["kazenai.economic.cost_confidence"] = cost_confidence
    if accounting_owner:
        attrs["kazenai.budget.accounting_owner"] = accounting_owner
    if lifecycle_version:
        attrs["kazenai.budget.lifecycle_version"] = lifecycle_version
    if business_subject_ref:
        attrs["kazenai.business.subject_ref"] = business_subject_ref
    if feature_id:
        attrs["kazenai.business.feature_id"] = feature_id
    if model:
        attrs["gen_ai.request.model"] = model
    if provider:
        attrs["gen_ai.system"] = provider
    if input_tokens is not None:
        attrs["gen_ai.usage.input_tokens"] = int(input_tokens)
    if output_tokens is not None:
        attrs["gen_ai.usage.output_tokens"] = int(output_tokens)
    return sanitize_attributes(attrs)


@contextmanager
def control_span(
    name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    kind: str = "internal",
) -> Iterator[Any]:
    """Duration span around a real operation. No-op when OTel disabled/unavailable."""
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return
    from opentelemetry.trace import SpanKind

    kind_map = {
        "internal": SpanKind.INTERNAL,
        "client": SpanKind.CLIENT,
        "server": SpanKind.SERVER,
        "producer": SpanKind.PRODUCER,
        "consumer": SpanKind.CONSUMER,
    }
    attrs = sanitize_attributes(attributes or {})
    try:
        span_cm = tracer.start_as_current_span(
            name,
            kind=kind_map.get(kind, SpanKind.INTERNAL),
            attributes=attrs,
        )
    except Exception:
        _COUNTERS.export_failed += 1
        _log.debug("control_span setup failed (non-fatal)", exc_info=True)
        yield None
        return
    with span_cm as span:
        _COUNTERS.spans_started += 1
        yield span


def record_span_event(span: Any, name: str, attributes: Mapping[str, Any] | None = None) -> None:
    if span is None:
        return
    try:
        span.add_event(name, attributes=sanitize_attributes(attributes or {}))
    except Exception:
        _COUNTERS.export_failed += 1


def inject_w3c_trace_headers(headers: MutableMapping[str, str]) -> None:
    """Inject W3C ``traceparent``/``tracestate`` into outbound HTTP headers (fail-open)."""
    if not otel_enabled() or not otel_available():
        return
    try:
        from opentelemetry import propagate

        propagate.inject(headers)
    except Exception:
        _COUNTERS.export_failed += 1
        _log.debug("W3C trace header inject failed (non-fatal)", exc_info=True)


def start_control_span(
    name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    kind: str = "internal",
    attach: bool = False,
) -> tuple[Any, Any]:
    """Start a span that outlives a ``with`` block (stream attempt lifetime).

    Returns ``(span, token)`` where ``token`` is the context detach token when
    ``attach=True``, or ``(span, None)`` when the span is kept alive without
    becoming the caller's active context (stream return must not leave
    ``kazenai.model_attempt`` attached across the consumer boundary).

    Caller must ``end_control_span``.
    """
    tracer = _get_tracer()
    if tracer is None:
        return None, None
    try:
        from opentelemetry.trace import SpanKind

        kind_map = {
            "internal": SpanKind.INTERNAL,
            "client": SpanKind.CLIENT,
            "server": SpanKind.SERVER,
            "producer": SpanKind.PRODUCER,
            "consumer": SpanKind.CONSUMER,
        }
        attrs = sanitize_attributes(attributes or {})
        span = tracer.start_span(name, kind=kind_map.get(kind, SpanKind.INTERNAL), attributes=attrs)
        token = None
        if attach:
            from opentelemetry import context as otel_context
            from opentelemetry.trace import set_span_in_context

            token = otel_context.attach(set_span_in_context(span))
        _COUNTERS.spans_started += 1
        return span, token
    except Exception:
        _COUNTERS.export_failed += 1
        _log.debug("start_control_span failed (non-fatal)", exc_info=True)
        return None, None


def end_control_span(span: Any, token: Any = None, *, error: BaseException | None = None) -> None:
    """End a span started by ``start_control_span`` and detach its context."""
    if span is None:
        return
    try:
        if error is not None:
            try:
                span.record_exception(error)
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR, str(error)[:200]))
            except Exception:
                pass
        span.end()
    except Exception:
        _COUNTERS.export_failed += 1
    if token is not None:
        try:
            from opentelemetry import context as otel_context

            otel_context.detach(token)
        except Exception:
            pass


def workflow_span(
    name: str = "kazenai.workflow",
    *,
    attributes: Mapping[str, Any] | None = None,
    kind: str = "internal",
) -> Any:
    """Duration span for a framework/workflow execution.

    Unlike short event-delivery spans, this context manager covers the full
    kickoff/invoke lifetime so workflow.start→end share one parent span.
    """
    return control_span(name, kind=kind, attributes=attributes)


def link_recovery_span(
    name: str,
    *,
    linked_trace_id: str | None = None,
    linked_span_id: str | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> Any:
    """Start a linked late-recovery span without inventing parentage.

    Returns a context manager (span) or nullcontext when OTel is unavailable.
    """
    tracer = _get_tracer()
    if tracer is None:
        return nullcontext(None)
    from opentelemetry.trace import Link, SpanContext, TraceFlags, NonRecordingSpan
    from opentelemetry import trace

    links = []
    if linked_trace_id and linked_span_id:
        try:
            tid = int(linked_trace_id, 16)
            sid = int(linked_span_id, 16)
            ctx = SpanContext(
                trace_id=tid,
                span_id=sid,
                is_remote=True,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )
            links.append(Link(ctx))
        except Exception:
            pass
    attrs = sanitize_attributes(attributes or {})
    return tracer.start_as_current_span(name, links=links, attributes=attrs)


def resolve_otlp_http_traces_endpoint(raw: str | None = None) -> str:
    """Normalize OTLP HTTP traces URL (4318 default; avoid missing/double /v1/traces)."""
    endpoint = (raw or os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
    if not endpoint:
        return ""
    # Explicit full traces path wins.
    if endpoint.rstrip("/").endswith("/v1/traces"):
        return endpoint.rstrip("/")
    # Base endpoint → append path once.
    return endpoint.rstrip("/") + "/v1/traces"


def default_otlp_http_base() -> str:
    """Default OTLP base by protocol: HTTP→4318, gRPC→4317."""
    proto = (
        os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL")
        or os.getenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL")
        or "http/protobuf"
    ).strip().lower()
    if proto == "grpc":
        return "http://localhost:4317"
    return "http://localhost:4318"
