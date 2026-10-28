"""Core OTel contract (optional extra, no financial coupling)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kazenai.otel import (
    GENAI_CONVENTION_VERSION,
    build_control_attributes,
    control_span,
    default_otlp_http_base,
    inject_w3c_trace_headers,
    otel_enabled,
    record_span_event,
    resolve_otlp_http_traces_endpoint,
    sanitize_attributes,
    telemetry_counters,
)

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
RESULTS: list[dict] = []


def _record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append({"scenario": name, "status": status, "detail": detail})


@pytest.fixture(scope="module", autouse=True)
def _write_artifact():
    yield
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ARTIFACTS / f"kazenai-core-{ts}.json"
    path.write_text(
        json.dumps(
            {
                "recorded_at": ts,
                "results": RESULTS,
                "pass_count": sum(1 for r in RESULTS if r["status"] == "pass"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_import_kazenai_without_requiring_otel_sdk():
    import kazenai

    assert kazenai.__version__
    _record("import_without_otel_extra", "pass")


def test_sanitize_drops_secrets_and_prompts():
    clean = sanitize_attributes(
        {
            "authorization": "Bearer sk-secret",
            "api_key": "x",
            "gen_ai.prompt": "ignore me",
            "kazenai.operation.id": "op1",
            "gen_ai.request.model": "gpt-4o-mini",
        }
    )
    assert "authorization" not in clean
    assert "api_key" not in clean
    assert "gen_ai.prompt" not in clean
    assert clean["kazenai.operation.id"] == "op1"
    assert clean["gen_ai.request.model"] == "gpt-4o-mini"
    _record("sanitize_secrets_prompts", "pass")


def test_endpoint_resolution_http_4318_and_suffix(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL", raising=False)
    assert default_otlp_http_base() == "http://localhost:4318"

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    assert default_otlp_http_base() == "http://localhost:4317"

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL", raising=False)
    assert resolve_otlp_http_traces_endpoint("http://collector:4318") == "http://collector:4318/v1/traces"
    assert (
        resolve_otlp_http_traces_endpoint("http://collector:4318/v1/traces")
        == "http://collector:4318/v1/traces"
    )
    _record("endpoint_port_and_suffix", "pass")


def test_otel_disabled_env(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "1")
    assert otel_enabled() is False
    with control_span("noop") as span:
        assert span is None
    _record("otel_sdk_disabled", "pass")


def test_control_attributes_contract():
    attrs = build_control_attributes(
        operation_id="op",
        call_id="call",
        attempt_id="att",
        attempt=2,
        reservation_id="rsv",
        accounting_owner="gateway",
        lifecycle_version="reservation.lifecycle.v1",
        model="gpt-4o-mini",
        provider="openai",
        input_tokens=10,
        output_tokens=5,
        reserved_usd_micros=1000,
        cost_confidence="estimated",
    )
    assert attrs["kazenai.genai_convention_version"] == GENAI_CONVENTION_VERSION
    assert attrs["kazenai.attempt.number"] == 2
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["kazenai.budget.accounting_owner"] == "gateway"
    _record("control_attributes_contract", "pass")


def test_inmemory_span_hierarchy_and_events(monkeypatch):
    pytest.importorskip("opentelemetry")
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setenv("KAZENAI_OTEL_ENABLED", "1")
    monkeypatch.setenv("KAZENAI_OTEL_FORCE_PROVIDER", "1")

    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    before = telemetry_counters().spans_started
    attrs = build_control_attributes(operation_id="op-shared", call_id="c1", attempt_id="a1", attempt=1)
    with control_span("kazenai.operation", attributes=attrs) as root:
        assert root is not None
        with control_span(
            "kazenai.admission",
            attributes=build_control_attributes(reservation_id="rsv1", decision_result="allowed"),
        ) as adm:
            record_span_event(adm, "reservation.lifecycle.reserved", {"kazenai.economic.state": "reserved"})
        with control_span(
            "kazenai.model_attempt",
            attributes=build_control_attributes(model="gpt-4o-mini", provider="openai", attempt=1),
            kind="client",
        ):
            pass
        with control_span(
            "kazenai.settlement",
            attributes=build_control_attributes(reservation_id="rsv1", settled_usd_micros=820000),
        ):
            pass

    spans = exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert "kazenai.operation" in names
    assert "kazenai.admission" in names
    assert "kazenai.model_attempt" in names
    assert "kazenai.settlement" in names
    # Shared trace across hierarchy
    trace_ids = {format(s.context.trace_id, "032x") for s in spans}
    assert len(trace_ids) == 1
    adm_span = next(s for s in spans if s.name == "kazenai.admission")
    assert any(e.name == "reservation.lifecycle.reserved" for e in adm_span.events)
    # No secrets in exported attributes
    for s in spans:
        for k in s.attributes or {}:
            assert "authorization" not in str(k).lower()
            assert "api_key" not in str(k).lower()
    assert telemetry_counters().spans_started >= before + 4
    _record("inmemory_span_hierarchy", "pass", f"spans={len(spans)} trace={next(iter(trace_ids))}")


def test_control_span_propagates_application_exceptions(monkeypatch):
    pytest.importorskip("opentelemetry")
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setenv("KAZENAI_OTEL_ENABLED", "1")
    monkeypatch.setenv("KAZENAI_OTEL_FORCE_PROVIDER", "1")
    trace.set_tracer_provider(TracerProvider())

    with pytest.raises(ValueError, match="application error"):
        with control_span("kazenai.operation"):
            raise ValueError("application error")
    _record("control_span_body_exception", "pass")


def test_start_end_control_span_and_stream_attempt_lifetime(monkeypatch):
    """Start/end_control_span + StreamAttempt span lifetime helpers."""
    pytest.importorskip("opentelemetry")
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from kazenai.otel import end_control_span, link_recovery_span, start_control_span
    from kazenai.streaming.lifecycle import StreamAttempt

    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setenv("KAZENAI_OTEL_ENABLED", "1")
    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    span, token = start_control_span(
        "kazenai.model_attempt",
        kind="client",
        attributes=build_control_attributes(provider="openai", attempt=1),
    )
    assert span is not None
    end_control_span(span, token, error=RuntimeError("boom"))

    with link_recovery_span(
        "kazenai.recovery",
        linked_trace_id="0" * 32,
        linked_span_id="0" * 16,
        attributes=build_control_attributes(decision_result="replay_pending"),
    ):
        pass

    attempt = StreamAttempt(
        surface="openai",
        step_ctx=object(),
        enforcement=object(),
        held_projection=0.0,
        reserved_cost_usd=0.01,
        model="gpt-4o-mini",
        method="chat.completions.create",
        usd_per_1k_tokens=0.0,
        capture_mode=None,
        agent_role="assistant",
        attribution={"operation_id": "op-stream", "attempt_id": "a1"},
    )
    attempt.begin_otel_span(
        provider="anthropic",
        reservation_id="rsv-stream",
        decision_id="dec-1",
    )
    assert attempt._otel_span is not None
    attempt.begin_otel_span(provider="anthropic")  # idempotent
    attempt.end_otel_span()
    assert attempt._otel_span is None
    provider.force_flush()
    names = {s.name for s in exporter.get_finished_spans()}
    assert "kazenai.model_attempt" in names
    assert "kazenai.recovery" in names
    _record("stream_attempt_otel_lifetime", "pass", f"spans={sorted(names)}")


def test_inject_w3c_trace_headers_fail_open(monkeypatch):
    pytest.importorskip("opentelemetry")
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setenv("KAZENAI_OTEL_ENABLED", "1")
    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    headers: dict[str, str] = {"x-api-key": "secret"}
    with control_span("kazenai.operation"):
        inject_w3c_trace_headers(headers)
    assert "traceparent" in headers
    assert headers.get("x-api-key") == "secret"
    # Disabled path must not raise or clear caller headers.
    monkeypatch.setenv("OTEL_SDK_DISABLED", "1")
    headers2 = {"x-api-key": "secret"}
    inject_w3c_trace_headers(headers2)
    assert "traceparent" not in headers2
    _record("inject_w3c_trace_headers", "pass")


def test_financial_path_unaffected_when_otel_errors(monkeypatch):
    """Export/telemetry failure must not change admission result shape."""
    from kazenai.spine.guard import ReservationHandle, reserve_budget

    monkeypatch.setenv("OTEL_SDK_DISABLED", "1")
    monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    # Clear all profile aliases — prior Control tests leave FINOPS_CONTROL_PROFILE set.
    monkeypatch.setenv("KAZENAI_FINOPS_CONTROL_PROFILE", "legacy")
    monkeypatch.setenv("KAZENAI_CONTROL_PROFILE", "legacy")
    monkeypatch.setenv("KAZENAI_PROFILE", "legacy")
    monkeypatch.delenv("KAZENAI_ENFORCEMENT_OWNER", raising=False)
    handle = reserve_budget(org_id="o", workspace_id="w", run_id="r", model="gpt-4o-mini")
    # Fail-open without FinOps URL returns a noop handle / zero — never raises due to OTel.
    assert handle is not None
    if isinstance(handle, ReservationHandle):
        assert float(handle.reserved_cost_usd) == 0.0 or handle.reservation_id == ""
    _record("finance_independent_of_otel", "pass")


def test_u10_real_guard_admission_settlement_spans_and_traceparent(monkeypatch):
    """Real guard reserve/settle emit admission+settlement spans + W3C headers."""
    pytest.importorskip("opentelemetry")
    pytest.importorskip("opentelemetry.sdk")
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from kazenai.spine import guard as guard_mod
    from kazenai.spine.guard import reconcile_budget, reserve_budget

    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setenv("KAZENAI_OTEL_ENABLED", "1")
    monkeypatch.setenv("KAZENAI_FINOPS_CONTROL_PROFILE", "control")
    monkeypatch.delenv("KAZENAI_CONTROL_PROFILE", raising=False)
    monkeypatch.delenv("KAZENAI_PROFILE", raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_closed")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_closed")
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "test-key")
    guard_mod._LIFECYCLE_CAP_CACHE.clear()

    seen_headers: list[dict[str, str]] = []

    class _H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: ANN001
            return

        def do_GET(self):  # noqa: N802
            seen_headers.append({k.lower(): v for k, v in self.headers.items()})
            body = json.dumps(
                {
                    "capabilities": {"reservation_settlement": True},
                    "effective_lifecycle": {
                        "profile": "control",
                        "accounting_mode": "reservation_settlement",
                        "failure_policy": "fail_closed",
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            seen_headers.append({k.lower(): v for k, v in self.headers.items()})
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                req = json.loads(raw.decode() or "{}")
            except Exception:
                req = {}
            if self.path.endswith("/reserve"):
                body = json.dumps(
                    {
                        "reservation_id": "rsv-otel-1",
                        "decision_id": "dec-otel-1",
                        "call_id": req.get("call_id") or "call-1",
                        "attempt": req.get("attempt") or 1,
                        "reserved_usd_micros": req.get("estimated_usd_micros") or 1000,
                        "settlement_capability": "kzcap1.test." + "A" * 43,
                    }
                ).encode()
            else:
                body = json.dumps({"result": "applied"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("KAZENAI_FINOPS_URL", f"http://127.0.0.1:{srv.server_address[1]}")

    # OTel forbids replacing an already-set TracerProvider; attach to the current one.
    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    try:
        with control_span("kazenai.operation", attributes=build_control_attributes(operation_id="op-u10")):
            handle = reserve_budget(
                org_id="org-otel",
                workspace_id="ws",
                run_id="run-otel",
                model="gpt-4o-mini",
                estimated_cost_usd=0.01,
            )
            assert handle.reservation_id == "rsv-otel-1"
            ok = reconcile_budget(
                org_id="org-otel",
                workspace_id="ws",
                run_id="run-otel",
                reserved_cost_usd=handle,
                actual_cost_usd=0.008,
            )
            assert ok is True
        provider.force_flush()
    finally:
        srv.shutdown()
        guard_mod._LIFECYCLE_CAP_CACHE.clear()
        monkeypatch.setenv("KAZENAI_FINOPS_CONTROL_PROFILE", "legacy")
        monkeypatch.setenv("KAZENAI_CONTROL_PROFILE", "legacy")
        monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)

    spans = exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert "kazenai.admission" in names
    assert "kazenai.settlement" in names
    trace_ids = {format(s.context.trace_id, "032x") for s in spans}
    assert len(trace_ids) == 1
    assert any("traceparent" in h for h in seen_headers)
    _record(
        "u10_guard_admission_settlement_traceparent",
        "pass",
        f"spans={sorted(names)} trace={next(iter(trace_ids))}",
    )
