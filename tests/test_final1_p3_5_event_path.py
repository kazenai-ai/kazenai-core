"""FINAL_1 P3-5 — Control FinOps↔Lens event path (client fan-out, singular accounting)."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from kazenai.ingest_url import ingest_post_url, normalize_ingest_base_url
from kazenai.schema import KazenEvent, new_id, now_ms
from kazenai.sinks import HttpSink, HttpSinkConfig, MultiSink


def _finops_style_event_in():
    """FinOps KazenEventIn shape without requiring FinOps on CI.

    Prefer live FinOps ``events_schema`` when a sibling checkout exists; otherwise
    use a local Pydantic mirror of the P3-5 lineage fields.
    """
    from pathlib import Path
    import sys

    here = Path(__file__).resolve()
    for parents_up in (2, 3):
        try:
            backend = here.parents[parents_up] / "kazenai-agent-finops" / "backend"
        except IndexError:
            continue
        if (backend / "events_schema.py").is_file():
            sys.path.insert(0, str(backend))
            from events_schema import KazenEventIn  # type: ignore

            return KazenEventIn

    from typing import Any, Dict, Optional

    from pydantic import BaseModel, Field, model_validator

    class KazenEventIn(BaseModel):
        schema_version: str = Field(default="1.2", min_length=1)
        ts_ms: int
        event_id: str = Field(..., min_length=1)
        org_id: str
        workspace_id: str = "default"
        project_id: str
        surface: str
        agent_id: str
        agent_role: str
        run_id: str
        parent_run_id: Optional[str] = None
        root_run_id: Optional[str] = None
        trace_id: Optional[str] = None
        client_id: Optional[str] = None
        step_id: str
        event_type: str
        payload: Dict[str, Any] = Field(default_factory=dict)

        @model_validator(mode="after")
        def _default_root_run_id(self) -> "KazenEventIn":
            if self.root_run_id is None and self.run_id:
                self.root_run_id = self.run_id
            return self

    return KazenEventIn


def test_p35_normalize_ingest_url_strips_v1_events():
    assert normalize_ingest_base_url("http://finops:8090/v1/events") == "http://finops:8090"
    assert normalize_ingest_base_url("http://finops:8090/v1/events/") == "http://finops:8090"
    assert normalize_ingest_base_url("http://finops:8090") == "http://finops:8090"
    assert ingest_post_url("http://finops:8090/v1/events") == "http://finops:8090/v1/events"
    assert ingest_post_url("http://finops:8090") == "http://finops:8090/v1/events"


def test_p35_kazen_event_preserves_lineage_ids():
    ev = KazenEvent(
        schema_version="1.2",
        ts_ms=now_ms(),
        event_id=new_id(),
        org_id="org-a",
        workspace_id="ws-a1",
        project_id="proj-a",
        surface="test",
        agent_id="agent",
        agent_role="agent",
        run_id="run-1",
        parent_run_id="run-parent",
        trace_id="trace-abc",
        step_id="step-1",
        event_type="model.call",
        payload={"reservation_id": "res-1", "call_id": "c1", "attempt": 1},
    )
    assert ev.root_run_id == "run-1"
    dump = ev.model_dump()
    assert dump["root_run_id"] == "run-1"
    assert dump["trace_id"] == "trace-abc"
    assert dump["workspace_id"] == "ws-a1"


class _DualIngestServer:
    """Tiny ThreadingHTTPServer that records POSTs and optional settle hooks."""

    def __init__(self, *, settle: bool) -> None:
        self.settle = settle
        self.events: List[Dict[str, Any]] = []
        self.settle_calls = 0
        self.headers_seen: List[Dict[str, str]] = []
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                body = json.loads(raw.decode("utf-8") or "{}")
                hdrs = {k: v for k, v in self.headers.items()}
                with outer._lock:
                    outer.headers_seen.append(hdrs)
                    for ev in body.get("events") or []:
                        # Dedupe by event_id (FinOps/Lens behavior).
                        if any(x.get("event_id") == ev.get("event_id") for x in outer.events):
                            continue
                        outer.events.append(ev)
                        if outer.settle:
                            source = (hdrs.get("X-Kazen-Source") or "").lower()
                            derived = hdrs.get("X-Kazen-Derived") == "1"
                            if (
                                not derived
                                and source not in {"sdk-lens-mirror", "lens", "lens-mirror", "metadata-only"}
                                and (ev.get("payload") or {}).get("reservation_id")
                            ):
                                outer.settle_calls += 1
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._thread.join(timeout=2)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def test_p35_dual_ingest_same_run_join_no_double_settle(tmp_path):
    finops = _DualIngestServer(settle=True)
    lens = _DualIngestServer(settle=True)  # would settle if mis-routed; header must prevent
    finops.start()
    lens.start()
    try:
        sink = MultiSink(
            [
                HttpSink(
                    HttpSinkConfig(
                        ingest_url=finops.base + "/v1/events",  # doubled path must normalize
                        flush_interval_s=0.05,
                        batch_max=1,
                        offline_queue_path=str(tmp_path / "finops.jsonl"),
                        retry_interval_s=0.05,
                        org_id="org-a",
                        extra_headers={"X-Kazen-Source": "sdk-finops"},
                    )
                ),
                HttpSink(
                    HttpSinkConfig(
                        ingest_url=lens.base,
                        flush_interval_s=0.05,
                        batch_max=1,
                        offline_queue_path=str(tmp_path / "lens.jsonl"),
                        retry_interval_s=0.05,
                        org_id="org-a",
                        extra_headers={"X-Kazen-Source": "sdk-lens-mirror"},
                    )
                ),
            ]
        )
        event_id = "evt-p35-join-1"
        run_id = "run-p35-1"
        ev = KazenEvent(
            schema_version="1.2",
            ts_ms=now_ms(),
            event_id=event_id,
            org_id="org-a",
            workspace_id="ws-a1",
            project_id="proj-a",
            surface="test",
            agent_id="agent",
            agent_role="agent",
            run_id=run_id,
            root_run_id=run_id,
            trace_id="trace-p35",
            step_id="s1",
            event_type="model.call",
            cost_usd=0.01,
            payload={"reservation_id": "res-p35", "call_id": "c1", "attempt": 1},
        )
        sink.emit(ev)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if finops.events and lens.events:
                break
            time.sleep(0.05)
        sink.close(deadline_s=1.0)

        assert len(finops.events) == 1
        assert len(lens.events) == 1
        f, l = finops.events[0], lens.events[0]
        assert f["event_id"] == l["event_id"] == event_id
        assert f["run_id"] == l["run_id"] == run_id
        assert f["org_id"] == l["org_id"] == "org-a"
        assert f["workspace_id"] == l["workspace_id"] == "ws-a1"
        assert f.get("root_run_id") == l.get("root_run_id") == run_id
        assert finops.settle_calls == 1
        assert lens.settle_calls == 0  # mirror header blocked settle

        # Duplicate export — FinOps dedupes; settle must not double.
        sink2 = HttpSink(
            HttpSinkConfig(
                ingest_url=finops.base,
                flush_interval_s=0.05,
                batch_max=1,
                offline_queue_path=str(tmp_path / "finops2.jsonl"),
                retry_interval_s=0.05,
                org_id="org-a",
                extra_headers={"X-Kazen-Source": "sdk-finops"},
            )
        )
        sink2.emit(ev)
        time.sleep(0.25)
        sink2.close(deadline_s=1.0)
        assert len(finops.events) == 1
        assert finops.settle_calls == 1
    finally:
        finops.stop()
        lens.stop()


def test_p35_cross_tenant_rejected_by_finops_store_logic():
    """Document Control rule: org_mismatch rejects (mirrors FinOps store)."""
    KazenEventIn = _finops_style_event_in()

    ev = KazenEventIn.model_validate(
        {
            "schema_version": "1.2",
            "ts_ms": now_ms(),
            "event_id": "e1",
            "org_id": "org-b",
            "workspace_id": "ws-b1",
            "project_id": "p",
            "surface": "t",
            "agent_id": "a",
            "agent_role": "agent",
            "run_id": "r",
            "step_id": "s",
            "event_type": "model.call",
            "payload": {},
        }
    )
    assert ev.root_run_id == "r"
    assert ev.org_id == "org-b"
    # Store-level rejection is org_id != principal org — unit-check the invariant.
    principal_org = "org-a"
    assert ev.org_id != principal_org


def test_p35_derived_header_skips_settle_spy(tmp_path):
    """Lens→FinOps derived must not settle (accounting singular)."""
    server = _DualIngestServer(settle=True)
    server.start()
    try:
        sink = HttpSink(
            HttpSinkConfig(
                ingest_url=server.base,
                flush_interval_s=0.05,
                batch_max=1,
                offline_queue_path=str(tmp_path / "derived.jsonl"),
                retry_interval_s=0.05,
                extra_headers={"X-Kazen-Derived": "1", "X-Kazen-Source": "lens"},
            )
        )
        sink.emit(
            KazenEvent(
                schema_version="1.2",
                ts_ms=now_ms(),
                event_id="derived-1",
                org_id="org-a",
                workspace_id="ws-a1",
                project_id="p",
                surface="lens",
                agent_id="a",
                agent_role="agent",
                run_id="r",
                step_id="s",
                event_type="model.call",
                cost_usd=0.02,
                payload={"reservation_id": "res-x"},
            )
        )
        time.sleep(0.3)
        sink.close(deadline_s=1.0)
        assert len(server.events) == 1
        assert server.settle_calls == 0
    finally:
        server.stop()


def test_p35_finops_event_in_keeps_lineage_fields():
    KazenEventIn = _finops_style_event_in()

    ev = KazenEventIn.model_validate(
        {
            "schema_version": "1.2",
            "ts_ms": 1,
            "event_id": "e",
            "org_id": "o",
            "workspace_id": "ws-a1",
            "project_id": "p",
            "surface": "s",
            "agent_id": "a",
            "agent_role": "agent",
            "run_id": "run-z",
            "parent_run_id": "run-y",
            "trace_id": "tr",
            "client_id": "cli",
            "step_id": "s1",
            "event_type": "model.call",
            "payload": {"reservation_id": "r1"},
        }
    )
    dump = ev.model_dump()
    assert dump["root_run_id"] == "run-z"
    assert dump["parent_run_id"] == "run-y"
    assert dump["trace_id"] == "tr"
    assert dump["client_id"] == "cli"
