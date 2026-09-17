"""FINAL_1 P3-3 — private-by-default capture; canary secrets must not leak."""

from __future__ import annotations

import json
import logging
import sqlite3
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kazenai import CaptureMode, inputs_absent, monitor, redact_secrets, resolve_capture_mode
from kazenai.capture_policy import build_content_ref, event_log_fields, sanitize_event_dict
from kazenai.retry_queue import RetryQueue
from kazenai.schema import KazenEvent, new_id, now_ms
from kazenai.sinks import HttpSink, HttpSinkConfig, MemorySink

CANARY = "CANARY_SECRET_p33_do_not_leak_9f3a"


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            choices=[SimpleNamespace(message=SimpleNamespace(content=f"echo:{CANARY}"))],
        )


class FakeOpenAI:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())


def _standalone(monkeypatch) -> None:
    monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)
    monkeypatch.delenv("KAZENAI_FINOPS_INGEST_URL", raising=False)
    monkeypatch.delenv("KAZENAI_INGEST_URL", raising=False)
    monkeypatch.delenv("KAZENAI_CAPTURE_MODE", raising=False)
    monkeypatch.delenv("KAZENAI_CAPTURE_CONSENT", raising=False)
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")


def _assert_no_canary(blob: str) -> None:
    assert CANARY not in blob


def test_p33_default_mode_is_metadata():
    assert resolve_capture_mode() is CaptureMode.METADATA


def test_p33_full_without_consent_falls_back_to_redacted(monkeypatch):
    monkeypatch.delenv("KAZENAI_CAPTURE_CONSENT", raising=False)
    assert resolve_capture_mode(explicit="full", consent=False) is CaptureMode.REDACTED
    assert resolve_capture_mode(explicit="full", consent=True) is CaptureMode.FULL


def test_p33_redact_nested_and_string_canary():
    payload = {
        "messages": [{"role": "user", "content": f"please use {CANARY}"}],
        "api_key": CANARY,
        "nested": {"token": CANARY, "ok": "hello"},
    }
    scrubbed = redact_secrets(payload)
    blob = json.dumps(scrubbed)
    _assert_no_canary(blob)
    assert scrubbed["api_key"] == "[REDACTED]"
    assert scrubbed["nested"]["token"] == "[REDACTED]"
    assert "[REDACTED]" in scrubbed["messages"][0]["content"]


def test_p33_metadata_ref_omits_bodies():
    ref = build_content_ref({"messages": [{"content": CANARY}]}, mode=CaptureMode.METADATA)
    assert ref["kind"] == "omitted"
    assert ref["present"] is False
    assert inputs_absent(ref)
    assert CANARY not in json.dumps(ref)


def test_p33_default_monitor_timeline_and_logs_have_no_canary(monkeypatch, caplog):
    _standalone(monkeypatch)
    client = FakeOpenAI()
    with caplog.at_level(logging.INFO):
        monitor(client, max_budget_usd=1.0, agent_id="p33")
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"secret {CANARY}"}],
        )
    _assert_no_canary(caplog.text)

    with tempfile.TemporaryDirectory() as tmp:
        timeline = str(Path(tmp) / "t.jsonl")
        client2 = FakeOpenAI()
        monitored = monitor(client2, max_budget_usd=1.0, timeline_path=timeline)
        monitored.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"secret {CANARY}"}],
        )
        raw = Path(timeline).read_text(encoding="utf-8")
        _assert_no_canary(raw)
        events = [json.loads(line) for line in raw.splitlines() if line.strip()]
        model_calls = [e for e in events if e.get("event_type") == "model.call"]
        assert model_calls
        iref = model_calls[0]["payload"]["inputs_ref"]
        oref = model_calls[0]["payload"]["outputs_ref"]
        assert inputs_absent(iref)
        assert inputs_absent(oref)
        assert model_calls[0]["payload"]["capture"]["bodies"] == "off"


def test_p33_sqlite_spool_and_http_payload_have_no_canary(monkeypatch):
    _standalone(monkeypatch)
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "offline.sqlite3")
        bad = KazenEvent(
            schema_version="1.0",
            ts_ms=now_ms(),
            event_id=new_id(),
            org_id="o",
            project_id="p",
            workspace_id="w",
            surface="test",
            agent_id="a",
            agent_role="agent",
            run_id="r",
            step_id="s",
            event_type="model.call",
            payload={
                "inputs_ref": {"kind": "inline", "value": {"messages": [{"content": CANARY}]}},
                "outputs_ref": {"kind": "inline", "value": CANARY},
            },
        )
        mem = MemorySink()
        mem.emit(bad)
        _assert_no_canary(json.dumps(mem.snapshot()))

        q = RetryQueue(db)
        q.enqueue(sanitize_event_dict(bad.model_dump()))
        with sqlite3.connect(db) as con:
            row = con.execute("SELECT payload_json FROM pending_events").fetchone()
        assert row is not None
        _assert_no_canary(row[0])

        offline = Path(tmp) / "offline_events.jsonl"
        sink = HttpSink(
            HttpSinkConfig(
                ingest_url="http://127.0.0.1:1",
                timeout_s=0.2,
                flush_interval_s=0.05,
                batch_max=1,
                offline_queue_path=str(offline),
            )
        )
        try:
            sink.emit(bad)
            time.sleep(0.5)
        finally:
            sink.close()
        sqlite_offline = offline.with_suffix(".sqlite3")
        found = False
        if sqlite_offline.exists():
            with sqlite3.connect(str(sqlite_offline)) as con:
                for (payload,) in con.execute("SELECT payload_json FROM pending_events"):
                    _assert_no_canary(payload)
                    found = True
        if offline.exists():
            _assert_no_canary(offline.read_text(encoding="utf-8"))
            found = True
        assert found, "expected offline spool after failed HTTP"


def test_p33_opt_in_redacted_scrubs_canary_but_keeps_structure(monkeypatch):
    _standalone(monkeypatch)
    with tempfile.TemporaryDirectory() as tmp:
        timeline = str(Path(tmp) / "t.jsonl")
        client = FakeOpenAI()
        monitor(
            client,
            max_budget_usd=1.0,
            timeline_path=timeline,
            capture_mode="redacted",
        )
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"secret {CANARY}"}],
        )
        raw = Path(timeline).read_text(encoding="utf-8")
        _assert_no_canary(raw)
        ev = json.loads(raw.splitlines()[0])
        iref = ev["payload"]["inputs_ref"]
        assert iref["kind"] == "inline"
        assert iref["present"] is True
        assert not inputs_absent(iref)
        assert "[REDACTED]" in json.dumps(iref["value"])


def test_p33_event_log_fields_never_include_bodies():
    ev = KazenEvent(
        schema_version="1.0",
        ts_ms=now_ms(),
        event_id=new_id(),
        org_id="o",
        project_id="p",
        workspace_id="w",
        surface="test",
        agent_id="a",
        agent_role="agent",
        run_id="r",
        step_id="s",
        event_type="model.call",
        payload={
            "inputs_ref": {"kind": "inline", "value": CANARY},
            "outputs_ref": {"kind": "inline", "value": CANARY},
        },
    )
    fields = event_log_fields(ev)
    _assert_no_canary(json.dumps(fields))
    assert fields["inputs_absent"] is True
    assert fields["outputs_absent"] is True


def test_p33_error_text_from_budget_exceeded_has_no_prompt(monkeypatch):
    _standalone(monkeypatch)
    from kazenai import BudgetExceeded

    client = FakeOpenAI()
    monitor(client, max_budget_usd=0.000001)
    with pytest.raises(BudgetExceeded) as ei:
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"leak {CANARY}"}],
        )
    _assert_no_canary(str(ei.value))
