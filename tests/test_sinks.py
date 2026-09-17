"""Event sink tests."""

from __future__ import annotations

import json
from pathlib import Path

from kazenai.schema import KazenEvent, new_id, now_ms
from kazenai.sinks import JsonlSink, MemorySink, MultiSink


def _event(**kwargs) -> KazenEvent:
    base = dict(
        ts_ms=now_ms(),
        event_id=new_id(),
        org_id="org",
        project_id="proj",
        surface="test",
        agent_id="agent",
        agent_role="agent",
        run_id="run",
        step_id="step",
        event_type="model.call",
        payload={},
    )
    base.update(kwargs)
    return KazenEvent(**base)


def test_memory_sink_stores_events():
    sink = MemorySink(max_events=10)
    ev = _event()
    sink.emit(ev)
    assert len(sink.events) == 1
    assert sink.events[0].event_id == ev.event_id


def test_memory_sink_trims_to_max():
    sink = MemorySink(max_events=2)
    for _ in range(5):
        sink.emit(_event())
    assert len(sink.snapshot()) == 2


def test_jsonl_sink_appends_line(tmp_path):
    path = str(tmp_path / "events.jsonl")
    sink = JsonlSink(path)
    sink.emit(_event(event_id="e1"))
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event_id"] == "e1"


def test_jsonl_sink_closes_file_handle(tmp_path):
    """P3-2 — emit must not leak an open file handle (ResourceWarning)."""
    import warnings

    path = str(tmp_path / "events.jsonl")
    sink = JsonlSink(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        for _ in range(3):
            sink.emit(_event())
    assert len(Path(path).read_text(encoding="utf-8").strip().splitlines()) == 3


def test_multi_sink_fail_open():
    class _Broken:
        def emit(self, event):
            raise RuntimeError("boom")

    sink = MultiSink([_Broken(), MemorySink()])
    sink.emit(_event())
    mem = sink._sinks[1]
    assert len(mem.events) == 1


def test_memory_sink_snapshot_copy():
    sink = MemorySink()
    sink.emit(_event())
    snap = sink.snapshot()
    snap.append({"tampered": True})
    assert len(sink.snapshot()) == 1
