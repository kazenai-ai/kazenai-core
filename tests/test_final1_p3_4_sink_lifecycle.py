"""FINAL_1 P3-4 — bounded durable sink lifecycle."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kazenai.retry_queue import RetryQueue
from kazenai.schema import KazenEvent, new_id, now_ms
from kazenai.sinks import HttpSink, HttpSinkConfig
from kazenai.spine import guard as guard_mod


def _event(*, event_id: str | None = None, org_id: str = "org-a", event_type: str = "model.call") -> KazenEvent:
    return KazenEvent(
        schema_version="1.0",
        ts_ms=now_ms(),
        event_id=event_id or new_id(),
        org_id=org_id,
        project_id="p",
        workspace_id="w",
        surface="t",
        agent_id="a",
        agent_role="agent",
        run_id="r",
        step_id="s",
        event_type=event_type,
        payload={},
    )


def test_p34_retry_queue_lease_ack_and_dedupe(tmp_path):
    q = RetryQueue(str(tmp_path / "q.sqlite3"), org_id="org-a", max_pending=100)
    ev = {"event_id": "e1", "org_id": "org-a", "event_type": "model.call"}
    assert q.enqueue(ev) == "applied"
    assert q.enqueue(ev) == "duplicate"
    leased = q.lease(limit=10)
    assert len(leased) == 1
    assert q.pending_count() == 1  # still pending until ack
    q.ack([leased[0].row_id])
    assert q.pending_count() == 0


def test_p34_retry_queue_nack_poison_and_tenant(tmp_path):
    q = RetryQueue(
        str(tmp_path / "q.sqlite3"),
        org_id="org-a",
        max_attempts=2,
        lease_ttl_ms=1,
    )
    assert q.enqueue({"event_id": "x", "org_id": "org-b"}) == "rejected_tenant"
    assert q.enqueue({"event_id": "p1", "org_id": "org-a", "event_type": "model.call"}) == "applied"
    first = q.lease(limit=1)
    assert first
    q.nack([first[0].row_id])
    time.sleep(0.01)
    second = q.lease(limit=1)
    assert second
    q.nack([second[0].row_id])
    assert q.pending_count() == 0
    assert q.dead_letter_count() == 1


def test_p34_retry_queue_disk_bound(tmp_path):
    q = RetryQueue(str(tmp_path / "q.sqlite3"), max_pending=2, max_disk_bytes=10**9)
    assert q.enqueue({"event_id": "a", "org_id": "o"}) == "applied"
    assert q.enqueue({"event_id": "b", "org_id": "o"}) == "applied"
    assert q.enqueue({"event_id": "c", "org_id": "o"}) == "rejected_disk"
    assert q.stats["disk_full"] >= 1


def test_p34_http_sink_burst_bounded_threads(tmp_path):
    cfg = HttpSinkConfig(
        ingest_url="http://127.0.0.1:8090",
        flush_interval_s=0.05,
        batch_max=20,
        queue_max=500,
        offline_queue_path=str(tmp_path / "offline.jsonl"),
        retry_interval_s=0.05,
        org_id="org-a",
    )
    before = threading.active_count()
    sink = HttpSink(cfg)
    after_start = threading.active_count()
    # One flush worker + one retry worker.
    assert after_start - before <= 3

    with patch("urllib.request.urlopen") as urlopen:
        resp = MagicMock()
        resp.read.return_value = b"{}"
        urlopen.return_value.__enter__.return_value = resp
        for _ in range(100):
            sink.emit(_event())
        time.sleep(0.4)
        mid = threading.active_count()
        assert mid - before <= 4
        sink.close(deadline_s=2.0)
    st = sink.stats()
    assert st["delivered"] >= 1
    assert st["queue_max"] == 500


def test_p34_http_sink_queue_backpressure_spools(tmp_path):
    cfg = HttpSinkConfig(
        ingest_url="http://127.0.0.1:1",
        flush_interval_s=10.0,  # slow flush so queue fills
        batch_max=1,
        queue_max=2,
        offline_queue_path=str(tmp_path / "offline.jsonl"),
        retry_interval_s=10.0,
        start_workers=True,
        org_id="org-a",
    )
    sink = HttpSink(cfg)
    # Fill queue without waiting for worker drain of failed posts.
    sink.emit(_event(event_id="a"))
    sink.emit(_event(event_id="b"))
    # Third should hit Full → spool
    sink.emit(_event(event_id="c"))
    time.sleep(0.05)
    st = sink.stats()
    assert st["dropped_queue"] + st["spooled"] >= 1
    assert sink._retry is not None
    assert sink._retry.pending_count() + st["delivered"] + st["spooled"] >= 1
    sink.close(deadline_s=1.0)


def test_p34_offline_reconnect_ack(tmp_path):
    cfg = HttpSinkConfig(
        ingest_url="http://127.0.0.1:8090",
        flush_interval_s=0.05,
        batch_max=2,
        queue_max=100,
        offline_queue_path=str(tmp_path / "offline.jsonl"),
        retry_interval_s=0.05,
        org_id="org-a",
    )
    sink = HttpSink(cfg)
    with patch("urllib.request.urlopen", side_effect=OSError("down")):
        sink.emit(_event(event_id="offline-1"))
        time.sleep(0.2)
    assert sink._retry.pending_count() >= 1 or sink.stats()["spooled"] >= 1

    with patch("urllib.request.urlopen") as urlopen:
        resp = MagicMock()
        resp.read.return_value = b"{}"
        urlopen.return_value.__enter__.return_value = resp
        # Wait for retry worker to drain.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and sink._retry.pending_count() > 0:
            time.sleep(0.05)
        sink.close(deadline_s=1.0)
    assert sink._retry.pending_count() == 0
    assert sink.stats()["delivered"] >= 1


def test_p34_shutdown_drains_or_spools(tmp_path):
    cfg = HttpSinkConfig(
        ingest_url="http://127.0.0.1:1",
        flush_interval_s=5.0,
        batch_max=50,
        queue_max=100,
        offline_queue_path=str(tmp_path / "offline.jsonl"),
        retry_interval_s=5.0,
        close_deadline_s=2.0,
        org_id="org-a",
    )
    sink = HttpSink(cfg)
    for i in range(5):
        sink.emit(_event(event_id=f"shutdown-{i}"))
    sink.close(deadline_s=2.0)
    # Either delivered (unlikely) or durable pending / jsonl.
    pending = sink._retry.pending_count() if sink._retry else 0
    jsonl = Path(tmp_path / "offline.jsonl")
    assert pending >= 1 or (jsonl.exists() and jsonl.stat().st_size > 0) or sink.stats()["spooled"] >= 1


def test_p34_process_crash_recovery(tmp_path):
    """Simulate crash: events on disk survive; new sink drain recovers."""
    db = tmp_path / "offline.sqlite3"
    q = RetryQueue(str(db), org_id="org-a")
    assert q.enqueue({"event_id": "crash-1", "org_id": "org-a", "event_type": "model.call"}) == "applied"
    # "Crash" — drop queue object without ack.
    del q

    cfg = HttpSinkConfig(
        ingest_url="http://127.0.0.1:8090",
        flush_interval_s=0.05,
        batch_max=10,
        offline_queue_path=str(tmp_path / "offline.jsonl"),
        retry_interval_s=0.05,
        org_id="org-a",
    )
    sink = HttpSink(cfg)
    with patch("urllib.request.urlopen") as urlopen:
        resp = MagicMock()
        resp.read.return_value = b"{}"
        urlopen.return_value.__enter__.return_value = resp
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and sink._retry.pending_count() > 0:
            time.sleep(0.05)
        sink.close(deadline_s=1.0)
    assert sink._retry.pending_count() == 0
    assert sink.stats()["delivered"] >= 1


def test_p34_poison_event_goes_to_dlq(tmp_path):
    q = RetryQueue(str(tmp_path / "q.sqlite3"), max_attempts=1)
    # Insert unparseable row directly.
    import sqlite3

    with sqlite3.connect(str(tmp_path / "q.sqlite3")) as con:
        con.execute(
            "INSERT INTO pending_events(created_at_ms, event_id, org_id, payload_json) VALUES(?,?,?,?)",
            (int(time.time() * 1000), "bad", "o", "{not-json"),
        )
        con.commit()
    leased = q.lease(limit=10)
    assert leased == []
    assert q.dead_letter_count() == 1


def test_p34_duplicate_delivery_deduped_on_spool(tmp_path):
    cfg = HttpSinkConfig(
        ingest_url="http://127.0.0.1:1",
        flush_interval_s=0.05,
        batch_max=1,
        offline_queue_path=str(tmp_path / "offline.jsonl"),
        retry_interval_s=10.0,
        org_id="org-a",
    )
    sink = HttpSink(cfg)
    with patch("urllib.request.urlopen", side_effect=OSError("down")):
        sink.emit(_event(event_id="dup-1"))
        time.sleep(0.15)
        sink.emit(_event(event_id="dup-1"))
        time.sleep(0.1)
    assert sink._retry.pending_count() == 1
    assert sink.stats()["duplicates"] >= 1 or sink._retry.stats["deduped"] >= 1
    sink.close(deadline_s=1.0)


def test_p34_mandatory_event_fail_closed_when_spool_impossible(tmp_path, monkeypatch):
    cfg = HttpSinkConfig(
        ingest_url="http://127.0.0.1:1",
        flush_interval_s=10.0,
        batch_max=1,
        queue_max=1,
        offline_queue_path=str(tmp_path / "offline.jsonl"),
        retry_interval_s=10.0,
        max_pending=0,  # spool rejects
        fail_closed_mandatory=True,
        org_id="org-a",
    )
    sink = HttpSink(cfg)
    sink.emit(_event(event_id="fill"))
    with pytest.raises(RuntimeError, match="mandatory"):
        sink.emit(_event(event_id="fin", event_type="finops.budget.settled"))
    sink.close(deadline_s=0.5)


def test_p34_default_sink_reused_across_emits(monkeypatch):
    guard_mod.reset_default_sink()
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops:8090")
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "k")
    monkeypatch.delenv("KAZENAI_AGENTLENS_MIRROR", raising=False)
    a = guard_mod._default_sink()
    b = guard_mod._default_sink()
    assert a is b
    before = threading.active_count()
    for _ in range(20):
        guard_mod._emit_event(_event(), event_sink=None)
    after = threading.active_count()
    assert after - before <= 2
    guard_mod.reset_default_sink()
