"""HttpSink batching tests."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

from kazenai.schema import KazenEvent, new_id, now_ms
from kazenai.sinks import HttpSink, HttpSinkConfig


def _event() -> KazenEvent:
    return KazenEvent(
        ts_ms=now_ms(),
        event_id=new_id(),
        org_id="o",
        project_id="p",
        surface="t",
        agent_id="a",
        agent_role="agent",
        run_id="r",
        step_id="s",
        event_type="model.call",
    )


def test_http_sink_posts_batch(tmp_path):
    cfg = HttpSinkConfig(
        ingest_url="http://127.0.0.1:8090",
        api_key="key",
        flush_interval_s=0.05,
        batch_max=2,
        offline_queue_path=str(tmp_path / "offline.jsonl"),
    )
    sink = HttpSink(cfg)
    with patch("urllib.request.urlopen") as urlopen:
        resp = MagicMock()
        resp.read.return_value = b"{}"
        urlopen.return_value.__enter__.return_value = resp
        sink.emit(_event())
        sink.emit(_event())
        time.sleep(0.15)
        sink.close()
    assert urlopen.called


def test_http_sink_offline_fallback(tmp_path):
    cfg = HttpSinkConfig(
        ingest_url="http://127.0.0.1:8090",
        flush_interval_s=0.05,
        batch_max=1,
        offline_queue_path=str(tmp_path / "offline.jsonl"),
    )
    sink = HttpSink(cfg)
    sink.emit(_event())
    with patch("urllib.request.urlopen", side_effect=OSError("down")):
        time.sleep(0.12)
        sink.close()
    offline = tmp_path / "offline.jsonl"
    assert offline.exists() or (tmp_path / "offline.sqlite3").exists()
