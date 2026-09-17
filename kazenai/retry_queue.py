"""SQLite-backed offline event queue with lease/ack, dedupe, and disk bounds.

FINAL_1 P3-4: durable client spool for HttpSink — never delete-on-read.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_log = logging.getLogger("kazenai.retry_queue")


@dataclass
class LeasedEvent:
    row_id: int
    event: Dict[str, Any]
    attempts: int
    event_id: Optional[str]
    org_id: Optional[str]


class RetryQueue:
    """Tenant-aware durable spool with lease/ack and optional size bounds."""

    def __init__(
        self,
        path: str,
        *,
        org_id: Optional[str] = None,
        max_pending: int = 10_000,
        max_disk_bytes: int = 64 * 1024 * 1024,
        max_attempts: int = 8,
        lease_ttl_ms: int = 30_000,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._org_id = (org_id or os.getenv("KAZENAI_ORG_ID") or "").strip() or None
        self._max_pending = int(max_pending)
        self._max_disk_bytes = int(max_disk_bytes)
        self._max_attempts = int(max_attempts)
        self._lease_ttl_ms = int(lease_ttl_ms)
        self._lock = threading.Lock()
        self.stats = {
            "enqueued": 0,
            "deduped": 0,
            "acked": 0,
            "nacked": 0,
            "poison": 0,
            "disk_full": 0,
            "rejected_tenant": 0,
        }
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self._path), timeout=30.0)
        con.execute("PRAGMA journal_mode=WAL;")
        return con

    def _init_db(self) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at_ms INTEGER NOT NULL,
                  event_id TEXT,
                  org_id TEXT,
                  payload_json TEXT NOT NULL,
                  attempts INTEGER NOT NULL DEFAULT 0,
                  lease_until_ms INTEGER NOT NULL DEFAULT 0,
                  poison INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            con.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_pending_event_id "
                "ON pending_events(event_id) WHERE event_id IS NOT NULL AND event_id != '';"
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS dead_letter (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  moved_at_ms INTEGER NOT NULL,
                  event_id TEXT,
                  org_id TEXT,
                  payload_json TEXT NOT NULL,
                  attempts INTEGER NOT NULL,
                  reason TEXT
                );
                """
            )
            con.commit()

    def _disk_bytes(self) -> int:
        try:
            return int(self._path.stat().st_size)
        except OSError:
            return 0

    def enqueue(self, event: Dict[str, Any]) -> str:
        """Enqueue event. Returns applied|duplicate|rejected_disk|rejected_tenant."""
        event_id = str(event.get("event_id") or "").strip() or None
        org_id = str(event.get("org_id") or "").strip() or None
        if self._org_id and org_id and org_id != self._org_id:
            self.stats["rejected_tenant"] += 1
            _log.warning("retry_queue rejected cross-tenant event org=%s expected=%s", org_id, self._org_id)
            return "rejected_tenant"

        with self._lock, self._connect() as con:
            if self._disk_bytes() >= self._max_disk_bytes:
                self.stats["disk_full"] += 1
                _log.error("retry_queue disk full path=%s bytes=%s", self._path, self._disk_bytes())
                return "rejected_disk"
            row = con.execute("SELECT COUNT(1) FROM pending_events WHERE poison=0").fetchone()
            pending = int(row[0]) if row else 0
            if pending >= self._max_pending:
                self.stats["disk_full"] += 1
                _log.error("retry_queue pending cap reached path=%s pending=%s", self._path, pending)
                return "rejected_disk"

            if event_id:
                existing = con.execute(
                    "SELECT id FROM pending_events WHERE event_id=? LIMIT 1",
                    (event_id,),
                ).fetchone()
                if existing:
                    self.stats["deduped"] += 1
                    return "duplicate"
                dead = con.execute(
                    "SELECT id FROM dead_letter WHERE event_id=? LIMIT 1",
                    (event_id,),
                ).fetchone()
                if dead:
                    self.stats["deduped"] += 1
                    return "duplicate"

            try:
                con.execute(
                    "INSERT INTO pending_events(created_at_ms, event_id, org_id, payload_json) VALUES(?,?,?,?)",
                    (
                        int(time.time() * 1000),
                        event_id,
                        org_id or self._org_id,
                        json.dumps(event, ensure_ascii=False),
                    ),
                )
                con.commit()
                self.stats["enqueued"] += 1
                return "applied"
            except sqlite3.IntegrityError:
                self.stats["deduped"] += 1
                return "duplicate"

    def lease(self, *, limit: int = 50) -> List[LeasedEvent]:
        """Lease pending rows without deleting. Expired leases are reclaimable."""
        now = int(time.time() * 1000)
        out: List[LeasedEvent] = []
        with self._lock, self._connect() as con:
            rows = con.execute(
                """
                SELECT id, payload_json, attempts, event_id, org_id
                FROM pending_events
                WHERE poison=0 AND lease_until_ms <= ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (now, int(limit)),
            ).fetchall()
            for rid, raw, attempts, event_id, org_id in rows:
                try:
                    payload = json.loads(str(raw))
                except Exception:
                    # Poison parse — move to DLQ immediately.
                    con.execute(
                        "INSERT INTO dead_letter(moved_at_ms, event_id, org_id, payload_json, attempts, reason) "
                        "VALUES(?,?,?,?,?,?)",
                        (now, event_id, org_id, str(raw), int(attempts or 0), "json_parse"),
                    )
                    con.execute("DELETE FROM pending_events WHERE id=?", (int(rid),))
                    self.stats["poison"] += 1
                    continue
                lease_until = now + self._lease_ttl_ms
                con.execute(
                    "UPDATE pending_events SET lease_until_ms=?, attempts=attempts+1 WHERE id=?",
                    (lease_until, int(rid)),
                )
                out.append(
                    LeasedEvent(
                        row_id=int(rid),
                        event=payload,
                        attempts=int(attempts or 0) + 1,
                        event_id=str(event_id) if event_id else None,
                        org_id=str(org_id) if org_id else None,
                    )
                )
            con.commit()
        return out

    def ack(self, row_ids: Sequence[int]) -> None:
        ids = [int(i) for i in row_ids]
        if not ids:
            return
        with self._lock, self._connect() as con:
            con.execute(
                f"DELETE FROM pending_events WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            )
            con.commit()
            self.stats["acked"] += len(ids)

    def nack(self, row_ids: Sequence[int], *, reason: str = "delivery_failed") -> None:
        """Release lease; move to DLQ when attempts exceed max."""
        ids = [int(i) for i in row_ids]
        if not ids:
            return
        now = int(time.time() * 1000)
        with self._lock, self._connect() as con:
            for rid in ids:
                row = con.execute(
                    "SELECT payload_json, attempts, event_id, org_id FROM pending_events WHERE id=?",
                    (rid,),
                ).fetchone()
                if not row:
                    continue
                raw, attempts, event_id, org_id = row
                if int(attempts or 0) >= self._max_attempts:
                    con.execute(
                        "INSERT INTO dead_letter(moved_at_ms, event_id, org_id, payload_json, attempts, reason) "
                        "VALUES(?,?,?,?,?,?)",
                        (now, event_id, org_id, str(raw), int(attempts or 0), reason),
                    )
                    con.execute("DELETE FROM pending_events WHERE id=?", (rid,))
                    self.stats["poison"] += 1
                else:
                    con.execute(
                        "UPDATE pending_events SET lease_until_ms=0 WHERE id=?",
                        (rid,),
                    )
                    self.stats["nacked"] += 1
            con.commit()

    def drain(self, *, limit: int = 500) -> List[Dict[str, Any]]:
        """Deprecated compatibility: lease then immediately ack (test helper only)."""
        leased = self.lease(limit=limit)
        if leased:
            self.ack([item.row_id for item in leased])
        return [item.event for item in leased]

    def pending_count(self) -> int:
        with self._lock, self._connect() as con:
            row = con.execute("SELECT COUNT(1) FROM pending_events WHERE poison=0").fetchone()
            return int(row[0]) if row else 0

    def dead_letter_count(self) -> int:
        with self._lock, self._connect() as con:
            row = con.execute("SELECT COUNT(1) FROM dead_letter").fetchone()
            return int(row[0]) if row else 0

    def snapshot_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "pending": self.pending_count(),
            "dead_letter": self.dead_letter_count(),
            "disk_bytes": self._disk_bytes(),
            "path": str(self._path),
            "org_id": self._org_id,
        }
