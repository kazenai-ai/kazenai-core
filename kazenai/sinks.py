"""Event sinks with bounded, lifecycle-managed HTTP delivery (FINAL_1 P3-4)."""

from __future__ import annotations

import atexit
import ipaddress
import json
import logging
import os
import queue
import threading
import time
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .capture_policy import sanitize_event_dict
from .ingest_url import ingest_post_url, normalize_ingest_base_url
from .schema import KazenEvent

_log = logging.getLogger("kazenai.sinks")


def _event_record(event: KazenEvent) -> Dict[str, Any]:
    """Serialize event for durable/network boundaries with secret scrubbing."""
    return sanitize_event_dict(event.model_dump())


def _check_ingest_url_ssrf(url: str) -> None:
    """Raise ValueError if url resolves to a private/loopback IP in production."""
    env = os.getenv("KAZENAI_ENV", "development").lower()
    if env not in ("production", "prod"):
        return
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return
    host = parsed.hostname or ""
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            raise ValueError(
                f"ingest_url must not point to a private/loopback address in production: {host!r}"
            )
    except ValueError as exc:
        if "private" in str(exc) or "loopback" in str(exc) or "reserved" in str(exc):
            raise


class EventSink(ABC):
    @abstractmethod
    def emit(self, event: KazenEvent) -> None: ...

    def close(self, *, deadline_s: float = 2.0) -> None:  # noqa: ARG002
        return None

    def stats(self) -> Dict[str, Any]:
        return {}


class MultiSink(EventSink):
    def __init__(self, sinks: Iterable[EventSink]) -> None:
        self._sinks = [s for s in sinks if s is not None]

    def emit(self, event: KazenEvent) -> None:
        for s in self._sinks:
            try:
                s.emit(event)
            except Exception:
                continue

    def close(self, *, deadline_s: float = 2.0) -> None:
        for s in self._sinks:
            try:
                s.close(deadline_s=deadline_s)
            except Exception:
                continue

    def stats(self) -> Dict[str, Any]:
        return {"sinks": [s.stats() for s in self._sinks]}


class MemorySink(EventSink):
    def __init__(self, *, max_events: int = 2000) -> None:
        self._max_events = int(max_events)
        self._lock = threading.Lock()
        self._events: List[Dict[str, Any]] = []
        self._dropped = 0

    def emit(self, event: KazenEvent) -> None:
        rec = _event_record(event)
        with self._lock:
            self._events.append(rec)
            overflow = len(self._events) - self._max_events
            if overflow > 0:
                self._events = self._events[overflow:]
                self._dropped += overflow

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)

    @property
    def events(self) -> List[KazenEvent]:
        with self._lock:
            return [KazenEvent.model_validate(rec) for rec in self._events]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"kind": "memory", "size": len(self._events), "dropped": self._dropped}


class JsonlSink(EventSink):
    def __init__(self, path: str, *, max_bytes: int = 64 * 1024 * 1024) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._max_bytes = int(max_bytes)
        self._dropped_disk = 0

    def emit(self, event: KazenEvent) -> None:
        line = json.dumps(_event_record(event), ensure_ascii=False)
        with self._lock:
            try:
                size = self._path.stat().st_size if self._path.exists() else 0
            except OSError:
                size = 0
            if size + len(line.encode("utf-8")) > self._max_bytes:
                self._dropped_disk += 1
                _log.error("jsonl_sink disk bound exceeded path=%s", self._path)
                return
            try:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                self._dropped_disk += 1
                _log.exception("jsonl_sink write failed path=%s", self._path)

    def stats(self) -> Dict[str, Any]:
        return {"kind": "jsonl", "path": str(self._path), "dropped_disk": self._dropped_disk}


@dataclass
class HttpSinkConfig:
    ingest_url: str
    api_key: str = ""
    timeout_s: float = 2.0
    batch_max: int = 50
    flush_interval_s: float = 1.0
    offline_queue_path: str = ""
    queue_max: int = 2000
    retry_interval_s: float = 1.0
    close_deadline_s: float = 5.0
    org_id: str = ""
    max_pending: int = 10_000
    max_disk_bytes: int = 64 * 1024 * 1024
    start_workers: bool = True
    # When True, financial/mandatory events raise if they cannot be queued or spooled.
    fail_closed_mandatory: bool = False
    # Extra HTTP headers (e.g. X-Kazen-Source for Lens mirror — never settle).
    extra_headers: Dict[str, str] = field(default_factory=dict)


_MANDATORY_EVENT_TYPES = frozenset(
    {
        "finops.budget.reserved",
        "finops.budget.settled",
        "finops.budget.denied",
        "finops.circuit_breaker.opened",
    }
)


@dataclass
class _HttpStats:
    delivered: int = 0
    spooled: int = 0
    dropped_queue: int = 0
    dropped_disk: int = 0
    worker_errors: int = 0
    retry_batches: int = 0
    duplicates: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "delivered": self.delivered,
            "spooled": self.spooled,
            "dropped_queue": self.dropped_queue,
            "dropped_disk": self.dropped_disk,
            "worker_errors": self.worker_errors,
            "retry_batches": self.retry_batches,
            "duplicates": self.duplicates,
        }


class HttpSink(EventSink):
    """
    Lifecycle-managed HTTP sink: one worker thread, bounded memory queue,
    durable RetryQueue spool with lease/ack reconnect drain.
    """

    def __init__(self, cfg: HttpSinkConfig) -> None:
        # Normalize so env values ending in /v1/events do not double-append.
        normalized = HttpSinkConfig(
            ingest_url=normalize_ingest_base_url(cfg.ingest_url) or cfg.ingest_url,
            api_key=cfg.api_key,
            timeout_s=cfg.timeout_s,
            batch_max=cfg.batch_max,
            flush_interval_s=cfg.flush_interval_s,
            offline_queue_path=cfg.offline_queue_path,
            queue_max=cfg.queue_max,
            retry_interval_s=cfg.retry_interval_s,
            close_deadline_s=cfg.close_deadline_s,
            org_id=cfg.org_id,
            max_pending=cfg.max_pending,
            max_disk_bytes=cfg.max_disk_bytes,
            start_workers=False,
            fail_closed_mandatory=cfg.fail_closed_mandatory,
            extra_headers=dict(cfg.extra_headers or {}),
        )
        _check_ingest_url_ssrf(normalized.ingest_url)
        self._cfg = normalized
        self._q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=max(1, int(cfg.queue_max)))
        self._stop = threading.Event()
        self._started = False
        self._stats = _HttpStats()
        self._stats_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._retry_thread: Optional[threading.Thread] = None

        self._offline_path: Optional[Path] = None
        self._retry: Any = None
        if cfg.offline_queue_path:
            p = Path(cfg.offline_queue_path)
        else:
            home = Path(os.getenv("KAZENAI_HOME", str(Path.home() / ".kazenai")))
            org = (cfg.org_id or os.getenv("KAZENAI_ORG_ID") or "local").strip() or "local"
            p = home / "spool" / org / "offline_events.jsonl"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            self._offline_path = p
            from .retry_queue import RetryQueue

            self._retry = RetryQueue(
                str(p.with_suffix(".sqlite3")),
                org_id=cfg.org_id or None,
                max_pending=cfg.max_pending,
                max_disk_bytes=cfg.max_disk_bytes,
            )
        except Exception:
            _log.exception("http_sink spool init failed")
            self._offline_path = None
            self._retry = None

        self._atexit_registered = False
        if cfg.start_workers:
            self.start()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="kazenai_http_sink", daemon=True)
        self._thread.start()
        self._retry_thread = threading.Thread(
            target=self._retry_loop, name="kazenai_http_retry", daemon=True
        )
        self._retry_thread.start()
        if not self._atexit_registered:
            atexit.register(self.close)
            self._atexit_registered = True

    def emit(self, event: KazenEvent) -> None:
        rec = _event_record(event)
        mandatory = str(getattr(event, "event_type", "") or "") in _MANDATORY_EVENT_TYPES
        try:
            self._q.put_nowait(rec)
            return
        except queue.Full:
            with self._stats_lock:
                self._stats.dropped_queue += 1
            # Prefer durable spool over silent loss for optional telemetry.
            if self._spool([rec]):
                return
            with self._stats_lock:
                self._stats.dropped_disk += 1
            _log.error(
                "http_sink queue full and spool failed event_type=%s event_id=%s",
                event.event_type,
                getattr(event, "event_id", None),
            )
            if mandatory or self._cfg.fail_closed_mandatory:
                raise RuntimeError("http_sink: mandatory event could not be queued or spooled")

    def close(self, *, deadline_s: Optional[float] = None) -> None:
        deadline = float(deadline_s if deadline_s is not None else self._cfg.close_deadline_s)
        self._stop.set()
        # Unblock workers.
        try:
            self._q.put_nowait(None)
        except Exception:
            pass
        end = time.monotonic() + max(0.1, deadline)
        # Drain remaining in-memory items to POST or spool before join returns.
        leftover: List[Dict[str, Any]] = []
        while time.monotonic() < end:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            leftover.append(item)
        if leftover:
            if not self._post_batch(leftover):
                self._spool(leftover)
        for t in (self._thread, self._retry_thread):
            if t is not None:
                remaining = max(0.0, end - time.monotonic())
                try:
                    t.join(timeout=remaining)
                except Exception:
                    pass

    def stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            base = self._stats.as_dict()
        base.update(
            {
                "kind": "http",
                "queue_size": self._q.qsize(),
                "queue_max": self._cfg.queue_max,
                "started": self._started,
                "retry": self._retry.snapshot_stats() if self._retry is not None else None,
            }
        )
        return base

    def _run(self) -> None:
        buf: List[Dict[str, Any]] = []
        last_flush = time.monotonic()
        while not self._stop.is_set():
            timeout = max(0.05, float(self._cfg.flush_interval_s))
            try:
                item = self._q.get(timeout=timeout)
                if item is None:
                    break
                buf.append(item)
            except queue.Empty:
                pass

            now = time.monotonic()
            should_flush = bool(buf) and (
                len(buf) >= int(self._cfg.batch_max)
                or (now - last_flush) >= float(self._cfg.flush_interval_s)
                or self._stop.is_set()
            )
            if not should_flush:
                continue
            batch = buf[: int(self._cfg.batch_max)]
            del buf[: int(self._cfg.batch_max)]
            last_flush = now
            if not self._post_batch(batch):
                self._spool(batch)
        # Final flush after stop.
        if buf:
            if not self._post_batch(buf):
                self._spool(buf)

    def _retry_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._drain_retry_once()
            except Exception:
                with self._stats_lock:
                    self._stats.worker_errors += 1
                _log.exception("http_sink retry worker error")
            self._stop.wait(timeout=max(0.1, float(self._cfg.retry_interval_s)))

    def _drain_retry_once(self) -> None:
        if self._retry is None:
            return
        leased = self._retry.lease(limit=int(self._cfg.batch_max))
        if not leased:
            return
        with self._stats_lock:
            self._stats.retry_batches += 1
        batch = [item.event for item in leased]
        if self._post_batch(batch, count_delivered=True):
            self._retry.ack([item.row_id for item in leased])
        else:
            self._retry.nack([item.row_id for item in leased], reason="http_post_failed")

    def _post_batch(self, batch: List[Dict[str, Any]], *, count_delivered: bool = True) -> bool:
        if not batch:
            return True
        try:
            body = json.dumps({"events": batch}, ensure_ascii=False).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self._cfg.api_key}"} if self._cfg.api_key else {}),
                **dict(self._cfg.extra_headers or {}),
            }
            req = urllib.request.Request(
                ingest_post_url(self._cfg.ingest_url),
                data=body,
                method="POST",
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=float(self._cfg.timeout_s)) as resp:
                _ = resp.read(1)
            if count_delivered:
                with self._stats_lock:
                    self._stats.delivered += len(batch)
            return True
        except Exception as exc:
            with self._stats_lock:
                self._stats.worker_errors += 1
            _log.warning("http_sink post failed count=%s err=%s", len(batch), exc)
            return False

    def _spool(self, batch: List[Dict[str, Any]]) -> bool:
        if not batch:
            return True
        if self._retry is None:
            if self._offline_path is None:
                return False
            try:
                with self._offline_path.open("a", encoding="utf-8") as fh:
                    for ev in batch:
                        fh.write(json.dumps(sanitize_event_dict(ev), ensure_ascii=False) + "\n")
                with self._stats_lock:
                    self._stats.spooled += len(batch)
                return True
            except OSError:
                with self._stats_lock:
                    self._stats.dropped_disk += len(batch)
                _log.exception("http_sink jsonl spool failed")
                return False
        ok = True
        for ev in batch:
            result = self._retry.enqueue(sanitize_event_dict(ev))
            if result == "applied":
                with self._stats_lock:
                    self._stats.spooled += 1
            elif result == "duplicate":
                with self._stats_lock:
                    self._stats.duplicates += 1
            else:
                ok = False
                with self._stats_lock:
                    self._stats.dropped_disk += 1
        return ok
