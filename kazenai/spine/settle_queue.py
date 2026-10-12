"""File-backed SDK settlement delivery queue.

Journals exact lifecycle settle intent *before* delivery attempts. Removes an
entry only when FinOps returns ``applied`` / ``duplicate`` for the same
fingerprint. HTTP 200 + ``result=pending`` is not an acknowledgment.

Layout under the queue root::

    pending/<fingerprint>.json   one durable record per exact intent
    claims/<identity>.claim      one immutable fingerprint per
                                 (org, destination, idempotency_key)
    quarantine/                  malformed records moved aside for inspection
    .flush.lock                  single-owner recovery lock
    .last_flush.json             outcome of the most recent flush

Claims are published with ``os.link`` from a fully written temp file, which
fails atomically when the target exists. That makes "one fingerprint per
idempotency key" hold across processes sharing a local directory. It is
not cross-host coordination.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import stat
import sys
import tempfile
import time
import uuid
from typing import Any, Iterator, Mapping, Optional

_log = logging.getLogger("kazenai.spine.settle_queue")

# A claim whose intent has no pending record is either mid-enqueue or orphaned
# by a crash between ack steps. Only orphans older than this may be reclaimed.
STALE_CLAIM_S = float(os.getenv("KAZENAI_SDK_SETTLE_STALE_CLAIM_S", "300") or 300)


class SettleIntentConflict(ValueError):
    """Same idempotency key reused with a different exact intent fingerprint."""


def _chmod_private_dir(path: str) -> None:
    try:
        os.chmod(path, 0o700)
    except Exception:
        pass


def _chmod_private_file(path: str) -> None:
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def _fsync_dir(path: str) -> None:
    """Persist directory entries (rename/link/unlink) where the OS supports it."""
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_private_tmp(directory: str, stem: str, payload: Mapping[str, Any]) -> str:
    tmp = os.path.join(directory, f".{stem}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    return tmp


def _default_root() -> str:
    raw = (os.getenv("KAZENAI_SDK_SETTLE_QUEUE_DIR") or "").strip()
    if raw:
        os.makedirs(raw, exist_ok=True)
        _chmod_private_dir(raw)
        return raw
    # Prefer XDG/state home over world-readable shared /tmp when available.
    state_home = (os.getenv("XDG_STATE_HOME") or "").strip()
    if not state_home:
        home = (os.getenv("HOME") or "").strip()
        if home:
            state_home = os.path.join(home, ".local", "state")
    if state_home:
        fallback = os.path.join(state_home, "kazenai", "sdk-settle-queue")
    else:
        fallback = os.path.join(tempfile.gettempdir(), "kazenai-sdk-settle-queue")
    os.makedirs(fallback, exist_ok=True)
    _chmod_private_dir(fallback)
    return fallback


def intent_fingerprint(
    body: Mapping[str, Any],
    *,
    org_id: str = "",
    base_url: str = "",
) -> str:
    """Stable identity for one exact settle intent (destination + amount + ids)."""
    material = {
        "idempotency_key": str(body.get("idempotency_key") or ""),
        "org_id": str(org_id or body.get("org_id") or ""),
        "base_url": str(base_url or "").rstrip("/"),
        "event": str(body.get("event") or ""),
        "reservation_id": str(body.get("reservation_id") or ""),
        "call_id": str(body.get("call_id") or ""),
        "attempt": int(body.get("attempt") or 0),
        "actual_usd_micros": body.get("actual_usd_micros"),
        "lifecycle_version": str(body.get("lifecycle_version") or ""),
    }
    raw = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def claim_identity(*, org_id: str, base_url: str, idempotency_key: str) -> str:
    """Identity that may hold exactly one fingerprint (destination + tenant + key)."""
    material = "\x00".join([org_id, base_url.rstrip("/"), idempotency_key]).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


class DurableSettleQueue:
    """Process-restart durable queue of settle POST bodies."""

    def __init__(self, root: str | None = None) -> None:
        self.root = root or _default_root()
        self.pending_dir = os.path.join(self.root, "pending")
        self.claims_dir = os.path.join(self.root, "claims")
        self.quarantine_dir = os.path.join(self.root, "quarantine")
        for d in (self.pending_dir, self.claims_dir, self.quarantine_dir):
            os.makedirs(d, exist_ok=True)
        for d in (self.root, self.pending_dir, self.claims_dir, self.quarantine_dir):
            _chmod_private_dir(d)
        self._backfill_claims()

    def _path_for_fingerprint(self, fingerprint: str) -> str:
        return os.path.join(self.pending_dir, f"{fingerprint[:40]}.json")

    def _claim_path(self, identity: str) -> str:
        return os.path.join(self.claims_dir, f"{identity[:40]}.claim")

    # -- claims ---------------------------------------------------------------

    def _read_claim(self, path: str) -> Optional[dict[str, Any]]:
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            return raw if isinstance(raw, dict) else None
        except FileNotFoundError:
            return None
        except Exception:
            return None

    def _acquire_claim(self, identity: str, fingerprint: str, idem: str) -> None:
        """Bind ``identity`` to ``fingerprint`` or raise ``SettleIntentConflict``.

        Claim files are immutable once linked; a different fingerprint can only
        take over an orphaned claim (no pending record, older than the stale bound).
        """
        path = self._claim_path(identity)
        for _ in range(8):
            existing = self._read_claim(path)
            if existing is not None:
                other_fp = str(existing.get("fingerprint") or "")
                if other_fp == fingerprint:
                    return
                if self._claim_is_orphaned(existing, path):
                    self._remove_claim_if(path, other_fp)
                    continue
                raise SettleIntentConflict(
                    "settle intent conflict: idempotency_key="
                    f"{idem!r} already journals fingerprint {other_fp[:12]}… "
                    f"≠ {fingerprint[:12]}…"
                )
            tmp = _write_private_tmp(
                self.claims_dir,
                identity[:40],
                {"fingerprint": fingerprint, "claimed_at": time.time()},
            )
            try:
                os.link(tmp, path)
            except FileExistsError:
                continue  # lost the race; re-read the winner's fingerprint
            finally:
                with contextlib.suppress(OSError):
                    os.remove(tmp)
            _fsync_dir(self.claims_dir)
            return
        raise SettleIntentConflict(
            f"settle intent conflict: could not claim idempotency_key={idem!r}"
        )

    def _claim_is_orphaned(self, claim: Mapping[str, Any], path: str) -> bool:
        fp = str(claim.get("fingerprint") or "")
        if fp and os.path.exists(self._path_for_fingerprint(fp)):
            return False
        try:
            claimed_at = float(claim.get("claimed_at") or os.path.getmtime(path))
        except Exception:
            return False
        return (time.time() - claimed_at) >= STALE_CLAIM_S

    def _remove_claim_if(self, path: str, fingerprint: str) -> None:
        """Remove the claim only if it still binds ``fingerprint``.

        Rename-to-tombstone makes the check-and-remove atomic against a
        concurrent replacement; a claim that changed underneath is restored.
        """
        tomb = f"{path}.{uuid.uuid4().hex}.tomb"
        try:
            os.rename(path, tomb)
        except FileNotFoundError:
            return
        moved = self._read_claim(tomb)
        if moved is not None and str(moved.get("fingerprint") or "") == fingerprint:
            with contextlib.suppress(FileNotFoundError):
                os.remove(tomb)
            _fsync_dir(self.claims_dir)
            return
        # Not ours — put it back unless another claim was published meanwhile.
        with contextlib.suppress(FileExistsError):
            os.link(tomb, path)
        with contextlib.suppress(FileNotFoundError):
            os.remove(tomb)

    def _release_claim_for(self, record: Mapping[str, Any]) -> None:
        body = record.get("body") if isinstance(record.get("body"), dict) else {}
        idem = str(body.get("idempotency_key") or "")
        if not idem:
            return
        identity = claim_identity(
            org_id=str(record.get("org_id") or ""),
            base_url=str(record.get("base_url") or ""),
            idempotency_key=idem,
        )
        self._remove_claim_if(self._claim_path(identity), str(record.get("fingerprint") or ""))

    def _backfill_claims(self) -> None:
        """Give records journaled by older SDKs (no claims dir) their claim."""
        for item in self.list_pending():
            body = item.get("body") if isinstance(item.get("body"), dict) else {}
            idem = str(body.get("idempotency_key") or "")
            fp = str(item.get("fingerprint") or "")
            if not idem or not fp:
                continue
            identity = claim_identity(
                org_id=str(item.get("org_id") or ""),
                base_url=str(item.get("base_url") or ""),
                idempotency_key=idem,
            )
            if os.path.exists(self._claim_path(identity)):
                continue
            try:
                self._acquire_claim(identity, fp, idem)
            except SettleIntentConflict:
                _log.warning(
                    "settle queue holds conflicting legacy records for idempotency_key=%r",
                    idem,
                )

    # -- records --------------------------------------------------------------

    def enqueue(
        self,
        body: Mapping[str, Any],
        *,
        error: str = "",
        base_url: str = "",
        api_key: str = "",
        org_id: str = "",
        persist_api_key: bool = False,
    ) -> str:
        """Journal intent. Returns absolute path of the durable record.

        File identity is the intent fingerprint (org/base/amount/ids), not the
        bare idempotency key — conflicting reuse cannot overwrite another intent.
        API keys are not persisted by default; flush resolves them from the
        caller / environment.
        """
        org = str(org_id or body.get("org_id") or "")
        base = (base_url or "").rstrip("/")
        fingerprint = intent_fingerprint(body, org_id=org, base_url=base)
        path = self._path_for_fingerprint(fingerprint)
        idem = str(body.get("idempotency_key") or "")
        if idem:
            # Same destination + same idempotency key with a different fingerprint
            # is conflicting reuse. Distinct destinations keep separate claims.
            self._acquire_claim(
                claim_identity(org_id=org, base_url=base, idempotency_key=idem),
                fingerprint,
                idem,
            )
        if os.path.exists(path):
            # Same fingerprint → idempotent refresh of metadata only.
            try:
                with open(path, encoding="utf-8") as fh:
                    existing = json.load(fh)
                if (
                    isinstance(existing, dict)
                    and existing.get("fingerprint") == fingerprint
                    and existing.get("body") == dict(body)
                    and str(existing.get("org_id") or "") == org
                    and str(existing.get("base_url") or "") == base
                ):
                    return path
            except Exception:
                pass
        payload = {
            "fingerprint": fingerprint,
            "record_id": uuid.uuid4().hex,
            "body": dict(body),
            "error": error,
            "base_url": base,
            # Prefer runtime credential resolution; only persist when explicitly requested.
            "api_key": (api_key or "") if persist_api_key else "",
            "org_id": org,
            "enqueued_at": time.time(),
        }
        tmp = _write_private_tmp(self.pending_dir, fingerprint[:40], payload)
        try:
            os.replace(tmp, path)
        except Exception:
            with contextlib.suppress(OSError):
                os.remove(tmp)
            raise
        _chmod_private_file(path)
        _fsync_dir(self.pending_dir)
        return path

    def _quarantine(self, path: str, reason: str) -> None:
        name = os.path.basename(path)
        target = os.path.join(self.quarantine_dir, f"{int(time.time())}.{uuid.uuid4().hex[:8]}.{name}")
        try:
            os.replace(path, target)
            _fsync_dir(self.pending_dir)
            _log.warning("settle queue quarantined %s (%s) -> %s", name, reason, target)
        except FileNotFoundError:
            pass
        except Exception:
            _log.error("settle queue could not quarantine %s (%s)", name, reason, exc_info=True)

    def list_pending(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        try:
            names = sorted(os.listdir(self.pending_dir))
        except FileNotFoundError:
            return out
        for name in names:
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.pending_dir, name)
            try:
                with open(path, encoding="utf-8") as fh:
                    raw = json.load(fh)
            except FileNotFoundError:
                continue  # acked concurrently
            except Exception as exc:
                # Records are published by atomic rename, so a parse failure is
                # real corruption rather than a torn in-progress write.
                self._quarantine(path, f"unreadable: {type(exc).__name__}")
                continue
            if not isinstance(raw, dict) or not isinstance(raw.get("body"), dict):
                self._quarantine(path, "malformed record")
                continue
            raw["_path"] = path
            out.append(raw)
        return out

    def ack(self, path: str, *, fingerprint: str | None = None) -> bool:
        """Remove a pending record only when it still matches ``fingerprint``."""
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            return False
        except Exception:
            raw = None
        if fingerprint and (not isinstance(raw, dict) or raw.get("fingerprint") != fingerprint):
            return False
        try:
            os.remove(path)
        except FileNotFoundError:
            return False
        except Exception:
            return False
        _fsync_dir(self.pending_dir)
        if isinstance(raw, dict):
            self._release_claim_for(raw)
        return True

    # -- operations -----------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Backlog and health signals for supervisors and metrics exporters."""
        pending = self.list_pending()
        now = time.time()
        ages = [now - float(p.get("enqueued_at") or now) for p in pending]
        try:
            quarantined = len([n for n in os.listdir(self.quarantine_dir) if n.endswith(".json")])
        except FileNotFoundError:
            quarantined = 0
        last: dict[str, Any] = {}
        try:
            with open(os.path.join(self.root, ".last_flush.json"), encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                last = loaded
        except Exception:
            pass
        return {
            "depth": len(pending),
            "oldest_age_s": round(max(ages), 3) if ages else 0.0,
            "quarantined": quarantined,
            "last_flush_at": last.get("at"),
            "last_flush_ok_at": last.get("ok_at"),
            "last_flush_applied": last.get("applied"),
            "last_flush_failed": last.get("failed"),
        }

    def _record_flush(self, result: Mapping[str, int]) -> None:
        prev: dict[str, Any] = {}
        path = os.path.join(self.root, ".last_flush.json")
        try:
            with open(path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                prev = loaded
        except Exception:
            pass
        now = time.time()
        payload = {
            "at": now,
            "ok_at": now if not result.get("failed") else prev.get("ok_at"),
            "applied": int(result.get("applied") or 0),
            "failed": int(result.get("failed") or 0),
        }
        try:
            tmp = _write_private_tmp(self.root, ".last_flush", payload)
            os.replace(tmp, path)
        except Exception:
            _log.debug("could not record flush outcome", exc_info=True)

    @contextlib.contextmanager
    def flush_lock(self) -> Iterator[bool]:
        """Single-owner recovery lock. Yields False when another flusher holds it."""
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows: caller owns single-flusher rule
            yield True
            return
        fd = os.open(os.path.join(self.root, ".flush.lock"), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


_QUEUE: DurableSettleQueue | None = None


def get_settle_queue() -> DurableSettleQueue:
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = DurableSettleQueue()
    return _QUEUE


def reset_settle_queue_for_tests(root: str | None = None) -> DurableSettleQueue:
    """Replace the process singleton (tests only)."""
    global _QUEUE
    _QUEUE = DurableSettleQueue(root=root)
    return _QUEUE


def journal_settle_intent(
    body: Mapping[str, Any],
    *,
    error: str = "pre_send",
    base_url: str = "",
    org_id: str = "",
    api_key: str | None = None,
    persist_api_key: bool = False,
) -> str:
    """Persist exact intent before delivery. Raises on journal failure (observable)."""
    return get_settle_queue().enqueue(
        body,
        error=error,
        base_url=base_url,
        api_key=api_key or "",
        org_id=org_id,
        persist_api_key=persist_api_key,
    )


def enqueue_failed_settle(
    body: Mapping[str, Any],
    *,
    error: str,
    base_url: str = "",
    api_key: str | None = None,
    org_id: str = "",
) -> None:
    """Best-effort enqueue used only when journal-before-send was skipped."""
    try:
        journal_settle_intent(
            body,
            error=error,
            base_url=base_url,
            org_id=org_id,
            api_key=api_key,
            persist_api_key=False,
        )
    except Exception:
        _log.debug("settle queue enqueue failed", exc_info=True)


def queue_stats() -> dict[str, Any]:
    return get_settle_queue().stats()


def flush_pending_settlements(
    *,
    default_base_url: str = "",
    default_api_key: str | None = None,
    timeout_s: float = 2.0,
) -> dict[str, int]:
    """Redeliver queued settle bodies. Only applied/duplicate ack the *same* fingerprint.

    Holds the queue's single-owner flush lock; a concurrent flusher gets
    ``{"applied": 0, "failed": 0, "locked": 1}`` without touching records.
    """
    q = get_settle_queue()
    with q.flush_lock() as owned:
        if not owned:
            return {"applied": 0, "failed": 0, "locked": 1}
        result = _flush_locked(
            q,
            default_base_url=default_base_url,
            default_api_key=default_api_key,
            timeout_s=timeout_s,
        )
        q._record_flush(result)
        return result


def _flush_locked(
    q: DurableSettleQueue,
    *,
    default_base_url: str,
    default_api_key: str | None,
    timeout_s: float,
) -> dict[str, int]:
    from kazenai.spine.guard import _post_settle_sync

    applied = 0
    failed = 0
    for item in q.list_pending():
        body = item.get("body") if isinstance(item.get("body"), dict) else None
        path = str(item.get("_path") or "")
        fingerprint = str(item.get("fingerprint") or "")
        if not body or not path:
            continue
        base = str(item.get("base_url") or default_base_url or "").rstrip("/")
        org = str(item.get("org_id") or body.get("org_id") or "")
        key = item.get("api_key")
        if key is None or key == "":
            key = default_api_key
        if not base:
            failed += 1
            continue
        # Recompute fingerprint — refuse to ack if the on-disk record drifted.
        expected = fingerprint or intent_fingerprint(body, org_id=org, base_url=base)
        try:
            ok = _post_settle_sync(
                base,
                org,
                dict(body),
                timeout_s=timeout_s,
                api_key=str(key) if key is not None else None,
                enqueue_on_failure=False,
                journal_before_send=False,
            )
            if ok:
                if q.ack(path, fingerprint=expected):
                    applied += 1
                else:
                    # Replacement with different fingerprint survived — do not count applied
                    # against the wrong intent.
                    failed += 1
            else:
                failed += 1
        except Exception:
            failed += 1
    return {"applied": applied, "failed": failed}


def main(argv: list[str] | None = None) -> int:
    """``python -m kazenai.spine.settle_queue {flush,stats}`` — supervised recovery entrypoint.

    Exit codes: 0 ok, 1 some records failed, 3 another flusher holds the lock.
    """
    parser = argparse.ArgumentParser(prog="python -m kazenai.spine.settle_queue")
    parser.add_argument("command", choices=["flush", "stats"])
    parser.add_argument("--root", default=None, help="queue dir (default KAZENAI_SDK_SETTLE_QUEUE_DIR)")
    parser.add_argument("--base-url", default="", help="FinOps URL for records lacking one")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args(argv)

    global _QUEUE
    if args.root:
        _QUEUE = DurableSettleQueue(root=args.root)
    if args.command == "stats":
        print(json.dumps(queue_stats(), sort_keys=True))
        return 0
    base = args.base_url or (
        os.getenv("KAZENAI_FINOPS_URL") or os.getenv("KAZENAI_FINOPS_INGEST_URL") or ""
    ).strip()
    result = flush_pending_settlements(default_base_url=base, timeout_s=args.timeout)
    print(json.dumps({**result, **queue_stats()}, sort_keys=True))
    if result.get("locked"):
        return 3
    return 1 if result.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
