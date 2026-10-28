"""Settlement spool: cross-process identity and operated recovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from kazenai.spine import guard
from kazenai.spine import settle_queue as sq
from kazenai.spine.settle_queue import SettleIntentConflict

REPO = Path(__file__).resolve().parents[1]


def _body(amount=12000, **extra):
    body = {
        "event": "usage_known",
        "reservation_id": "rsv-a1",
        "call_id": "call-a1",
        "attempt": 1,
        "org_id": "org-a1",
        "idempotency_key": "settle:rsv-a1:usage",
        "actual_usd_micros": amount,
        "lifecycle_version": "reservation.lifecycle.v1",
    }
    body.update(extra)
    return body


_RACER = r"""
import json, os, sys, time
sys.path.insert(0, {repo!r})
from kazenai.spine import settle_queue as sq
root, go, amount = sys.argv[1], sys.argv[2], int(sys.argv[3])
q = sq.DurableSettleQueue(root)
while not os.path.exists(go):
    time.sleep(0.0005)
body = {{
    "event": "usage_known", "reservation_id": "rsv-a1", "call_id": "call-a1",
    "attempt": 1, "org_id": "org-a1", "idempotency_key": "settle:rsv-a1:usage",
    "actual_usd_micros": amount, "lifecycle_version": "reservation.lifecycle.v1",
}}
try:
    q.enqueue(body, org_id="org-a1", base_url="https://finops.invalid")
    print("ok")
except sq.SettleIntentConflict:
    print("conflict")
"""


@pytest.mark.parametrize("round_", range(3))
def test_a1_concurrent_writers_persist_one_fingerprint(tmp_path, round_):
    root = tmp_path / "spool"
    go = tmp_path / "go"
    script = _RACER.format(repo=str(REPO))
    amounts = [11, 22, 33, 44, 55, 66]
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(root), str(go), str(a)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for a in amounts
    ]
    time.sleep(0.5)  # let every racer import and spin on the barrier
    go.write_text("go")
    outs = []
    for p in procs:
        out, err = p.communicate(timeout=60)
        assert p.returncode == 0, err
        outs.append(out.strip())
    assert outs.count("ok") == 1, outs
    assert outs.count("conflict") == len(amounts) - 1, outs
    pending = sq.DurableSettleQueue(str(root)).list_pending()
    assert len(pending) == 1
    winner = amounts[outs.index("ok")]
    assert pending[0]["body"]["actual_usd_micros"] == winner


def test_a1_same_fingerprint_is_idempotent(tmp_path):
    q = sq.DurableSettleQueue(str(tmp_path / "q"))
    p1 = q.enqueue(_body(), org_id="org-a1", base_url="https://finops.invalid")
    p2 = q.enqueue(_body(), org_id="org-a1", base_url="https://finops.invalid")
    assert p1 == p2
    assert len(q.list_pending()) == 1


def test_a1_writer_vs_flusher_claim_held_until_ack(tmp_path):
    q = sq.reset_settle_queue_for_tests(str(tmp_path / "q"))
    q.enqueue(_body(12000), org_id="org-a1", base_url="https://finops.invalid")
    seen: list[str] = []

    def send_while_conflicting_writer(*args, **kwargs):
        try:
            q.enqueue(_body(99000), org_id="org-a1", base_url="https://finops.invalid")
            seen.append("ok")
        except SettleIntentConflict:
            seen.append("conflict")
        return True

    with patch.object(guard, "_post_settle_sync", send_while_conflicting_writer):
        result = sq.flush_pending_settlements(default_base_url="https://finops.invalid", default_api_key="configured-key")
    assert seen == ["conflict"]
    assert result == {"applied": 1, "failed": 0, "rejected": 0}
    assert q.list_pending() == []
    # Claim released on ack: the key is reusable for a new intent.
    q.enqueue(_body(99000), org_id="org-a1", base_url="https://finops.invalid")
    assert len(q.list_pending()) == 1


def test_a1_interrupted_write_leaves_no_visible_record(tmp_path):
    q = sq.DurableSettleQueue(str(tmp_path / "q"))
    Path(q.pending_dir, ".deadbeef.abc.tmp").write_text("{partial")
    Path(q.claims_dir, ".deadbeef.abc.tmp").write_text("{partial")
    assert q.list_pending() == []
    q.enqueue(_body(), org_id="org-a1", base_url="https://finops.invalid")
    assert len(q.list_pending()) == 1


def test_a1_stale_orphan_claim_is_reclaimable_fresh_one_is_not(tmp_path, monkeypatch):
    q = sq.DurableSettleQueue(str(tmp_path / "q"))
    identity = sq.claim_identity(
        org_id="org-a1", base_url="https://finops.invalid", idempotency_key="settle:rsv-a1:usage"
    )
    claim = Path(q._claim_path(identity))
    claim.write_text(json.dumps({"fingerprint": "f" * 64, "claimed_at": time.time()}))
    with pytest.raises(SettleIntentConflict):
        q.enqueue(_body(), org_id="org-a1", base_url="https://finops.invalid")

    claim.write_text(json.dumps({"fingerprint": "f" * 64, "claimed_at": time.time() - 10_000}))
    q.enqueue(_body(), org_id="org-a1", base_url="https://finops.invalid")
    assert len(q.list_pending()) == 1


def test_a1_backfills_claims_for_legacy_records(tmp_path):
    root = tmp_path / "q"
    pending = root / "pending"
    pending.mkdir(parents=True)
    body = _body(12000)
    fp = sq.intent_fingerprint(body, org_id="org-a1", base_url="https://finops.invalid")
    (pending / f"{fp[:40]}.json").write_text(
        json.dumps(
            {
                "fingerprint": fp,
                "body": body,
                "org_id": "org-a1",
                "base_url": "https://finops.invalid",
                "enqueued_at": time.time(),
            }
        )
    )
    q = sq.DurableSettleQueue(str(root))
    with pytest.raises(SettleIntentConflict):
        q.enqueue(_body(27000), org_id="org-a1", base_url="https://finops.invalid")


def test_a5_corrupt_records_are_quarantined_and_counted(tmp_path, caplog):
    q = sq.DurableSettleQueue(str(tmp_path / "q"))
    q.enqueue(_body(), org_id="org-a1", base_url="https://finops.invalid")
    Path(q.pending_dir, "corrupt.json").write_text("{not json")
    Path(q.pending_dir, "wrong-shape.json").write_text(json.dumps(["x"]))
    with caplog.at_level("WARNING", logger="kazenai.spine.settle_queue"):
        rows = q.list_pending()
    assert len(rows) == 1
    assert "quarantined" in caplog.text
    stats = q.stats()
    assert stats["depth"] == 1
    assert stats["quarantined"] == 2
    assert stats["oldest_age_s"] >= 0


def test_a5_flush_is_single_owner_and_records_outcome(tmp_path):
    q = sq.reset_settle_queue_for_tests(str(tmp_path / "q"))
    q.enqueue(_body(), org_id="org-a1", base_url="https://finops.invalid")
    with q.flush_lock() as owned:
        assert owned
        assert sq.flush_pending_settlements() == {"applied": 0, "failed": 0, "locked": 1}
    assert len(q.list_pending()) == 1

    with patch.object(guard, "_post_settle_sync", lambda *a, **k: False):
        assert sq.flush_pending_settlements(default_base_url="https://finops.invalid", default_api_key="configured-key")["failed"] == 1
    stats = q.stats()
    assert stats["last_flush_failed"] == 1
    assert stats["last_flush_ok_at"] is None

    monkeypatch = pytest.MonkeyPatch()
    later = time.time() + 10_000
    monkeypatch.setattr(sq.time, "time", lambda: later)
    try:
        with patch.object(guard, "_post_settle_sync", lambda *a, **k: True):
            assert sq.flush_pending_settlements(default_base_url="https://finops.invalid", default_api_key="configured-key") == {
            "applied": 1,
            "failed": 0,
            "rejected": 0,
        }
    finally:
        monkeypatch.undo()
    stats = q.stats()
    assert stats["depth"] == 0
    assert stats["last_flush_ok_at"] is not None


def test_a5_cli_exit_codes(tmp_path, capsys):
    root = str(tmp_path / "q")
    q = sq.reset_settle_queue_for_tests(root)
    assert sq.main(["stats", "--root", root]) == 0
    assert json.loads(capsys.readouterr().out)["depth"] == 0

    q.enqueue(_body(), org_id="org-a1", base_url="https://finops.invalid")
    with patch.object(guard, "_post_settle_sync", lambda *a, **k: False):
        assert sq.main(["flush", "--root", root, "--base-url", "https://finops.invalid"]) == 1
    clock = pytest.MonkeyPatch()
    later = time.time() + 10_000
    clock.setattr(sq.time, "time", lambda: later)
    try:
        with patch.object(guard, "_post_settle_sync", lambda *a, **k: True):
            assert sq.main(["flush", "--root", root, "--base-url", "https://finops.invalid"]) == 0
    finally:
        clock.undo()
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["applied"] == 1 and out["depth"] == 0


def test_a5_records_and_claims_are_private(tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX permissions")
    q = sq.DurableSettleQueue(str(tmp_path / "q"))
    path = q.enqueue(_body(), org_id="org-a1", base_url="https://finops.invalid")
    claims = list(Path(q.claims_dir).glob("*.claim"))
    assert len(claims) == 1
    for p in [Path(path), *claims]:
        assert oct(p.stat().st_mode & 0o777) == "0o600"


def test_a1_remove_claim_never_drops_a_different_claim(tmp_path):
    """Check-then-remove is exact: a claim that changed is left in place."""
    q = sq.DurableSettleQueue(str(tmp_path / "q"))
    path = Path(q._claim_path("ident"))
    path.write_text(json.dumps({"fingerprint": "b" * 64, "claimed_at": time.time()}))
    assert q._remove_claim_if(str(path), "a" * 64) is False
    assert json.loads(path.read_text())["fingerprint"] == "b" * 64
    assert q._remove_claim_if(str(path), "b" * 64) is True
    assert not path.exists()


def test_a1_writer_that_lost_its_claim_withdraws_its_record(tmp_path, monkeypatch):
    """A writer stalled past STALE_CLAIM_S must not leave a second, divergent record."""
    q = sq.DurableSettleQueue(str(tmp_path / "q"))
    identity = sq.claim_identity(
        org_id="org-a1", base_url="https://finops.invalid", idempotency_key="settle:rsv-a1:usage"
    )
    other = _body(99000)
    other_fp = sq.intent_fingerprint(other, org_id="org-a1", base_url="https://finops.invalid")
    real_replace = os.replace

    def stalled_replace(src, dst):
        real_replace(src, dst)
        if dst.startswith(q.pending_dir):
            # While this writer stalled, another intent took the claim over.
            Path(q._claim_path(identity)).write_text(
                json.dumps({"fingerprint": other_fp, "claimed_at": time.time()})
            )
            Path(q._path_for_fingerprint(other_fp)).write_text(
                json.dumps({"fingerprint": other_fp, "body": other, "org_id": "org-a1",
                            "base_url": "https://finops.invalid", "enqueued_at": time.time()})
            )

    monkeypatch.setattr(sq.os, "replace", stalled_replace)
    with pytest.raises(SettleIntentConflict):
        q.enqueue(_body(12000), org_id="org-a1", base_url="https://finops.invalid")
    monkeypatch.setattr(sq.os, "replace", real_replace)
    amounts = [r["body"]["actual_usd_micros"] for r in q.list_pending()]
    assert amounts == [99000]


def test_a1_released_claim_is_reclaimed_not_reported_as_conflict(tmp_path, monkeypatch):
    q = sq.DurableSettleQueue(str(tmp_path / "q"))
    identity = sq.claim_identity(
        org_id="org-a1", base_url="https://finops.invalid", idempotency_key="settle:rsv-a1:usage"
    )
    real_replace = os.replace

    def released_replace(src, dst):
        real_replace(src, dst)
        if dst.startswith(q.pending_dir):
            os.remove(q._claim_path(identity))  # released by a concurrent ack

    monkeypatch.setattr(sq.os, "replace", released_replace)
    q.enqueue(_body(12000), org_id="org-a1", base_url="https://finops.invalid")
    monkeypatch.setattr(sq.os, "replace", real_replace)
    assert len(q.list_pending()) == 1
    assert q._claim_binds(identity, q.list_pending()[0]["fingerprint"])


def _http_error(status, error_code):
    import io
    import urllib.error

    body = json.dumps({"detail": {"error_code": error_code}}).encode()
    return urllib.error.HTTPError("https://finops.invalid/v1/budget/settle", status, "x", {}, io.BytesIO(body))


@pytest.mark.parametrize(
    "status,code,permanent",
    [
        (409, "idempotency_payload_mismatch", True),
        (422, "invalid_amount", True),
        (400, "invalid_amount", True),
        (409, "invalid_transition", True),
        (401, "unauthorized", True),
        (408, "timeout", False),
        (429, "slow_down", False),
        (503, "unavailable", False),
    ],
)
def test_a5_permanent_settle_rejections_are_classified(status, code, permanent):
    rejection = guard._permanent_settle_rejection(_http_error(status, code))
    assert (rejection is not None) is permanent
    if permanent:
        assert rejection.error_code == code


def test_a5_rejected_record_is_quarantined_not_retried_forever(tmp_path):
    q = sq.reset_settle_queue_for_tests(str(tmp_path / "q"))
    q.enqueue(_body(), org_id="org-a1", base_url="https://finops.invalid")

    def conflict(*a, **k):
        assert k.get("raise_on_permanent") is True
        raise guard.SettleRejected(409, "idempotency_payload_mismatch")

    with patch.object(guard, "_post_settle_sync", conflict):
        assert sq.flush_pending_settlements(default_base_url="https://finops.invalid", default_api_key="configured-key") == {
            "applied": 0,
            "failed": 0,
            "rejected": 1,
        }
    stats = q.stats()
    assert stats["depth"] == 0
    assert stats["quarantined"] == 1
    with patch.object(guard, "_post_settle_sync", conflict):
        assert sq.main(["flush", "--root", q.root]) == 0  # nothing left to retry


def test_flush_never_follows_a_stored_destination_or_key(tmp_path):
    """§8: the destination and key come from the process, never from the record.

    A record whose stored URL differs from the configured URL is quarantined and
    not sent. A matching record is sent with the configured key, not the stored one.
    """
    q = sq.reset_settle_queue_for_tests(str(tmp_path / "q"))
    token = "kzcap1.test." + "B" * 43
    q.enqueue(
        _body(settlement_capability=token, idempotency_key="settle-evil"),
        org_id="org-a1",
        base_url="https://evil.example",
        api_key="stolen-key",
    )
    q.enqueue(
        _body(settlement_capability=token, idempotency_key="settle-same"),
        org_id="org-a1",
        base_url="https://FINOPS.invalid/",
        api_key="stolen-key",
    )
    for spool in Path(q.pending_dir).glob("*.json"):
        assert (spool.stat().st_mode & 0o777) == 0o600
        assert token in spool.read_text()
        assert "stolen-key" not in spool.read_text()
    sent: list[tuple[str, str, str]] = []

    def capture(base, org, body, **kwargs):
        sent.append((base, str(kwargs.get("api_key") or ""), str(body.get("idempotency_key"))))
        return True

    with patch.object(guard, "_post_settle_sync", capture):
        result = sq.flush_pending_settlements(
            default_base_url="https://finops.invalid",
            default_api_key="configured-key",
        )
    assert sent == [("https://finops.invalid", "configured-key", "settle-same")]
    assert result["applied"] == 1
    assert result["rejected"] == 1
    assert q.stats()["quarantined"] == 1
    assert q.stats()["depth"] == 0


def test_a5_post_settle_raises_only_when_asked(monkeypatch):
    import urllib.request

    def refuse(*a, **k):
        raise _http_error(409, "idempotency_payload_mismatch")

    monkeypatch.setattr("kazenai.spine.transport.urlopen", refuse)
    kwargs = dict(timeout_s=1.0, enqueue_on_failure=False, journal_before_send=False)
    assert guard._post_settle_sync("https://finops.invalid", "org-a1", _body(), **kwargs) is False
    with pytest.raises(guard.SettleRejected):
        guard._post_settle_sync(
            "https://finops.invalid", "org-a1", _body(), raise_on_permanent=True, **kwargs
        )


def test_kill_switch_leaves_the_queue_and_import_starts_no_thread(tmp_path, monkeypatch):
    monkeypatch.setenv("KAZENAI_SDK_SETTLE_AUTO_FLUSH", "0")
    sq.shutdown_settle_flusher()
    q = sq.reset_settle_queue_for_tests(root=str(tmp_path / "q"))
    q.enqueue(_body(), org_id="org-a1", base_url="https://finops.invalid")
    time.sleep(0.05)
    assert q.stats()["depth"] == 1
    assert sq._DAEMON is None or not sq._DAEMON.is_alive()


def test_world_writable_queue_refuses_to_flush(tmp_path, monkeypatch):
    monkeypatch.setenv("KAZENAI_SDK_SETTLE_AUTO_FLUSH", "0")
    root = tmp_path / "q"
    q = sq.reset_settle_queue_for_tests(root=str(root))
    q.enqueue(_body(), org_id="org-a1")
    os.chmod(root, 0o777)
    calls: list[str] = []

    def _post(*args, **kwargs):
        calls.append("posted")
        return True

    with patch.object(guard, "_post_settle_sync", _post):
        result = sq.flush_pending_settlements(default_base_url="https://finops.invalid", default_api_key="k")
    assert result.get("refused") == 1
    assert calls == []
    assert q.stats()["depth"] == 1


def test_failed_flush_persists_backoff_and_skips_until_due(tmp_path, monkeypatch):
    monkeypatch.setenv("KAZENAI_SDK_SETTLE_AUTO_FLUSH", "0")
    monkeypatch.setattr(sq.random, "uniform", lambda _a, b: b)
    q = sq.reset_settle_queue_for_tests(root=str(tmp_path / "q"))
    q.enqueue(
        {**_body(), "settlement_capability": "kzcap1.test." + "A" * 43},
        org_id="org-a1",
    )
    with patch.object(guard, "_post_settle_sync", lambda *a, **k: False):
        first = sq.flush_pending_settlements(default_base_url="https://finops.invalid", default_api_key="k")
    assert first["failed"] == 1
    pending = q.list_pending()
    assert pending[0]["attempts"] == 1
    assert pending[0]["next_attempt_at"] > time.time()
    assert "kzcap1.test." not in json.dumps({k: v for k, v in pending[0].items() if k == "error"})
    calls: list[str] = []

    def _post(*args, **kwargs):
        calls.append("posted")
        return True

    with patch.object(guard, "_post_settle_sync", _post):
        second = sq.flush_pending_settlements(default_base_url="https://finops.invalid", default_api_key="k")
    assert second["applied"] == 0
    assert calls == []


def test_auth_failure_is_quarantined_and_does_not_block_the_next_record(tmp_path, monkeypatch, caplog):
    import io
    import urllib.error
    import urllib.request
    from email.message import Message

    monkeypatch.setenv("KAZENAI_SDK_SETTLE_AUTO_FLUSH", "0")
    q = sq.reset_settle_queue_for_tests(str(tmp_path / "q"))
    q.enqueue({**_body(), "reservation_id": "poison"}, org_id="org-a1")
    q.enqueue({**_body(), "reservation_id": "good", "idempotency_key": "settle:good"}, org_id="org-a1")
    seen: list[str] = []

    def urlopen(req, timeout=0):
        body = json.loads(req.data.decode())
        seen.append(body["reservation_id"])
        if body["reservation_id"] == "poison":
            raise urllib.error.HTTPError(
                req.full_url,
                401,
                "no",
                Message(),
                io.BytesIO(b'{"detail":{"error_code":"unauthorized","secret":"do-not-keep"}}'),
            )

        class _Ok:
            def read(self):
                return b'{"result":"applied"}'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return _Ok()

    monkeypatch.setattr("kazenai.spine.transport.urlopen", urlopen)
    with caplog.at_level("WARNING"):
        result = sq.flush_pending_settlements(default_base_url="https://finops.invalid", default_api_key="k")
    assert set(seen) == {"poison", "good"}
    assert result["applied"] == 1
    assert result["rejected"] == 1
    quarantined = list(Path(q.quarantine_dir).glob("*.json"))
    assert quarantined
    assert "do-not-keep" not in quarantined[0].read_text()
    assert "unauthorized" in caplog.text
    assert "do-not-keep" not in caplog.text


def test_retry_after_is_persisted_and_capped(tmp_path, monkeypatch):
    import io
    import urllib.error
    import urllib.request
    from email.message import Message

    monkeypatch.setenv("KAZENAI_SDK_SETTLE_AUTO_FLUSH", "0")
    monkeypatch.setattr(sq.random, "uniform", lambda _a, _b: 0)
    q = sq.reset_settle_queue_for_tests(str(tmp_path / "q"))
    q.enqueue(_body(), org_id="org-a1")
    headers = Message()
    headers["Retry-After"] = "50"

    def urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 429, "slow", headers, io.BytesIO(b"{}"))

    monkeypatch.setattr("kazenai.spine.transport.urlopen", urlopen)
    started = time.time()
    result = sq.flush_pending_settlements(default_base_url="https://finops.invalid", default_api_key="k")
    assert result["failed"] == 1
    pending = q.list_pending()
    assert pending[0]["next_attempt_at"] >= started + 49
    headers.replace_header("Retry-After", "9999")
    monkeypatch.setattr(sq.time, "time", lambda: started + 10_000)
    sq.flush_pending_settlements(default_base_url="https://finops.invalid", default_api_key="k")
    pending = q.list_pending()
    assert pending[0]["next_attempt_at"] <= started + 10_000 + 300.1


def test_retry_after_http_date_is_capped(tmp_path, monkeypatch):
    import io
    import urllib.error
    import urllib.request
    from datetime import datetime, timedelta, timezone
    from email.message import Message
    from email.utils import format_datetime

    monkeypatch.setenv("KAZENAI_SDK_SETTLE_AUTO_FLUSH", "0")
    monkeypatch.setattr(sq.random, "uniform", lambda _a, _b: 0)
    q = sq.reset_settle_queue_for_tests(str(tmp_path / "q"))
    q.enqueue(_body(), org_id="org-a1")
    headers = Message()
    headers["Retry-After"] = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=40), usegmt=True)

    def urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 429, "slow", headers, io.BytesIO(b"{}"))

    monkeypatch.setattr("kazenai.spine.transport.urlopen", urlopen)
    started = time.time()
    result = sq.flush_pending_settlements(default_base_url="https://finops.invalid", default_api_key="k")
    assert result["failed"] == 1
    pending = q.list_pending()
    assert pending[0]["next_attempt_at"] >= started + 30
    headers.replace_header(
        "Retry-After",
        format_datetime(datetime.now(timezone.utc) + timedelta(seconds=9999), usegmt=True),
    )
    monkeypatch.setattr(sq.time, "time", lambda: started + 10_000)
    sq.flush_pending_settlements(default_base_url="https://finops.invalid", default_api_key="k")
    pending = q.list_pending()
    assert pending[0]["next_attempt_at"] <= started + 10_000 + 300.1


def test_record_journaled_before_send_is_not_redelivered_while_its_sender_owns_it(tmp_path, monkeypatch):
    """The auto-flusher must not race the request that journaled the intent.

    Journal-before-send writes the record and then sends it. If the background
    flusher also sends it, FinOps sees the same lifecycle event twice at once and
    the sender's copy can be refused, failing a call that was allowed.
    """
    q = sq.reset_settle_queue_for_tests(str(tmp_path / "spool"))
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops.test")
    sent: list[dict] = []
    monkeypatch.setattr(guard, "_post_settle_sync", lambda base, org, body, **kw: sent.append(body) or True)

    q.enqueue(_body(event="provider_started", actual_usd_micros=None), error="pre_send", org_id="org-a1", base_url="http://finops.test")
    assert sq.flush_pending_settlements()["applied"] == 0
    assert sent == []

    # A sender that died mid-send leaves the record behind; it is recovered after the grace period.
    real_time = time.time
    monkeypatch.setattr(sq.time, "time", lambda: real_time() + sq._pre_send_grace_s() + 1)
    assert sq.flush_pending_settlements()["applied"] == 1
    assert [b["event"] for b in sent] == ["provider_started"]


def test_record_journaled_after_a_failed_send_is_due_immediately(tmp_path, monkeypatch):
    q = sq.reset_settle_queue_for_tests(str(tmp_path / "spool"))
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "http://finops.test")
    sent: list[dict] = []
    monkeypatch.setattr(guard, "_post_settle_sync", lambda base, org, body, **kw: sent.append(body) or True)
    q.enqueue(_body(), error="connect_error", org_id="org-a1", base_url="http://finops.test")
    assert sq.flush_pending_settlements()["applied"] == 1
    assert len(sent) == 1
