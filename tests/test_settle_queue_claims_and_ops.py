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
        result = sq.flush_pending_settlements()
    assert seen == ["conflict"]
    assert result == {"applied": 1, "failed": 0}
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
        assert sq.flush_pending_settlements()["failed"] == 1
    stats = q.stats()
    assert stats["last_flush_failed"] == 1
    assert stats["last_flush_ok_at"] is None

    with patch.object(guard, "_post_settle_sync", lambda *a, **k: True):
        assert sq.flush_pending_settlements() == {"applied": 1, "failed": 0}
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
        assert sq.main(["flush", "--root", root]) == 1
    with patch.object(guard, "_post_settle_sync", lambda *a, **k: True):
        assert sq.main(["flush", "--root", root]) == 0
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
