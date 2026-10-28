# Settlement recovery

The Control lifecycle journals an exact settlement request before attempting to
deliver it to FinOps. A record is removed only after FinOps returns
`result=applied` or `result=duplicate` for the matching intent fingerprint.
HTTP success with `result=pending` remains queued.

## Deployment requirements

1. Set `KAZENAI_SDK_SETTLE_QUEUE_DIR` to a persistent directory owned by the
   SDK process. Keep directory mode `0700` and record mode `0600`.
2. The SDK starts one background flusher in each process after the first
   record is journaled (importing the package starts nothing).
   `KAZENAI_SDK_SETTLE_AUTO_FLUSH=0` turns it off. Records left behind by a
   process that exited are delivered by the next process that flushes, or by
   the recovery command run on a schedule (cron, systemd timer, or a sidecar
   loop) with the trusted FinOps URL and API key in its environment:

   ```bash
   KAZENAI_SDK_SETTLE_QUEUE_DIR=/var/lib/myapp/kazenai-settle \
   KAZENAI_FINOPS_URL=https://finops.example.com \
   KAZENAI_FINOPS_API_KEY=... \
   python -m kazenai.spine.settle_queue flush
   ```

   Exit codes: `0` all delivered, `1` some records still pending or quarantined
   as rejected, `3` another flusher holds the queue lock. Flushes take an exclusive lock
   (`.flush.lock`), so overlapping schedules cannot double-deliver. Failed
   deliveries back off: each record stores `attempts` and `next_attempt_at`,
   and a `Retry-After` from FinOps (seconds or an HTTP date) is honoured up to
   300 seconds. A record journaled just before its send belongs to that send:
   the flusher leaves it alone for 10 seconds (or four times
   `KAZENAI_FINOPS_RESERVE_TIMEOUT_S`, if longer), so a live call and the
   flusher never deliver the same event at once. If the send ends without an
   acknowledgement, the record is handed to the flusher immediately.
3. Export backlog signals from `python -m kazenai.spine.settle_queue stats`
   (or `kazenai.spine.settle_queue.queue_stats()`): `depth`, `oldest_age_s`,
   `quarantined`, `last_flush_at`, `last_flush_ok_at`, `last_flush_failed`.
   Alert on growing `oldest_age_s` and any non-zero `quarantined`.
4. Many SDK processes on one host may share a queue directory. Each
   `(org, FinOps URL, idempotency_key)` binds exactly one intent fingerprint
   through an atomic claim file (`claims/`); a conflicting reuse raises
   `SettleIntentConflict` in every process. The queue directory must be on a
   local filesystem that supports hard links and `flock` (not NFS); otherwise
   journaling fails and the call is refused rather than sent unjournaled.
5. Keep credentials in runtime configuration. API keys are not written to queue
   files unless a caller explicitly requests persistence.
6. A flush sends records only to the process's configured FinOps URL, with the
   process's own key, never a URL or key stored in a record. A record journaled
   for a different FinOps URL is moved to `quarantine/` as
   `destination_mismatch` and not sent. Check the configured URL, then replay
   the record by hand if it is genuinely yours.

## Guarantees

- Sync and async settlement paths journal before network delivery.
- Process cancellation, connection failure and abrupt process exit retain the
  pending record.
- A retained intent replays into Agent FinOps exactly as recorded.
- Intent identity includes destination, organization, reservation/call/attempt,
  lifecycle event, and known amount.

## Limits

- This is a settlement delivery queue, not durable agent execution.
- There is no claim of exactly-once provider execution or exactly-once billing.
- Record, claim and directory metadata are fsynced on POSIX. Durability across
  host power loss depends on the filesystem honouring fsync.
- Corrupt or malformed records are moved to `quarantine/` with a WARNING log
  and counted in `stats`; they are never redelivered automatically.
- A record FinOps refuses for good (HTTP 409 `idempotency_payload_mismatch`, or
  a 400/422 validation error) is quarantined as `rejected_<code>` instead of
  being retried on every flush. FinOps already holds a different intent for
  that key; review the quarantined record before replaying anything by hand.
- Claims are removed only under a lock (`claims/.lock`) and only while they
  still bind the expected fingerprint, so a takeover cannot drop another
  writer's claim.
- A writer that stalls longer than `KAZENAI_SDK_SETTLE_STALE_CLAIM_S` between
  claiming and journaling can lose its claim to another intent. It then
  withdraws its own record and raises `SettleIntentConflict`, so only one
  intent stays journaled.
- A claim orphaned by a crash (no pending record) blocks a *different* intent
  for the same key for `KAZENAI_SDK_SETTLE_STALE_CLAIM_S` (default 300 s).
- Multiple hosts require a shared transactional queue or one designated
  recovery owner; a local filesystem path is not cross-host coordination.
