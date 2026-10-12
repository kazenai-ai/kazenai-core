# Settlement recovery

The Control lifecycle journals an exact settlement request before attempting to
deliver it to FinOps. A record is removed only after FinOps returns
`result=applied` or `result=duplicate` for the matching intent fingerprint.
HTTP success with `result=pending` remains queued.

## Deployment requirements

1. Set `KAZENAI_SDK_SETTLE_QUEUE_DIR` to a persistent directory owned by the
   SDK process. Keep directory mode `0700` and record mode `0600`.
2. Run the recovery command on a schedule (cron, systemd timer, or a sidecar
   loop) with the trusted FinOps URL and API key in its environment. The SDK
   does not start it automatically.

   ```bash
   KAZENAI_SDK_SETTLE_QUEUE_DIR=/var/lib/myapp/kazenai-settle \
   KAZENAI_FINOPS_URL=https://finops.example.com \
   KAZENAI_FINOPS_API_KEY=... \
   python -m kazenai.spine.settle_queue flush
   ```

   Exit codes: `0` all delivered, `1` some records still pending, `3` another
   flusher holds the queue lock. Flushes take an exclusive lock
   (`.flush.lock`), so overlapping schedules cannot double-deliver.
3. Export backlog signals from `python -m kazenai.spine.settle_queue stats`
   (or `kazenai.spine.settle_queue.queue_stats()`): `depth`, `oldest_age_s`,
   `quarantined`, `last_flush_at`, `last_flush_ok_at`, `last_flush_failed`.
   Alert on growing `oldest_age_s` and any non-zero `quarantined`.
4. Many SDK processes on one host may share a queue directory. Each
   `(org, FinOps URL, idempotency_key)` binds exactly one intent fingerprint
   through an atomic claim file (`claims/`); a conflicting reuse raises
   `SettleIntentConflict` in every process.
5. Keep credentials in runtime configuration. API keys are not written to queue
   files unless a caller explicitly requests persistence.

## Proven guarantees

- Sync and async settlement paths journal before network delivery.
- Process cancellation, connection failure, and the tested `os._exit` boundary
  retain the pending record.
- A local public-path proof replays the retained intent into FinOps/PostgreSQL.
- Intent identity includes destination, organization, reservation/call/attempt,
  lifecycle event, and known amount.

## Limits

- This is a settlement delivery queue, not durable agent execution.
- There is no claim of exactly-once provider execution or exactly-once billing.
- Record, claim, and directory metadata are fsynced on POSIX. Host/power-loss
  durability still depends on the filesystem honouring fsync and has not been
  independently proven.
- Corrupt or malformed records are moved to `quarantine/` with a WARNING log
  and counted in `stats`; they are never redelivered automatically.
- A claim orphaned by a crash (no pending record) blocks a *different* intent
  for the same key for `KAZENAI_SDK_SETTLE_STALE_CLAIM_S` (default 300 s).
- Multiple hosts require a shared transactional queue or one designated
  recovery owner; a local filesystem path is not cross-host coordination.
