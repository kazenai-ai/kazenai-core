# kazenai — runtime spend, policy & audit enforcement for AI agents

Drop it in front of any model or framework. It hard-caps spend **before** the call,
kills runaway loops, and writes a policy-linked audit trail you can sell on later.

Provider-agnostic (OpenAI, Anthropic, Bedrock, anything callable). Works fully
in-process — no backend required to start enforcing.

---

## Install

```bash
pip install kazenai
# local dev (until the 1.0.1 build is live on PyPI):
#   pip install -e ./kazenai-core
```

Only dependency footprint is the event schema + httpx/pydantic — it does not pull
in the rest of the KazenAI workspace.

---

## The whole product surface (6 imports)

```python
from kazenai import (
    monitor,            # zero-code drop-in: wrap a client, get enforcement + audit
    guarded_llm_call,   # explicit guard for ANY provider / custom call
    BudgetExceeded,     # raised when a hard cap is hit
    KazenCircuitBreaker,# raised when the breaker opens (runaway / overspend)
    StreamCutoffError,  # raised when a stream is cut mid-flight on budget
    FinOpsConfig,       # optional: tune soft-pause %, loop threshold
)
```

Everything else in the package is internal/advanced — ignore it.

---

## Mode 1 — zero-code drop-in (in-process enforcement)

```python
import openai
from kazenai import monitor, KazenCircuitBreaker

client = monitor(
    openai.OpenAI(),
    org_id="acme",
    max_budget_usd=5.00,      # hard ceiling for this client/run
    stream_enforcement=True,  # cut a stream mid-flight if it would blow the cap
    timeline_path="audit.jsonl",  # tamper-evident local audit log
)

try:
    while True:  # a runaway agent loop
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "keep going..."}],
        )
except KazenCircuitBreaker as e:
    print("stopped before the bill ran away:", e)
```

No FinOps service needed. `monitor()` enforces the budget, opens a circuit breaker
on runaway/overspend, runs loop-anomaly detection, and emits the `KazenEvent`
stream locally (`audit.jsonl`). Anthropic clients are auto-detected.

## Mode 2 — hosted / atomic enforcement (multi-process, shared budgets)

Point it at the FinOps service for atomic, cross-process reservation
(run/day/month windows + per-run step caps, Redis-backed, fail-closed):

```bash
export KAZENAI_FINOPS_INGEST_URL="https://finops.yourco.com"
export KAZENAI_FINOPS_API_KEY="sk-..."
```

Same `monitor(...)` call — it now reserves against the shared budget before every
call and reconciles actuals after. In production deployment modes this is
fail-closed: if the budget service is unreachable, the call is denied, not waved through.

## Mode 3 — explicit guard (any provider / custom function)

```python
from kazenai import guarded_llm_call

resp = guarded_llm_call(
    lambda: my_provider.generate(prompt),
    model="claude-opus-4-8",
    org_id="acme",
    run_id="run-123",
    projected_cost_usd=0.02,   # pre-call reservation amount
    feature="summarizer",
)
```

Reserve → call → reconcile actual cost → emit `model.call` + budget events.
`aguarded_llm_call` is the async variant.

---

## The audit trail (this is the asset, not just a log)

Every call emits structured `KazenEvent`s:

| event_type | when |
|---|---|
| `finops.budget.reserve` | pre-call reservation granted |
| `finops.budget.denied` | reservation refused (cap/step limit) |
| `model.call` | call completed, with `tokens_used`, `cost_usd`, `latency_ms` |
| `finops.circuit_breaker.opened` | runaway/overspend tripped the breaker |
| `finops.loop.anomaly` | repeated near-identical calls detected |

Sinks: in-memory (always), `timeline_path` JSONL (local audit), and HTTP to the
FinOps service when configured. This per-org event stream is the compounding,
hard-to-copy data layer — keep it from day one.

---

## Environment variables

| Var | Purpose |
|---|---|
| `KAZENAI_FINOPS_INGEST_URL` | hosted FinOps base URL (enables atomic reserve) |
| `KAZENAI_FINOPS_API_KEY` | auth for the FinOps service |
| `KAZENAI_TIMELINE_PATH` | path for the local JSONL audit timeline |
| `KAZENAI_FINOPS_RESERVE_TIMEOUT_S` | reserve call timeout (default 0.8s) |
| `KAZENAI_DEPLOYMENT_MODE` | `production`/`staging` → enforcement fail-closed |

---

## Before you demo this to anyone

This is a **cost** product. The number has to be right. There is a known
mispricing in the usage→cost matching path (the per-1k rate table itself is
correct; the bug is in model-name resolution / `from_usage`). Fix and add a
regression test pinning real provider invoices before the first design-partner
call — a wrong number is the one thing that kills a FinOps tool on contact.
