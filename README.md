# kazenai (kazenai-core)

> **Control FINAL_1 scope:** Certified path is sync non-streaming OpenAI Chat Completions + Anthropic Messages via `monitor()` — see `docs/integrations/control-supported-matrix.md`. Framework adapters (LangChain/CrewAI/LangGraph/AutoGen) are **not Control-certified** in FINAL_1.


> **Stop your AI agents from burning your budget. Catch loops before they catch you.**

[![PyPI](https://img.shields.io/pypi/v/kazenai.svg)](https://pypi.org/project/kazenai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Install:** [PyPI · kazenai](https://pypi.org/project/kazenai/) · Customer package: [`kazenai-finops`](https://pypi.org/project/kazenai-finops/) · **Products:** [kazenai.com](https://kazenai.com)

> This repository publishes to PyPI as **`kazenai`** (version aligned with `1.0.3`). For most agent integrations, install **`kazenai-finops`**, which depends on this package. Editable sibling-path installs are for workspace contributors only.

---

## What is KazenAI?

KazenAI provides **Agent FinOps** reliability primitives for AI agents: local budget checks, loop detection, and optional event ingest into Agent FinOps / Agent Lens.

When you run an AI agent in production, cost and control failures are common:

1. **It loops** — and burns LLM budget quickly
2. **It fails with weak evidence** — HTTP 200 with a wrong business outcome
3. **Debugging is hard** — non-deterministic traces without a shared event model

`monitor()` intercepts supported LLM calls, enforces budget limits locally (no network required for the hard cap), detects loops before another provider call, and can emit canonical `KazenEvent` batches when FinOps ingest is configured.

```python
import os
from openai import OpenAI
from kazenai import monitor, BudgetExceeded

# API key for FinOps ingest is env-only (not a monitor kwarg):
#   export KAZENAI_FINOPS_API_KEY=kz_...
#   export KAZENAI_FINOPS_INGEST_URL=https://finops.example.com

client = monitor(
    OpenAI(),
    agent_id="support-agent",
    max_budget_usd=5.00,
    debug=True,           # see cost per call in your terminal
)

try:
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
    )
except BudgetExceeded as e:
    print("hard local/shared cap:", e)
```

That's it for the Control-certified sync OpenAI path. Your call sites stay the same.

---

## The problem in one screenshot

```
[KazenAI] step=1  gpt-4o-mini  cost=$0.0043  total=$0.0043  proj=$0.21/$5.00  OK
[KazenAI] step=2  gpt-4o-mini  cost=$0.0041  total=$0.0084  proj=$0.19/$5.00  OK
[KazenAI] step=8  gpt-4o-mini  cost=$0.0039  total=$0.43    proj=$4.91/$5.00  WARN:budget_87pct
[KazenAI] step=9  gpt-4o-mini  BLOCKED:budget  spent=$5.00  limit=$5.00

BudgetExceeded: cumulative_cost exceeded (hard pre-call cap)
# Soft trajectory pause raises KazenCircuitBreaker after a completed call
```

Illustrative debug output — not a customer bill or certified savings claim.

---

## Features (shipped in this package)

- **`monitor(client, ...)`** — Control-certified for sync OpenAI Chat Completions + Anthropic Messages
- **`FinOpsController`** — trajectory projection + soft circuit breaker (`KazenCircuitBreaker`)
- **`HttpSink`** — batch ingest to Agent FinOps with offline `RetryQueue`
- **Canonical `KazenEvent`** — via dependency on `kazen-event-schema`
- **Framework integrations** (see `examples/`): present for evaluation — **not Control-certified** in FINAL_1
  - LangChain, CrewAI, LangGraph helpers
  - AutoGen helper may lag; treat as experimental

### Environment variables (FinOps ingest)

| Variable | Purpose |
|----------|---------|
| `KAZENAI_FINOPS_INGEST_URL` / `KAZENAI_FINOPS_URL` | Base URL for ingest |
| `KAZENAI_FINOPS_API_KEY` | API key for `POST /v1/events` |
| `KAZENAI_BUDGET_USD` | Per-run soft budget for circuit breaker |
| `KAZENAI_ORG_ID` / `KAZENAI_PROJECT_ID` | Tenant labels on events |


## Certified Control provider contract (FINAL_1)

| Mode | Status |
|------|--------|
| Sync OpenAI `chat.completions.create` (non-streaming) | **Certified** via `monitor()` |
| Sync Anthropic `messages.create` (non-streaming) | **Certified** via `monitor()` |
| OpenAI Responses API / async clients / Control streaming | **`UnsupportedModeError`** |
| Soft trajectory pause | `KazenCircuitBreaker` (after a completed call) |
| Hard local/shared budget deny | `BudgetExceeded` (before provider) |

`kazenai_finops.KazenBudgetExceeded` is a **deprecated alias of** `KazenCircuitBreaker`, not hard `BudgetExceeded`.

## Roadmap

Future capabilities (broader adapters, replay, TypeScript SDK) are product/roadmap items — see [../docs/ROADMAP.md](../docs/ROADMAP.md) where present. Do not treat them as Control-certified from this README alone.

---

## Installation

```bash
python -m pip install kazenai
# Typical customer install (re-exports + FinOps extras):
python -m pip install kazenai-finops openai
```

### Workspace / contributor install (optional)

```bash
pip install -e ../kazen-event-schema
pip install --no-deps -e .
```

CI may install pinned wheels from `vendor/` for `kazen-event-schema` and
`kazenai-contracts`. **`kazen-event-schema` is on PyPI**; `kazenai-contracts` remains
workspace/vendor until published separately.

To keep Cursor out of GitHub contributors, enable the strip hook once per clone:

```bash
git config core.hooksPath .githooks
```

Python 3.10, 3.11, 3.12 supported. No C extensions.

---

## Quickstart

```python
from openai import OpenAI
from kazenai import monitor

# export KAZENAI_FINOPS_API_KEY=...   # optional ingest; not a monitor kwarg
monitored_client = monitor(
    OpenAI(),
    agent_id="my-run-001",
    max_budget_usd=5.00,
)

result = monitored_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello KazenAI"}],
)
```

---

## Quick Start

### Raw OpenAI (Control-certified)

```python
import openai
from kazenai import monitor, BudgetExceeded, KazenCircuitBreaker, LoopDetected

client = openai.OpenAI()

# monitor() patches sync chat.completions (certified Control path).
# FinOps API key: KAZENAI_FINOPS_API_KEY env (not a kwarg).
monitor(
    client,
    agent_id="my-agent",
    max_budget_usd=0.50,
    debug=True,
)

try:
    for i in range(100):       # simulated loop
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Step {i}: do the thing"}],
        )
except BudgetExceeded as e:
    print(f"Hard cap before provider: {e}")
except LoopDetected as e:
    print(f"Loop blocked before provider: {e}")
except KazenCircuitBreaker as e:
    print(f"Soft pause after a completed call: {e}")
```

### LangChain / CrewAI (not Control-certified)

Framework wrappers may exist for evaluation. Prefer wrapping the OpenAI/Anthropic
client with `monitor()` for the FINAL_1 supported path. See
`docs/integrations/control-supported-matrix.md`.

---

## Why local enforcement matters

Most observability tools record what happened. **On the certified path, KazenAI can block what's about to happen.**

```
Traditional tools:  LLM call → response → log cost → dashboard shows overspend
KazenAI (local):    Pre-flight check → BLOCKED → LLM call never made
```

Local enforcement means:
- **Works for the hard cap without FinOps network** — `backend_url=None` still denies
- **Low overhead** — budget check is local on the hot path
- **Backend outage ≠ unprotected hard-cap path** — optional ingest may still fail open depending on configuration

---

## Repository Structure

```
kazenai-core/
├── kazenai/
│   ├── __init__.py          # public API: monitor()
│   ├── schema.py            # canonical KazenEvent (shared SDK + backend)
│   ├── context.py           # RunContext with parent_step_id
│   ├── monitor.py           # monitor() entry point
│   ├── interceptor.py       # LLM/tool call interception
│   ├── enforcement.py       # local budget + rate limit enforcement
│   ├── loop_detector.py     # H1 (Jaccard) + H2 (chain fingerprint)
│   ├── cost_tracker.py      # pricing table
│   ├── client.py            # async API client + sampling
│   ├── retry_queue.py       # SQLite retry queue for offline resilience
│   ├── debug.py             # debug=True terminal output
│   ├── config.py            # pydantic-settings
│   └── integrations/
│       ├── langchain.py
│       ├── autogen.py
│       ├── crewai.py
│       └── generic.py
├── examples/
├── tests/
├── pyproject.toml
└── README.md
```

---

## Design principles

1. **Local-first hard-cap enforcement** — SDK can deny without backend
2. **Low overhead** — keep hot-path checks fast
3. **Fail-open for internal SDK faults** — SDK bugs should not crash the agent (budget deny is intentional)
4. **Canonical schema** — `KazenEvent` shared with FinOps / Lens
5. **Honest Control matrix** — do not claim framework certification without evidence

---

## Status

**PyPI:** `kazenai` **1.0.3** published. Control FINAL_1 certifies the sync OpenAI + Anthropic `monitor()` path above — not a general "every agent framework" claim, and not customer production certification by itself.

Products and design-partner enquiries: [kazenai.com](https://kazenai.com) · **founder@kazenai.com**

---

## Contributing

1. Star or open an issue describing a cost/loop pain point
2. Try the Control-certified examples and report what breaks
3. Keep PRs aligned with the Control supported matrix when claiming certification

---

## License

Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.
