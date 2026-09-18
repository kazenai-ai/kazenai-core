# kazenai-core

> **Control FINAL_1 scope:** Certified path is sync non-streaming OpenAI Chat Completions + Anthropic Messages via `monitor()` — see `docs/integrations/control-supported-matrix.md`. Framework adapters (LangChain/CrewAI/LangGraph/AutoGen) are **not Control-certified** in FINAL_1.


> **Stop your AI agents from burning your budget. Catch loops before they catch you.**

[![Local package](https://img.shields.io/badge/package-local%20v1.0.1-blue.svg)](../WORKSPACE.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![LLM calls guarded](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kazenai-ai/kazenai-finops-sdk/main/badge/llm-guard.json)](https://github.com/kazenai-ai/kazenai-finops-sdk/blob/main/scripts/audit_llm_calls_all.py)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Quickstart:** [kazenai.com/onboarding](https://kazenai.com/onboarding) · Customer package: `kazenai-finops`

> Publishing status: this checkout uses local sibling-path installs for
> `kazenai` v1.0.1. PyPI currently does not provide this workspace version.

---

## What is KazenAI?

KazenAI is **reliability infrastructure for AI agents**.

When you run an AI agent in production, three things will eventually go wrong:

1. **It loops** — and burns your entire monthly LLM budget in 40 minutes
2. **It fails silently** — returns 200 OK but the business outcome is wrong
3. **You can't debug it** — because AI agents are non-deterministic and single-trace debugging is meaningless

KazenAI intercepts every LLM and tool call your agent makes, enforces budget limits locally (no network required), detects loops before they become expensive, and gives you full observability — with one function call.

```python
import os
from openai import OpenAI
from kazenai import monitor, BudgetExceeded

# API key for FinOps ingest is env-only (not a monitor kwarg):
#   export KAZENAI_FINOPS_API_KEY=kz_...
#   export KAZENAI_FINOPS_INGEST_URL=http://127.0.0.1:8090

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

That's it. Your agent code doesn't change.

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

No more waking up to a $47K bill.

---

## Features (shipped in this repo)

- **`monitor(client, ...)`** — OpenAI-compatible client wrapper with budget + loop enforcement
- **`FinOpsController`** — trajectory projection + soft circuit breaker (`KazenCircuitBreaker`)
- **`HttpSink`** — batch ingest to `kazenai-agent-finops` with offline `RetryQueue`
- **Canonical `KazenEvent`** — shared schema with orchestrator + FinOps API
- **Framework integrations** (see `examples/`):
  - **LangChain** — `KazenCallbackHandler` + optional `wrap_langchain_runnable()`
  - **CrewAI** — `wrap_crew_kickoff()` using `RunContext`
  - **LangGraph** — `wrap_graph_invoke()` using `RunContext`
- **AutoGen** — planned; not yet in this package

### Environment variables (FinOps ingest)

| Variable | Purpose |
|----------|---------|
| `KAZENAI_FINOPS_INGEST_URL` | Base URL (e.g. `http://127.0.0.1:8090`) |
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

Future capabilities (probabilistic replay, drift monitor, TypeScript SDK) are listed in [../docs/ROADMAP.md](../docs/ROADMAP.md). AgentLens P2/P3 are **scaffold** stage, not shipped products.

---

## Installation

Local (sibling checkout of `kazen-event-schema`):

```bash
pip install -e ../kazen-event-schema
pip install --no-deps -e .
```

CI installs pinned wheels from `vendor/` first (`kazen-event-schema` and
`kazenai-contracts`), because neither is on PyPI yet.

To keep Cursor out of GitHub contributors, enable the strip hook once per clone:

```bash
git config core.hooksPath .githooks
```

Python 3.10, 3.11, 3.12 supported. No C extensions. Installs in under 30 seconds.

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

### Raw OpenAI (no framework)

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

### LangChain

```python
from kazenai import monitor

chain = your_langchain_chain  # LCEL chain, agent, etc.
monitored = monitor(
    chain,
    agent_id="customer-support",
    # api_key is env-only for monitor(); framework adapters may take api_key separately
    max_budget_usd=2.00,
    debug=True,
)

result = monitored.invoke({"input": "help me with my order"})
```

### CrewAI

```python
from kazenai import monitor

crew = YourCrew()
monitored = monitor(
    crew,
    agent_id="research-crew",
    # api_key is env-only for monitor(); framework adapters may take api_key separately
    max_budget_usd=10.00,
    h2_max_reps=3,   # block if same tool chain repeats 3 times
)

result = monitored.kickoff(inputs={"topic": "AI trends"})
```

---

## Why local enforcement matters

Most observability tools record what happened. **KazenAI blocks what's about to happen.**

```
Traditional tools:  LLM call → response → log cost → dashboard shows $47K
KazenAI:            Pre-flight check → BLOCKED → LLM call never made
```

Local enforcement means:
- **No network dependency** — works with `backend_url=None`
- **No latency added** — budget check completes in <1ms
- **No backend outage = no protection failure** — the agent doesn't need to reach our servers to be protected

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
│   ├── cost_tracker.py      # 14-model pricing table
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
│   ├── basic_agent.py       # raw OpenAI example
│   ├── langchain_example.py
│   ├── loop_example.py      # trigger loop detection
│   └── benchmark_latency.py # verify <5ms overhead
├── tests/
├── pyproject.toml
└── README.md
```

---

## Design principles

1. **Local-first enforcement** — SDK must block without backend
2. **Zero-blocking** — SDK overhead <5ms on hot path
3. **Fail-open always** — internal errors never crash your agent
4. **Canonical schema** — KazenEvent used by both SDK and backend (`extra='forbid'`)
5. **DX over features** — `debug=True` gives value in 2 minutes

---

## Star this repo ⭐

If you've ever woken up to an unexpected LLM bill, or spent hours debugging an agent that returned 200 OK but did nothing useful — **star this repo**. It tells us this matters to you, and it helps us ship faster.

We're building in public. Follow [@kazenai](https://x.com/kazenai) for weekly progress updates.

---

## Status

**Week 1** — Building core SDK  
**Week 2** — Design partner onboarding  
**Week 3** — Hosted dashboard + paid tiers  
**Week 4** — Public launch

Early access: [kazenai.com](https://kazenai.com) or email **founder@kazenai.com**

---

## Contributing

We're pre-1.0 and moving fast. The best way to contribute right now is:

1. ⭐ Star the repo
2. Open an issue describing a pain point you've hit with AI agent costs or loops
3. Try the examples and report what breaks

Full contribution guide coming with v1.0.

---

## License

Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.
