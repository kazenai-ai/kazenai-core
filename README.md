# kazenai-core

> **Stop your AI agents from burning your budget. Catch loops before they catch you.**

[![PyPI version](https://badge.fury.io/py/kazenai-finops.svg)](https://badge.fury.io/py/kazenai-finops)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status: Building in Public](https://img.shields.io/badge/status-building%20in%20public-orange.svg)](https://x.com/kazenai)

---

## What is KazenAI?

KazenAI is **reliability infrastructure for AI agents**.

When you run an AI agent in production, three things will eventually go wrong:

1. **It loops** — and burns your entire monthly LLM budget in 40 minutes
2. **It fails silently** — returns 200 OK but the business outcome is wrong
3. **You can't debug it** — because AI agents are non-deterministic and single-trace debugging is meaningless

KazenAI intercepts every LLM and tool call your agent makes, enforces budget limits locally (no network required), detects loops before they become expensive, and gives you full observability — with one function call.

```python
from kazenai import monitor

agent = monitor(
    agent,
    agent_id="support-agent",
    api_key="kz_...",
    max_budget_usd=5.00,
    debug=True,           # see cost per call in your terminal
)
```

That's it. Your agent code doesn't change.

---

## The problem in one screenshot

```
[KazenAI] step=1  gpt-4o-mini  cost=$0.0043  total=$0.0043  proj=$0.21/$5.00  OK
[KazenAI] step=2  gpt-4o-mini  cost=$0.0041  total=$0.0084  proj=$0.19/$5.00  OK
[KazenAI] step=8  gpt-4o-mini  cost=$0.0039  total=$0.43    proj=$4.91/$5.00  WARN:budget_87pct
[KazenAI] step=9  gpt-4o-mini  BLOCKED:budget  spent=$5.00  limit=$5.00

KazenBudgetExceeded: Run agent-001: budget $5.00 exceeded (spent $4.97, attempted $5.42)
```

No more waking up to a $47K bill.

---

## Features (MVP — shipping Week 1)

- **`monitor(agent, ...)`** — wraps LangChain, AutoGen, and CrewAI agents with one call
- **Local budget enforcement** — blocks calls before they're made, no network required, <1ms
- **Real-time debug output** — `debug=True` prints cost + status after every LLM call
- **Loop detection (H1)** — Jaccard similarity catches near-duplicate inputs before they spiral
- **Loop detection (H2)** — tool chain fingerprinting catches recursive tool patterns
- **Rate limiting** — `max_calls_per_minute` prevents runaway agents from moving too fast
- **Event sampling** — `sample_rate=0.3` to control backend traffic at scale
- **Offline resilience** — RetryQueue (SQLite) stores events locally if backend is down
- **Parent-child tracing** — `RunContext` with `parent_step_id` for multi-agent pipelines
- **Canonical event schema** — `KazenEvent` shared by SDK and backend (no field drift)
- **OpenAI patcher** — transparent monkey-patch, no code changes required
- **LangChain integration** — `KazenCallbackHandler` + `LangChainProxy`
- **AutoGen integration** — wraps `initiate_chat`
- **CrewAI integration** — wraps `kickoff()`

## Planned Modules (Roadmap)

- **AgentLens P1** — step-level trace capture dashboard (Month 2)
- **AgentLens P2** — Probabilistic Replay Engine: run any trace N times, get statistical distribution of outcomes (Month 5) — *no competitor has built this*
- **AgentLens P3** — Semantic Drift Monitor: detect when your agent's behaviour changes after a model update (Month 6)
- **TypeScript SDK** — for Node.js agent frameworks (Month 6)

---

## Installation

```bash
pip install kazenai-finops
```

Python 3.10, 3.11, 3.12 supported. No C extensions. Installs in under 30 seconds.

---

## Quick Start

### Raw OpenAI (no framework)

```python
import openai
from kazenai import monitor, KazenBudgetExceeded

client = openai.OpenAI()

# monitor() patches the OpenAI client transparently
with_monitoring = monitor(
    client,
    agent_id="my-agent",
    api_key="kz_...",          # get yours at kazenai.com
    max_budget_usd=0.50,
    debug=True,
)

try:
    for i in range(100):       # simulated loop
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Step {i}: do the thing"}],
        )
except KazenBudgetExceeded as e:
    print(f"Blocked at step {e.run_id}: spent ${e.spent:.4f}")
```

### LangChain

```python
from kazenai import monitor

chain = your_langchain_chain  # LCEL chain, agent, etc.
monitored = monitor(
    chain,
    agent_id="customer-support",
    api_key="kz_...",
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
    api_key="kz_...",
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

Apache 2.0 — use it for anything, attribution appreciated.
