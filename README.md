# kazenai

> **Stop your AI agents from burning your budget. Catch loops before they catch you.**

[![PyPI](https://img.shields.io/pypi/v/kazenai.svg)](https://pypi.org/project/kazenai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Install:** [PyPI · kazenai](https://pypi.org/project/kazenai/) · **Customer SDK:** [`kazenai-finops`](https://pypi.org/project/kazenai-finops/) · **Products:** [kazenai.com](https://kazenai.com)

---

## What is this?

`kazenai` is the core Python SDK behind KazenAI’s economic control layer for AI agents. One call — `monitor()` — wraps your OpenAI or Anthropic client and:

1. **Blocks overspend before the provider call** (`BudgetExceeded`)
2. **Detects runaway loops** before another LLM request
3. **Optionally ships evidence** to Agent FinOps / Agent Lens as canonical `KazenEvent`s

Most observability tools tell you what went wrong after the bill. KazenAI can stop the call that would cause it.

```
Traditional tools:  LLM call → response → log cost → dashboard shows overspend
KazenAI:            Pre-flight check → BLOCKED → LLM call never made
```

---

## Install

```bash
python -m pip install kazenai openai
# Most teams install the customer package instead:
python -m pip install kazenai-finops openai
```

`kazenai-finops` re-exports this SDK and is the recommended install for product integrations.

Requires Python 3.10+.

---

## Quick start

```python
import os
from openai import OpenAI
from kazenai import monitor, BudgetExceeded

# Optional FinOps ingest (env only — not monitor kwargs):
#   export KAZENAI_FINOPS_API_KEY=kz_...
#   export KAZENAI_FINOPS_INGEST_URL=https://finops.example.com

client = monitor(
    OpenAI(),
    agent_id="support-agent",
    max_budget_usd=5.00,
    debug=True,
)

try:
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
    )
except BudgetExceeded as e:
    print("hard budget cap:", e)
```

Your call sites stay the same — `monitor()` wraps the client.

With `debug=True` you’ll see spend as the run progresses:

```
[KazenAI] step=1  gpt-4o-mini  cost=$0.0043  total=$0.0043  proj=$0.21/$5.00  OK
[KazenAI] step=8  gpt-4o-mini  cost=$0.0039  total=$0.43    proj=$4.91/$5.00  WARN:budget_87pct
[KazenAI] step=9  gpt-4o-mini  BLOCKED:budget  spent=$5.00  limit=$5.00
```

Illustrative debug output — not a customer bill or savings claim.

---

## What you get

| Capability | Behavior |
|------------|----------|
| Hard budget cap | `BudgetExceeded` **before** the provider call |
| Soft trajectory pause | `KazenCircuitBreaker` after a completed call when projection trips |
| Loop detection | Blocks high-risk repeat patterns before another provider call |
| Event ingest | Optional `HttpSink` batches to Agent FinOps when URL + API key are set |
| Shared schema | Canonical `KazenEvent` via [`kazen-event-schema`](https://pypi.org/project/kazen-event-schema/) |

### Supported today

| Path | Status |
|------|--------|
| Sync OpenAI `chat.completions.create` (non-streaming) | Supported via `monitor()` |
| Sync Anthropic `messages.create` (non-streaming) | Supported via `monitor()` |
| Streaming / Responses API / async clients | Not on the supported control path |
| LangChain / CrewAI / LangGraph / AutoGen helpers | Available for evaluation — wrap the underlying client with `monitor()` for the supported path |

---

## Why local enforcement matters

- **Hard cap works without FinOps network** — deny even when ingest is offline
- **Low overhead** — budget and loop checks stay on the hot path
- **Backend outage ≠ unprotected spend** on the local hard-cap path

---

## Optional FinOps environment variables

| Variable | Purpose |
|----------|---------|
| `KAZENAI_FINOPS_INGEST_URL` / `KAZENAI_FINOPS_URL` | Base URL for event ingest |
| `KAZENAI_FINOPS_API_KEY` | API key for `POST /v1/events` |
| `KAZENAI_BUDGET_USD` | Soft per-run budget for the circuit breaker |
| `KAZENAI_ORG_ID` / `KAZENAI_PROJECT_ID` | Tenant labels on events |

---

## Packages in the stack

| Package | Role |
|---------|------|
| [`kazenai`](https://pypi.org/project/kazenai/) | Core `monitor()` engine (this repo) |
| [`kazenai-finops`](https://pypi.org/project/kazenai-finops/) | Customer-facing SDK (recommended install) |
| [`kazen-event-schema`](https://pypi.org/project/kazen-event-schema/) | Shared event contract across FinOps / Lens / SDKs |

---

## Contributing

Issues and PRs that improve budget/loop reliability are welcome. Keep claims aligned with the supported paths above.

Products and design-partner enquiries: [kazenai.com](https://kazenai.com) · **founder@kazenai.com**

---

## License

Apache License 2.0. See LICENSE and NOTICE.
