# kazenai

> **Stop your AI agents from burning your budget. Catch loops before they catch you.**

[![PyPI](https://img.shields.io/pypi/v/kazenai.svg)](https://pypi.org/project/kazenai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Install:** [PyPI · kazenai](https://pypi.org/project/kazenai/) · **Customer SDK:** [`kazenai-finops`](https://pypi.org/project/kazenai-finops/) · **Docs:** [docs.kazenai.com](https://docs.kazenai.com/) · **Products:** [kazenai.com](https://kazenai.com)

**Source:** [github.com/kazenai-ai/kazenai-core](https://github.com/kazenai-ai/kazenai-core)

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
python -m pip install "kazenai[openai]==1.1.2"
# Most teams install the customer package instead:
python -m pip install "kazenai-finops[openai]==1.1.2"
# Anthropic:
# python -m pip install "kazenai-finops[anthropic]==1.1.2"
# Both providers:
# python -m pip install "kazenai-finops[providers]==1.1.2"
```

`kazenai-finops` re-exports this SDK and is the recommended install for product integrations.
Provider extras bound the SDK ranges so a first-run install cannot silently
resolve OpenAI 3.x or Anthropic 1.x.

Tested cells are OpenAI 1.109.1 and an isolated OpenAI 2.48.0 cell within the
install range `openai>=1.40,<3`, plus Anthropic 0.125.0 within
`anthropic>=0.39,<1`. Other versions in those ranges are compatible candidates,
not independently verified versions. Verification uses synthetic provider
transports and real local KazenAI services; the OpenAI and Anthropic paths are
also verified against the live OpenAI and Anthropic APIs on 2026-10-12 (gpt-6-luna, claude-sonnet-5-5) (`tests/test_live_provider_smoke.py`, opt-in). Source of truth:
[docs/integrations/control-supported-matrix.md](docs/integrations/control-supported-matrix.md).

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
| OpenAI `chat.completions.create`, sync/async, non-streaming and streaming | Fixture-verified and live-verified (2026-10-12) via `monitor()` |
| Anthropic `messages.create`, sync/async, non-streaming and streaming | Fixture-verified and live-verified (2026-10-12) via `monitor()` |
| OpenAI Responses API / Realtime | Unsupported on the Control path |
| LangChain | Fixture- and live-verified (2026-10-12) when the actual provider client is monitored/bound (`ChatOpenAI` via `bind_monitored_chat_openai`, `ChatAnthropic` via `bind_monitored_chat_anthropic`) |
| LangGraph | Fixture- and live-verified (2026-10-12) with an explicit declared boundary; arbitrary raw clients in nodes are not intercepted |
| LiteLLM proxy | Live-verified (2026-10-12): app-owned and gateway-owned accounting with Agent FinOps; gateway-owned streaming settlement not yet verified |
| CrewAI | Not in the supported matrix. 1.1.2 routes CrewAI 1.15.26 LLM/task/tool events to the owning crew (tested on the real event bus); cancellation and native `akickoff` cells are still unverified |
| AutoGen | Unsupported |

### Streaming

```python
from kazenai import StreamCutoffError, monitor
from openai import OpenAI

client = monitor(
    OpenAI(),
    max_budget_usd=1.00,
    stream_cutoff_usd=0.01,  # optional local observable-output guard
)

try:
    with client.chat.completions.create(
        model="gpt-4o-mini",
        stream=True,
        messages=[{"role": "user", "content": "Explain this result"}],
    ) as stream:
        for chunk in stream:
            ...
except StreamCutoffError:
    # The client attempted to close future output. Final provider billing may
    # remain pending when authoritative final usage was not received.
    pass
```

`monitor()` requests final OpenAI usage and settles exact cost when authoritative
usage arrives. OpenAI `chat.completions.stream(...)`, Anthropic
`messages.create(..., stream=True)`, and the
`messages.stream(...)` context manager follow the same finalize-once accounting
lifecycle. A shared reservation is marked in-flight at provider dispatch. Early
cancellation, provider failure, or missing final usage moves it to pending
reconciliation—never an exact zero and never released as an unstarted call.

`stream_cutoff_usd` estimates only output observable by the client. It cannot see
hidden reasoning or guarantee that the provider stopped generating or billing
immediately. `stream_enforcement=True` is deprecated.

The official lazy OpenAI helper is supported as well. Its reservation is marked
`provider_started` when the context manager is entered:

```python
with client.chat.completions.stream(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain this result"}],
) as stream:
    for event in stream:
        ...
```

---

## Local and shared enforcement

- **Hard cap works without FinOps network** — deny even when ingest is offline
- **Low overhead** — budget and loop checks stay on the hot path
- **Shared policies can fail closed** — production/staging modes deny when a
  required shared reservation cannot be obtained

An explicitly configured development fail-open path retains only the local
client safeguards. It does not preserve a cross-process or account-wide budget
guarantee.

### Multi-model workflows

Each `monitor(..., max_budget_usd=...)` client has its own cap, so a workflow
with one client per provider would add those caps up. Share one instead:

```python
from kazenai import SharedBudget, monitor

budget = SharedBudget(max_usd=1.00)
planner = monitor(OpenAI(), budget=budget)
writer = monitor(Anthropic(), budget=budget)
```

In LangChain, bind chat models with `bind_monitored_chat_openai` /
`bind_monitored_chat_anthropic` (both accept `budget=`). In LangGraph, declare
every provider client the nodes use:
`declare_langgraph_control_boundary(graph, planner, writer)` then
`wrap_graph(graph, capability="control", guarded_clients=[planner, writer])`.
Across processes or machines, give every client the same `workflow_id` and set a
workflow limit in Agent FinOps.

A provider refusal before any output (HTTP 4xx such as unknown model, invalid
key or rate limit) settles at exactly $0; timeouts and 5xx stay pending for
reconciliation. Models with no known price are blocked before dispatch unless
priced via `KAZENAI_PRICING_JSON`.

---

## Optional FinOps environment variables

| Variable | Purpose |
|----------|---------|
| `KAZENAI_FINOPS_INGEST_URL` / `KAZENAI_FINOPS_URL` | FinOps base URL for shared reservations and event ingest |
| `KAZENAI_FINOPS_API_KEY` | Credential for FinOps budget and event requests |
| `KAZENAI_DEPLOYMENT_MODE` | `production` / `staging` require fail-closed shared reservations |
| `KAZENAI_FINOPS_RESERVE_TIMEOUT_S` | Timeout for a shared pre-call reservation |
| `KAZENAI_TIMELINE_PATH` | Optional local JSONL evidence path |
| `KAZENAI_SDK_SETTLE_QUEUE_DIR` | Persistent private directory for pending settlement intents |

Pass `max_budget_usd` (or `budget=SharedBudget(...)`), `org_id`, `project_id`,
`workspace_id` and attribution fields directly to `monitor()`; they are not
inferred from similarly named environment variables.

For shared Control deployments, read [Settlement recovery](docs/settlement-recovery.md).
The current queue requires an explicitly supervised flusher and one process per
queue directory; it is not a general multiprocess job queue.

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

Maintainers: follow [RELEASING.md](RELEASING.md) for the clean-checkout and
public-PyPI verification sequence.

Products and design-partner enquiries: [kazenai.com](https://kazenai.com) · **founder@kazenai.com**

---

## License

Apache License 2.0. See LICENSE and NOTICE.
