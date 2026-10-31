# kazenai

> **A hard spending limit for every customer's AI.**

[![PyPI](https://img.shields.io/pypi/v/kazenai.svg)](https://pypi.org/project/kazenai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/kazenai-ai/kazenai-core/blob/main/LICENSE)

**Install:** [PyPI · kazenai](https://pypi.org/project/kazenai/) · **Customer SDK:** [`kazenai-finops`](https://pypi.org/project/kazenai-finops/) · **Docs:** [docs.kazenai.com](https://docs.kazenai.com/) · **Products:** [kazenai.com](https://kazenai.com)

**Source:** [github.com/kazenai-ai/kazenai-core](https://github.com/kazenai-ai/kazenai-core)

---

## What is this?

Flat-priced AI has one weak spot: a few customers, or one stuck agent, can cost
more than they pay. KazenAI checks every model call against that customer's
limit before it runs. Over the limit, the call doesn't happen.

`kazenai` is the core Python SDK. One call, `monitor()`, wraps your OpenAI or Anthropic client and:

1. **Blocks overspend before the provider call** (`BudgetExceeded`)
2. **Detects runaway loops** before another LLM request
3. **Connects to Agent FinOps** (optional) for a limit per customer across every process, with a receipt for every allow and deny

Most observability tools tell you what went wrong after the bill. KazenAI can stop the call that would cause it.

```
Traditional tools:  LLM call → response → log cost → dashboard shows overspend
KazenAI:            Pre-flight check → BLOCKED → LLM call never made
```

---

## Install

```bash
python -m pip install "kazenai[openai]==1.2.1"
# Most teams install the customer package instead:
python -m pip install "kazenai-finops[openai]==1.2.1"
# Anthropic:
# python -m pip install "kazenai-finops[anthropic]==1.2.1"
# Both providers:
# python -m pip install "kazenai-finops[providers]==1.2.1"
```

`kazenai-finops` re-exports this SDK and is the recommended install for product integrations.
Provider extras keep the provider SDKs within the supported ranges
(`openai>=1.40,<3`, `anthropic>=0.39,<1`). To send inline images, add the
`images` extra (`kazenai[openai,images]`): it sizes each image before the call.

KazenAI is tested on OpenAI 1.109.1 and 2.48.0 (Chat Completions), OpenAI 2.54.0
(Responses and the live runs) and Anthropic 0.125.0, and the OpenAI and Anthropic
paths are verified against the live APIs. Full list:
[Supported integrations](https://github.com/kazenai-ai/kazenai-core/blob/main/docs/integrations/control-supported-matrix.md).

Requires Python 3.10+.

---

## Quick start

```python
from openai import OpenAI
from kazenai import monitor, BudgetExceeded

client = monitor(
    OpenAI(max_retries=0),
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

This local cap needs no account. To enforce a limit per customer across every
process, connect Agent FinOps.

---

## Connect to Agent FinOps (hosted limits)

```bash
export KAZENAI_FINOPS_URL=https://finops.example.com   # your Agent FinOps URL
export KAZENAI_FINOPS_API_KEY=kz_...
export KAZENAI_TIER_MISMATCH_MULTIPLIER=2              # required, see below
```

```python
from kazenai import use_attribution

client = monitor(
    OpenAI(max_retries=0),
    agent_id="support-agent",
    org_id="<organization ID>",      # both IDs are on the Setup page in Agent FinOps
    workspace_id="<workspace ID>",
    require_customer_id=True,        # refuse a call that names no customer
)

with use_attribution(business_subject_ref="cust_123"):   # the customer's limit applies
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Summarize this ticket"}],
    )
```

An Agent FinOps API key belongs to one organization and one workspace;
`monitor()` must name the same two, or each reservation is refused with HTTP 403
and the call raises `BudgetUnavailable`. A per-customer limit applies only to
calls that carry that customer's ID (`business_subject_ref`); with
`require_customer_id=True` a call without one raises `CustomerIdRequired` before
it reaches the provider. Details, including worker threads:
[QUICKSTART](QUICKSTART.md#2-connect-a-shared-finops-authority).

Setting a FinOps URL turns on the Control lifecycle (the default since 1.2.0):

- every guarded call reserves against your limits in Agent FinOps **before** it
  is sent and settles after it returns;
- a refused reservation raises `BudgetExceeded`, and the provider is never called;
- if Agent FinOps cannot be reached, the call is refused (fail closed), unless
  an admin chose fail-open for the organization: then it runs with only the
  local caps and is not charged to the organization's limits
  ([details](https://docs.kazenai.com/reference/configuration/#when-agent-finops-is-unreachable)).

Two settings are required under Control, and `monitor()` refuses to start
without them:

| Requirement | Why | If missing |
|---|---|---|
| `KAZENAI_TIER_MISMATCH_MULTIPLIER` (finite, at least 1) | When a provider bills a call at a service tier KazenAI has no certified price for, KazenAI holds the requested-tier price times this number against your limit and keeps the call for reconciliation. Use at least the price ratio of the most expensive tier your provider can apply. | `PricingConfigError` from `monitor()` |
| Provider clients built with `max_retries=0` | Each retry is a new billable attempt, so KazenAI reserves for each one itself. | `UnsupportedModeError` (`unsupported_provider_retries`) |

Hosted Agent FinOps also lets an admin pause an organization: new reservations
are refused (`BudgetExceeded`) until it is resumed, and calls already admitted
still settle.

`KAZENAI_PROFILE=legacy` selects the pre-1.2 check/reconcile path. It is
deprecated, warns, and is removed in 2.0. Hosted Agent FinOps refuses it.

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
| OpenAI `chat.completions.create` / `.stream` / `.parse`, sync and async, streaming and non-streaming | Supported, verified against the live API |
| Anthropic `messages.create` / `.stream`, sync and async, streaming and non-streaming | Supported, verified against the live API |
| LangChain `ChatOpenAI` / `ChatAnthropic` | Supported when bound with `bind_monitored_chat_openai` / `bind_monitored_chat_anthropic` |
| LangGraph | Supported when the graph declares every provider client its nodes use; raw clients created inside nodes are not intercepted |
| LiteLLM proxy | Supported with application-owned or gateway-owned accounting (gateway-owned streaming settlement not yet verified); see [LiteLLM gateway](#litellm-gateway) |
| OpenAI `responses.create` / `.stream` / `.parse`, sync and async, streaming and non-streaming | Supported, verified against the live API (new in 1.2.0). Refused before dispatch: `background=True`, `previous_response_id` / `conversation` / stored `prompt`, built-in tools |
| CrewAI 1.15.26 (`kickoff`, `akickoff`, `kickoff_async`) | Supported, verified against the live API, when every LLM is bound with `bind_crew_llm_client`; memory, knowledge, `Crew(stream=True)` and `llm_overlay` are refused under Control. Python 3.10–3.13 (CrewAI doesn't install on 3.14 yet) |
| Other OpenAI endpoints (embeddings, images, audio, batches, fine-tuning, Realtime, legacy `completions`) and any provider method saved before `monitor()` | Refused before dispatch with `UnsupportedModeError`; `monitor(..., allow_unmetered=("/embeddings",))` sends named endpoints without a budget check |
| Bedrock/Vertex wrappers, AutoGen | Not supported |

Cancelling CrewAI's native `akickoff` stops the crew; `kickoff_async` runs on a
worker thread that cancellation does not stop, so that crew finishes and is
charged.

### LiteLLM gateway

To enforce limits in a LiteLLM proxy for every application behind it, install
`kazenai` in the proxy's environment and load its callback:

```yaml
litellm_settings:
  callbacks: kazenai.integrations.litellm_gateway.proxy_handler_instance
environment_variables:
  KAZENAI_ENFORCEMENT_OWNER: gateway
```

Set the usual Agent FinOps variables in the proxy, and keep its retries off
(`num_retries: 0`). Applications send `metadata={"kazenai_operation_id": ...}` and
do not reserve again. `kazenai` does not depend on or bundle LiteLLM; the callback
uses only LiteLLM's MIT-licensed `CustomLogger`.

Inline images are priced before the call. Files and documents sent by reference
are refused. `observe_unbounded=("images",)` lets image calls through without an
exact bound and records them as `spend_bound: "none"`. Responses API paths that
need OpenAI's exact input-token counter require an openai 2.x client.

### Streaming

```python
from kazenai import StreamCutoffError, monitor
from openai import OpenAI

client = monitor(
    OpenAI(max_retries=0),
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
- **Low overhead** — a guarded call adds about three network round trips to
  Agent FinOps over one kept-alive connection: a few milliseconds when your app
  runs in the same cloud region
- **Shared policies fail closed by default** — a call is refused when the
  shared reservation cannot be obtained, unless the organization chose fail-open

Fail-open (an organization setting in Agent FinOps) keeps only the local client
safeguards while Agent FinOps is unreachable. It does not preserve a
cross-process or account-wide budget guarantee for those calls.

### Multi-model workflows

Each `monitor(..., max_budget_usd=...)` client has its own cap, so a workflow
with one client per provider would add those caps up. Share one instead:

```python
from kazenai import SharedBudget, monitor

budget = SharedBudget(max_usd=1.00)
planner = monitor(OpenAI(max_retries=0), budget=budget)
writer = monitor(Anthropic(max_retries=0), budget=budget)
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

## FinOps environment variables

| Variable | Purpose |
|----------|---------|
| `KAZENAI_FINOPS_URL` (or `KAZENAI_FINOPS_INGEST_URL`) | Agent FinOps base URL. Setting either selects the Control lifecycle |
| `KAZENAI_FINOPS_API_KEY` | Credential for FinOps budget and event requests |
| `KAZENAI_TIER_MISMATCH_MULTIPLIER` | Required under Control (finite, at least 1); see above |
| `KAZENAI_PROFILE` | `legacy` selects the deprecated pre-1.2 path; leave unset |
| `KAZENAI_DEPLOYMENT_MODE` | `production` / `staging` require fail-closed shared reservations |
| `KAZENAI_FINOPS_RESERVE_TIMEOUT_S` | Timeout for one attempt of a shared pre-call reservation (default 2 s, including connection setup). A reservation that gets no answer is sent once more with the same idempotency key |
| `KAZENAI_FINOPS_KEEPALIVE_S` | Seconds an idle connection to Agent FinOps is kept for reuse (default 15); `0` opens a new connection per request |
| `KAZENAI_TIMELINE_PATH` | Optional local JSONL evidence path |
| `KAZENAI_SDK_SETTLE_QUEUE_DIR` | Persistent private directory for pending settlement intents |
| `KAZENAI_PRICING_LIVE` | `0` turns off the signed live price catalog fetched from FinOps (the bundled snapshot always applies otherwise) |
| `KAZENAI_PRICING_REFRESH_S` / `KAZENAI_PRICING_CACHE_DIR` | Live catalog refresh interval (default 900 s) and private cache directory |
| `KAZENAI_ENDPOINT_PRICE_MULTIPLIERS` | Exact-host map (values >= 1) for proxies that bill at list price; other non-first-party hosts are priced at 1.1x |

Pass `max_budget_usd` (or `budget=SharedBudget(...)`), `org_id`, `project_id`,
`workspace_id` and attribution fields directly to `monitor()`; they are not
inferred from similarly named environment variables.

For shared Control deployments, read [Settlement recovery](https://github.com/kazenai-ai/kazenai-core/blob/main/docs/settlement-recovery.md).
Run the settlement flusher on a schedule; processes on one host can share a
queue directory, but the queue does not coordinate across hosts.

---

## Packages in the stack

| Package | Role |
|---------|------|
| [`kazenai`](https://pypi.org/project/kazenai/) | Core `monitor()` engine (this repo) |
| [`kazenai-finops`](https://pypi.org/project/kazenai-finops/) | Customer-facing SDK (recommended install) |
| [`kazen-event-schema`](https://pypi.org/project/kazen-event-schema/) | Shared event contract between the SDKs and Agent FinOps |

---

## Documentation

- [docs.kazenai.com](https://docs.kazenai.com/): guides and reference
- [Supported integrations](https://github.com/kazenai-ai/kazenai-core/blob/main/docs/integrations/control-supported-matrix.md): every provider and framework path and how it is verified
- [Price catalog](https://github.com/kazenai-ai/kazenai-core/blob/main/docs/PRICE_CATALOG.md): rates, pricing rules and their source pages
- [Settlement recovery](https://github.com/kazenai-ai/kazenai-core/blob/main/docs/settlement-recovery.md): running the settlement flusher in production
- [Changelog](https://github.com/kazenai-ai/kazenai-core/blob/main/CHANGELOG.md)

---

## Contributing

Issues and pull requests that improve budget and loop reliability are welcome —
see [CONTRIBUTING.md](https://github.com/kazenai-ai/kazenai-core/blob/main/CONTRIBUTING.md).

Products and design-partner enquiries: [kazenai.com](https://kazenai.com) · **founder@kazenai.com**

## License

Apache License 2.0. See [LICENSE](https://github.com/kazenai-ai/kazenai-core/blob/main/LICENSE) and [NOTICE](https://github.com/kazenai-ai/kazenai-core/blob/main/NOTICE).
