# KazenAI Core quick start

`kazenai` is the Python runtime used by KazenAI to enforce local and shared
spending policies around supported OpenAI and Anthropic calls. Most
product integrations should install the customer-facing package, which
re-exports the Core API:

```bash
python -m pip install "kazenai-finops[openai]==1.2.1"
```

Install Core directly when you specifically want the runtime package:

```bash
python -m pip install "kazenai[openai]==1.2.1"
```

Python 3.10, 3.11 and 3.12 are supported.

## 1. Put a local hard cap around a client

```python
from openai import OpenAI
from kazenai import BudgetExceeded, monitor

client = monitor(
    OpenAI(max_retries=0),
    agent_id="support-agent",
    org_id="acme",
    max_budget_usd=5.00,
)

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Summarize this ticket"}],
    )
except BudgetExceeded as exc:
    print("The call was blocked before provider dispatch:", exc)
```

`max_budget_usd` is an in-process limit for that one client. It is not an
account-wide provider billing limit.

When several clients make up one workflow (for example an OpenAI step and an
Anthropic step), give them one shared cap instead:

```python
from kazenai import SharedBudget

budget = SharedBudget(max_usd=1.00)
openai_client = monitor(OpenAI(max_retries=0), budget=budget)
anthropic_client = monitor(Anthropic(max_retries=0), budget=budget)
```

A model with no known price is blocked before dispatch (`UnknownModelError`);
add its price with `KAZENAI_PRICING_JSON`. For multiple workers or services
sharing a budget, use the shared FinOps authority described below.

## 2. Connect a shared FinOps authority

```bash
export KAZENAI_FINOPS_URL="https://finops.example.com"
export KAZENAI_FINOPS_API_KEY="kz_..."
export KAZENAI_TIER_MISMATCH_MULTIPLIER="2"
export KAZENAI_DEPLOYMENT_MODE="production"
```

```python
client = monitor(
    OpenAI(max_retries=0),
    agent_id="support-agent",
    org_id="<organization ID>",      # both IDs are on the Setup page in Agent FinOps
    workspace_id="<workspace ID>",
    max_budget_usd=5.00,
)
```

An Agent FinOps API key belongs to one organization and one workspace, and
`monitor()` must name the same two. It does not read them from the key or from
`KAZENAI_ORG_ID`. Without them it uses `local` and `default`, Agent FinOps
refuses each reservation with HTTP 403, and the call raises `BudgetUnavailable`.
`guarded_llm_call()` (section 4) takes the same `org_id` and `workspace_id`.

Name the customer on each call. A per-customer limit in Agent FinOps (dimension
`business_subject`) applies only to calls that carry that customer's ID as
`business_subject_ref`:

```python
from kazenai import use_attribution

with use_attribution(business_subject_ref="cust_123"):
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Summarize this ticket"}],
    )
```

Use the same ID the limit was set for. When a client only ever serves one
customer, `monitor(..., business_subject_ref="cust_123")` names that customer on
every call instead. If both name a customer and the IDs differ, the call raises
`AttributionConflict` before it reaches the provider. A call without a customer
ID is checked against the organization and workspace limits only; no
per-customer limit applies to it. The ID is held in a context variable: calls in
the same thread, and in asyncio tasks started inside the `with` block, carry it.
A thread started inside the block, such as a `ThreadPoolExecutor` worker, does not.
To carry the ID into a worker, submit the work in a copy of the current context:

```python
import contextvars

with use_attribution(business_subject_ref="cust_123"):
    pool.submit(contextvars.copy_context().run, handle_ticket, ticket)
```

To make a call without a customer ID fail instead of skipping the customer's
limit, pass `require_customer_id=True` to `monitor()` (kazenai 1.2.1 and later).
Such a call then raises `CustomerIdRequired` before it reserves budget or reaches
the provider.

Setting the FinOps URL selects the Control lifecycle (the default since 1.2.0).
The `monitor(...)` call now reserves the estimated cost in Agent FinOps
before dispatch and settles authoritative usage afterward. A refused
reservation raises `BudgetExceeded`; the provider is never called. If Agent
FinOps cannot be reached the call is refused, except under explicit local
development (`KAZENAI_ENV=dev` with `KAZENAI_ENFORCEMENT_MODE=fail_open`), which
keeps only the local client safeguards.

`monitor()` refuses to start under Control unless:

- `KAZENAI_TIER_MISMATCH_MULTIPLIER` is a finite number of at least 1
  (`PricingConfigError` otherwise). It is how much of the requested-tier price
  KazenAI holds against the limit when a provider bills a call at a service
  tier KazenAI has no certified price for; such calls wait for reconciliation.
- the provider client is built with `max_retries=0` (`UnsupportedModeError`
  otherwise), so each retry is reserved as its own billable attempt.

An admin can pause an organization in Agent FinOps. While it is paused, new
reservations are refused with `BudgetExceeded`; calls already admitted still
settle. `KAZENAI_PROFILE=legacy` selects the deprecated pre-1.2 path, which
hosted Agent FinOps refuses.

## 3. Stream with finalization-safe accounting

OpenAI `create(stream=True)` is supported:

```python
from openai import OpenAI
from kazenai import StreamCutoffError, monitor

client = monitor(
    OpenAI(max_retries=0),
    max_budget_usd=1.00,
    stream_cutoff_usd=0.05,  # optional observable-output guard
)

try:
    with client.chat.completions.create(
        model="gpt-4o-mini",
        stream=True,
        messages=[{"role": "user", "content": "Explain the result"}],
    ) as stream:
        for chunk in stream:
            print(chunk)
except StreamCutoffError:
    # KazenAI attempted to close future output. Provider billing can remain
    # pending when authoritative terminal usage was not received.
    pass
```

The official lazy OpenAI helper is also supported. Provider dispatch and the
shared reservation's `provider_started` transition occur when the context
manager is entered, not when it is constructed:

```python
with client.chat.completions.stream(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain the result"}],
) as stream:
    for event in stream:
        print(event)
```

Anthropic's synchronous manager follows the same accounting lifecycle:

```python
from anthropic import Anthropic
from kazenai import monitor

client = monitor(Anthropic(max_retries=0), max_budget_usd=1.00)

with client.messages.stream(
    model="claude-sonnet-4-5",
    max_tokens=256,
    messages=[{"role": "user", "content": "Explain the result"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="")
```

For a streamed call, KazenAI records a reservation, marks it
`provider_started` at dispatch, and settles it when authoritative terminal usage
arrives. Cancellation, a provider failure after dispatch, or missing final
usage produces an `outcome_unknown`/pending-reconciliation state. It is never
treated as an exact zero-cost call and never released as though dispatch did not
happen. The one exception is a definitive provider refusal (HTTP 4xx such as a
bad request, invalid key, unknown model or rate limit, but not 408/409/499):
nothing was generated, so it settles at exactly $0 and releases the hold.

`stream_cutoff_usd` estimates only output visible to the client. It cannot see
hidden reasoning or guarantee that the provider immediately stopped generating
or billing. `stream_enforcement=True` is deprecated; configure
`stream_cutoff_usd` explicitly if you accept that limitation.

## 4. Guard a custom synchronous call explicitly

```python
from kazenai import guarded_llm_call

response = guarded_llm_call(
    lambda: my_provider.generate(prompt),
    model="my-priced-model",
    org_id="acme",
    run_id="run-123",
    projected_cost_usd=0.02,
    feature="summarizer",
)
```

This explicit guard reserves around a custom callable. KazenAI does not parse
arbitrary provider responses, so the caller passes valid price and usage
information.

## What is supported

Every path below is covered by the test suite, and the OpenAI, Anthropic,
LiteLLM, LangChain and LangGraph paths are also verified against the live
provider APIs. Full details:
[Supported integrations](docs/integrations/control-supported-matrix.md).

| Path | Status |
|---|---|
| OpenAI `chat.completions.create` and `.parse`, sync and async | Supported, non-streaming and `stream=True` |
| OpenAI `chat.completions.stream` | Supported |
| Anthropic `messages.create`, sync and async | Supported, non-streaming and `stream=True` |
| Anthropic `messages.stream` | Supported |
| LiteLLM | Application-owned and gateway-owned accounting (gateway-owned streaming settlement not yet verified) |
| LangChain | `ChatOpenAI` via `bind_monitored_chat_openai`, `ChatAnthropic` via `bind_monitored_chat_anthropic`; multi-provider chains via `declare_langchain_control_boundary(runnable, *clients)` |
| LangGraph | Supported with an explicit declared boundary listing every provider client the nodes use; raw clients inside arbitrary nodes are not intercepted |
| Anthropic beta Messages (`client.beta.messages`) | Not supported; raises `UnsupportedModeError` on a monitored client |
| OpenAI `responses.create`, `.parse` and `.stream`, sync and async | Supported, non-streaming and `stream=True`, function tools (new in 1.2.0, verified against the live API) |
| OpenAI Responses `background=True`, `previous_response_id` / `conversation` / stored `prompt`, built-in tools | Not supported; raises `UnsupportedModeError` before dispatch |
| OpenAI Realtime/WebSocket, legacy `completions`, Responses `compact` / `connect`, embeddings, images, audio, batches, fine-tuning | Not supported; raises `UnsupportedModeError` on a monitored client (`allow_unmetered=("/embeddings",)` sends named endpoints without a budget check) |
| A provider method or `with_raw_response` wrapper saved before `monitor()` | Raises `UnsupportedModeError`; read methods from the client after wrapping |
| Bedrock/Vertex Anthropic wrappers | Not supported |
| CrewAI 1.15.26 | `wrap_crew(...)` kickoff / `akickoff` / `kickoff_async` with every LLM bound by `bind_crew_llm_client` (two providers: `guarded_clients=[...]`); memory, knowledge, `Crew(stream=True)`, `llm_overlay` and Anthropic structured output are not supported. Cancelling `akickoff` stops the crew; cancelling `kickoff_async` does not (its thread finishes and is charged) |
| AutoGen | Not supported |

## Next references

- [README.md](README.md) — capability overview and configuration
- [DEPLOY.md](DEPLOY.md) — application deployment guidance
- [RELEASING.md](RELEASING.md) — maintainer release procedure
- [docs.kazenai.com](https://docs.kazenai.com/) — product documentation
