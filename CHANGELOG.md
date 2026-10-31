# Changelog

Notable changes to `kazenai`, newest first. Each release is also published on
[GitHub Releases](https://github.com/kazenai-ai/kazenai-core/releases).

## 1.2.1 - 2026-10-31

An opt-in guard so a call can't silently skip a customer's limit, and corrected
hosted setup docs.

```bash
python -m pip install "kazenai[openai]==1.2.1"
```

### New

- `monitor(..., require_customer_id=True)` refuses a call that names no customer
  ID. The call raises `CustomerIdRequired` (a `KazenBlock`) before it reserves
  budget or reaches the provider. Without the option, a call made where the ID
  isn't set, such as in a `ThreadPoolExecutor` worker, is checked against the
  organization and workspace limits only. Off by default, so nothing changes
  unless you turn it on.

### Fixed

- A call refused because the organization is paused said "budget exhausted".
  `BudgetExceeded` now gives the reason from the decision receipt: "Pre-call
  reservation denied: organization is paused for new admissions". Plan-limit and
  customer-limit messages are unchanged.
- The LiteLLM gateway path gave a paused organization the same "budget
  exhausted"; `GatewayAdmissionDenied` now carries the receipt's reason.
- A reservation that got no answer (a timeout, a refused or dropped connection,
  or a 502/503/504) failed the call at once. It is now sent once more with the
  same body and idempotency key, so Agent FinOps answers a repeat with the
  reservation it already made and never reserves twice. A refusal (402) and any
  other answer are not retried.
- `KAZENAI_FINOPS_RESERVE_TIMEOUT_S` defaults to 2 s (was 0.8 s). The first call
  in a new process also opens the connection inside that time, which from a
  distant region could take most of 0.8 s. With both attempts, a call waits at
  most about 4 s for an Agent FinOps that never answers; set the variable lower
  to shorten that.

### Docs

- The README and QUICKSTART show the `org_id` and `workspace_id` that hosted
  Agent FinOps needs. Without them each reservation is refused with HTTP 403 and
  the call raises `BudgetUnavailable`.
- They also show how to name the customer on each call (`use_attribution`) and
  how to carry the ID into a worker thread (`contextvars.copy_context().run`).

### Compatibility

- Supports Python 3.13 and 3.14, now tested in CI alongside 3.10–3.12 and listed
  in the package metadata.

## 1.2.0 - 2026-10-28

Hard limits now apply by default. When Agent FinOps is configured, every call
reserves its cost before it is sent, and a call the limit has no room for is
refused. This release also adds the OpenAI Responses API and certified CrewAI
support, and closes gaps that let some OpenAI calls skip the budget check.

**Upgrade from any earlier release.** Versions 1.0.1 to 1.1.3 are affected by
the [security issue](#security) below.

```bash
python -m pip install "kazenai[openai]==1.2.0"
```

### Before you upgrade

- **A FinOps URL now means hard limits.** With `KAZENAI_FINOPS_URL` set, each
  call reserves budget in Agent FinOps before it is sent and settles afterwards.
  If Agent FinOps can't be reached, the call is refused. The old
  check-then-reconcile mode (`KAZENAI_PROFILE=legacy`) is deprecated and hosted
  Agent FinOps no longer accepts it.
- **Two settings are required with Agent FinOps.** `monitor()` raises at setup
  without them:
  - `KAZENAI_TIER_MISMATCH_MULTIPLIER`, a number of at least 1. If a provider
    bills a call at a service tier KazenAI didn't request, KazenAI holds the
    requested price times this number, and further calls to that model are
    refused for the organization until an operator reviews it.
  - Provider clients built with `max_retries=0`, so every attempt is reserved.
- **Paid endpoints that can't be bounded are refused.** Embeddings, images,
  audio, batches and fine-tuning now raise `UnsupportedModeError` on a monitored
  client. To allow one without a budget check, name it:
  `monitor(client, allow_unmetered=("/embeddings",))`.
- **In staging and production, the FinOps URL must be public HTTPS.** With
  `KAZENAI_DEPLOYMENT_MODE` (or `KAZENAI_ENV`) set to `staging` or `production`,
  `monitor()` refuses a FinOps or ingest URL that uses `http://` or points at a
  private, loopback or link-local address, including through DNS. Development
  mode is unchanged.
- **APIs from earlier KazenAI products are removed** (see [Removed](#removed)).

Details: [Configuration](https://docs.kazenai.com/reference/configuration/#connect-agent-finops-control)
and [Errors](https://docs.kazenai.com/reference/errors/).

### New

- **OpenAI Responses API.** `responses.create`, `responses.parse` and
  `responses.stream`, sync and async, with function tools. Background mode,
  server-side conversation state and built-in tools (web search, file search,
  code interpreter and others) are refused, because their cost can't be bounded
  before the call. [OpenAI guide](https://docs.kazenai.com/guides/openai/#responses-api)
- **CrewAI 1.15.26, certified.** `wrap_crew` covers `kickoff`, `akickoff` and
  `kickoff_async` with OpenAI and Anthropic, including two providers on one
  budget. A budget denial stops the crew instead of being retried.
  [Frameworks guide](https://docs.kazenai.com/guides/frameworks/#crewai)
- **Signed live prices.** Connected to Agent FinOps, the SDK fetches KazenAI's
  signed price catalog and verifies it. The prices bundled with the package stay
  the floor. `KAZENAI_PRICING_LIVE=0` turns it off. The signature check uses
  `cryptography`, now a dependency of `kazenai`, so every install gets the feed.
- **LiteLLM gateway callback.** `kazenai.integrations.litellm_gateway` enforces
  Agent FinOps limits inside a LiteLLM proxy for every application behind it
  (`callbacks: kazenai.integrations.litellm_gateway.proxy_handler_instance`).
  `kazenai` does not depend on LiteLLM.
- **Fail-open, per organization** (Agent FinOps). An org admin can choose what
  happens while the SDK cannot reach Agent FinOps: refuse the call (fail-closed,
  the default) or let it run with only the SDK's local caps (fail-open). It
  applies only when Agent FinOps does not answer (no connection, a timeout, HTTP
  408, 429 or 5xx); an over-limit call, a pause or an invalid key is still
  refused. Calls let through this way are not charged to the org's limits, and
  the SDK logs them. The SDK learns the setting from Agent FinOps and remembers
  it on disk; a process that has never seen it fails closed. The LiteLLM gateway
  callback always fails closed.
- **Faster calls with Agent FinOps.** A guarded call's reserve and settlements
  now share one kept-alive connection instead of opening a new one each time,
  which cuts the added time to about three network round trips: a few
  milliseconds in the same cloud region, and roughly a third of before from
  farther away. `KAZENAI_FINOPS_KEEPALIVE_S` (default 15) sets how long an idle
  connection is kept; `0` restores a new connection per request.
- **Automatic settlement retry.** Settlements that couldn't be delivered are
  retried in the background with backoff (`KAZENAI_SDK_SETTLE_AUTO_FLUSH=0`
  turns it off).
- **Shadow mode** (Agent FinOps). A limit set to shadow never blocks. It records
  what it would have decided, so you can see the effect before enforcing.
  [Shadow mode](https://docs.kazenai.com/guides/shadow-mode/)
- **Pause an organization** (Agent FinOps). An admin can stop all new spend for
  an organization; calls then raise `BudgetExceeded`.
  [Pause an organization](https://docs.kazenai.com/reference/configuration/#pause-an-organization)

### Security

In 1.0.1 to 1.1.3, these calls on a monitored OpenAI client reached OpenAI
without a budget check and were not recorded:

- `responses.parse`, and raw or streaming response wrappers first used before
  `monitor()`;
- legacy `completions.create` and Realtime connections;
- any method, wrapper or `client.request` saved before `monitor()`, and
  `client.post(...)`.

The same applied to Anthropic clients. 1.2.0 checks every request a monitored
client sends, and refuses a paid request that didn't come through a guarded
method before it reaches the provider. Fields in `extra_body` are checked like
top-level arguments, and saved settlements are sent only to the Agent FinOps URL
they were created for. Code that builds its own client from the API key, or
sends raw HTTP, is outside what `monitor()` can cover.

Report security issues privately to founder@kazenai.com.

### Pricing

- Calls through a host other than the provider's own API (Azure, Bedrock,
  Vertex, data-residency domains, proxies) are priced at 1.1×. Declare a proxy
  that charges list price with `KAZENAI_ENDPOINT_PRICE_MULTIPLIERS`.
- Anthropic fast mode is refused, and `inference_geo: "us"` is priced at 1.1×.
- Models with a long-context price band (GPT-5.4 and later, GPT-6, Claude Haiku
  5.5) are priced from the provider's exact input count, so the SDK first makes
  one free count request (`responses.input_tokens.count`, which needs `openai`
  2.x, or `messages.count_tokens`). Chat Completions has no such counter, so a
  Chat Completions call to these models is priced at the standard band when its
  size in bytes shows it cannot reach the long-context threshold (272,000 tokens
  for GPT models). Calls with images or audio, or that could reach the threshold,
  are refused (`long_context_chat_unsupported`); use the Responses API for those.
- Scheduled changes are built in: Gemini 3.6 and 3.8 Flash prices double on
  2026-12-31. The GPT-5.6 Sol promotional price ends on 2026-11-22; after that,
  calls to it are refused until an updated signed catalog confirms the new price.

All rules: [PRICE_CATALOG.md](https://github.com/kazenai-ai/kazenai-core/blob/v1.2.0/docs/PRICE_CATALOG.md).

### Fixed

- FinOps telemetry events keep the call's workspace, so keys bound to one
  workspace no longer have them rejected.
- A crew started from inside another crew gets its own run instead of raising
  `AttributionConflict`.
- Settlements Agent FinOps rejects for good are set aside instead of retried on
  every flush; `flush` reports them and exits `1`.
- An async stream denied before sending no longer leaves a "coroutine was never
  awaited" warning.
- Requests that carry your KazenAI key (event delivery, the signed price catalog,
  alert webhooks) no longer follow redirects. A redirect used to copy the key to
  the host it pointed at.
- A model call whose cost is unknown (the provider reported no usage) is recorded
  with no cost instead of $0.

### Removed

Leftovers from earlier KazenAI products; nothing in the current product uses them.

- `KazenMemory` (`kazenai.memory`).
- `kazenai.reliability_policy` and its root exports (`AutonomyTier`,
  `FailureKind`, `ReliabilityAction`, `ReliabilityContext`,
  `ReliabilityDecision`, `ReliabilitySurface`, `parse_autonomy_tier_from_headers`,
  `resolve_reliability`, `should_block`, `should_escalate`).
- `kazenai.deployment.brain_strict_enabled()` and `graph_strict_enabled()`, and
  the startup checks for services that no longer exist.

### Compatibility

Python 3.10 or later; `openai>=1.40,<3`; `anthropic>=0.39,<1`. Requires
`kazen-event-schema>=0.6.6,<0.7` and `cryptography>=50.0.0`. Released together
with `kazenai-finops` 1.2.0.

## 1.1.3 - 2026-10-12

Documentation and packaging release. No change to runtime behaviour or the
public API; upgrading from 1.1.2 is safe.

### Changes

- **Clearer documentation.** The README, quick start, architecture,
  deployment, release and supported-integrations pages are rewritten for
  people using KazenAI; links on the PyPI page now resolve.
- **Smaller source distribution.** The sdist ships the package, licence and
  README only; the test suite lives in the GitHub repository.
- **Cleaner source.** Code comments and docstrings describe behaviour in plain
  language.
- **Standalone tests.** The test suite runs from a checkout of this repository
  alone, with no sibling repositories or private packages.

Requires `kazen-event-schema>=0.6.4,<0.7`. Release together with
`kazenai-finops` 1.1.3.

### Install

```bash
python -m pip install "kazenai[openai]==1.1.3"
# or, for most product integrations:
python -m pip install "kazenai-finops[openai]==1.1.3"
```

## 1.1.2 - 2026-10-12

Async clients, one budget across several clients, accurate streamed pricing,
LangChain `ChatAnthropic`, multi-provider graphs, and a refreshed price table.
Supported paths are listed in
[Supported integrations](docs/integrations/control-supported-matrix.md).

Requires `kazen-event-schema>=0.6.4,<0.7`. Agent FinOps and Agent Lens require
`kazenai>=1.1.2`.

### New: one budget across several clients (`SharedBudget`)

```python
from kazenai import SharedBudget, monitor

budget = SharedBudget(max_usd=1.00)
openai_client = monitor(OpenAI(), budget=budget)
anthropic_client = monitor(Anthropic(), budget=budget)
```

Every call on any of those clients checks and charges the same cap before
dispatch. Without it, each `monitor(max_budget_usd=...)` client has its own cap,
so a 3-model workflow at $1.00 per client could spend $3.00. One process only;
use Agent FinOps (`workflow_id` + a workflow limit) across processes.

### LangChain ChatAnthropic and multi-provider graphs

- `bind_monitored_chat_anthropic(chat, anthropic_client, async_client=...)`
  puts a LangChain `ChatAnthropic` under Control (invoke/stream/ainvoke/astream),
  like `bind_monitored_chat_openai` for `ChatOpenAI`. Its `betas` option is
  rejected: beta calls go through `client.beta.messages`, which is not guarded.
- A graph or chain that calls several providers declares all of them:
  `declare_langgraph_control_boundary(graph, openai_client, anthropic_client)`
  then `wrap_graph(graph, capability="control", guarded_clients=[...])`
  (same for `declare_langchain_control_boundary` / `wrap_langchain_runnable`).
  Every client passed must be one the graph declared.
- `client.beta.messages` on a monitored Anthropic client now raises
  `UnsupportedModeError` instead of dispatching unguarded.
- `bound_guarded_client(chat_model)` returns the monitored client a bound chat
  model uses, so you can declare it without keeping a separate reference.
- Streams requested through `with_raw_response` are metered when the caller
  calls `parse()` (previously that stream was not costed).

### Provider rejections release the hold

A provider HTTP 4xx before any output (bad request, auth, model not found,
unprocessable, rate limited; not 408/409/499) now settles the reservation at
exactly $0. Previously it stayed `pending_reconciliation` and tied up budget.
Timeouts, connection errors and 5xx still stay pending. Applies to `monitor()`
(sync/async, streaming/non-streaming) and the LiteLLM gateway callback.

### Ordinary API keys work with the SDK

API keys created the normal way in Agent FinOps can now be used with
`monitor()` for their own organization. Previously some keys were refused and,
under fail-closed, blocked every call. Requires the matching Agent FinOps
release.

### Behaviour change: unpriced models are blocked by default

A model with no price used to project as $0 outside production, so a local
budget cap silently did not bind. From 1.1.2 the unknown-model policy defaults
to `block` in every environment: `monitor()` raises `UnknownModelError` before
dispatch. Add the price with `KAZENAI_PRICING_JSON`, pass `usd_per_1k_tokens`,
or opt in to `KAZENAI_UNKNOWN_MODEL_POLICY=estimate`. `warn` remains as a
development-only opt-in and is ignored in production-like deployments.

### Pricing refresh

- One canonical table (`kazenai.model_pricing`), verified against the official
  OpenAI, Anthropic, Gemini, and DeepSeek pricing pages on 2026-10-12
  (`PRICING_LAST_VERIFIED`, `PRICING_SOURCES`). `cost_engine` derives from it;
  the second, disagreeing table is gone.
- Adds the gpt-6/gpt-5.x, gpt-4.1, o-series, Claude 5.x/Opus 4.8 and Gemini
  2.5/3.x families. Corrects gpt-4o ($2.50/$10), Claude Opus 4.5–4.8 ($5/$25)
  and Claude Haiku 4.5 ($1/$5).
- Removes retired or unverifiable rows (Claude 3.x, gpt-4-turbo, Llama, Nova,
  Mistral, Qwen, Cohere, older DeepSeek ids). These are now unknown and
  blocked unless priced via `KAZENAI_PRICING_JSON`.
- An unpriced version bump (`gpt-5.3`, `claude-opus-5-6`) no longer inherits a
  sibling's price through prefix matching; dated snapshots still resolve.
- The injected output bound uses `max_completion_tokens` for OpenAI reasoning
  families (gpt-5/gpt-6/o-series), which reject `max_tokens`.

### Fixes

- **Credentials stay out of logs.** `ReservationHandle` and `HttpSinkConfig` no
  longer include the FinOps API key in their `repr`, so logging a handle or a
  traceback that captures locals does not print the key.
- **Structured-output calls are budgeted.** OpenAI `chat.completions.parse` and
  `beta.chat.completions.parse` (used by LangChain `with_structured_output`) went
  straight to the provider without a budget check or a cost record. They are now
  guarded exactly like `create`, sync and async. `beta.chat.completions.stream`
  is refused with `UnsupportedModeError` under Control instead of running
  unguarded.
- **CrewAI `bind_crew_llm_client` keeps the crew runnable.** On CrewAI 1.15.26
  native LLMs it replaced the agent's LLM with an object CrewAI rejected ("LLM
  must be resolved"). It now swaps the monitored client into the existing LLM
  (sync and async), so crews run and each LLM call is checked before dispatch.
  Set `max_retry_limit=0` on agents so a refused call surfaces as
  `BudgetExceeded`. CrewAI stays outside the supported matrix.

- **Streamed calls are priced correctly.** Settlement used the stream-cutoff
  estimate (the model's *output* rate) as a flat price, so every streamed token,
  input included, was charged at the output rate: 2.6–2.9x overstated in a live
  run. Streams now settle like non-streamed calls, pricing provider-reported
  input and output tokens separately. A caller's explicit `usd_per_1k_tokens`
  still applies.
- **LangChain non-streaming calls recorded $0.** langchain-openai calls
  `chat.completions.with_raw_response.create`, which returns a raw wrapper
  without `.usage`; the pre-call check ran but the cost was never recorded.
  `monitor()` now reads usage through the SDK's cached `parse()` (OpenAI and
  Anthropic, sync and async).
- **Framework wrappers no longer replace the client's tenant.** `wrap_graph` /
  `wrap_langchain_runnable` default to `org_id="local"`; that default overrode
  the org a monitored client was configured with, so reservations went to the
  wrong tenant. Run/step lineage still comes from the framework; an explicitly
  set ambient tenant still wins.
- **LiteLLM gateway hook no longer leaks `litellm_params`** into the provider
  request (real OpenAI/Anthropic reject unknown fields).
- **Settlement no longer charges an extra micro-dollar for float noise**
  (0.00010200000000000001 USD now settles as 102 micros, not 103).
- **Frameworks live-verified.** LangChain, LangGraph (incl. checkpoint resume and
  a 3-model, 2-provider workflow under one FinOps budget), and LiteLLM app-owned
  and gateway-owned accounting passed live runs on 2026-10-12 with two model
  sets. Gateway-owned streaming settlement is not yet live-verified.
- **Live-provider verified.** OpenAI Chat Completions and Anthropic Messages via
  `monitor()` (sync/async, streaming/non-streaming) passed a paid live smoke on
  2026-10-12 with `gpt-6-luna` and `claude-sonnet-5-5`: real responses, recorded
  cost equal to the pricing table on the reported usage, and pre-dispatch
  `BudgetExceeded`.

- **Settlement spool is safe across processes.** Each
  `(org, FinOps URL, idempotency_key)` binds exactly one intent fingerprint
  through an atomic claim file. Conflicting reuse raises `SettleIntentConflict`
  in every process sharing the queue directory; previously two processes could
  both persist different amounts under one key.
- **Settlement recovery is operable.** `python -m kazenai.spine.settle_queue
  flush|stats` with a single-owner lock and exit codes; `queue_stats()`
  (depth, oldest age, quarantined, last flush); corrupt records are quarantined
  with a WARNING instead of skipped; directory metadata is fsynced on POSIX. See
  [`docs/settlement-recovery.md`](docs/settlement-recovery.md).
- **CrewAI listener routes real events.** Ownership is resolved from the
  crew's agents, tasks, LLMs, and tools, the event's agent/task references, and
  the active kickoff scope. CrewAI 1.15.26 LLM/task/tool events were previously
  dropped. CrewAI is still a preview integration.
- **Event registry.** `finops.trajectory` and `finops.loop.anomaly`, which the SDK
  emits, are registered in `kazen-event-schema` 0.6.4, so strict consumers accept them.

### Also included since 1.1.1

Journal-before-send settlement, shared
JWT issuer policy (`kazenai.auth.jwt`, including `extract_role`), async provider
paths, LangChain/LangGraph/LiteLLM adapter contracts, and OTel control spans.
Agent FinOps and Agent Lens import these APIs, which is why they require 1.1.2.

### Install

```bash
python -m pip install "kazenai[openai]==1.1.2"
```

## 1.1.1 - 2026-10-04

Compatibility and packaging patch. No change to supported integrations.

### What changed and why

- Added optional provider extras so first-run installs cannot silently resolve
  unsupported OpenAI 3.x or Anthropic 1.x majors.
- Synchronized active install guidance, release docs, and package metadata to 1.1.1.

### Install

```bash
python -m pip install "kazenai[openai]==1.1.1"
python -m pip install "kazenai[anthropic]==1.1.1"
# Most product integrations should use:
python -m pip install "kazenai-finops[openai]==1.1.1"
```

### Supported provider methods and SDK ranges

- Sync OpenAI Chat Completions: non-streaming, `stream=True`, `.stream()` helper
- Sync Anthropic Messages: non-streaming, `stream=True`, `.stream()` helper
- Supported ranges: `openai>=1.40,<2`, `anthropic>=0.39,<1`
- Python 3.10–3.12

### Explicitly unsupported

- Async clients
- OpenAI Responses API
- OpenAI Realtime / WebSocket
- Bedrock / Vertex wrappers
- Later provider SDK majors outside the supported ranges

### Behavior change

None intended versus 1.1.0.

### Security / privacy impact

None. Capture remains metadata-default; bodies require explicit opt-in with consent.

### Upgrade from 1.1.0

```bash
python -m pip install -U "kazenai[openai]==1.1.1"
```

Prefer extras over bare `pip install openai` / `anthropic`.

### Links

- Docs: https://docs.kazenai.com/
- Issues: https://github.com/kazenai-ai/kazenai-core/issues
- Source tag: `v1.1.1`

### Known limitation

Streaming cancellation / missing authoritative usage remains financially
**pending/unknown** rather than exact zero. Client close does not prove the
provider stopped billing.

Earlier releases: [GitHub Releases](https://github.com/kazenai-ai/kazenai-core/releases).
