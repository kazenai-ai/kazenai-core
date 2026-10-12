# kazenai 1.1.2 (release candidate)

Control 1.1.x line. Promotes the FINAL_1_b_final working tree and the Phase 1
bug fixes from the 2026-10-11 re-audit. Capability claims are bounded by
[`docs/integrations/control-supported-matrix.md`](docs/integrations/control-supported-matrix.md);
nothing here widens that matrix.

## Release order

1. `kazen-event-schema` 0.6.4 — this release requires `kazen-event-schema>=0.6.4,<0.7`.
2. `kazenai` 1.1.2.
3. Agent FinOps and Agent Lens images (both require `kazenai>=1.1.2,<2.0`).

`kazenai-finops` 1.1.1 (`kazenai>=1.1.1,<2.0`) resolves 1.1.2 without a re-release.

## New: one budget across several clients (`SharedBudget`)

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

## LangChain ChatAnthropic and multi-provider graphs

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

## Provider rejections release the hold

A provider HTTP 4xx before any output (bad request, auth, model not found,
unprocessable, rate limited; not 408/409/499) now settles the reservation at
exactly $0. Previously it stayed `pending_reconciliation` and tied up budget.
Timeouts, connection errors and 5xx still stay pending. Applies to `monitor()`
(sync/async, streaming/non-streaming) and the LiteLLM gateway callback.

## Ordinary API keys work with the SDK

API keys created the normal way in Agent FinOps can now be used with
`monitor()` for their own organization. Previously some keys were refused and,
under fail-closed, blocked every call. Requires the matching Agent FinOps
release.

## Behaviour change: unpriced models are blocked by default

A model with no price used to project as $0 outside production, so a local
budget cap silently did not bind. From 1.1.2 the unknown-model policy defaults
to `block` in every environment: `monitor()` raises `UnknownModelError` before
dispatch. Add the price with `KAZENAI_PRICING_JSON`, pass `usd_per_1k_tokens`,
or opt in to `KAZENAI_UNKNOWN_MODEL_POLICY=estimate`. `warn` remains as a
development-only opt-in and is ignored in production-like deployments.

## Pricing refresh

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

## Fixes

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

- **Settlement spool is safe across processes (A1).** Each
  `(org, FinOps URL, idempotency_key)` binds exactly one intent fingerprint
  through an atomic claim file. Conflicting reuse raises `SettleIntentConflict`
  in every process sharing the queue directory; previously two processes could
  both persist different amounts under one key.
- **Settlement recovery is operable (A5).** `python -m kazenai.spine.settle_queue
  flush|stats` with a single-owner lock and exit codes; `queue_stats()`
  (depth, oldest age, quarantined, last flush); corrupt records are quarantined
  with a WARNING instead of skipped; directory metadata is fsynced on POSIX. See
  [`docs/settlement-recovery.md`](docs/settlement-recovery.md).
- **CrewAI listener routes real events (A2).** Ownership is resolved from the
  crew's agents, tasks, LLMs, and tools, the event's agent/task references, and
  the active kickoff scope. CrewAI 1.15.26 LLM/task/tool events were previously
  dropped. CrewAI remains outside the supported matrix.
- **Event registry.** `finops.trajectory` and `finops.loop.anomaly`, which the SDK
  emits, are registered in `kazen-event-schema` 0.6.4, so strict consumers accept them.

## Also included since 1.1.1

The FINAL_1_b_final working tree: journal-before-send settlement, shared
JWT issuer policy (`kazenai.auth.jwt`, including `extract_role`), async provider
paths, LangChain/LangGraph/LiteLLM adapter contracts, and OTel control spans.
Agent FinOps and Agent Lens import these APIs, which is why they require 1.1.2.

## Install

```bash
python -m pip install "kazenai[openai]==1.1.2"
```
