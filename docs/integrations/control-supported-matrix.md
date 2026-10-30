# Supported integrations

This page lists what KazenAI Control supports in `kazenai` 1.2.0, how each
integration was verified, and the known limits. The machine-readable version is
[`control-supported-matrix.json`](./control-supported-matrix.json); it is checked
by the test suite, so the two stay in step.

Packages: `kazenai` 1.2.0, `kazenai-finops` 1.2.0 and `kazen-event-schema` 0.6.6,
on PyPI since 2026-10-28. Hosted Agent FinOps (limits and run receipts)
requires `kazenai>=1.2.0`: it refuses the legacy check path that earlier releases
use, so they fail closed against it.

## How to read the status

| Status | Meaning |
|---|---|
| `live-verified` | Exercised against the paid OpenAI/Anthropic APIs (or the named remote service) |
| `integration-verified` | Exercised against real KazenAI services (Agent FinOps and its run-receipts service) |
| `fixture-verified` | Covered by tests that use the official SDK objects with simulated provider responses |
| `implemented-unverified` | Code exists but is not yet a supported path; do not rely on it |
| `planned` / `unsupported` | Not part of this release |

## Providers

| Integration | Status | Details |
|---|---|---|
| OpenAI Chat Completions — sync and async, streaming and non-streaming, `parse()` structured output | live-verified | Tested on openai 1.109.1 and 2.48.0 (install range `openai>=1.40,<3`); live run 2026-10-12. Recorded cost equals the published price for the reported input/output tokens; calls over budget are refused before dispatch |
| OpenAI Responses — `create`, `parse`, `stream`; sync and async, streaming and non-streaming; function tools | live-verified | New in 1.2.0; tested on openai 1.109.1 and 2.54.0; live run 2026-10-13 on gpt-6-luna and gpt-6.1-sol (`parse`, `create`, streaming, the `stream()` helper; sync and async; [evidence](../verification/live/2026-10-13-openai-responses.json)). Recorded cost equals the published price for the reported tokens. Refused before dispatch: `background=True`, `previous_response_id` / `conversation` / stored `prompt`, built-in tools (web search, file search, code interpreter, MCP and others), and resuming a stream by `response_id` |
| Other OpenAI endpoints: Realtime, legacy Completions, Responses `compact` / `connect`, embeddings, images, audio, batches, fine-tuning | unsupported | Refused before dispatch with `UnsupportedModeError` on a monitored client since 1.2.0. The same applies to anything else sent through the monitored client without a budget check: a provider method, `with_raw_response` wrapper or `client.request` saved before `monitor()`, or a request another monitored client's call triggers. Reads, deletes and free endpoints (files, moderations, token counting) still work. `monitor(..., allow_unmetered=("/embeddings",))` sends named endpoints without a budget check |
| Anthropic Messages — sync and async, streaming and non-streaming | live-verified | Tested on anthropic 0.125.0 (install range `anthropic>=0.39,<1`); live run 2026-10-12. `client.beta.messages` and Bedrock/Vertex wrappers are not supported |

## Frameworks and gateways

| Integration | Status | Details |
|---|---|---|
| LangChain 1.4.4 | live-verified | `ChatOpenAI` via `bind_monitored_chat_openai`, `ChatAnthropic` via `bind_monitored_chat_anthropic`; invoke, stream, ainvoke, astream and `with_structured_output`. ChatAnthropic `betas` are refused |
| LangGraph 1.2.14 | live-verified | A graph declares every provider client its nodes use (`declare_langgraph_control_boundary`). Interrupt and resume from a SQLite checkpoint, and multi-model workflows across providers |
| LiteLLM proxy 1.104.2 — application-owned accounting | live-verified | `monitor()` around an OpenAI client pointed at the proxy: one reservation per call, exact settlement, refusal before dispatch |
| LiteLLM proxy — gateway-owned accounting | live-verified | The KazenAI LiteLLM callback reserves and settles in the proxy (no double counting with the SDK); HTTP 402 before dispatch. Gateway-owned streaming settlement is not live-verified yet |
| CrewAI 1.15.26 | live-verified | `kickoff`, native `akickoff` and `kickoff_async` with native OpenAI and Anthropic LLMs bound by `bind_crew_llm_client`; live run 2026-10-13 (including a two-provider crew on one Agent FinOps budget). A denial stops the crew without retries. Refused under Control: any unbound LLM (agent, manager, planning, function-calling), memory, knowledge and embedders, `Crew(stream=True)`, `llm_overlay`, non-native LLM classes. Not supported: Anthropic structured output (beta Messages). Cancelling native `akickoff` stops the crew and emits `workflow.error` (never `workflow.end`); a call already sent completes and is charged. `kickoff_async` runs `kickoff` on a worker thread: cancelling the awaiter does not stop it, so the crew finishes, is charged and emits `workflow.end`. [Evidence](../verification/live/2026-10-13-crewai.json) |
| AutoGen | unsupported | |

## Budgets and accounting

| Capability | Status | Details |
|---|---|---|
| Multi-model workflow budget | live-verified | Steps on different models and providers share one budget: across processes through Agent FinOps (a `workflow_id` limit), or in one process with `SharedBudget`. The next step is refused before dispatch when the budget runs out. Without either, each `monitor()` client keeps its own independent budget |
| Shared reserve/settle with Agent FinOps (PostgreSQL) | integration-verified | Explicit reservation lifecycle; crash and replay of a settlement is tested |
| Provider rejection release | live-verified | A provider HTTP 4xx (except 408/409/499) settles at $0 and releases the hold; timeouts and 5xx stay pending for reconciliation |
| Ordinary API keys with the SDK | live-verified | Keys created the normal way in Agent FinOps work with `monitor()` for their own organization |
| Event ingest | fixture-verified | `kazen-event-schema` wire versions 1.0–1.2, field aliases and conflict checks |

## Observability

| Integration | Status | Details |
|---|---|---|
| OpenTelemetry to Agent FinOps run receipts | fixture-verified | OTLP/HTTP JSON with status, links and resource attributes |
| OpenTelemetry to Datadog APM | live-verified | OpenTelemetry Collector contrib 0.128.0 with the Datadog exporter (2026-10-12): spans appear in APM Traces with the right service, environment and model resource, with no keys, auth headers or prompt text. Keep the Datadog API key on the Collector. Datadog LLM Observability rendering is not claimed |
| Run-receipt incident workflows | integration-verified | Authenticated local services with synthetic events |

## Known limits

- The settlement queue coordinates processes on one host; it is not cross-host
  coordination. Run `python -m kazenai.spine.settle_queue flush` on a schedule —
  the SDK does not start it. See [Settlement recovery](../settlement-recovery.md).
- A settlement replayed after a restart is not yet linked to the original trace.
- Control covers requests sent through a monitored client. It cannot stop code
  that holds the provider key and creates another client, sends raw HTTP, or
  keeps the SDK's private `_client` from before `monitor()`.
- `declare_langgraph_control_boundary` is an assertion by the caller; it does not
  intercept raw provider clients created inside graph nodes.
- LangGraph's default correlation store is local and expires after 24 hours.
- Not supported or not claimed: OpenAI Realtime, Responses background mode and
  built-in tools, universal provider
  support, exactly-once execution, guaranteed savings, and Datadog LLM
  Observability rendering.
