# Control FINAL_1_b_final — supported integration matrix

**Machine-readable source of truth:** [`control-supported-matrix.json`](./control-supported-matrix.json)

This matrix records bounded local evidence as of 2026-10-11, updated 2026-10-12 for the Phase 1 bug fixes and a paid live-provider smoke. Hermetic provider transports prove SDK behavior; the `live-verified` rows below were also exercised against the paid OpenAI and Anthropic APIs. `implemented-unverified` means code exists but may not be sold as a certified path.

**Packages:** published on PyPI: `kazenai` 1.1.1 and `kazenai-finops` 1.1.1 (`kazen-event-schema>=0.6.3`). Release candidates: `kazen-event-schema` 0.6.4 (aliases/conflicts, `finops.trajectory` / `finops.loop.anomaly` registration), `kazenai` 1.1.2 (requires schema `>=0.6.4`), and `kazenai-finops` 1.1.2 (requires `kazenai>=1.1.2`). Publish order: schema 0.6.4 → `kazenai` 1.1.2 → `kazenai-finops` 1.1.2 → Agent FinOps / Agent Lens (both require `kazenai>=1.1.2`). Public docs and site describe 1.1.2 and deploy after these packages are on PyPI.

## Evidence tiers

| Tier | Meaning |
|---|---|
| `unsupported` | Explicitly outside the release |
| `planned` | Intended; no implementation claim |
| `implemented-unverified` | Code exists; an acceptance defect or missing gate remains |
| `fixture-verified` | Hermetic/fake-transport tests and dated evidence |
| `integration-verified` | Real local KazenAI services exercised |
| `live-verified` | Paid provider or remote destination exercised |

## Current verified envelope

| Cell | Status | Boundary |
|---|---|---|
| OpenAI Chat Completions sync/async, stream/non-stream | live-verified | 1.109.1 ordinary suite and isolated 2.48.0 fixtures; live 2026-10-12 with gpt-6-luna on openai 2.54.0 (response, exact split-priced cost, pre-dispatch deny); Responses unsupported |
| Anthropic Messages sync/async, stream/non-stream | live-verified | 0.125.0 fixtures; live 2026-10-12 with claude-sonnet-5-5 (response, exact split-priced cost, pre-dispatch deny); Bedrock/Vertex wrappers unsupported |
| Local and PostgreSQL admission/reserve/settle | integration-verified | Explicit lifecycle; tested single-writer crash/replay path |
| LiteLLM application-owned accounting | live-verified | LiteLLM 1.104.2 proxy to real OpenAI/Anthropic + real Agent FinOps: one reservation per call, exact settlement, pre-dispatch deny; no universal router/provider claim |
| LiteLLM gateway-owned accounting | live-verified | Same proxy with the Kazen callback, non-streaming: one gateway reservation (no SDK double count), exact settlement, HTTP 402 before dispatch; gateway streaming settlement not live-verified |
| JWT runtime policy | fixture-verified | Python + Platform `jose` parity locally; HS256 only |
| Canonical event ingest | fixture-verified | Schema 0.6.4, wire 1.0–1.2 normalization, aliases/conflicts |
| OTel to Lens | fixture-verified | Local OTLP/JSON projection including status/links/resource |
| OTel to Datadog APM | live-verified | 2026-10-12: OTel Collector contrib 0.128.0 with the Datadog exporter (`DD_SITE=datadoghq.com`); a GenAI span (model, input/output tokens, `service.name`, `deployment.environment`) visible in APM Traces with correct service, env and resource; no keys, auth headers or prompt bodies in the payload. Datadog API key stays on the Collector. Datadog LLM Observability product view and `ddtrace` instrumentation not claimed |
| LangChain | live-verified | langchain 1.4.4 / langchain-openai 1.7.0: `ChatOpenAI` via `bind_monitored_chat_openai` and `ChatAnthropic` via `bind_monitored_chat_anthropic`, invoke/stream/ainvoke/astream live; `with_structured_output` (Chat Completions `parse`) guarded; ChatAnthropic `betas` rejected |
| LangGraph | live-verified | langgraph 1.2.14: interrupt/resume from SQLite and a 3-model, 2-provider workflow live; a graph declares every provider client its nodes use |
| Multi-model workflow budget | live-verified | Steps on different models/providers share one budget, via Agent FinOps (`workflow_id` limit) or, in one process, `SharedBudget`; the next step is denied before dispatch when it runs out |
| Provider rejection release | live-verified | Provider HTTP 4xx (except 408/409/499) settles at $0 and releases the hold; timeouts and 5xx stay pending |
| Ordinary API keys with the SDK | live-verified | Keys created the normal way work with `monitor()` for their own organization |
| CrewAI | implemented-unverified | Hook-abort Control cell passes; 1.1.2 routes real-bus LLM/task/tool events to the owning crew; cancellation, nested runs and native `akickoff` Observe cells unverified — not advertised. `bind_crew_llm_client` runs a sequential two-provider crew under one budget (hermetic and live 2026-10-12, kazenai-demo-agent example 19) |
| Agent Lens incident workflows | integration-verified | Authenticated local services with synthetic events |

## Active limitations

- The file-backed settlement queue coordinates processes on one host through atomic claim files (1.1.2); it is not cross-host coordination. Recovery runs via the supervised `python -m kazenai.spine.settle_queue flush` command (single-owner lock, quarantine for corrupt records, backlog stats); the SDK does not start it automatically. Host power-loss durability is not proved.
- Core and Platform CI use a vendored JWT corpus with a SHA-256 manifest; tests fail if it drifts from `kazenai-contracts`.
- Core CI fails when the committed KazenEvent schema export drifts from the model, and core tests fail if the SDK emits an unregistered event type. FinOps/Lens/Platform do not yet share one producer-to-persistence CI gate.
- Detached settlement replay is not yet linked to the original trace/span (A6).
- The async behavioral coverage gate runs three maintained suites in Core CI and passes locally at 69% versus its 35% floor.
- `declare_langgraph_control_boundary` is a caller assertion. It does not intercept arbitrary raw clients in graph nodes.
- LangGraph's default correlation side store is local, temporary, and 24-hour bounded.
- OpenAI Responses, Realtime, generic replay, universal-provider support, exactly-once execution, guaranteed savings, and Datadog LLM Observability rendering are unsupported or not claimed. Datadog is verified only as an APM trace destination through an OTel Collector.

## Rule for changes

1. Edit `control-supported-matrix.json` first.
2. Add tests and dated evidence before raising a status.
3. Provision shared fixtures inside each standalone CI workflow that claims conformance.
4. Keep unsupported claims excluded from customer material.
