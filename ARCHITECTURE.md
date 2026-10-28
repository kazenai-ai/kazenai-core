# Architecture

`kazenai` is an in-process SDK. It wraps OpenAI and Anthropic clients, checks
each call against a budget before it is sent, and records what the call cost
once the provider reports usage. It can work entirely locally or with Agent
FinOps as a shared spending authority.

## Public API

```python
from kazenai import monitor, SharedBudget, BudgetExceeded, Enforcement
```

`monitor()` is the entry point for OpenAI Chat Completions and Responses, and Anthropic
Messages, sync and async, streaming and non-streaming. Import from the package
root; module layout below it is internal and may change between releases.

## How a call flows

1. **Admission.** `monitor()` estimates the call's cost from the model's price
   and output limit, checks the local budget (`max_budget_usd` or a
   `SharedBudget`) and loop detector, and — when Agent FinOps is configured —
   reserves the estimate there. A call that does not fit is refused with
   `BudgetExceeded` before anything is sent.
2. **Dispatch.** The reservation is marked as started, then the provider SDK
   sends the request.
3. **Settlement.** When the provider reports usage, the call is priced on its
   input and output tokens and the reservation settles at that amount. If usage
   never arrives (cancellation, timeout, server error), the reservation stays
   pending for reconciliation instead of settling at zero. A definitive provider
   refusal (HTTP 4xx) settles at $0.
4. **Evidence.** Each step emits a `KazenEvent` (from `kazen-event-schema`) to
   the configured sinks and, optionally, an OpenTelemetry span.

## Modules

| Module | Responsibility |
|--------|----------------|
| `kazenai/monitor.py` | `monitor()` and the OpenAI / Anthropic sync wrappers |
| `kazenai/_async_provider.py` | Async OpenAI / Anthropic wrappers |
| `kazenai/streaming/` | Stream lifecycle and provider stream wrappers |
| `kazenai/enforcement.py`, `kazenai/budget.py` | Local caps; `SharedBudget` shares one cap across clients |
| `kazenai/loop_detector.py` | Repeated-call (loop) detection |
| `kazenai/model_pricing.py` | The price table (`PRICING_LAST_VERIFIED`, `PRICING_SOURCES`) |
| `kazenai/provider_errors.py` | Which provider errors settle at $0 and which stay pending |
| `kazenai/spine/` | Shared reservations with Agent FinOps and the settlement journal |
| `kazenai/attribution.py` | Customer, feature and workflow attribution |
| `kazenai/integrations/` | LangChain, LangGraph, CrewAI and LiteLLM adapters |
| `kazenai/otel.py` | Optional OpenTelemetry spans |

## Design rules

- **Fail closed in production.** When a shared reservation is required and
  cannot be obtained, the call is refused.
- **Never record an unknown outcome as zero.** Missing usage means pending, not
  free.
- **Metadata by default.** Prompts and completions are not captured unless
  explicitly enabled; secrets are redacted before any sink.
- **No hidden network calls on import.** Everything optional (FinOps, OTel,
  frameworks) is enabled by configuration or extras.
