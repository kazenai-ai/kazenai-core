# KazenAI Core quick start

`kazenai` is the Python runtime used by KazenAI to enforce local and shared
spending policies around supported synchronous OpenAI and Anthropic calls. Most
product integrations should install the customer-facing package, which
re-exports the Core API:

```bash
python -m pip install "kazenai-finops==1.1.0" openai
```

Install Core directly when you specifically want the runtime package:

```bash
python -m pip install "kazenai==1.1.0" openai
```

Python 3.10, 3.11 and 3.12 are supported.

## 1. Put a local hard cap around a client

```python
from openai import OpenAI
from kazenai import BudgetExceeded, monitor

client = monitor(
    OpenAI(),
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

`max_budget_usd` is an in-process client/run limit. It is not an account-wide
provider billing limit. For multiple workers or services sharing a budget, use
the shared FinOps authority described below.

## 2. Connect a shared FinOps authority

```bash
export KAZENAI_FINOPS_URL="https://finops.example.com"
export KAZENAI_FINOPS_API_KEY="kz_..."
export KAZENAI_DEPLOYMENT_MODE="production"
```

The same `monitor(...)` call now reserves the estimated cost before dispatch
and reconciles authoritative usage afterward. Production and staging modes fail
closed when a required shared reservation cannot be obtained. Explicit
development fail-open behavior retains only the local client safeguards; it
does not preserve the shared budget guarantee.

## 3. Stream with finalization-safe accounting

OpenAI `create(stream=True)` is supported:

```python
from openai import OpenAI
from kazenai import StreamCutoffError, monitor

client = monitor(
    OpenAI(),
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

client = monitor(Anthropic(), max_budget_usd=1.00)

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
arrives. Cancellation, provider failure, or missing final usage produces an
`outcome_unknown`/pending-reconciliation state. It is never treated as an exact
zero-cost call and never released as though dispatch did not happen.

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

This explicit guard can reserve around a custom callable, but it does not make
arbitrary provider response parsing a certified integration. The caller remains
responsible for passing valid price and usage information.

## Supported boundary in 1.1.0

| Path | Status |
|---|---|
| Sync OpenAI `chat.completions.create` | Supported, non-streaming and `stream=True` |
| Sync OpenAI `chat.completions.stream` | Supported |
| Sync Anthropic `messages.create` | Supported, non-streaming and `stream=True` |
| Sync Anthropic `messages.stream` | Supported |
| Async provider clients | Not supported |
| OpenAI Responses streaming | Not supported |
| OpenAI Realtime/WebSocket | Not supported |
| Bedrock/Vertex Anthropic wrappers | Not supported |
| Framework helpers | Evaluation surface; wrap the underlying supported client |

## Next references

- [README.md](README.md) — capability overview and configuration
- [DEPLOY.md](DEPLOY.md) — application deployment guidance
- [RELEASING.md](RELEASING.md) — maintainer release procedure
- [docs.kazenai.com](https://docs.kazenai.com/) — product documentation
