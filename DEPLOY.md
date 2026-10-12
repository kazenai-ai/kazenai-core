# Deploying KazenAI Core

`kazenai` is a Python SDK embedded in an application process. This repository
does not publish or run a standalone Core HTTP service. The optional shared
budget authority and event ingest endpoint are supplied by Agent FinOps.

## Supported runtime

- Python 3.10, 3.11 or 3.12
- OpenAI Chat Completions clients, sync and async, streaming and non-streaming
- Anthropic Messages clients, sync and async, streaming and non-streaming

See [QUICKSTART.md](QUICKSTART.md) for the exact supported call paths and
[docs/integrations/control-supported-matrix.md](docs/integrations/control-supported-matrix.md)
for the evidence behind each one.

## Verify a source checkout

Use a fresh virtual environment and install the development extras:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
ruff check kazenai tests
mypy kazenai
pytest --cov=kazenai --cov-fail-under=80
```

CI must pass on Python 3.10, 3.11 and 3.12 before a release. Maintainers should
follow [RELEASING.md](RELEASING.md) rather than publishing artifacts made from a
working tree.

## Build an application image

Pin the released version in the application that embeds the SDK:

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "-m", "your_application"]
```

```text
# requirements.txt
kazenai[openai]==1.1.2
```

Most customer integrations should pin `kazenai-finops[openai]==1.1.2` instead; that
package installs a compatible Core version transitively and bounds the
provider SDK range.

## Runtime configuration

Secrets belong in the deployment platform's secret manager, not in the image or
repository.

| Variable | Purpose |
|---|---|
| `KAZENAI_FINOPS_URL` / `KAZENAI_FINOPS_INGEST_URL` | Shared FinOps authority and event-ingest base URL |
| `KAZENAI_FINOPS_API_KEY` | Credential used for FinOps requests |
| `KAZENAI_DEPLOYMENT_MODE` | Use `production` or `staging` to require fail-closed shared reservations |
| `KAZENAI_FINOPS_RESERVE_TIMEOUT_S` | Timeout for a shared pre-call reservation |
| `KAZENAI_TIMELINE_PATH` | Optional local JSONL evidence path |
| `KAZENAI_SDK_SETTLE_QUEUE_DIR` | Persistent directory for the settlement journal; run `python -m kazenai.spine.settle_queue flush` on a schedule (see [docs/settlement-recovery.md](docs/settlement-recovery.md)) |

The local `max_budget_usd` passed to `monitor()` remains an in-process guard for
that client (`SharedBudget` shares one in-process cap across several clients).
A shared cross-process guarantee requires the FinOps authority to be configured
and successfully reserving calls. Pass tenant and attribution values directly
to `monitor()` rather than relying on application-specific environment names.

## Deployment smoke test

Core has no HTTP health endpoint. Exercise the same provider path that the
application will use and verify all of the following in a non-production
tenant:

1. A permitted request completes and records exactly one accounting lifecycle.
2. A request above the configured budget is denied before provider dispatch.
3. A completed stream settles authoritative usage once.
4. A cancelled or failed stream remains pending/outcome-unknown rather than
   settling to zero.
5. In production mode, an unavailable required FinOps authority denies the call.

Keep provider keys and FinOps credentials out of captured test output.
