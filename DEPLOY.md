# kazenai-core — Deploy

Python SDK for agent monitoring, FinOps budget enforcement, and event ingest. Published to PyPI as `kazenai`; not a standalone HTTP service.

## Prerequisites

- Python 3.10+
- Optional: FinOps ingest endpoint for cloud telemetry

## Build / test

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest --cov=kazenai --cov-fail-under=80
python -m build   # wheel + sdist (requires build package)
```

## Docker

No first-party runtime image. Embed in your agent container:

```dockerfile
RUN pip install kazenai
ENV KAZENAI_FINOPS_INGEST_URL=https://finops.example.com
ENV KAZENAI_FINOPS_API_KEY=kz_...
```

## Required environment (runtime)

| Variable | Purpose |
|----------|---------|
| `KAZENAI_FINOPS_INGEST_URL` | FinOps base URL for event batch ingest |
| `KAZENAI_FINOPS_API_KEY` | API key for `POST /v1/events` |
| `KAZENAI_BUDGET_USD` | Per-run soft budget / circuit breaker |
| `KAZENAI_ORG_ID` / `KAZENAI_PROJECT_ID` | Tenant labels on events |

## Health / verification

SDK has no HTTP health endpoint. Verify FinOps connectivity:

```bash
curl -sf "${KAZENAI_FINOPS_INGEST_URL}/health"
```

Or run the package smoke test: `pytest tests/ -q`.
