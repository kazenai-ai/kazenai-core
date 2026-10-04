# Architecture notes — `kazenai` Core

This document is for maintainers and open-source readers. It describes current
shape and planned modularization. Public import paths stay stable unless a
release notes a breaking change.

## Public API (stable)

```python
from kazenai import monitor, BudgetExceeded, Enforcement
```

`monitor()` remains the certified Control entrypoint for sync OpenAI Chat
Completions and sync Anthropic Messages (see the supported-surfaces matrix and
docs). Callers must not depend on private module layout.

## Known concentration: `kazenai/monitor.py`

`monitor.py` is currently the certified Control entrypoint implementation and
has grown into a large module (OpenAI patch, Anthropic patch, streaming
finalize/emit, local hold + shared reserve helpers, event payload assembly).

This is tracked technical debt, not an invitation to rewrite behavior ad hoc.

### Next engineering track — mechanical split (no behavior change)

Planned extractions modules (same tests, same public API):

| Module | Responsibility |
|--------|----------------|
| `kazenai/monitor_openai.py` | OpenAI `chat.completions` patch + stream manager wiring |
| `kazenai/monitor_anthropic.py` | Anthropic `messages` patch + stream manager wiring |
| `kazenai/monitor_emit.py` | `model.call` payload builders (reservation + attribution fields) |
| reserve helpers | Local hold / FinOps reserve wrappers (stay near spine or a small helper module) |

`kazenai/monitor.py` becomes a **thin facade**: `monitor()`, shared constants,
and re-exports used by tests. `from kazenai import monitor` does not change.

### Rule going forward

- **New surfaces** do not grow the god file; they land in new modules.
- Splits must be mechanical: move code, keep tests green, no semantic change.
- Do not couple a modularization PR to a product feature or a forced PyPI bump
  unless the release explicitly ships the refactor.

## Related modules (already split)

- `kazenai/streaming/` — stream lifecycle, OpenAI/Anthropic stream wrappers
- `kazenai/spine/guard.py` — shared FinOps reservation
- `kazenai/attribution.py` — ContextVar attribution
- `kazenai/enforcement.py` — local hard caps

## Versioning note

Attribution-on-`model.call` and gated live provider smokes may land on `main`
before a PyPI patch. Until a patch release is published, public PyPI install
truth remains the last tagged package version (see `RELEASING.md`).
