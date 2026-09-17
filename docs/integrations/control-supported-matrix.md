# Control FINAL_1 — supported integration matrix (P4-1)

**Machine-readable source of truth:** [`control-supported-matrix.json`](./control-supported-matrix.json)

This matrix records what Control FINAL_1 **actually certifies**. Fake/local
transport proves integration only — not live-provider operation. Adding a
capability requires deferred tests **before** raising its status above
`implemented-unverified`.

## Envelope (locked 2026-09-16)

| Dimension | Control FINAL_1 |
|-----------|-----------------|
| Adapter | Sync non-streaming **OpenAI Chat Completions** + **Anthropic Messages** via `kazenai.monitor` |
| Package | `from kazenai import monitor` (distribution `kazenai` 1.0.1 local) |
| Demo | Fake provider + real local Control services |
| Live provider | Separate USER-GO (`live-verified` not claimed) |
| Topology | Additive Control compose; Brain / Builder / Copilot / Home **absent** |

## Evidence tiers

| Tier | Meaning |
|------|---------|
| `unsupported` | Explicitly out of Control FINAL_1 (or FINAL_2) |
| `planned` | Intended; no implementation claim |
| `implemented-unverified` | Code exists; not Control-certified |
| `fixture-verified` | Hermetic/fake-transport tests + dated artifact |
| `integration-verified` | Real local Control services exercised |
| `live-verified` | Paid provider path — USER-GO only |

## Supported (certified) cells

| ID | Status | Evidence |
|----|--------|----------|
| `sdk.kazenai.monitor` | fixture-verified | P3-1, P3-2 |
| `provider.openai.chat_completions.sync` | fixture-verified | P3-2 |
| `provider.anthropic.messages.sync` | fixture-verified | P3-2 |
| `admission.pg_reserve_settle` | integration-verified | P2-2, P2-3, ADR-015 |
| `admission.local_max_budget_usd` | fixture-verified | P3-1 / G06 |
| `pricing.token_cost_engine` | fixture-verified | P3-1 |
| `privacy.capture_metadata_default` | fixture-verified | P3-3 |
| `telemetry.lineage_ids` | fixture-verified | P3-5 |
| `telemetry.sinks_lifecycle` | fixture-verified | P3-4 |
| `telemetry.finops_lens_multisink` | fixture-verified | P3-5 / ADR-016 |
| `auth.control_identity` | integration-verified | P1-4 |
| `tools.model_call_only` | fixture-verified | monitor helpers |
| `example.control_loop_and_failure` | fixture-verified | P4-5 |

## Explicitly unsupported / unverified

| ID | Status | Why |
|----|--------|-----|
| OpenAI Responses / async / Control streaming | unsupported | Rejected on certified path (P3-2) |
| LangChain / CrewAI / AutoGen | unsupported | Not Control-certified |
| LangGraph example / adapter | implemented-unverified | P4-5 D1: fake-client offline only; ChatAnthropic unverified |
| `example.control_loop_and_failure` | fixture-verified | P4-5 / D2–D7 fixture path |
| Generic replay / drift calibration | unsupported | FINAL_2 |
| MCP-only spend (Cursor / Claude Code / Brain MCP) | unsupported | Not Control topology |
| Subscription invoice attribution | unsupported | Forbidden claim |
| Checkpoint/resume | unsupported | Soft CB ≠ durable resume |
| Universal provider / guaranteed savings | unsupported | Forbidden claims |
| Live provider operation | unsupported | USER-GO; fake ≠ live |

## Rule for changes

1. Edit `control-supported-matrix.json` first.
2. Add tests + dated `artifacts/final-1-*` before raising status to `fixture-verified+`.
3. `tests/test_final1_p4_1_capability_manifest.py` must pass.
4. Do not advertise forbidden claims unless the cell remains `unsupported`.
