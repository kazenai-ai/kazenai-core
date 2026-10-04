# Control FINAL_1 + FINAL_LENS — supported integration matrix

**Machine-readable source of truth:** [`control-supported-matrix.json`](./control-supported-matrix.json)

This matrix records what Control FINAL_1 plus the Train B streaming release
**actually certifies**. Official SDKs with hermetic mock transports prove the
client integration only — not live-provider operation. Adding a
capability requires deferred tests **before** raising its status above
`implemented-unverified`.

## Envelope (streaming extension verified 2026-10-02)

| Dimension | Control FINAL_1 |
|-----------|-----------------|
| Adapter | Sync **OpenAI Chat Completions** + **Anthropic Messages**, non-streaming and selected streaming surfaces, via `kazenai.monitor` |
| Package | `from kazenai import monitor` (distribution `kazenai` 1.1.1 prepared) |
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
| `sdk.kazenai_finops.wrapper` | fixture-verified | identity re-export only (`monitor is kazenai.monitor`) |
| `provider.openai.chat_completions.sync` | fixture-verified | P3-2 |
| `provider.openai.chat_completions.sync.stream` | fixture-verified | official OpenAI SDK + mock transport; `create(stream=True)` and `.stream()` helper |
| `provider.anthropic.messages.sync` | fixture-verified | P3-2 |
| `provider.anthropic.messages.sync.stream` | fixture-verified | official Anthropic SDK + mock transport |
| `mode.streaming.control` | fixture-verified | Train B cleanup artifact `cleanup-train-b-streaming-verification-20261002T012400Z.json` |
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
| `lens.incident_inbox` | integration-verified | FINAL_LENS P1/P6 |
| `lens.incident_reviews` | integration-verified | FINAL_LENS P2/P6 |
| `lens.regression_export` | integration-verified | FINAL_LENS P3/P6 |
| `lens.run_comparison` | integration-verified | FINAL_LENS P4/P6 |
| `lens.incident_digest_local_capture` | integration-verified | FINAL_LENS P5/P6 |

The FINAL_LENS cells mean authenticated real **local** services with synthetic
events and a provider-incapable capture transport. They do not mean external
notification, remote deployment, customer validation, causal improvement or ROI.

## Explicitly unsupported / unverified

| ID | Status | Why |
|----|--------|-----|
| OpenAI Responses / async / Realtime | unsupported | Rejected or outside the certified path |
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
