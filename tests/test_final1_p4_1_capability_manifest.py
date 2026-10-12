"""FINAL_1 P4-1 — Control capability matrix integrity.

Every advertised cell (fixture-verified+) must cite dated evidence + tests.
Forbidden claims must only appear alongside unsupported labels in the matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

def _matrix_root() -> Path:
    """Prefer workspace monorepo matrix when artifacts exist; else in-repo copy."""
    core = Path(__file__).resolve().parents[1]
    workspace = Path(__file__).resolve().parents[2]
    ws_matrix = workspace / "docs" / "integrations" / "control-supported-matrix.json"
    if ws_matrix.is_file() and (workspace / "artifacts").is_dir():
        return workspace
    return core


ROOT = _matrix_root()
MATRIX = ROOT / "docs" / "integrations" / "control-supported-matrix.json"


def _file_present(rel: str) -> bool:
    """Resolve evidence/test paths for workspace or standalone core checkout."""
    from glob import glob

    roots = [ROOT]
    if ROOT.name == "kazenai-core":
        roots.append(ROOT.parent)
    for root in roots:
        if "*" in rel:
            if any(Path(m).is_file() for m in glob(str(root / rel))):
                return True
            continue
        candidates = [root / rel]
        if rel.startswith("kazenai-core/"):
            candidates.append(root / rel[len("kazenai-core/") :])
        if any(path.is_file() for path in candidates):
            return True
    return False


def _standalone_core_ci() -> bool:
    """True when FINAL_1 artifacts are not available (GitHub Actions core-only)."""
    return not _file_present("artifacts/final-1-p3-1-verification-20260916T195500Z.json")


ADVERTISED = frozenset(
    {"fixture-verified", "integration-verified", "live-verified"}
)
FORBIDDEN_PHRASES = (
    "guaranteed savings",
    "universal provider",
    "subscription invoice",
)


def _load() -> dict:
    assert MATRIX.is_file(), f"missing matrix: {MATRIX}"
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def test_p41_matrix_schema_and_unique_ids():
    data = _load()
    assert data["schema_version"] == "final1.p4-1.v1"
    cells = data["cells"]
    assert len(cells) >= 20
    ids = [c["id"] for c in cells]
    assert len(ids) == len(set(ids))
    for tier in data["evidence_tiers"]:
        assert tier in {
            "unsupported",
            "planned",
            "implemented-unverified",
            "fixture-verified",
            "integration-verified",
            "live-verified",
        }


def test_p41_advertised_cells_have_evidence_and_tests():
    data = _load()
    missing = []
    standalone = _standalone_core_ci()
    for cell in data["cells"]:
        status = cell["status"]
        if status not in ADVERTISED:
            continue
        evidence = cell.get("evidence") or []
        tests = cell.get("tests") or []
        if not evidence or not tests:
            missing.append(cell["id"])
            continue
        for path in evidence:
            if _file_present(path):
                continue
            if standalone and (
                path.startswith("artifacts/")
                or path.startswith("docs/")
                or path.startswith("kazenai-agent-")
                or path.startswith("kazenai-examples/")
                or path.startswith("kazenai-finops-sdk/")
                or path.startswith("kazenai-contracts/")
                or path.startswith("scripts/")
                or path.startswith("tests/e2e/")
            ):
                # Citations remain required; workspace evidence is not shipped in core CI.
                continue
            assert False, f"{cell['id']}: missing evidence {path}"
        for tref in tests:
            path = tref.split("::", 1)[0]
            if path.endswith("*"):
                continue
            if _file_present(path):
                continue
            # Standalone core CI only enforces in-repo unit tests under tests/
            # (not workspace Playwright under tests/e2e/ or sibling repos).
            if standalone and not (
                path.startswith("kazenai-core/tests/")
                or (
                    path.startswith("tests/")
                    and not path.startswith("tests/e2e/")
                )
            ):
                continue
            assert False, f"{cell['id']}: missing test {path}"
    assert not missing, f"advertised cells missing evidence/tests: {missing}"


def test_p41_required_unsupported_exclusions_present():
    data = _load()
    by_id = {c["id"]: c for c in data["cells"]}
    # FINAL_1_b_final P8 certifies langchain/langgraph/crewai + narrow checkpoint resume.
    required_unsupported = [
        "provider.openai.responses",
        "capability.generic_replay",
        "capability.drift_calibration",
        "capability.mcp_only_spend_control",
        "capability.subscription_cost_attribution",
        "capability.universal_provider",
        "capability.guaranteed_savings",
        "framework.autogen",
    ]
    for cid in required_unsupported:
        assert cid in by_id, f"missing exclusion cell {cid}"
        assert by_id[cid]["status"] == "unsupported", cid

    # Train B: Control streaming is certified (sync OpenAI + Anthropic surfaces).
    assert by_id["mode.streaming.control"]["status"] in ADVERTISED
    assert by_id["provider.openai.chat_completions.sync.stream"]["status"] in ADVERTISED
    assert by_id["provider.anthropic.messages.sync.stream"]["status"] in ADVERTISED


def test_p41_live_provider_is_scoped_to_the_verified_paths():
    by_id = {c["id"]: c for c in _load()["cells"]}
    live = by_id["capability.live_provider"]
    assert live["status"] == "live-verified"
    assert "kazenai-core/tests/test_live_provider_smoke.py" in live["tests"]
    for excluded in ("Responses", "Bedrock", "framework"):
        assert excluded in live["notes"], excluded
    # Unverified surfaces must not ride along.
    assert by_id["provider.openai.responses"]["status"] == "unsupported"


def test_p41_litellm_has_machine_readable_cells():
    by_id = {c["id"]: c for c in _load()["cells"]}
    for cid in ("gateway.litellm.app_owned", "gateway.litellm.gateway_owned"):
        assert by_id[cid]["status"] in ADVERTISED, cid
    # Gateway-owned streaming settlement was not exercised live; the cell must say so.
    assert "streaming settlement is not live-verified" in by_id["gateway.litellm.gateway_owned"]["notes"]
    assert "own independent budget" in by_id["capability.multi_model_workflow_budget"]["notes"]


def test_p41_p8_frameworks_fixture_verified():
    data = _load()
    by_id = {c["id"]: c for c in data["cells"]}
    for cid in (
        "framework.langchain",
        "framework.langgraph",
        "capability.checkpoint_resume",
    ):
        assert by_id[cid]["status"] in ADVERTISED, cid
    # Ownership routing for real CrewAI 1.15.26 sources is tested (A2), but
    # cancellation / nested / native akickoff Observe cells are not; the cell
    # must stay unadvertised until they are.
    crew = by_id["framework.crewai"]
    assert crew["status"] == "implemented-unverified"
    assert "not advertised" in crew.get("notes", "")
    assert "kazenai-core/tests/test_crewai_ownership_a2.py" in crew["tests"]


def test_p41_envelope_matches_locked_providers():
    data = _load()
    env = data["envelope"]["adapter"].lower()
    assert "openai" in env and "anthropic" in env
    assert "sync" in env or "non-streaming" in env
    by_id = {c["id"]: c for c in data["cells"]}
    assert by_id["provider.openai.chat_completions.sync"]["status"] in ADVERTISED
    assert by_id["provider.anthropic.messages.sync"]["status"] in ADVERTISED


def test_p41_forbidden_claims_not_in_matrix_as_supported():
    data = _load()
    for cell in data["cells"]:
        blob = json.dumps(cell).lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase in blob and cell["status"] in ADVERTISED:
                pytest.fail(f"{cell['id']} advertises forbidden phrase {phrase!r}")


def test_p41_human_matrix_markdown_exists():
    md = ROOT / "docs" / "integrations" / "control-supported-matrix.md"
    assert md.is_file()
    text = md.read_text(encoding="utf-8")
    assert "control-supported-matrix.json" in text
    assert "fixture-verified" in text
