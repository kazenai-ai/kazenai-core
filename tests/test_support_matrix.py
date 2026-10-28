"""Control capability matrix integrity.

Every advertised cell (fixture-verified+) must cite dated evidence + tests.
Forbidden claims must only appear alongside unsupported labels in the matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs" / "integrations" / "control-supported-matrix.json"


def _core_test_path(ref: str) -> Path | None:
    """Return the in-repo path for a core test reference, else None."""
    path = ref.split("::", 1)[0]
    if path.startswith("kazenai-core/"):
        path = path[len("kazenai-core/") :]
    if path.startswith("tests/") and path.endswith(".py"):
        return ROOT / path
    return None


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


def test_matrix_schema_and_unique_ids():
    data = _load()
    assert data["schema_version"] == "kazenai.support-matrix.v1"
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


def test_advertised_cells_have_evidence_and_tests():
    """Advertised cells cite dated evidence and at least one guarding suite.

    Core test references must exist in this repository; references to suites in
    other repositories are descriptive labels.
    """
    data = _load()
    missing = []
    for cell in data["cells"]:
        if cell["status"] not in ADVERTISED:
            continue
        evidence = [e for e in cell.get("evidence") or [] if isinstance(e, str) and e.strip()]
        tests = [t for t in cell.get("tests") or [] if isinstance(t, str) and t.strip()]
        if not evidence or not tests:
            missing.append(cell["id"])
            continue
        for ref in tests:
            path = _core_test_path(ref)
            if path is not None:
                assert path.is_file(), f"{cell['id']}: missing test {ref}"
    assert not missing, f"advertised cells missing evidence/tests: {missing}"


def test_required_unsupported_exclusions_present():
    data = _load()
    by_id = {c["id"]: c for c in data["cells"]}
    # Framework cells: langchain/langgraph/crewai + narrow checkpoint resume.
    required_unsupported = [
        "provider.openai.unguarded_surfaces",
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

    # Control streaming is certified (sync OpenAI + Anthropic surfaces).
    assert by_id["mode.streaming.control"]["status"] in ADVERTISED
    assert by_id["provider.openai.chat_completions.sync.stream"]["status"] in ADVERTISED
    assert by_id["provider.anthropic.messages.sync.stream"]["status"] in ADVERTISED


def test_live_provider_is_scoped_to_the_verified_paths():
    by_id = {c["id"]: c for c in _load()["cells"]}
    live = by_id["capability.live_provider"]
    assert live["status"] == "live-verified"
    assert "kazenai-core/tests/test_live_provider_smoke.py" in live["tests"]
    for excluded in ("Realtime", "Bedrock", "framework"):
        assert excluded in live["notes"], excluded
    # Responses carries its own live evidence (2026-10-13 smoke).
    responses = by_id["provider.openai.responses"]
    assert responses["status"] == "live-verified"
    assert "kazenai-core/tests/test_live_responses_smoke.py" in responses["tests"]
    assert any(e.startswith("Live provider run") for e in responses["evidence"])


def test_litellm_has_machine_readable_cells():
    by_id = {c["id"]: c for c in _load()["cells"]}
    for cid in ("gateway.litellm.app_owned", "gateway.litellm.gateway_owned"):
        assert by_id[cid]["status"] in ADVERTISED, cid
    # Gateway-owned streaming settlement was not exercised live; the cell must say so.
    assert "streaming settlement is not live-verified" in by_id["gateway.litellm.gateway_owned"]["notes"]
    assert "own independent budget" in by_id["capability.multi_model_workflow_budget"]["notes"]


def test_frameworks_fixture_verified():
    data = _load()
    by_id = {c["id"]: c for c in data["cells"]}
    for cid in (
        "framework.langchain",
        "framework.langgraph",
        "framework.crewai",
        "capability.checkpoint_resume",
    ):
        assert by_id[cid]["status"] in ADVERTISED, cid
    # CrewAI is certified on the certification cells plus live cells, with its
    # refusals and unsupported paths spelled out.
    crew = by_id["framework.crewai"]
    assert crew["status"] == "live-verified"
    assert "kazenai-core/tests/test_crewai_control_certification.py" in crew["tests"]
    for limit in ("memory", "llm_overlay", "Crew(stream=True)", "Anthropic structured output"):
        assert limit in crew["notes"], limit
    assert "kazenai-core/tests/test_crewai_event_ownership.py" in crew["tests"]


def test_envelope_matches_locked_providers():
    data = _load()
    env = data["envelope"]["adapter"].lower()
    assert "openai" in env and "anthropic" in env
    assert "sync" in env or "non-streaming" in env
    by_id = {c["id"]: c for c in data["cells"]}
    assert by_id["provider.openai.chat_completions.sync"]["status"] in ADVERTISED
    assert by_id["provider.anthropic.messages.sync"]["status"] in ADVERTISED


def test_forbidden_claims_not_in_matrix_as_supported():
    data = _load()
    for cell in data["cells"]:
        blob = json.dumps(cell).lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase in blob and cell["status"] in ADVERTISED:
                pytest.fail(f"{cell['id']} advertises forbidden phrase {phrase!r}")


def test_human_matrix_markdown_exists():
    md = ROOT / "docs" / "integrations" / "control-supported-matrix.md"
    assert md.is_file()
    text = md.read_text(encoding="utf-8")
    assert "control-supported-matrix.json" in text
    assert "fixture-verified" in text


def test_release_state_matches_the_docs():
    """Packages may be described as on PyPI only once the matrix says they were published."""
    release = _load()["release"]
    assert release["state"] in {"unreleased", "published"}
    if release["state"] == "published":
        return
    for doc in (MATRIX.with_suffix(".md"), ROOT / "README.md", ROOT / "QUICKSTART.md"):
        if not doc.is_file():
            continue
        for line in doc.read_text(encoding="utf-8").splitlines():
            if release["kazenai"] in line:
                assert "on PyPI" not in line, f"{doc.name}: {line.strip()[:120]}"
    assert "not yet released" in MATRIX.with_suffix(".md").read_text(encoding="utf-8")


def test_live_evidence_files_exist_and_end_with_a_passing_run():
    linked = [
        evidence.split(": ", 1)[1]
        for cell in _load()["cells"]
        for evidence in cell.get("evidence", [])
        if ": kazenai-core/docs/verification/live/" in evidence
    ]
    assert linked, "no live evidence summaries are linked"
    for ref in linked:
        path = ROOT / ref.removeprefix("kazenai-core/")
        assert path.is_file(), ref
        summary = json.loads(path.read_text(encoding="utf-8"))
        assert summary["runs"], ref
        assert summary["runs"][-1]["all_passed"] is True, ref
        assert all(len(run["sha256"]) == 64 for run in summary["runs"]), ref
