"""FINAL_1 P4-1 — Control capability matrix integrity.

Every advertised cell (fixture-verified+) must cite dated evidence + tests.
Forbidden claims must only appear alongside unsupported labels in the matrix.
"""

from __future__ import annotations

import json
import re
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
    candidates = [ROOT / rel]
    if rel.startswith("kazenai-core/"):
        candidates.append(ROOT / rel[len("kazenai-core/") :])
    if ROOT.name == "kazenai-core":
        candidates.append(ROOT.parent / rel)
    return any(p.is_file() for p in candidates)


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
                or path.startswith("scripts/")
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
            if standalone and not (
                path.startswith("kazenai-core/tests/") or path.startswith("tests/")
            ):
                continue
            assert False, f"{cell['id']}: missing test {path}"
    assert not missing, f"advertised cells missing evidence/tests: {missing}"


def test_p41_required_unsupported_exclusions_present():
    data = _load()
    by_id = {c["id"]: c for c in data["cells"]}
    required_unsupported = [
        "mode.async_clients",
        "mode.streaming.control",
        "provider.openai.responses",
        "capability.generic_replay",
        "capability.drift_calibration",
        "capability.mcp_only_spend_control",
        "capability.subscription_cost_attribution",
        "capability.checkpoint_resume",
        "capability.universal_provider",
        "capability.guaranteed_savings",
        "capability.live_provider",
        "framework.langchain",
        "framework.crewai",
        "framework.autogen",
    ]
    for cid in required_unsupported:
        assert cid in by_id, f"missing exclusion cell {cid}"
        assert by_id[cid]["status"] == "unsupported", cid


def test_p41_langgraph_not_advertised_as_verified():
    data = _load()
    cell = next(c for c in data["cells"] if c["id"] == "framework.langgraph")
    assert cell["status"] in {"unsupported", "implemented-unverified", "planned"}
    assert cell["status"] not in ADVERTISED


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


def test_p41_doc_demotions_have_control_banner():
    data = _load()
    for item in data.get("doc_demotions") or []:
        rel = item["path"]
        # Workspace paths like kazenai-core/README.md → README.md in standalone core.
        candidates = [ROOT / rel]
        if rel.startswith("kazenai-core/"):
            candidates.append(ROOT / rel[len("kazenai-core/") :])
        if ROOT.name == "kazenai-core" and not rel.startswith("kazenai-core/"):
            candidates.append(ROOT / rel)
        path = next((p for p in candidates if p.is_file()), None)
        if path is None and _standalone_core_ci():
            continue
        assert path is not None, rel
        text = path.read_text(encoding="utf-8")
        assert re.search(r"Control FINAL_1|FINAL_1 Control|not Control", text, re.I), (
            f"{path} missing Control FINAL_1 demotion banner"
        )


def test_p41_human_matrix_markdown_exists():
    md = ROOT / "docs" / "integrations" / "control-supported-matrix.md"
    assert md.is_file()
    text = md.read_text(encoding="utf-8")
    assert "control-supported-matrix.json" in text
    assert "fixture-verified" in text
