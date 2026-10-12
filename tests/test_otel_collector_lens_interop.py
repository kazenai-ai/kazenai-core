"""Live Collector → Lens /v1/traces interop."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "kazenai-examples" / "otel-collector" / "live_otel_lens_interop.py"
ARTIFACTS = REPO / "artifacts"
LENS_PYTHON = REPO / "kazenai-agent-lens" / "backend" / ".venv" / "bin" / "python"


@pytest.fixture(scope="module")
def _live_artifact():
    if not SCRIPT.is_file():
        pytest.skip("live interop script missing")
    py = str(LENS_PYTHON) if LENS_PYTHON.is_file() else sys.executable
    proc = subprocess.run(
        [py, str(SCRIPT)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        combined = f"{proc.stdout}\n{proc.stderr}"
        # Ordinary suite must not hard-fail when Docker/live collector is unavailable.
        if "docker_available" in combined or "docker info failed" in combined:
            pytest.skip(
                "live OTEL interop skipped (Docker/collector unavailable)\n"
                f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
            )
        pytest.fail(
            "live OTEL interop failed\n"
            f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"
        )
    ts_glob = sorted(ARTIFACTS.glob("kazenai-live-*.json"))
    if not ts_glob:
        pytest.fail("live interop artifact not written")
    data = json.loads(ts_glob[-1].read_text(encoding="utf-8"))
    yield data


def test_otel_local_interop_verified_gate(_live_artifact):
    names = {r["scenario"]: r for r in _live_artifact.get("results") or []}
    assert names.get("OTEL_LOCAL_INTEROP_VERIFIED", {}).get("status") == "pass"
    assert names.get("shared_trace_id", {}).get("status") == "pass"
    assert names.get("schema_invalid_422", {}).get("status") in {"pass", "skip"}
