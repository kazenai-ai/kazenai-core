"""F821 gate (P3-2): no undefined names anywhere in this repo.

This bug class produced real failures: the Brain MCP server crashed at launch
(missing asyncio import), the orchestrator MCP server silently skipped its
security overlay (missing rule-guard imports, NameError swallowed), and
routing/usage.py raised NameError on its malformed-response fallback path.
Star-import re-export modules blind pyflakes ("unable to detect undefined
names"); those warnings are excluded — converting the star imports to explicit
__all__ re-exports is tracked separately.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

pyflakes_api = pytest.importorskip("pyflakes.api")
from pyflakes.reporter import Reporter  # noqa: E402

SCAN_ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {
    ".venv", ".venv-test", ".venv-build", ".venv-bootstrap", ".venv-bench",
    "node_modules", "build", "dist", "__pycache__", ".git", ".mypy_cache",
}


def test_no_undefined_names() -> None:
    problems: list[str] = []
    for path in sorted(SCAN_ROOT.rglob("*.py")):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        out, err = io.StringIO(), io.StringIO()
        pyflakes_api.check(
            path.read_text(encoding="utf-8", errors="ignore"), str(path), Reporter(out, err)
        )
        for line in out.getvalue().splitlines():
            if "unable to detect undefined names" in line or "may be undefined" in line:
                continue
            if "undefined name" in line or "referenced before assignment" in line:
                problems.append(line)
    assert problems == [], "undefined names found:\n" + "\n".join(problems)
