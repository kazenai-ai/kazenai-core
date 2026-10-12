#!/usr/bin/env python3
"""U13/F08: coverage gate for ``kazenai._async_provider`` via real async fault suites.

Measures line coverage while executing the maintained async/provider cancellation
and settlement suites — not a vacuous import-only probe.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT.parent / "artifacts" / "final1b-final-reaudit-fix-20261011"
# Meaningful floor once real suites execute the module. Raise as coverage grows.
FAIL_UNDER = 35.0

COVERAGERC = dedent(
    """\
    [run]
    include = */kazenai/_async_provider.py
    omit =
    branch = False
    parallel = False

    [report]
    include = */kazenai/_async_provider.py
    show_missing = True
    skip_empty = False
    """
)

# Prefer suites that exercise async provider / cancel paths when present.
TEST_TARGETS = [
    "tests/test_final1b_final_p5_async_providers.py",
    "tests/test_final1b_final_p5_remaining.py",
    "tests/test_final1b_final_eight_cell_cancel_matrix.py",
]


def _existing_targets() -> list[str]:
    found: list[str] = []
    for rel in TEST_TARGETS:
        path = ROOT / rel
        if path.exists():
            found.append(str(path))
    if found:
        return found
    # Fallback: any test file mentioning _async_provider.
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if "_async_provider" in text or "patch_openai_async" in text:
            found.append(str(path))
    return found


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    data_file = ARTIFACTS / ".coverage_async_provider"
    if data_file.exists():
        data_file.unlink()
    rc_path = ARTIFACTS / "async_provider.coveragerc"
    rc_path.write_text(COVERAGERC)
    targets = _existing_targets()
    if not targets:
        out = {
            "recorded_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "pass": False,
            "error": "no async provider test targets found",
            "fail_under": FAIL_UNDER,
        }
        (ARTIFACTS / "async-provider-coverage-gate.json").write_text(
            json.dumps(out, indent=2) + "\n"
        )
        print(json.dumps(out, indent=2))
        return 1

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    py = str(ROOT / ".venv" / "bin" / "python")
    if not Path(py).exists():
        py = sys.executable

    run = subprocess.run(
        [
            py,
            "-m",
            "coverage",
            "run",
            f"--rcfile={rc_path}",
            f"--data-file={data_file}",
            "-m",
            "pytest",
            "-q",
            "--tb=no",
            "--no-cov",
            *targets,
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    report = subprocess.run(
        [
            py,
            "-m",
            "coverage",
            "report",
            f"--rcfile={rc_path}",
            f"--data-file={data_file}",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )

    pct = 0.0
    for line in (report.stdout or "").splitlines():
        if "_async_provider.py" in line:
            for p in line.split():
                if p.endswith("%"):
                    try:
                        pct = float(p.rstrip("%"))
                    except ValueError:
                        pass
            break

    out = {
        "recorded_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "module": "kazenai._async_provider",
        "module_coverage_pct": pct,
        "fail_under": FAIL_UNDER,
        "coverage_pct": pct,
        "pytest_returncode": run.returncode,
        "targets": targets,
        "stderr": ((run.stderr or "") + (report.stderr or ""))[-1200:],
        "report": report.stdout,
        "coverage_returncode": report.returncode,
        "pass": (
            run.returncode == 0
            and report.returncode == 0
            and pct >= FAIL_UNDER
        ),
        "note": (
            "F08: executes real async/provider suites under coverage; "
            "import-only probes are not accepted as regression protection."
        ),
    }
    path = ARTIFACTS / "async-provider-coverage-gate.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "report"}, indent=2))
    print(report.stdout)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
