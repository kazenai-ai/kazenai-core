"""Package metadata and first-run install-contract checks for kazenai 1.1.1."""

from __future__ import annotations

import re
from pathlib import Path

import kazenai

_ROOT = Path(__file__).resolve().parents[1]
_EXPECTED = "1.1.1"
_OPENAI_BOUND = "openai>=1.40,<2"
_ANTHROPIC_BOUND = "anthropic>=0.39,<1"


def _pyproject_text() -> str:
    return (_ROOT / "pyproject.toml").read_text(encoding="utf-8")


def _optional_extra(name: str) -> list[str]:
    text = _pyproject_text()
    block = re.search(
        rf'(?ms)^{re.escape(name)}\s*=\s*\[(.*?)\]',
        text,
    )
    assert block is not None, f"missing optional-dependencies.{name}"
    return re.findall(r'"([^"]+)"', block.group(1))


def test_runtime_and_pyproject_version_are_1_1_1() -> None:
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', _pyproject_text())
    assert match is not None
    assert match.group(1) == _EXPECTED
    assert kazenai.__version__ == _EXPECTED


def test_optional_provider_extras_use_certified_bounds() -> None:
    assert _optional_extra("openai") == [_OPENAI_BOUND]
    assert _optional_extra("anthropic") == [_ANTHROPIC_BOUND]
    assert _optional_extra("providers") == [_OPENAI_BOUND, _ANTHROPIC_BOUND]
    # Providers remain available for the test/dev extra without becoming mandatory.
    dev = _optional_extra("dev")
    assert _OPENAI_BOUND in dev
    assert _ANTHROPIC_BOUND in dev


def test_active_first_run_docs_prefer_bounded_extras() -> None:
    for relative in ("README.md", "QUICKSTART.md"):
        text = (_ROOT / relative).read_text(encoding="utf-8")
        assert "kazenai[openai]==1.1.1" in text or "kazenai-finops[openai]==1.1.1" in text
        assert not re.search(
            r'pip install(?:\s+"?)kazenai(?:-finops)?(?!\[[^\]]+\])(?:"?==[0-9.]+)?\s+"?(?:openai|anthropic)\b',
            text,
        )


def test_supported_matrix_package_line_matches_release() -> None:
    matrix = (_ROOT / "docs/integrations/control-supported-matrix.md").read_text(
        encoding="utf-8"
    )
    assert "1.1.1 prepared" in matrix
