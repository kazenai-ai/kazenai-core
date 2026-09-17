"""Hermetic test defaults for standalone CI (no host .env leakage)."""

from __future__ import annotations

import pytest

# Host/dev shells often export KAZENAI_* from a workspace .env; clear the ones
# that change default pricing behavior so unit tests match GitHub Actions.
_CLEAR_ENV = (
    "KAZENAI_UNKNOWN_MODEL_POLICY",
    "KAZENAI_DEPLOYMENT_MODE",
    "KAZENAI_ENV",
    "KAZENAI_PRICING_JSON",
)


@pytest.fixture(autouse=True)
def _hermetic_kazenai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _CLEAR_ENV:
        monkeypatch.delenv(key, raising=False)
