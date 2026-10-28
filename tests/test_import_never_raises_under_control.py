"""0.11: importing the SDK never fails; Control configuration errors surface on use.

Services (FinOps, Lens) import ``kazenai`` for JWT, signing and catalog helpers
under ``KAZENAI_PROFILE=control`` and never price a provider call. An import-time
``TokenCostEngine()`` made every such process crash without the customer-side
``KAZENAI_TIER_MISMATCH_MULTIPLIER``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def _clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("KAZENAI_")}
    env.update(extra)
    return env


def test_import_succeeds_under_control_without_the_multiplier():
    env = _clean_env(KAZENAI_PROFILE="control", KAZENAI_CONTROL_PROFILE="1")
    code = "import kazenai, kazenai.auth.jwt, kazenai.ingest_signature, kazenai.price_catalog; print('ok')"
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "ok"


def test_first_control_pricing_use_still_refuses_without_the_multiplier(monkeypatch):
    from kazenai.price_catalog import PricingConfigError
    from kazenai.spine import guard

    monkeypatch.delenv("KAZENAI_TIER_MISMATCH_MULTIPLIER", raising=False)
    monkeypatch.setenv("KAZENAI_CONTROL_PROFILE", "1")
    monkeypatch.setattr(guard, "_COST_ENGINE", guard._LazyCostEngine())
    with pytest.raises(PricingConfigError):
        guard._COST_ENGINE.cost_usd(model="gpt-4o-mini", input_tokens=10, output_tokens=0)
