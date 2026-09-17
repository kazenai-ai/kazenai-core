"""Core re-export of P1-1 principal fixtures + JWT audience reuse."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kazenai.auth import evaluate_fixture, reject_client_privileged_headers, PrincipalError

CONTRACTS = Path(__file__).resolve().parents[1].parent / "kazenai-contracts"
# Prefer sibling monorepo checkout; fall back one more level for nested layouts.
if not (CONTRACTS / "fixtures" / "auth").is_dir():
    CONTRACTS = Path(__file__).resolve().parents[2] / "kazenai-contracts"
FIXTURE_DIR = CONTRACTS / "fixtures" / "auth"


@pytest.mark.skipif(not FIXTURE_DIR.is_dir(), reason="kazenai-contracts fixtures not present")
@pytest.mark.parametrize(
    "name",
    [
        "valid_human_principal.json",
        "valid_service_act_as.json",
        "unauthorized_act_as_human.json",
        "forged_org_not_in_membership.json",
        "forged_workspace_not_authorized.json",
        "invalid_audience.json",
        "stale_revoked_membership.json",
        "client_privileged_headers_forbidden.json",
    ],
)
def test_core_evaluates_contract_fixtures(name: str) -> None:
    fixture = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    result = evaluate_fixture(fixture)
    assert result.get("ok") is True, result


def test_strip_privileged_on_browser_ingress() -> None:
    with pytest.raises(PrincipalError, match="privileged"):
        reject_client_privileged_headers(
            {"X-Kazen-Act-As-Org": "org-b", "X-Kazen-User-Scope": "victim"},
            ingress="browser_client",
        )
