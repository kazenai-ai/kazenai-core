"""Core re-export of principal fixtures + JWT audience reuse."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from kazenai.auth import evaluate_fixture, reject_client_privileged_headers, PrincipalError

# Vendored corpus: standalone CI checks out only this public repo and cannot
# install the private kazenai-contracts wheel. When the wheel is installed, the
# vendored copy must match it byte-for-byte.
VENDORED_DIR = Path(__file__).resolve().parent / "fixtures" / "contracts_principal"
FIXTURES = [
    "valid_human_principal.json",
    "valid_service_act_as.json",
    "unauthorized_act_as_human.json",
    "forged_org_not_in_membership.json",
    "forged_workspace_not_authorized.json",
    "invalid_audience.json",
    "stale_revoked_membership.json",
    "client_privileged_headers_forbidden.json",
    "self_act_as_own_org.json",
    "self_act_as_own_org_foreign_workspace.json",
]


def test_vendored_corpus_matches_manifest_and_contracts_wheel() -> None:
    manifest = {}
    for line in (VENDORED_DIR / "CORPUS.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        manifest[name] = digest
    assert set(manifest) == set(FIXTURES)
    has_wheel = importlib.util.find_spec("kazenai_contracts") is not None
    for name, digest in manifest.items():
        data = (VENDORED_DIR / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, f"{name} differs from CORPUS.sha256"
        if has_wheel:
            from kazenai_contracts.resources import auth_fixture_text

            assert auth_fixture_text(name).encode() == data, (
                f"vendored {name} drifted from the kazenai-contracts wheel; re-vendor it"
            )


@pytest.mark.parametrize("name", FIXTURES)
def test_core_evaluates_contract_fixtures(name: str) -> None:
    fixture = json.loads((VENDORED_DIR / name).read_text(encoding="utf-8"))
    result = evaluate_fixture(fixture)
    assert result.get("ok") is True, result


def test_strip_privileged_on_browser_ingress() -> None:
    with pytest.raises(PrincipalError, match="privileged"):
        reject_client_privileged_headers(
            {"X-Kazen-Act-As-Org": "org-b", "X-Kazen-User-Scope": "victim"},
            ingress="browser_client",
        )
