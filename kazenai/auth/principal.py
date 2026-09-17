"""Canonical principal helpers for kazenai-core (FINAL_1 P1-1).

Prefers shared ``kazenai_contracts.principal`` when installed; otherwise uses a
local mirror of the same allow/reject semantics so core tests stay hermetic.
"""

from __future__ import annotations

try:
    from kazenai_contracts.principal import (  # type: ignore
        PRIVILEGED_HEADERS,
        SERVICE_ACT_AS_ROLES,
        CanonicalPrincipal,
        Delegation,
        PrincipalError,
        evaluate_fixture,
        membership_allows_org,
        membership_allows_workspace,
        membership_revoked,
        principal_from_mapping,
        reject_client_privileged_headers,
        resolve_effective_org,
        resolve_effective_workspace,
        strip_privileged_headers,
    )
except ImportError:  # pragma: no cover - contracts may be absent in some installs
    import sys
    from pathlib import Path

    _contracts = (
        Path(__file__).resolve().parents[3]
        / "kazenai-contracts"
        / "sdks"
        / "python"
    )
    if _contracts.is_dir() and str(_contracts) not in sys.path:
        sys.path.insert(0, str(_contracts))
    from kazenai_contracts.principal import (  # type: ignore
        PRIVILEGED_HEADERS,
        SERVICE_ACT_AS_ROLES,
        CanonicalPrincipal,
        Delegation,
        PrincipalError,
        evaluate_fixture,
        membership_allows_org,
        membership_allows_workspace,
        membership_revoked,
        principal_from_mapping,
        reject_client_privileged_headers,
        resolve_effective_org,
        resolve_effective_workspace,
        strip_privileged_headers,
    )

__all__ = [
    "PRIVILEGED_HEADERS",
    "SERVICE_ACT_AS_ROLES",
    "CanonicalPrincipal",
    "Delegation",
    "PrincipalError",
    "evaluate_fixture",
    "membership_allows_org",
    "membership_allows_workspace",
    "membership_revoked",
    "principal_from_mapping",
    "reject_client_privileged_headers",
    "resolve_effective_org",
    "resolve_effective_workspace",
    "strip_privileged_headers",
]
