"""Canonical principal helpers for kazenai-core (FINAL_1 P1-1 / LF-05).

Prefers shared ``kazenai_contracts.principal`` when installed; otherwise uses the
in-tree mirror ``kazenai.auth._principal_impl``. No sibling-path ``sys.path``
injection (public installs must not require a private contracts checkout).
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
except ImportError:  # pragma: no cover - contracts optional for public wheel
    from kazenai.auth._principal_impl import (
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
