"""Shared authentication helpers."""

from kazenai.auth.jwt import (
    JwtVerifyConfig,
    JwtVerifyError,
    extract_org_id,
    jwt_secret_from_env,
    verify_hs256_token,
)
from kazenai.auth.principal import (
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
    "JwtVerifyConfig",
    "JwtVerifyError",
    "extract_org_id",
    "jwt_secret_from_env",
    "verify_hs256_token",
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
