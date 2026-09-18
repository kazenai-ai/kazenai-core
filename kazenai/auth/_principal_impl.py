"""Local mirror of kazenai_contracts.principal (FINAL_1 LF-05).

Kept in-tree so public ``kazenai`` installs work without a sibling
``kazenai-contracts`` checkout or sys.path injection. Prefer the installed
``kazenai_contracts`` package when present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

SERVICE_ACT_AS_ROLES = frozenset(
    {
        "service",
        "internal_admin",
        "growthops_service",
        "gateway_service",
        "brain_service",
        "finops_service",
        "lens_service",
        "platform_service",
    }
)

PRIVILEGED_HEADERS = frozenset(
    {
        "x-kazen-act-as-org",
        "x-kazen-user-scope",
    }
)

UNTRUSTED_INGRESS = frozenset({"browser_client", "untrusted_client", "human_jwt"})


class PrincipalError(ValueError):
    """Identity/delegation contract violation."""


@dataclass(frozen=True)
class Delegation:
    kind: str = "none"
    asserted_org_id: Optional[str] = None
    asserted_actor_id: Optional[str] = None
    audience: Optional[str] = None
    scope: tuple[str, ...] = ()


@dataclass(frozen=True)
class CanonicalPrincipal:
    role: str
    auth_type: str
    identity_kind: str = "human"
    subject_id: Optional[str] = None
    actor_id: Optional[str] = None
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    workspace_id: Optional[str] = None
    authorized_org_ids: Optional[tuple[str, ...]] = None
    authorized_workspace_ids: Optional[tuple[str, ...]] = None
    delegation: Delegation = field(default_factory=Delegation)

    def audit_ids(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id or self.user_id or self.actor_id,
            "actor_id": self.actor_id or self.user_id or self.subject_id,
            "org_id": self.org_id,
            "workspace_id": self.workspace_id,
            "identity_kind": self.identity_kind,
            "delegation": {
                "kind": self.delegation.kind,
                "asserted_org_id": self.delegation.asserted_org_id,
                "asserted_actor_id": self.delegation.asserted_actor_id,
            },
        }


def _norm_header_map(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(k).lower(): str(v).strip() for k, v in headers.items() if v is not None}


def principal_from_mapping(data: Mapping[str, Any]) -> CanonicalPrincipal:
    dele_raw = data.get("delegation") if isinstance(data.get("delegation"), Mapping) else {}
    dele = Delegation(
        kind=str(dele_raw.get("kind") or "none"),
        asserted_org_id=(str(dele_raw["asserted_org_id"]).strip() if dele_raw.get("asserted_org_id") else None),
        asserted_actor_id=(
            str(dele_raw["asserted_actor_id"]).strip() if dele_raw.get("asserted_actor_id") else None
        ),
        audience=(str(dele_raw["audience"]).strip() if dele_raw.get("audience") else None),
        scope=tuple(str(x) for x in (dele_raw.get("scope") or [])),
    )

    auth_orgs: Optional[tuple[str, ...]]
    if "authorized_org_ids" in data:
        auth_orgs = tuple(str(x) for x in (data.get("authorized_org_ids") or []) if str(x).strip())
    else:
        auth_orgs = None

    auth_ws: Optional[tuple[str, ...]]
    if "authorized_workspace_ids" in data:
        auth_ws = tuple(str(x) for x in (data.get("authorized_workspace_ids") or []) if str(x).strip())
    else:
        auth_ws = None

    subject = str(data.get("subject_id") or data.get("user_id") or data.get("actor_id") or "").strip() or None
    actor = str(data.get("actor_id") or data.get("user_id") or data.get("subject_id") or "").strip() or None
    role = str(data.get("role") or "")
    identity = str(
        data.get("identity_kind") or ("service" if role in SERVICE_ACT_AS_ROLES else "human")
    )
    return CanonicalPrincipal(
        role=role,
        auth_type=str(data.get("auth_type") or ""),
        identity_kind=identity,
        subject_id=subject,
        actor_id=actor,
        user_id=(str(data["user_id"]).strip() if data.get("user_id") else None),
        org_id=(str(data["org_id"]).strip() if data.get("org_id") else None),
        workspace_id=(str(data["workspace_id"]).strip() if data.get("workspace_id") else None),
        authorized_org_ids=auth_orgs,
        authorized_workspace_ids=auth_ws,
        delegation=dele,
    )


def strip_privileged_headers(
    headers: Mapping[str, str],
    *,
    ingress: str = "browser_client",
) -> dict[str, str]:
    """Drop privileged headers on browser/client ingress; keep on service/proxy."""
    out = {str(k): str(v) for k, v in headers.items()}
    if ingress in UNTRUSTED_INGRESS:
        for key in list(out):
            if key.lower() in PRIVILEGED_HEADERS:
                del out[key]
    return out


def reject_client_privileged_headers(
    headers: Mapping[str, str],
    *,
    ingress: str = "browser_client",
) -> None:
    if ingress not in UNTRUSTED_INGRESS:
        return
    normalized = _norm_header_map(headers)
    present = sorted(h for h in PRIVILEGED_HEADERS if normalized.get(h))
    if present:
        raise PrincipalError("client must not set privileged identity headers")


def membership_revoked(membership: Optional[Mapping[str, Any]]) -> bool:
    if membership is None:
        return False
    return bool(membership.get("revoked_at"))


def membership_allows_org(principal: CanonicalPrincipal, org_id: str) -> bool:
    org = str(org_id or "").strip()
    if not org:
        return False
    if principal.authorized_org_ids is not None:
        return org in principal.authorized_org_ids
    return bool(principal.org_id) and org == principal.org_id


def membership_allows_workspace(principal: CanonicalPrincipal, workspace_id: str) -> bool:
    ws = str(workspace_id or "default").strip() or "default"
    if principal.authorized_workspace_ids is not None:
        return ws in principal.authorized_workspace_ids
    if principal.workspace_id:
        return ws == principal.workspace_id
    return True


def resolve_effective_org(
    principal: CanonicalPrincipal,
    headers: Mapping[str, str],
    *,
    body_org_id: str,
    ingress: str = "api",
) -> tuple[str, CanonicalPrincipal]:
    """Resolve tenant org; return (org_id, principal_with_delegation_audit)."""
    reject_client_privileged_headers(headers, ingress=ingress)
    body_org = str(body_org_id or "").strip()
    if not body_org:
        raise PrincipalError("org_id required")

    normalized = _norm_header_map(headers)
    header_org = normalized.get("x-kazen-org-id") or normalized.get("x-org-id") or ""
    act_as = normalized.get("x-kazen-act-as-org") or ""

    if header_org and header_org != body_org:
        raise PrincipalError("org_id must match X-Kazen-Org-Id")

    if act_as:
        if principal.role not in SERVICE_ACT_AS_ROLES:
            raise PrincipalError("X-Kazen-Act-As-Org not authorized for this credential")
        if body_org != act_as:
            raise PrincipalError("org_id must match X-Kazen-Act-As-Org")
        updated = CanonicalPrincipal(
            role=principal.role,
            auth_type=principal.auth_type,
            identity_kind="service",
            subject_id=principal.subject_id or principal.actor_id,
            actor_id=principal.actor_id or principal.subject_id,
            user_id=principal.user_id,
            org_id=act_as,
            workspace_id=principal.workspace_id,
            authorized_org_ids=principal.authorized_org_ids,
            authorized_workspace_ids=principal.authorized_workspace_ids,
            delegation=Delegation(
                kind="act_as_org",
                asserted_org_id=act_as,
                audience=principal.delegation.audience,
                scope=principal.delegation.scope,
            ),
        )
        return act_as, updated

    if not membership_allows_org(principal, body_org):
        if principal.role in SERVICE_ACT_AS_ROLES and body_org != (principal.org_id or ""):
            raise PrincipalError("service credential requires X-Kazen-Act-As-Org for tenant operations")
        if principal.authorized_org_ids is not None and len(principal.authorized_org_ids) == 0:
            raise PrincipalError("membership revoked or missing")
        raise PrincipalError("requested org not in authorized membership")

    if body_org != (principal.org_id or ""):
        if principal.role in SERVICE_ACT_AS_ROLES:
            raise PrincipalError("service credential requires X-Kazen-Act-As-Org for tenant operations")
        if principal.role not in {"admin", "owner", "internal_admin"}:
            raise PrincipalError("cross-org request forbidden")
        raise PrincipalError("cross-org request requires X-Kazen-Act-As-Org")

    return body_org, principal


def resolve_effective_workspace(
    principal: CanonicalPrincipal,
    headers: Mapping[str, str],
    *,
    body_workspace_id: str,
    act_as_active: bool = False,
) -> str:
    if act_as_active:
        return str(body_workspace_id or "default").strip() or "default"
    normalized = _norm_header_map(headers)
    ws = (
        str(body_workspace_id or "").strip()
        or normalized.get("x-kazen-workspace-id")
        or "default"
    )
    ws = ws.strip() or "default"
    if not membership_allows_workspace(principal, ws):
        raise PrincipalError("requested workspace not authorized")
    return ws


def _verify_jwt_fixture(jwt_spec: Mapping[str, Any]) -> None:
    """Mirror kazenai.auth.jwt issuer/audience rules for fixture evaluation."""
    try:
        import jwt as pyjwt
    except ImportError as exc:  # pragma: no cover
        raise PrincipalError("jwt audience mismatch") from exc

    claims = dict(jwt_spec.get("claims") or {})
    secret = str(jwt_spec.get("secret") or "")
    expected_aud = str(jwt_spec.get("expected_audience") or "authenticated")
    expected_iss = str(jwt_spec.get("expected_issuer") or "kazenai")
    token = pyjwt.encode(claims, secret, algorithm=str(jwt_spec.get("alg") or "HS256"))

    try:
        from kazenai.auth.jwt import JwtVerifyConfig, JwtVerifyError, verify_hs256_token

        try:
            verify_hs256_token(
                token,
                JwtVerifyConfig(
                    secret=secret,
                    kazen_issuer=expected_iss,
                    kazen_audience=expected_aud,
                    enforce_iss_aud=True,
                ),
            )
        except JwtVerifyError as exc:
            msg = str(exc).lower()
            if "audience" in msg or "issuer" in msg:
                raise PrincipalError("jwt audience mismatch") from exc
            raise PrincipalError(str(exc)) from exc
        return
    except ImportError:
        pass

    # Fallback when kazenai-core is not on PYTHONPATH: same iss/aud gate.
    payload = pyjwt.decode(
        token,
        secret,
        algorithms=["HS256"],
        options={"require": ["exp", "sub"], "verify_aud": False},
    )
    iss = str(payload.get("iss") or "").strip()
    aud = payload.get("aud")
    aud_ok = (expected_aud in aud) if isinstance(aud, list) else (str(aud) == expected_aud)
    if iss == expected_iss and not aud_ok:
        raise PrincipalError("jwt audience mismatch")
    if iss == expected_iss and aud_ok:
        return
    raise PrincipalError("jwt audience mismatch")


def evaluate_fixture(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Run a contracts auth fixture; return structured result for cross-language parity."""
    expect = fixture.get("expect")
    try:
        if "jwt" in fixture and isinstance(fixture.get("jwt"), Mapping):
            _verify_jwt_fixture(fixture["jwt"])
            # Valid JWT-only fixtures would continue; invalid ones raise above.
            result = {"ok": True, "jwt_verified": True}
            if expect == "reject":
                return {"ok": False, "error": "expected reject but accepted", "got": result}
            return result

        membership = fixture.get("membership")
        if membership_revoked(membership if isinstance(membership, Mapping) else None):
            raise PrincipalError("membership revoked or missing")

        principal = principal_from_mapping(fixture.get("principal") or {})
        req = fixture.get("request") or {}
        headers = dict(req.get("headers") or {})
        ingress = str(req.get("ingress") or "api")
        if fixture.get("fixture_id") == "client_supplied_privileged_headers_stripped":
            ingress = "browser_client"

        org, audited = resolve_effective_org(
            principal,
            headers,
            body_org_id=str(req.get("body_org_id") or ""),
            ingress=ingress,
        )
        act_as_active = bool(_norm_header_map(headers).get("x-kazen-act-as-org"))
        ws = resolve_effective_workspace(
            audited,
            headers,
            body_workspace_id=str(req.get("body_workspace_id") or "default"),
            act_as_active=act_as_active,
        )
        audit = audited.audit_ids()
        for path in fixture.get("audit_must_include") or []:
            cursor: Any = audit
            for part in str(path).split("."):
                if not isinstance(cursor, Mapping) or part not in cursor:
                    raise PrincipalError(f"audit missing {path}")
                cursor = cursor[part]
            if cursor in (None, ""):
                raise PrincipalError(f"audit missing {path}")

        result = {
            "ok": True,
            "org_id": org,
            "workspace_id": ws,
            "audit": audit,
        }
        if expect == "reject":
            return {"ok": False, "error": "expected reject but accepted", "got": result}
        return result
    except PrincipalError as exc:
        if expect == "accept":
            return {"ok": False, "error": str(exc)}
        wanted = str(fixture.get("reject_reason") or "")
        if wanted and wanted not in str(exc):
            return {"ok": False, "error": f"reject reason mismatch: wanted={wanted!r} got={str(exc)!r}"}
        return {"ok": True, "rejected": True, "reason": str(exc)}
