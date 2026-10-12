"""Shared JWT verification for KazenAI Python services.

Implements exact issuer profiles (no prefix matching), profile-bound HS256
keyrings, and stable error categories. Direct asymmetric Supabase JWKS
verification is unsupported in this release.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import jwt


ERROR_CATEGORIES = frozenset(
    {
        "malformed",
        "signature",
        "algorithm",
        "issuer",
        "audience",
        "expired",
        "not_yet_valid",
        "missing_claim",
        "unavailable_key",
    }
)


class JwtVerifyError(ValueError):
    """JWT failed verification with a stable public error category."""

    def __init__(self, category: str, message: str | None = None) -> None:
        cat = str(category or "malformed").strip() or "malformed"
        if cat not in ERROR_CATEGORIES:
            cat = "malformed"
        self.category = cat
        # Public message must never include the token or key material.
        super().__init__(message or cat)


@dataclass(frozen=True)
class IssuerProfile:
    name: str
    issuer: str
    algorithms: tuple[str, ...] = ("HS256",)
    audiences: frozenset[str] = frozenset({"authenticated"})
    identity_type: str = "kazen_session"
    keys: tuple[str, ...] = ()
    # Optional HS256 rotation map (kid -> secret). When non-empty, unknown kid is rejected.
    kid_keys: tuple[tuple[str, str], ...] = ()
    require_sub: bool = True
    require_exp: bool = True
    validate_nbf: bool = True
    validate_iat: bool = True
    leeway_seconds: int = 0
    allow_missing_kid: bool = True


@dataclass(frozen=True)
class JwtVerifyConfig:
    """Multi-profile verification config.

    ``secret`` / ``kazen_issuer`` / ``kazen_audience`` / ``supabase_issuer`` /
    ``enforce_iss_aud`` remain as compatibility shorthands that expand into
    exact issuer profiles.
    """

    profiles: tuple[IssuerProfile, ...] = ()
    # Compatibility fields (prefer building profiles explicitly in new code).
    secret: str = ""
    kazen_issuer: str = "kazenai"
    kazen_audience: str = "authenticated"
    supabase_issuer: str = ""
    enforce_iss_aud: bool = True
    leeway_seconds: int = 0

    @classmethod
    def from_env(
        cls,
        *,
        secret: str | None = None,
        enforce_iss_aud: bool | None = None,
    ) -> "JwtVerifyConfig":
        mode = os.getenv("KAZENAI_DEPLOYMENT_MODE", os.getenv("KAZENAI_ENV", "development")).lower()
        if enforce_iss_aud is None:
            # Profiles are always enforced when configured. The legacy
            # development relaxation required an explicit opt-in.
            relax = os.getenv("KAZENAI_JWT_RELAX_ISSUER", "").strip().lower() in {
                "1",
                "true",
                "yes",
            }
            enforce_iss_aud = not relax
        kazen_issuer = os.getenv("KAZENAI_JWT_ISSUER", "kazenai").strip() or "kazenai"
        kazen_audience = os.getenv("KAZENAI_JWT_AUDIENCE", "authenticated").strip() or "authenticated"
        supabase_issuer = _supabase_issuer_from_env()
        leeway = int(os.getenv("KAZENAI_JWT_LEEWAY_SECONDS", "0") or "0")
        leeway = max(0, min(leeway, 300))

        # Profile-bound keyrings — do not cross-try unrelated secrets across issuers.
        kazen_keys = _unique_nonempty(
            [
                secret,
                os.getenv("KAZENAI_JWT_SECRET", ""),
                # Platform mint order prefers SUPABASE_JWT_SECRET for iss=kazenai cookies.
                os.getenv("SUPABASE_JWT_SECRET", ""),
                os.getenv("KAZENAI_PLATFORM_JWT_SECRET", ""),
                os.getenv("JWT_SECRET", ""),
            ]
        )
        supabase_keys = _unique_nonempty(
            [
                os.getenv("SUPABASE_JWT_SECRET", ""),
                os.getenv("KAZENAI_SUPABASE_JWT_SECRET", ""),
            ]
        )
        # If caller passed an explicit secret only, keep it on kazen profile.
        if secret and secret.strip() and not kazen_keys:
            kazen_keys = (secret.strip(),)

        profiles: list[IssuerProfile] = []
        if kazen_keys:
            profiles.append(
                IssuerProfile(
                    name="kazenai-hs256",
                    issuer=kazen_issuer,
                    audiences=frozenset({kazen_audience}),
                    identity_type="kazen_session",
                    keys=kazen_keys,
                    leeway_seconds=leeway,
                )
            )
        if _supabase_issuer_configured(supabase_issuer) and supabase_keys:
            supabase_aud = (
                os.getenv("SUPABASE_JWT_AUDIENCE", "authenticated").strip() or "authenticated"
            )
            profiles.append(
                IssuerProfile(
                    name="supabase-hs256-legacy",
                    issuer=supabase_issuer,
                    audiences=frozenset({supabase_aud}),
                    identity_type="supabase_hs256_legacy",
                    keys=supabase_keys,
                    leeway_seconds=leeway,
                )
            )

        # Compatibility single-secret constructor used by older unit tests.
        primary = (secret or "").strip() or (kazen_keys[0] if kazen_keys else "")
        return cls(
            profiles=tuple(profiles),
            secret=primary,
            kazen_issuer=kazen_issuer,
            kazen_audience=kazen_audience,
            supabase_issuer=supabase_issuer,
            enforce_iss_aud=bool(enforce_iss_aud),
            leeway_seconds=leeway,
        )

    def resolved_profiles(self) -> tuple[IssuerProfile, ...]:
        if self.profiles:
            return self.profiles
        # Expand legacy single-secret config into exact profiles.
        profiles: list[IssuerProfile] = []
        keys = _unique_nonempty([self.secret])
        if keys:
            profiles.append(
                IssuerProfile(
                    name="kazenai-hs256",
                    issuer=self.kazen_issuer or "kazenai",
                    audiences=frozenset({self.kazen_audience or "authenticated"}),
                    identity_type="kazen_session",
                    keys=keys,
                    leeway_seconds=self.leeway_seconds,
                )
            )
        if _supabase_issuer_configured(self.supabase_issuer) and keys:
            profiles.append(
                IssuerProfile(
                    name="supabase-hs256-legacy",
                    issuer=self.supabase_issuer,
                    audiences=frozenset({self.kazen_audience or "authenticated"}),
                    identity_type="supabase_hs256_legacy",
                    keys=keys,
                    leeway_seconds=self.leeway_seconds,
                )
            )
        return tuple(profiles)


def _supabase_issuer_from_env() -> str:
    explicit = os.getenv("SUPABASE_JWT_ISSUER", "").strip()
    if explicit:
        return explicit
    base = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    if not base:
        return ""
    return f"{base}/auth/v1"


def _supabase_issuer_configured(issuer: str) -> bool:
    # The issuer is trusted configuration, either explicitly supplied or already
    # resolved by _supabase_issuer_from_env. Do not consult ambient state again.
    return bool(str(issuer or "").strip())


def _unique_nonempty(values: Iterable[Optional[str]]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.append(item)
    return tuple(seen)


def jwt_secret_from_env() -> str:
    for key in (
        "KAZENAI_JWT_SECRET",
        "SUPABASE_JWT_SECRET",
        "KAZENAI_SUPABASE_JWT_SECRET",
        "KAZENAI_PLATFORM_JWT_SECRET",
        "JWT_SECRET",
    ):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return ""


def extract_org_id(payload: Mapping[str, Any]) -> str:
    """Trusted org id from top-level or app_metadata only (never user_metadata)."""
    app_metadata = payload.get("app_metadata") if isinstance(payload.get("app_metadata"), dict) else {}
    return str(payload.get("org_id") or app_metadata.get("org_id") or "").strip()


def extract_role(payload: Mapping[str, Any], *, default: str = "member") -> str:
    """Trusted role from top-level or app_metadata only (never user_metadata)."""
    app_metadata = payload.get("app_metadata") if isinstance(payload.get("app_metadata"), dict) else {}
    return str(payload.get("role") or app_metadata.get("role") or default).strip() or default


def _as_numeric_date(value: Any, claim: str) -> int:
    """Shared NumericDate coercion with TypeScript ``asNumericDate``."""
    if isinstance(value, bool) or isinstance(value, (list, tuple, dict)):
        raise JwtVerifyError("malformed", f"invalid {claim} claim")
    if isinstance(value, str) and not value.strip():
        raise JwtVerifyError("malformed", f"invalid {claim} claim")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise JwtVerifyError("malformed", f"invalid {claim} claim") from exc


def _assert_subject_claim(payload: Mapping[str, Any], *, require: bool) -> None:
    sub = payload.get("sub")
    if sub is None:
        if require:
            raise JwtVerifyError("missing_claim", "missing subject")
        return
    if not isinstance(sub, str):
        raise JwtVerifyError("malformed", "invalid subject claim")
    if require and not sub.strip():
        raise JwtVerifyError("missing_claim", "missing subject")


def _kid_keyring(profile: IssuerProfile) -> dict[str, str]:
    return dict(profile.kid_keys)


def _hs256_verification_keys(profile: IssuerProfile, header: Mapping[str, Any]) -> tuple[str, ...]:
    ring = _kid_keyring(profile)
    kid_raw = header.get("kid")
    if kid_raw is not None and not isinstance(kid_raw, str):
        raise JwtVerifyError("malformed", "invalid kid header")
    if ring:
        if kid_raw is None or not kid_raw.strip():
            if profile.allow_missing_kid and profile.keys:
                return profile.keys
            raise JwtVerifyError("missing_claim", "missing kid")
        key = ring.get(kid_raw)
        if key is None:
            raise JwtVerifyError("unavailable_key", "unknown signing key")
        return (key,)
    return profile.keys


def _audience_matches(payload: Mapping[str, Any], allowed: frozenset[str]) -> bool:
    aud = payload.get("aud")
    if aud is None:
        return False
    if isinstance(aud, (list, tuple, set)):
        actual = {str(item) for item in aud if str(item)}
    else:
        actual = {str(aud)} if str(aud) else set()
    return bool(actual & set(allowed))


def _unverified_payload(token: str) -> dict[str, Any]:
    try:
        return dict(jwt.decode(token, options={"verify_signature": False}))
    except jwt.PyJWTError as exc:
        raise JwtVerifyError("malformed", "invalid jwt payload") from exc


def _map_pyjwt_error(exc: Exception) -> JwtVerifyError:
    if isinstance(exc, jwt.ExpiredSignatureError):
        return JwtVerifyError("expired", "token expired")
    if isinstance(exc, jwt.ImmatureSignatureError):
        return JwtVerifyError("not_yet_valid", "token not yet valid")
    invalid_subject = getattr(jwt, "InvalidSubjectError", None)
    if invalid_subject is not None and isinstance(exc, invalid_subject):
        return JwtVerifyError("malformed", "invalid subject claim")
    # PyJWT >= 2.10 validates jti during decode, before our claim checks.
    # Older supported PyJWT versions leave it to the shared policy below.
    invalid_jti = getattr(jwt.exceptions, "InvalidJTIError", None)
    if invalid_jti is not None and isinstance(exc, invalid_jti):
        return JwtVerifyError("malformed", "invalid jti claim")
    if isinstance(exc, jwt.InvalidSignatureError):
        return JwtVerifyError("signature", "invalid signature")
    if isinstance(exc, jwt.MissingRequiredClaimError):
        return JwtVerifyError("missing_claim", "missing required claim")
    if isinstance(exc, jwt.InvalidAudienceError):
        return JwtVerifyError("audience", "audience mismatch")
    if isinstance(exc, jwt.InvalidIssuerError):
        return JwtVerifyError("issuer", "issuer mismatch")
    if isinstance(exc, jwt.DecodeError):
        return JwtVerifyError("malformed", "invalid jwt")
    return JwtVerifyError("signature", "jwt verification failed")


def verify_hs256_token(
    token: str,
    config: JwtVerifyConfig,
    *,
    now: int | None = None,
) -> dict[str, Any]:
    """Verify a token against exact issuer profiles.

    ``now`` is a controlled unix timestamp for fixture evaluation.
    """
    if not isinstance(token, str) or token.count(".") != 2:
        raise JwtVerifyError("malformed", "invalid jwt")

    profiles = config.resolved_profiles()
    if not profiles:
        raise JwtVerifyError("unavailable_key", "jwt secret not configured")

    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise JwtVerifyError("malformed", "invalid jwt header") from exc

    alg = str(header.get("alg") or "")
    if alg == "none" or not alg:
        raise JwtVerifyError("algorithm", "unsupported jwt alg")
    allowed_algs = {a for p in profiles for a in p.algorithms}
    if alg not in allowed_algs:
        raise JwtVerifyError("algorithm", "unsupported jwt alg")

    # Unknown critical headers → malformed (parity with the TypeScript verifier).
    crit = header.get("crit")
    if crit is not None:
        if not isinstance(crit, (list, tuple)) or any(not isinstance(c, str) for c in crit):
            raise JwtVerifyError("malformed", "invalid crit header")
        understood = {"alg", "typ", "kid"}
        for name in crit:
            if name not in understood or name not in header:
                raise JwtVerifyError("malformed", f"unknown critical header: {name}")

    if header.get("kid") is not None and not isinstance(header.get("kid"), str):
        raise JwtVerifyError("malformed", "invalid kid header")

    try:
        unverified = _unverified_payload(token)
    except JwtVerifyError:
        raise

    iss = str(unverified.get("iss") or "").strip()
    if not iss:
        raise JwtVerifyError("missing_claim", "missing issuer")

    profile = next((p for p in profiles if p.issuer == iss), None)
    if profile is None:
        raise JwtVerifyError("issuer", "jwt issuer mismatch")

    if alg not in profile.algorithms:
        raise JwtVerifyError("algorithm", "unsupported jwt alg")

    try:
        verification_keys = _hs256_verification_keys(profile, header)
    except JwtVerifyError:
        raise
    if not verification_keys:
        raise JwtVerifyError("unavailable_key", "jwt secret not configured")

    clock = int(now if now is not None else time.time())
    last_error: JwtVerifyError | None = None
    payload: dict[str, Any] | None = None

    for key in verification_keys:
        try:
            payload = _decode_with_clock(
                token,
                key=key,
                algorithms=list(profile.algorithms),
                leeway=profile.leeway_seconds,
                clock=clock,
                require_exp=profile.require_exp,
                require_sub=profile.require_sub,
                validate_nbf=profile.validate_nbf,
                validate_iat=profile.validate_iat,
            )
            break
        except JwtVerifyError as exc:
            last_error = exc
            # Wrong key → try next key in this profile's ring only.
            if exc.category == "signature":
                continue
            raise

    if payload is None:
        raise last_error or JwtVerifyError("signature", "invalid signature")

    # Exact issuer re-check after signature verification.
    if str(payload.get("iss") or "").strip() != profile.issuer:
        raise JwtVerifyError("issuer", "jwt issuer mismatch")

    if not _audience_matches(payload, profile.audiences):
        raise JwtVerifyError("audience", "audience mismatch")

    _assert_subject_claim(payload, require=profile.require_sub)

    return dict(payload)


def _decode_with_clock(
    token: str,
    *,
    key: str,
    algorithms: Sequence[str],
    leeway: int,
    clock: int,
    require_exp: bool,
    require_sub: bool,
    validate_nbf: bool,
    validate_iat: bool,
) -> dict[str, Any]:
    """Decode and verify signature; apply exp/nbf/iat against the controlled clock."""
    required = []
    if require_sub:
        required.append("sub")
    try:
        options = {
            "require": required,
            "verify_aud": False,
            "verify_iss": False,
            # Validate temporal claims ourselves so fixture clocks are reliable
            # across PyJWT versions that no longer read time.time().
            "verify_exp": False,
            "verify_nbf": False,
            "verify_iat": False,
            "verify_sub": False,
        }
        payload = dict(
            jwt.decode(
                token,
                key,
                algorithms=list(algorithms),
                options=options,
                leeway=0,
            )
        )
    except jwt.PyJWTError as exc:
        raise _map_pyjwt_error(exc) from exc

    if require_exp:
        if payload.get("exp") is None:
            raise JwtVerifyError("missing_claim", "missing required claim")
        exp = _as_numeric_date(payload["exp"], "exp")
        if exp <= (clock - int(leeway)):
            raise JwtVerifyError("expired", "token expired")

    if validate_nbf and payload.get("nbf") is not None:
        nbf = _as_numeric_date(payload["nbf"], "nbf")
        if nbf > (clock + int(leeway)):
            raise JwtVerifyError("not_yet_valid", "token not yet valid")

    if validate_iat and payload.get("iat") is not None:
        iat = _as_numeric_date(payload["iat"], "iat")
        if iat > (clock + int(leeway)):
            raise JwtVerifyError("not_yet_valid", "token not yet valid")

    # Optional jti must be a string when the claim is present (incl. null).
    if "jti" in payload and not isinstance(payload.get("jti"), str):
        raise JwtVerifyError("malformed", "invalid jti claim")

    _assert_subject_claim(payload, require=require_sub)

    return payload


# Back-compat alias used by older imports/tests.
verify_token = verify_hs256_token
