"""Shared HS256 JWT verification for KazenAI Python services."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import jwt


class JwtVerifyError(ValueError):
    """JWT failed verification."""


@dataclass(frozen=True)
class JwtVerifyConfig:
    secret: str
    kazen_issuer: str = "kazenai"
    kazen_audience: str = "authenticated"
    supabase_issuer: str = ""
    enforce_iss_aud: bool = True
    leeway_seconds: int = 0

    @classmethod
    def from_env(cls, *, secret: str | None = None, enforce_iss_aud: bool | None = None) -> "JwtVerifyConfig":
        resolved = (secret or jwt_secret_from_env()).strip()
        supabase_issuer = (
            os.getenv("SUPABASE_JWT_ISSUER", "").strip()
            or f"{os.getenv('SUPABASE_URL', '').rstrip('/')}/auth/v1".strip("/")
        )
        if enforce_iss_aud is None:
            mode = os.getenv("KAZENAI_DEPLOYMENT_MODE", os.getenv("KAZENAI_ENV", "development")).lower()
            enforce_iss_aud = mode in {"saas", "staging", "production", "prod"}
        return cls(
            secret=resolved,
            kazen_issuer=os.getenv("KAZENAI_JWT_ISSUER", "kazenai").strip() or "kazenai",
            kazen_audience=os.getenv("KAZENAI_JWT_AUDIENCE", "authenticated").strip() or "authenticated",
            supabase_issuer=supabase_issuer,
            enforce_iss_aud=bool(enforce_iss_aud),
        )


def jwt_secret_from_env() -> str:
    for key in (
        "KAZENAI_JWT_SECRET",
        "JWT_SECRET",
        "SUPABASE_JWT_SECRET",
        "KAZENAI_SUPABASE_JWT_SECRET",
    ):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return ""


def extract_org_id(payload: dict[str, Any]) -> str:
    app_metadata = payload.get("app_metadata") if isinstance(payload.get("app_metadata"), dict) else {}
    user_metadata = payload.get("user_metadata") if isinstance(payload.get("user_metadata"), dict) else {}
    return str(
        payload.get("org_id")
        or app_metadata.get("org_id")
        or user_metadata.get("org_id")
        or ""
    ).strip()


def _audience_matches(payload: dict[str, Any], expected: str) -> bool:
    aud = payload.get("aud")
    if aud is None:
        return False
    if isinstance(aud, list):
        return expected in [str(item) for item in aud]
    return str(aud) == expected


def _issuer_allowed(payload: dict[str, Any], config: JwtVerifyConfig) -> bool:
    iss = str(payload.get("iss") or "").strip()
    if not iss:
        return not config.enforce_iss_aud
    if iss == config.kazen_issuer:
        return _audience_matches(payload, config.kazen_audience)
    if config.supabase_issuer and (iss == config.supabase_issuer or iss.startswith(f"{config.supabase_issuer}/")):
        return True
    return not config.enforce_iss_aud


def verify_hs256_token(token: str, config: JwtVerifyConfig) -> dict[str, Any]:
    if not config.secret:
        raise JwtVerifyError("jwt secret not configured")
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise JwtVerifyError("invalid jwt header") from exc
    if str(header.get("alg") or "") != "HS256":
        raise JwtVerifyError("unsupported jwt alg")

    try:
        payload = jwt.decode(
            token,
            config.secret,
            algorithms=["HS256"],
            options={
                "require": ["exp", "sub"],
                "verify_aud": False,
            },
            leeway=config.leeway_seconds,
        )
    except jwt.PyJWTError as exc:
        raise JwtVerifyError(str(exc)) from exc

    if config.enforce_iss_aud and not _issuer_allowed(payload, config):
        raise JwtVerifyError("jwt issuer or audience mismatch")
    return dict(payload)
