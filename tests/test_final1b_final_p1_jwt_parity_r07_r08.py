"""R07/R08 — cross-language JWT parity (subject, leeway, iat, kid keyring, env keyrings)."""

from __future__ import annotations

import os
from unittest.mock import patch

import jwt
import pytest

from kazenai.auth.jwt import IssuerProfile, JwtVerifyConfig, JwtVerifyError, verify_hs256_token


def _audit_profile(*, leeway: int = 30) -> IssuerProfile:
    key = "audit-only-key-32-characters-long-123"
    return IssuerProfile(
        name="audit",
        issuer="kazenai",
        keys=(key,),
        audiences=frozenset({"authenticated"}),
        leeway_seconds=leeway,
    )


def test_r07_numeric_sub_rejects_malformed() -> None:
    key = "audit-only-key-32-characters-long-123"
    now = 1_700_000_000
    token = jwt.encode(
        {"sub": 123, "iss": "kazenai", "aud": "authenticated", "exp": now + 600},
        key,
        algorithm="HS256",
    )
    with pytest.raises(JwtVerifyError) as excinfo:
        verify_hs256_token(token, JwtVerifyConfig(profiles=(_audit_profile(),)), now=now)
    assert excinfo.value.category == "malformed"


def test_r07_within_leeway_accepts() -> None:
    key = "audit-only-key-32-characters-long-123"
    now = 1_700_000_000
    token = jwt.encode(
        {"sub": "audit-user", "iss": "kazenai", "aud": "authenticated", "exp": now - 5},
        key,
        algorithm="HS256",
    )
    payload = verify_hs256_token(token, JwtVerifyConfig(profiles=(_audit_profile(),)), now=now)
    assert payload["sub"] == "audit-user"


def test_numeric_jti_rejects_malformed_after_signature_verification() -> None:
    key = "audit-only-key-32-characters-long-123"
    now = 1_700_000_000
    claims = {
        "sub": "audit-user", "iss": "kazenai", "aud": "authenticated",
        "exp": now + 600, "jti": 123,
    }
    config = JwtVerifyConfig(profiles=(_audit_profile(),))
    token = jwt.encode(claims, key, algorithm="HS256")
    with pytest.raises(JwtVerifyError) as excinfo:
        verify_hs256_token(token, config, now=now)
    assert excinfo.value.category == "malformed"

    # Classifying a malformed claim must not bypass signature verification.
    token = jwt.encode(claims, "wrong-signing-key-32-characters-long", algorithm="HS256")
    with pytest.raises(JwtVerifyError) as excinfo:
        verify_hs256_token(token, config, now=now)
    assert excinfo.value.category == "signature"


def test_r07_future_iat_uses_controlled_clock() -> None:
    key = "audit-only-key-32-characters-long-123"
    now = 1_700_000_000
    token = jwt.encode(
        {
            "sub": "audit-user",
            "iss": "kazenai",
            "aud": "authenticated",
            "exp": now + 600,
            "iat": now + 300,
        },
        key,
        algorithm="HS256",
    )
    with pytest.raises(JwtVerifyError) as excinfo:
        verify_hs256_token(token, JwtVerifyConfig(profiles=(_audit_profile(),)), now=now)
    assert excinfo.value.category == "not_yet_valid"


def test_r08_unknown_kid_rejected_when_keyring_configured() -> None:
    key = "audit-only-key-32-characters-long-123"
    now = 1_700_000_000
    token = jwt.encode(
        {"sub": "audit-user", "iss": "kazenai", "aud": "authenticated", "exp": now + 600},
        key,
        algorithm="HS256",
        headers={"kid": "unknown-kid"},
    )
    profile = IssuerProfile(
        name="audit",
        issuer="kazenai",
        keys=(key,),
        kid_keys=(("primary", key),),
        audiences=frozenset({"authenticated"}),
    )
    with pytest.raises(JwtVerifyError) as excinfo:
        verify_hs256_token(token, JwtVerifyConfig(profiles=(profile,)), now=now)
    assert excinfo.value.category == "unavailable_key"


def test_r08_from_env_separates_supabase_keys() -> None:
    env = {
        "KAZENAI_JWT_SECRET": "kazen-only-secret-32-characters!!",
        "SUPABASE_JWT_SECRET": "supabase-only-secret-32-characters",
        "SUPABASE_URL": "https://example.supabase.co",
    }
    with patch.dict(os.environ, env, clear=True):
        config = JwtVerifyConfig.from_env()
    profiles = {p.name: p for p in config.resolved_profiles()}
    assert "kazen-only-secret-32-characters!!" in profiles["kazenai-hs256"].keys
    assert "supabase-only-secret-32-characters" in profiles["kazenai-hs256"].keys
    assert profiles["supabase-hs256-legacy"].keys == ("supabase-only-secret-32-characters",)
    assert "kazen-only-secret-32-characters!!" not in profiles["supabase-hs256-legacy"].keys


def test_explicit_supabase_issuer_does_not_require_environment(monkeypatch) -> None:
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_JWT_ISSUER", raising=False)
    key = "audit-only-key-32-characters-long-123"
    issuer = "https://audit-project.supabase.co/auth/v1"
    now = 1_700_000_000
    claims = {"sub": "audit-user", "iss": issuer, "aud": "authenticated", "exp": now + 600}
    config = JwtVerifyConfig(secret=key, supabase_issuer=issuer)
    token = jwt.encode(claims, key, algorithm="HS256")
    assert verify_hs256_token(token, config, now=now)["sub"] == "audit-user"

    # A token's own issuer cannot establish a trust profile.
    with pytest.raises(JwtVerifyError) as excinfo:
        verify_hs256_token(token, JwtVerifyConfig(secret=key), now=now)
    assert excinfo.value.category == "issuer"

    token = jwt.encode({**claims, "iss": issuer + "/evil"}, key, algorithm="HS256")
    with pytest.raises(JwtVerifyError) as excinfo:
        verify_hs256_token(token, config, now=now)
    assert excinfo.value.category == "issuer"
