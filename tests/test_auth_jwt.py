"""JWT verifier unit tests."""

from __future__ import annotations

import time

import jwt
import pytest

from kazenai.auth.jwt import JwtVerifyConfig, JwtVerifyError, extract_org_id, verify_hs256_token


def test_extract_org_id_from_metadata() -> None:
    payload = {
        "app_metadata": {"org_id": "org-meta"},
        "user_metadata": {"org_id": "org-user"},
    }
    assert extract_org_id({"org_id": "org-top"}) == "org-top"
    assert extract_org_id(payload) == "org-meta"


def test_kazen_issuer_requires_audience() -> None:
    secret = "shared-secret"
    token = jwt.encode(
        {
            "sub": "svc",
            "org_id": "org-1",
            "exp": int(time.time()) + 300,
            "iss": "kazenai",
            "aud": "wrong",
        },
        secret,
        algorithm="HS256",
    )
    with pytest.raises(JwtVerifyError):
        verify_hs256_token(token, JwtVerifyConfig(secret=secret, enforce_iss_aud=True))
