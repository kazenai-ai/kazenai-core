"""Defect regressions retained after repair."""

from __future__ import annotations

import time

import jwt
import pytest

from kazenai.auth.jwt import JwtVerifyConfig, JwtVerifyError, verify_hs256_token
from kazenai.schema import KazenEvent as CoreEvent


def test_p0_rejects_prefix_extended_supabase_issuer() -> None:
    secret = "p0-kazenai-jwt-secret-32b!!"
    trusted = "https://example.supabase.co/auth/v1"
    token = jwt.encode(
        {
            "sub": "user-1",
            "org_id": "org-1",
            "exp": int(time.time()) + 3600,
            "iss": f"{trusted}/evil",
            "aud": "authenticated",
        },
        secret,
        algorithm="HS256",
    )
    with pytest.raises(JwtVerifyError) as excinfo:
        verify_hs256_token(
            token,
            JwtVerifyConfig(
                secret=secret,
                supabase_issuer=trusted,
                enforce_iss_aud=True,
            ),
        )
    assert excinfo.value.category == "issuer"


def test_p0_tokens_17_survives_core_schema() -> None:
    raw = {
        "schema_version": "1.2",
        "ts_ms": 1_700_000_000_000,
        "event_id": "evt_p0_tokens17_core",
        "org_id": "org_p0",
        "workspace_id": "default",
        "project_id": "default",
        "surface": "kazenai-sdk",
        "agent_id": "agent_p0",
        "agent_role": "worker",
        "run_id": "run_p0",
        "step_id": "step_p0",
        "event_type": "model.call",
        "tokens": 17,
        "cost_usd": 0.0,
    }
    event = CoreEvent.model_validate(raw)
    assert event.tokens_used == 17
    assert event.tokens == 17
    dumped = event.model_dump()
    assert dumped.get("tokens_used") == 17
    assert dumped.get("tokens") == 17
