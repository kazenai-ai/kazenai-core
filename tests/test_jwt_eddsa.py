"""Platform-signed Ed25519 tokens: services verify with public keys only."""

from __future__ import annotations

import json
import time

import jwt as pyjwt
import pytest

cryptography = pytest.importorskip("cryptography")
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from kazenai.auth.jwt import (  # noqa: E402
    JwtVerifyError,
    load_public_keys,
    platform_token_profile,
    sign_eddsa_token,
    verify_eddsa_token,
)


def _keypair():
    key = Ed25519PrivateKey.generate()
    private_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_pem, public_pem


PRIVATE, PUBLIC = _keypair()
OTHER_PRIVATE, _OTHER_PUBLIC = _keypair()
NOW = 1_800_000_000


def _claims(**overrides):
    base = {
        "iss": "kazenai-platform",
        "aud": "kazenai-finops",
        "sub": "user:u1",
        "token_use": "delegation",
        "iat": NOW,
        "exp": NOW + 60,
        "jti": "dlg_1",
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


def _profile(**kw):
    return platform_token_profile(
        audience=kw.pop("audience", "kazenai-finops"),
        token_use=kw.pop("token_use", "delegation"),
        max_ttl_seconds=kw.pop("max_ttl_seconds", 120),
        public_keys=kw.pop("public_keys", (("k1", PUBLIC),)),
    )


def _verify(token, **kw):
    return verify_eddsa_token(token, _profile(**kw), now=NOW + 5)


def test_valid_delegation_token():
    token = sign_eddsa_token(_claims(), private_key_pem=PRIVATE, kid="k1")
    payload = _verify(token)
    assert payload["sub"] == "user:u1"


@pytest.mark.parametrize(
    "claims,category",
    [
        (_claims(iss="kazenai"), "issuer"),
        (_claims(aud="kazenai-lens"), "audience"),
        (_claims(token_use="session"), "token_use"),
        (_claims(token_use=None), "token_use"),
        (_claims(exp=NOW - 10), "expired"),
        (_claims(nbf=NOW + 600), "not_yet_valid"),
        (_claims(exp=NOW + 3600), "expired"),  # lifetime above the profile maximum
        (_claims(sub=None), "missing_claim"),
        (_claims(jti=123), "malformed"),
    ],
)
def test_claim_rejections(claims, category):
    token = sign_eddsa_token(claims, private_key_pem=PRIVATE, kid="k1")
    with pytest.raises(JwtVerifyError) as exc:
        _verify(token)
    assert exc.value.category == category


def test_session_profile_rejects_delegation_and_vice_versa():
    session = sign_eddsa_token(
        _claims(aud="kazenai-platform", token_use="session", exp=NOW + 3600),
        private_key_pem=PRIVATE,
        kid="k1",
    )
    with pytest.raises(JwtVerifyError) as exc:
        _verify(session)  # FinOps delegation profile
    assert exc.value.category in {"audience", "token_use"}
    payload = verify_eddsa_token(
        session,
        platform_token_profile(audience="kazenai-platform", token_use="session", public_keys=(("k1", PUBLIC),)),
        now=NOW + 5,
    )
    assert payload["token_use"] == "session"


def test_wrong_key_unknown_kid_and_missing_kid():
    forged = sign_eddsa_token(_claims(), private_key_pem=OTHER_PRIVATE, kid="k1")
    with pytest.raises(JwtVerifyError) as exc:
        _verify(forged)
    assert exc.value.category == "signature"
    unknown = sign_eddsa_token(_claims(), private_key_pem=PRIVATE, kid="k9")
    with pytest.raises(JwtVerifyError) as exc:
        _verify(unknown)
    assert exc.value.category == "unavailable_key"
    no_kid = pyjwt.encode(_claims(), PRIVATE, algorithm="EdDSA")
    with pytest.raises(JwtVerifyError) as exc:
        _verify(no_kid)
    assert exc.value.category == "missing_claim"


def test_hs256_token_with_shared_secret_is_refused():
    """The old shared-secret path cannot produce a token a service accepts."""
    hs = pyjwt.encode(_claims(), "x" * 64, algorithm="HS256", headers={"kid": "k1"})
    with pytest.raises(JwtVerifyError) as exc:
        _verify(hs)
    assert exc.value.category == "algorithm"


def test_alg_none_refused():
    none_token = pyjwt.encode(_claims(), None, algorithm="none", headers={"kid": "k1"})
    with pytest.raises(JwtVerifyError) as exc:
        _verify(none_token)
    assert exc.value.category == "algorithm"


def test_no_keys_configured():
    token = sign_eddsa_token(_claims(), private_key_pem=PRIVATE, kid="k1")
    with pytest.raises(JwtVerifyError) as exc:
        _verify(token, public_keys=())
    assert exc.value.category == "unavailable_key"


def test_load_public_keys_from_env(monkeypatch):
    monkeypatch.setenv("KAZENAI_SESSION_JWT_PUBLIC_KEYS", json.dumps({"k1": PUBLIC, "k2": PUBLIC}))
    assert [kid for kid, _ in load_public_keys()] == ["k1", "k2"]
    # gen-local-keys writes PEMs as one line with literal \n, matching Platform.
    escaped = PUBLIC.strip().replace("\n", "\\n")
    monkeypatch.setenv("KAZENAI_SESSION_JWT_PUBLIC_KEYS", json.dumps({"k1": escaped}))
    loaded = load_public_keys()
    token = sign_eddsa_token(_claims(), private_key_pem=PRIVATE, kid="k1")
    assert _verify(token, public_keys=loaded)["sub"] == "user:u1"
    monkeypatch.setenv("KAZENAI_SESSION_JWT_PUBLIC_KEYS", json.dumps({"k1": "not a key"}))
    with pytest.raises(JwtVerifyError):
        load_public_keys()
    monkeypatch.setenv("KAZENAI_SESSION_JWT_PUBLIC_KEYS", "{bad json")
    with pytest.raises(JwtVerifyError):
        load_public_keys()


def test_rotation_accepts_either_kid():
    new_private, new_public = _keypair()
    ring = (("old", PUBLIC), ("new", new_public))
    old_tok = sign_eddsa_token(_claims(), private_key_pem=PRIVATE, kid="old")
    new_tok = sign_eddsa_token(_claims(), private_key_pem=new_private, kid="new")
    assert _verify(old_tok, public_keys=ring)["jti"] == "dlg_1"
    assert _verify(new_tok, public_keys=ring)["jti"] == "dlg_1"
