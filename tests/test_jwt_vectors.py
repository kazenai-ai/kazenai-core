"""Shared JWT vector corpus against core verifier."""

from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.util
import json
from pathlib import Path

import jwt
import pytest

from kazenai.auth.jwt import IssuerProfile, JwtVerifyConfig, JwtVerifyError, verify_hs256_token

ROOT = Path(__file__).resolve().parents[2]
# Vendored corpus: standalone CI checks out only this repo. When the sibling
# kazenai-contracts checkout exists, the vendored copy must match it byte-for-byte.
VENDORED_DIR = Path(__file__).resolve().parent / "fixtures" / "contracts_auth"
SIBLING_DIR = ROOT / "kazenai-contracts" / "fixtures" / "auth"
VECTORS_PATH = VENDORED_DIR / "jwt_vectors.v1.json"
POLICY_PATH = VENDORED_DIR / "jwt_policy.v1.json"


def test_vendored_corpus_matches_manifest_and_sibling():
    manifest = {}
    for line in (VENDORED_DIR / "CORPUS.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        manifest[name] = digest
    assert set(manifest) == {"jwt_policy.v1.json", "jwt_vectors.v1.json"}
    for name, digest in manifest.items():
        data = (VENDORED_DIR / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, f"{name} differs from CORPUS.sha256"
        sibling = SIBLING_DIR / name
        if sibling.is_file():
            assert sibling.read_bytes() == data, (
                f"vendored {name} drifted from kazenai-contracts; re-vendor it"
            )
        if importlib.util.find_spec("kazenai_contracts") is not None:
            from kazenai_contracts.resources import auth_fixture_text

            assert auth_fixture_text(name).encode() == data, (
                f"vendored {name} drifted from the kazenai-contracts wheel"
            )


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _mint(claims: dict, key: str | None, header: dict | None = None, *, unsigned: bool = False) -> str:
    # Manual compact mint so JSON null jti / numeric kid survive (PyJWT coerces).
    hdr = dict(header or {"alg": "HS256", "typ": "JWT"})
    if "alg" not in hdr:
        hdr["alg"] = "none" if unsigned else "HS256"
    h = _b64url(json.dumps(hdr, separators=(",", ":"), allow_nan=False).encode())
    body = _b64url(json.dumps(claims, separators=(",", ":"), allow_nan=False).encode())
    if unsigned or not key:
        return f"{h}.{body}."
    sig = _b64url(
        hmac.new(key.encode("utf-8"), f"{h}.{body}".encode("ascii"), hashlib.sha256).digest()
    )
    return f"{h}.{body}.{sig}"


def _config_from_corpus(corpus: dict) -> JwtVerifyConfig:
    keys = corpus["synthetic_keys"]
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    profiles = []
    for profile in policy["profiles"]:
        key_names = corpus["runtime_keyrings"][profile["name"]]
        profiles.append(
            IssuerProfile(
                name=profile["name"],
                issuer=profile["issuer"],
                algorithms=tuple(profile["algorithms"]),
                audiences=frozenset(profile["audiences"]),
                identity_type=profile["identity_type"],
                keys=tuple(keys[name] for name in key_names),
                require_sub=profile["require_sub"],
                require_exp=profile["require_exp"],
                validate_nbf=profile["validate_nbf"],
                validate_iat=profile.get("validate_iat", True),
                leeway_seconds=int(profile["leeway_seconds"]),
                allow_missing_kid=profile["allow_missing_kid"],
            )
        )
    return JwtVerifyConfig(profiles=tuple(profiles))


def _config_for_vector(corpus: dict, vector: dict) -> JwtVerifyConfig:
    config = _config_from_corpus(corpus)
    override = vector.get("verification") or {}
    leeway = override.get("leeway_seconds")
    if leeway is None:
        return config
    adjusted = []
    for profile in config.profiles:
        adjusted.append(
            IssuerProfile(
                name=profile.name,
                issuer=profile.issuer,
                algorithms=profile.algorithms,
                audiences=profile.audiences,
                identity_type=profile.identity_type,
                keys=profile.keys,
                kid_keys=profile.kid_keys,
                require_sub=profile.require_sub,
                require_exp=profile.require_exp,
                validate_nbf=profile.validate_nbf,
                validate_iat=profile.validate_iat,
                leeway_seconds=int(leeway),
                allow_missing_kid=profile.allow_missing_kid,
            )
        )
    return JwtVerifyConfig(profiles=tuple(adjusted))


def test_shared_jwt_vector_corpus() -> None:
    # Shared corpus is required for standalone conformance.
    assert VECTORS_PATH.is_file(), f"missing JWT corpus at {VECTORS_PATH}"
    assert POLICY_PATH.is_file(), f"missing JWT policy at {POLICY_PATH}"
    corpus = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
    now = int(corpus["clock"]["now_unix"])
    keys = corpus["synthetic_keys"]

    for vector in corpus["vectors"]:
        config = _config_for_vector(corpus, vector)
        if vector.get("raw_token") is not None:
            token = vector["raw_token"]
        else:
            sign_key_name = vector.get("sign_key")
            key = keys.get(sign_key_name) if sign_key_name else None
            claims = dict(vector.get("claims") or {})
            token = _mint(
                claims,
                key,
                header=vector.get("header"),
                unsigned=bool(vector.get("unsigned")),
            )
            if vector.get("tamper_org_id") is not None and token.count(".") == 2:
                header_b64, payload_b64, sig = token.split(".")
                payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=="))
                payload["org_id"] = vector["tamper_org_id"]
                token = f"{header_b64}.{_b64url(json.dumps(payload, separators=(',', ':')).encode())}.{sig}"
        expect = vector["expect"]
        if expect.get("accept"):
            payload = verify_hs256_token(token, config, now=now)
            assert str(payload.get("sub") or "")
        else:
            with pytest.raises(JwtVerifyError) as excinfo:
                verify_hs256_token(token, config, now=now)
            assert excinfo.value.category == expect["error_category"], vector["id"]


def test_prefix_extended_supabase_issuer_rejected() -> None:
    secret = "p0-kazenai-jwt-secret-32b!!"
    trusted = "https://example.supabase.co/auth/v1"
    token = jwt.encode(
        {
            "sub": "user-1",
            "org_id": "org-1",
            "exp": 1700003600,
            "iss": f"{trusted}/evil",
            "aud": "authenticated",
        },
        secret,
        algorithm="HS256",
    )
    with pytest.raises(JwtVerifyError) as excinfo:
        verify_hs256_token(
            token,
            JwtVerifyConfig(secret=secret, supabase_issuer=trusted, enforce_iss_aud=True),
            now=1700000000,
        )
    assert excinfo.value.category == "issuer"
