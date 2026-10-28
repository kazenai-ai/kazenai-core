"""Raw-body ingest envelope. A changed byte, header, or reserialization fails closed."""

from __future__ import annotations

import json

import pytest

from kazenai.ingest_signature import (
    IngestSignatureError,
    sign_request,
    tenant_key_id,
    verify_signature,
)


def _body() -> bytes:
    return b'{"events":[{"event_id":"e1","n":1}]}'


def test_tenant_key_id_is_not_the_secret():
    key = "kaz-live-secret"
    key_id = tenant_key_id(key)
    assert key_id.startswith("tenant:")
    assert key not in key_id
    assert tenant_key_id(key) == key_id


def test_valid_envelope_and_field_mutations():
    raw = _body()
    headers = sign_request(
        secret="signing-secret",
        method="post",
        path="/v1/events",
        key_id="tenant:abc",
        raw_body=raw,
        timestamp=1_700_000_000,
        nonce="nonce-1",
    )
    kwargs = dict(
        secret="signing-secret",
        method="POST",
        path="/v1/events",
        key_id="tenant:abc",
        timestamp=headers["X-Kazen-Timestamp"],
        nonce=headers["X-Kazen-Nonce"],
        content_type=headers["Content-Type"],
        raw_body=raw,
        signature=headers["X-Kazen-Signature"],
        now_s=1_700_000_000,
    )
    verify_signature(**kwargs)
    for field, value in (
        ("method", "GET"),
        ("path", "/v1/events?x=1"),
        ("key_id", "other"),
        ("timestamp", "1700000001"),
        ("nonce", "nonce-2"),
        ("content_type", "text/plain"),
        ("raw_body", raw + b" "),
    ):
        mutated = dict(kwargs)
        mutated[field] = value
        with pytest.raises(IngestSignatureError):
            verify_signature(**mutated)


def test_reserialized_equivalent_json_is_a_different_signature():
    compact = b'{"a":1,"b":2}'
    pretty = json.dumps({"b": 2, "a": 1}).encode("utf-8")
    assert compact != pretty
    headers = sign_request(
        secret="signing-secret",
        method="POST",
        path="/v1/events",
        key_id="k",
        raw_body=compact,
        timestamp=1_700_000_000,
        nonce="n",
    )
    with pytest.raises(IngestSignatureError):
        verify_signature(
            secret="signing-secret",
            method="POST",
            path="/v1/events",
            key_id="k",
            timestamp=headers["X-Kazen-Timestamp"],
            nonce=headers["X-Kazen-Nonce"],
            content_type="application/json",
            raw_body=pretty,
            signature=headers["X-Kazen-Signature"],
            now_s=1_700_000_000,
        )


@pytest.mark.parametrize(
    "values",
    [[], ["application/json; charset=utf-8"], ["text/plain"], ["application/json", "application/json"], [" "]],
)
def test_content_type_is_exact_never_normalised(values):
    from kazenai.ingest_signature import content_type_values

    with pytest.raises(IngestSignatureError):
        content_type_values({}, values)


def test_content_type_accepts_the_single_exact_value():
    from kazenai.ingest_signature import content_type_values

    assert content_type_values({}, ["application/json"]) == "application/json"


def test_nonce_outlives_the_whole_acceptance_window():
    from kazenai.ingest_signature import nonce_ttl_s

    # A timestamp is accepted up to skew either side, so an envelope is
    # verifiable for 2 * skew; the nonce must be remembered at least that long.
    for skew in (10, 300, 900):
        assert nonce_ttl_s(skew) > 2 * skew


def test_signed_path_ignores_scheme_host_and_query():
    from kazenai.ingest_signature import signed_path

    assert signed_path("https://finops.example/v1/events?x=1") == "/v1/events"
    assert signed_path("http://lens:8780/api/v1/events") == "/api/v1/events"
