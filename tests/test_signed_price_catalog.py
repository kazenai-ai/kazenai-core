"""Signed price catalog: export, verify, refuse, install, and the FinOps feed."""

from __future__ import annotations

import base64
import io
import json
import os
import stat
import urllib.error
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

cryptography = pytest.importorskip("cryptography")
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

import kazenai.catalog_document as cd  # noqa: E402
import kazenai.catalog_feed as feed  # noqa: E402
import kazenai.price_catalog as pc  # noqa: E402
from kazenai.request_pricing import prepare_provider_call, settle_usage, settlement_evidence  # noqa: E402

_MESSAGES = [{"role": "user", "content": "hi"}]


def _keypair():
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return pem, base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


PEM, PUBLIC = _keypair()
OTHER_PEM, OTHER_PUBLIC = _keypair()


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setenv(cd.PUBLIC_KEYS_ENV, json.dumps({"kzprice-test": PUBLIC}))
    monkeypatch.setenv("KAZENAI_PRICING_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)
    monkeypatch.delenv("KAZENAI_ENDPOINT_PRICE_MULTIPLIERS", raising=False)
    cd.reset_for_tests()
    feed.reset_for_tests()
    yield
    cd.reset_for_tests()
    feed.reset_for_tests()


def _signed(version: str = "2026-10-17.1", *, issued_at=None, expires_in=timedelta(days=30), mutate=None, pem=PEM, kid="kzprice-test"):
    payload = cd.export_payload(version=version, issued_at=issued_at, expires_in=expires_in)
    if mutate is not None:
        mutate(payload)
    return cd.sign_payload(payload, private_key_pem=pem, kid=kid)


def _gpt4o_input(payload, value="99"):
    for row in payload["rows"]:
        if row["model"] == "gpt-4o" and row["request_tier"] == "standard":
            row["rates"]["input"] = value


# --- the document ------------------------------------------------------------------


def test_bundled_catalog_round_trips_through_the_document():
    bundle = cd.parse_payload(cd.export_payload())
    assert set(bundle.rows) == set(pc.BUNDLED.rows)
    for key, row in pc.BUNDLED.rows.items():
        parsed = bundle.rows[key]
        assert parsed.rates.as_json() == row.rates.as_json()
        assert parsed.scheduled == row.scheduled and parsed.valid_until == row.valid_until
        assert (parsed.certified, parsed.long_context_threshold) == (row.certified, row.long_context_threshold)
    assert dict(bundle.aliases) == dict(pc.BUNDLED.aliases)
    assert bundle.openai_patch_sizing == pc.BUNDLED.openai_patch_sizing
    assert bundle.anthropic_tool_definition_tokens == pc.BUNDLED.anthropic_tool_definition_tokens


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(extra=1),
        lambda p: p["rows"][0].update(unknown="x"),
        lambda p: p["rows"][0]["rates"].update(input="-1"),
        lambda p: p["rows"][0]["rates"].update(input="NaN"),
        lambda p: p["rows"].append(dict(p["rows"][0])),
        lambda p: p["aliases"].update({"gpt-4o-*": "gpt-4o"}),
        lambda p: p["aliases"].update({"x": "not-a-model"}),
        lambda p: p["endpoints"].update(non_first_party_uplift="0.9"),
        lambda p: p["anthropic"]["inference_geo_multipliers"].update(us="0.5"),
        lambda p: p.update(rounding_rule="floor"),
        lambda p: p.update(catalog_version="latest"),
    ],
)
def test_malformed_or_unsafe_payloads_are_refused(mutate):
    payload = cd.export_payload(version="2026-10-17.1")
    mutate(payload)
    with pytest.raises(cd.CatalogDocumentError):
        cd.parse_payload(payload)


# --- signatures and ordering -------------------------------------------------------


def test_a_signed_catalog_installs_and_prices_new_values():
    before = prepare_provider_call({"model": "gpt-4o", "max_tokens": 10, "messages": _MESSAGES}, provider="openai", surface="chat")
    cd.accept_envelope(_signed(mutate=_gpt4o_input))
    after = prepare_provider_call({"model": "gpt-4o", "max_tokens": 10, "messages": _MESSAGES}, provider="openai", surface="chat")
    assert pc.active_catalog_version() == "2026-10-17.1"
    assert after.decision.catalog_version == "2026-10-17.1"
    assert after.decision.rate_vector.input == Decimal("99")
    assert after.reserved_micro_usd > before.reserved_micro_usd


def test_a_new_model_needs_no_sdk_release():
    from kazenai.pricing_policy import UnknownModelError

    with pytest.raises(UnknownModelError):
        prepare_provider_call({"model": "gpt-9", "max_tokens": 10, "messages": _MESSAGES}, provider="openai", surface="chat")

    def _add(payload):
        row = next(r for r in payload["rows"] if r["model"] == "gpt-4o" and r["request_tier"] == "standard")
        payload["rows"].append({**json.loads(json.dumps(row)), "model": "gpt-9"})

    cd.accept_envelope(_signed(mutate=_add))
    assert prepare_provider_call({"model": "gpt-9", "max_tokens": 10, "messages": _MESSAGES}, provider="openai", surface="chat").reserved_micro_usd > 0


def test_tampering_unknown_keys_and_other_algorithms_are_refused():
    envelope = _signed()
    tampered = dict(envelope, payload=envelope["payload"].replace('"output":"10"', '"output":"0.01"', 1))
    assert tampered["payload"] != envelope["payload"]
    for case in (
        tampered,
        _signed(pem=OTHER_PEM),
        dict(envelope, signatures=[dict(envelope["signatures"][0], alg="RS256")]),
        dict(envelope, signatures=[]),
        dict(envelope, format="kazenai.price-catalog.signed.v0"),
    ):
        with pytest.raises(cd.CatalogDocumentError):
            cd.accept_envelope(case)
    assert pc.active_bundle().source == "bundled"


def test_non_canonical_payload_text_is_refused():
    envelope = _signed()
    pretty = json.dumps(json.loads(envelope["payload"]), indent=2)
    key = serialization.load_pem_private_key(PEM, password=None)
    sig = base64.urlsafe_b64encode(key.sign(pretty.encode())).rstrip(b"=").decode()
    with pytest.raises(cd.CatalogDocumentError, match="canonical"):
        cd.accept_envelope({**envelope, "payload": pretty, "signatures": [{"kid": "kzprice-test", "alg": "Ed25519", "sig": sig}]})


def test_without_a_pinned_key_nothing_is_trusted(monkeypatch):
    monkeypatch.setattr(cd, "TRUSTED_CATALOG_KEYS", {})
    monkeypatch.delenv(cd.PUBLIC_KEYS_ENV, raising=False)
    with pytest.raises(cd.CatalogDocumentError, match="no pricing public key"):
        cd.accept_envelope(_signed())


def test_an_environment_key_cannot_replace_a_pinned_key(monkeypatch):
    monkeypatch.setattr(cd, "TRUSTED_CATALOG_KEYS", {"kzprice-test": PUBLIC})
    monkeypatch.setenv(cd.PUBLIC_KEYS_ENV, json.dumps({"kzprice-test": OTHER_PUBLIC}))
    with pytest.raises(cd.CatalogDocumentError, match="cannot replace pinned key"):
        cd.trusted_keys()


def test_rollback_older_than_bundled_expired_and_future_catalogs_are_refused():
    cd.accept_envelope(_signed("2026-10-17.2"))
    with pytest.raises(cd.CatalogDocumentError, match="rollback"):
        cd.accept_envelope(_signed("2026-10-17.1"))
    cd.accept_envelope(_signed("2026-10-17.2"))  # same version: idempotent reload
    with pytest.raises(cd.CatalogDocumentError, match="older than the bundled"):
        cd.accept_envelope(_signed("2026-10-01"))
    with pytest.raises(cd.CatalogDocumentError, match="expired"):
        cd.accept_envelope(_signed("2026-10-18", issued_at=datetime.now(timezone.utc) - timedelta(days=40)))
    with pytest.raises(cd.CatalogDocumentError, match="future"):
        cd.accept_envelope(_signed("2026-10-19", issued_at=datetime.now(timezone.utc) + timedelta(days=2)))
    assert pc.active_catalog_version() == "2026-10-17.2"


def test_an_expired_live_catalog_falls_back_to_the_bundled_snapshot():
    bundle = cd.accept_envelope(_signed("2026-10-17.1", expires_in=timedelta(hours=1)))
    assert pc.active_bundle().version == "2026-10-17.1"
    later = pc.parse_instant(bundle.expires_at) + timedelta(seconds=1)
    assert pc.active_bundle(later).source == "bundled"


# --- the FinOps feed -------------------------------------------------------------


class _Response(io.BytesIO):
    def __init__(self, body: bytes, etag: str = '"e1"'):
        super().__init__(body)
        self.headers = {"ETag": etag}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _serve(monkeypatch, envelopes: list, seen: list):
    def urlopen(request, timeout=0):  # noqa: ARG001
        seen.append({"url": request.full_url, "headers": dict(request.header_items())})
        item = envelopes.pop(0)
        if isinstance(item, int):
            raise urllib.error.HTTPError(request.full_url, item, "status", {}, io.BytesIO(b""))
        return _Response(json.dumps(item).encode())

    monkeypatch.setattr(feed.outbound_http, "urlopen", urlopen)


def test_feed_fetches_from_finops_verifies_installs_and_caches(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "https://finops.invalid")
    monkeypatch.setenv("KAZENAI_FINOPS_API_KEY", "sdk-key")
    seen: list = []
    _serve(monkeypatch, [_signed("2026-10-17.3"), 304], seen)
    assert feed.enabled() is True
    assert feed.refresh_now() == "installed"
    assert seen[0]["url"] == "https://finops.invalid/v1/pricing/catalog"
    assert {k.lower(): v for k, v in seen[0]["headers"].items()}["x-api-key"] == "sdk-key"
    assert pc.active_catalog_version() == "2026-10-17.3"
    path = feed.cache_path()
    assert path and oct(os.stat(path).st_mode & 0o777) == oct(0o600)
    assert not os.stat(os.path.dirname(path)).st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    assert feed.refresh_now() == "not_modified"
    assert {k.lower(): v for k, v in seen[1]["headers"].items()}["if-none-match"] == '"e1"'
    # A fresh process loads the verified cache before any network call.
    cd.reset_for_tests()
    feed.reset_for_tests()
    feed._load_cache()
    assert pc.active_catalog_version() == "2026-10-17.3"


def test_feed_refuses_a_bad_catalog_and_keeps_the_previous_one(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "https://finops.invalid")
    seen: list = []
    _serve(monkeypatch, [_signed("2026-10-17.4"), _signed("2026-10-17.5", pem=OTHER_PEM), 404, 503], seen)
    assert feed.refresh_now() == "installed"
    assert feed.refresh_now() == "refused"
    assert feed.refresh_now() == "none_published"
    assert feed.refresh_now() == "http_error"
    assert pc.active_catalog_version() == "2026-10-17.4"


def test_feed_is_off_without_finops_without_keys_or_when_disabled(monkeypatch):
    monkeypatch.setattr(cd, "TRUSTED_CATALOG_KEYS", {})
    assert feed.enabled() is False
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "https://finops.invalid")
    assert feed.enabled() is True
    monkeypatch.setenv("KAZENAI_PRICING_LIVE", "0")
    assert feed.enabled() is False
    monkeypatch.delenv("KAZENAI_PRICING_LIVE")
    monkeypatch.delenv(cd.PUBLIC_KEYS_ENV)
    assert feed.enabled() is False


def test_ensure_current_never_raises_and_refreshes_in_the_background(monkeypatch):
    monkeypatch.setenv("KAZENAI_FINOPS_URL", "https://finops.invalid")

    def boom(request, timeout=0):  # noqa: ARG001
        raise OSError("network down")

    monkeypatch.setattr(feed.outbound_http, "urlopen", boom)
    feed.ensure_current()
    thread = feed._STATE["thread"]
    assert thread is not None
    thread.join(timeout=5)
    assert pc.active_bundle().source == "bundled"
    # The call path still prices on the bundled snapshot.
    assert prepare_provider_call({"model": "gpt-4o", "max_tokens": 10, "messages": _MESSAGES}, provider="openai", surface="chat").reserved_micro_usd > 0


# --- settlement evidence for FinOps re-pricing --------------------------------------


def test_settle_bodies_carry_the_priced_token_components():
    from kazenai.lifecycle_transitions import build_settle_payload

    pricing = prepare_provider_call({"model": "gpt-4o", "max_tokens": 10, "messages": _MESSAGES}, provider="openai", surface="chat")
    usage = {"prompt_tokens": 1200, "completion_tokens": 30, "prompt_tokens_details": {"cached_tokens": 200}, "service_tier": "default"}
    micros = settle_usage(pricing, usage)
    evidence = settlement_evidence(pricing)
    assert evidence["tokens"] == {
        "input_tokens": 1000,
        "cached_input_tokens": 200,
        "cache_write_5m_tokens": 0,
        "cache_write_1h_tokens": 0,
        "output_tokens": 30,
        "audio_input_tokens": 0,
    }
    assert evidence["catalog_version"] == pricing.decision.catalog_version
    body = build_settle_payload(
        reservation_id="r", call_id="c", attempt=1, org_id="o", event="usage_known",
        actual_usd_micros=micros, usage_evidence=evidence,
    )
    assert body["usage_evidence"]["tokens"]["output_tokens"] == 30
    assert "usage_evidence" not in build_settle_payload(
        reservation_id="r", call_id="c", attempt=1, org_id="o", event="provider_started", usage_evidence=evidence
    )


def test_guard_settle_posts_the_evidence(monkeypatch):
    from types import SimpleNamespace

    from kazenai.spine import guard

    pricing = prepare_provider_call({"model": "gpt-4o", "max_tokens": 10, "messages": _MESSAGES}, provider="openai", surface="chat")
    settle_usage(pricing, {"prompt_tokens": 100, "completion_tokens": 5})
    handle = SimpleNamespace(pricing_call=pricing)
    evidence = guard._handle_settlement_evidence(handle)
    assert evidence["tokens"]["input_tokens"] == 100
    assert guard._handle_settlement_evidence(SimpleNamespace()) == {}


def test_price_catalog_doc_is_generated_from_the_code():
    """docs/PRICE_CATALOG.md must match the catalog; regenerate with `price_catalog_tool.py doc`."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("price_catalog_tool", root / "scripts" / "price_catalog_tool.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert (root / "docs" / "PRICE_CATALOG.md").read_text(encoding="utf-8") == tool._render_doc()
