"""Signed price catalog documents.

The bundled snapshot in :mod:`kazenai.price_catalog` is the default and the
floor. A newer catalog is published as data, signed offline with a KazenAI
pricing key, and distributed by Agent FinOps. FinOps and the SDK verify it
against the public keys pinned below and install it only when:

* the Ed25519 signature verifies with a trusted key id;
* the payload is a well-formed ``kazenai.price-catalog.v1`` document;
* its version is newer than the bundled snapshot and than any catalog this
  process (or its cache) already accepted (no rollback);
* it is unexpired and was not issued in the future.

Anything else is refused and the previous catalog stays active, so a tampered
bucket, a man in the middle, or a replayed old file cannot lower a price.

Envelope (``kazenai.price-catalog.signed.v1``)::

    {"format": "kazenai.price-catalog.signed.v1",
     "payload": "<the exact canonical JSON that was signed>",
     "signatures": [{"kid": "kzprice-2026-10", "alg": "Ed25519", "sig": "<base64url>"}]}

Verification needs the ``cryptography`` package (``kazenai[server]``). Without
it a signed catalog is never trusted and the bundled snapshot stays active.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .price_catalog import (
    BUNDLED,
    ROUNDING_RULE,
    CatalogBundle,
    CatalogRow,
    PricingConfigError,
    RateVector,
    install_live_bundle,
    parse_instant,
)

_log = logging.getLogger("kazenai.catalog_document")

SCHEMA = "kazenai.price-catalog.v1"
ENVELOPE_FORMAT = "kazenai.price-catalog.signed.v1"
MAX_ENVELOPE_BYTES = 2 * 1024 * 1024
ISSUED_AT_MAX_SKEW = timedelta(hours=1)

# KazenAI pricing public keys: kid -> base64url raw Ed25519 public key (32 bytes).
# The operator generates the key pair offline (scripts/price_catalog_tool.py
# keygen), keeps the private key offline or in a KMS, and pins the public key
# here before a release. Until a key is pinned, no live catalog is trusted.
TRUSTED_CATALOG_KEYS: Dict[str, str] = {"kzprice-2026-10": "pOKKh2JTvCt0n4IevCeIBy7_gbmRougGttWcoHlt8TQ"}

# Self-hosted or test deployments may add keys (same shape) through the
# environment. Pinned keys cannot be removed this way.
PUBLIC_KEYS_ENV = "KAZENAI_PRICING_CATALOG_PUBLIC_KEYS"

_VERSION_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:\.(\d{1,6}))?$")
_KID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_TOP = {
    "schema", "catalog_version", "issued_at", "expires_at", "currency", "unit",
    "rounding_rule", "rows", "aliases", "anthropic", "openai", "endpoints",
}
_ROW = {
    "provider", "model", "request_tier", "context_band", "expected_reported_tier",
    "rates", "certified", "source_url", "verified_on", "long_context_threshold",
    "pricing_rule", "scheduled", "valid_until",
}
_RATES = {
    "input", "cached_input", "cache_write_5m", "cache_write_1h", "output",
    "image", "tool_per_call", "audio_input", "audio_cached_input",
}
_ANTHROPIC = {
    "client_tool_versions", "tool_system_prompt_tokens", "tool_system_prompt_fallback",
    "tool_definition_tokens", "allowed_speeds", "inference_geo_multipliers",
}
_OPENAI = {"image_tile_models", "image_patch_multipliers", "image_patch_sizing"}
_ENDPOINTS = {"first_party_api_hosts", "non_first_party_uplift"}


class CatalogDocumentError(PricingConfigError):
    """A catalog document or envelope was refused. The previous catalog stays."""


def version_key(version: str) -> Tuple[int, int, int, int]:
    """Order catalog versions ``YYYY-MM-DD`` or ``YYYY-MM-DD.N``."""
    match = _VERSION_RE.match(str(version or ""))
    if match is None:
        raise CatalogDocumentError(f"catalog_version {version!r} must be YYYY-MM-DD or YYYY-MM-DD.N")
    year, month, day, serial = match.groups()
    return int(year), int(month), int(day), int(serial or 0)


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _utc_z(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- export --------------------------------------------------------------------


def _rates_json(rates: RateVector) -> dict:
    return rates.as_json()


def _row_json(row: CatalogRow) -> dict:
    return {
        "provider": row.provider,
        "model": row.model,
        "request_tier": row.request_tier,
        "context_band": row.context_band,
        "expected_reported_tier": row.expected_reported_tier,
        "rates": _rates_json(row.rates),
        "certified": bool(row.certified),
        "source_url": row.source_url,
        "verified_on": row.verified_on,
        "long_context_threshold": row.long_context_threshold,
        "pricing_rule": row.pricing_rule,
        "scheduled": [{"effective_from": when, "rates": _rates_json(rates)} for when, rates in row.scheduled],
        "valid_until": row.valid_until,
    }


def export_payload(
    bundle: CatalogBundle = BUNDLED,
    *,
    version: Optional[str] = None,
    issued_at: Optional[datetime] = None,
    expires_in: timedelta = timedelta(days=30),
) -> dict:
    """The catalog as a ``kazenai.price-catalog.v1`` payload (unsigned)."""
    issued = issued_at or datetime.now(timezone.utc)
    chosen = version or bundle.version
    version_key(chosen)
    return {
        "schema": SCHEMA,
        "catalog_version": chosen,
        "issued_at": _utc_z(issued),
        "expires_at": _utc_z(issued + expires_in),
        "currency": "USD",
        "unit": "per_1m_tokens",
        "rounding_rule": ROUNDING_RULE,
        "rows": [_row_json(row) for _key, row in sorted(bundle.rows.items())],
        "aliases": dict(sorted(bundle.aliases.items())),
        "anthropic": {
            "client_tool_versions": list(bundle.anthropic_client_tool_versions),
            "tool_system_prompt_tokens": {k: list(v) for k, v in sorted(bundle.anthropic_tool_system_prompt_tokens.items())},
            "tool_system_prompt_fallback": int(bundle.anthropic_tool_system_prompt_fallback),
            "tool_definition_tokens": dict(sorted(bundle.anthropic_tool_definition_tokens.items())),
            "allowed_speeds": list(bundle.anthropic_allowed_speeds),
            "inference_geo_multipliers": {k: format(v, "f") for k, v in sorted(bundle.anthropic_inference_geo_multipliers.items())},
        },
        "openai": {
            "image_tile_models": {k: list(v) for k, v in sorted(bundle.openai_tile_models.items())},
            "image_patch_multipliers": {k: format(v, "f") for k, v in sorted(bundle.openai_patch_multipliers.items())},
            "image_patch_sizing": {
                model: {detail: [edge, budget] for detail, (edge, budget) in sorted(sizing.items())}
                for model, sizing in sorted(bundle.openai_patch_sizing.items())
            },
        },
        "endpoints": {
            "first_party_api_hosts": {k: list(v) for k, v in sorted(bundle.first_party_api_hosts.items())},
            "non_first_party_uplift": format(bundle.non_first_party_uplift, "f"),
        },
    }


# --- parse ---------------------------------------------------------------------


def _fail(message: str) -> None:
    raise CatalogDocumentError(message)


def _exact_keys(node: Any, allowed: set, where: str, *, required: Iterable[str] = ()) -> Mapping[str, Any]:
    if not isinstance(node, Mapping):
        _fail(f"{where} must be an object")
    unknown = set(node) - allowed
    if unknown:
        _fail(f"{where} has unknown fields {sorted(unknown)}")
    missing = [key for key in required if key not in node]
    if missing:
        _fail(f"{where} is missing {missing}")
    return node


def _decimal(value: Any, where: str, *, minimum: Decimal = Decimal("0")) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        _fail(f"{where} must be a decimal string")
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        _fail(f"{where} is not a decimal")
    if not number.is_finite() or number < minimum:
        _fail(f"{where} must be finite and >= {minimum}")
    return number


def _int(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{where} must be an integer >= {minimum}")
    return int(value)


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{where} must be a non-empty string")
    return value.strip()


def _parse_rates(node: Any, where: str) -> RateVector:
    rates = _exact_keys(node, _RATES, where, required=("input", "output"))

    def _opt(name: str) -> Optional[Decimal]:
        value = rates.get(name)
        return None if value is None else _decimal(value, f"{where}.{name}")

    return RateVector(
        input=_decimal(rates["input"], f"{where}.input"),
        cached_input=_opt("cached_input"),
        cache_write_5m=_opt("cache_write_5m"),
        cache_write_1h=_opt("cache_write_1h"),
        output=_decimal(rates["output"], f"{where}.output"),
        image=_opt("image"),
        tool_per_call=_opt("tool_per_call"),
        audio_input=_opt("audio_input"),
        audio_cached_input=_opt("audio_cached_input"),
    )


def _instant(value: Any, where: str) -> Optional[str]:
    if value is None:
        return None
    text = _text(value, where)
    try:
        parse_instant(text)
    except PricingConfigError:
        _fail(f"{where} must be an ISO-8601 UTC instant")
    return text


def _parse_row(node: Any, index: int, version: str) -> CatalogRow:
    where = f"rows[{index}]"
    row = _exact_keys(
        node,
        _ROW,
        where,
        required=("provider", "model", "request_tier", "context_band", "expected_reported_tier", "rates", "source_url", "verified_on", "pricing_rule"),
    )
    threshold = row.get("long_context_threshold")
    scheduled = []
    for position, entry in enumerate(row.get("scheduled") or ()):
        item = _exact_keys(entry, {"effective_from", "rates"}, f"{where}.scheduled[{position}]", required=("effective_from", "rates"))
        scheduled.append(
            (
                _instant(item["effective_from"], f"{where}.scheduled[{position}].effective_from"),
                _parse_rates(item["rates"], f"{where}.scheduled[{position}].rates"),
            )
        )
    certified = row.get("certified", True)
    if not isinstance(certified, bool):
        _fail(f"{where}.certified must be a boolean")
    return CatalogRow(
        provider=_text(row["provider"], f"{where}.provider").lower(),
        model=_text(row["model"], f"{where}.model").lower(),
        request_tier=_text(row["request_tier"], f"{where}.request_tier").lower(),
        context_band=_text(row["context_band"], f"{where}.context_band").lower(),
        expected_reported_tier=_text(row["expected_reported_tier"], f"{where}.expected_reported_tier"),
        rates=_parse_rates(row["rates"], f"{where}.rates"),
        certified=certified,
        source_url=_text(row["source_url"], f"{where}.source_url"),
        verified_on=_text(row["verified_on"], f"{where}.verified_on"),
        long_context_threshold=None if threshold is None else _int(threshold, f"{where}.long_context_threshold", minimum=1),
        pricing_rule=_text(row["pricing_rule"], f"{where}.pricing_rule"),
        label="catalog",
        catalog_version=version,
        scheduled=tuple(scheduled),
        valid_until=_instant(row.get("valid_until"), f"{where}.valid_until"),
    )


def parse_payload(payload: Mapping[str, Any], *, sha256: Optional[str] = None) -> CatalogBundle:
    """Validate a payload and return the bundle it describes. Raises on anything odd."""
    top = _exact_keys(payload, _TOP, "catalog", required=tuple(_TOP))
    if top["schema"] != SCHEMA:
        _fail(f"catalog schema must be {SCHEMA}")
    version = _text(top["catalog_version"], "catalog_version")
    version_key(version)
    if top["currency"] != "USD" or top["unit"] != "per_1m_tokens":
        _fail("catalog currency/unit must be USD per_1m_tokens")
    if top["rounding_rule"] != ROUNDING_RULE:
        _fail("catalog rounding_rule is not the rule this SDK implements")
    issued = _instant(top["issued_at"], "issued_at")
    expires = _instant(top["expires_at"], "expires_at")
    if parse_instant(expires) <= parse_instant(issued):
        _fail("expires_at must be after issued_at")
    rows_node = top["rows"]
    if not isinstance(rows_node, list) or not rows_node:
        _fail("rows must be a non-empty list")
    rows: Dict[Tuple[str, str, str, str], CatalogRow] = {}
    for index, node in enumerate(rows_node):
        row = _parse_row(node, index, version)
        if row.key in rows:
            _fail(f"duplicate catalog key {row.key!r}")
        rows[row.key] = row
    models = {row.model for row in rows.values()}
    aliases_node = top["aliases"]
    if not isinstance(aliases_node, Mapping):
        _fail("aliases must be an object")
    aliases: Dict[str, str] = {}
    for presented, target in aliases_node.items():
        name = _text(presented, "aliases key").lower()
        canonical = _text(target, f"aliases[{presented!r}]").lower()
        if any(ch in name for ch in "*?[]{}()|+^$\\") or canonical not in models:
            _fail(f"alias {presented!r} must be exact and point at a catalog model")
        aliases[name] = canonical

    anth = _exact_keys(top["anthropic"], _ANTHROPIC, "anthropic", required=tuple(_ANTHROPIC))
    prompt_tokens: Dict[str, Tuple[int, int]] = {}
    for model, pair in _exact_keys(anth["tool_system_prompt_tokens"], set(anth["tool_system_prompt_tokens"]), "anthropic.tool_system_prompt_tokens").items():
        if not isinstance(pair, list) or len(pair) != 2:
            _fail("anthropic.tool_system_prompt_tokens values must be [auto_none, any_tool]")
        prompt_tokens[str(model).lower()] = (_int(pair[0], "tool_system_prompt_tokens"), _int(pair[1], "tool_system_prompt_tokens"))
    definitions = {
        _text(name, "tool_definition_tokens key"): _int(value, f"tool_definition_tokens[{name!r}]")
        for name, value in dict(anth["tool_definition_tokens"]).items()
    }
    geo = {
        _text(name, "inference_geo key").lower(): _decimal(value, f"inference_geo_multipliers[{name!r}]", minimum=Decimal("1"))
        for name, value in dict(anth["inference_geo_multipliers"]).items()
    }
    speeds = tuple(_text(item, "allowed_speeds").lower() for item in anth["allowed_speeds"])
    tools = tuple(_text(item, "client_tool_versions") for item in anth["client_tool_versions"])

    oai = _exact_keys(top["openai"], _OPENAI, "openai", required=tuple(_OPENAI))
    tiles: Dict[str, Tuple[int, int]] = {}
    for model, pair in dict(oai["image_tile_models"]).items():
        if not isinstance(pair, list) or len(pair) != 2:
            _fail("openai.image_tile_models values must be [base, tile]")
        tiles[str(model).lower()] = (_int(pair[0], "image_tile_models", minimum=1), _int(pair[1], "image_tile_models", minimum=1))
    patches = {
        str(model).lower(): _decimal(value, f"image_patch_multipliers[{model!r}]", minimum=Decimal("1"))
        for model, value in dict(oai["image_patch_multipliers"]).items()
    }
    sizing: Dict[str, Dict[str, Tuple[int, Optional[int]]]] = {}
    for model, details in dict(oai["image_patch_sizing"]).items():
        if not isinstance(details, Mapping) or not details:
            _fail("openai.image_patch_sizing values must be non-empty objects")
        entry: Dict[str, Tuple[int, Optional[int]]] = {}
        for detail, limits in details.items():
            if detail not in {"low", "high", "original"} or not isinstance(limits, list) or len(limits) != 2:
                _fail("openai.image_patch_sizing entries must be low/high/original: [edge, budget|null]")
            budget = None if limits[1] is None else _int(limits[1], "image_patch_sizing budget", minimum=1)
            entry[detail] = (_int(limits[0], "image_patch_sizing edge", minimum=1), budget)
        sizing[str(model).lower()] = entry

    ends = _exact_keys(top["endpoints"], _ENDPOINTS, "endpoints", required=tuple(_ENDPOINTS))
    hosts = {
        _text(provider, "first_party_api_hosts key").lower(): tuple(_text(host, "first_party_api_hosts").lower() for host in values)
        for provider, values in dict(ends["first_party_api_hosts"]).items()
    }
    uplift = _decimal(ends["non_first_party_uplift"], "endpoints.non_first_party_uplift", minimum=Decimal("1"))
    return CatalogBundle(
        version=version,
        source="live",
        rows=rows,
        aliases=aliases,
        anthropic_client_tool_versions=tools,
        anthropic_tool_system_prompt_tokens=prompt_tokens,
        anthropic_tool_system_prompt_fallback=_int(anth["tool_system_prompt_fallback"], "tool_system_prompt_fallback"),
        anthropic_tool_definition_tokens=definitions,
        openai_tile_models=tiles,
        openai_patch_multipliers=patches,
        openai_patch_sizing=sizing,
        anthropic_inference_geo_multipliers=geo,
        anthropic_allowed_speeds=speeds,
        first_party_api_hosts=hosts,
        non_first_party_uplift=uplift,
        issued_at=issued,
        expires_at=expires,
        sha256=sha256,
    )


# --- signatures ------------------------------------------------------------------


def _b64url_decode(value: str, where: str) -> bytes:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value or ""):
        _fail(f"{where} must be unpadded base64url")
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (binascii.Error, ValueError):
        _fail(f"{where} is not base64url")
    return b""


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def trusted_keys() -> Dict[str, bytes]:
    """Pinned keys plus any added through ``KAZENAI_PRICING_CATALOG_PUBLIC_KEYS``."""
    keys: Dict[str, str] = dict(TRUSTED_CATALOG_KEYS)
    raw = os.getenv(PUBLIC_KEYS_ENV, "").strip()
    if raw:
        try:
            extra = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CatalogDocumentError(f"{PUBLIC_KEYS_ENV} is not valid JSON") from exc
        if not isinstance(extra, dict):
            raise CatalogDocumentError(f"{PUBLIC_KEYS_ENV} must be an object of kid -> key")
        for kid, value in extra.items():
            if kid in keys and keys[kid] != value:
                raise CatalogDocumentError(f"{PUBLIC_KEYS_ENV} cannot replace pinned key {kid!r}")
            keys[str(kid)] = str(value)
    out: Dict[str, bytes] = {}
    for kid, value in keys.items():
        if not _KID_RE.match(kid):
            raise CatalogDocumentError(f"pricing key id {kid!r} is invalid")
        raw_key = _b64url_decode(value, f"pricing key {kid!r}")
        if len(raw_key) != 32:
            raise CatalogDocumentError(f"pricing key {kid!r} must be a 32-byte Ed25519 public key")
        out[kid] = raw_key
    return out


def _verify_signature(public_key: bytes, message: bytes, signature: bytes) -> bool:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise CatalogDocumentError("verifying a signed catalog needs the `cryptography` package (kazenai[server])") from exc
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
        return True
    except (InvalidSignature, ValueError):
        return False


def verify_envelope(envelope: Any, *, keys: Optional[Mapping[str, bytes]] = None) -> Tuple[dict, str, str]:
    """Return ``(payload, kid, sha256)`` for a correctly signed envelope."""
    if isinstance(envelope, (bytes, bytearray)):
        if len(envelope) > MAX_ENVELOPE_BYTES:
            _fail("signed catalog is too large")
        try:
            envelope = json.loads(bytes(envelope).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            _fail("signed catalog is not JSON")
    node = _exact_keys(envelope, {"format", "payload", "signatures"}, "envelope", required=("format", "payload", "signatures"))
    if node["format"] != ENVELOPE_FORMAT:
        _fail(f"envelope format must be {ENVELOPE_FORMAT}")
    payload_text = node["payload"]
    if not isinstance(payload_text, str) or len(payload_text.encode("utf-8")) > MAX_ENVELOPE_BYTES:
        _fail("envelope payload must be a JSON string")
    message = payload_text.encode("utf-8")
    trusted = dict(keys) if keys is not None else trusted_keys()
    if not trusted:
        _fail("no pricing public key is pinned; signed catalogs are not trusted")
    signatures = node["signatures"]
    if not isinstance(signatures, list) or not signatures:
        _fail("envelope has no signatures")
    verified_kid = ""
    for index, entry in enumerate(signatures):
        item = _exact_keys(entry, {"kid", "alg", "sig"}, f"signatures[{index}]", required=("kid", "alg", "sig"))
        if item["alg"] != "Ed25519":
            continue
        public = trusted.get(str(item["kid"]))
        if public is None:
            continue
        if _verify_signature(public, message, _b64url_decode(item["sig"], f"signatures[{index}].sig")):
            verified_kid = str(item["kid"])
            break
    if not verified_kid:
        _fail("no signature verifies with a trusted pricing key")
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        _fail("signed payload is not JSON")
    if canonical_bytes(payload) != message:
        _fail("signed payload is not in canonical form")
    return payload, verified_kid, hashlib.sha256(message).hexdigest()


def sign_payload(payload: Mapping[str, Any], *, private_key_pem: bytes, kid: str) -> dict:
    """Envelope for ``payload`` signed with an Ed25519 PKCS#8 PEM key (tooling only)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if not _KID_RE.match(kid):
        raise CatalogDocumentError("kid must match [A-Za-z0-9_.-]{1,64}")
    parse_payload(payload)
    key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise CatalogDocumentError("the pricing signing key must be Ed25519")
    message = canonical_bytes(payload)
    return {
        "format": ENVELOPE_FORMAT,
        "payload": message.decode("utf-8"),
        "signatures": [{"kid": kid, "alg": "Ed25519", "sig": _b64url(key.sign(message))}],
    }


# --- activation ------------------------------------------------------------------

_HIGHEST_ACCEPTED: Optional[str] = None


def accept_envelope(
    envelope: Any,
    *,
    minimum_version: Optional[str] = None,
    now: Optional[datetime] = None,
    keys: Optional[Mapping[str, bytes]] = None,
    install: bool = True,
) -> CatalogBundle:
    """Verify, check freshness and ordering, and (by default) install a catalog.

    ``minimum_version`` is the highest version the caller has already accepted
    (for example from its disk cache or database); an equal version is accepted
    again (an idempotent reload), an older one is refused.
    """
    global _HIGHEST_ACCEPTED  # noqa: PLW0603
    payload, kid, digest = verify_envelope(envelope, keys=keys)
    bundle = parse_payload(payload, sha256=digest)
    moment = now or datetime.now(timezone.utc)
    if version_key(bundle.version) < version_key(BUNDLED.version):
        _fail(f"catalog {bundle.version} is older than the bundled snapshot {BUNDLED.version}")
    floor = [value for value in (minimum_version, _HIGHEST_ACCEPTED) if value]
    for seen in floor:
        if version_key(bundle.version) < version_key(seen):
            _fail(f"catalog {bundle.version} is older than accepted catalog {seen} (rollback refused)")
    if bundle.expired(moment):
        _fail(f"catalog {bundle.version} expired at {bundle.expires_at}")
    issued = parse_instant(bundle.issued_at)
    if issued is not None and issued - moment > ISSUED_AT_MAX_SKEW:
        _fail(f"catalog {bundle.version} is issued in the future")
    if install:
        install_live_bundle(bundle)
        if _HIGHEST_ACCEPTED is None or version_key(bundle.version) > version_key(_HIGHEST_ACCEPTED):
            _HIGHEST_ACCEPTED = bundle.version
        _log.info("price catalog %s installed (key %s, sha256 %s)", bundle.version, kid, digest[:16])
    return bundle


def reset_for_tests() -> None:
    global _HIGHEST_ACCEPTED  # noqa: PLW0603
    _HIGHEST_ACCEPTED = None
    install_live_bundle(None)
