from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Mapping, Optional, Tuple

from .model_pricing import MODEL_PRICING
from .price_catalog import (
    BUILTIN_ALIASES,
    CATALOG_VERSION,
    CatalogRow,
    PricingConfigError,
    _rates,
    cost_micro_usd,
    decimal_to_micro_usd,
    lookup_row,
    micro_usd_to_decimal,
    provider_for_model,
)
from .pricing_policy import (
    UnknownModelError,  # noqa: F401  (re-exported for callers)
    enforce_unknown_model,
    estimate_unknown_model_pricing,
    resolve_model_key,
    unknown_model_policy,
)

# Builtin exact aliases. Operator aliases are merged at engine init.
_ALIASES = BUILTIN_ALIASES

_log = logging.getLogger("kazenai.cost_engine")

_RATE_FIELDS = {
    "input",
    "cached_input",
    "cache_write_5m",
    "cache_write_1h",
    "output",
    "image",
    "tool_per_call",
}
_ROW_FIELDS = {
    "provider",
    "model",
    "request_tier",
    "context_band",
    "expected_reported_tier",
    "rates",
    "source",
    "verified_on",
    "long_context_threshold",
    "pricing_rule",
}
_TOP_FIELDS = {"currency", "unit", "source", "verified_on", "rows"}
_ALIAS_FORBIDDEN = set("*?[]{}()|+^$\\")


@dataclass(frozen=True)
class ModelPricing:
    input_per_1k: float
    output_per_1k: float


def _per_1k(row: CatalogRow) -> ModelPricing:
    return ModelPricing(
        input_per_1k=float(row.rates.input) / 1000.0,
        output_per_1k=float(row.rates.output) / 1000.0,
    )


_DEFAULT_PRICING: Dict[str, ModelPricing] = {
    key: ModelPricing(input_per_1k=inp / 1000.0, output_per_1k=out / 1000.0)
    for key, (inp, out) in MODEL_PRICING.items()
}
_DEFAULT_PRICING["ollama"] = ModelPricing(input_per_1k=0.0, output_per_1k=0.0)

_OPERATOR_ALIASES: Dict[str, str] = {}
_ACTIVE_CATALOG: Optional[Dict[tuple, CatalogRow]] = None


def operator_aliases() -> Dict[str, str]:
    return dict(_OPERATOR_ALIASES)


def active_catalog() -> Dict[tuple, CatalogRow]:
    if _ACTIVE_CATALOG is None:
        from .price_catalog import current_table

        return current_table()
    return dict(_ACTIVE_CATALOG)


def _fail(message: str) -> None:
    # Location and validation error only. Never include the raw JSON document.
    raise PricingConfigError(message)


def _parse_rate(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, float):
        try:
            number = Decimal(str(value))
        except Exception:
            _fail(f"KAZENAI_PRICING_JSON rate {field_name} is not a finite decimal")
        if not number.is_finite():
            _fail(f"KAZENAI_PRICING_JSON rate {field_name} is not finite")
    else:
        try:
            number = Decimal(str(value))
        except Exception:
            _fail(f"KAZENAI_PRICING_JSON rate {field_name} is not a finite decimal")
    if not number.is_finite() or number < 0:
        _fail(f"KAZENAI_PRICING_JSON rate {field_name} must be finite and >= 0")
    return format(number, "f")


def _load_pricing_json(raw: str) -> Dict[tuple, CatalogRow]:
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        _fail("KAZENAI_PRICING_JSON is not valid JSON")
    if not isinstance(document, dict):
        _fail("KAZENAI_PRICING_JSON must be an object")
    unknown = set(document) - _TOP_FIELDS
    if unknown:
        _fail("KAZENAI_PRICING_JSON contains an unknown field")
    currency = document.get("currency", "USD")
    unit = document.get("unit", "per_1m_tokens")
    if currency != "USD":
        _fail("KAZENAI_PRICING_JSON currency is not USD")
    if unit != "per_1m_tokens":
        _fail("KAZENAI_PRICING_JSON unit is not per_1m_tokens")
    source = str(document.get("source") or "").strip()
    verified = str(document.get("verified_on") or "").strip()
    if not source or not verified:
        _fail("KAZENAI_PRICING_JSON requires source and verified_on")
    rows = document.get("rows")
    if not isinstance(rows, list):
        _fail("KAZENAI_PRICING_JSON rows must be a list")
    overrides: Dict[tuple, CatalogRow] = {}
    for row in rows:
        if not isinstance(row, dict):
            _fail("KAZENAI_PRICING_JSON row is not an object")
        if set(row) - _ROW_FIELDS:
            _fail("KAZENAI_PRICING_JSON row contains an unknown field")
        rates = row.get("rates")
        if not isinstance(rates, dict) or set(rates) - _RATE_FIELDS or "input" not in rates or "output" not in rates:
            _fail("KAZENAI_PRICING_JSON rates are missing or contain an unknown field")
        try:
            vector = _rates(
                _parse_rate(rates.get("input"), "input") or "0",
                _parse_rate(rates.get("cached_input"), "cached_input"),
                _parse_rate(rates.get("cache_write_5m"), "cache_write_5m"),
                _parse_rate(rates.get("cache_write_1h"), "cache_write_1h"),
                _parse_rate(rates.get("output"), "output") or "0",
                _parse_rate(rates.get("image"), "image"),
                _parse_rate(rates.get("tool_per_call"), "tool_per_call"),
            )
        except PricingConfigError:
            raise
        except Exception:
            _fail("KAZENAI_PRICING_JSON rate is not a finite decimal")
        key = (
            str(row.get("provider") or "").strip().lower(),
            str(row.get("model") or "").strip().lower(),
            str(row.get("request_tier") or "standard").strip().lower(),
            str(row.get("context_band") or "short").strip().lower(),
        )
        if not key[0] or not key[1]:
            _fail("KAZENAI_PRICING_JSON row is missing provider or model")
        if key in overrides:
            _fail("KAZENAI_PRICING_JSON contains a duplicate canonical key")
        overrides[key] = CatalogRow(
            provider=key[0],
            model=key[1],
            request_tier=key[2],
            context_band=key[3],
            expected_reported_tier=str(row.get("expected_reported_tier") or "default"),
            rates=vector,
            certified=True,
            source_url=str(row.get("source") or source),
            verified_on=str(row.get("verified_on") or verified),
            long_context_threshold=row.get("long_context_threshold"),
            pricing_rule=str(row.get("pricing_rule") or "operator_override"),
            label="operator_override",
            catalog_version=CATALOG_VERSION,
        )
    return overrides


def _load_aliases(raw: str, catalog: Mapping[tuple, CatalogRow]) -> Dict[str, str]:
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        _fail("KAZENAI_MODEL_ALIASES_JSON is not valid JSON")
    if not isinstance(document, dict):
        _fail("KAZENAI_MODEL_ALIASES_JSON must be an object of exact ids")
    certified_models = {row.model for row in catalog.values() if row.certified and row.request_tier == "standard"}
    aliases: Dict[str, str] = {}
    for key, target in document.items():
        presented = str(key or "").strip().lower()
        if not presented or any(ch in presented for ch in _ALIAS_FORBIDDEN):
            _fail("KAZENAI_MODEL_ALIASES_JSON keys must be exact ids")
        canonical = str(target or "").strip().lower()
        if not canonical or any(ch in canonical for ch in _ALIAS_FORBIDDEN):
            _fail("KAZENAI_MODEL_ALIASES_JSON targets must be exact certified ids")
        if canonical not in certified_models:
            _fail("KAZENAI_MODEL_ALIASES_JSON target is not a certified catalog row")
        aliases[presented] = canonical
    return aliases


def _require_multiplier_when_control() -> None:
    raw = os.getenv("KAZENAI_TIER_MISMATCH_MULTIPLIER", "").strip()
    from .lifecycle_config import control_profile_active

    if not raw:
        if control_profile_active():
            _fail(
                "KAZENAI_TIER_MISMATCH_MULTIPLIER is required under Control "
                "(no verified built-in maximum for an unknown reported tier). "
                "Set it to a finite number >= 1; see "
                "https://docs.kazenai.com/reference/configuration/"
            )
        return
    try:
        number = Decimal(raw)
    except Exception:
        _fail("KAZENAI_TIER_MISMATCH_MULTIPLIER is not finite")
    if not number.is_finite() or number < 1:
        _fail("KAZENAI_TIER_MISMATCH_MULTIPLIER must be finite and >= 1")


class TokenCostEngine:
    """Provider-agnostic token to USD engine. Integer micro-USD is the ceiling."""

    def __init__(self, *, pricing: Optional[Dict[str, Any]] = None) -> None:
        global _OPERATOR_ALIASES
        _require_multiplier_when_control()
        from .price_catalog import active_bundle, current_table

        # The active catalog (a verified live catalog, else the bundled
        # snapshot), priced at this instant: scheduled prices applied and
        # expired rows uncertified.
        self._catalog: Dict[tuple, CatalogRow] = current_table()
        raw = os.getenv("KAZENAI_PRICING_JSON", "").strip()
        if raw:
            try:
                overrides = _load_pricing_json(raw)
            except PricingConfigError:
                _log.error("KAZENAI_PRICING_JSON rejected at engine init")
                raise
            for key, row in overrides.items():
                if key in self._catalog and self._catalog[key].label != "operator_override":
                    pass
                self._catalog[key] = row
        alias_raw = os.getenv("KAZENAI_MODEL_ALIASES_JSON", "").strip()
        aliases = dict(active_bundle().aliases)
        if alias_raw:
            aliases.update(_load_aliases(alias_raw, self._catalog))
        _OPERATOR_ALIASES = aliases
        self._aliases = aliases
        global _ACTIVE_CATALOG
        _ACTIVE_CATALOG = self._catalog

        merged: Dict[str, ModelPricing] = dict(_DEFAULT_PRICING)
        for row in self._catalog.values():
            if row.request_tier == "standard" and row.context_band == "short":
                merged[row.model] = _per_1k(row)
        if isinstance(pricing, dict):
            for key, value in pricing.items():
                if isinstance(value, ModelPricing):
                    merged[str(key)] = value
                elif isinstance(value, dict):
                    merged[str(key)] = ModelPricing(
                        input_per_1k=float(value.get("input_per_1k", 0.0) or 0.0),
                        output_per_1k=float(value.get("output_per_1k", 0.0) or 0.0),
                    )
        self._pricing = merged
        self._constructor_models = {str(key) for key in (pricing or {})}

    def _row(self, model: str, *, tier: str = "standard", band: str = "short") -> Optional[CatalogRow]:
        resolved = resolve_model_key(model, {row.model for row in self._catalog.values()}, self._aliases)
        if resolved is None:
            return None
        provider = provider_for_model(resolved)
        if provider is None:
            # Operator override may introduce a model the builtin map does not know.
            matches = [row for row in self._catalog.values() if row.model == resolved]
            providers = {row.provider for row in matches}
            provider = next(iter(providers)) if len(providers) == 1 else None
        if provider is None:
            return None
        return lookup_row(provider, resolved, tier, band, table=self._catalog)

    def price(self, model: str) -> ModelPricing:
        if model in getattr(self, "_constructor_models", ()):
            return self._pricing[model]
        row = self._row(model)
        if row is not None:
            from .lifecycle_config import control_profile_active

            if control_profile_active() and not row.certified:
                raise UnknownModelError(
                    f"uncertified model {model!r} is refused under Control"
                )
            return _per_1k(row)
        known = set(self._pricing)
        enforce_unknown_model(model, known_keys=known, aliases=self._aliases)
        if unknown_model_policy() == "estimate":
            inp, out = estimate_unknown_model_pricing()
            return ModelPricing(inp, out)
        _log.warning(
            "unknown model %r — cost will be reported as $0.00. "
            "Set KAZENAI_PRICING_JSON to add pricing or pin to a known model name.",
            model,
        )
        return ModelPricing(0.0, 0.0)

    def cost_micro_usd(self, *, model: str, input_tokens: int, output_tokens: int) -> int:
        row = self._row(model)
        if row is None:
            usd = Decimal(str(self.cost_usd(model=model, input_tokens=input_tokens, output_tokens=output_tokens)))
            return decimal_to_micro_usd(usd)
        return cost_micro_usd(
            row.rates,
            input_tokens=max(0, int(input_tokens)),
            output_tokens=max(0, int(output_tokens)),
        )

    def cost_usd(self, *, model: str, input_tokens: int, output_tokens: int) -> float:
        row = self._row(model)
        if row is None or model in getattr(self, "_constructor_models", ()):
            pricing = self.price(model)
            raw = (Decimal(str(pricing.input_per_1k)) * Decimal(max(0, int(input_tokens)))) + (
                Decimal(str(pricing.output_per_1k)) * Decimal(max(0, int(output_tokens)))
            )
            return float(micro_usd_to_decimal(decimal_to_micro_usd(raw)))
        return float(
            micro_usd_to_decimal(
                cost_micro_usd(
                    row.rates,
                    input_tokens=max(0, int(input_tokens)),
                    output_tokens=max(0, int(output_tokens)),
                )
            )
        )

    def from_usage(self, *, model: str, usage: Any, pricing: Any = None) -> Tuple[int, int, float]:
        from .request_pricing import settle_usage, usage_components

        components = usage_components(usage) if usage is not None else {
            "input_tokens": 0,
            "output_tokens": 0,
        }
        input_tokens = int(components.get("input_tokens") or 0) + int(components.get("cached_input_tokens") or 0)
        input_tokens += int(components.get("cache_write_5m_tokens") or 0) + int(components.get("cache_write_1h_tokens") or 0)
        output_tokens = int(components.get("output_tokens") or 0)
        if pricing is not None:
            micros = settle_usage(pricing, usage)
            return input_tokens, output_tokens, float(micro_usd_to_decimal(micros))
        row = self._row(model)
        if row is None:
            return input_tokens, output_tokens, float(self.cost_usd(model=model, input_tokens=input_tokens, output_tokens=output_tokens))
        # Frozen-basis callers pass pricing. Without it, use the current row
        # but still count cache components so a cache-only usage is not $0.
        micros = cost_micro_usd(row.rates, **{k: int(v) for k, v in components.items()})
        return input_tokens, output_tokens, float(micro_usd_to_decimal(micros))

    def known_models(self) -> list[str]:
        return sorted(self._pricing.keys())
