"""One versioned price ceiling for Core and FinOps.

Checked against official provider pages on 2026-10-17. Rows that page does not
support stay uncertified and are refused under Control.

This module is the bundled snapshot. A newer catalog signed with a pinned
KazenAI pricing key can replace it at runtime (``kazenai.catalog_document``);
the bundled snapshot is the default and the floor.

Installed SDK literals, read from the local environments on 2026-10-16.
No network call was used to discover a type literal.

- openai 1.109.1
  ``openai.types.chat.completion_create_params.CompletionCreateParams.service_tier``
  and ``openai.types.responses.response_create_params.ResponseCreateParams.service_tier``
  are ``Literal["auto", "default", "flex", "scale", "priority"]``.
  Unset requests are pinned to the literal ``default``.
  This 1.109.1 client has no ``responses.input_tokens.count``.
- anthropic 0.125.0
  ``anthropic.types.message_create_params.MessageCreateParams.service_tier``
  is ``Literal["auto", "standard_only"]``.
  Unset requests are pinned to the literal ``standard_only``.
  ``anthropic.types.usage.Usage.service_tier`` is
  ``Literal["standard", "priority", "batch"]``.
  Request ``standard_only`` reporting ``standard`` is the expected mapping.

OpenAI 2.x exact input counting is ``responses.input_tokens.count``. Ordinary
certified paths stay compatible with the 1.x floor. Paths that need that
counter refuse before dispatch when the installed client lacks it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_EVEN
from typing import Dict, Iterable, Mapping, Optional, Tuple

CATALOG_VERSION = "2026-10-17"
VERIFIED_ON = "2026-10-17"
ROUNDING_RULE = "decimal_half_even_to_nanos_then_ceil_micro_usd"

OPENAI_SOURCE = "https://developers.openai.com/api/docs/pricing"
ANTHROPIC_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
GOOGLE_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"
DEEPSEEK_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing"
OPENAI_VISION_SOURCE = "https://developers.openai.com/api/docs/guides/images-vision"
ANTHROPIC_VISION_SOURCE = "https://platform.claude.com/docs/en/build-with-claude/vision"
ANTHROPIC_TOOLS_SOURCE = ANTHROPIC_SOURCE

# Client-tool version strings copied from the Anthropic pricing page
# (tool-use section), which names these exact type literals. Web search,
# web fetch, and code execution are documented there and are refused.
ANTHROPIC_CLIENT_TOOL_VERSIONS = (
    "text_editor_20250728",
    "computer_toolset_20260801",
    "computer_20251124",
    "computer_20250124",
    "browser_toolset_20260801",
)

OPENAI_PIN_REQUEST_TIER = "default"
ANTHROPIC_PIN_REQUEST_TIER = "standard_only"
OPENAI_EXPECTED_REPORTED = "default"
ANTHROPIC_EXPECTED_REPORTED = "standard"

# OpenAI long context is strictly above 272K input tokens (pricing page).
OPENAI_LONG_THRESHOLD = 272_000
# Claude Haiku 5.5 long context is strictly over 100,000 prompt tokens.
ANTHROPIC_HAIKU_55_THRESHOLD = 100_000
# Gemini 2.5 Pro long context is strictly over 200,000 prompt tokens.
GEMINI_25_PRO_THRESHOLD = 200_000

UNKNOWN_CHARGE_AFTER_DAYS = 35


class PricingConfigError(ValueError):
    """Malformed operator pricing or alias configuration. Startup must fail."""


def decimal_to_micro_usd(amount: Decimal) -> int:
    """The one Decimal-to-micro-USD rule.

    Half-even to nanos, then ceiling to integer micro-USD. Pre-call bounds
    and settlement both use this. Binary float is not a budget decision.
    """
    if not isinstance(amount, Decimal):
        amount = Decimal(amount)
    if not amount.is_finite():
        raise PricingConfigError("non-finite USD amount")
    if amount < 0:
        raise PricingConfigError("negative USD amount")
    nanos = amount.quantize(Decimal("0.000000001"), rounding=ROUND_HALF_EVEN)
    return int((nanos * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING))


def micro_usd_to_decimal(micros: int) -> Decimal:
    return Decimal(int(micros)) / Decimal(1_000_000)


def _d(value: Optional[str]) -> Optional[Decimal]:
    if value is None:
        return None
    number = Decimal(value)
    if not number.is_finite() or number < 0:
        raise PricingConfigError("rate must be finite and >= 0")
    return number


@dataclass(frozen=True)
class RateVector:
    """USD per 1M tokens, except ``tool_per_call`` which is USD per call.

    ``None`` means the provider page does not publish that component. Settlement
    then bills the component at the uncached input rate so a missing discount
    cannot under-charge. A published zero is free.
    """

    input: Decimal
    cached_input: Optional[Decimal]
    cache_write_5m: Optional[Decimal]
    cache_write_1h: Optional[Decimal]
    output: Decimal
    image: Optional[Decimal]
    tool_per_call: Optional[Decimal]
    # Audio input where the provider prices it apart from text (Gemini 2.5).
    # ``None``: audio tokens bill at the text input rate.
    audio_input: Optional[Decimal] = None
    audio_cached_input: Optional[Decimal] = None

    def as_json(self) -> dict:
        def _s(value: Optional[Decimal]) -> Optional[str]:
            return None if value is None else format(value, "f")

        out = {
            "input": format(self.input, "f"),
            "cached_input": _s(self.cached_input),
            "cache_write_5m": _s(self.cache_write_5m),
            "cache_write_1h": _s(self.cache_write_1h),
            "output": format(self.output, "f"),
            "image": _s(self.image),
            "tool_per_call": _s(self.tool_per_call),
        }
        # Only when published, so rows without audio keep their rate-card hash.
        if self.audio_input is not None:
            out["audio_input"] = _s(self.audio_input)
        if self.audio_cached_input is not None:
            out["audio_cached_input"] = _s(self.audio_cached_input)
        return out

    def scaled(self, multiplier: Decimal) -> "RateVector":
        """Every published component times ``multiplier`` (data residency, regional endpoints)."""
        if multiplier == 1:
            return self

        def _m(value: Optional[Decimal]) -> Optional[Decimal]:
            return None if value is None else value * multiplier

        return RateVector(
            input=self.input * multiplier,
            cached_input=_m(self.cached_input),
            cache_write_5m=_m(self.cache_write_5m),
            cache_write_1h=_m(self.cache_write_1h),
            output=self.output * multiplier,
            image=_m(self.image),
            tool_per_call=_m(self.tool_per_call),
            audio_input=_m(self.audio_input),
            audio_cached_input=_m(self.audio_cached_input),
        )


def _rates(
    inp: str,
    cached: Optional[str],
    write_5m: Optional[str],
    write_1h: Optional[str],
    out: str,
    image: Optional[str] = None,
    tool: Optional[str] = None,
    *,
    audio: Optional[str] = None,
    audio_cached: Optional[str] = None,
) -> RateVector:
    return RateVector(
        input=_d(inp),  # type: ignore[arg-type]
        cached_input=_d(cached),
        cache_write_5m=_d(write_5m),
        cache_write_1h=_d(write_1h),
        output=_d(out),  # type: ignore[arg-type]
        image=_d(image),
        tool_per_call=_d(tool),
        audio_input=_d(audio),
        audio_cached_input=_d(audio_cached),
    )


def parse_instant(value: Optional[str]) -> Optional[datetime]:
    """ISO-8601 UTC instant (``2027-01-01T00:00:00Z``) or ``None``."""
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise PricingConfigError(f"invalid instant {value!r}") from exc
    if moment.tzinfo is None:
        raise PricingConfigError(f"instant {value!r} must carry a UTC offset")
    return moment.astimezone(timezone.utc)


@dataclass(frozen=True)
class CatalogRow:
    provider: str
    model: str
    request_tier: str
    context_band: str
    expected_reported_tier: str
    rates: RateVector
    certified: bool
    source_url: str
    verified_on: str
    long_context_threshold: Optional[int]
    pricing_rule: str
    label: str = "catalog"
    catalog_version: str = CATALOG_VERSION
    # Published future prices: ((effective_from ISO instant, RateVector), ...).
    # The latest entry whose instant has passed replaces ``rates``.
    scheduled: Tuple[Tuple[str, RateVector], ...] = ()
    # Promotional or otherwise time-limited prices. At or after this instant the
    # row is uncertified (refused under Control) until a newer catalog says more.
    valid_until: Optional[str] = None

    @property
    def key(self) -> Tuple[str, str, str, str]:
        return (self.provider, self.model, self.request_tier, self.context_band)

    def at(self, moment: Optional[datetime] = None) -> "CatalogRow":
        """This row as priced at ``moment`` (default now, UTC)."""
        if not self.scheduled and self.valid_until is None:
            return self
        now = moment or datetime.now(timezone.utc)
        rates = self.rates
        for effective_from, scheduled_rates in sorted(self.scheduled, key=lambda item: parse_instant(item[0]) or now):
            starts = parse_instant(effective_from)
            if starts is not None and starts <= now:
                rates = scheduled_rates
        certified = self.certified
        ends = parse_instant(self.valid_until)
        if ends is not None and now >= ends:
            certified = False
        if rates is self.rates and certified == self.certified:
            return self
        return replace(self, rates=rates, certified=certified)


def _row(
    provider: str,
    model: str,
    tier: str,
    band: str,
    reported: str,
    rates: RateVector,
    source: str,
    *,
    threshold: Optional[int] = None,
    rule: str = "standard_token",
    certified: bool = True,
    label: str = "catalog",
    version: str = CATALOG_VERSION,
    verified: str = VERIFIED_ON,
    scheduled: Tuple[Tuple[str, RateVector], ...] = (),
    valid_until: Optional[str] = None,
) -> CatalogRow:
    return CatalogRow(
        provider=provider,
        model=model,
        request_tier=tier,
        context_band=band,
        expected_reported_tier=reported,
        rates=rates,
        certified=certified,
        source_url=source,
        verified_on=verified,
        long_context_threshold=threshold,
        pricing_rule=rule,
        label=label,
        catalog_version=version,
        scheduled=scheduled,
        valid_until=valid_until,
    )


# GPT-5.6 Sol's promotional price is published "at least through November 21,
# 2026" (pricing page, checked 2026-10-17). After that the row is uncertified
# until a newer catalog confirms the price; it is never silently kept.
GPT_56_SOL_PROMO_VALID_UNTIL = "2026-11-22T00:00:00Z"
# Gemini 3.6 Flash and 3.8 Flash double "starting January 1, 2027" (pricing page,
# checked 2026-10-17). Applied one day early so no time zone can under-price.
GEMINI_FLASH_2027_EFFECTIVE = "2026-12-31T00:00:00Z"

# (model, in, cached, write, out, long_in, long_cached, long_write, long_out)
# long_* None means the page has no long-context price band.
_OPENAI_STANDARD: tuple = (
    ("gpt-6-astra", "10", "1", "12.50", "50", "20", "2", "25", "75"),
    ("gpt-6.1-sol", "2", "0.10", "2.50", "10", "4", "0.20", "5", "15"),
    ("gpt-6-sol", "2", "0.20", "2.50", "10", "4", "0.40", "5", "15"),
    ("gpt-6-luna", "0.10", "0.01", "0.125", "0.50", "0.20", "0.02", "0.25", "0.75"),
    ("gpt-5.6-sol", "4", "0.40", "5", "20", "8", "0.80", "10", "30"),
    ("gpt-5.6-terra", "2", "0.20", "2.50", "12", "4", "0.40", "5", "18"),
    ("gpt-5.6-luna", "0.20", "0.02", "0.25", "1.20", "0.40", "0.04", "0.50", "1.80"),
    ("gpt-5.5", "5", "0.50", None, "30", "10", "1", None, "45"),
    ("gpt-5.5-pro", "30", None, None, "180", "60", None, None, "270"),
    ("gpt-5.4", "2.50", "0.25", None, "15", "5", "0.50", None, "22.50"),
    ("gpt-5.4-mini", "0.75", "0.075", None, "4.50", None, None, None, None),
    ("gpt-5.4-nano", "0.20", "0.02", None, "1.25", None, None, None, None),
    ("gpt-5.4-pro", "30", None, None, "180", "60", None, None, "270"),
    ("gpt-5.2", "1.75", "0.175", None, "14", None, None, None, None),
    ("gpt-5.2-pro", "21", None, None, "168", None, None, None, None),
    ("gpt-5.1", "1.25", "0.125", None, "10", None, None, None, None),
    ("gpt-5", "1.25", "0.125", None, "10", None, None, None, None),
    ("gpt-5-mini", "0.25", "0.025", None, "2", None, None, None, None),
    ("gpt-5-nano", "0.05", "0.005", None, "0.40", None, None, None, None),
    ("gpt-5-pro", "15", None, None, "120", None, None, None, None),
    ("gpt-4.1", "2", "0.50", None, "8", None, None, None, None),
    ("gpt-4.1-mini", "0.40", "0.10", None, "1.60", None, None, None, None),
    ("gpt-4.1-nano", "0.10", "0.025", None, "0.40", None, None, None, None),
    ("gpt-4o", "2.50", "1.25", None, "10", None, None, None, None),
    ("gpt-4o-2024-05-13", "5", None, None, "15", None, None, None, None),
    ("gpt-4o-mini", "0.15", "0.075", None, "0.60", None, None, None, None),
    ("o1", "15", "7.50", None, "60", None, None, None, None),
    ("o1-pro", "150", None, None, "600", None, None, None, None),
    ("o3", "2", "0.50", None, "8", None, None, None, None),
    ("o3-pro", "20", None, None, "80", None, None, None, None),
    ("o3-mini", "1.10", "0.55", None, "4.40", None, None, None, None),
    ("o4-mini", "1.10", "0.275", None, "4.40", None, None, None, None),
)

# Anthropic: in, write 5m, write 1h, cache read, out. Read is the published
# hit price (0.1x except where the page states 0.025x or 0.05x).
_ANTHROPIC_STANDARD: tuple = (
    ("claude-fable-5-1", "10", "12.50", "20", "0.25", "50"),
    ("claude-fable-5", "10", "12.50", "20", "1", "50"),
    ("claude-opus-5-5", "4", "5", "8", "0.20", "20"),
    ("claude-opus-5", "5", "6.25", "10", "0.50", "25"),
    ("claude-opus-4-8", "5", "6.25", "10", "0.50", "25"),
    ("claude-opus-4-7", "5", "6.25", "10", "0.50", "25"),
    ("claude-opus-4-6", "5", "6.25", "10", "0.50", "25"),
    ("claude-opus-4-5", "5", "6.25", "10", "0.50", "25"),
    ("claude-opus-4-1", "15", "18.75", "30", "1.50", "75"),
    ("claude-sonnet-5-5", "2", "2.50", "4", "0.10", "10"),
    ("claude-sonnet-5", "2", "2.50", "4", "0.20", "10"),
    ("claude-sonnet-4-6", "3", "3.75", "6", "0.30", "15"),
    ("claude-sonnet-4-5", "3", "3.75", "6", "0.30", "15"),
    ("claude-haiku-4-5", "1", "1.25", "2", "0.10", "5"),
)


def _openai_rows() -> Iterable[CatalogRow]:
    for model, inp, cached, write, out, lin, lc, lw, lo in _OPENAI_STANDARD:
        threshold = OPENAI_LONG_THRESHOLD if lin is not None else None
        valid_until = GPT_56_SOL_PROMO_VALID_UNTIL if model == "gpt-5.6-sol" else None
        yield _row(
            "openai",
            model,
            "standard",
            "short",
            OPENAI_EXPECTED_REPORTED,
            _rates(inp, cached, write, None, out),
            OPENAI_SOURCE,
            threshold=threshold,
            rule="openai_standard_short",
            valid_until=valid_until,
        )
        if lin is not None:
            yield _row(
                "openai",
                model,
                "standard",
                "long",
                OPENAI_EXPECTED_REPORTED,
                _rates(lin, lc, lw, None, lo),
                OPENAI_SOURCE,
                threshold=threshold,
                rule="openai_standard_long",
                valid_until=valid_until,
            )
    # Certified unexpected reported tiers for gpt-4o. Priority processing was
    # renamed Fast on the pricing page (2026-07-30); the installed SDK still
    # reports the literal "priority". Batch numbers are the Batch table.
    yield _row(
        "openai",
        "gpt-4o",
        "priority",
        "short",
        "priority",
        _rates("4.25", "2.125", None, None, "17"),
        OPENAI_SOURCE,
        rule="openai_fast_reported_priority",
    )
    yield _row(
        "openai",
        "gpt-4o",
        "batch",
        "short",
        "batch",
        _rates("1.25", None, None, None, "5"),
        OPENAI_SOURCE,
        rule="openai_batch",
    )
    yield _row(
        "openai",
        "gpt-4o-mini",
        "priority",
        "short",
        "priority",
        _rates("0.25", "0.125", None, None, "1"),
        OPENAI_SOURCE,
        rule="openai_fast_reported_priority",
    )
    yield _row(
        "openai",
        "gpt-4o-mini",
        "batch",
        "short",
        "batch",
        _rates("0.075", None, None, None, "0.30"),
        OPENAI_SOURCE,
        rule="openai_batch",
    )


def _anthropic_rows() -> Iterable[CatalogRow]:
    for model, inp, w5, w1, read, out in _ANTHROPIC_STANDARD:
        yield _row(
            "anthropic",
            model,
            "standard",
            "short",
            ANTHROPIC_EXPECTED_REPORTED,
            _rates(inp, read, w5, w1, out),
            ANTHROPIC_SOURCE,
            rule="anthropic_standard",
        )
    yield _row(
        "anthropic",
        "claude-haiku-5-5",
        "standard",
        "short",
        ANTHROPIC_EXPECTED_REPORTED,
        _rates("0.10", "0.01", "0.125", "0.20", "0.50"),
        ANTHROPIC_SOURCE,
        threshold=ANTHROPIC_HAIKU_55_THRESHOLD,
        rule="anthropic_haiku_5_5_short",
    )
    yield _row(
        "anthropic",
        "claude-haiku-5-5",
        "standard",
        "long",
        ANTHROPIC_EXPECTED_REPORTED,
        _rates("0.50", "0.05", "0.625", "1", "2.50"),
        ANTHROPIC_SOURCE,
        threshold=ANTHROPIC_HAIKU_55_THRESHOLD,
        rule="anthropic_haiku_5_5_long",
    )
    # Batch table is half of standard. Cache multipliers stack with batch
    # (same pricing page): 1.25x / 2x / published read ratio.
    for model, inp, w5, w1, read, out in (
        ("claude-sonnet-4-6", "1.50", "1.875", "3", "0.15", "7.50"),
        ("claude-haiku-4-5", "0.50", "0.625", "1", "0.05", "2.50"),
    ):
        yield _row(
            "anthropic",
            model,
            "batch",
            "short",
            "batch",
            _rates(inp, read, w5, w1, out),
            ANTHROPIC_SOURCE,
            rule="anthropic_batch",
        )


def _google_rows() -> Iterable[CatalogRow]:
    # (model, in, cached, out, threshold, audio_in, audio_cached). Audio input is
    # priced apart from text on 2.5 Flash and 2.5 Flash-Lite (pricing page).
    text = (
        ("gemini-3.8-flash", "0.75", "0.075", "3.75", None, None, None),
        ("gemini-3.6-flash", "0.75", "0.075", "3.75", None, None, None),
        ("gemini-3.5-flash-lite", "0.30", "0.03", "2.50", None, None, None),
        ("gemini-2.5-flash", "0.30", "0.03", "2.50", None, "1.00", "0.10"),
        ("gemini-2.5-flash-lite", "0.10", "0.01", "0.40", None, "0.30", "0.03"),
        ("gemini-2.5-pro", "1.25", "0.125", "10", GEMINI_25_PRO_THRESHOLD, None, None),
    )
    scheduled_2027 = {
        "gemini-3.8-flash": ((GEMINI_FLASH_2027_EFFECTIVE, _rates("1.50", "0.15", None, None, "7.50")),),
        "gemini-3.6-flash": ((GEMINI_FLASH_2027_EFFECTIVE, _rates("1.50", "0.15", None, None, "7.50")),),
    }
    for model, inp, cached, out, threshold, audio, audio_cached in text:
        yield _row(
            "google",
            model,
            "standard",
            "short",
            "standard",
            _rates(inp, cached, None, None, out, audio=audio, audio_cached=audio_cached),
            GOOGLE_SOURCE,
            threshold=threshold,
            rule="gemini_text_standard",
            scheduled=scheduled_2027.get(model, ()),
        )
    yield _row(
        "google",
        "gemini-2.5-pro",
        "standard",
        "long",
        "standard",
        _rates("2.50", "0.25", None, None, "15"),
        GOOGLE_SOURCE,
        threshold=GEMINI_25_PRO_THRESHOLD,
        rule="gemini_2_5_pro_long",
    )


def _deepseek_rows() -> Iterable[CatalogRow]:
    # Peak-hour rates. A ceiling must not reserve the cheaper off-peak price.
    for model, miss, hit, out in (
        ("deepseek-flash", "0.30", "0.006", "1.20"),
        ("deepseek-v4-pro", "1.32", "0.044", "3.96"),
    ):
        yield _row(
            "deepseek",
            model,
            "standard",
            "short",
            "standard",
            _rates(miss, hit, None, None, out),
            DEEPSEEK_SOURCE,
            rule="deepseek_peak",
        )


def _build_catalog() -> Dict[Tuple[str, str, str, str], CatalogRow]:
    table: Dict[Tuple[str, str, str, str], CatalogRow] = {}
    for row in (
        *_openai_rows(),
        *_anthropic_rows(),
        *_google_rows(),
        *_deepseek_rows(),
    ):
        if row.key in table:
            raise PricingConfigError(f"duplicate catalog key {row.key!r}")
        table[row.key] = row
    return table


PRICE_CATALOG: Dict[Tuple[str, str, str, str], CatalogRow] = _build_catalog()

# Exact snapshot / legacy ids the provider page says bill as another row.
# No prefixes. deepseek-v4-flash and deepseek-v4-flash-vision-exp are served
# as deepseek-flash at the Flash price (DeepSeek pricing page).
BUILTIN_ALIASES: Dict[str, str] = {
    "deepseek-v4-flash": "deepseek-flash",
    "deepseek-v4-flash-vision-exp": "deepseek-flash",
    # Anthropic pinned snapshots behind the pre-4.6 aliases. Each model page
    # lists "Claude API: <snapshot>" and "Claude API alias: <alias>" with the
    # same prices (checked 2026-10-16):
    #   https://platform.claude.com/docs/en/models/haiku-4-5/overview
    #   https://platform.claude.com/docs/en/models/sonnet-4-5/overview
    #   https://platform.claude.com/docs/en/models/opus-4-5/overview
    "claude-haiku-4-5-20251001": "claude-haiku-4-5",
    "claude-sonnet-4-5-20250929": "claude-sonnet-4-5",
    "claude-opus-4-5-20251101": "claude-opus-4-5",
    # OpenAI snapshots listed under the model's "Snapshots" section, which
    # publishes one text-token price for the model (checked 2026-10-16):
    #   https://developers.openai.com/api/docs/models/<model>
    # gpt-4o-2024-05-13 is priced separately on the pricing page and keeps its row.
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4o-2024-11-20": "gpt-4o",
    "gpt-4.1-2025-04-14": "gpt-4.1",
    "gpt-4.1-mini-2025-04-14": "gpt-4.1-mini",
    "gpt-4.1-nano-2025-04-14": "gpt-4.1-nano",
    "gpt-5-mini-2025-08-07": "gpt-5-mini",
}

PRICING_SOURCES: Dict[str, str] = {
    "openai": OPENAI_SOURCE,
    "anthropic": ANTHROPIC_SOURCE,
    "google": GOOGLE_SOURCE,
    "deepseek": DEEPSEEK_SOURCE,
}


def iter_rows() -> Iterable[CatalogRow]:
    return PRICE_CATALOG.values()


def certified_rows() -> list[CatalogRow]:
    return [row for row in PRICE_CATALOG.values() if row.certified]


def uncertified_rows() -> list[CatalogRow]:
    return [row for row in PRICE_CATALOG.values() if not row.certified]


def standard_short_rates() -> Dict[str, Tuple[float, float]]:
    """Input/output USD per 1M for the pinned standard short row.

    Compatibility view for callers that still read a two-number table.
    """
    out: Dict[str, Tuple[float, float]] = {}
    for row in PRICE_CATALOG.values():
        if row.request_tier == "standard" and row.context_band == "short":
            out[row.model] = (float(row.rates.input), float(row.rates.output))
    return out


def provider_for_model(model: str, *, table: Optional[Mapping[Tuple[str, str, str, str], CatalogRow]] = None) -> Optional[str]:
    catalog = table if table is not None else current_table()
    key = str(model or "").strip().lower()
    found = {row.provider for row in catalog.values() if row.model == key}
    if len(found) == 1:
        return next(iter(found))
    return None


def canonical_model_id(model: str) -> str:
    """Resolve a snapshot alias to the catalog id. Unknown ids are returned unchanged."""
    key = str(model or "").strip()
    return active_bundle().aliases.get(key, key)


def catalog_row_hash(row: CatalogRow) -> str:
    """Stable identity of one server rate card. The client cannot supply this."""
    import hashlib
    import json

    payload = {
        "provider": row.provider,
        "model": row.model,
        "request_tier": row.request_tier,
        "context_band": row.context_band,
        "catalog_version": row.catalog_version,
        "rates": row.rates.as_json(),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def lookup_row(
    provider: str,
    model: str,
    request_tier: str,
    context_band: str,
    *,
    table: Optional[Mapping[Tuple[str, str, str, str], CatalogRow]] = None,
) -> Optional[CatalogRow]:
    catalog = table if table is not None else current_table()
    return catalog.get((provider, model, request_tier, context_band))


def model_declares_long_band(
    provider: str,
    model: str,
    request_tier: str = "standard",
    *,
    table: Optional[Mapping[Tuple[str, str, str, str], CatalogRow]] = None,
) -> bool:
    return lookup_row(provider, model, request_tier, "long", table=table) is not None


def effective_component(rate: Optional[Decimal], input_rate: Decimal) -> Decimal:
    return input_rate if rate is None else rate


def component_cost(
    rates: RateVector,
    *,
    input_tokens: int = 0,
    cached_input_tokens: int = 0,
    cache_write_5m_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
    output_tokens: int = 0,
    image_tokens: int = 0,
    tool_calls: int = 0,
    audio_input_tokens: int = 0,
    audio_cached_input_tokens: int = 0,
) -> Decimal:
    """USD as Decimal. Image tokens use the image rate when published, else input.

    Audio tokens are counted apart from ``input_tokens`` and bill at the audio
    rate when the row publishes one, else at the text rate.
    """
    uncached = max(0, int(input_tokens))
    cached = max(0, int(cached_input_tokens))
    w5 = max(0, int(cache_write_5m_tokens))
    w1 = max(0, int(cache_write_1h_tokens))
    output = max(0, int(output_tokens))
    images = max(0, int(image_tokens))
    calls = max(0, int(tool_calls))
    million = Decimal(1_000_000)
    image_rate = rates.image if rates.image is not None else rates.input
    tool_rate = rates.tool_per_call if rates.tool_per_call is not None else Decimal(0)
    total = (
        Decimal(uncached) * rates.input
        + Decimal(cached) * effective_component(rates.cached_input, rates.input)
        + Decimal(w5) * effective_component(rates.cache_write_5m, rates.input)
        + Decimal(w1) * effective_component(rates.cache_write_1h, rates.input)
        + Decimal(output) * rates.output
        + Decimal(images) * image_rate
    ) / million
    total += Decimal(calls) * tool_rate
    audio = max(0, int(audio_input_tokens))
    audio_cached = max(0, int(audio_cached_input_tokens))
    if audio or audio_cached:
        audio_rate = rates.audio_input if rates.audio_input is not None else rates.input
        audio_cached_rate = (
            rates.audio_cached_input
            if rates.audio_cached_input is not None
            else effective_component(rates.cached_input, rates.input)
            if rates.audio_input is None
            else audio_rate
        )
        total += (Decimal(audio) * audio_rate + Decimal(audio_cached) * audio_cached_rate) / million
    return total


def cost_micro_usd(rates: RateVector, **tokens: int) -> int:
    return decimal_to_micro_usd(component_cost(rates, **tokens))


# --- Anthropic tool overhead (pricing page, tool-use section, checked 2026-10-17)
#
# With any ``tools`` the API adds a tool-use system prompt (per model and
# tool_choice), and some client tools add definition tokens on top. The byte
# estimate cannot see either, so the pre-call bound adds them. "About" figures
# on the page carry a 10% margin; exact figures are copied as published.
ANTHROPIC_TOOL_SYSTEM_PROMPT_TOKENS: Dict[str, Tuple[int, int]] = {
    # model: (tool_choice auto/none, tool_choice any/tool)
    "claude-opus-5-5": (286, 286),
    "claude-opus-5": (286, 406),
    "claude-opus-4-8": (290, 410),
    "claude-opus-4-7": (675, 804),
    "claude-opus-4-6": (497, 589),
    "claude-opus-4-5": (496, 588),
    "claude-opus-4-1": (313, 315),
    "claude-sonnet-5-5": (286, 286),
    "claude-sonnet-5": (354, 474),
    "claude-sonnet-4-6": (497, 589),
    "claude-sonnet-4-5": (496, 588),
    "claude-haiku-5-5": (286, 406),
    "claude-haiku-4-5": (496, 588),
}
# A model the table does not list gets the largest published figure.
ANTHROPIC_TOOL_SYSTEM_PROMPT_FALLBACK = 804
ANTHROPIC_TOOL_DEFINITION_TOKENS: Dict[str, int] = {
    # 974 on Claude 4.7+ (745 on earlier models); the larger figure everywhere.
    "text_editor_20250728": 974,
    # "about 4,590 on Claude Sonnet 5, about 4,520 on the other supported models"
    "computer_toolset_20260801": 5_050,
    # "about 6,670 on Claude Sonnet 5 and about 6,610 on the other supported models"
    "browser_toolset_20260801": 7_340,
    # Earlier computer versions: 466-499 system prompt + about 735 per definition.
    "computer_20251124": 1_310,
    "computer_20250124": 1_310,
}


def anthropic_tool_overhead_tokens(model: str, tools: Iterable[object], tool_choice: object = None) -> int:
    """Input tokens the API adds for ``tools`` beyond their JSON bytes."""
    listed = [tool for tool in (tools or ()) if tool is not None]
    if not listed:
        return 0
    bundle = active_bundle()
    pair = bundle.anthropic_tool_system_prompt_tokens.get(str(model or "").strip().lower())
    choice = tool_choice.get("type") if isinstance(tool_choice, Mapping) else getattr(tool_choice, "type", tool_choice)
    forced = str(choice or "auto").strip().lower() in {"any", "tool"}
    if pair is None:
        total = bundle.anthropic_tool_system_prompt_fallback
    else:
        total = pair[1] if forced else pair[0]
    for tool in listed:
        tool_type = tool.get("type") if isinstance(tool, Mapping) else getattr(tool, "type", None)
        total += int(bundle.anthropic_tool_definition_tokens.get(str(tool_type or ""), 0))
    return int(total)


# --- request price modifiers (pricing pages, checked 2026-10-17) ------------
#
# Anthropic ``inference_geo: "us"``: 1.1x on every token category for Claude 4.6
# and later; applied to every model as a ceiling. ``speed: "fast"`` is premium
# fast mode: refused under Control until a catalog certifies its rate card.
# OpenAI regional data-residency domains (``<cc>.api.openai.com``) and FedRAMP
# endpoints carry a 10% uplift; Anthropic regional cloud endpoints carry a 10%
# premium. Any host other than the first-party API is priced with that uplift.
ANTHROPIC_INFERENCE_GEO_MULTIPLIERS: Dict[str, Decimal] = {"global": Decimal("1"), "us": Decimal("1.1")}
ANTHROPIC_ALLOWED_SPEEDS = ("standard",)
FIRST_PARTY_API_HOSTS: Dict[str, Tuple[str, ...]] = {
    "openai": ("api.openai.com",),
    "anthropic": ("api.anthropic.com",),
}
NON_FIRST_PARTY_UPLIFT = Decimal("1.1")


# --- image token formulas copied from the fetched vision pages ---------------

# Tile models: base + tiles. low = base only. high/auto follow the 2048/768/512 rule.
# gpt-4o-mini base 2833 / tile 5667; gpt-4o and gpt-4.1 base 85 / tile 170.
_TILE_MODELS = {
    "gpt-5.1": (70, 140),
    "gpt-5": (70, 140),
    "gpt-4o": (85, 170),
    "gpt-4.1": (85, 170),
    "gpt-4o-mini": (2833, 5667),
    "o1": (75, 150),
    "o1-pro": (75, 150),
    "o3": (75, 150),
}

# Patch models: 32px patches, then a multiplier, ceil. detail=auto uses the
# most expensive supported detail (price ceiling), which can be `low` when the
# page says low has the larger patch budget.
_PATCH_MULTIPLIER = {
    "gpt-6-astra": Decimal("1.2"),
    "gpt-5.6-sol": Decimal("1.2"),
    "gpt-5.6-terra": Decimal("1.2"),
    "gpt-5.6-luna": Decimal("1.2"),
    "gpt-5.5": Decimal("1.2"),
    "gpt-5.4": Decimal("1.2"),
    "gpt-5.4-mini": Decimal("1.2"),
    "gpt-5.4-nano": Decimal("1.2"),
    "gpt-5.2": Decimal("1.2"),
    "gpt-5-mini": Decimal("1.2"),
    "gpt-5-nano": Decimal("1.5"),
    "gpt-4.1-mini": Decimal("1.62"),
    "gpt-4.1-nano": Decimal("2.46"),
    "o4-mini": Decimal("1.72"),
}


def _fit(width: int, height: int, max_edge: int) -> tuple[int, int]:
    long_edge = max(width, height)
    if long_edge <= max_edge or long_edge <= 0:
        return width, height
    scale = Decimal(max_edge) / Decimal(long_edge)
    return (
        max(1, int((Decimal(width) * scale).to_integral_value(rounding=ROUND_HALF_EVEN))),
        max(1, int((Decimal(height) * scale).to_integral_value(rounding=ROUND_HALF_EVEN))),
    )


def _tile_tokens(width: int, height: int, base: int, tile: int) -> int:
    w, h = _fit(width, height, 2048)
    short = min(w, h)
    if short > 768:
        if w <= h:
            w = 768
            h = max(1, int(h * 768 / short))
        else:
            h = 768
            w = max(1, int(w * 768 / short))
    import math

    tiles = math.ceil(w / 512) * math.ceil(h / 512)
    return int(base + tile * tiles)


def _patches(width: int, height: int) -> int:
    import math

    return math.ceil(width / 32) * math.ceil(height / 32)


def _patch_tokens(width: int, height: int, *, max_edge: int, budget: Optional[int], multiplier: Decimal) -> int:
    import math

    w, h = _fit(max(1, width), max(1, height), max_edge)
    if budget is not None and w * h > 0 and _patches(w, h) > budget:
        shrink = (Decimal(budget) * Decimal(32) * Decimal(32) / Decimal(w * h)).sqrt()
        w = max(1, int((Decimal(w) * shrink).to_integral_value(rounding=ROUND_HALF_EVEN)))
        h = max(1, int((Decimal(h) * shrink).to_integral_value(rounding=ROUND_HALF_EVEN)))
        # Stay inside the budget after integer pixels.
        while budget is not None and _patches(w, h) > budget and (w > 1 or h > 1):
            if w >= h and w > 1:
                w -= 1
            elif h > 1:
                h -= 1
            else:
                break
    count = _patches(w, h)
    return int(math.ceil(Decimal(count) * multiplier))


# Patch sizing per model and detail (vision guide, checked 2026-10-17):
# (max long edge in px, patch budget or None). The API rejects more than 30,000
# patches, so that is the ceiling where the page gives no budget.
_PATCH_REJECT_ABOVE = 30_000
_PATCH_SIZING: Dict[str, Dict[str, Tuple[int, Optional[int]]]] = {
    "gpt-6-astra": {"low": (512, None), "high": (65_535, 2_500), "original": (65_535, _PATCH_REJECT_ABOVE)},
    "gpt-5.6-sol": {"low": (512, None), "high": (2_048, 2_500), "original": (65_535, _PATCH_REJECT_ABOVE)},
    "gpt-5.6-terra": {"low": (512, None), "high": (2_048, 2_500), "original": (65_535, _PATCH_REJECT_ABOVE)},
    "gpt-5.6-luna": {"low": (512, None), "high": (2_048, 2_500), "original": (65_535, _PATCH_REJECT_ABOVE)},
    "gpt-5.5": {"low": (512, None), "high": (2_048, 2_500), "original": (6_000, 10_000)},
    # gpt-5.4 low uses a larger budget than high ("can use more tokens than high").
    "gpt-5.4": {"low": (2_048, 6_144), "high": (2_048, 2_500), "original": (6_000, 10_000)},
    "gpt-5.4-mini": {"low": (2_048, 6_144), "high": (2_048, 2_500), "original": (6_000, 10_000)},
    "gpt-5.4-nano": {"low": (2_048, 6_144), "high": (2_048, 2_500), "original": (6_000, 10_000)},
    # low, high, and auto share one limit; original is not supported.
    "gpt-5.2": {"low": (2_048, 6_144), "high": (2_048, 6_144)},
    "gpt-4.1-mini": {"low": (2_048, 6_144), "high": (2_048, 6_144)},
}
# Models with a published multiplier but no sizing table: every detail is priced
# at the largest allowance any listed model publishes.
_PATCH_SIZING_FALLBACK: Dict[str, Tuple[int, Optional[int]]] = {
    "low": (65_535, _PATCH_REJECT_ABOVE),
    "high": (65_535, _PATCH_REJECT_ABOVE),
    "original": (65_535, _PATCH_REJECT_ABOVE),
}


def openai_image_tokens(model: str, width: int, height: int, detail: str) -> int:
    """Billable image input tokens. ``auto`` is the most expensive detail."""
    key = str(model or "").strip().lower()
    chosen = str(detail or "auto").strip().lower()
    bundle = active_bundle()
    if key in bundle.openai_tile_models:
        base, tile = bundle.openai_tile_models[key]
        low = base
        high = _tile_tokens(width, height, base, tile)
        if chosen == "low":
            return low
        if chosen == "auto":
            return max(low, high)
        return high
    if key in bundle.openai_patch_multipliers:
        mult = bundle.openai_patch_multipliers[key]
        sizing = bundle.openai_patch_sizing.get(key, _PATCH_SIZING_FALLBACK)
        table = {
            name: _patch_tokens(width, height, max_edge=edge, budget=budget, multiplier=mult)
            for name, (edge, budget) in sizing.items()
        }
        # auto (the default when detail is omitted) is the most expensive
        # detail the model supports; an unlisted detail is priced the same way.
        if chosen == "auto" or chosen not in table:
            return max(table.values())
        return table[chosen]
    # Models without a published image formula cannot be priced from pixels.
    raise PricingConfigError(f"no certified image formula for {key!r}")


def anthropic_image_tokens(model: str, width: int, height: int) -> int:
    """Visual tokens: ceil(width/28) * ceil(height/28), capped by the tier.

    Standard models: long edge 1568 and 1568 tokens. Claude 4.7 and later:
    long edge 2576 and 4784 tokens. Oversized images are charged at the cap,
    which is the page's maximum after downscale.
    """
    import math

    key = str(model or "")
    high_res = key.startswith(
        (
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-fable-",
            "claude-haiku-5",
        )
    )
    max_edge = 2576 if high_res else 1568
    max_tokens = 4784 if high_res else 1568
    w, h = max(1, int(width)), max(1, int(height))
    if max(w, h) > max_edge:
        w, h = _fit(w, h, max_edge)
    tokens = math.ceil(w / 28) * math.ceil(h / 28)
    return min(int(tokens), max_tokens)


def image_formula_source(provider: str) -> str:
    if provider == "anthropic":
        return ANTHROPIC_VISION_SOURCE
    return OPENAI_VISION_SOURCE


# --- the active catalog ------------------------------------------------------
#
# ``BUNDLED`` is this module's snapshot. ``kazenai.catalog_document`` can install
# a newer catalog verified against a pinned KazenAI pricing key; it is used
# only while unexpired, and the bundled snapshot remains the floor.


@dataclass(frozen=True)
class CatalogBundle:
    version: str
    source: str
    rows: Mapping[Tuple[str, str, str, str], CatalogRow] = field(repr=False)
    aliases: Mapping[str, str] = field(repr=False)
    anthropic_client_tool_versions: Tuple[str, ...]
    anthropic_tool_system_prompt_tokens: Mapping[str, Tuple[int, int]] = field(repr=False)
    anthropic_tool_system_prompt_fallback: int
    anthropic_tool_definition_tokens: Mapping[str, int] = field(repr=False)
    openai_tile_models: Mapping[str, Tuple[int, int]] = field(repr=False)
    openai_patch_multipliers: Mapping[str, Decimal] = field(repr=False)
    openai_patch_sizing: Mapping[str, Mapping[str, Tuple[int, Optional[int]]]] = field(repr=False)
    anthropic_inference_geo_multipliers: Mapping[str, Decimal] = field(repr=False)
    anthropic_allowed_speeds: Tuple[str, ...]
    first_party_api_hosts: Mapping[str, Tuple[str, ...]] = field(repr=False)
    non_first_party_uplift: Decimal
    issued_at: Optional[str] = None
    expires_at: Optional[str] = None
    sha256: Optional[str] = None

    def expired(self, moment: Optional[datetime] = None) -> bool:
        ends = parse_instant(self.expires_at)
        return ends is not None and (moment or datetime.now(timezone.utc)) >= ends


BUNDLED = CatalogBundle(
    version=CATALOG_VERSION,
    source="bundled",
    rows=PRICE_CATALOG,
    aliases=BUILTIN_ALIASES,
    anthropic_client_tool_versions=tuple(ANTHROPIC_CLIENT_TOOL_VERSIONS),
    anthropic_tool_system_prompt_tokens=ANTHROPIC_TOOL_SYSTEM_PROMPT_TOKENS,
    anthropic_tool_system_prompt_fallback=ANTHROPIC_TOOL_SYSTEM_PROMPT_FALLBACK,
    anthropic_tool_definition_tokens=ANTHROPIC_TOOL_DEFINITION_TOKENS,
    openai_tile_models=_TILE_MODELS,
    openai_patch_multipliers=_PATCH_MULTIPLIER,
    openai_patch_sizing=_PATCH_SIZING,
    anthropic_inference_geo_multipliers=ANTHROPIC_INFERENCE_GEO_MULTIPLIERS,
    anthropic_allowed_speeds=ANTHROPIC_ALLOWED_SPEEDS,
    first_party_api_hosts=FIRST_PARTY_API_HOSTS,
    non_first_party_uplift=NON_FIRST_PARTY_UPLIFT,
)

_LIVE: Optional[CatalogBundle] = None


def install_live_bundle(bundle: Optional[CatalogBundle]) -> None:
    """Make a verified catalog active (``None`` returns to the bundled snapshot)."""
    global _LIVE  # noqa: PLW0603
    _LIVE = bundle


def active_bundle(moment: Optional[datetime] = None) -> CatalogBundle:
    live = _LIVE
    if live is not None and not live.expired(moment):
        return live
    return BUNDLED


def active_catalog_version() -> str:
    return active_bundle().version


def current_table(moment: Optional[datetime] = None) -> Dict[Tuple[str, str, str, str], CatalogRow]:
    """The active catalog's rows as priced at ``moment`` (default now, UTC)."""
    bundle = active_bundle(moment)
    return {key: row.at(moment) for key, row in bundle.rows.items()}
