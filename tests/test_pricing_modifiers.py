"""Price modifiers and request shapes that used to reserve less than the bill.

Each case is from the provider pages checked on 2026-10-17 (see
docs/PRICE_CATALOG.md): fast mode, US-only inference, regional / third-party
endpoints, Anthropic tool overhead, per-model OpenAI image sizing, Gemini audio
input, dated price changes, and fields smuggled through ``extra_body``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest

import kazenai.price_catalog as pc
from kazenai import monitor
from kazenai.enforcement import UnsupportedModeError
from kazenai.price_catalog import PricingConfigError, anthropic_tool_overhead_tokens, openai_image_tokens
from kazenai.request_pricing import (
    clear_tier_breaker,
    prepare_provider_call,
    prepare_provider_call_async,
    settle_usage,
)

anthropic = pytest.importorskip("anthropic")
openai = pytest.importorskip("openai")

_MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)
    # These tests are about the endpoint rule itself.
    monkeypatch.delenv("KAZENAI_ENDPOINT_PRICE_MULTIPLIERS", raising=False)
    clear_tier_breaker()
    yield
    clear_tier_breaker()


def _client(url: str) -> SimpleNamespace:
    return SimpleNamespace(base_url=url)


def _anthropic(requests: list, *, asynchronous: bool = False):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500, json={}, request=request)

    if asynchronous:
        return anthropic.AsyncAnthropic(
            api_key="t", base_url="https://api.anthropic.com", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
    return anthropic.Anthropic(
        api_key="t", base_url="https://api.anthropic.com", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _monitored(client):
    return monitor(client, org_id="o", project_id="p", agent_id="a", max_budget_usd=50, loop_anomaly_threshold=1.01)


# --- fast mode ---------------------------------------------------------------


@pytest.mark.parametrize("where", ["top_level", "extra_body"])
def test_fast_mode_is_refused_before_dispatch_sync_and_async(where):
    speed = {"speed": "fast"} if where == "top_level" else {"extra_body": {"speed": "fast"}}
    base = {"model": "claude-opus-5-5", "max_tokens": 8, "messages": _MESSAGES}
    with pytest.raises(UnsupportedModeError) as exc:
        prepare_provider_call({**base, **speed}, provider="anthropic", surface="messages")
    assert exc.value.error_code == "fast_mode_unsupported"
    if where == "extra_body":
        # anthropic 0.125.0 has no `speed` argument, so this is how callers send it.
        sent: list = []
        client = _monitored(_anthropic(sent))
        with pytest.raises(UnsupportedModeError):
            client.messages.create(**base, **speed)
        assert sent == []
        sent_async: list = []
        aclient = _monitored(_anthropic(sent_async, asynchronous=True))

        async def _go():
            with pytest.raises(UnsupportedModeError):
                await aclient.messages.create(**base, **speed)

        asyncio.run(_go())
        assert sent_async == []


def test_standard_speed_is_allowed():
    pricing = prepare_provider_call(
        {"model": "claude-opus-5-5", "max_tokens": 8, "messages": _MESSAGES, "speed": "standard"},
        provider="anthropic",
        surface="messages",
    )
    assert pricing.reserved_micro_usd > 0


# --- inference_geo -----------------------------------------------------------


def test_us_inference_geo_reserves_and_settles_at_1_1x():
    base = {"model": "claude-sonnet-5-5", "max_tokens": 1000, "messages": _MESSAGES}
    plain = prepare_provider_call(dict(base), provider="anthropic", surface="messages")
    us = prepare_provider_call({**base, "inference_geo": "us"}, provider="anthropic", surface="messages")
    assert us.decision.price_multiplier == Decimal("1.1")
    assert "inference_geo:us" in us.decision.price_modifiers
    assert us.reserved_micro_usd == pc.decimal_to_micro_usd(
        pc.micro_usd_to_decimal(plain.reserved_micro_usd) * Decimal("1.1")
    ) or abs(us.reserved_micro_usd - plain.reserved_micro_usd * 1.1) <= 1
    usage = {"input_tokens": 100_000, "output_tokens": 10_000, "service_tier": "standard"}
    assert settle_usage(us, usage) > settle_usage(plain, usage)
    explicit_global = prepare_provider_call({**base, "inference_geo": "global"}, provider="anthropic", surface="messages")
    assert explicit_global.reserved_micro_usd == plain.reserved_micro_usd


def test_unknown_inference_geo_is_refused():
    with pytest.raises(UnsupportedModeError) as exc:
        prepare_provider_call(
            {"model": "claude-sonnet-5-5", "max_tokens": 8, "messages": _MESSAGES, "inference_geo": "eu"},
            provider="anthropic",
            surface="messages",
        )
    assert exc.value.error_code == "inference_geo_unsupported"


# --- endpoints ---------------------------------------------------------------


def _reserve_on(url: str, provider: str = "openai") -> int:
    if provider == "openai":
        kwargs = {"model": "gpt-4o", "max_tokens": 1000, "messages": _MESSAGES}
        surface = "chat"
    else:
        kwargs = {"model": "claude-sonnet-5-5", "max_tokens": 1000, "messages": _MESSAGES}
        surface = "messages"
    return prepare_provider_call(kwargs, provider=provider, surface=surface, client=_client(url)).reserved_micro_usd


@pytest.mark.parametrize(
    "provider,first_party,other",
    [
        ("openai", "https://api.openai.com/v1", "https://eu.api.openai.com/v1"),
        ("openai", "https://api.openai.com/v1", "https://my-azure.openai.azure.com/openai"),
        ("anthropic", "https://api.anthropic.com", "https://bedrock-runtime.us-east-1.amazonaws.com"),
    ],
)
def test_non_first_party_endpoints_carry_the_10_percent_uplift(provider, first_party, other):
    listed = _reserve_on(first_party, provider)
    uplifted = _reserve_on(other, provider)
    assert uplifted > listed
    assert abs(uplifted - listed * 1.1) <= 2


def test_an_operator_may_declare_a_list_priced_host_but_never_below_list(monkeypatch):
    listed = _reserve_on("https://api.openai.com/v1")
    monkeypatch.setenv("KAZENAI_ENDPOINT_PRICE_MULTIPLIERS", json.dumps({"llm-proxy.internal": "1"}))
    assert _reserve_on("http://llm-proxy.internal:4000/v1") == listed
    monkeypatch.setenv("KAZENAI_ENDPOINT_PRICE_MULTIPLIERS", json.dumps({"llm-proxy.internal": "0.9"}))
    with pytest.raises(PricingConfigError):
        _reserve_on("http://llm-proxy.internal:4000/v1")
    monkeypatch.setenv("KAZENAI_ENDPOINT_PRICE_MULTIPLIERS", json.dumps({"*.internal": "1"}))
    with pytest.raises(PricingConfigError):
        _reserve_on("http://llm-proxy.internal:4000/v1")


def test_unexpected_tier_settlement_keeps_the_request_multiplier():
    pricing = prepare_provider_call(
        {"model": "gpt-4o", "max_tokens": 100, "messages": _MESSAGES},
        provider="openai",
        surface="chat",
        client=_client("https://eu.api.openai.com/v1"),
    )
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0, "service_tier": "priority"}
    held = settle_usage(pricing, usage)
    # gpt-4o priority input is $4.25 / MTok; the regional uplift still applies.
    assert held == pc.decimal_to_micro_usd(Decimal("4.25") * Decimal("1.1"))


# --- extra_body cannot bypass a refusal ---------------------------------------


@pytest.mark.parametrize(
    "provider,surface,kwargs,code",
    [
        ("openai", "chat", {"model": "gpt-4o", "messages": _MESSAGES, "extra_body": {"service_tier": "priority"}}, "service_tier_unsupported"),
        ("openai", "chat", {"model": "gpt-4o", "messages": _MESSAGES, "extra_body": {"web_search_options": {}}}, "web_search_options_unsupported"),
        ("openai", "chat", {"model": "gpt-4o", "messages": _MESSAGES, "extra_body": {"n": 0}}, "n_not_integer"),
        ("anthropic", "messages", {"model": "claude-sonnet-5-5", "max_tokens": 8, "messages": _MESSAGES, "extra_body": {"tools": [{"type": "web_fetch_20250910", "name": "f"}]}}, "anthropic_server_tool_unsupported"),
        ("anthropic", "messages", {"model": "claude-sonnet-5-5", "max_tokens": 8, "messages": _MESSAGES, "extra_body": {"mcp_servers": [{}]}}, "mcp_servers_unsupported"),
    ],
)
def test_extra_body_fields_are_checked_like_top_level_fields(provider, surface, kwargs, code):
    with pytest.raises(UnsupportedModeError) as exc:
        prepare_provider_call(dict(kwargs), provider=provider, surface=surface)
    assert exc.value.error_code == code


def test_extra_body_null_tier_is_replaced_by_the_pin():
    kwargs = {"model": "gpt-4o", "messages": _MESSAGES, "extra_body": {"service_tier": None}}
    prepare_provider_call(kwargs, provider="openai", surface="chat")
    assert kwargs["extra_body"]["service_tier"] == "default"


def test_extra_body_model_and_max_tokens_are_what_gets_priced():
    cheap = prepare_provider_call(
        {"model": "gpt-4o-mini", "max_tokens": 10, "messages": _MESSAGES}, provider="openai", surface="chat"
    )
    smuggled = prepare_provider_call(
        {"model": "gpt-4o-mini", "max_tokens": 10, "messages": _MESSAGES, "extra_body": {"model": "o1-pro", "max_tokens": 4000}},
        provider="openai",
        surface="chat",
    )
    assert smuggled.decision.canonical_model_id == "o1-pro"
    assert smuggled.reserved_micro_usd > cheap.reserved_micro_usd * 1000


def test_responses_background_in_extra_body_is_refused():
    sent: list = []

    def handler(request):
        sent.append(request)
        return httpx.Response(500, json={}, request=request)

    client = _monitored(
        openai.OpenAI(api_key="t", base_url="https://api.openai.com/v1", max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    )
    with pytest.raises(UnsupportedModeError) as exc:
        client.responses.create(model="gpt-4o-mini", input="hi", extra_body={"background": True})
    assert exc.value.error_code == "responses_background_unsupported"
    assert sent == []


# --- Anthropic tool overhead -------------------------------------------------


def test_anthropic_tool_overhead_is_added_to_the_estimate():
    base = {"model": "claude-sonnet-5", "max_tokens": 8, "messages": _MESSAGES}
    plain = prepare_provider_call(dict(base), provider="anthropic", surface="messages")
    tools = [{"type": "browser_toolset_20260801", "name": "browser"}]
    with_tools = prepare_provider_call({**base, "tools": tools, "tool_choice": {"type": "any"}}, provider="anthropic", surface="messages")
    added = with_tools.decision.projected_input_tokens - plain.decision.projected_input_tokens
    # 474 tool-use system prompt (Sonnet 5, tool_choice any) + ~6,670 toolset (+10%).
    assert added >= 474 + 7_340
    assert anthropic_tool_overhead_tokens("claude-sonnet-5", tools, {"type": "auto"}) == 354 + 7_340
    assert anthropic_tool_overhead_tokens("claude-opus-4-7", [{"name": "mine"}], None) == 675
    # A model the published table does not list gets the largest figure.
    assert anthropic_tool_overhead_tokens("claude-fable-5-1", [{"name": "mine"}], None) == 804
    assert anthropic_tool_overhead_tokens("claude-sonnet-5", [], None) == 0


# --- OpenAI image sizing -----------------------------------------------------


def test_openai_low_and_high_detail_follow_the_per_model_table():
    # gpt-5.4 low: 2048 px edge, 6,144-patch budget (larger than high).
    assert openai_image_tokens("gpt-5.4", 2048, 2048, "low") == 4_916
    assert openai_image_tokens("gpt-5.4", 2048, 2048, "high") == 3_000
    # gpt-5.2 and gpt-4.1-mini: low and high share the 2048 / 6,144 limit.
    assert openai_image_tokens("gpt-5.2", 2048, 2048, "high") == 4_916
    assert openai_image_tokens("gpt-4.1-mini", 2048, 2048, "low") == 6_636
    # gpt-6-astra high: 2,500 patches with a 65,535 px edge (elongated image).
    assert openai_image_tokens("gpt-6-astra", 3000, 32, "high") == 113
    # auto is never cheaper than any explicit detail.
    for model in ("gpt-5.4", "gpt-5.2", "gpt-5.5", "gpt-6-astra", "gpt-5-nano"):
        auto = openai_image_tokens(model, 1500, 900, "auto")
        for detail in ("low", "high", "original"):
            assert auto >= openai_image_tokens(model, 1500, 900, detail)


# --- Gemini audio (LiteLLM helpers) -------------------------------------------


def test_gemini_audio_input_is_priced_at_the_audio_rate():
    from kazenai.integrations.litellm_control import (
        estimate_cost_from_response,
        estimate_cost_usd,
        estimate_reserve_usd,
    )

    assert estimate_cost_usd(model="gemini/gemini-2.5-flash", prompt_tokens=1_000_000, completion_tokens=0) == pytest.approx(0.30)
    assert estimate_reserve_usd(model="gemini/gemini-2.5-flash", prompt_tokens=1_000_000, completion_tokens=0) == pytest.approx(1.00)
    assert estimate_reserve_usd(model="gemini/gemini-2.5-flash", prompt_tokens=1_000_000, completion_tokens=0, audio_prompt_tokens=0) == pytest.approx(0.30)
    response = {"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0, "prompt_tokens_details": {"audio_tokens": 400_000}}}
    assert estimate_cost_from_response(model="gemini-2.5-flash", response=response) == pytest.approx(0.58)
    lite = {"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0, "prompt_tokens_details": {"audio_tokens": 1_000_000}}}
    assert estimate_cost_from_response(model="gemini-2.5-flash-lite", response=lite) == pytest.approx(0.30)
    # A model without an audio rate is unchanged.
    assert estimate_reserve_usd(model="gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=0) == pytest.approx(0.15)


# --- dated prices ------------------------------------------------------------


class _Clock(datetime):
    fixed = datetime(2026, 10, 17, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.fixed if tz is None else cls.fixed.astimezone(tz)


def test_scheduled_gemini_increase_applies_on_its_date(monkeypatch):
    key = ("google", "gemini-3.6-flash", "standard", "short")
    assert pc.current_table(datetime(2026, 12, 1, tzinfo=timezone.utc))[key].rates.input == Decimal("0.75")
    assert pc.current_table(datetime(2027, 1, 2, tzinfo=timezone.utc))[key].rates.output == Decimal("7.50")
    monkeypatch.setattr(pc, "datetime", _Clock)
    _Clock.fixed = datetime(2027, 1, 2, tzinfo=timezone.utc)
    from kazenai.integrations.litellm_control import estimate_cost_usd

    assert estimate_cost_usd(model="gemini-3.6-flash", prompt_tokens=1_000_000, completion_tokens=0) == pytest.approx(1.50)


def test_expired_promotional_row_is_refused_under_control(monkeypatch):
    monkeypatch.setenv("KAZENAI_CONTROL_PROFILE", "1")
    kwargs = {"model": "gpt-5.6-sol", "max_output_tokens": 8, "input": "hi"}
    # gpt-5.6-sol declares a long-context band, so it needs the exact counter.
    counter = SimpleNamespace(count=lambda **_k: SimpleNamespace(input_tokens=1_000))
    client = SimpleNamespace(base_url="https://api.openai.com/v1", responses=SimpleNamespace(input_tokens=counter))
    monkeypatch.setattr(pc, "datetime", _Clock)
    _Clock.fixed = datetime(2026, 11, 1, tzinfo=timezone.utc)
    priced = prepare_provider_call(dict(kwargs), provider="openai", surface="responses", client=client)
    assert priced.reserved_micro_usd > 0
    _Clock.fixed = datetime(2026, 11, 23, tzinfo=timezone.utc)
    from kazenai.pricing_policy import UnknownModelError

    with pytest.raises(UnknownModelError):
        prepare_provider_call(dict(kwargs), provider="openai", surface="responses", client=client)


def test_async_path_applies_the_same_rules():
    async def _go():
        with pytest.raises(UnsupportedModeError):
            await prepare_provider_call_async(
                {"model": "claude-opus-5-5", "max_tokens": 8, "messages": _MESSAGES, "extra_body": {"speed": "fast"}},
                provider="anthropic",
                surface="messages",
            )
        priced = await prepare_provider_call_async(
            {"model": "claude-sonnet-5-5", "max_tokens": 1000, "messages": _MESSAGES, "inference_geo": "us"},
            provider="anthropic",
            surface="messages",
            client=_client("https://api.anthropic.com"),
        )
        assert priced.decision.price_multiplier == Decimal("1.1")

    asyncio.run(_go())
