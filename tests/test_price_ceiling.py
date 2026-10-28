"""One price ceiling: exact ids, frozen basis, and fail-closed dispatch."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest

from kazenai import monitor
from kazenai.cost_engine import PricingConfigError, TokenCostEngine
from kazenai.enforcement import UnsupportedModeError
from kazenai.price_catalog import (
    CATALOG_VERSION,
    PRICE_CATALOG,
    anthropic_image_tokens,
    cost_micro_usd,
    decimal_to_micro_usd,
    lookup_row,
)
from kazenai.pricing_policy import UnknownModelError
from kazenai.request_pricing import (
    clear_tier_breaker,
    prepare_provider_call,
    prepare_provider_call_async,
    settle_usage,
    tier_audit_rows,
)

openai = pytest.importorskip("openai")

_CHAT = {
    "id": "chatcmpl-x",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "service_tier": "default"},
}


def _openai(requests, *, counter=None, counter_error=None):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_CHAT, request=request)

    client = openai.OpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    if counter is not None or counter_error is not None:
        def count(**kwargs):
            if counter_error is not None:
                raise counter_error
            if counter is not None:
                counter.append(kwargs)
            return SimpleNamespace(input_tokens=300_000)

        client.responses.input_tokens = SimpleNamespace(count=count)
    else:
        # OpenAI 2.x exposes responses.input_tokens.count. This fixture is the
        # client that does not, which is what the unavailable-counter cases assert.
        client.responses.input_tokens = None
    return client


def _async_openai(requests, *, counter=None):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_CHAT, request=request)

    client = openai.AsyncOpenAI(
        api_key="test",
        base_url="https://openai.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    if counter is not None:
        async def count(**kwargs):
            counter.append(kwargs)
            return SimpleNamespace(input_tokens=300_000)

        client.responses.input_tokens = SimpleNamespace(count=count)
    return client


def _monitored(client, **kw):
    return monitor(
        client,
        org_id="org-1",
        project_id="p",
        agent_id="a",
        max_budget_usd=50,
        loop_anomaly_threshold=1.01,
        **kw,
    )


@pytest.fixture(autouse=True)
def _local_budget(monkeypatch):
    monkeypatch.setenv("KAZENAI_ENV", "dev")
    monkeypatch.setenv("KAZENAI_DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("KAZENAI_ENFORCEMENT_MODE", "fail_open")
    monkeypatch.setenv("KAZENAI_FINOPS_RESERVATION_MODE", "fail_open")
    monkeypatch.delenv("KAZENAI_FINOPS_URL", raising=False)


@pytest.fixture(autouse=True)
def _clear_breaker():
    clear_tier_breaker()
    yield
    clear_tier_breaker()


def test_decimal_to_micro_usd_ceils():
    assert decimal_to_micro_usd(Decimal("0.0000001")) == 1
    assert decimal_to_micro_usd(Decimal("1")) == 1_000_000


def test_prefix_ids_do_not_inherit():
    engine = TokenCostEngine()
    for model in ("gpt-4o-2024-99-99", "claude-sonnet-latest", "gpt-4o-mini-2099-01-01"):
        with pytest.raises(UnknownModelError):
            engine.price(model)


def test_cache_write_is_reserved_not_the_read_price():
    row = lookup_row("anthropic", "claude-sonnet-4-6", "standard", "short")
    assert row is not None
    assert row.rates.cache_write_5m == row.rates.input * Decimal("1.25")
    assert row.rates.cache_write_1h == row.rates.input * Decimal("2")
    assert row.rates.cached_input == row.rates.input * Decimal("0.1")
    kwargs = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]}],
    }
    pricing = prepare_provider_call(dict(kwargs), provider="anthropic", surface="messages")
    assert pricing.decision.cache_write_kind == "5m"
    plain = prepare_provider_call(
        {"model": "claude-sonnet-4-6", "max_tokens": 10, "messages": [{"role": "user", "content": "hello"}]},
        provider="anthropic",
        surface="messages",
    )
    assert pricing.reserved_micro_usd > plain.reserved_micro_usd
    hour = prepare_provider_call(
        {
            "model": "claude-sonnet-4-6",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}],
        },
        provider="anthropic",
        surface="messages",
    )
    assert hour.decision.cache_write_kind == "1h"
    assert hour.reserved_micro_usd > pricing.reserved_micro_usd


def test_from_usage_cache_is_not_zero():
    engine = TokenCostEngine()
    class Usage:
        cache_read_input_tokens = 1000
        output_tokens = 0

    for usage in ({"cache_read_input_tokens": 1000, "output_tokens": 0}, Usage()):
        _inp, _out, cost = engine.from_usage(model="claude-sonnet-4-6", usage=usage)
        assert cost > 0


def test_n_multiplies_output_and_bad_n_refuses_before_dispatch():
    one = prepare_provider_call(
        {"model": "gpt-4o-mini", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]},
        provider="openai",
        surface="chat",
    )
    three = prepare_provider_call(
        {"model": "gpt-4o-mini", "max_tokens": 100, "n": 3, "messages": [{"role": "user", "content": "hi"}]},
        provider="openai",
        surface="chat",
    )
    assert three.decision.n == 3
    assert one.decision.n == 1
    delta = cost_micro_usd(one.decision.rate_vector, output_tokens=200)
    assert three.reserved_micro_usd - one.reserved_micro_usd == delta
    for bad in (0, -1, 1.5):
        requests = []
        client = _monitored(_openai(requests))
        with pytest.raises(UnsupportedModeError):
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"hi {bad}"}],
                n=bad,
            )
        assert requests == []
        async_requests = []
        async_client = _monitored(_async_openai(async_requests))

        async def _go(value=bad, captured=async_requests):
            with pytest.raises(UnsupportedModeError):
                await async_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": f"hi {value}"}],
                    n=value,
                )
            assert captured == []

        asyncio.run(_go())


def test_tools_and_tiers_refuse_before_dispatch():
    refused = [
        {"tools": [{"type": "web_search"}]},
        {"web_search_options": {}},
        {"modalities": ["audio"]},
        {"mcp_servers": [{}]},
        {"container": {"id": "c"}},
        {"service_tier": "priority"},
        {"service_tier": "flex"},
        {"service_tier": "scale"},
        {"service_tier": "auto"},
    ]
    for extra in refused:
        requests = []
        client = _monitored(_openai(requests))
        with pytest.raises(UnsupportedModeError):
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                **extra,
            )
        assert requests == []
    kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function", "function": {"name": "f"}}]}
    pricing = prepare_provider_call(dict(kwargs), provider="openai", surface="chat")
    assert pricing.reserved_micro_usd > 0
    pinned = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    prepare_provider_call(pinned, provider="openai", surface="chat")
    assert pinned["service_tier"] == "default"
    anth = {"model": "claude-sonnet-4-6", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    prepare_provider_call(anth, provider="anthropic", surface="messages")
    assert anth["service_tier"] == "standard_only"
    background_requests = []
    background = _monitored(_openai(background_requests))
    with pytest.raises(UnsupportedModeError) as background_exc:
        background.responses.create(model="gpt-4o-mini", input="hi", background=True)
    assert background_exc.value.error_code == "responses_background_unsupported"
    assert background_requests == []
    with pytest.raises(UnsupportedModeError):
        prepare_provider_call(
            {"model": "claude-sonnet-4-6", "max_tokens": 8, "messages": [], "tools": [{"type": "web_search_20250305", "name": "web_search"}]},
            provider="anthropic",
            surface="messages",
        )
    allowed = prepare_provider_call(
        {"model": "claude-sonnet-4-6", "max_tokens": 8, "messages": [], "tools": [{"name": "mine"}]},
        provider="anthropic",
        surface="messages",
    )
    assert allowed.reserved_micro_usd > 0


def test_async_tool_refusal_does_not_dispatch():
    requests = []
    client = _monitored(_async_openai(requests))

    async def _go():
        with pytest.raises(UnsupportedModeError):
            await client.responses.create(model="gpt-4o-mini", input="hi", tools=[{"type": "web_search"}])

    asyncio.run(_go())
    assert requests == []


def test_reported_standard_is_not_a_mismatch_and_priority_holds():
    pricing = prepare_provider_call(
        {"model": "claude-sonnet-4-6", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
        provider="anthropic",
        surface="messages",
    )
    micros = settle_usage(pricing, {"input_tokens": 10, "output_tokens": 2, "service_tier": "standard"})
    assert micros > 0
    assert pricing.hold_for_reconciliation is False
    gpt = prepare_provider_call(
        {"model": "gpt-4o", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
        provider="openai",
        surface="chat",
    )
    held = settle_usage(gpt, {"prompt_tokens": 100, "completion_tokens": 10, "service_tier": "priority"}, org_id="org-1")
    assert gpt.hold_for_reconciliation is True
    assert gpt.reservation_state == "pending_reconciliation"
    assert gpt.reservation_state != "outcome_unknown"
    assert gpt.usage_evidence["service_tier"] == "priority"
    assert gpt.exposure_source == "certified_unexpected_tier"
    assert held > 0


def test_unknown_tier_opens_breaker_and_audits(caplog):
    pricing = prepare_provider_call(
        {"model": "gpt-4o", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
        provider="openai",
        surface="chat",
        org_id="org-1",
    )
    with caplog.at_level(logging.WARNING, logger="kazenai.request_pricing"):
        held = settle_usage(
            pricing,
            {"prompt_tokens": 100, "completion_tokens": 10, "service_tier": "flex"},
            org_id="org-1",
        )
    assert pricing.pricing_basis == "unknown_tier"
    assert pricing.reservation_state == "pending_reconciliation"
    assert held >= pricing.reserved_micro_usd
    assert tier_audit_rows()
    assert "unknown service tier" in caplog.text
    with pytest.raises(UnsupportedModeError) as exc:
        prepare_provider_call(
            {"model": "gpt-4o", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
            provider="openai",
            surface="chat",
            org_id="org-1",
        )
    assert exc.value.error_code == "unknown_tier_circuit_open"


def test_long_context_counter_and_chat_refusal():
    # Chat Completions on a long-context model: a request whose bytes cannot reach the
    # threshold is priced at the standard band; one that could, or carries an image, is refused.
    allowed = []
    client = _monitored(_openai(allowed))
    client.chat.completions.create(model="gpt-5.4", messages=[{"role": "user", "content": "hi"}])
    assert len(allowed) == 1
    requests = []
    client = _monitored(_openai(requests))
    with pytest.raises(UnsupportedModeError) as chat_exc:
        client.chat.completions.create(
            model="gpt-5.4", messages=[{"role": "user", "content": "x" * 272_000}]
        )
    assert chat_exc.value.error_code == "long_context_chat_unsupported"
    image = {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
    with pytest.raises(UnsupportedModeError) as image_exc:
        client.chat.completions.create(
            model="gpt-5.4", messages=[{"role": "user", "content": [image, {"type": "text", "text": "hi"}]}]
        )
    assert image_exc.value.error_code == "long_context_chat_unsupported"
    assert requests == []
    missing_requests = []
    bare = _monitored(_openai(missing_requests))
    with pytest.raises(UnsupportedModeError) as missing:
        bare.responses.create(model="gpt-5.4", input="hi")
    assert missing.value.error_code == "input_counter_unavailable"
    assert missing_requests == []
    counted = []
    counted_requests = []
    counted_client = _monitored(_openai(counted_requests, counter=counted))
    counted_client.responses.create(model="gpt-5.4", input="hi", max_output_tokens=16)
    assert counted
    assert counted_requests
    boom = []
    boom_client = _monitored(_openai(boom, counter_error=RuntimeError("down")))
    with pytest.raises(UnsupportedModeError) as failed:
        boom_client.responses.create(model="gpt-5.4", input="hi")
    assert failed.value.error_code == "input_counter_failed"
    assert boom == []
    ordinary = []
    ordinary_client = _monitored(_openai(ordinary))
    ordinary_client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert ordinary


def test_async_counter_is_awaited():
    counted = []
    requests = []
    client = _monitored(_async_openai(requests, counter=counted))

    async def _go():
        await client.responses.create(model="gpt-5.4", input="hi", max_output_tokens=16)

    asyncio.run(_go())
    assert counted
    assert requests


def test_counted_request_uses_the_long_band():
    class Counter:
        def count(self, **kwargs):
            return SimpleNamespace(input_tokens=272_001)

    short_client = SimpleNamespace(responses=SimpleNamespace(input_tokens=SimpleNamespace(count=lambda **k: SimpleNamespace(input_tokens=100))))
    long_client = SimpleNamespace(responses=SimpleNamespace(input_tokens=Counter()))
    short = prepare_provider_call(
        {"model": "gpt-5.4", "input": "hi", "max_output_tokens": 10},
        provider="openai",
        surface="responses",
        client=short_client,
    )
    long = prepare_provider_call(
        {"model": "gpt-5.4", "input": "hi", "max_output_tokens": 10},
        provider="openai",
        surface="responses",
        client=long_client,
    )
    assert short.decision.context_band == "short"
    assert long.decision.context_band == "long"
    assert long.decision.threshold_decision == "long"
    assert long.decision.input_token_count_source == "provider_exact_counter"


def test_async_prepare_awaits_counter():
    seen = []

    class Counter:
        async def count(self, **kwargs):
            seen.append(kwargs)
            return {"input_tokens": 272_001}

    client = SimpleNamespace(responses=SimpleNamespace(input_tokens=Counter()))

    async def _go():
        return await prepare_provider_call_async(
            {"model": "gpt-5.4", "input": "hi", "max_output_tokens": 10},
            provider="openai",
            surface="responses",
            client=client,
        )

    pricing = asyncio.run(_go())
    assert seen
    assert pricing.decision.context_band == "long"


def test_images_and_documents():
    assert anthropic_image_tokens("claude-sonnet-4-6", 200, 200) == 64
    for content in (
        [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}],
        [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}, "file_id": "file_1"}],
    ):
        requests = []
        client = _monitored(_openai(requests))
        with pytest.raises(UnsupportedModeError):
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
            )
        assert requests == []
    with pytest.raises(UnsupportedModeError) as doc:
        prepare_provider_call(
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": [{"type": "input_file", "file_id": "file_1"}]}],
            },
            provider="openai",
            surface="chat",
            observe_unbounded=("images",),
        )
    assert doc.value.error_code == "document_reference_unsupported"
    tiny = base64.b64encode(b"\x89PNG\r\n").decode()
    with pytest.raises(UnsupportedModeError):
        prepare_provider_call(
            {
                "model": "claude-sonnet-4-6",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": tiny}}]}],
            },
            provider="anthropic",
            surface="messages",
        )


def test_observe_unbounded_images_reserve_text_only():
    pricing = prepare_provider_call(
        {
            "model": "gpt-4o",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}],
        },
        provider="openai",
        surface="chat",
        observe_unbounded=("images",),
    )
    assert pricing.decision.spend_bound == "none"
    fields = pricing.decision.reserve_body_fields()
    assert fields["pricing"]["spend_bound"] == "none"
    assert fields["provider"] == "openai"
    assert fields["pricing"]["canonical_model_id"] == "gpt-4o"
    assert "rate_vector" not in fields
    assert "canonical_model_id" not in fields


def test_malformed_pricing_json_fails_closed(monkeypatch, caplog):
    sentinel = "SENTINEL_DO_NOT_LOG_9f3a"
    monkeypatch.setenv("KAZENAI_PRICING_JSON", json.dumps({"nope": sentinel}))
    with caplog.at_level(logging.ERROR, logger="kazenai.cost_engine"):
        with pytest.raises(PricingConfigError) as exc:
            TokenCostEngine()
    assert sentinel not in str(exc.value)
    assert sentinel not in caplog.text
    assert "KAZENAI_PRICING_JSON rejected" in caplog.text


def test_operator_alias_is_exact(monkeypatch):
    monkeypatch.setenv("KAZENAI_MODEL_ALIASES_JSON", json.dumps({"my-gpt-4o": "gpt-4o"}))
    engine = TokenCostEngine()
    assert engine.price("my-gpt-4o") == engine.price("gpt-4o")
    pricing = prepare_provider_call(
        {"model": "my-gpt-4o", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
        provider="openai",
        surface="chat",
    )
    fields = pricing.decision.reserve_body_fields()
    assert fields["model"] == "my-gpt-4o"
    assert fields["pricing"]["canonical_model_id"] == "gpt-4o"
    assert fields["pricing"]["catalog_version"] == CATALOG_VERSION
    assert fields["pricing"]["input_token_count_source"] in {"estimate", "provider_count"}
    assert fields["pricing"]["rule"] in {"standard", "long_context"}
    monkeypatch.setenv("KAZENAI_MODEL_ALIASES_JSON", json.dumps({"gpt-4o-*": "gpt-4o"}))
    with pytest.raises(PricingConfigError):
        TokenCostEngine()


def test_multiplier_and_control(monkeypatch):
    monkeypatch.setenv("KAZENAI_TIER_MISMATCH_MULTIPLIER", "0.5")
    with pytest.raises(PricingConfigError):
        TokenCostEngine()
    monkeypatch.delenv("KAZENAI_TIER_MISMATCH_MULTIPLIER", raising=False)
    monkeypatch.setenv("KAZENAI_CONTROL_PROFILE", "1")
    with pytest.raises(PricingConfigError):
        TokenCostEngine()


def test_settlement_uses_the_frozen_vector():
    pricing = prepare_provider_call(
        {"model": "gpt-4o-mini", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
        provider="openai",
        surface="chat",
    )
    frozen = pricing.decision.rate_vector
    key = ("openai", "gpt-4o-mini", "standard", "short")
    original = PRICE_CATALOG[key]
    PRICE_CATALOG[key] = replace(original, rates=replace(frozen, output=Decimal("999")))
    try:
        micros = settle_usage(pricing, {"prompt_tokens": 1000, "completion_tokens": 0, "service_tier": "default"})
    finally:
        PRICE_CATALOG[key] = original
    assert micros == cost_micro_usd(frozen, input_tokens=1000)
    assert pricing.decision.catalog_version == CATALOG_VERSION
    assert pricing.decision.rounding_rule


def test_pillow_image_price_when_installed():
    pytest.importorskip("PIL")
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (200, 200), "white").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    pricing = prepare_provider_call(
        {
            "model": "claude-sonnet-4-6",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}}]}],
        },
        provider="anthropic",
        surface="messages",
    )
    text = prepare_provider_call(
        {"model": "claude-sonnet-4-6", "max_tokens": 1, "messages": [{"role": "user", "content": "x"}]},
        provider="anthropic",
        surface="messages",
    )
    assert pricing.reserved_micro_usd > text.reserved_micro_usd


@pytest.mark.parametrize(
    "extra,code",
    [
        ({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}, "anthropic_server_tool_unsupported"),
        ({"tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}]}, "anthropic_server_tool_unsupported"),
        ({"tools": [{"type": "code_execution_20250522", "name": "code_execution"}]}, "anthropic_server_tool_unsupported"),
        ({"tools": [{"type": "bash_20990101", "name": "bash"}]}, "anthropic_tool_unsupported"),
        ({"mcp_servers": [{"type": "url", "url": "https://mcp.invalid", "name": "m"}]}, None),
        ({"container": "container_abc"}, None),
    ],
)
def test_each_refused_anthropic_tool_surface_refuses_before_reserve(extra, code):
    """1.3: web search, web fetch, code execution, uncertified tool versions, MCP and containers."""
    kwargs = {"model": "claude-sonnet-4-6", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}], **extra}
    with pytest.raises(UnsupportedModeError) as exc:
        prepare_provider_call(kwargs, provider="anthropic", surface="messages")
    if code is not None:
        assert exc.value.error_code == code


def test_a_certified_anthropic_client_tool_version_still_reserves():
    from kazenai.price_catalog import ANTHROPIC_CLIENT_TOOL_VERSIONS

    version = ANTHROPIC_CLIENT_TOOL_VERSIONS[0]
    pricing = prepare_provider_call(
        {
            "model": "claude-sonnet-4-6",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": version, "name": version.split("_2")[0]}],
        },
        provider="anthropic",
        surface="messages",
    )
    assert pricing.reserved_micro_usd > 0
