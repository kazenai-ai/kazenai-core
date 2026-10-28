"""Pre-dispatch price ceiling: tools, n, tier, images, long context.

Local refusals run before any provider counter. The counter runs before
reserve. A counter exception refuses and does not dispatch.
"""

from __future__ import annotations

import base64
import binascii
import inspect
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Optional

from .enforcement import UnsupportedModeError
from .price_catalog import (
    ANTHROPIC_PIN_REQUEST_TIER,
    OPENAI_PIN_REQUEST_TIER,
    ROUNDING_RULE,
    UNKNOWN_CHARGE_AFTER_DAYS,
    CatalogRow,
    PricingConfigError,
    RateVector,
    active_bundle,
    anthropic_image_tokens,
    anthropic_tool_overhead_tokens,
    cost_micro_usd,
    decimal_to_micro_usd,
    lookup_row,
    micro_usd_to_decimal,
    model_declares_long_band,
    openai_image_tokens,
    provider_for_model,
)
from .pricing_context import set_current_pricing

_log = logging.getLogger("kazenai.request_pricing")

_REFUSED_TOOL_PREFIXES = ("web_search", "web_fetch", "code_execution")
_DOCUMENT_TYPES = {
    "file",
    "input_file",
    "document",
    "input_document",
    "container",
}
_IMAGE_TYPES = {"image", "image_url", "input_image"}
_ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "GIF", "WEBP"}
_MAX_ENCODED_CHARS = 10 * 1024 * 1024
_MAX_DECODED_BYTES = 8 * 1024 * 1024
_MAX_DIMENSION = 8000
_MAX_PIXELS = 20_000_000
_MAX_FRAMES = 1

# In-process admission breaker. Key is org + provider + model + reported tier.
_TIER_BREAKER: dict[tuple[str, str, str, str], dict[str, Any]] = {}
_TIER_AUDIT: list[dict[str, Any]] = []


class CallPricing:
    """Frozen catalog basis plus the mutable settlement outcome for one call."""

    def __init__(self, decision: "PricingDecision") -> None:
        self.decision = decision
        self.hold_for_reconciliation = False
        self.reservation_state = "reserved"
        self.pricing_basis = decision.pricing_basis
        self.held_exposure_usd = decision.reserved_usd
        self.usage_evidence: dict[str, Any] = {}
        self.exposure_source = "certified_row"
        # Token components the settlement was priced from (FinOps re-prices them).
        self.settled_components: dict[str, int] = {}

    @property
    def reserved_usd(self) -> float:
        return self.decision.reserved_usd

    @property
    def reserved_micro_usd(self) -> int:
        return self.decision.reserved_micro_usd


@dataclass(frozen=True)
class PricingDecision:
    catalog_version: str
    provider: str
    canonical_model_id: str
    presented_model_id: str
    request_tier: str
    expected_reported_tier: str
    context_band: str
    rate_vector: RateVector
    rounding_rule: str
    pricing_source: str
    pricing_basis: str
    label: str
    input_token_count_source: str
    pricing_rule: str
    long_context_threshold: Optional[int]
    threshold_decision: str
    spend_bound: str
    n: int
    reserved_micro_usd: int
    cache_write_kind: str
    certified: bool
    verified_on: str
    projected_input_tokens: int = 0
    projected_output_tokens: int = 0
    projected_image_tokens: int = 0
    # Request price modifiers already folded into ``rate_vector`` (1 = none).
    price_multiplier: Decimal = Decimal("1")
    price_modifiers: tuple = ()

    @property
    def reserved_usd(self) -> float:
        return float(micro_usd_to_decimal(self.reserved_micro_usd))

    def reserve_body_fields(self) -> dict[str, Any]:
        """Reserve body fields. Pricing identity is nested so it matches the contract.

        The rate vector is not sent. FinOps looks up its own catalog row.
        """
        rule = "long_context" if self.context_band == "long" else "standard"
        source = self.input_token_count_source
        if source not in {"estimate", "provider_count"}:
            source = "provider_count" if "provider" in source else "estimate"
        threshold = self.threshold_decision
        if threshold not in {"not_applicable", "below", "crossed"}:
            threshold = "crossed" if self.context_band == "long" else "not_applicable"
        return {
            "provider": self.provider,
            "model": self.presented_model_id,
            "pricing": {
                "catalog_version": self.catalog_version,
                "canonical_model_id": self.canonical_model_id,
                "request_tier": self.request_tier,
                "context_band": self.context_band,
                "input_token_count_source": source,
                "rule": rule,
                "long_context_threshold": self.long_context_threshold,
                "threshold_decision": threshold,
                "spend_bound": self.spend_bound if self.spend_bound in {"bounded", "none"} else "bounded",
                "basis": "rate_card",
                "usage": {
                    "input_tokens": 0 if self.cache_write_kind in {"5m", "1h"} else self.projected_input_tokens,
                    "cache_write_5m_tokens": self.projected_input_tokens if self.cache_write_kind == "5m" else 0,
                    "cache_write_1h_tokens": self.projected_input_tokens if self.cache_write_kind == "1h" else 0,
                    "output_tokens": self.projected_output_tokens,
                    "image_tokens": self.projected_image_tokens,
                },
            },
        }


def tier_audit_rows() -> list[dict[str, Any]]:
    return list(_TIER_AUDIT)


def clear_tier_breaker() -> None:
    _TIER_BREAKER.clear()
    _TIER_AUDIT.clear()


def _breaker_key(org: str, provider: str, model: str, tier: str) -> tuple[str, str, str, str]:
    return (str(org or ""), provider, model, tier)


def unresolved_tier_state(opened_at: float, *, now: Optional[float] = None) -> str:
    """After 35 days the hold is terminal ``unknown_charge`` and is not released."""
    moment = time.time() if now is None else float(now)
    if moment - float(opened_at) >= UNKNOWN_CHARGE_AFTER_DAYS * 86400:
        return "unknown_charge"
    return "pending_reconciliation"


def _refuse(message: str, code: str) -> None:
    raise UnsupportedModeError(message, error_code=code)


def _integer_n(value: Any) -> int:
    if value is None:
        return 1
    if isinstance(value, bool) or isinstance(value, float):
        _refuse(
            "n must be an integer >= 1. Missing n means 1.",
            "n_not_integer",
        )
    if isinstance(value, str):
        _refuse("n must be an integer >= 1.", "n_not_integer")
    try:
        number = int(value)
    except (TypeError, ValueError):
        _refuse("n must be an integer >= 1.", "n_not_integer")
    if number < 1 or int(value) != number:
        _refuse("n must be an integer >= 1.", "n_not_integer")
    # Reject 1.0 that arrived as a non-int. ints only.
    if type(value) is not int and not (isinstance(value, int) and not isinstance(value, bool)):
        _refuse("n must be an integer >= 1.", "n_not_integer")
    return number


def _tool_type(tool: Any) -> Optional[str]:
    if isinstance(tool, Mapping):
        raw = tool.get("type")
    else:
        raw = getattr(tool, "type", None)
    if raw is None:
        return None
    return str(raw)


def _validate_request_modifiers(kwargs: Mapping[str, Any], *, provider: str) -> None:
    """Refuse request options whose price the active catalog does not certify."""
    if provider != "anthropic":
        return
    bundle = active_bundle()
    speed = kwargs.get("speed")
    if speed is not None and str(speed).strip().lower() not in bundle.anthropic_allowed_speeds:
        _refuse(
            f"Anthropic speed={speed!r} (fast mode) is premium-priced and not certified; it is refused before dispatch.",
            "fast_mode_unsupported",
        )
    geo = kwargs.get("inference_geo")
    if geo is not None and str(geo).strip().lower() not in bundle.anthropic_inference_geo_multipliers:
        _refuse(
            f"Anthropic inference_geo={geo!r} has no certified price multiplier.",
            "inference_geo_unsupported",
        )


def _client_host(client: Any) -> Optional[str]:
    """Host the provider client sends to, or ``None`` when it cannot be read."""
    if client is None:
        return None
    raw = getattr(client, "base_url", None)
    if raw is None:
        inner = getattr(client, "_client", None)
        raw = getattr(inner, "base_url", None) if inner is not None else None
    if raw is None:
        return None
    host = getattr(raw, "host", None)
    if host is None:
        from urllib.parse import urlsplit

        host = urlsplit(str(raw)).hostname
    host = str(host or "").strip().lower()
    return host or None


def _operator_endpoint_multipliers() -> dict[str, Decimal]:
    """``KAZENAI_ENDPOINT_PRICE_MULTIPLIERS``: exact host -> multiplier (>= 1).

    For a proxy or gateway that bills at the first-party list price, an operator
    may declare ``{"llm-proxy.internal": "1"}``. Values below 1 are refused: an
    operator can raise a ceiling, never lower it below list price.
    """
    raw = os.getenv("KAZENAI_ENDPOINT_PRICE_MULTIPLIERS", "").strip()
    if not raw:
        return {}
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PricingConfigError("KAZENAI_ENDPOINT_PRICE_MULTIPLIERS is not valid JSON") from exc
    if not isinstance(document, dict):
        raise PricingConfigError("KAZENAI_ENDPOINT_PRICE_MULTIPLIERS must be an object of exact hosts")
    out: dict[str, Decimal] = {}
    for host, value in document.items():
        name = str(host or "").strip().lower()
        if not name or any(ch in name for ch in "*?/[] "):
            raise PricingConfigError("KAZENAI_ENDPOINT_PRICE_MULTIPLIERS keys must be exact hosts")
        try:
            number = Decimal(str(value))
        except Exception as exc:
            raise PricingConfigError("KAZENAI_ENDPOINT_PRICE_MULTIPLIERS values must be decimals") from exc
        if not number.is_finite() or number < 1:
            raise PricingConfigError("KAZENAI_ENDPOINT_PRICE_MULTIPLIERS values must be finite and >= 1")
        out[name] = number
    return out


def _price_multiplier(kwargs: Mapping[str, Any], *, provider: str, client: Any) -> tuple[Decimal, tuple[str, ...]]:
    """Multiplier on every rate for this request, with the reasons."""
    bundle = active_bundle()
    multiplier = Decimal("1")
    reasons: list[str] = []
    if provider == "anthropic":
        geo = kwargs.get("inference_geo")
        if geo is not None:
            factor = bundle.anthropic_inference_geo_multipliers.get(str(geo).strip().lower(), Decimal("1"))
            if factor != 1:
                multiplier *= factor
                reasons.append(f"inference_geo:{str(geo).strip().lower()}")
    host = _client_host(client)
    first_party = bundle.first_party_api_hosts.get(provider, ())
    if host is not None and first_party and host not in first_party:
        declared = _operator_endpoint_multipliers().get(host)
        factor = declared if declared is not None else bundle.non_first_party_uplift
        if factor != 1:
            multiplier *= factor
            reasons.append("endpoint:" + ("operator_declared" if declared is not None else "non_first_party"))
    return multiplier, tuple(reasons)


def _validate_tools(kwargs: Mapping[str, Any], *, provider: str, surface: str) -> None:
    if "mcp_servers" in kwargs and kwargs.get("mcp_servers") is not None:
        _refuse("mcp_servers is not a certified client tool.", "mcp_servers_unsupported")
    if "container" in kwargs and kwargs.get("container") is not None:
        _refuse("container is not a certified client tool.", "container_unsupported")
    if provider == "openai" and surface == "chat":
        if "web_search_options" in kwargs and kwargs.get("web_search_options") is not None:
            _refuse(
                "Chat Completions web_search_options is refused before dispatch.",
                "web_search_options_unsupported",
            )
        modalities = kwargs.get("modalities")
        if modalities is not None and any(str(item) == "audio" for item in modalities):
            _refuse("Chat Completions audio output is refused before dispatch.", "audio_output_unsupported")
        if "audio" in kwargs and kwargs.get("audio") is not None:
            _refuse("Chat Completions audio output is refused before dispatch.", "audio_output_unsupported")
    for tool in kwargs.get("tools") or ():
        tool_type = _tool_type(tool)
        if provider == "anthropic":
            if tool_type is None or tool_type == "custom":
                continue
            if tool_type.startswith(_REFUSED_TOOL_PREFIXES) or any(
                tool_type == item or tool_type.startswith(item + "_") for item in ("web_search", "web_fetch", "code_execution")
            ):
                _refuse(
                    f"Anthropic tool {tool_type!r} is refused before dispatch.",
                    "anthropic_server_tool_unsupported",
                )
            if tool_type not in active_bundle().anthropic_client_tool_versions:
                _refuse(
                    f"Anthropic tool {tool_type!r} is not a certified client-tool version.",
                    "anthropic_tool_unsupported",
                )
            continue
        if surface == "responses":
            if tool_type not in {"function", "custom"}:
                _refuse(
                    f"OpenAI Responses tool {tool_type!r} is refused before dispatch.",
                    "responses_builtin_tool_unsupported",
                )
            continue
        if tool_type != "function":
            _refuse(
                f"Chat Completions tool {tool_type!r} is refused before dispatch.",
                "chat_tool_unsupported",
            )


_SDK_TRANSPORT_KEYS = ("extra_body", "extra_headers", "extra_query", "timeout")


def _effective_request(kwargs: Mapping[str, Any]) -> dict:
    """The request the provider will receive: ``extra_body`` overlays top-level fields.

    The provider SDKs merge ``extra_body`` into the JSON body, so a field placed
    there (``service_tier``, ``speed``, ``tools``, ``model``, ``max_tokens`` ...)
    is billed exactly like a top-level one. Every refusal and price is computed
    on this view, never on the top-level arguments alone.
    """
    extra = kwargs.get("extra_body")
    view = {key: value for key, value in kwargs.items() if key not in _SDK_TRANSPORT_KEYS}
    if extra is None:
        return view
    if not isinstance(extra, Mapping):
        _refuse("extra_body must be a JSON object to be priced.", "extra_body_unsupported")
    view.update(dict(extra))
    return view


def _pin_service_tier(kwargs: dict, *, provider: str) -> str:
    """Inject the standard-only literal when unset. Any other explicit value refuses.

    ``extra_body`` wins over top-level fields in the sent request, so it is
    checked too, and an explicit ``null`` there is replaced by the pin (an
    unset OpenAI tier means ``auto``, which is not the standard tier).
    """
    pin = ANTHROPIC_PIN_REQUEST_TIER if provider == "anthropic" else OPENAI_PIN_REQUEST_TIER
    extra = kwargs.get("extra_body")
    if isinstance(extra, Mapping) and "service_tier" in extra:
        requested_extra = extra.get("service_tier")
        if requested_extra in (None, ""):
            kwargs["extra_body"] = {**dict(extra), "service_tier": pin}
        elif str(requested_extra) != pin:
            _refuse(
                f"service_tier {str(requested_extra)!r} is not the pinned standard tier ({pin!r}).",
                "service_tier_unsupported",
            )
    if "service_tier" not in kwargs or kwargs.get("service_tier") in (None, ""):
        kwargs["service_tier"] = pin
        return "standard"
    requested = str(kwargs.get("service_tier"))
    if requested != pin:
        _refuse(
            f"service_tier {requested!r} is not the pinned standard tier ({pin!r}).",
            "service_tier_unsupported",
        )
    return "standard"


def _cache_write_kind(kwargs: Mapping[str, Any]) -> str:
    """Return ``1h`` when a 1-hour write can happen, else ``5m``, else ``none``."""
    found = "none"

    def _see(node: Any) -> None:
        nonlocal found
        if isinstance(node, Mapping):
            control = node.get("cache_control")
            if isinstance(control, Mapping):
                ttl = str(control.get("ttl") or "5m")
                if ttl in {"1h", "1hour", "hour"}:
                    found = "1h"
                elif found != "1h":
                    found = "5m"
            for value in node.values():
                _see(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                _see(value)

    _see(kwargs)
    return found


def _walk_content(kwargs: Mapping[str, Any]) -> list[Any]:
    nodes: list[Any] = []
    for key in ("messages", "input", "system"):
        value = kwargs.get(key)
        if value is not None:
            nodes.append(value)
    return nodes


def _iter_parts(node: Any):
    if isinstance(node, Mapping):
        yield node
        for value in node.values():
            yield from _iter_parts(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _iter_parts(item)


def _is_document(part: Mapping[str, Any]) -> bool:
    kind = str(part.get("type") or "")
    if kind in _DOCUMENT_TYPES:
        return True
    if "file_id" in part and kind not in _IMAGE_TYPES:
        return True
    source = part.get("source")
    if isinstance(source, Mapping) and source.get("type") in {"file", "file_id"}:
        return True
    return False


def _image_payload(part: Mapping[str, Any]) -> Optional[dict]:
    kind = str(part.get("type") or "")
    if kind not in _IMAGE_TYPES and "image_url" not in part and not (
        isinstance(part.get("source"), Mapping) and str(part["source"].get("type") or "") in {"base64", "url"}
        and kind == "image"
    ):
        if kind not in _IMAGE_TYPES:
            return None
    detail = "auto"
    if isinstance(part.get("image_url"), Mapping):
        detail = str(part["image_url"].get("detail") or "auto")
        url = part["image_url"].get("url")
        return {"detail": detail, "url": url, "source": None, "file_id": part.get("file_id")}
    if isinstance(part.get("image_url"), str):
        return {"detail": str(part.get("detail") or "auto"), "url": part.get("image_url"), "source": None, "file_id": part.get("file_id")}
    source = part.get("source") if isinstance(part.get("source"), Mapping) else None
    if kind == "input_image":
        return {
            "detail": str(part.get("detail") or "auto"),
            "url": part.get("image_url"),
            "source": source,
            "file_id": part.get("file_id"),
        }
    if kind == "image":
        return {
            "detail": str(part.get("detail") or "auto"),
            "url": None if not isinstance(source, Mapping) else source.get("url"),
            "source": source,
            "file_id": part.get("file_id") or (source.get("file_id") if isinstance(source, Mapping) else None),
        }
    return None


def _decode_inline(encoded: str) -> bytes:
    if len(encoded) > _MAX_ENCODED_CHARS:
        _refuse("inline image encoded length exceeds the limit.", "image_encoded_too_large")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        _refuse("inline image is not valid base64.", "image_malformed")
    if len(raw) > _MAX_DECODED_BYTES:
        _refuse("inline image decoded size exceeds the limit.", "image_decoded_too_large")
    return raw


def _split_data_url(url: str) -> Optional[str]:
    if not str(url).startswith("data:"):
        return None
    try:
        header, payload = str(url).split(",", 1)
    except ValueError:
        _refuse("inline image data URL is malformed.", "image_malformed")
    if ";base64" not in header:
        _refuse("inline image data URL is not base64.", "image_malformed")
    return payload


def _inspect_image_bytes(raw: bytes) -> tuple[str, int, int, int]:
    stripped = raw.lstrip()
    if stripped.startswith((b"<svg", b"<?xml", b"<SVG", b"%PDF", b"<!DOCTYPE")):
        _refuse("SVG and document images are refused.", "image_format_rejected")
    try:
        from PIL import Image as _Image
    except ImportError:
        _refuse(
            "inline images require the optional Pillow extra (kazenai[images]).",
            "image_pillow_missing",
        )
    bomb = getattr(_Image, "DecompressionBombWarning", None)
    previous = getattr(_Image, "MAX_IMAGE_PIXELS", None)
    try:
        _Image.MAX_IMAGE_PIXELS = _MAX_PIXELS
        with warnings.catch_warnings():
            if bomb is not None:
                warnings.simplefilter("error", bomb)
            try:
                from io import BytesIO

                image = _Image.open(BytesIO(raw))
                # Metadata only. Never load() and never decode raster pixels.
                fmt = str(image.format or "")
                width, height = image.size
                frames = int(getattr(image, "n_frames", 1) or 1)
            except Exception as exc:
                name = type(exc).__name__
                if "DecompressionBomb" in name or "DecompressionBomb" in str(exc):
                    _refuse("image tripped decompression-bomb protection.", "image_decompression_bomb")
                _refuse("image header is truncated or malformed.", "image_malformed")
    finally:
        if previous is not None:
            _Image.MAX_IMAGE_PIXELS = previous
    if fmt.upper() not in _ALLOWED_IMAGE_FORMATS:
        _refuse(f"image format {fmt!r} is not in the allowlist.", "image_format_rejected")
    if frames > _MAX_FRAMES:
        _refuse("image frame count exceeds the limit.", "image_too_many_frames")
    if width > _MAX_DIMENSION or height > _MAX_DIMENSION or width < 1 or height < 1:
        _refuse("image dimensions exceed the limit.", "image_dimensions")
    if width * height > _MAX_PIXELS:
        _refuse("image pixel count exceeds the limit.", "image_pixels")
    return fmt, int(width), int(height), frames


def _inline_b64(payload: dict) -> Optional[str]:
    source = payload.get("source")
    if isinstance(source, Mapping) and str(source.get("type") or "") == "base64":
        return str(source.get("data") or "")
    url = payload.get("url")
    if isinstance(url, str):
        return _split_data_url(url)
    return None


def _collect_images(kwargs: Mapping[str, Any]) -> tuple[list[dict], bool]:
    """Return image descriptors and whether a document-by-reference is present."""
    images: list[dict] = []
    document = False
    for part in _iter_parts(_walk_content(kwargs)):
        if not isinstance(part, Mapping):
            continue
        if _is_document(part):
            document = True
            continue
        image = _image_payload(part)
        if image is not None:
            images.append(image)
    return images, document


def _client_has_exact_counter(client: Any, *, provider: str, surface: str) -> bool:
    if client is None:
        return False
    if provider == "anthropic":
        messages = getattr(client, "messages", None)
        return callable(getattr(messages, "count_tokens", None))
    if provider == "openai" and surface == "responses":
        responses = getattr(client, "responses", None)
        resource = getattr(responses, "input_tokens", None)
        return callable(getattr(resource, "count", None))
    return False


# Request fields the provider token counters accept. The rest of a create call
# (max_tokens, max_output_tokens, temperature, store, metadata, ...) does not change
# the input count, and the official SDKs reject it with TypeError.
_COUNTER_FIELDS = {
    "anthropic": frozenset(
        {"model", "messages", "system", "tools", "tool_choice", "thinking", "mcp_servers",
         "output_config", "output_format", "cache_control"}
    ),
    "openai": frozenset(
        {"model", "input", "instructions", "tools", "tool_choice", "parallel_tool_calls", "reasoning",
         "text", "truncation", "conversation", "previous_response_id", "personality"}
    ),
}


def _counter_method(client: Any, provider: str) -> Any:
    if provider == "anthropic":
        return client.messages.count_tokens
    return client.responses.input_tokens.count


def _openai_text_format(text_format: Any) -> Any:
    """The ``text.format`` the OpenAI SDK sends for ``parse``/``stream(text_format=...)``."""
    try:
        from openai.lib._parsing._responses import type_to_text_format_param

        return type_to_text_format_param(text_format)
    except Exception:
        return None


def _counter_kwargs(counter: Any, kwargs: Mapping[str, Any], *, provider: str) -> dict[str, Any]:
    """Only the fields the counter takes: by its signature, else the known list."""
    known = set(_COUNTER_FIELDS.get(provider, ()))
    try:
        params = inspect.signature(counter).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            accepted = known
        else:
            accepted = {name for name in params if name != "timeout" and not name.startswith("extra_")}
    except (TypeError, ValueError):
        accepted = known
    out = {k: v for k, v in kwargs.items() if k in accepted}
    if provider == "openai" and "text_format" in kwargs and "text" in accepted and "text" not in out:
        fmt = _openai_text_format(kwargs["text_format"])
        if fmt is not None:
            out["text"] = {"format": fmt}
    return out


def _call_counter(client: Any, kwargs: Mapping[str, Any], *, provider: str, surface: str) -> int:
    try:
        counter = _counter_method(client, provider)
        result = counter(**_counter_kwargs(counter, kwargs, provider=provider))
    except UnsupportedModeError:
        raise
    except Exception as exc:
        _refuse(
            f"exact input counter failed ({type(exc).__name__}); the request was not dispatched.",
            "input_counter_failed",
        )
    if hasattr(result, "__await__"):
        _refuse("exact input counter returned an awaitable on the sync path.", "input_counter_failed")
    tokens = getattr(result, "input_tokens", None)
    if tokens is None and isinstance(result, Mapping):
        tokens = result.get("input_tokens")
    try:
        return max(0, int(tokens))
    except (TypeError, ValueError):
        _refuse("exact input counter returned no integer input_tokens.", "input_counter_failed")
    return 0


async def _call_counter_async(client: Any, kwargs: Mapping[str, Any], *, provider: str, surface: str) -> int:
    try:
        counter = _counter_method(client, provider)
        result = counter(**_counter_kwargs(counter, kwargs, provider=provider))
        if hasattr(result, "__await__"):
            result = await result
    except UnsupportedModeError:
        raise
    except Exception as exc:
        _refuse(
            f"exact input counter failed ({type(exc).__name__}); the request was not dispatched.",
            "input_counter_failed",
        )
    tokens = getattr(result, "input_tokens", None)
    if tokens is None and isinstance(result, Mapping):
        tokens = result.get("input_tokens")
    try:
        return max(0, int(tokens))
    except (TypeError, ValueError):
        _refuse("exact input counter returned no integer input_tokens.", "input_counter_failed")
    return 0


def _output_tokens(kwargs: Mapping[str, Any], n: int) -> int:
    from .monitor import _max_output_tokens

    return max(0, int(_max_output_tokens(kwargs))) * n


def _text_input_tokens(kwargs: Mapping[str, Any]) -> int:
    from .monitor import _estimate_input_tokens

    return int(_estimate_input_tokens(kwargs))


# Chat Completions has no exact input counter. For a model with a long-context band,
# the standard price is certain when the request cannot reach the threshold: every
# token covers at least one UTF-8 byte of text, so the request's JSON size in bytes,
# plus generous framing for roles, tool rendering and priming, bounds its input
# tokens. Images and audio are not bounded by their bytes, so they still need a count.
# Tool definitions and JSON schemas reach the model rewritten as text (comment lines
# per description line, unions for enums, indentation), which can be a few times
# their JSON size, so they count four times.
_CHAT_BOUND_FRAMING = 256
_CHAT_BOUND_PER_MESSAGE = 16
_CHAT_BOUND_PER_TOOL = 64
_CHAT_BOUND_RENDERED_FACTOR = 4
_CHAT_RENDERED_FIELDS = ("tools", "functions", "response_format")
_CHAT_TEXT_PART_TYPES = {"text", "refusal"}


def _openai_response_format(value: Any) -> Any:
    """The ``response_format`` the OpenAI SDK sends for ``parse(response_format=Model)``."""
    try:
        from openai.lib._parsing._completions import type_to_response_format_param

        return type_to_response_format_param(value)
    except Exception:
        return None


def _strict_json_bytes(value: Any, depth: int = 0) -> Optional[int]:
    """Bytes of ``value`` as JSON, or None when part of it cannot be measured."""
    if depth > 64:
        return None
    if value is None or isinstance(value, bool):
        return 5
    if isinstance(value, (int, float)):
        return len(repr(value))
    if isinstance(value, str):
        return len(value.encode("utf-8", errors="replace")) + 2
    if isinstance(value, Mapping):
        total = 2
        for key, item in value.items():
            size = _strict_json_bytes(item, depth + 1)
            if size is None:
                return None
            total += len(str(key).encode("utf-8", errors="replace")) + 4 + size
        return total
    if isinstance(value, (list, tuple)):
        total = 2
        for item in value:
            size = _strict_json_bytes(item, depth + 1)
            if size is None:
                return None
            total += size + 1
        return total
    dump = getattr(value, "model_dump", None)
    if callable(dump) and not isinstance(value, type):
        try:
            return _strict_json_bytes(dump(), depth + 1)
        except Exception:
            return None
    return None


def _chat_input_upper_bound(view: Mapping[str, Any]) -> Optional[int]:
    """An upper bound on a Chat Completions request's input tokens, or None if it has none."""
    messages = view.get("messages") or []
    if not isinstance(messages, (list, tuple)):
        return None
    for message in messages:
        audio = message.get("audio") if isinstance(message, Mapping) else getattr(message, "audio", None)
        if audio is not None:
            return None  # a reference to earlier audio output: its tokens are not in this request
        content = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
        if isinstance(content, (list, tuple)):
            for part in content:
                kind = part.get("type") if isinstance(part, Mapping) else getattr(part, "type", None)
                if kind not in _CHAT_TEXT_PART_TYPES:
                    return None  # image, audio or file: tokens are not bounded by bytes
    body = {key: value for key, value in view.items() if key != "model"}
    response_format = body.get("response_format")
    if isinstance(response_format, type):
        converted = _openai_response_format(response_format)
        if converted is None:
            return None
        body["response_format"] = converted
    size = _strict_json_bytes({key: value for key, value in body.items() if key not in _CHAT_RENDERED_FIELDS})
    rendered = _strict_json_bytes({key: value for key, value in body.items() if key in _CHAT_RENDERED_FIELDS})
    if size is None or rendered is None:
        return None
    tools = view.get("tools") or view.get("functions") or []
    tool_count = len(tools) if isinstance(tools, (list, tuple)) else 0
    return (
        size
        + _CHAT_BOUND_RENDERED_FACTOR * rendered
        + _CHAT_BOUND_FRAMING
        + _CHAT_BOUND_PER_MESSAGE * len(messages)
        + _CHAT_BOUND_PER_TOOL * tool_count
    )


def _require_chat_below_long_band(view: Mapping[str, Any], *, provider: str, canonical: str, images: list) -> None:
    """Allow Chat Completions on a long-context model only when the request provably stays short."""
    threshold = _select_row(provider, canonical, "short").long_context_threshold
    bound = None if images else _chat_input_upper_bound(view)
    if threshold is None or bound is None or bound > int(threshold):
        _refuse(
            "Chat Completions has no exact input counter, and this request could reach the model's "
            "long-context price band (images, audio, or very long input). Use the Responses API.",
            "long_context_chat_unsupported",
        )


def _check_breaker(org: str, provider: str, model: str) -> None:
    now = time.time()
    for key, entry in list(_TIER_BREAKER.items()):
        if key[0] != org or key[1] != provider or key[2] != model:
            continue
        state = unresolved_tier_state(float(entry["opened_at"]), now=now)
        entry["state"] = state
        _refuse(
            f"admission is closed for {provider} {model} tier {key[3]!r} until an operator resolves it "
            f"(state {state}).",
            "unknown_tier_circuit_open",
        )


def _open_breaker(
    *,
    org: str,
    provider: str,
    model: str,
    tier: str,
    held_micro_usd: int,
) -> None:
    key = _breaker_key(org, provider, model, tier)
    if key not in _TIER_BREAKER:
        _TIER_BREAKER[key] = {
            "opened_at": time.time(),
            "held_micro_usd": int(held_micro_usd),
            "state": "pending_reconciliation",
        }
    # The existing webhook alert cannot address both the org admin and the
    # operator. One audit row and one log line, no mailer.
    row = {
        "kind": "unknown_tier",
        "org_id": org,
        "provider": provider,
        "model": model,
        "tier": tier,
        "held_micro_usd": int(held_micro_usd),
        "state": "pending_reconciliation",
    }
    _TIER_AUDIT.append(row)
    _log.warning(
        "unknown service tier held for operator reconciliation org=%s provider=%s model=%s tier=%s held_micro_usd=%s",
        org,
        provider,
        model,
        tier,
        held_micro_usd,
    )


def _multiplier() -> Decimal:
    import os

    raw = os.getenv("KAZENAI_TIER_MISMATCH_MULTIPLIER", "").strip()
    if not raw:
        from .lifecycle_config import control_profile_active

        if control_profile_active():
            raise PricingConfigError(
                "KAZENAI_TIER_MISMATCH_MULTIPLIER is required under Control when a reported "
                "tier has no certified rate vector"
            )
        return Decimal(1)
    try:
        number = Decimal(raw)
    except Exception:
        raise PricingConfigError("KAZENAI_TIER_MISMATCH_MULTIPLIER is not a finite decimal") from None
    if not number.is_finite() or number < 1:
        raise PricingConfigError("KAZENAI_TIER_MISMATCH_MULTIPLIER must be finite and >= 1")
    return number


def _catalog_table():
    from .catalog_feed import ensure_current
    from .cost_engine import TokenCostEngine, active_catalog

    # Live signed catalog from FinOps when configured; never blocks or raises.
    ensure_current()
    TokenCostEngine()
    return active_catalog()


def _resolve_canonical(presented: str, aliases: Mapping[str, str]) -> tuple[str, str]:
    from .pricing_policy import resolve_model_key

    table = _catalog_table()
    known = {row.model for row in table.values()}
    canonical = resolve_model_key(presented, known, aliases)
    if canonical is None:
        from .pricing_policy import enforce_unknown_model

        enforce_unknown_model(presented, known_keys=set(known), aliases=aliases)
        return presented, presented
    return presented, canonical


def _aliases() -> dict[str, str]:
    from .cost_engine import operator_aliases

    from .price_catalog import active_bundle

    merged = dict(active_bundle().aliases)
    merged.update(operator_aliases())
    return merged


def _select_row(provider: str, canonical: str, band: str) -> CatalogRow:
    table = _catalog_table()
    row = lookup_row(provider, canonical, "standard", band, table=table)
    if row is None:
        from .pricing_policy import UnknownModelError

        raise UnknownModelError(f"no price row for {(provider, canonical, 'standard', band)!r}")
    from .lifecycle_config import control_profile_active

    if control_profile_active() and not row.certified:
        from .pricing_policy import UnknownModelError

        raise UnknownModelError(
            f"uncertified price row for {canonical!r} is refused under Control"
        )
    return row


def _image_tokens_for(provider: str, model: str, images: list[dict]) -> int:
    total = 0
    for image in images:
        encoded = _inline_b64(image)
        file_id = image.get("file_id")
        url = image.get("url")
        source = image.get("source") if isinstance(image.get("source"), Mapping) else None
        if file_id or (isinstance(source, Mapping) and str(source.get("type") or "") in {"url", "file"}):
            _refuse(
                "URL and file_id images are refused without an exact whole-request counter.",
                "image_reference_unsupported",
            )
        if isinstance(url, str) and url and not url.startswith("data:"):
            _refuse(
                "URL images are refused without an exact whole-request counter.",
                "image_reference_unsupported",
            )
        if not encoded:
            _refuse("only inline images can be priced without an exact counter.", "image_not_inline")
        raw = _decode_inline(encoded)
        _fmt, width, height, _frames = _inspect_image_bytes(raw)
        detail = str(image.get("detail") or "auto")
        if provider == "anthropic":
            total += anthropic_image_tokens(model, width, height)
        else:
            try:
                total += openai_image_tokens(model, width, height, detail)
            except PricingConfigError:
                _refuse(f"no certified image formula for {model!r}.", "image_formula_uncertified")
    return total


def _build(
    *,
    presented: str,
    canonical: str,
    provider: str,
    row: CatalogRow,
    n: int,
    input_tokens: int,
    output_tokens: int,
    image_tokens: int,
    count_source: str,
    threshold_decision: str,
    spend_bound: str,
    cache_kind: str,
    basis: str,
    label: str,
    multiplier: Decimal = Decimal("1"),
    modifiers: tuple = (),
) -> CallPricing:
    rates = row.rates.scaled(multiplier)
    write_5m = input_tokens if cache_kind == "5m" else 0
    write_1h = input_tokens if cache_kind == "1h" else 0
    uncached = 0 if cache_kind in {"5m", "1h"} else input_tokens
    micros = cost_micro_usd(
        rates,
        input_tokens=uncached,
        cache_write_5m_tokens=write_5m,
        cache_write_1h_tokens=write_1h,
        output_tokens=output_tokens,
        image_tokens=image_tokens,
    )
    if micros <= 0 and (input_tokens or output_tokens or image_tokens):
        micros = 0
    decision = PricingDecision(
        catalog_version=row.catalog_version,
        provider=provider,
        canonical_model_id=canonical,
        presented_model_id=presented,
        request_tier=row.request_tier,
        expected_reported_tier=row.expected_reported_tier,
        context_band=row.context_band,
        rate_vector=rates,
        rounding_rule=ROUNDING_RULE,
        pricing_source=row.source_url,
        pricing_basis=basis,
        label=label,
        input_token_count_source=count_source,
        pricing_rule=row.pricing_rule,
        long_context_threshold=row.long_context_threshold,
        threshold_decision=threshold_decision,
        spend_bound=spend_bound,
        n=n,
        reserved_micro_usd=micros,
        cache_write_kind=cache_kind,
        certified=row.certified,
        verified_on=row.verified_on,
        projected_input_tokens=int(input_tokens),
        projected_output_tokens=int(output_tokens),
        projected_image_tokens=int(image_tokens),
        price_multiplier=multiplier,
        price_modifiers=tuple(modifiers),
    )
    pricing = CallPricing(decision)
    set_current_pricing(pricing)
    return pricing


def _prepare(
    kwargs: dict,
    *,
    provider: str,
    surface: str,
    client: Any,
    org_id: str,
    observe_unbounded: tuple,
    counted_input: Optional[int],
) -> CallPricing:
    _pin_service_tier(kwargs, provider=provider)
    view = _effective_request(kwargs)
    _check_breaker(org_id, provider, str(view.get("model") or ""))
    n = _integer_n(view.get("n"))
    _validate_tools(view, provider=provider, surface=surface)
    _validate_request_modifiers(view, provider=provider)
    images, document = _collect_images(view)
    if document:
        _refuse(
            "files and documents sent by reference are refused, including under observe_unbounded.",
            "document_reference_unsupported",
        )
    observe_images = "images" in set(observe_unbounded or ())
    if images and not observe_images:
        # Local shape checks before the counter, including Pillow metadata.
        if not _client_has_exact_counter(client, provider=provider, surface=surface):
            pass
        else:
            for image in images:
                if image.get("file_id"):
                    continue
                url = image.get("url")
                source = image.get("source") if isinstance(image.get("source"), Mapping) else None
                if isinstance(url, str) and url.startswith(("http://", "https://")):
                    continue
                if isinstance(source, Mapping) and str(source.get("type") or "") == "url":
                    continue
                encoded = _inline_b64(image)
                if encoded:
                    _inspect_image_bytes(_decode_inline(encoded))
    presented = str(view.get("model") or "").strip()
    _presented, canonical = _resolve_canonical(presented.lower(), _aliases())
    table = _catalog_table()
    resolved_provider = provider_for_model(canonical, table=table)
    if resolved_provider not in (None, provider):
        provider = resolved_provider
    needs_counter = model_declares_long_band(provider, canonical, "standard", table=table)
    if needs_counter and surface == "chat":
        _require_chat_below_long_band(view, provider=provider, canonical=canonical, images=images)
        needs_counter = False  # provably below the threshold: the standard band applies
    count_source = "local_byte_estimate"
    threshold_decision = "short"
    input_tokens = _text_input_tokens(view)
    if needs_counter or (images and not observe_images and _client_has_exact_counter(client, provider=provider, surface=surface)):
        if needs_counter and not _client_has_exact_counter(client, provider=provider, surface=surface):
            _refuse(
                "this path requires the provider exact input counter, which this client does not expose.",
                "input_counter_unavailable",
            )
        if counted_input is None and (needs_counter or images):
            if needs_counter or (images and _client_has_exact_counter(client, provider=provider, surface=surface)):
                if not _client_has_exact_counter(client, provider=provider, surface=surface):
                    _refuse(
                        "exact input counter is required and this client does not expose it.",
                        "input_counter_unavailable",
                    )
                counted_input = _call_counter(client, view, provider=provider, surface=surface)
        if counted_input is not None:
            input_tokens = int(counted_input)
            count_source = "provider_exact_counter"
    if provider == "anthropic" and count_source == "local_byte_estimate":
        # The API adds a tool-use system prompt and client-tool definitions the
        # request bytes do not show. An exact counter already includes them.
        input_tokens += anthropic_tool_overhead_tokens(canonical, view.get("tools") or (), view.get("tool_choice"))
    multiplier, modifiers = _price_multiplier(view, provider=provider, client=client)
    image_tokens = 0
    spend_bound = "bounded"
    if images and observe_images:
        spend_bound = "none"
        image_tokens = 0
    elif images and count_source != "provider_exact_counter":
        image_tokens = _image_tokens_for(provider, canonical, images)
    row = _select_row(provider, canonical, "short")
    if row.long_context_threshold is not None and input_tokens > int(row.long_context_threshold):
        long_row = _select_row(provider, canonical, "long")
        row = long_row
        threshold_decision = "long"
    elif row.long_context_threshold is not None:
        threshold_decision = "short"
    else:
        threshold_decision = "no_long_band"
    cache_kind = _cache_write_kind(view)
    output_tokens = _output_tokens(view, n)
    basis = "operator_override" if row.label == "operator_override" else "catalog"
    return _build(
        presented=presented,
        canonical=canonical,
        provider=provider,
        row=row,
        n=n,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        image_tokens=image_tokens,
        count_source=count_source,
        threshold_decision=threshold_decision,
        spend_bound=spend_bound,
        cache_kind=cache_kind,
        basis=basis,
        label=row.label,
        multiplier=multiplier,
        modifiers=modifiers,
    )


def prepare_provider_call(
    kwargs: dict,
    *,
    provider: str,
    surface: str,
    client: Any = None,
    org_id: str = "",
    observe_unbounded: tuple = (),
) -> CallPricing:
    return _prepare(
        kwargs,
        provider=provider,
        surface=surface,
        client=client,
        org_id=org_id,
        observe_unbounded=observe_unbounded,
        counted_input=None,
    )


async def prepare_provider_call_async(
    kwargs: dict,
    *,
    provider: str,
    surface: str,
    client: Any = None,
    org_id: str = "",
    observe_unbounded: tuple = (),
) -> CallPricing:
    # Local refusals first, via the sync validator's checks, but the counter
    # itself must be awaited. Run local checks by calling the pieces, then count.
    _pin_service_tier(kwargs, provider=provider)
    view = _effective_request(kwargs)
    _check_breaker(org_id, provider, str(view.get("model") or ""))
    _integer_n(view.get("n"))
    _validate_tools(view, provider=provider, surface=surface)
    _validate_request_modifiers(view, provider=provider)
    images, document = _collect_images(view)
    if document:
        _refuse(
            "files and documents sent by reference are refused, including under observe_unbounded.",
            "document_reference_unsupported",
        )
    presented = str(view.get("model") or "").strip()
    _presented, canonical = _resolve_canonical(presented.lower(), _aliases())
    table = _catalog_table()
    needs_counter = model_declares_long_band(provider, canonical, "standard", table=table)
    counted: Optional[int] = None
    observe_images = "images" in set(observe_unbounded or ())
    if needs_counter and surface == "chat":
        _require_chat_below_long_band(view, provider=provider, canonical=canonical, images=images)
        needs_counter = False  # provably below the threshold: the standard band applies
    use_counter = needs_counter or (images and not observe_images and _client_has_exact_counter(client, provider=provider, surface=surface))
    if use_counter:
        if not _client_has_exact_counter(client, provider=provider, surface=surface):
            _refuse(
                "this path requires the provider exact input counter, which this client does not expose.",
                "input_counter_unavailable",
            )
        counted = await _call_counter_async(client, view, provider=provider, surface=surface)
    # Reuse the sync builder with the already-counted input so it does not call again.
    return _prepare(
        kwargs,
        provider=provider,
        surface=surface,
        client=client,
        org_id=org_id,
        observe_unbounded=observe_unbounded,
        counted_input=counted,
    )


def _usage_field(usage: Any, *names: str) -> int:
    for name in names:
        value = None
        if isinstance(usage, Mapping):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


def _nested(usage: Any, *path: str) -> int:
    current = usage
    for part in path:
        if current is None:
            return 0
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    try:
        return int(current or 0)
    except (TypeError, ValueError):
        return 0


def usage_components(usage: Any) -> dict[str, int]:
    prompt = _usage_field(usage, "prompt_tokens", "input_tokens")
    output = _usage_field(usage, "completion_tokens", "output_tokens")
    cached = _nested(usage, "prompt_tokens_details", "cached_tokens")
    if cached == 0:
        cached = _nested(usage, "input_tokens_details", "cached_tokens")
    if cached == 0:
        cached = _usage_field(usage, "cache_read_input_tokens")
    audio = _nested(usage, "prompt_tokens_details", "audio_tokens")
    if audio == 0:
        audio = _nested(usage, "input_tokens_details", "audio_tokens")
    write_5m = _nested(usage, "cache_creation", "ephemeral_5m_input_tokens")
    write_1h = _nested(usage, "cache_creation", "ephemeral_1h_input_tokens")
    creation = _usage_field(usage, "cache_creation_input_tokens")
    if write_5m == 0 and write_1h == 0 and creation:
        write_5m = creation
    # OpenAI prompt_tokens already include cached tokens. Anthropic input_tokens
    # are the uncached portion. When cache_read is present beside input_tokens
    # and prompt_tokens is absent, do not subtract.
    if isinstance(usage, Mapping):
        has_prompt = "prompt_tokens" in usage or "input_tokens_details" in usage or "prompt_tokens_details" in usage
    else:
        has_prompt = getattr(usage, "prompt_tokens", None) is not None or getattr(usage, "prompt_tokens_details", None) is not None
    if has_prompt and cached:
        uncached = max(0, prompt - cached)
    elif _usage_field(usage, "cache_read_input_tokens") or creation or write_5m or write_1h:
        uncached = prompt
    else:
        uncached = prompt
        cached = 0
    # Audio input is reported inside the prompt total; bill it at the audio rate.
    audio = min(max(0, audio), uncached)
    return {
        "input_tokens": uncached - audio,
        "cached_input_tokens": cached,
        "cache_write_5m_tokens": write_5m,
        "cache_write_1h_tokens": write_1h,
        "output_tokens": output,
        "audio_input_tokens": audio,
    }


def reported_service_tier(usage: Any, response: Any = None) -> Optional[str]:
    for source in (usage, response):
        if source is None:
            continue
        value = source.get("service_tier") if isinstance(source, Mapping) else getattr(source, "service_tier", None)
        if value:
            return str(value)
    return None


def settle_usage(pricing: CallPricing, usage: Any, *, reported_tier: Optional[str] = None, org_id: str = "") -> int:
    """Micro-USD for this usage on the frozen basis. May hold a tier mismatch."""
    decision = pricing.decision
    tier = reported_tier if reported_tier is not None else reported_service_tier(usage)
    rates = decision.rate_vector
    pricing.exposure_source = "certified_row"
    if tier and tier != decision.expected_reported_tier:
        table = _catalog_table()
        other = lookup_row(decision.provider, decision.canonical_model_id, tier, decision.context_band, table=table)
        if other is None:
            other = lookup_row(decision.provider, decision.canonical_model_id, tier, "short", table=table)
        if other is not None and other.certified:
            rates = other.rates.scaled(decision.price_multiplier)
            pricing.hold_for_reconciliation = True
            pricing.reservation_state = "pending_reconciliation"
            pricing.exposure_source = "certified_unexpected_tier"
            pricing.usage_evidence = {
                "service_tier": tier,
                "provider": decision.provider,
                "canonical_model_id": decision.canonical_model_id,
                "exposure_source": "certified_unexpected_tier",
                "catalog_version": other.catalog_version,
                "pricing_rule": other.pricing_rule,
            }
            _open_breaker(
                org=org_id,
                provider=decision.provider,
                model=decision.canonical_model_id,
                tier=tier,
                held_micro_usd=0,
            )
        else:
            multiplier = _multiplier()
            components = usage_components(usage)
            pricing.settled_components = dict(components)
            base = cost_micro_usd(decision.rate_vector, **components)
            held = decimal_to_micro_usd(micro_usd_to_decimal(base) * multiplier)
            if held < decision.reserved_micro_usd:
                held = decision.reserved_micro_usd
            pricing.hold_for_reconciliation = True
            pricing.reservation_state = "pending_reconciliation"
            pricing.pricing_basis = "unknown_tier"
            pricing.exposure_source = "operator_risk_multiplier"
            pricing.held_exposure_usd = float(micro_usd_to_decimal(held))
            pricing.usage_evidence = {
                "service_tier": tier,
                "provider": decision.provider,
                "canonical_model_id": decision.canonical_model_id,
                "exposure_source": "operator_risk_multiplier",
                "pricing_basis": "unknown_tier",
                "label": "estimate",
                "multiplier": format(multiplier, "f"),
            }
            _open_breaker(
                org=org_id,
                provider=decision.provider,
                model=decision.canonical_model_id,
                tier=tier,
                held_micro_usd=held,
            )
            return held
    components = usage_components(usage)
    pricing.settled_components = dict(components)
    micros = cost_micro_usd(rates, **components)
    if pricing.hold_for_reconciliation:
        pricing.held_exposure_usd = float(micro_usd_to_decimal(micros))
    return micros


def settlement_evidence(pricing: Optional[CallPricing]) -> dict[str, Any]:
    """``usage_evidence`` for a settle body: the priced token components and modifiers.

    FinOps re-prices these tokens on the reservation's frozen rate card and keeps
    the higher of its figure and the SDK's, so a stale SDK cannot under-charge.
    Tier-mismatch evidence rides along so FinOps can open its durable breaker.
    """
    if pricing is None:
        return {}
    out: dict[str, Any] = dict(pricing.usage_evidence or {})
    if pricing.settled_components:
        out["tokens"] = {key: int(value) for key, value in pricing.settled_components.items()}
    decision = pricing.decision
    if decision.price_multiplier != 1:
        out["price_multiplier"] = format(decision.price_multiplier, "f")
        out["price_modifiers"] = list(decision.price_modifiers)
    out.setdefault("catalog_version", decision.catalog_version)
    return out


def attach_pricing(reserved: Any, pricing: Optional[CallPricing]) -> None:
    if reserved is None or pricing is None:
        return
    try:
        setattr(reserved, "pricing_call", pricing)
    except Exception:
        return


def pricing_evidence(pricing: Optional[CallPricing]) -> dict[str, Any]:
    if pricing is None:
        return {}
    fields = pricing.decision.reserve_body_fields()
    if pricing.usage_evidence:
        fields["usage_evidence"] = dict(pricing.usage_evidence)
    fields["reservation_state"] = pricing.reservation_state
    fields["pricing_basis"] = pricing.pricing_basis
    return fields
