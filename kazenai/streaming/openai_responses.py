"""OpenAI Responses API streaming helpers.

Responses streams carry usage only on the terminal event (``response.completed``,
``response.incomplete`` or ``response.failed``) as ``event.response.usage``; there
is no ``stream_options.include_usage``. Reasoning tokens are counted inside
``output_tokens``.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from .lifecycle import StreamOutcome, StreamUsage
from .proxy import (
    AsyncStreamIteratorProxy,
    AsyncStreamManagerProxy,
    StreamIteratorProxy,
    StreamManagerProxy,
)


SURFACE_CREATE_STREAM = "openai.responses.create.stream"
SURFACE_MANAGER_STREAM = "openai.responses.stream"

_TERMINAL_EVENTS = frozenset({"response.completed", "response.incomplete", "response.failed"})

# Deltas of generated output (text, tool arguments, refusals, reasoning). Used only
# for the local stream_cutoff_usd estimate, never for settlement.
_OUTPUT_DELTA_EVENTS = frozenset(
    {
        "response.output_text.delta",
        "response.refusal.delta",
        "response.function_call_arguments.delta",
        "response.custom_tool_call_input.delta",
        "response.reasoning_text.delta",
        "response.reasoning_summary_text.delta",
    }
)


def _value(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, dict):
        value = obj.get(name)
    return value


def usage_from_response_usage(usage: Any) -> Optional[StreamUsage]:
    """Convert ``ResponseUsage`` to a StreamUsage; authoritative when both counts exist."""
    if usage is None:
        return None

    def _int(obj: Any, name: str) -> Optional[int]:
        value = _value(obj, name)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    inp = _int(usage, "input_tokens")
    out = _int(usage, "output_tokens")
    total = _int(usage, "total_tokens")
    if total is None and inp is not None and out is not None:
        total = inp + out
    raw = {
        key: value
        for key, value in {
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": total,
            "cached_tokens": _int(_value(usage, "input_tokens_details"), "cached_tokens"),
            "reasoning_tokens": _int(_value(usage, "output_tokens_details"), "reasoning_tokens"),
        }.items()
        if value is not None
    }
    return StreamUsage(
        prompt_tokens=inp,
        completion_tokens=out,
        input_tokens=inp,
        output_tokens=out,
        total_tokens=total,
        raw=raw,
        authoritative=inp is not None and out is not None,
    )


def observe_responses_event(event: Any) -> Tuple[str, Optional[StreamUsage]]:
    """Observe one Responses stream event (raw SSE or ``responses.stream`` helper)."""
    try:
        etype = _value(event, "type")
        if etype in _OUTPUT_DELTA_EVENTS:
            delta = _value(event, "delta")
            return (str(delta) if delta else ""), None
        if etype in _TERMINAL_EVENTS:
            # A failed response without usage stays unresolved (pending), never $0.
            return "", usage_from_response_usage(_value(_value(event, "response"), "usage"))
    except Exception:
        pass
    return "", None


def wrap_responses_stream(upstream: Any, attempt: Any) -> StreamIteratorProxy:
    return StreamIteratorProxy(
        upstream,
        attempt,
        observe_chunk=observe_responses_event,
        extract_final_usage=None,
    )


class ResponsesEnteredStreamProxy(StreamIteratorProxy):
    """Keeps the official helper's ``until_done`` / ``get_final_response``."""

    def until_done(self) -> "ResponsesEnteredStreamProxy":
        for _event in self:
            pass
        return self

    def get_final_response(self) -> Any:
        self.until_done()
        fn = getattr(self._upstream, "get_final_response", None)
        if not callable(fn):
            raise AttributeError("get_final_response")
        try:
            return fn()
        except Exception as exc:
            if not self._attempt.finalized:
                self._attempt.finalize(StreamOutcome.PROVIDER_ERROR, error=exc)
            raise


def wrap_responses_entered_helper(entered: Any, attempt: Any) -> ResponsesEnteredStreamProxy:
    return ResponsesEnteredStreamProxy(
        entered,
        attempt,
        observe_chunk=observe_responses_event,
        extract_final_usage=None,
    )


def wrap_responses_manager(
    manager: Any,
    *,
    attempt_factory: Any,
    enter_scope: Any = None,
) -> StreamManagerProxy:
    return StreamManagerProxy(
        manager,
        attempt_factory=attempt_factory,
        wrap_entered_stream=wrap_responses_entered_helper,
        enter_scope=enter_scope,
    )


def wrap_responses_async_stream(upstream: Any, attempt: Any) -> AsyncStreamIteratorProxy:
    return AsyncStreamIteratorProxy(
        upstream,
        attempt,
        observe_chunk=observe_responses_event,
        extract_final_usage=None,
    )


class ResponsesAsyncEnteredStreamProxy(AsyncStreamIteratorProxy):
    async def until_done(self) -> "ResponsesAsyncEnteredStreamProxy":
        async for _event in self:
            pass
        return self

    async def get_final_response(self) -> Any:
        await self.until_done()
        fn = getattr(self._upstream, "get_final_response", None)
        if not callable(fn):
            raise AttributeError("get_final_response")
        try:
            result = fn()
            if hasattr(result, "__await__"):
                return await result
            return result
        except Exception as exc:
            if not self._attempt.finalized:
                from .proxy import _finalize_attempt

                await _finalize_attempt(self._attempt, StreamOutcome.PROVIDER_ERROR, error=exc)
            raise


def wrap_responses_async_entered_helper(entered: Any, attempt: Any) -> ResponsesAsyncEnteredStreamProxy:
    return ResponsesAsyncEnteredStreamProxy(
        entered,
        attempt,
        observe_chunk=observe_responses_event,
        extract_final_usage=None,
    )


def wrap_responses_async_manager(
    manager: Any,
    *,
    attempt_factory: Any,
    enter_scope: Any = None,
) -> AsyncStreamManagerProxy:
    return AsyncStreamManagerProxy(
        manager,
        attempt_factory=attempt_factory,
        wrap_entered_stream=wrap_responses_async_entered_helper,
        enter_scope=enter_scope,
    )
