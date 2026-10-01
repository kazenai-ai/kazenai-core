"""OpenAI Chat Completions streaming helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from .lifecycle import StreamOutcome, StreamUsage
from .proxy import StreamIteratorProxy, StreamManagerProxy


SURFACE_CREATE_STREAM = "openai.chat.completions.create.stream"
SURFACE_HELPER_STREAM = "openai.chat.completions.stream"


def merge_stream_options(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Copy kwargs and ensure stream_options.include_usage without dropping caller options."""
    out = dict(kwargs)
    out["stream"] = True
    existing = out.get("stream_options")
    if existing is None:
        out["stream_options"] = {"include_usage": True}
        return out
    if isinstance(existing, Mapping):
        merged = dict(existing)
        merged["include_usage"] = True
        out["stream_options"] = merged
        return out
    # Unknown type — leave as-is rather than clobber.
    return out


def _value(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, dict):
        value = obj.get(name)
    return value


def _observable_delta(chunk: Any) -> str:
    """Flatten observable billable deltas for local cutoff estimation only."""
    try:
        choices = _value(chunk, "choices")
        if not choices:
            return ""
        first = choices[0]
        delta = _value(first, "delta")
        if delta is None:
            return ""
        parts: list[str] = []
        for name in ("content", "refusal"):
            value = _value(delta, name)
            if value:
                parts.append(str(value))

        function_call = _value(delta, "function_call")
        if function_call is not None:
            for name in ("name", "arguments"):
                value = _value(function_call, name)
                if value:
                    parts.append(str(value))

        tool_calls = _value(delta, "tool_calls") or []
        for tool_call in tool_calls:
            function = _value(tool_call, "function")
            if function is None:
                continue
            for name in ("name", "arguments"):
                value = _value(function, name)
                if value:
                    parts.append(str(value))
        return "".join(parts)
    except Exception:
        return ""


def observe_openai_chunk(chunk: Any) -> Tuple[str, Optional[StreamUsage]]:
    text = _observable_delta(chunk)
    usage = getattr(chunk, "usage", None)
    if usage is None and isinstance(chunk, dict):
        usage = chunk.get("usage")
    if usage is None:
        return text, None

    def _field(obj: Any, *names: str) -> Optional[int]:
        for name in names:
            val = getattr(obj, name, None)
            if val is None and isinstance(obj, dict):
                val = obj.get(name)
            if val is not None:
                return int(val)
        return None

    prompt_tokens = _field(usage, "prompt_tokens")
    completion_tokens = _field(usage, "completion_tokens")
    total_tokens = _field(usage, "total_tokens")
    raw = {
        key: value
        for key, value in {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }.items()
        if value is not None
    }
    return text, StreamUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total_tokens,
        raw=raw,
        authoritative=bool(
            total_tokens is not None
            or (prompt_tokens is not None and completion_tokens is not None)
        ),
    )


def observe_openai_helper_event(event: Any) -> Tuple[str, Optional[StreamUsage]]:
    """Observe the SDK ``chat.completions.stream`` event without double counting.

    The helper emits a ``chunk`` event plus convenience events derived from the
    same chunk. Only the raw chunk event is authoritative for usage and local
    output estimation.
    """
    if _value(event, "type") != "chunk":
        return "", None
    chunk = _value(event, "chunk")
    if chunk is None:
        return "", None
    return observe_openai_chunk(chunk)


def _final_usage_from_helper(stream: Any) -> Optional[StreamUsage]:
    get_final = getattr(stream, "get_final_completion", None)
    if not callable(get_final):
        return None
    try:
        completion = get_final()
    except Exception:
        return None
    _text, usage = observe_openai_chunk({"choices": [], "usage": _value(completion, "usage")})
    return usage


def wrap_openai_stream(upstream: Any, attempt: Any) -> StreamIteratorProxy:
    return StreamIteratorProxy(
        upstream,
        attempt,
        observe_chunk=observe_openai_chunk,
        extract_final_usage=None,
    )


class OpenAIEnteredStreamProxy(StreamIteratorProxy):
    """Preserve the official helper's completion methods through KazenAI."""

    def until_done(self) -> "OpenAIEnteredStreamProxy":
        for _event in self:
            pass
        return self

    def get_final_completion(self) -> Any:
        self.until_done()
        fn = getattr(self._upstream, "get_final_completion", None)
        if not callable(fn):
            raise AttributeError("get_final_completion")
        try:
            return fn()
        except Exception as exc:
            if not self._attempt.finalized:
                self._attempt.finalize(StreamOutcome.PROVIDER_ERROR, error=exc)
            raise


def wrap_openai_entered_helper(entered: Any, attempt: Any) -> OpenAIEnteredStreamProxy:
    return OpenAIEnteredStreamProxy(
        entered,
        attempt,
        observe_chunk=observe_openai_helper_event,
        extract_final_usage=_final_usage_from_helper,
    )


def wrap_openai_manager(
    manager: Any,
    *,
    attempt_factory: Any,
    enter_scope: Any = None,
) -> StreamManagerProxy:
    return StreamManagerProxy(
        manager,
        attempt_factory=attempt_factory,
        wrap_entered_stream=wrap_openai_entered_helper,
        enter_scope=enter_scope,
    )
