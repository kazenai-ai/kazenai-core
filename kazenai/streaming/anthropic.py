"""Anthropic Messages streaming helpers."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from .lifecycle import StreamOutcome, StreamUsage
from .proxy import StreamIteratorProxy, StreamManagerProxy


SURFACE_CREATE_STREAM = "anthropic.messages.create.stream"
SURFACE_MANAGER_STREAM = "anthropic.messages.stream"


def _usage_from_obj(
    usage: Any,
    *,
    authoritative: bool = True,
) -> Optional[StreamUsage]:
    if usage is None:
        return None

    def _field(*names: str) -> Optional[int]:
        for name in names:
            val = getattr(usage, name, None)
            if val is None and isinstance(usage, dict):
                val = usage.get(name)
            if val is not None:
                return int(val)
        return None

    inp = _field("input_tokens")
    out = _field("output_tokens")
    total = None
    if inp is not None and out is not None:
        total = inp + out
    raw = {
        key: value
        for key, value in {
            "input_tokens": inp,
            "output_tokens": out,
            "cache_creation_input_tokens": _field("cache_creation_input_tokens"),
            "cache_read_input_tokens": _field("cache_read_input_tokens"),
        }.items()
        if value is not None
    }
    return StreamUsage(
        input_tokens=inp,
        output_tokens=out,
        prompt_tokens=inp,
        completion_tokens=out,
        total_tokens=total,
        raw=raw,
        # A terminal Anthropic usage event may contain only the cumulative
        # output count; StreamUsage.merged combines it with message_start input
        # usage before deciding whether the overall snapshot is complete.
        authoritative=bool(authoritative and (inp is not None or out is not None)),
    )


def observe_anthropic_event(event: Any) -> Tuple[str, Optional[StreamUsage]]:
    text = ""
    usage = None
    try:
        etype = getattr(event, "type", None) or (event.get("type") if isinstance(event, dict) else None)
        if etype == "content_block_delta":
            delta = getattr(event, "delta", None) or (event.get("delta") if isinstance(event, dict) else None)
            if delta is not None:
                parts: list[str] = []
                # Text, tool JSON, extended thinking and signatures all consume
                # output even though only text is user-visible.
                for name in ("text", "partial_json", "thinking", "signature"):
                    value = getattr(delta, name, None)
                    if value is None and isinstance(delta, dict):
                        value = delta.get(name)
                    if value:
                        parts.append(str(value))
                text = "".join(parts)
        elif etype in ("message_start", "message_delta", "message_stop"):
            msg = getattr(event, "message", None) or (event.get("message") if isinstance(event, dict) else None)
            u = getattr(event, "usage", None) or (event.get("usage") if isinstance(event, dict) else None)
            if u is None and msg is not None:
                u = getattr(msg, "usage", None) or (msg.get("usage") if isinstance(msg, dict) else None)
            usage = _usage_from_obj(
                u,
                # ``message_start`` usage is an initial snapshot. Its output
                # count is not final even when both counters are present.
                authoritative=etype in ("message_delta", "message_stop"),
            )
    except Exception:
        pass
    return text, usage


def _final_usage_from_stream(stream: Any) -> Optional[StreamUsage]:
    # Prefer get_final_message only when the stream is already exhausted/complete.
    get_final = getattr(stream, "get_final_message", None)
    if callable(get_final):
        try:
            msg = get_final()
            return _usage_from_obj(getattr(msg, "usage", None))
        except Exception:
            return None
    return None


def wrap_anthropic_event_stream(upstream: Any, attempt: Any) -> StreamIteratorProxy:
    return StreamIteratorProxy(
        upstream,
        attempt,
        observe_chunk=observe_anthropic_event,
        extract_final_usage=_final_usage_from_stream,
    )


class AnthropicEnteredStreamProxy(StreamIteratorProxy):
    """Preserves text_stream / get_final_message / until_done on entered streams."""

    def get_final_message(self) -> Any:
        fn = getattr(self._upstream, "get_final_message", None)
        if not callable(fn):
            raise AttributeError("get_final_message")
        try:
            msg = fn()
        except Exception as exc:
            if not self._attempt.finalized:
                self._attempt.finalize(StreamOutcome.PROVIDER_ERROR, error=exc)
            raise
        usage = _usage_from_obj(getattr(msg, "usage", None))
        if not self._attempt.finalized:
            self._attempt.finalize(StreamOutcome.COMPLETE, usage=usage)
        return msg

    def get_final_text(self) -> Any:
        fn = getattr(self._upstream, "get_final_text", None)
        if not callable(fn):
            raise AttributeError("get_final_text")
        try:
            result = fn()
        except Exception as exc:
            if not self._attempt.finalized:
                self._attempt.finalize(StreamOutcome.PROVIDER_ERROR, error=exc)
            raise
        if not self._attempt.finalized:
            self._attempt.finalize(
                StreamOutcome.COMPLETE,
                usage=_final_usage_from_stream(self._upstream),
            )
        return result

    def until_done(self) -> Any:
        fn = getattr(self._upstream, "until_done", None)
        if not callable(fn):
            raise AttributeError("until_done")
        try:
            result = fn()
        except Exception as exc:
            if not self._attempt.finalized:
                self._attempt.finalize(StreamOutcome.PROVIDER_ERROR, error=exc)
            raise
        if not self._attempt.finalized:
            self._attempt.finalize(
                StreamOutcome.COMPLETE,
                usage=_final_usage_from_stream(self._upstream),
            )
        return result

    @property
    def text_stream(self) -> Any:
        upstream_ts = getattr(self._upstream, "text_stream", None)
        if upstream_ts is None:
            raise AttributeError("text_stream")

        attempt = self._attempt
        parent = self

        class _TextStream:
            def __iter__(self_inner):
                completed = False
                try:
                    for text in upstream_ts:
                        attempt.note_first_byte()
                        attempt.observe_text_estimate(str(text or ""))
                        if attempt.should_cutoff():
                            parent._close_upstream()
                            attempt.finalize(StreamOutcome.CUTOFF)
                            from ..enforcement import StreamCutoffError

                            raise StreamCutoffError(
                                "stream cutoff: local stream_cutoff_usd exceeded",
                                blocked_usd=attempt.estimated_spend_usd,
                                total_tokens=attempt.estimated_output_tokens,
                            )
                        yield text
                    completed = True
                    if not attempt.finalized:
                        usage = _final_usage_from_stream(parent._upstream)
                        attempt.finalize(StreamOutcome.COMPLETE, usage=usage)
                except Exception as exc:
                    from ..enforcement import StreamCutoffError

                    if isinstance(exc, StreamCutoffError):
                        raise
                    parent._close_upstream()
                    if not attempt.finalized:
                        attempt.finalize(StreamOutcome.PROVIDER_ERROR, error=exc)
                    raise
                finally:
                    if not completed and not attempt.finalized:
                        parent.close()

        return _TextStream()


def wrap_anthropic_entered(entered: Any, attempt: Any) -> AnthropicEnteredStreamProxy:
    return AnthropicEnteredStreamProxy(
        entered,
        attempt,
        observe_chunk=observe_anthropic_event,
        extract_final_usage=_final_usage_from_stream,
    )


def wrap_anthropic_manager(
    manager: Any,
    attempt: Any = None,
    *,
    attempt_factory: Any = None,
    enter_scope: Any = None,
) -> StreamManagerProxy:
    return StreamManagerProxy(
        manager,
        attempt,
        attempt_factory=attempt_factory,
        wrap_entered_stream=wrap_anthropic_entered,
        enter_scope=enter_scope,
    )
