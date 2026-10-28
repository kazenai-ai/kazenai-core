"""Iterator and context-manager proxies that finalize a StreamAttempt exactly once."""

from __future__ import annotations

import inspect
from contextlib import asynccontextmanager, nullcontext
from typing import Any, AsyncIterator, Callable, Iterator, Optional

from ..enforcement import StreamCutoffError
from .lifecycle import StreamAttempt, StreamOutcome


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _finalize_attempt(
    attempt: StreamAttempt,
    outcome: StreamOutcome,
    *,
    usage: Any = None,
    error: Optional[BaseException] = None,
) -> None:
    """Prefer async finalize when settlement callbacks are awaitable."""
    afn = getattr(attempt, "finalize_async", None)
    if callable(afn) and (
        getattr(attempt, "reconcile_async_fn", None) is not None
        or getattr(attempt, "pending_async_fn", None) is not None
    ):
        await afn(outcome, usage=usage, error=error)
        return
    attempt.finalize(outcome, usage=usage, error=error)


class StreamIteratorProxy:
    """Wraps a sync stream iterator; observes chunks and finalizes on terminal paths."""

    def __init__(
        self,
        upstream: Any,
        attempt: StreamAttempt,
        *,
        observe_chunk: Any,
        extract_final_usage: Any = None,
    ) -> None:
        self._upstream = upstream
        self._iterator: Optional[Any] = None
        self._attempt = attempt
        self._observe_chunk = observe_chunk
        self._extract_final_usage = extract_final_usage
        self._closed = False
        self._exhausted = False
        self._upstream_entered = False

    def __iter__(self) -> Iterator[Any]:
        # A generator gives the common ``for chunk in stream: break`` path a
        # ``finally`` hook.  Explicit ``next(stream)`` remains supported by
        # __next__, while context-manager/close are still the deterministic
        # cross-interpreter cancellation contract.
        def _iterate() -> Iterator[Any]:
            try:
                while True:
                    try:
                        yield self.__next__()
                    except StopIteration:
                        return
            finally:
                if not self._attempt.finalized:
                    self.close()

        return _iterate()

    def _iter_upstream(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self._upstream)
        return self._iterator

    def __next__(self) -> Any:
        try:
            chunk = next(self._iter_upstream())
        except StopIteration:
            self._exhausted = True
            usage = None
            if self._extract_final_usage is not None:
                try:
                    usage = self._extract_final_usage(self._upstream)
                except Exception:
                    usage = None
            self._attempt.finalize(StreamOutcome.COMPLETE, usage=usage)
            raise
        except StreamCutoffError:
            self._close_upstream()
            self._attempt.finalize(StreamOutcome.CUTOFF)
            raise
        except Exception as exc:
            self._close_upstream()
            self._attempt.finalize(StreamOutcome.PROVIDER_ERROR, error=exc)
            raise

        self._attempt.note_first_byte()
        try:
            text, usage = self._observe_chunk(chunk)
            if text:
                self._attempt.observe_text_estimate(text)
            if usage is not None:
                self._attempt.observe_usage(usage)
        except Exception:
            pass

        if self._attempt.should_cutoff():
            self._close_upstream()
            self._attempt.finalize(StreamOutcome.CUTOFF)
            raise StreamCutoffError(
                "stream cutoff: local stream_cutoff_usd exceeded",
                blocked_usd=self._attempt.estimated_spend_usd,
                total_tokens=self._attempt.estimated_output_tokens,
            )
        return chunk

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_upstream()
        if not self._attempt.finalized:
            self._attempt.finalize(StreamOutcome.CLIENT_CANCELLED)

    def _close_upstream(self) -> None:
        close = getattr(self._upstream, "close", None)
        if not callable(close):
            self._attempt.mark_upstream_close(ok=False)
            return
        try:
            close()
            self._attempt.mark_upstream_close(ok=True)
        except Exception:
            self._attempt.mark_upstream_close(ok=False)

    def __enter__(self) -> "StreamIteratorProxy":
        enter = getattr(self._upstream, "__enter__", None)
        if callable(enter):
            try:
                entered = enter()
                self._upstream_entered = True
                if entered is not None and entered is not self._upstream:
                    self._upstream = entered
                    self._iterator = None
            except Exception as exc:
                self._attempt.finalize(StreamOutcome.PROVIDER_ERROR, error=exc)
                raise
        return self

    def __exit__(self, exc_type, exc, tb) -> Any:
        result = None
        exit_fn = getattr(self._upstream, "__exit__", None)
        if self._upstream_entered and callable(exit_fn):
            try:
                result = exit_fn(exc_type, exc, tb)
                self._attempt.mark_upstream_close(ok=True)
            except Exception as upstream_exc:
                if not self._attempt.finalized:
                    self._attempt.finalize(StreamOutcome.PROVIDER_ERROR, error=upstream_exc)
                raise
            finally:
                self._closed = True
        else:
            self._closed = True
            self._close_upstream()

        if not self._attempt.finalized:
            if exc_type is not None and issubclass(exc_type, StreamCutoffError):
                self._attempt.finalize(StreamOutcome.CUTOFF)
            elif self._exhausted:
                self._attempt.finalize(StreamOutcome.COMPLETE)
            else:
                # Leaving a context without EOF is a consumer cancellation,
                # even when the surrounding block itself did not raise.
                self._attempt.finalize(StreamOutcome.CLIENT_CANCELLED, error=exc)
        return result

    def __del__(self) -> None:
        # Best effort only.  Shared reservations also have server-side expiry;
        # callers should use a context manager or close() for deterministic
        # release on non-CPython runtimes.
        try:
            if not self._attempt.finalized:
                self.close()
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._upstream, name)


class StreamManagerProxy:
    """Wraps Anthropic-style ``messages.stream(...)`` managers."""

    def __init__(
        self,
        manager: Any,
        attempt: Optional[StreamAttempt] = None,
        *,
        attempt_factory: Optional[Callable[[], StreamAttempt]] = None,
        wrap_entered_stream: Any,
        enter_scope: Optional[Callable[[], Any]] = None,
    ) -> None:
        if attempt is None and attempt_factory is None:
            raise TypeError("attempt or attempt_factory is required")
        self._manager = manager
        self._attempt = attempt
        self._attempt_factory = attempt_factory
        self._wrap_entered_stream = wrap_entered_stream
        self._enter_scope = enter_scope
        self._entered: Optional[Any] = None

    def __enter__(self) -> Any:
        if self._attempt is None:
            assert self._attempt_factory is not None
            self._attempt = self._attempt_factory()
        try:
            scope = self._enter_scope() if self._enter_scope is not None else nullcontext()
            with scope:
                entered = self._manager.__enter__()
        except Exception as exc:
            self._attempt.finalize(StreamOutcome.PROVIDER_ERROR, error=exc)
            raise
        self._entered = self._wrap_entered_stream(entered, self._attempt)
        return self._entered

    def __exit__(self, exc_type, exc, tb) -> Any:
        try:
            result = self._manager.__exit__(exc_type, exc, tb)
        except Exception as upstream_exc:
            if self._attempt is not None and not self._attempt.finalized:
                self._attempt.finalize(StreamOutcome.PROVIDER_ERROR, error=upstream_exc)
            raise
        else:
            if self._entered is not None:
                close = getattr(self._entered, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            elif self._attempt is not None and not self._attempt.finalized:
                if exc_type is None:
                    self._attempt.finalize(StreamOutcome.CLIENT_CANCELLED)
                else:
                    self._attempt.finalize(StreamOutcome.CLIENT_CANCELLED, error=exc)
            return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)


class AsyncStreamIteratorProxy:
    """Wraps an async stream iterator; observes chunks and finalizes once."""

    def __init__(
        self,
        upstream: Any,
        attempt: StreamAttempt,
        *,
        observe_chunk: Any,
        extract_final_usage: Any = None,
    ) -> None:
        self._upstream = upstream
        self._iterator: Optional[Any] = None
        self._attempt = attempt
        self._observe_chunk = observe_chunk
        self._extract_final_usage = extract_final_usage
        self._closed = False
        self._exhausted = False
        self._upstream_entered = False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    def _aiter_upstream(self) -> Any:
        if self._iterator is None:
            self._iterator = self._upstream.__aiter__()
        return self._iterator

    async def __anext__(self) -> Any:
        try:
            chunk = await self._aiter_upstream().__anext__()
        except StopAsyncIteration:
            self._exhausted = True
            usage = None
            if self._extract_final_usage is not None:
                try:
                    usage = await _maybe_await(self._extract_final_usage(self._upstream))
                except Exception:
                    usage = None
            await _finalize_attempt(self._attempt, StreamOutcome.COMPLETE, usage=usage)
            raise
        except StreamCutoffError:
            await self._aclose_upstream()
            await _finalize_attempt(self._attempt, StreamOutcome.CUTOFF)
            raise
        except Exception as exc:
            await self._aclose_upstream()
            await _finalize_attempt(
                self._attempt, StreamOutcome.PROVIDER_ERROR, error=exc
            )
            raise

        self._attempt.note_first_byte()
        try:
            text, usage = self._observe_chunk(chunk)
            if text:
                self._attempt.observe_text_estimate(text)
            if usage is not None:
                self._attempt.observe_usage(usage)
        except Exception:
            pass

        if self._attempt.should_cutoff():
            await self._aclose_upstream()
            await _finalize_attempt(self._attempt, StreamOutcome.CUTOFF)
            raise StreamCutoffError(
                "stream cutoff: local stream_cutoff_usd exceeded",
                blocked_usd=self._attempt.estimated_spend_usd,
                total_tokens=self._attempt.estimated_output_tokens,
            )
        return chunk

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._aclose_upstream()
        if not self._attempt.finalized:
            await _finalize_attempt(self._attempt, StreamOutcome.CLIENT_CANCELLED)

    close = aclose  # deterministic close alias for mixed sync/async callers

    async def _aclose_upstream(self) -> None:
        close = getattr(self._upstream, "aclose", None)
        if not callable(close):
            close = getattr(self._upstream, "close", None)
        if not callable(close):
            self._attempt.mark_upstream_close(ok=False)
            return
        try:
            await _maybe_await(close())
            self._attempt.mark_upstream_close(ok=True)
        except Exception:
            self._attempt.mark_upstream_close(ok=False)

    async def __aenter__(self) -> "AsyncStreamIteratorProxy":
        enter = getattr(self._upstream, "__aenter__", None)
        if callable(enter):
            try:
                entered = await enter()
                self._upstream_entered = True
                if entered is not None and entered is not self._upstream:
                    self._upstream = entered
                    self._iterator = None
            except Exception as exc:
                await _finalize_attempt(
                    self._attempt, StreamOutcome.PROVIDER_ERROR, error=exc
                )
                raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> Any:
        result = None
        exit_fn = getattr(self._upstream, "__aexit__", None)
        if self._upstream_entered and callable(exit_fn):
            try:
                result = await exit_fn(exc_type, exc, tb)
                self._attempt.mark_upstream_close(ok=True)
            except Exception as upstream_exc:
                if not self._attempt.finalized:
                    await _finalize_attempt(
                        self._attempt, StreamOutcome.PROVIDER_ERROR, error=upstream_exc
                    )
                raise
            finally:
                self._closed = True
        else:
            self._closed = True
            await self._aclose_upstream()

        if not self._attempt.finalized:
            if exc_type is not None and issubclass(exc_type, StreamCutoffError):
                await _finalize_attempt(self._attempt, StreamOutcome.CUTOFF)
            elif self._exhausted:
                await _finalize_attempt(self._attempt, StreamOutcome.COMPLETE)
            else:
                await _finalize_attempt(
                    self._attempt, StreamOutcome.CLIENT_CANCELLED, error=exc
                )
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._upstream, name)


class AsyncStreamManagerProxy:
    """Wraps async ``.stream()`` managers; reserve only on ``__aenter__``."""

    def __init__(
        self,
        manager: Any,
        attempt: Optional[StreamAttempt] = None,
        *,
        attempt_factory: Optional[Callable[[], Any]] = None,
        wrap_entered_stream: Any,
        enter_scope: Optional[Callable[[], Any]] = None,
    ) -> None:
        if attempt is None and attempt_factory is None:
            raise TypeError("attempt or attempt_factory is required")
        self._manager = manager
        self._attempt = attempt
        self._attempt_factory = attempt_factory
        self._wrap_entered_stream = wrap_entered_stream
        self._enter_scope = enter_scope
        self._entered: Optional[Any] = None
        self._closed = False

    def close(self) -> None:
        """Dispose an unentered SDK stream manager without creating an attempt.

        OpenAI's async stream manager eagerly stores the request coroutine even
        though it does not await it until ``__aenter__``.  Dropping an unentered
        manager otherwise emits ``coroutine was never awaited``.  The SDK does
        not currently expose a public close method for that state, so close any
        still-pending coroutine owned by the manager while no Kazen attempt has
        been created.
        """
        if self._closed or self._attempt is not None or self._entered is not None:
            return
        values = getattr(self._manager, "__dict__", {}).values()
        for value in values:
            if inspect.iscoroutine(value):
                value.close()
        self._closed = True

    async def aclose(self) -> None:
        """Async spelling for deterministic disposal of an unentered manager."""
        self.close()

    async def __aenter__(self) -> Any:
        if self._closed:
            raise RuntimeError("stream manager is closed")
        if self._attempt is None:
            assert self._attempt_factory is not None
            try:
                produced = self._attempt_factory()
                self._attempt = await _maybe_await(produced)
            except BaseException:
                # Denied before dispatch: dispose of the request coroutine the SDK
                # created eagerly, so it is neither sent nor left un-awaited.
                self.close()
                raise
        try:
            scope = self._enter_scope() if self._enter_scope is not None else nullcontext()
            if hasattr(scope, "__aenter__"):
                async with scope:
                    entered = await self._manager.__aenter__()
            else:
                with scope:
                    entered = await self._manager.__aenter__()
        except Exception as exc:
            await _finalize_attempt(
                self._attempt, StreamOutcome.PROVIDER_ERROR, error=exc
            )
            raise
        self._entered = self._wrap_entered_stream(entered, self._attempt)
        return self._entered

    async def __aexit__(self, exc_type, exc, tb) -> Any:
        try:
            result = await self._manager.__aexit__(exc_type, exc, tb)
        except Exception as upstream_exc:
            if self._attempt is not None and not self._attempt.finalized:
                await _finalize_attempt(
                    self._attempt, StreamOutcome.PROVIDER_ERROR, error=upstream_exc
                )
            raise
        else:
            if self._entered is not None:
                close = getattr(self._entered, "aclose", None) or getattr(
                    self._entered, "close", None
                )
                if callable(close):
                    try:
                        await _maybe_await(close())
                    except Exception:
                        pass
            elif self._attempt is not None and not self._attempt.finalized:
                await _finalize_attempt(
                    self._attempt, StreamOutcome.CLIENT_CANCELLED, error=exc
                )
            return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)


@asynccontextmanager
async def async_nullcontext():
    yield
