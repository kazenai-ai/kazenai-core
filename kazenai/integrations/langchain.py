"""LangChain integration — Observe callbacks + Control-aware Runnable wrapper.

Economic admission happens only at a ``kazenai.monitor()``-guarded provider
client (or gateway-owned enforcement). This module must not call ``monitor()``
on arbitrary Runnables.

Control on ``wrap_langchain_runnable`` requires either:

* ``bind_monitored_chat_openai`` on a ``ChatOpenAI`` instance or
  ``bind_monitored_chat_anthropic`` on a ``ChatAnthropic`` instance (preferred), or
* ``declare_langchain_control_boundary(runnable, *guarded_clients)`` when the
  Runnable performs provider calls via those guarded clients internally (one per
  provider in a multi-model chain).
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Callable, Iterator, Optional

from .adapter_contract import (
    CapabilityLevel,
    UnsupportedFrameworkPath,
    active_or_new_operation,
    emit_framework_event,
    enter_operation,
    extract_chat_usage,
    is_monitored_client,
    mark_attached,
    mark_detached,
    reject_openai_responses_api,
    require_control_client,
)

try:
    from langchain_core.callbacks.base import BaseCallbackHandler as _LangChainBaseCallbackHandler
except ImportError:  # pragma: no cover - langchain-core optional in minimal installs
    _LangChainBaseCallbackHandler = object  # type: ignore[misc,assignment]


def declare_langchain_control_boundary(runnable: Any, *guarded_clients: Any) -> Any:
    """Mark a Runnable whose provider calls go through the given monitored clients.

    Pass every client the Runnable uses (for example one OpenAI and one Anthropic
    client in a multi-model chain).
    """
    if not guarded_clients:
        raise UnsupportedFrameworkPath("declare_langchain_control_boundary needs at least one monitored client", code="unguarded_control_boundary")
    for client in guarded_clients:
        require_control_client(client, label="declared control boundary")
    try:
        setattr(runnable, "_kazenai_langchain_control_boundary", guarded_clients[0])
        setattr(runnable, "_kazenai_langchain_control_boundaries", tuple(guarded_clients))
    except Exception:
        pass
    return runnable


def _declared_boundaries(runnable: Any) -> tuple[Any, ...]:
    many = getattr(runnable, "_kazenai_langchain_control_boundaries", None)
    if many:
        return tuple(many)
    one = getattr(runnable, "_kazenai_langchain_control_boundary", None)
    return (one,) if one is not None else ()


def _chat_completions_resource(openai_client: Any) -> Any | None:
    return getattr(getattr(getattr(openai_client, "chat", None), "completions", None), "create", None) and getattr(
        getattr(openai_client, "chat", None), "completions", None
    )


def _apply_guarded_openai_to_chat_model(chat_model: Any, guarded: Any) -> None:
    """langchain-openai 1.7: ``client`` is Completions; ``root_client`` is OpenAI."""
    completions = getattr(getattr(guarded, "chat", None), "completions", None)
    if completions is None:
        raise UnsupportedFrameworkPath(
            "Guarded OpenAI client must expose chat.completions for ChatOpenAI bind",
            code="unguarded_control_boundary",
        )
    if hasattr(chat_model, "root_client"):
        try:
            setattr(chat_model, "root_client", guarded)
        except Exception:
            pass
    if hasattr(chat_model, "client"):
        try:
            setattr(chat_model, "client", completions)
        except Exception:
            pass
    if hasattr(chat_model, "openai_api_client"):
        try:
            setattr(chat_model, "openai_api_client", guarded)
        except Exception:
            pass
    try:
        setattr(chat_model, "_kazenai_langchain_control_bound", True)
        setattr(chat_model, "_kazenai_bound_guarded_client", guarded)
    except Exception:
        pass


def _apply_guarded_async_openai_to_chat_model(chat_model: Any, guarded_async: Any) -> None:
    """U05: bind monitored AsyncOpenAI so ainvoke cannot use an unguarded transport."""
    async_completions = getattr(getattr(guarded_async, "chat", None), "completions", None)
    if hasattr(chat_model, "root_async_client"):
        try:
            setattr(chat_model, "root_async_client", guarded_async)
        except Exception:
            pass
    if hasattr(chat_model, "async_client") and async_completions is not None:
        try:
            setattr(chat_model, "async_client", async_completions)
        except Exception:
            pass
    try:
        setattr(chat_model, "_kazenai_bound_guarded_async_client", guarded_async)
    except Exception:
        pass


def _monitor_existing_async_transport(chat_model: Any, monitor_fn: Any, monitor_kwargs: dict) -> Any | None:
    """Re-monitor ChatOpenAI's existing async client when caller did not pass one."""
    existing = getattr(chat_model, "root_async_client", None)
    if existing is None:
        existing = getattr(chat_model, "async_client", None)
        # async_client may be Completions; climb to root when possible.
        root = getattr(existing, "_client", None) or getattr(existing, "_parent", None)
        if root is not None:
            existing = root
    if existing is None:
        return None
    if is_monitored_client(existing):
        return existing
    try:
        return monitor_fn(existing, **monitor_kwargs)
    except Exception:
        return None


def _is_bound_monitored_chat_openai(runnable: Any, guarded_client: Any) -> bool:
    if type(runnable).__name__ == "ChatAnthropic":
        return getattr(runnable, "_kazenai_bound_guarded_client", None) is guarded_client
    if type(runnable).__name__ != "ChatOpenAI":
        return False
    if getattr(runnable, "_kazenai_bound_guarded_client", None) is guarded_client:
        return True
    root = getattr(runnable, "root_client", None)
    if root is guarded_client and is_monitored_client(guarded_client):
        return True
    return False


def _assert_control_runnable_boundary(runnable: Any, guarded_client: Any) -> None:
    """Reject Control when guarded_client is supplied but the Runnable cannot use it."""
    if _is_bound_monitored_chat_openai(runnable, guarded_client):
        return
    if any(client is guarded_client for client in _declared_boundaries(runnable)):
        return
    if type(runnable).__name__ == "ChatOpenAI":
        raise UnsupportedFrameworkPath(
            "ChatOpenAI Control requires bind_monitored_chat_openai(...); "
            "guarded_client alone does not guard model dispatch",
            code="unguarded_control_boundary",
        )
    if type(runnable).__name__ == "ChatAnthropic":
        raise UnsupportedFrameworkPath(
            "ChatAnthropic Control requires bind_monitored_chat_anthropic(...); "
            "guarded_client alone does not guard model dispatch",
            code="unguarded_control_boundary",
        )
    raise UnsupportedFrameworkPath(
        "LangChain Control requires bind_monitored_chat_openai / bind_monitored_chat_anthropic, "
        "or declare_langchain_control_boundary(runnable, *guarded_clients) when the "
        "Runnable calls those monitored clients",
        code="unguarded_control_boundary",
    )


class KazenCallbackHandler(_LangChainBaseCallbackHandler):
    """LangChain-compatible callback for Observe (wire schema 1.2).

    Subclasses ``langchain_core`` ``BaseCallbackHandler`` so callback managers
    can read ``ignore_*`` flags. Does **not** authorize settlement or reserve budget.
    """

    raise_error: bool = False

    def __init__(
        self,
        *,
        org_id: str = "local",
        project_id: str = "default",
        workspace_id: str = "default",
        agent_id: str = "langchain",
        run_id: str = "",
        operation_id: str | None = None,
        on_event: Optional[Any] = None,
        capability: str = "observe",
    ) -> None:
        self.org_id = org_id
        self.project_id = project_id
        self.workspace_id = workspace_id
        self.agent_id = agent_id
        self._on_event = on_event
        self._op = active_or_new_operation(
            framework="langchain",
            org_id=org_id,
            workspace_id=workspace_id,
            project_id=project_id,
            agent_id=agent_id,
            run_id=run_id or None,
            operation_id=operation_id,
            capability=capability,
            reuse_active=False,
        )
        self.run_id = self._op.run_id
        self._parent_run_ids: dict[str, str] = {}

    def _emit(self, event_type: str, *, tokens: int | None = None, payload: dict | None = None) -> None:
        emit_framework_event(
            op=self._op,
            event_type=event_type,
            tokens_used=tokens,
            payload=payload,
            on_event=self._on_event,
        )

    def on_chain_start(
        self,
        serialized: dict,
        inputs: dict,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        self._emit(
            "workflow.start",
            payload={"phase": "chain_start", "run_id": str(run_id or "")},
        )

    def on_chain_end(self, outputs: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        self._emit(
            "workflow.end",
            payload={"phase": "chain_end", "run_id": str(run_id or "")},
        )

    def on_chain_error(self, error: BaseException, *, run_id: Any = None, **kwargs: Any) -> None:
        self._emit(
            "workflow.error",
            payload={"phase": "chain_error", "error_type": type(error).__name__, "run_id": str(run_id or "")},
        )

    def on_chat_model_start(self, serialized: dict, messages: list, *, run_id: Any = None, parent_run_id: Any = None, **kwargs: Any) -> None:
        if parent_run_id is not None:
            self._parent_run_ids[str(run_id)] = str(parent_run_id)
        self._emit("workflow.step", payload={"phase": "chat_model_start", "run_id": str(run_id or "")})

    def on_llm_start(self, serialized: dict, prompts: list, *, run_id: Any = None, parent_run_id: Any = None, **kwargs: Any) -> None:
        self._emit("workflow.step", payload={"phase": "llm_start", "run_id": str(run_id or "")})

    def on_llm_end(self, response: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        tokens, meta = extract_chat_usage(response)
        # Missing usage is unknown — do not coerce to 0 for ledger purposes.
        self._emit(
            "model.call",
            tokens=tokens,
            payload={"phase": "llm_end", "run_id": str(run_id or ""), **meta},
        )

    def on_llm_error(self, error: BaseException, *, run_id: Any = None, **kwargs: Any) -> None:
        self._emit(
            "model.call",
            payload={"phase": "llm_error", "error_type": type(error).__name__, "run_id": str(run_id or "")},
        )

    def on_tool_end(self, output: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        self._emit("tool.call", payload={"phase": "tool_end", "run_id": str(run_id or "")})

    def on_tool_error(self, error: BaseException, *, run_id: Any = None, **kwargs: Any) -> None:
        self._emit(
            "tool.call",
            payload={"phase": "tool_error", "error_type": type(error).__name__, "run_id": str(run_id or "")},
        )


class LangChainRunnableAdapter:
    """Documented Runnable path: invoke / ainvoke / stream / astream with context."""

    def __init__(
        self,
        runnable: Any,
        *,
        org_id: str = "local",
        workspace_id: str = "default",
        project_id: str = "default",
        agent_id: str = "langchain",
        operation_id: str | None = None,
        capability: CapabilityLevel | str = CapabilityLevel.OBSERVE,
        on_event: Callable[[Any], Any] | None = None,
        guarded_client: Any | None = None,
        guarded_clients: Any | None = None,
    ) -> None:
        self._runnable = runnable
        self._on_event = on_event
        clients = tuple(guarded_clients or ()) or ((guarded_client,) if guarded_client is not None else ())
        if not clients and getattr(runnable, "_kazenai_bound_guarded_client", None) is not None:
            # A chat model bound with bind_monitored_chat_openai/_anthropic carries its client.
            clients = (runnable._kazenai_bound_guarded_client,)
        guarded_client = clients[0] if clients else None
        self._guarded_client = guarded_client
        self._guarded_clients = clients
        self._capability = (
            capability if isinstance(capability, CapabilityLevel) else CapabilityLevel(str(capability))
        )
        self._op_kwargs = dict(
            framework="langchain",
            org_id=org_id,
            workspace_id=workspace_id,
            project_id=project_id,
            agent_id=agent_id,
            operation_id=operation_id,
            capability=self._capability,
        )
        if self._capability == CapabilityLevel.CONTROL:
            owner = None
            try:
                from kazenai.enforcement_owner import configured_enforcement_owner, get_enforcement_owner

                owner = configured_enforcement_owner() or (
                    get_enforcement_owner().owner if get_enforcement_owner() else None
                )
            except Exception:
                owner = None
            if owner != "gateway":
                if guarded_client is None:
                    raise UnsupportedFrameworkPath(
                        "LangChain Control requires guarded_client=monitor(...) "
                        "or KAZENAI_ENFORCEMENT_OWNER=gateway",
                        code="unguarded_control_boundary",
                    )
                for client in clients:
                    reject_openai_responses_api(client)
                    require_control_client(client)
                    _assert_control_runnable_boundary(runnable, client)
        mark_attached(self)

    def _enter(self):
        op = active_or_new_operation(**self._op_kwargs, reuse_active=True)
        return enter_operation(op)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        with self._enter() as op:
            emit_framework_event(op=op, event_type="workflow.start", on_event=self._on_event, payload={"mode": "invoke"})
            try:
                result = self._runnable.invoke(*args, **kwargs)
            except Exception as exc:
                emit_framework_event(
                    op=op,
                    event_type="workflow.error",
                    on_event=self._on_event,
                    payload={"mode": "invoke", "error_type": type(exc).__name__},
                )
                raise
            emit_framework_event(op=op, event_type="workflow.end", on_event=self._on_event, payload={"mode": "invoke"})
            return result

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        with self._enter() as op:
            emit_framework_event(op=op, event_type="workflow.start", on_event=self._on_event, payload={"mode": "ainvoke"})
            try:
                result = await self._runnable.ainvoke(*args, **kwargs)
            except Exception as exc:
                emit_framework_event(
                    op=op,
                    event_type="workflow.error",
                    on_event=self._on_event,
                    payload={"mode": "ainvoke", "error_type": type(exc).__name__},
                )
                raise
            emit_framework_event(op=op, event_type="workflow.end", on_event=self._on_event, payload={"mode": "ainvoke"})
            return result

    def stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        with self._enter() as op:
            emit_framework_event(op=op, event_type="workflow.start", on_event=self._on_event, payload={"mode": "stream"})
            try:
                for chunk in self._runnable.stream(*args, **kwargs):
                    yield chunk
            finally:
                emit_framework_event(op=op, event_type="workflow.end", on_event=self._on_event, payload={"mode": "stream"})

    async def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        with self._enter() as op:
            emit_framework_event(op=op, event_type="workflow.start", on_event=self._on_event, payload={"mode": "astream"})
            try:
                async for chunk in self._runnable.astream(*args, **kwargs):
                    yield chunk
            finally:
                emit_framework_event(op=op, event_type="workflow.end", on_event=self._on_event, payload={"mode": "astream"})

    def detach(self) -> None:
        mark_detached(self)


def wrap_langchain_runnable(runnable: Any, **kwargs: Any) -> LangChainRunnableAdapter:
    """Wrap a LangChain Runnable with framework context (not provider monitor())."""
    return LangChainRunnableAdapter(runnable, **kwargs)


def bind_monitored_chat_openai(chat_model: Any, openai_client: Any, **monitor_kwargs: Any) -> Any:
    """Inject a monitored OpenAI client into a LangChain chat model for Control.

    langchain-openai 1.7 keeps ``root_client`` as ``OpenAI`` and ``client`` as
    ``chat.completions.Completions``. Rejects Responses API bindings before dispatch.
    """
    from ..monitor import monitor

    reject_openai_responses_api(chat_model)
    # Control-certified binds use Chat Completions even when langchain-openai infers Responses.
    try:
        if hasattr(chat_model, "use_responses_api"):
            object.__setattr__(chat_model, "use_responses_api", False)
    except Exception:
        try:
            setattr(chat_model, "use_responses_api", False)
        except Exception:
            pass
    async_raw = monitor_kwargs.pop("async_client", None)
    guarded = monitor(openai_client, **monitor_kwargs)
    try:
        setattr(guarded, "_kazenai_monitored", True)
    except Exception:
        pass

    _apply_guarded_openai_to_chat_model(chat_model, guarded)

    guarded_async = None
    if async_raw is not None:
        try:
            guarded_async = monitor(async_raw, **monitor_kwargs)
        except Exception:
            guarded_async = None
    if guarded_async is None:
        guarded_async = _monitor_existing_async_transport(chat_model, monitor, monitor_kwargs)
    if guarded_async is not None:
        _apply_guarded_async_openai_to_chat_model(chat_model, guarded_async)
    else:
        # U05: do not certify Control while an unguarded async transport remains.
        remaining = getattr(chat_model, "root_async_client", None)
        if remaining is not None and not is_monitored_client(remaining):
            raise UnsupportedFrameworkPath(
                "ChatOpenAI Control bind requires a monitored async client for ainvoke; "
                "pass async_client=monitor(AsyncOpenAI(...)) or ensure root_async_client "
                "can be re-monitored",
                code="unguarded_control_boundary",
            )
    require_control_client(guarded, label="OpenAI Chat Completions client")
    return chat_model


def bind_monitored_chat_anthropic(chat_model: Any, anthropic_client: Any | None = None, **monitor_kwargs: Any) -> Any:
    """Route a LangChain ``ChatAnthropic`` through monitored Anthropic clients for Control.

    langchain-anthropic keeps its SDK clients in ``_client`` / ``_async_client``;
    both are replaced with ``monitor()``-guarded clients (sync and async), so
    invoke/stream/ainvoke/astream reserve before dispatch and settle from usage.
    Pass ``async_client=`` to supply the async client, otherwise the model's own
    async client is monitored. ``betas`` route through ``client.beta.messages``,
    which Control does not guard, so they are rejected here.
    """
    from ..monitor import monitor

    if type(chat_model).__name__ != "ChatAnthropic":
        raise UnsupportedFrameworkPath(
            "bind_monitored_chat_anthropic expects a langchain_anthropic.ChatAnthropic",
            code="unguarded_control_boundary",
        )
    if getattr(chat_model, "betas", None) or (getattr(chat_model, "model_kwargs", None) or {}).get("betas"):
        raise UnsupportedFrameworkPath(
            "ChatAnthropic betas use client.beta.messages, which Control does not guard",
            code="unguarded_control_boundary",
        )
    async_raw = monitor_kwargs.pop("async_client", None)
    sync_raw = anthropic_client if anthropic_client is not None else chat_model._client
    guarded = sync_raw if is_monitored_client(sync_raw) else monitor(sync_raw, **monitor_kwargs)
    async_raw = async_raw if async_raw is not None else chat_model._async_client
    guarded_async = async_raw if is_monitored_client(async_raw) else monitor(async_raw, **monitor_kwargs)
    for client in (guarded, guarded_async):
        try:
            setattr(client, "_kazenai_monitored", True)
        except Exception:
            pass
    # cached_property slots: an instance __dict__ entry replaces the lazy client.
    chat_model.__dict__["_client"] = guarded
    chat_model.__dict__["_async_client"] = guarded_async
    for name, value in (
        ("_kazenai_bound_guarded_client", guarded),
        ("_kazenai_bound_guarded_async_client", guarded_async),
    ):
        try:
            setattr(chat_model, name, value)
        except Exception:
            object.__setattr__(chat_model, name, value)
    require_control_client(guarded, label="Anthropic Messages client")
    require_control_client(guarded_async, label="Anthropic Messages async client")
    return chat_model


def bound_guarded_client(chat_model: Any) -> Any:
    """The monitored provider client a chat model was bound to.

    Use it to declare boundaries for chains or graphs built from bound models:
    ``declare_langchain_control_boundary(chain, bound_guarded_client(gpt), ...)``.
    """
    client = getattr(chat_model, "_kazenai_bound_guarded_client", None)
    if client is None:
        raise UnsupportedFrameworkPath(
            f"{type(chat_model).__name__} is not bound; call bind_monitored_chat_openai / "
            "bind_monitored_chat_anthropic first",
            code="unguarded_control_boundary",
        )
    return client


def wrap_model_call_middleware(*, on_event: Callable[[Any], Any] | None = None, **op_kwargs: Any) -> Any:
    """Return ``AgentMiddleware`` for ``create_agent`` (langchain 1.4+).

    Economic admission remains at the guarded provider boundary.
    """
    try:
        from langchain.agents.middleware.types import AgentMiddleware
    except ImportError as exc:  # pragma: no cover
        raise UnsupportedFrameworkPath(
            "LangChain agent middleware requires langchain>=1.4 with AgentMiddleware",
            code="unsupported_framework_path",
        ) from exc

    class KazenModelCallMiddleware(AgentMiddleware):
        def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
            op = active_or_new_operation(framework="langchain", reuse_active=True, **op_kwargs)
            with enter_operation(op):
                emit_framework_event(
                    op=op,
                    event_type="workflow.step",
                    on_event=on_event,
                    payload={"phase": "wrap_model_call"},
                )
                return handler(request)

        async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
            op = active_or_new_operation(framework="langchain", reuse_active=True, **op_kwargs)
            with enter_operation(op):
                emit_framework_event(
                    op=op,
                    event_type="workflow.step",
                    on_event=on_event,
                    payload={"phase": "awrap_model_call"},
                )
                return await handler(request)

    return KazenModelCallMiddleware()
