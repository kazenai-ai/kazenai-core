"""CrewAI integration — Observe listeners + Control kickoff/akickoff wrappers.

Control requires a monitored OpenAI/Anthropic client that is actually bound to
the crew/agent LLM path (or gateway ownership). Listener callbacks never
authorize settlement.
"""

from __future__ import annotations

import contextvars
import os
import threading
import warnings
from typing import Any, Callable, Optional

from .adapter_contract import (
    CapabilityLevel,
    UnsupportedFrameworkPath,
    emit_framework_event,
    enter_operation,
    extract_chat_usage,
    is_monitored_client,
    mark_attached,
    mark_detached,
    new_operation,
    reject_openai_responses_api,
    require_control_client,
)


_LISTENER_LOCK = threading.Lock()
_REGISTERED_LISTENERS: dict[int, Any] = {}


class CrewExecutionScope:
    """Per-crew execution state — nested stack, not a global singleton."""

    def __init__(self, op: Any, crew: Any = None) -> None:
        self.op = op
        self.crew = crew
        self.events: list[Any] = []


# Stack per crew so overlapping executions do not overwrite each other (R15).
_ACTIVE_OPS: dict[int, list[CrewExecutionScope]] = {}
# Task-local scope for concurrent kickoffs of the same crew (U07).
_TASK_SCOPE: contextvars.ContextVar[CrewExecutionScope | None] = contextvars.ContextVar(
    "kazenai_crew_task_scope",
    default=None,
)


def _iter_llm_holders(crew_or_agent: Any) -> list[Any]:
    holders: list[Any] = [crew_or_agent]
    agents = getattr(crew_or_agent, "agents", None)
    if agents:
        holders.extend(list(agents))
    return holders


def _client_identity(obj: Any) -> Any:
    if obj is None:
        return None
    for attr in ("client", "_client", "openai_client", "async_client"):
        inner = getattr(obj, attr, None)
        if inner is not None:
            return inner
    return obj


def crew_binds_guarded_client(crew_or_agent: Any, guarded_client: Any) -> bool:
    """True when the monitored client is attached to the crew/agent LLM path."""
    if guarded_client is None:
        return False
    if getattr(crew_or_agent, "_kazenai_bound_guarded_client", None) is guarded_client:
        return True
    for holder in _iter_llm_holders(crew_or_agent):
        if getattr(holder, "_kazenai_bound_guarded_client", None) is guarded_client:
            return True
        llm = getattr(holder, "llm", None)
        if llm is None:
            continue
        if llm is guarded_client:
            return True
        if getattr(llm, "_kazenai_bound_guarded_client", None) is guarded_client:
            return True
        if _client_identity(llm) is guarded_client:
            return True
    return False


def declare_crew_control_boundary(crew: Any, guarded_client: Any) -> Any:
    """Mark that ``crew`` performs provider calls via ``guarded_client``."""
    require_control_client(guarded_client, label="declared crew control boundary")
    try:
        setattr(crew, "_kazenai_bound_guarded_client", guarded_client)
        setattr(crew, "_kazenai_crew_control_boundary", guarded_client)
    except Exception:
        pass
    return crew


def _safe_attr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _ownership_graph(crew: Any) -> tuple[list[Any], set[str]]:
    """Objects (and their stable string ids) that emit bus events on ``crew``'s behalf.

    CrewAI 1.15.x emits LLM events from ``agent.llm``, task events from the
    Task, and tool events from tool-usage helpers that reference the agent/task.
    None of those sources expose ``.crew`` reliably, so ownership is resolved
    against this graph instead (A2).
    """
    objs: list[Any] = []
    keys: set[str] = set()

    def add(obj: Any) -> None:
        if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
            return
        objs.append(obj)
        for attr in ("id", "key"):
            val = _safe_attr(obj, attr)
            if val is not None and not callable(val):
                keys.add(str(val))

    with warnings.catch_warnings():
        # Reading deprecated CrewAI fields (function_calling_llm) is intentional.
        warnings.simplefilter("ignore", DeprecationWarning)
        add(crew)
        for attr in ("manager_agent", "manager_llm", "planning_llm", "function_calling_llm"):
            add(_safe_attr(crew, attr))
        for agent in list(_safe_attr(crew, "agents") or []):
            add(agent)
            for attr in ("llm", "function_calling_llm"):
                add(_safe_attr(agent, attr))
            for tool in list(_safe_attr(agent, "tools") or []):
                add(tool)
        for task in list(_safe_attr(crew, "tasks") or []):
            add(task)
            add(_safe_attr(task, "agent"))
            for tool in list(_safe_attr(task, "tools") or []):
                add(tool)
    return objs, keys


class KazenCrewListener:
    """Observe-level listener for crew/task/tool/LLM notifications."""

    def __init__(self, *, on_event: Callable[[Any], Any] | None = None) -> None:
        self._on_event = on_event
        self._crew_ref: Any | None = None
        # Strong refs keep ids stable (no reuse after GC) while the crew is bound.
        self._owned_objs: list[Any] = []
        self._owned_ids: frozenset[int] = frozenset()
        self._owned_keys: frozenset[str] = frozenset()

    def bind_crew(self, crew: Any) -> None:
        self._crew_ref = crew
        self.refresh_ownership()

    def refresh_ownership(self) -> None:
        """Rebuild the ownership registry (agents/tasks may change before kickoff)."""
        crew = self._crew_ref
        objs, keys = _ownership_graph(crew) if crew is not None else ([], set())
        with _LISTENER_LOCK:
            self._owned_objs = objs
            self._owned_ids = frozenset(id(o) for o in objs)
            self._owned_keys = frozenset(keys)

    def clear_ownership(self) -> None:
        with _LISTENER_LOCK:
            self._owned_objs = []
            self._owned_ids = frozenset()
            self._owned_keys = frozenset()

    def owns_event(self, source: Any, event: Any = None) -> bool:
        """True when a bus event belongs to this listener's crew.

        Order: an active kickoff scope for a *different* crew always rejects
        (events carry the emitting context); then the source / the event's
        agent and task references are matched against the ownership registry;
        finally an event emitted inside this crew's own kickoff scope is accepted.
        """
        owned = self._crew_ref
        if owned is None:
            return True
        local = _TASK_SCOPE.get()
        if local is not None and local.crew is not None and local.crew is not owned:
            return False
        if source is owned:
            return True
        owned_ids = self._owned_ids
        candidates = [source]
        if event is not None:
            for attr in ("from_agent", "agent", "from_task", "task"):
                candidates.append(_safe_attr(event, attr))
        for cand in candidates:
            if cand is None:
                continue
            if id(cand) in owned_ids:
                return True
            if _safe_attr(cand, "crew") is owned:
                return True
        if event is not None:
            for attr in ("agent_id", "task_id"):
                val = _safe_attr(event, attr)
                if val and str(val) in self._owned_keys:
                    return True
        return local is not None and local.crew is owned

    def _op_for(self, crew: Any = None):
        # Prefer task-local scope only when it belongs to this crew (F04).
        # Never fall back to another crew's ContextVar / top-of-stack.
        target = crew if crew is not None else self._crew_ref
        local = _TASK_SCOPE.get()
        if local is not None and (target is None or local.crew is target):
            return local.op
        if target is None:
            return None
        with _LISTENER_LOCK:
            stack = _ACTIVE_OPS.get(id(target))
        return stack[-1].op if stack else None

    def on_crew_start(self, crew: Any = None, **kwargs: Any) -> None:
        op = self._op_for(crew)
        if op is None:
            return
        emit_framework_event(op=op, event_type="workflow.start", on_event=self._on_event, payload={"phase": "crew_start"})

    def on_crew_end(self, crew: Any = None, **kwargs: Any) -> None:
        op = self._op_for(crew)
        if op is None:
            return
        emit_framework_event(op=op, event_type="workflow.end", on_event=self._on_event, payload={"phase": "crew_end"})

    def on_crew_error(self, crew: Any = None, error: BaseException | None = None, **kwargs: Any) -> None:
        op = self._op_for(crew)
        if op is None:
            return
        emit_framework_event(
            op=op,
            event_type="workflow.error",
            on_event=self._on_event,
            payload={"phase": "crew_error", "error_type": type(error).__name__ if error else "error"},
        )

    def on_task_start(self, crew: Any = None, **kwargs: Any) -> None:
        op = self._op_for(crew)
        if op is None:
            return
        emit_framework_event(op=op, event_type="workflow.step", on_event=self._on_event, payload={"phase": "task_start"})

    def on_task_end(self, crew: Any = None, **kwargs: Any) -> None:
        op = self._op_for(crew)
        if op is None:
            return
        emit_framework_event(op=op, event_type="workflow.step", on_event=self._on_event, payload={"phase": "task_end"})

    def on_tool_start(self, crew: Any = None, **kwargs: Any) -> None:
        op = self._op_for(crew)
        if op is None:
            return
        emit_framework_event(op=op, event_type="tool.call", on_event=self._on_event, payload={"phase": "tool_start"})

    def on_llm_start(self, crew: Any = None, **kwargs: Any) -> None:
        op = self._op_for(crew)
        if op is None:
            return
        emit_framework_event(op=op, event_type="workflow.step", on_event=self._on_event, payload={"phase": "llm_start"})

    def on_llm_end(
        self,
        crew: Any = None,
        response: Any = None,
        usage: Any = None,
        **kwargs: Any,
    ) -> None:
        op = self._op_for(crew)
        if op is None:
            return
        tokens, meta = extract_chat_usage(response if response is not None else usage)
        emit_framework_event(
            op=op,
            event_type="model.call",
            tokens_used=tokens,
            on_event=self._on_event,
            payload={"phase": "llm_end", **meta},
        )

    def on_llm_error(self, crew: Any = None, error: BaseException | None = None, **kwargs: Any) -> None:
        op = self._op_for(crew)
        if op is None:
            return
        emit_framework_event(
            op=op,
            event_type="model.call",
            on_event=self._on_event,
            payload={"phase": "llm_error", "error_type": type(error).__name__ if error else "error"},
        )

    def on_tool_end(self, crew: Any = None, **kwargs: Any) -> None:
        op = self._op_for(crew)
        if op is None:
            return
        emit_framework_event(op=op, event_type="tool.call", on_event=self._on_event, payload={"phase": "tool_end"})


def _register_listener(crew: Any, listener: KazenCrewListener) -> None:
    """Register on CrewAI's event bus using the pinned decorator-style ``on`` API."""
    listener.bind_crew(crew)
    key = id(crew)
    with _LISTENER_LOCK:
        if key in _REGISTERED_LISTENERS:
            return
        _REGISTERED_LISTENERS[key] = {"listener": listener, "handlers": []}

    bus = None
    for mod_name in (
        "crewai.utilities.events",
        "crewai.events.event_bus",
        "crewai.events",
    ):
        try:
            mod = __import__(mod_name, fromlist=["crewai_event_bus", "event_bus"])
            bus = getattr(mod, "crewai_event_bus", None) or getattr(mod, "event_bus", None)
            if bus is not None:
                break
        except Exception:
            continue

    registered = False
    if bus is not None:
        on_fn = getattr(bus, "on", None)
        # CrewAI 1.x: crewai_event_bus.on(EventType) returns a decorator.
        if callable(on_fn):
            event_types: list[Any] = []
            for mod_name in (
                "crewai.utilities.events",
                "crewai.events.types",
                "crewai.events",
            ):
                try:
                    mod = __import__(mod_name, fromlist=["*"])
                except Exception:
                    continue
                for name in (
                    "CrewKickoffStartedEvent",
                    "CrewKickoffCompletedEvent",
                    "CrewKickoffFailedEvent",
                    "TaskStartedEvent",
                    "TaskCompletedEvent",
                    "ToolUsageStartedEvent",
                    "ToolUsageFinishedEvent",
                    "LLMCallStartedEvent",
                    "LLMCallCompletedEvent",
                    "LLMCallFailedEvent",
                ):
                    ev = getattr(mod, name, None)
                    if ev is not None and ev not in event_types:
                        event_types.append(ev)
            for ev_type in event_types:
                try:
                    decorator = on_fn(ev_type)

                    def _make_handler(ltn: KazenCrewListener, et: Any):
                        # CrewAI 1.15.26 emits ``handler(source, event_instance, **extra)``
                        # (three positional args). Accept *args for forward compatibility.
                        def _handler(*args: Any, **kwargs: Any) -> None:
                            source = args[0] if args else kwargs.get("source")
                            event = args[1] if len(args) > 1 else kwargs.get("event")
                            payload = event if event is not None else kwargs
                            owned = ltn._crew_ref or crew
                            # F04/A2: route only sources owned by this crew's
                            # graph or emitted inside its kickoff scope.
                            if not ltn.owns_event(source, event):
                                return
                            name = getattr(et, "__name__", str(et))
                            if "KickoffStarted" in name or "CrewKickoffStarted" in name:
                                ltn.on_crew_start(crew=owned)
                            elif "KickoffCompleted" in name or "CrewKickoffCompleted" in name:
                                ltn.on_crew_end(crew=owned)
                            elif "KickoffFailed" in name:
                                ltn.on_crew_error(
                                    crew=owned,
                                    error=getattr(payload, "error", None),
                                )
                            elif "TaskStarted" in name:
                                ltn.on_task_start(crew=owned)
                            elif "TaskCompleted" in name or "TaskFinished" in name:
                                ltn.on_task_end(crew=owned)
                            elif "ToolUsageStarted" in name:
                                ltn.on_tool_start(crew=owned)
                            elif "ToolUsageFinished" in name or "ToolUsageCompleted" in name:
                                ltn.on_tool_end(crew=owned)
                            elif "LLMCallStarted" in name:
                                ltn.on_llm_start(crew=owned)
                            elif "LLMCallFailed" in name:
                                ltn.on_llm_error(
                                    crew=owned,
                                    error=getattr(payload, "error", None),
                                )
                            elif "LLMCallCompleted" in name or (
                                "LLM" in name and "Failed" not in name
                            ):
                                usage = getattr(payload, "usage", None) or getattr(
                                    payload, "token_usage", None
                                )
                                ltn.on_llm_end(crew=owned, usage=usage)
                        return _handler

                    handler = _make_handler(listener, ev_type)
                    if callable(decorator):
                        decorated = decorator(handler)
                        handler = decorated if callable(decorated) else handler
                    with _LISTENER_LOCK:
                        entry = _REGISTERED_LISTENERS.get(key)
                        if isinstance(entry, dict):
                            entry.setdefault("handlers", []).append((ev_type, handler, bus))
                    registered = True
                except Exception:
                    continue
        if not registered:
            for meth in ("register", "listen"):
                register = getattr(bus, meth, None)
                if not callable(register):
                    continue
                try:
                    register(listener)
                    registered = True
                    break
                except TypeError:
                    try:
                        register("*", listener)
                        registered = True
                        break
                    except Exception:
                        pass
                except Exception:
                    pass

    if registered:
        return

    # Fallback: attach to crew instance hooks if present.
    for attr in ("event_listeners", "listeners", "_listeners"):
        try:
            existing = getattr(crew, attr, None)
            if isinstance(existing, list):
                existing.append(listener)
                return
            if existing is None:
                setattr(crew, attr, [listener])
                return
        except Exception:
            continue


def _unregister_listener(crew: Any) -> None:
    key = id(crew)
    with _LISTENER_LOCK:
        entry = _REGISTERED_LISTENERS.pop(key, None)
    if not isinstance(entry, dict):
        return
    listener = entry.get("listener")
    if isinstance(listener, KazenCrewListener):
        listener.clear_ownership()
    for item in entry.get("handlers") or []:
        try:
            ev_type, handler, bus = item
        except Exception:
            continue
        for meth in ("off", "remove", "unregister"):
            fn = getattr(bus, meth, None)
            if not callable(fn):
                continue
            try:
                fn(ev_type, handler)
                break
            except TypeError:
                try:
                    fn(handler)
                    break
                except Exception:
                    pass
            except Exception:
                pass


class CrewAdapter:
    """Wraps kickoff and native akickoff with scoped operation context."""

    def __init__(
        self,
        crew: Any,
        *,
        org_id: str = "local",
        workspace_id: str = "default",
        project_id: str = "default",
        agent_id: str = "crewai",
        operation_id: str | None = None,
        capability: CapabilityLevel | str = CapabilityLevel.OBSERVE,
        on_event: Callable[[Any], Any] | None = None,
        guarded_client: Any | None = None,
        listener: KazenCrewListener | None = None,
    ) -> None:
        if getattr(crew, "kickoff", None) is None and getattr(crew, "akickoff", None) is None:
            raise TypeError("crew object has no kickoff/akickoff")
        self._crew = crew
        self._on_event = on_event
        self._listener = listener or KazenCrewListener(on_event=on_event)
        self._capability = (
            capability if isinstance(capability, CapabilityLevel) else CapabilityLevel(str(capability))
        )
        self._op_kwargs = dict(
            framework="crewai",
            org_id=org_id,
            workspace_id=workspace_id,
            project_id=project_id,
            agent_id=agent_id,
            operation_id=operation_id,
            capability=self._capability,
        )
        if self._capability == CapabilityLevel.CONTROL:
            gateway = os.getenv("KAZENAI_ENFORCEMENT_OWNER", "").strip().lower() == "gateway"
            if guarded_client is None and not gateway:
                raise UnsupportedFrameworkPath(
                    "CrewAI Control requires guarded_client=monitor(OpenAI(...)) bound via "
                    "bind_crew_llm_client / declare_crew_control_boundary, or gateway ownership",
                    code="unguarded_control_boundary",
                )
            if guarded_client is not None:
                reject_openai_responses_api(guarded_client)
                require_control_client(guarded_client, label="CrewAI LLM client")
                # R12: an unrelated monitored client must not certify Control.
                if not crew_binds_guarded_client(crew, guarded_client):
                    raise UnsupportedFrameworkPath(
                        "CrewAI Control requires the guarded_client to be bound to the crew/agent "
                        "LLM path (bind_crew_llm_client or declare_crew_control_boundary); "
                        "passing an unrelated monitored client is not an enforcement boundary",
                        code="unguarded_control_boundary",
                    )
        _register_listener(crew, self._listener)
        mark_attached(self)

    def _bind(self, op: Any) -> Any:
        self._listener.refresh_ownership()
        scope = CrewExecutionScope(op, crew=self._crew)
        with _LISTENER_LOCK:
            _ACTIVE_OPS.setdefault(id(self._crew), []).append(scope)
        token = _TASK_SCOPE.set(scope)
        return token

    def _unbind(self, token: Any = None) -> None:
        if token is not None:
            try:
                _TASK_SCOPE.reset(token)
            except Exception:
                pass
        with _LISTENER_LOCK:
            stack = _ACTIVE_OPS.get(id(self._crew))
            if not stack:
                return
            stack.pop()
            if not stack:
                _ACTIVE_OPS.pop(id(self._crew), None)

    @staticmethod
    def _is_hook_aborted(exc: BaseException) -> bool:
        return type(exc).__name__ == "HookAborted"

    def kickoff(self, *args: Any, **kwargs: Any) -> Any:
        from kazenai.otel import build_control_attributes, workflow_span

        op = new_operation(**self._op_kwargs)
        with enter_operation(op):
            with workflow_span(
                "kazenai.workflow",
                attributes=build_control_attributes(operation_id=op.operation_id),
            ):
                token = self._bind(op)
                emit_framework_event(op=op, event_type="workflow.start", on_event=self._on_event, payload={"mode": "kickoff"})
                terminal = "success"
                try:
                    return self._crew.kickoff(*args, **kwargs)
                except Exception as exc:
                    if self._is_hook_aborted(exc):
                        terminal = "abort"
                        emit_framework_event(
                            op=op,
                            event_type="workflow.suspend",
                            on_event=self._on_event,
                            payload={"mode": "kickoff", "signal": "HookAborted"},
                        )
                    else:
                        terminal = "error"
                        emit_framework_event(
                            op=op,
                            event_type="workflow.error",
                            on_event=self._on_event,
                            payload={"mode": "kickoff", "error_type": type(exc).__name__},
                        )
                    raise
                finally:
                    if terminal == "success":
                        emit_framework_event(op=op, event_type="workflow.end", on_event=self._on_event, payload={"mode": "kickoff"})
                    self._unbind(token)

    async def akickoff(self, *args: Any, **kwargs: Any) -> Any:
        """Native async kickoff when present. Thread-backed kickoff_async is not certified here."""
        native = getattr(self._crew, "akickoff", None)
        if native is None:
            raise UnsupportedFrameworkPath(
                "Native akickoff() unavailable on this CrewAI version; "
                "thread-backed kickoff_async is not Control-certified",
                code="unsupported_framework_path",
            )
        op = new_operation(**self._op_kwargs)
        with enter_operation(op):
            token = self._bind(op)
            emit_framework_event(op=op, event_type="workflow.start", on_event=self._on_event, payload={"mode": "akickoff"})
            terminal = "success"
            try:
                return await native(*args, **kwargs)
            except Exception as exc:
                if self._is_hook_aborted(exc):
                    terminal = "abort"
                    emit_framework_event(
                        op=op,
                        event_type="workflow.suspend",
                        on_event=self._on_event,
                        payload={"mode": "akickoff", "signal": "HookAborted"},
                    )
                else:
                    terminal = "error"
                    emit_framework_event(
                        op=op,
                        event_type="workflow.error",
                        on_event=self._on_event,
                        payload={"mode": "akickoff", "error_type": type(exc).__name__},
                    )
                raise
            finally:
                if terminal == "success":
                    emit_framework_event(op=op, event_type="workflow.end", on_event=self._on_event, payload={"mode": "akickoff"})
                self._unbind(token)

    def detach(self) -> None:
        self._unbind()
        _unregister_listener(self._crew)
        self._listener.clear_ownership()
        mark_detached(self)


def wrap_crew_kickoff(crew: Any, **monitor_kwargs: Any) -> Callable[..., Any]:
    """Compatibility shim: returns sync kickoff callable."""
    adapter = CrewAdapter(crew, **monitor_kwargs)

    def kickoff(*args: Any, **kwargs: Any) -> Any:
        return adapter.kickoff(*args, **kwargs)

    return kickoff


def wrap_crew(crew: Any, **kwargs: Any) -> CrewAdapter:
    """Preferred entry: kickoff + akickoff adapter."""
    return CrewAdapter(crew, **kwargs)


def abort_on_budget_denial(exc: BaseException) -> BaseException:
    """Convert Control denial into CrewAI's verified blocking signal when available."""
    try:
        from crewai.hooks import HookAborted  # type: ignore

        return HookAborted(str(exc))
    except Exception:
        pass
    return exc


def _bind_crewai_native_llm(llm: Any, guarded: Any, monitor_kwargs: dict) -> bool:
    """CrewAI 1.15 native providers keep SDK clients in ``_client`` / ``_async_client``.

    Swap in monitored clients there (sync and async) and keep the LLM object;
    replacing the LLM itself breaks the crew ("LLM must be resolved").
    """
    if not (hasattr(llm, "_get_sync_client") and hasattr(llm, "_get_async_client")):
        return False
    provider = type(llm).__name__.lower()
    client_is_anthropic = hasattr(guarded, "messages") and not hasattr(guarded, "chat")
    if ("anthropic" in provider) != client_is_anthropic:
        raise UnsupportedFrameworkPath(
            f"bind_crew_llm_client: {type(llm).__name__} needs a matching "
            f"{'Anthropic' if 'anthropic' in provider else 'OpenAI'} client",
            code="unguarded_control_boundary",
        )
    from ..monitor import monitor

    llm._client = guarded
    try:
        raw_async = llm._async_client if llm._async_client is not None else llm._build_async_client()
        llm._async_client = raw_async if is_monitored_client(raw_async) else monitor(raw_async, **monitor_kwargs)
    except Exception:
        # No async client could be built (e.g. no API key yet): async calls would be
        # unguarded, so refuse rather than certify half the LLM.
        raise UnsupportedFrameworkPath(
            "bind_crew_llm_client could not monitor the CrewAI LLM async client",
            code="unguarded_control_boundary",
        )
    try:
        object.__setattr__(llm, "_kazenai_bound_guarded_client", guarded)
    except Exception:
        pass
    return True


def bind_crew_llm_client(crew_or_agent: Any, openai_client: Any, **monitor_kwargs: Any) -> Any:
    """Attach a monitored provider client to a CrewAI crew/agent LLM.

    CrewAI 1.15 native LLMs (``LLM(model="gpt-4o-mini")`` or an Anthropic model)
    get monitored ``_client``/``_async_client``; pass an OpenAI client for OpenAI
    models and an Anthropic client for Anthropic models.
    """
    from ..monitor import monitor

    reject_openai_responses_api(openai_client)
    guarded = openai_client if is_monitored_client(openai_client) else monitor(openai_client, **monitor_kwargs)
    try:
        setattr(guarded, "_kazenai_monitored", True)
    except Exception:
        pass

    bound = False
    for holder in _iter_llm_holders(crew_or_agent):
        native_llm = getattr(holder, "llm", None)
        if native_llm is not None and _bind_crewai_native_llm(native_llm, guarded, dict(monitor_kwargs)):
            bound = True
            try:
                setattr(holder, "_kazenai_bound_guarded_client", guarded)
            except Exception:
                pass
            continue
        holder_bound = False
        for attr in ("llm", "client", "_client"):
            if not hasattr(holder, attr):
                continue
            try:
                target = getattr(holder, attr)
                if target is not None and hasattr(target, "client"):
                    setattr(target, "client", guarded)
                    try:
                        setattr(target, "_kazenai_bound_guarded_client", guarded)
                    except Exception:
                        pass
                    holder_bound = True
                    bound = True
                elif attr == "llm" and target is None:
                    continue
                else:
                    setattr(holder, attr, guarded)
                    holder_bound = True
                    bound = True
            except Exception:
                continue
        if holder_bound:
            try:
                setattr(holder, "_kazenai_bound_guarded_client", guarded)
            except Exception:
                pass

    if bound:
        try:
            setattr(crew_or_agent, "_kazenai_bound_guarded_client", guarded)
        except Exception:
            pass
    else:
        # U05: automatic bind must prove a supported LLM path; use
        # declare_crew_control_boundary for explicit caller-owned wiring.
        raise UnsupportedFrameworkPath(
            "bind_crew_llm_client found no LLM/client path to bind; "
            "use declare_crew_control_boundary(crew, guarded) for caller-owned Control",
            code="unguarded_control_boundary",
        )

    require_control_client(guarded, label="CrewAI provider client")
    return guarded
