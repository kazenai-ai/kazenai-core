"""CrewAI integration — Observe listeners + Control kickoff/akickoff wrappers.

Certified for CrewAI 1.15.26 native OpenAI and Anthropic LLMs: ``kickoff``,
native ``akickoff`` and ``kickoff_async``, sequential and hierarchical crews whose
every LLM is bound, nested crews, cancellation and concurrent kickoffs.

Control requires every LLM a crew can call to be bound to a monitored
OpenAI/Anthropic client (``bind_crew_llm_client``), or gateway ownership.
Crew features that call models or embedders KazenAI cannot see (memory,
knowledge, an unbound planning/manager/function-calling LLM, ``llm_overlay``,
``Crew(stream=True)``) are refused under Control. Listener callbacks never
authorize settlement.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import threading
import warnings
import weakref
from typing import Any, Callable, Iterable, Optional

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
# Keyed by listener, so two adapters on one crew each keep their own handlers.
_REGISTERED_LISTENERS: dict[int, Any] = {}
# id(object) -> ids of the listeners whose crew graph contains it. An object in
# more than one graph (an LLM shared by two crews) is ambiguous.
_OWNERSHIP: dict[int, set[int]] = {}


class CrewExecutionScope:
    """Per-crew execution state — nested stack, not a global singleton."""

    def __init__(self, op: Any, crew: Any = None) -> None:
        self.op = op
        self.crew = crew
        self.events: list[Any] = []


# Stack per crew so overlapping executions do not overwrite each other.
_ACTIVE_OPS: dict[int, list[CrewExecutionScope]] = {}
# Task-local scope for concurrent kickoffs of the same crew.
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
    against this graph instead.
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


def _release_ownership(listener_id: int, state: dict[str, frozenset[int]]) -> None:
    """Drop a listener's ownership entries (on detach, or when it is collected)."""
    with _LISTENER_LOCK:
        for oid in state.get("ids", frozenset()):
            owners = _OWNERSHIP.get(oid)
            if owners is not None:
                owners.discard(listener_id)
                if not owners:
                    _OWNERSHIP.pop(oid, None)
        state["ids"] = frozenset()


class KazenCrewListener:
    """Observe-level listener for crew/task/tool/LLM notifications."""

    def __init__(self, *, on_event: Callable[[Any], Any] | None = None) -> None:
        self._on_event = on_event
        self._crew_ref: Any | None = None
        # Strong refs keep ids stable (no reuse after GC) while the crew is bound.
        self._owned_objs: list[Any] = []
        self._owned_ids: frozenset[int] = frozenset()
        self._owned_keys: frozenset[str] = frozenset()
        # Shared with the finalizer (which must not reference ``self``), so a
        # listener collected without detach() cannot leave stale entries whose
        # object ids get reused.
        self._ownership_state: dict[str, frozenset[int]] = {"ids": frozenset()}
        weakref.finalize(self, _release_ownership, id(self), self._ownership_state)

    def bind_crew(self, crew: Any) -> None:
        self._crew_ref = crew
        self.refresh_ownership()

    def _set_owned(self, objs: list[Any], keys: set[str]) -> None:
        me = id(self)
        new_ids = frozenset(id(o) for o in objs)
        with _LISTENER_LOCK:
            for oid in self._owned_ids - new_ids:
                owners = _OWNERSHIP.get(oid)
                if owners is not None:
                    owners.discard(me)
                    if not owners:
                        _OWNERSHIP.pop(oid, None)
            for oid in new_ids:
                _OWNERSHIP.setdefault(oid, set()).add(me)
            self._owned_objs = objs
            self._owned_ids = new_ids
            self._ownership_state["ids"] = new_ids
            self._owned_keys = frozenset(keys)

    def refresh_ownership(self) -> None:
        """Rebuild the ownership registry (agents/tasks may change before kickoff)."""
        crew = self._crew_ref
        objs, keys = _ownership_graph(crew) if crew is not None else ([], set())
        self._set_owned(objs, keys)

    def clear_ownership(self) -> None:
        self._set_owned([], set())

    def owns_event(self, source: Any, event: Any = None) -> bool:
        """True when a bus event belongs to this listener's crew.

        Order: an active kickoff scope for a *different* crew always rejects
        (events carry the emitting context); then the source / the event's
        agent and task references are matched against the ownership registry;
        finally an event emitted inside this crew's own kickoff scope is accepted.
        """
        owned = self._crew_ref
        if owned is None:
            # An unbound listener owns nothing.
            return False
        local = _TASK_SCOPE.get()
        if local is not None and local.crew is not None and local.crew is not owned:
            return False
        in_own_scope = local is not None and local.crew is owned
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
                with _LISTENER_LOCK:
                    shared = len(_OWNERSHIP.get(id(cand), ())) > 1
                # An object several crews share only belongs to the one whose
                # kickoff it runs under.
                if not shared or in_own_scope:
                    return True
                continue
            if _safe_attr(cand, "crew") is owned:
                return True
        if event is not None:
            for attr in ("agent_id", "task_id"):
                val = _safe_attr(event, attr)
                if val and str(val) in self._owned_keys:
                    return True
        return local is not None and local.crew is owned

    def _op_for(self, crew: Any = None):
        # Prefer task-local scope only when it belongs to this crew.
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
    key = id(listener)
    with _LISTENER_LOCK:
        if key in _REGISTERED_LISTENERS:
            return
        _REGISTERED_LISTENERS[key] = {"listener": listener, "crew": crew, "handlers": []}

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
                            # Route only sources owned by this crew's
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


def _unregister_listener(listener: KazenCrewListener) -> None:
    key = id(listener)
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


def _is_bound(llm: Any, allowed: tuple[Any, ...]) -> bool:
    if llm is None:
        return False
    if any(llm is client for client in allowed):
        return True
    marker = getattr(llm, "_kazenai_bound_guarded_client", None)
    if marker is not None and any(marker is client for client in allowed):
        return True
    ident = _client_identity(llm)
    return any(ident is client for client in allowed)


def unguarded_crew_paths(crew: Any, allowed: Iterable[Any]) -> list[str]:
    """Every place ``crew`` could call a model or embedder outside ``allowed``.

    Empty means each LLM the crew can call is bound to one of the guarded
    clients. Memory, knowledge and embedders use models KazenAI cannot see.
    """
    allowed = tuple(c for c in allowed if c is not None)
    gaps: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        if _safe_attr(crew, "memory"):
            gaps.append("crew.memory")
        if _safe_attr(crew, "knowledge_sources") or _safe_attr(crew, "knowledge"):
            gaps.append("crew.knowledge")
        if _safe_attr(crew, "embedder"):
            gaps.append("crew.embedder")
        fc = _safe_attr(crew, "function_calling_llm")
        if fc is not None and not _is_bound(fc, allowed):
            gaps.append("crew.function_calling_llm")
        if _safe_attr(crew, "planning") and not _is_bound(_safe_attr(crew, "planning_llm"), allowed):
            gaps.append("crew.planning_llm")
        process = str(getattr(_safe_attr(crew, "process"), "value", _safe_attr(crew, "process")) or "")
        manager_agent = _safe_attr(crew, "manager_agent")
        manager_llm = _safe_attr(crew, "manager_llm")
        if manager_llm is not None and not _is_bound(manager_llm, allowed):
            gaps.append("crew.manager_llm")
        if process == "hierarchical" and manager_llm is None and manager_agent is None:
            gaps.append("crew.manager_llm (hierarchical default)")
        agents = list(_safe_attr(crew, "agents") or [])
        if manager_agent is not None:
            agents.append(manager_agent)
        for index, agent in enumerate(agents):
            label = f"agent[{index}]" if agent is not manager_agent else "crew.manager_agent"
            if not _is_bound(_safe_attr(agent, "llm"), allowed):
                gaps.append(f"{label}.llm")
            afc = _safe_attr(agent, "function_calling_llm")
            if afc is not None and not _is_bound(afc, allowed):
                gaps.append(f"{label}.function_calling_llm")
            planning = _safe_attr(agent, "planning_config")
            planning_llm = _safe_attr(planning, "llm") if planning is not None else None
            if planning_llm is not None and not _is_bound(planning_llm, allowed):
                gaps.append(f"{label}.planning_config.llm")
            if _safe_attr(agent, "memory"):
                gaps.append(f"{label}.memory")
            if _safe_attr(agent, "knowledge_sources") or _safe_attr(agent, "knowledge"):
                gaps.append(f"{label}.knowledge")
            if _safe_attr(agent, "embedder"):
                gaps.append(f"{label}.embedder")
    return gaps


def _overlay_active() -> bool:
    try:
        from crewai.llm_overlay import active  # type: ignore
    except Exception:
        return False
    try:
        return active.get() is not None
    except Exception:
        return False


_CANCEL_TYPES: tuple[type[BaseException], ...] = (asyncio.CancelledError, KeyboardInterrupt, GeneratorExit)


class CrewAdapter:
    """Wraps kickoff, native akickoff and kickoff_async with scoped operation context."""

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
        guarded_clients: Iterable[Any] | None = None,
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
        clients = [c for c in [guarded_client, *(guarded_clients or [])] if c is not None]
        self._guarded_clients = tuple(clients)
        if self._capability == CapabilityLevel.CONTROL:
            gateway = os.getenv("KAZENAI_ENFORCEMENT_OWNER", "").strip().lower() == "gateway"
            if not clients and not gateway:
                raise UnsupportedFrameworkPath(
                    "CrewAI Control requires guarded_client=monitor(OpenAI(...)) bound via "
                    "bind_crew_llm_client / declare_crew_control_boundary, or gateway ownership",
                    code="unguarded_control_boundary",
                )
            for client in clients:
                reject_openai_responses_api(client)
                require_control_client(client, label="CrewAI LLM client")
            if clients:
                # An unrelated monitored client must not certify Control.
                if not any(crew_binds_guarded_client(crew, c) for c in clients):
                    raise UnsupportedFrameworkPath(
                        "CrewAI Control requires the guarded_client to be bound to the crew/agent "
                        "LLM path (bind_crew_llm_client or declare_crew_control_boundary); "
                        "passing an unrelated monitored client is not an enforcement boundary",
                        code="unguarded_control_boundary",
                    )
                self._check_boundary()
        _register_listener(crew, self._listener)
        mark_attached(self)

    def _check_boundary(self) -> None:
        # A declared boundary is the caller's assertion about its own wiring.
        if getattr(self._crew, "_kazenai_crew_control_boundary", None) is not None:
            return
        gaps = unguarded_crew_paths(self._crew, self._guarded_clients)
        if gaps:
            raise UnsupportedFrameworkPath(
                "CrewAI Control: these paths can call a model outside the guarded clients: "
                + ", ".join(gaps)
                + ". Bind each LLM with bind_crew_llm_client (pass every client as "
                "guarded_clients=[...]) or turn the feature off.",
                code="unguarded_control_boundary",
            )

    def _preflight(self) -> None:
        """Runtime refusals: paths that drop the guard or the operation scope."""
        if self._capability != CapabilityLevel.CONTROL or not self._guarded_clients:
            return
        if _safe_attr(self._crew, "stream"):
            raise UnsupportedFrameworkPath(
                "CrewAI Control does not support Crew(stream=True): the crew runs after "
                "kickoff returns, outside the operation scope",
                code="unsupported_framework_path",
            )
        if _overlay_active():
            raise UnsupportedFrameworkPath(
                "CrewAI Control does not support llm_overlay: it rebuilds agent LLMs "
                "without the guarded client",
                code="unguarded_control_boundary",
            )
        self._check_boundary()

    def _new_op(self) -> Any:
        outer = _TASK_SCOPE.get()
        metadata: dict[str, Any] = {}
        if outer is not None and outer.crew is not self._crew:
            # A crew kicked off inside another crew's task or tool.
            metadata["parent_operation_id"] = outer.op.operation_id
        return new_operation(**self._op_kwargs, metadata=metadata)

    def _bind(self, op: Any) -> Any:
        self._listener.refresh_ownership()
        scope = CrewExecutionScope(op, crew=self._crew)
        with _LISTENER_LOCK:
            _ACTIVE_OPS.setdefault(id(self._crew), []).append(scope)
        return _TASK_SCOPE.set(scope)

    def _unbind(self, token: Any = None) -> None:
        # The scope this kickoff pushed (concurrent kickoffs of one crew pop their own).
        scope = _TASK_SCOPE.get() if token is not None else None
        if scope is not None and scope.crew is not self._crew:
            scope = None
        if token is not None:
            try:
                _TASK_SCOPE.reset(token)
            except Exception:
                pass
        with _LISTENER_LOCK:
            stack = _ACTIVE_OPS.get(id(self._crew))
            if not stack:
                return
            if scope is not None and scope in stack:
                stack.remove(scope)
            else:
                stack.pop()
            if not stack:
                _ACTIVE_OPS.pop(id(self._crew), None)

    @staticmethod
    def _is_hook_aborted(exc: BaseException) -> bool:
        return any(cls.__name__ == "HookAborted" for cls in type(exc).__mro__)

    def _terminal_event(self, op: Any, mode: str, exc: BaseException) -> None:
        if isinstance(exc, _CANCEL_TYPES):
            emit_framework_event(
                op=op,
                event_type="workflow.error",
                on_event=self._on_event,
                payload={"mode": mode, "terminal": "cancelled", "error_type": type(exc).__name__},
            )
        elif self._is_hook_aborted(exc):
            emit_framework_event(
                op=op,
                event_type="workflow.suspend",
                on_event=self._on_event,
                payload={"mode": mode, "signal": "HookAborted", "error_type": type(exc).__name__},
            )
        else:
            emit_framework_event(
                op=op,
                event_type="workflow.error",
                on_event=self._on_event,
                payload={"mode": mode, "error_type": type(exc).__name__},
            )

    def kickoff(self, *args: Any, **kwargs: Any) -> Any:
        from kazenai.otel import build_control_attributes, workflow_span

        self._preflight()
        op = self._new_op()
        with enter_operation(op):
            with workflow_span(
                "kazenai.workflow",
                attributes=build_control_attributes(operation_id=op.operation_id),
            ):
                bound = self._bind(op)
                emit_framework_event(op=op, event_type="workflow.start", on_event=self._on_event, payload={"mode": "kickoff"})
                try:
                    result = self._crew.kickoff(*args, **kwargs)
                except BaseException as exc:
                    self._terminal_event(op, "kickoff", exc)
                    raise
                else:
                    emit_framework_event(op=op, event_type="workflow.end", on_event=self._on_event, payload={"mode": "kickoff"})
                    return result
                finally:
                    self._unbind(bound)

    async def akickoff(self, *args: Any, **kwargs: Any) -> Any:
        """Native async kickoff (``Crew.akickoff``)."""
        from kazenai.otel import build_control_attributes, workflow_span

        native = getattr(self._crew, "akickoff", None)
        if native is None:
            raise UnsupportedFrameworkPath(
                "Native akickoff() unavailable on this CrewAI version; use kickoff_async()",
                code="unsupported_framework_path",
            )
        self._preflight()
        op = self._new_op()
        with enter_operation(op):
            with workflow_span(
                "kazenai.workflow",
                attributes=build_control_attributes(operation_id=op.operation_id),
            ):
                bound = self._bind(op)
                emit_framework_event(op=op, event_type="workflow.start", on_event=self._on_event, payload={"mode": "akickoff"})
                try:
                    result = await native(*args, **kwargs)
                except BaseException as exc:
                    self._terminal_event(op, "akickoff", exc)
                    raise
                else:
                    emit_framework_event(op=op, event_type="workflow.end", on_event=self._on_event, payload={"mode": "akickoff"})
                    return result
                finally:
                    self._unbind(bound)

    async def kickoff_async(self, *args: Any, **kwargs: Any) -> Any:
        """``kickoff`` on a worker thread (CrewAI's ``kickoff_async`` model).

        The thread runs to completion even if the awaiting task is cancelled, so
        its spend is still checked and settled; prefer ``akickoff`` when you need
        cancellation to stop model calls.
        """
        return await asyncio.to_thread(self.kickoff, *args, **kwargs)

    def detach(self) -> None:
        _unregister_listener(self._listener)
        self._listener.clear_ownership()
        mark_detached(self)


def wrap_crew_kickoff(crew: Any, **adapter_kwargs: Any) -> Callable[..., Any]:
    """Compatibility shim: returns sync kickoff callable."""
    adapter = CrewAdapter(crew, **adapter_kwargs)

    def kickoff(*args: Any, **kwargs: Any) -> Any:
        return adapter.kickoff(*args, **kwargs)

    return kickoff


def wrap_crew(crew: Any, **kwargs: Any) -> CrewAdapter:
    """Preferred entry: kickoff, akickoff and kickoff_async adapter."""
    return CrewAdapter(crew, **kwargs)


def abort_on_budget_denial(exc: BaseException) -> BaseException:
    """Return ``exc`` as a CrewAI stop signal (``HookAborted``) when CrewAI is installed.

    The result is still an instance of the original KazenAI exception, so callers
    that catch ``BudgetExceeded`` keep working, while CrewAI treats it as a
    deliberate stop and does not retry the task.
    """
    stop = _crew_stop_class(type(exc))
    if stop is None or isinstance(exc, stop):
        return exc
    converted = stop.__new__(stop)
    converted.__dict__.update(getattr(exc, "__dict__", {}))
    converted.args = exc.args
    converted.__cause__ = exc
    return converted


_STOP_CLASSES: dict[type, type] = {}


def _crew_stop_class(base: type) -> Optional[type]:
    try:
        from crewai.hooks.dispatch import HookAborted  # type: ignore
    except Exception:
        try:
            from crewai.hooks import HookAborted  # type: ignore
        except Exception:
            return None
    if issubclass(base, HookAborted):
        return base
    with _LISTENER_LOCK:
        cls = _STOP_CLASSES.get(base)
        if cls is None:
            cls = type(f"CrewAI{base.__name__}", (base, HookAborted), {"__module__": __name__})
            _STOP_CLASSES[base] = cls
    return cls


_STOP_PATCH_LOCK = threading.Lock()
_STOP_PATCHED: set[type] = set()


def _install_crewai_stop_patch(llm_cls: type) -> None:
    """Make a KazenAI denial inside a bound CrewAI LLM a non-retried CrewAI stop.

    CrewAI retries a failed task ``max_retry_limit`` times unless the error is a
    ``HookAborted``. A budget denial is a deliberate stop, so on LLMs bound with
    ``bind_crew_llm_client`` it is re-raised as ``BudgetExceeded`` *and*
    ``HookAborted``. Unbound LLMs are untouched.
    """
    from ..enforcement import KazenBlock

    with _STOP_PATCH_LOCK:
        if llm_cls in _STOP_PATCHED:
            return
        orig_call = llm_cls.__dict__.get("call")
        orig_acall = llm_cls.__dict__.get("acall")

        if orig_call is not None:

            def call(self: Any, *args: Any, **kwargs: Any) -> Any:
                try:
                    return orig_call(self, *args, **kwargs)
                except KazenBlock as exc:
                    if getattr(self, "_kazenai_bound_guarded_client", None) is None:
                        raise
                    raise abort_on_budget_denial(exc) from exc

            call.__wrapped__ = orig_call  # type: ignore[attr-defined]
            setattr(llm_cls, "call", call)

        if orig_acall is not None:

            async def acall(self: Any, *args: Any, **kwargs: Any) -> Any:
                try:
                    return await orig_acall(self, *args, **kwargs)
                except KazenBlock as exc:
                    if getattr(self, "_kazenai_bound_guarded_client", None) is None:
                        raise
                    raise abort_on_budget_denial(exc) from exc

            acall.__wrapped__ = orig_acall  # type: ignore[attr-defined]
            setattr(llm_cls, "acall", acall)
        _STOP_PATCHED.add(llm_cls)


def _native_provider(llm: Any) -> Optional[str]:
    """``"openai"`` / ``"anthropic"`` for the certified CrewAI native LLM classes."""
    try:
        from crewai.llms.providers.openai.completion import OpenAICompletion  # type: ignore

        if type(llm) is OpenAICompletion:
            return "openai"
    except Exception:
        pass
    try:
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion  # type: ignore

        if type(llm) is AnthropicCompletion:
            return "anthropic"
    except Exception:
        pass
    return None


def _is_crewai_llm(obj: Any) -> bool:
    try:
        from crewai.llms.base_llm import BaseLLM  # type: ignore
    except Exception:
        return False
    return isinstance(obj, BaseLLM)


def _bind_crewai_native_llm(llm: Any, guarded: Any, monitor_kwargs: dict) -> bool:
    """CrewAI 1.15 native providers keep SDK clients in ``_client`` / ``_async_client``.

    Swap in monitored clients there (sync and async, on one budget) and keep the
    LLM object; replacing the LLM itself breaks the crew ("LLM must be resolved").
    Only the native OpenAI and Anthropic classes are certified; any other CrewAI
    LLM (Gemini, Azure, Bedrock, LiteLLM-routed) is refused untouched.
    """
    if not _is_crewai_llm(llm) and not (hasattr(llm, "_get_sync_client") and hasattr(llm, "_get_async_client")):
        return False
    provider = _native_provider(llm)
    if provider is None:
        raise UnsupportedFrameworkPath(
            f"bind_crew_llm_client: {type(llm).__name__} is not a certified CrewAI LLM; "
            "use a native OpenAI or Anthropic model (LLM(model='gpt-…') or 'anthropic/claude-…')",
            code="unguarded_control_boundary",
        )
    from ..monitor import _detect_client_kind, monitor

    client_is_anthropic = _detect_client_kind(guarded) == "anthropic"
    if (provider == "anthropic") != client_is_anthropic:
        raise UnsupportedFrameworkPath(
            f"bind_crew_llm_client: {type(llm).__name__} needs a matching "
            f"{'Anthropic' if provider == 'anthropic' else 'OpenAI'} client",
            code="unguarded_control_boundary",
        )
    try:
        raw_async = llm._async_client if llm._async_client is not None else llm._build_async_client()
        if is_monitored_client(raw_async):
            guarded_async = raw_async
        else:
            sibling = getattr(guarded, "_kazenai_monitor_sibling", None)
            # One budget for sync and async calls of this LLM.
            guarded_async = sibling(raw_async) if callable(sibling) else monitor(raw_async, **monitor_kwargs)
    except Exception:
        # No async client could be built (e.g. no API key yet): async calls would be
        # unguarded, so refuse rather than certify half the LLM.
        raise UnsupportedFrameworkPath(
            "bind_crew_llm_client could not monitor the CrewAI LLM async client",
            code="unguarded_control_boundary",
        )
    llm._client = guarded
    llm._async_client = guarded_async
    try:
        object.__setattr__(llm, "_kazenai_bound_guarded_client", guarded)
    except Exception:
        pass
    _install_crewai_stop_patch(type(llm))
    return True


def bind_crew_llm_client(crew_or_agent: Any, openai_client: Any, **monitor_kwargs: Any) -> Any:
    """Attach a monitored provider client to a CrewAI crew/agent LLM.

    CrewAI 1.15 native LLMs (``LLM(model="gpt-…")`` or an ``anthropic/claude-…``
    model) get monitored ``_client``/``_async_client`` on one budget; pass an
    OpenAI client for OpenAI models and an Anthropic client for Anthropic models.
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
        if _is_crewai_llm(native_llm) or isinstance(native_llm, str):
            # A CrewAI LLM we cannot bind (or a bare model name) must never be
            # replaced by the raw client; it stays unbound and Control refuses it.
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
        # Automatic bind must prove a supported LLM path; use
        # declare_crew_control_boundary for explicit caller-owned wiring.
        raise UnsupportedFrameworkPath(
            "bind_crew_llm_client found no LLM/client path to bind; "
            "use declare_crew_control_boundary(crew, guarded) for caller-owned Control",
            code="unguarded_control_boundary",
        )

    require_control_client(guarded, label="CrewAI provider client")
    return guarded
