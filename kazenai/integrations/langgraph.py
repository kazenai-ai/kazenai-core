"""LangGraph integration — durable operation correlation + multi-mode wrap.

FinOps remains the financial authority. Checkpoints store only a minimal
correlation envelope (no credentials, clients, or span objects).
"""

from __future__ import annotations

import inspect
import os
from typing import Any, AsyncIterator, Callable, Iterator, Optional

from .adapter_contract import (
    CORRELATION_ENVELOPE_VERSION,
    CapabilityLevel,
    UnsupportedFrameworkPath,
    correlation_envelope,
    emit_framework_event,
    enter_operation,
    is_monitored_client,
    mark_attached,
    mark_detached,
    new_operation,
    reject_openai_responses_api,
    require_control_client,
    restore_operation_from_envelope,
)


CONFIG_KEY = "kazenai_correlation"

# Process-local resume cache. Keyed by tenancy + workflow + thread so two graphs
# sharing a common thread name cannot collide (U06). Values expire.
_THREAD_ENVELOPES: dict[tuple[str, ...], tuple[float, dict[str, Any]]] = {}
_THREAD_ENVELOPE_TTL_S = 24 * 3600.0


def _envelope_store_root() -> str:
    """Durable side-store root — LangGraph checkpointers drop custom config keys."""
    import tempfile

    raw = (os.getenv("KAZENAI_LANGGRAPH_ENVELOPE_DIR") or "").strip()
    if raw:
        os.makedirs(raw, exist_ok=True)
        return raw
    fallback = os.path.join(tempfile.gettempdir(), "kazenai-langgraph-envelopes")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def _durable_envelope_path(key: tuple[str, ...]) -> str:
    import hashlib

    digest = hashlib.sha256("|".join(key).encode("utf-8")).hexdigest()[:40]
    return os.path.join(_envelope_store_root(), f"{digest}.json")


def _persist_durable_envelope(key: tuple[str, ...], envelope: dict[str, Any]) -> None:
    """Write correlation envelope so AsyncSqliteSaver restart keeps operation_id (U06)."""
    import json
    import time

    path = _durable_envelope_path(key)
    payload = {"key": list(key), "envelope": dict(envelope), "stored_at": time.time()}
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _load_durable_envelope(key: tuple[str, ...]) -> dict[str, Any] | None:
    import json
    import time

    path = _durable_envelope_path(key)
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    stored_at = float(raw.get("stored_at") or 0.0)
    if stored_at and time.time() - stored_at > _THREAD_ENVELOPE_TTL_S:
        try:
            os.remove(path)
        except Exception:
            pass
        return None
    env = raw.get("envelope")
    return dict(env) if isinstance(env, dict) else None


def _checkpointer_store_id(graph: Any) -> str:
    """Bind side-store identity to the checkpoint store, not only thread_id (F06)."""
    override = (os.getenv("KAZENAI_LANGGRAPH_STORE_ID") or "").strip()
    if override:
        return override
    cp = getattr(graph, "checkpointer", None)
    if cp is None:
        cp = getattr(graph, "_checkpointer", None)
    if cp is None:
        return ""
    for attr in ("conn_string", "_conn_string", "db_path", "path", "sqlite_path"):
        raw = getattr(cp, attr, None)
        if raw:
            text = str(raw)
            return os.path.abspath(text) if text not in {":memory:", ""} else text
    conn = getattr(cp, "conn", None) or getattr(cp, "_conn", None)
    if conn is not None:
        for attr in ("path", "database", "db", "_database"):
            raw = getattr(conn, attr, None)
            if raw:
                text = str(raw)
                return os.path.abspath(text) if text not in {":memory:", ""} else text
        args = getattr(conn, "_args", None)
        if isinstance(args, (tuple, list)) and args and args[0]:
            text = str(args[0])
            return os.path.abspath(text) if text not in {":memory:", ""} else text
    # Distinct process-local savers without a path still must not collide.
    return f"checkpointer:{type(cp).__name__}:{id(cp)}"


def _thread_cache_key(
    *,
    org_id: str,
    workspace_id: str,
    project_id: str,
    agent_id: str,
    thread_id: str,
    checkpoint_ns: str = "",
    store_id: str = "",
) -> tuple[str, ...]:
    return (
        str(org_id or "").strip(),
        str(workspace_id or "").strip(),
        str(project_id or "").strip(),
        str(agent_id or "").strip(),
        str(checkpoint_ns or "").strip(),
        str(store_id or "").strip(),
        str(thread_id or "").strip(),
    )


def _store_thread_envelope(key: tuple[str, ...], envelope: dict[str, Any]) -> None:
    import time

    now = time.time()
    # Opportunistic TTL prune (bounded map).
    if len(_THREAD_ENVELOPES) > 256:
        stale = [k for k, (ts, _) in _THREAD_ENVELOPES.items() if now - ts > _THREAD_ENVELOPE_TTL_S]
        for k in stale:
            _THREAD_ENVELOPES.pop(k, None)
    _THREAD_ENVELOPES[key] = (now, dict(envelope))
    # LangGraph checkpointers retain only thread_id/checkpoint_id in configurable —
    # custom keys and invoke metadata are dropped. Persist a durable side copy so
    # process restart (cleared module cache) still recovers the same operation_id.
    _persist_durable_envelope(key, envelope)


def _load_thread_envelope(key: tuple[str, ...]) -> dict[str, Any] | None:
    import time

    hit = _THREAD_ENVELOPES.get(key)
    if hit is not None:
        ts, envelope = hit
        if time.time() - ts > _THREAD_ENVELOPE_TTL_S:
            _THREAD_ENVELOPES.pop(key, None)
        else:
            return dict(envelope)
    durable = _load_durable_envelope(key)
    if durable is not None:
        # Warm the process cache after restart recovery.
        _THREAD_ENVELOPES[key] = (time.time(), dict(durable))
        return dict(durable)
    return None


def declare_langgraph_control_boundary(graph: Any, *guarded_clients: Any) -> Any:
    """Mark that ``graph`` nodes perform provider calls via ``guarded_clients``.

    Pass every monitored client the nodes use — for a multi-model graph, one per
    provider (OpenAI, Anthropic, ...). Callers must bind the clients the nodes
    actually use; a pointer to an unrelated monitored client is not Control
    proof — see LangGraphAdapter.
    """
    if not guarded_clients:
        raise UnsupportedFrameworkPath(
            "declare_langgraph_control_boundary needs at least one monitored client",
            code="unguarded_control_boundary",
        )
    for client in guarded_clients:
        require_control_client(client, label="declared langgraph control boundary")
    first = guarded_clients[0]
    try:
        setattr(graph, "_kazenai_bound_guarded_client", first)
        setattr(graph, "_kazenai_langgraph_control_boundary", first)
        setattr(graph, "_kazenai_bound_guarded_clients", tuple(guarded_clients))
        # Stash identity so adapters can require graph-bound (not decoy) clients.
        setattr(graph, "_kazenai_control_boundary_id", id(first))
        setattr(graph, "_kazenai_control_boundary_ids", tuple(id(c) for c in guarded_clients))
    except Exception:
        pass
    return graph


def _result_is_interrupt(result: Any) -> bool:
    """LangGraph 1.2+ may return ``__interrupt__`` instead of raising on invoke."""
    if isinstance(result, dict) and result.get("__interrupt__"):
        return True
    if isinstance(result, (list, tuple)):
        for item in result:
            if _result_is_interrupt(item):
                return True
    return False


def _is_interrupt_exc(exc: BaseException) -> bool:
    name = type(exc).__name__
    return "Interrupt" in name or "GraphInterrupt" in name


def _is_cancel_exc(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in {"CancelledError", "GeneratorExit"} or isinstance(exc, GeneratorExit)


class LangGraphAdapter:
    """Proxy over a compiled graph exposing invoke/ainvoke/stream/astream."""

    def __init__(
        self,
        graph: Any,
        *,
        org_id: str = "local",
        workspace_id: str = "default",
        project_id: str = "default",
        agent_id: str = "langgraph",
        operation_id: str | None = None,
        capability: CapabilityLevel | str = CapabilityLevel.OBSERVE,
        on_event: Callable[[Any], Any] | None = None,
        stream_mode: str | None = None,
        guarded_client: Any | None = None,
        guarded_clients: Any | None = None,
    ) -> None:
        if not any(hasattr(graph, m) for m in ("invoke", "ainvoke", "stream", "astream")):
            raise TypeError("graph has no invoke/ainvoke/stream/astream")
        self._graph = graph
        self._on_event = on_event
        self._stream_mode = stream_mode
        self._capability = (
            capability if isinstance(capability, CapabilityLevel) else CapabilityLevel(str(capability))
        )
        # Template only — each independent invocation gets a fresh operation (R14).
        self._op_template = dict(
            framework="langgraph",
            org_id=org_id,
            workspace_id=workspace_id,
            project_id=project_id,
            agent_id=agent_id,
            operation_id=operation_id,
            capability=self._capability,
        )
        if self._capability == CapabilityLevel.CONTROL:
            gateway = os.getenv("KAZENAI_ENFORCEMENT_OWNER", "").strip().lower() == "gateway"
            graph_bound = (
                getattr(graph, "_kazenai_bound_guarded_client", None)
                or getattr(graph, "_kazenai_langgraph_control_boundary", None)
            )
            bound_all = tuple(getattr(graph, "_kazenai_bound_guarded_clients", None) or ())
            if not bound_all and graph_bound is not None:
                bound_all = (graph_bound,)
            requested = tuple(guarded_clients or ()) or ((guarded_client,) if guarded_client is not None else ())
            # Multi-provider graphs pass every client; each must be one the graph declared.
            for extra in requested[1:]:
                reject_openai_responses_api(extra)
                require_control_client(extra, label="LangGraph provider client")
                if not any(extra is b for b in bound_all) and not gateway:
                    raise UnsupportedFrameworkPath(
                        "LangGraph Control guarded_clients must all be declared via "
                        "declare_langgraph_control_boundary(graph, *clients)",
                        code="unguarded_control_boundary",
                    )
            guarded_client = requested[0] if requested else None
            declared = guarded_client or graph_bound
            if declared is None and not gateway:
                raise UnsupportedFrameworkPath(
                    "LangGraph Control requires guarded_client=monitor(...) (or "
                    "declare_langgraph_control_boundary), or KAZENAI_ENFORCEMENT_OWNER=gateway",
                    code="unguarded_control_boundary",
                )
            if declared is not None:
                reject_openai_responses_api(declared)
                require_control_client(declared, label="LangGraph provider client")
                # F03/U05: a decoy monitored client is not Control proof.
                # wrap_graph does NOT auto-declare — callers must bind explicitly.
                # Gateway-owned Control may omit.
                if not gateway:
                    if graph_bound is None:
                        raise UnsupportedFrameworkPath(
                            "LangGraph Control requires declare_langgraph_control_boundary("
                            "graph, guarded_client) before wrap_graph so the monitored "
                            "client is bound into the graph — passing an unrelated "
                            "monitored client is rejected",
                            code="unguarded_control_boundary",
                        )
                    if not any(declared is b for b in bound_all):
                        raise UnsupportedFrameworkPath(
                            "LangGraph Control guarded_client must be the same object "
                            "bound via declare_langgraph_control_boundary",
                            code="unguarded_control_boundary",
                        )
            self._guarded_client = declared
            self._guarded_clients = requested or ((declared,) if declared is not None else ())
        else:
            self._guarded_client = guarded_client
            self._guarded_clients = tuple(guarded_clients or ()) or ((guarded_client,) if guarded_client is not None else ())
        mark_attached(self)

    def _merge_config(self, config: dict[str, Any] | None, op) -> dict[str, Any]:
        cfg = dict(config or {})
        configurable = dict(cfg.get("configurable") or {})
        # Trusted tenancy stays in process context; envelope is correlation only.
        envelope = correlation_envelope(op)
        configurable[CONFIG_KEY] = envelope
        # Also stamp metadata for consumers that retain it (not AsyncSqliteSaver today).
        metadata = dict(cfg.get("metadata") or {})
        metadata[CONFIG_KEY] = envelope
        cfg["metadata"] = metadata
        thread_id = str(configurable.get("thread_id") or "").strip()
        if thread_id:
            key = _thread_cache_key(
                org_id=str(self._op_template.get("org_id") or ""),
                workspace_id=str(self._op_template.get("workspace_id") or ""),
                project_id=str(self._op_template.get("project_id") or ""),
                agent_id=str(self._op_template.get("agent_id") or ""),
                thread_id=thread_id,
                checkpoint_ns=str(configurable.get("checkpoint_ns") or ""),
                store_id=_checkpointer_store_id(self._graph),
            )
            _store_thread_envelope(key, envelope)
        cfg["configurable"] = configurable
        return cfg

    def _call(self, method: str, input: Any, config: dict | None, **kwargs: Any) -> Any:
        fn = getattr(self._graph, method)
        try:
            sig = inspect.signature(fn)
            accepts_config = "config" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (TypeError, ValueError):
            accepts_config = True
        if accepts_config:
            return fn(input, config=config, **kwargs)
        return fn(input, **kwargs)

    async def _acall(self, method: str, input: Any, config: dict | None, **kwargs: Any) -> Any:
        fn = getattr(self._graph, method)
        try:
            sig = inspect.signature(fn)
            accepts_config = "config" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (TypeError, ValueError):
            accepts_config = True
        if accepts_config:
            return await fn(input, config=config, **kwargs)
        return await fn(input, **kwargs)

    def _parse_checkpoint_snap(self, snap: Any) -> Any | None:
        for candidate in (
            getattr(snap, "config", None),
            getattr(snap, "metadata", None),
            getattr(snap, "values", None),
        ):
            if not isinstance(candidate, dict):
                continue
            configurable = candidate.get("configurable") if "configurable" in candidate else candidate
            if isinstance(configurable, dict):
                env = configurable.get(CONFIG_KEY)
                restored = restore_operation_from_envelope(env)
                if restored is not None:
                    return restored
            env = candidate.get(CONFIG_KEY) if isinstance(candidate, dict) else None
            restored = restore_operation_from_envelope(env)
            if restored is not None:
                return restored
        return None

    def _envelope_from_checkpoint(self, config: dict[str, Any] | None) -> Any | None:
        """U06: recover correlation envelope from checkpointer when resume omits it."""
        if config is None or not hasattr(self._graph, "get_state"):
            return None
        try:
            snap = self._graph.get_state(config)
        except Exception:
            return None
        return self._parse_checkpoint_snap(snap)

    async def _envelope_from_checkpoint_async(self, config: dict[str, Any] | None) -> Any | None:
        """U06: async twin — prefer ``aget_state`` for AsyncSqliteSaver restart."""
        if config is None:
            return None
        aget = getattr(self._graph, "aget_state", None)
        if callable(aget):
            try:
                snap = await aget(config)
                restored = self._parse_checkpoint_snap(snap)
                if restored is not None:
                    return restored
            except Exception:
                pass
        # Fall back to sync get_state only when no async saver path worked.
        return self._envelope_from_checkpoint(config)

    def _thread_has_checkpoint(self, config: dict[str, Any] | None) -> bool:
        """True when the checkpointer already has state for this thread (resume)."""
        if config is None or not hasattr(self._graph, "get_state"):
            return False
        try:
            snap = self._graph.get_state(config)
        except Exception:
            return False
        if snap is None:
            return False
        values = getattr(snap, "values", None)
        if values:
            return True
        nxt = getattr(snap, "next", None)
        if nxt:
            return True
        snap_cfg = getattr(snap, "config", None)
        if isinstance(snap_cfg, dict):
            ck = (snap_cfg.get("configurable") or {}).get("checkpoint_id")
            if ck:
                return True
        return False

    async def _thread_has_checkpoint_async(self, config: dict[str, Any] | None) -> bool:
        if config is None:
            return False
        aget = getattr(self._graph, "aget_state", None)
        if callable(aget):
            try:
                snap = await aget(config)
                if snap is None:
                    return False
                if getattr(snap, "values", None) or getattr(snap, "next", None):
                    return True
                snap_cfg = getattr(snap, "config", None)
                if isinstance(snap_cfg, dict):
                    ck = (snap_cfg.get("configurable") or {}).get("checkpoint_id")
                    if ck:
                        return True
                return False
            except Exception:
                pass
        return self._thread_has_checkpoint(config)

    def _bind_restored_op(self, restored: Any):
        """Apply restored correlation; detect explicit operation_id conflicts (F06)."""
        explicit = self._op_template.get("operation_id")
        if explicit and str(explicit) != str(restored.operation_id):
            raise UnsupportedFrameworkPath(
                "LangGraph correlation conflict: constructor operation_id="
                f"{explicit!r} differs from resumed {restored.operation_id!r}",
                code="correlation_conflict",
            )
        return new_operation(
            framework="langgraph",
            org_id=self._op_template["org_id"],
            workspace_id=self._op_template["workspace_id"],
            project_id=self._op_template["project_id"],
            agent_id=self._op_template["agent_id"],
            operation_id=restored.operation_id,
            run_id=restored.run_id,
            capability=self._capability,
            metadata={"restored_from_checkpoint": True},
        )

    def _cache_key_for_config(self, cfg: dict[str, Any]) -> tuple[str, ...] | None:
        thread_id = str(cfg.get("thread_id") or "").strip()
        if not thread_id:
            return None
        return _thread_cache_key(
            org_id=str(self._op_template.get("org_id") or ""),
            workspace_id=str(self._op_template.get("workspace_id") or ""),
            project_id=str(self._op_template.get("project_id") or ""),
            agent_id=str(self._op_template.get("agent_id") or ""),
            thread_id=thread_id,
            checkpoint_ns=str(cfg.get("checkpoint_ns") or ""),
            store_id=_checkpointer_store_id(self._graph),
        )

    def _fresh_op(self, config: dict[str, Any] | None):
        """Resume from checkpoint envelope when present; otherwise new operation_id."""
        cfg = (config or {}).get("configurable") or {}
        restored = restore_operation_from_envelope(cfg.get(CONFIG_KEY))
        if restored is None:
            restored = self._envelope_from_checkpoint(config)
        # Side-store only for threads that already have checkpoint state (F06).
        if restored is None and self._thread_has_checkpoint(config):
            key = self._cache_key_for_config(cfg)
            if key is not None:
                cached = _load_thread_envelope(key)
                if cached is not None:
                    restored = restore_operation_from_envelope(cached)
        if restored is None:
            kwargs = dict(self._op_template)
            # Preserve explicit constructor operation_id when provided (U06).
            # Independent invocations without an envelope and without a seed mint anew.
            if not kwargs.get("operation_id"):
                kwargs["operation_id"] = None
            return new_operation(**kwargs)
        return self._bind_restored_op(restored)

    async def _fresh_op_async(self, config: dict[str, Any] | None):
        """Async resume path — uses aget_state for durable restart (U06)."""
        cfg = (config or {}).get("configurable") or {}
        restored = restore_operation_from_envelope(cfg.get(CONFIG_KEY))
        if restored is None:
            restored = await self._envelope_from_checkpoint_async(config)
        if restored is None and await self._thread_has_checkpoint_async(config):
            key = self._cache_key_for_config(cfg)
            if key is not None:
                cached = _load_thread_envelope(key)
                if cached is not None:
                    restored = restore_operation_from_envelope(cached)
        if restored is None:
            return self._fresh_op(config)
        return self._bind_restored_op(restored)

    def invoke(self, input: Any = None, config: dict | None = None, **kwargs: Any) -> Any:
        from kazenai.otel import build_control_attributes, workflow_span

        op = self._fresh_op(config)
        with enter_operation(op):
            with workflow_span(
                "kazenai.workflow",
                attributes=build_control_attributes(
                    operation_id=op.operation_id,
                    workflow_id=str(getattr(op, "operation_id", "") or "") or None,
                ),
            ):
                emit_framework_event(op=op, event_type="workflow.start", on_event=self._on_event, payload={"mode": "invoke"})
                cfg = self._merge_config(config, op)
                try:
                    result = self._call("invoke", input, cfg, **kwargs)
                except Exception as exc:
                    if _is_interrupt_exc(exc):
                        emit_framework_event(
                            op=op,
                            event_type="workflow.suspend",
                            on_event=self._on_event,
                            payload={"mode": "invoke", "signal": type(exc).__name__},
                        )
                    else:
                        emit_framework_event(
                            op=op,
                            event_type="workflow.error",
                            on_event=self._on_event,
                            payload={"mode": "invoke", "error_type": type(exc).__name__},
                        )
                    raise
                if _result_is_interrupt(result):
                    emit_framework_event(
                        op=op,
                        event_type="workflow.suspend",
                        on_event=self._on_event,
                        payload={"mode": "invoke", "signal": "__interrupt__"},
                    )
                    return result
                emit_framework_event(op=op, event_type="workflow.end", on_event=self._on_event, payload={"mode": "invoke"})
                return result

    async def ainvoke(self, input: Any = None, config: dict | None = None, **kwargs: Any) -> Any:
        from kazenai.otel import build_control_attributes, workflow_span

        op = await self._fresh_op_async(config)
        with enter_operation(op):
            with workflow_span(
                "kazenai.workflow",
                attributes=build_control_attributes(
                    operation_id=op.operation_id,
                    workflow_id=str(getattr(op, "operation_id", "") or "") or None,
                ),
            ):
                emit_framework_event(op=op, event_type="workflow.start", on_event=self._on_event, payload={"mode": "ainvoke"})
                cfg = self._merge_config(config, op)
                try:
                    result = await self._acall("ainvoke", input, cfg, **kwargs)
                except Exception as exc:
                    if _is_interrupt_exc(exc):
                        emit_framework_event(
                            op=op,
                            event_type="workflow.suspend",
                            on_event=self._on_event,
                            payload={"mode": "ainvoke", "signal": type(exc).__name__, "error_type": type(exc).__name__},
                        )
                    else:
                        emit_framework_event(
                            op=op,
                            event_type="workflow.error",
                            on_event=self._on_event,
                            payload={"mode": "ainvoke", "error_type": type(exc).__name__},
                        )
                    raise
                if _result_is_interrupt(result):
                    emit_framework_event(
                        op=op,
                        event_type="workflow.suspend",
                        on_event=self._on_event,
                        payload={"mode": "ainvoke", "signal": "__interrupt__"},
                    )
                    return result
                emit_framework_event(op=op, event_type="workflow.end", on_event=self._on_event, payload={"mode": "ainvoke"})
                return result

    def stream(self, input: Any = None, config: dict | None = None, **kwargs: Any) -> Iterator[Any]:
        """Yield chunks without leaving framework ContextVar active in the consumer."""
        op = self._fresh_op(config)
        with enter_operation(op):
            emit_framework_event(op=op, event_type="workflow.start", on_event=self._on_event, payload={"mode": "stream"})
            cfg = self._merge_config(config, op)
            if self._stream_mode is not None and "stream_mode" not in kwargs:
                kwargs["stream_mode"] = self._stream_mode
            fn = getattr(self._graph, "stream")
            try:
                sig = inspect.signature(fn)
                accepts_config = "config" in sig.parameters or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                )
            except (TypeError, ValueError):
                accepts_config = True
            iterator = fn(input, config=cfg, **kwargs) if accepts_config else fn(input, **kwargs)

        terminal = "success"
        try:
            while True:
                with enter_operation(op):
                    try:
                        item = next(iterator)
                    except StopIteration:
                        break
                    except Exception as exc:
                        if _is_interrupt_exc(exc):
                            terminal = "suspend"
                            emit_framework_event(
                                op=op,
                                event_type="workflow.suspend",
                                on_event=self._on_event,
                                payload={"mode": "stream", "signal": type(exc).__name__},
                            )
                        else:
                            terminal = "error"
                            emit_framework_event(
                                op=op,
                                event_type="workflow.error",
                                on_event=self._on_event,
                                payload={"mode": "stream", "error_type": type(exc).__name__},
                            )
                        raise
                if _result_is_interrupt(item):
                    terminal = "suspend"
                    with enter_operation(op):
                        emit_framework_event(
                            op=op,
                            event_type="workflow.suspend",
                            on_event=self._on_event,
                            payload={"mode": "stream", "signal": "__interrupt__"},
                        )
                    yield item
                    break
                yield item  # consumer sees no active framework ContextVar
        except BaseException as exc:
            if _is_cancel_exc(exc):
                terminal = "cancel"
                with enter_operation(op):
                    emit_framework_event(
                        op=op,
                        event_type="workflow.error",
                        on_event=self._on_event,
                        payload={"mode": "stream", "error_type": type(exc).__name__},
                    )
            raise
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            if terminal == "success":
                with enter_operation(op):
                    emit_framework_event(
                        op=op, event_type="workflow.end", on_event=self._on_event, payload={"mode": "stream"}
                    )

    async def astream(self, input: Any = None, config: dict | None = None, **kwargs: Any) -> AsyncIterator[Any]:
        op = await self._fresh_op_async(config)
        with enter_operation(op):
            emit_framework_event(op=op, event_type="workflow.start", on_event=self._on_event, payload={"mode": "astream"})
            cfg = self._merge_config(config, op)
            if self._stream_mode is not None and "stream_mode" not in kwargs:
                kwargs["stream_mode"] = self._stream_mode
            fn = getattr(self._graph, "astream")
            try:
                sig = inspect.signature(fn)
                accepts_config = "config" in sig.parameters or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                )
            except (TypeError, ValueError):
                accepts_config = True
            iterator = fn(input, config=cfg, **kwargs) if accepts_config else fn(input, **kwargs)

        terminal = "success"
        try:
            while True:
                with enter_operation(op):
                    try:
                        item = await iterator.__anext__()
                    except StopAsyncIteration:
                        break
                    except Exception as exc:
                        if _is_interrupt_exc(exc):
                            terminal = "suspend"
                            emit_framework_event(
                                op=op,
                                event_type="workflow.suspend",
                                on_event=self._on_event,
                                payload={"mode": "astream", "signal": type(exc).__name__},
                            )
                        else:
                            terminal = "error"
                            emit_framework_event(
                                op=op,
                                event_type="workflow.error",
                                on_event=self._on_event,
                                payload={"mode": "astream", "error_type": type(exc).__name__},
                            )
                        raise
                if _result_is_interrupt(item):
                    terminal = "suspend"
                    with enter_operation(op):
                        emit_framework_event(
                            op=op,
                            event_type="workflow.suspend",
                            on_event=self._on_event,
                            payload={"mode": "astream", "signal": "__interrupt__"},
                        )
                    yield item
                    break
                yield item
        except BaseException as exc:
            if _is_cancel_exc(exc):
                terminal = "cancel"
                with enter_operation(op):
                    emit_framework_event(
                        op=op,
                        event_type="workflow.error",
                        on_event=self._on_event,
                        payload={"mode": "astream", "error_type": type(exc).__name__},
                    )
            raise
        finally:
            aclose = getattr(iterator, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except Exception:
                    pass
            else:
                close = getattr(iterator, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            if terminal == "success":
                with enter_operation(op):
                    emit_framework_event(
                        op=op, event_type="workflow.end", on_event=self._on_event, payload={"mode": "astream"}
                    )

    def resume(self, input: Any = None, config: dict | None = None, **kwargs: Any) -> Any:
        """Resume after interrupt — stable operation_id from checkpoint envelope."""
        return self.invoke(input, config=config, **kwargs)

    def detach(self) -> None:
        mark_detached(self)


def wrap_graph_invoke(graph: Any, **monitor_kwargs: Any) -> Callable[..., Any]:
    """Compatibility shim: returns sync ``invoke`` callable (historical API)."""
    adapter = wrap_graph(graph, **monitor_kwargs)

    def invoke(*args: Any, **kwargs: Any) -> Any:
        # Historical LangGraph: invoke(input, config) — preserve second positional config.
        if len(args) >= 2 and "config" not in kwargs:
            return adapter.invoke(args[0], args[1], *args[2:], **kwargs)
        if args:
            return adapter.invoke(args[0], **kwargs)
        return adapter.invoke(**kwargs)

    return invoke


def wrap_graph(graph: Any, **kwargs: Any) -> LangGraphAdapter:
    """Preferred entry: multi-mode LangGraph adapter.

    Control does **not** auto-declare a boundary. Callers must first
    ``declare_langgraph_control_boundary(graph, guarded_client)`` with the
    client graph nodes actually use. Passing an unused monitored client to
    ``wrap_graph`` alone is rejected (F03).
    """
    return LangGraphAdapter(graph, **kwargs)


def unsupported_control_entry(*_a: Any, **_k: Any) -> None:
    raise UnsupportedFrameworkPath(
        "This LangGraph entry point is not Control-verified; use wrap_graph() modes only",
        code="unsupported_framework_path",
    )


def envelope_version() -> str:
    return CORRELATION_ENVELOPE_VERSION
