"""LangGraph integration — wrap graph invoke."""

from __future__ import annotations

from typing import Any, Callable


def wrap_graph_invoke(graph: Any, **monitor_kwargs: Any) -> Callable[..., Any]:
    """Wrap compiled graph invoke with KazenAI RunContext."""
    from ..context import RunContext, use_context

    original = getattr(graph, "invoke", None)
    if original is None:
        raise TypeError("graph has no invoke method")

    org_id = str(monitor_kwargs.get("org_id") or "local")
    project_id = str(monitor_kwargs.get("project_id") or "default")
    agent_id = str(monitor_kwargs.get("agent_id") or "langgraph")

    def invoke(*args: Any, **kwargs: Any) -> Any:
        ctx = RunContext.new(org_id=org_id, project_id=project_id, agent_id=agent_id)
        with use_context(ctx):
            return original(*args, **kwargs)

    return invoke
