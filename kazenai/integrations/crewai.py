"""CrewAI integration — wrap Crew.kickoff."""

from __future__ import annotations

from typing import Any, Callable


def wrap_crew_kickoff(crew: Any, **monitor_kwargs: Any) -> Callable[..., Any]:
    """Return a wrapped kickoff that runs inside a KazenAI RunContext when configured."""
    from ..context import RunContext, use_context

    original = getattr(crew, "kickoff", None)
    if original is None:
        raise TypeError("crew object has no kickoff method")

    org_id = str(monitor_kwargs.get("org_id") or "local")
    project_id = str(monitor_kwargs.get("project_id") or "default")
    agent_id = str(monitor_kwargs.get("agent_id") or "crewai")

    def kickoff(*args: Any, **kwargs: Any) -> Any:
        ctx = RunContext.new(org_id=org_id, project_id=project_id, agent_id=agent_id)
        with use_context(ctx):
            return original(*args, **kwargs)

    return kickoff
