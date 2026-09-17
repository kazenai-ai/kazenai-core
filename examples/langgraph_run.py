#!/usr/bin/env python3
"""LangGraph-style invoke wrapper example (no langgraph package required).

  export KAZENAI_FINOPS_INGEST_URL=http://127.0.0.1:8090
  export KAZENAI_FINOPS_API_KEY=dev
  python examples/langgraph_run.py
"""

from __future__ import annotations

import os

from kazenai.integrations.langgraph import wrap_graph_invoke


class _FakeGraph:
    def __init__(self, fn):
        self._fn = fn

    def invoke(self, state: dict) -> dict:
        return self._fn(state)


def _node(state: dict) -> dict:
    return {**state, "answer": "ok"}


def main() -> None:
    graph = _FakeGraph(_node)
    wrapped_invoke = wrap_graph_invoke(
        graph,
        org_id=os.getenv("KAZENAI_ORG_ID", "local"),
        project_id=os.getenv("KAZENAI_PROJECT_ID", "default"),
        agent_id="example-langgraph",
    )
    result = wrapped_invoke({"query": "hello"})
    print(result)


if __name__ == "__main__":
    main()
