#!/usr/bin/env python3
"""Minimal LangChain + KazenAI FinOps ingest example.

  export KAZENAI_FINOPS_INGEST_URL=http://127.0.0.1:8090
  export KAZENAI_FINOPS_API_KEY=dev
  export KAZENAI_BUDGET_USD=1
  python examples/langchain_monitor.py
"""

from __future__ import annotations

import os

from kazenai.integrations.langchain import KazenCallbackHandler
from kazenai.schema import KazenEvent
from kazenai.sinks import HttpSink, HttpSinkConfig


def main() -> None:
    url = os.getenv("KAZENAI_FINOPS_INGEST_URL", "http://127.0.0.1:8090").rstrip("/")
    api_key = os.getenv("KAZENAI_FINOPS_API_KEY", "dev")

    sink = HttpSink(HttpSinkConfig(ingest_url=url + "/v1/events", api_key=api_key))

    def on_event(ev: KazenEvent) -> None:
        sink.emit(ev)
        print(ev.event_type, ev.cost_usd, ev.tokens_used)

    handler = KazenCallbackHandler(
        org_id=os.getenv("KAZENAI_ORG_ID", "local"),
        project_id=os.getenv("KAZENAI_PROJECT_ID", "default"),
        run_id="example-langchain",
        on_event=on_event,
    )

    # Simulate an LLM end callback without importing langchain
    class _FakeResponse:
        llm_output = {"token_usage": {"total_tokens": 120}}

    handler.on_llm_end(_FakeResponse())
    print("Emitted model.call to FinOps ingest")


if __name__ == "__main__":
    main()
