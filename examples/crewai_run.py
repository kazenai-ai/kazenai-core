#!/usr/bin/env python3
"""CrewAI-style kickoff wrapper example (no crewai package required).

  export KAZENAI_FINOPS_INGEST_URL=http://127.0.0.1:8090
  export KAZENAI_FINOPS_API_KEY=dev
  python examples/crewai_run.py
"""

from __future__ import annotations

import os

from kazenai.integrations.crewai import wrap_crew_kickoff


class _FakeCrew:
    def kickoff(self, inputs: dict | None = None) -> str:
        return f"done:{inputs}"


def main() -> None:
    crew = _FakeCrew()
    wrapped_kickoff = wrap_crew_kickoff(
        crew,
        org_id=os.getenv("KAZENAI_ORG_ID", "local"),
        project_id=os.getenv("KAZENAI_PROJECT_ID", "default"),
        agent_id="example-crewai",
    )
    out = wrapped_kickoff(inputs={"task": "demo"})
    print(out)


if __name__ == "__main__":
    main()
