"""A7: the committed KazenEvent JSON Schema export must match the installed model."""

from __future__ import annotations

import json
from pathlib import Path

from kazenai.schema import KazenEvent

EXPORT = Path(__file__).resolve().parents[1] / "contracts" / "kazenevent.schema.v1.json"


def test_committed_schema_export_is_current():
    expected = json.dumps(KazenEvent.model_json_schema(), indent=2, sort_keys=True) + "\n"
    assert EXPORT.read_text(encoding="utf-8") == expected, (
        "contracts/kazenevent.schema.v1.json is stale; run "
        "`python scripts/export_kazenevent_schema.py` and commit the result"
    )
