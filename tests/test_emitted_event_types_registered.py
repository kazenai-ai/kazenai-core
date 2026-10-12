"""Every ``event_type="..."`` literal core emits must be in the event-schema registry.

Consumers that call ``validate_kazen_event(strict=True)`` reject unregistered
types, so an unregistered emit is silently dropped downstream.
"""

from __future__ import annotations

import re
from pathlib import Path

from kazen_event_schema import KNOWN_EVENT_TYPES, normalize_event_type, validate_kazen_event

PKG = Path(__file__).resolve().parents[1] / "kazenai"
_LITERAL = re.compile(r'event_type\s*=\s*"([a-z0-9_.]+)"')


def _emitted_types() -> set[str]:
    found: set[str] = set()
    for path in PKG.rglob("*.py"):
        found.update(_LITERAL.findall(path.read_text(encoding="utf-8")))
    return found


def test_emitted_event_types_registered():
    emitted = _emitted_types()
    assert "model.call" in emitted  # scanner sanity
    missing = sorted(t for t in emitted if normalize_event_type(t) not in KNOWN_EVENT_TYPES)
    assert not missing, f"register in kazen-event-schema KNOWN_EVENT_TYPES: {missing}"
    for t in emitted:
        validate_kazen_event(
            {"org_id": "org", "agent_id": "agent", "event_type": t, "run_id": "r"},
            strict=True,
        )
