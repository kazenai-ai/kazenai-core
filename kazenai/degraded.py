"""Request-scoped degradation tracking for X-Kazen-Degraded response headers."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Iterable, List

_degraded: ContextVar[List[str]] = ContextVar("kazen_degraded_services", default=[])


def mark_degraded(*services: str) -> None:
    current = list(_degraded.get())
    for service in services:
        name = str(service or "").strip().lower()
        if name and name not in current:
            current.append(name)
    _degraded.set(current)


def degraded_services() -> List[str]:
    return list(_degraded.get())


def clear_degraded() -> None:
    _degraded.set([])


def degraded_header_value(services: Iterable[str] | None = None) -> str:
    names = list(services) if services is not None else degraded_services()
    return ",".join(names)
