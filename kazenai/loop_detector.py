from __future__ import annotations

import hashlib
import re
import threading
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from typing import Iterable, List, Optional


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# What monitor() does when a request looks like a loop. "record" (the default since
# 1.2.3) writes a finops.loop.anomaly event (reason "repeated_request") and sends
# the call; "block" raises LoopDetected before dispatch (the behaviour before 1.2.3);
# "off" skips scoring.
LOOP_GUARD_RECORD = "record"
LOOP_GUARD_BLOCK = "block"
LOOP_GUARD_OFF = "off"
LOOP_GUARD_MODES = (LOOP_GUARD_RECORD, LOOP_GUARD_BLOCK, LOOP_GUARD_OFF)


def resolve_loop_guard(mode: object) -> str:
    """Return a valid loop_guard mode, or raise ValueError naming the allowed ones."""
    value = mode.strip().lower() if isinstance(mode, str) else None
    if value not in LOOP_GUARD_MODES:
        raise ValueError(f"loop_guard must be one of {', '.join(repr(m) for m in LOOP_GUARD_MODES)}; got {mode!r}")
    return value


def _normalize_tokens(text: str) -> frozenset[str]:
    if not text:
        return frozenset()
    return frozenset(m.group(0) for m in _TOKEN_RE.finditer(text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = (len(a) + len(b) - inter)
    return inter / union if union else 0.0


def _tool_fingerprint(tool_names: Iterable[str]) -> str:
    normalized = [t.strip().lower() for t in tool_names if t and t.strip()]
    joined = "\n".join(normalized)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _dataclass(*args, **kwargs):
    # Python 3.9 dataclasses.dataclass does not support `slots=`.
    try:
        return dataclass(*args, **kwargs)
    except TypeError:
        kwargs.pop("slots", None)
        return dataclass(*args, **kwargs)


@_dataclass(frozen=True, slots=True)
class LoopScores:
    h1_jaccard: float
    h2_tools: float
    combined: float


class _History:
    """Recent inputs and tool fingerprints for one customer (or for calls naming none)."""

    __slots__ = ("inputs", "tool_fps", "tool_counts")

    def __init__(self, max_history: int, tool_history: int) -> None:
        self.inputs: deque[frozenset[str]] = deque(maxlen=max_history)
        self.tool_fps: deque[str] = deque(maxlen=tool_history)
        self.tool_counts: Counter[str] = Counter()


class LoopDetector:
    """
    Mandatory heuristics:
      Jaccard similarity (>0.85) on normalized input tokens.
      SHA1 fingerprinting of ordered tool-name sequences.

    History is kept per customer (``subject``) since 1.2.3, so two customers sending
    the same prompt don't count as a loop. Calls naming no customer share one history.
    At most ``max_subjects`` customer histories are kept; the least recently active
    one is dropped first. A looping customer is by definition active, so it stays.
    """

    def __init__(
        self,
        *,
        max_history: int = 32,
        tool_history: int = 64,
        h1_threshold: float = 0.85,
        max_subjects: int = 64,
    ) -> None:
        self._max_history = int(max_history)
        self._tool_history = int(tool_history)
        self._h1_threshold = float(h1_threshold)
        self._max_subjects = max(0, int(max_subjects))

        self._lock = threading.Lock()
        self._shared = _History(self._max_history, self._tool_history)
        self._subjects: OrderedDict[str, _History] = OrderedDict()

    def _history_for(self, subject: Optional[str]) -> _History:
        # Caller holds self._lock.
        key = subject.strip() if isinstance(subject, str) else None
        if not key or self._max_subjects == 0:
            return self._shared
        hist = self._subjects.get(key)
        if hist is None:
            hist = _History(self._max_history, self._tool_history)
            self._subjects[key] = hist
            while len(self._subjects) > self._max_subjects:
                self._subjects.popitem(last=False)
        else:
            self._subjects.move_to_end(key)
        return hist

    def score(
        self,
        *,
        input_text: Optional[str],
        tool_names: Optional[List[str]] = None,
        subject: Optional[str] = None,
    ) -> LoopScores:
        tokens = _normalize_tokens(input_text or "")

        with self._lock:
            hist = self._history_for(subject)

            # H1
            h1 = 0.0
            for prev in hist.inputs:
                sim = _jaccard(tokens, prev)
                if sim > h1:
                    h1 = sim
                    if h1 >= 0.999:  # fast path
                        break
            hist.inputs.append(tokens)

            # H2
            h2 = 0.0
            if tool_names is not None:
                fp = _tool_fingerprint(tool_names)
                # maintain counts over sliding window
                if len(hist.tool_fps) == hist.tool_fps.maxlen:
                    old = hist.tool_fps[0]
                    hist.tool_counts[old] -= 1
                    if hist.tool_counts[old] <= 0:
                        del hist.tool_counts[old]
                hist.tool_fps.append(fp)
                hist.tool_counts[fp] += 1

                c = hist.tool_counts[fp]
                h2 = 1.0 if c >= 3 else (c / 3.0)

        combined = max(h1, h2)
        return LoopScores(h1_jaccard=h1, h2_tools=h2, combined=combined)

    def is_loop(self, scores: LoopScores) -> bool:
        return scores.h1_jaccard > self._h1_threshold or scores.h2_tools >= 1.0
