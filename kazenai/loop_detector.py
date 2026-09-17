from __future__ import annotations

import hashlib
import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable, List, Optional


_TOKEN_RE = re.compile(r"[a-z0-9]+")


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


class LoopDetector:
    """
    Mandatory heuristics:
      H1: Jaccard similarity (>0.85) on normalized input tokens.
      H2: SHA1 fingerprinting of ordered tool-name sequences.
    """

    def __init__(
        self,
        *,
        max_history: int = 32,
        tool_history: int = 64,
        h1_threshold: float = 0.85,
    ) -> None:
        self._max_history = int(max_history)
        self._tool_history = int(tool_history)
        self._h1_threshold = float(h1_threshold)

        self._input_history: deque[frozenset[str]] = deque(maxlen=self._max_history)
        self._tool_fp_history: deque[str] = deque(maxlen=self._tool_history)
        self._tool_counts: Counter[str] = Counter()

    def score(self, *, input_text: Optional[str], tool_names: Optional[List[str]] = None) -> LoopScores:
        tokens = _normalize_tokens(input_text or "")

        # H1
        h1 = 0.0
        for prev in self._input_history:
            sim = _jaccard(tokens, prev)
            if sim > h1:
                h1 = sim
                if h1 >= 0.999:  # fast path
                    break
        self._input_history.append(tokens)

        # H2
        h2 = 0.0
        if tool_names is not None:
            fp = _tool_fingerprint(tool_names)
            # maintain counts over sliding window
            if len(self._tool_fp_history) == self._tool_fp_history.maxlen:
                old = self._tool_fp_history[0]
                self._tool_counts[old] -= 1
                if self._tool_counts[old] <= 0:
                    del self._tool_counts[old]
            self._tool_fp_history.append(fp)
            self._tool_counts[fp] += 1

            c = self._tool_counts[fp]
            h2 = 1.0 if c >= 3 else (c / 3.0)

        combined = max(h1, h2)
        return LoopScores(h1_jaccard=h1, h2_tools=h2, combined=combined)

    def is_loop(self, scores: LoopScores) -> bool:
        return scores.h1_jaccard > self._h1_threshold or scores.h2_tools >= 1.0
