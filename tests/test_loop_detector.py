"""LoopDetector heuristic tests."""

from __future__ import annotations

from kazenai.loop_detector import LoopDetector, LoopScores, _jaccard, _normalize_tokens, _tool_fingerprint


def test_normalize_tokens_empty():
    assert _normalize_tokens("") == frozenset()


def test_jaccard_identical():
    a = frozenset({"hello", "world"})
    assert _jaccard(a, a) == 1.0


def test_tool_fingerprint_stable():
    assert _tool_fingerprint(["Search", "search"]) == _tool_fingerprint(["search", "Search"])


def test_loop_detector_high_jaccard():
    det = LoopDetector()
    text = "fix the login bug in authentication module"
    det.score(input_text=text)
    scores = det.score(input_text=text)
    assert scores.h1_jaccard >= 0.85
    assert det.is_loop(scores) is True


def test_loop_detector_tool_repeat():
    det = LoopDetector()
    for _ in range(3):
        det.score(input_text="step", tool_names=["grep", "edit"])
    scores = det.score(input_text="next", tool_names=["grep", "edit"])
    assert scores.h2_tools >= 1.0
    assert det.is_loop(scores) is True


def test_loop_scores_dataclass():
    s = LoopScores(h1_jaccard=0.1, h2_tools=0.2, combined=0.2)
    assert s.combined == 0.2
