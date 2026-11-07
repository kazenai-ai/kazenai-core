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


def test_each_subject_has_its_own_history():
    det = LoopDetector()
    det.score(input_text="same words here", subject="cust_a")
    other = det.score(input_text="same words here", subject="cust_b")
    again = det.score(input_text="same words here", subject="cust_a")
    assert det.is_loop(other) is False
    assert det.is_loop(again) is True


def test_calls_without_a_subject_share_one_history():
    det = LoopDetector()
    det.score(input_text="same words here")
    assert det.is_loop(det.score(input_text="same words here", subject="  ")) is True


def test_tool_repeats_are_counted_per_subject():
    det = LoopDetector()
    for subject in ("a", "b", "a", "b"):
        scores = det.score(input_text=subject * 3, tool_names=["grep"], subject=subject)
    assert scores.h2_tools < 1.0
    assert det.is_loop(det.score(input_text="zzz", tool_names=["grep"], subject="a")) is True


def test_least_recently_active_subject_is_dropped_first():
    det = LoopDetector(max_subjects=2)
    det.score(input_text="alpha words", subject="a")
    det.score(input_text="bravo words", subject="b")
    det.score(input_text="alpha words", subject="a")  # a is now the most recent
    det.score(input_text="charlie words", subject="c")  # drops b
    assert det.is_loop(det.score(input_text="bravo words", subject="b")) is False
    assert det.is_loop(det.score(input_text="charlie words", subject="c")) is True


def test_max_subjects_zero_keeps_one_shared_history():
    det = LoopDetector(max_subjects=0)
    det.score(input_text="same words here", subject="cust_a")
    assert det.is_loop(det.score(input_text="same words here", subject="cust_b")) is True
