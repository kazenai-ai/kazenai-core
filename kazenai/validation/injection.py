"""Prompt-injection detection — canonical, shared across KazenAI services.

This is the single source of truth for injection patterns. Services that
previously had their own detector (orchestrator ``security/injection_detector``)
or only ad-hoc regex filters (brain ``modules/content_safety``) should converge
on this module.

The core API (``scan`` / ``scan_and_raise``) is framework-agnostic. The
``scan_and_raise_or_http`` helper lazily imports FastAPI so that importing this
module never requires a web framework.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class InjectionResult:
    detected: bool
    confidence: float          # 0.0 = definitely clean, 1.0 = definitely injection
    pattern_name: str = ""     # which pattern matched
    matched_text: str = ""     # the matched substring (redacted if sensitive)
    recommendation: str = ""   # what action to take


@dataclass
class InjectionPattern:
    name: str
    regex: "re.Pattern[str]"
    confidence: float
    description: str


class InjectionAttemptDetected(RuntimeError):
    """Raised when a prompt injection attempt is detected with high confidence."""

    def __init__(self, message: str, result: InjectionResult):
        super().__init__(message)
        self.result = result


def _compile(pattern: str, flags: int = re.IGNORECASE | re.DOTALL) -> "re.Pattern[str]":
    return re.compile(pattern, flags)


_PATTERNS: List[InjectionPattern] = [
    InjectionPattern(
        name="ignore_previous_instructions",
        regex=_compile(r"ignore\s+(all\s+)?(previous|prior|above|earlier|your)\s+(instructions?|directives?|rules?|prompts?)"),
        confidence=0.95,
        description="Classic prompt injection: override prior instructions",
    ),
    InjectionPattern(
        name="disregard_above",
        regex=_compile(r"(disregard|forget|override|ignore)\s+(everything|all|the)\s+(above|previous|prior|earlier)"),
        confidence=0.95,
        description="Override previous context",
    ),
    InjectionPattern(
        name="you_are_now",
        regex=_compile(r"\byou\s+are\s+now\b.{0,50}(AI|assistant|agent|bot|model|GPT|Claude|LLM|developer|admin|god)"),
        confidence=0.90,
        description="Persona hijack: 'you are now X'",
    ),
    InjectionPattern(
        name="override_system_prompt",
        regex=_compile(r"(override|bypass|ignore|replace|modify|change)\s+(the\s+)?(system\s+)?(prompt|instruction|directive|constraint|guard|safety)"),
        confidence=0.90,
        description="Explicit system prompt override",
    ),
    InjectionPattern(
        name="forget_guidelines",
        regex=_compile(r"(forget|ignore|bypass|discard)\s+(all\s+)?(your\s+)?(guidelines?|safety|ethics?|rules?|restrictions?|constraints?)"),
        confidence=0.88,
        description="Safety guideline bypass",
    ),
    InjectionPattern(
        name="print_system_prompt",
        regex=_compile(r"(print|output|reveal|show|display|repeat|return|echo)\s+(your\s+)?(system|original|full|complete)?\s*(prompt|instructions?|context|directives?)"),
        confidence=0.85,
        description="System prompt exfiltration attempt",
    ),
    InjectionPattern(
        name="repeat_verbatim",
        regex=_compile(r"repeat\s+(everything|all|the\s+following|verbatim|exactly|word\s+for\s+word)\s+(above|before|previously|in\s+your\s+context)"),
        confidence=0.85,
        description="Context exfiltration via verbatim repeat",
    ),
    InjectionPattern(
        name="dan_jailbreak",
        regex=_compile(r"\bDAN\b|\bDo\s+Anything\s+Now\b|\bjailbreak\b"),
        confidence=0.80,
        description="Known DAN/jailbreak keywords",
    ),
    InjectionPattern(
        name="act_as_unconstrained",
        regex=_compile(r"\bact\s+as\s+(an?\s+)?(unconstrained|unrestricted|unfiltered|jailbroken|DAN|evil|malicious)"),
        confidence=0.80,
        description="'Act as unconstrained AI' pattern",
    ),
    InjectionPattern(
        name="new_instructions_follow",
        regex=_compile(r"(new|updated?|actual|real|following)\s+instructions?\s*(follow|start|begin|are)"),
        confidence=0.75,
        description="Hidden instruction injection via 'new instructions follow'",
    ),
    InjectionPattern(
        name="hidden_text_delimiter",
        regex=_compile(r"----+\s*(SYSTEM|HIDDEN|SECRET|REAL)\s*----+"),
        confidence=0.70,
        description="Hidden delimiter injection pattern",
    ),
]

# Block threshold — text is rejected when match confidence exceeds this.
BLOCK_THRESHOLD = float(os.getenv("KAZENAI_INJECTION_BLOCK_THRESHOLD", "0.80"))


class PromptInjectionDetector:
    """Scan text for prompt injection patterns."""

    def __init__(self, patterns: Optional[List[InjectionPattern]] = None) -> None:
        self._patterns = patterns or _PATTERNS

    def scan(self, text: str) -> InjectionResult:
        """Return the highest-confidence injection match in *text* (or a clean result)."""
        if not text or not text.strip():
            return InjectionResult(detected=False, confidence=0.0)

        best: Optional[InjectionResult] = None
        for pattern in self._patterns:
            match = pattern.regex.search(text)
            if match:
                redacted = match.group(0)[:80].replace("\n", " ")
                result = InjectionResult(
                    detected=True,
                    confidence=pattern.confidence,
                    pattern_name=pattern.name,
                    matched_text=redacted,
                    recommendation=(
                        "Reject input — prompt injection attempt detected. "
                        f"Pattern: {pattern.name} ({pattern.description})"
                    ),
                )
                if best is None or result.confidence > best.confidence:
                    best = result
        return best or InjectionResult(detected=False, confidence=0.0)

    def scan_and_raise(self, text: str, block_threshold: float = BLOCK_THRESHOLD) -> None:
        """Raise :class:`InjectionAttemptDetected` if confidence >= *block_threshold*."""
        result = self.scan(text)
        if result.detected and result.confidence >= block_threshold:
            raise InjectionAttemptDetected(
                f"Prompt injection detected (confidence={result.confidence:.2f}, "
                f"pattern={result.pattern_name}): {result.matched_text[:60]}",
                result,
            )


_detector: Optional[PromptInjectionDetector] = None


def get_detector() -> PromptInjectionDetector:
    global _detector
    if _detector is None:
        _detector = PromptInjectionDetector()
    return _detector


def scan_and_raise_or_http(text: str, *, field: str = "prompt") -> None:
    """FastAPI-aware scan: raise HTTP 422 on detection, fail closed in production.

    FastAPI is imported lazily so this module stays framework-agnostic.
    """
    from fastapi import HTTPException

    try:
        get_detector().scan_and_raise(text or "")
    except InjectionAttemptDetected as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "InjectionAttemptDetected",
                "message": str(exc)[:200],
                "field": field,
            },
        ) from exc
