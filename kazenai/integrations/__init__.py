"""Framework and gateway integrations for KazenAI.

Ordinary ``import kazenai`` does not require LangChain/LangGraph/CrewAI/LiteLLM.
Install optional extras: ``kazenai[langchain]``, ``[langgraph]``, ``[crewai]``,
``[frameworks]``, ``[otel]``.

LiteLLM Control helpers: ``kazenai.integrations.litellm_control`` (no LiteLLM import).
Gateway CustomLogger: ``kazenai-examples/litellm-otel-control`` (gateway process only).

Shared contract: ``kazenai.integrations.adapter_contract``.
"""

from __future__ import annotations

from .adapter_contract import (
    CapabilityLevel,
    FrameworkOperation,
    UnsupportedFrameworkPath,
    correlation_envelope,
    enter_operation,
    new_operation,
)

__all__ = [
    "CapabilityLevel",
    "FrameworkOperation",
    "UnsupportedFrameworkPath",
    "correlation_envelope",
    "enter_operation",
    "new_operation",
]
