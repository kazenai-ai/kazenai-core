__version__ = "1.0.1"

from .context import BackgroundTaskContext, RunContext, get_current_context, use_context
from .enforcement import (
    BudgetExceeded,
    BudgetUnavailable,
    Enforcement,
    KazenBlock,
    LoopDetected,
    RateLimitExceeded,
    StreamCutoffError,
    UnsupportedModeError,
)
from .spine import aguarded_llm_call, guarded_embedding_call, guarded_llm_call
from .loop_detector import LoopDetector

# NOTE: this deliberately shadows the `kazenai.monitor` SUBMODULE with the
# `monitor()` function — the SDK's primary public entry point (renaming would
# break consumers like kazenai-finops-sdk). Consequences:
#   - `from kazenai import monitor` / `kazenai.monitor(...)` → the function.
#   - `from kazenai.monitor import X` → still works (resolved via sys.modules).
#   - `import kazenai.monitor as m` → binds the FUNCTION on Python 3.12+;
#     use `importlib.import_module("kazenai.monitor")` to get the module.
from .monitor import monitor, patch_anthropic, patch_openai
from .capture_policy import CaptureMode, inputs_absent, redact_secrets, resolve_capture_mode
from .cost_engine import TokenCostEngine
from .trajectory import CostTrajectoryPredictor
from .circuit_breaker import CircuitBreaker, KazenCircuitBreaker, StateSerializer
from .sinks import EventSink, HttpSink, HttpSinkConfig, JsonlSink, MemorySink, MultiSink
from .alerts import AlertDispatcher, WebhookAlert
from .finops import FinOpsController, FinOpsConfig
from .schema import KazenEvent
from .degraded import clear_degraded, degraded_header_value, degraded_services, mark_degraded
from .deployment import canonical_default_org, require_canonical_default_org
from .memory import KazenMemory
from .validation import (
    InjectionAttemptDetected,
    PromptInjectionDetector,
    ValidatedRequest,
    bounded_list,
    bounded_str,
    enforce_json_bounds,
    get_detector,
    scan_and_raise_or_http,
)
from .reliability_policy import (
    AutonomyTier,
    FailureKind,
    ReliabilityAction,
    ReliabilityContext,
    ReliabilityDecision,
    ReliabilitySurface,
    parse_autonomy_tier_from_headers,
    resolve as resolve_reliability,
    should_block,
    should_escalate,
)

__all__ = [
    "KazenMemory",
    "clear_degraded",
    "degraded_header_value",
    "degraded_services",
    "mark_degraded",
    "AutonomyTier",
    "FailureKind",
    "ReliabilityAction",
    "ReliabilityContext",
    "ReliabilityDecision",
    "ReliabilitySurface",
    "parse_autonomy_tier_from_headers",
    "resolve_reliability",
    "should_block",
    "should_escalate",
    "aguarded_llm_call",
    "BudgetExceeded",
    "BudgetUnavailable",
    "guarded_embedding_call",
    "guarded_llm_call",
    "CircuitBreaker",
    "CostTrajectoryPredictor",
    "Enforcement",
    "KazenBlock",
    "KazenCircuitBreaker",
    "KazenEvent",
    "LoopDetected",
    "LoopDetector",
    "TokenCostEngine",
    "RateLimitExceeded",
    "StreamCutoffError",
    "UnsupportedModeError",
    "RunContext",
    "BackgroundTaskContext",
    "StateSerializer",
    "EventSink",
    "MemorySink",
    "JsonlSink",
    "HttpSink",
    "HttpSinkConfig",
    "MultiSink",
    "AlertDispatcher",
    "WebhookAlert",
    "FinOpsController",
    "FinOpsConfig",
    "get_current_context",
    "CaptureMode",
    "inputs_absent",
    "redact_secrets",
    "resolve_capture_mode",
    "monitor",
    "patch_anthropic",
    "patch_openai",
    "use_context",
    "InjectionAttemptDetected",
    "PromptInjectionDetector",
    "ValidatedRequest",
    "bounded_list",
    "bounded_str",
    "enforce_json_bounds",
    "get_detector",
    "scan_and_raise_or_http",
    "canonical_default_org",
    "require_canonical_default_org",
]
