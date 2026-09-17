from __future__ import annotations

import argparse
import statistics
import time

from kazenai.context import RunContext, use_context
from kazenai.enforcement import Enforcement
from kazenai.loop_detector import LoopDetector


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = int(round((p / 100.0) * (len(values_sorted) - 1)))
    return values_sorted[max(0, min(k, len(values_sorted) - 1))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    args = ap.parse_args()

    ctx = RunContext.new(org_id="org", project_id="proj", agent_id="agent")
    enforcement = Enforcement(max_cost_usd=10.0, calls_per_minute=1000000)
    loop_detector = LoopDetector()

    samples_us: list[float] = []
    with use_context(ctx):
        for i in range(args.iters):
            step_ctx = ctx.child_step(step_id=f"s{i}")
            t0 = time.perf_counter_ns()
            with use_context(step_ctx):
                loop_detector.score(input_text="Summarize the following text about budgets and loops", tool_names=["toolA"])
                enforcement.check_local(projected_cost_usd=0.0)
                enforcement.record_call(cost_usd=0.0)
            t1 = time.perf_counter_ns()
            samples_us.append((t1 - t0) / 1000.0)

    p50 = _pct(samples_us, 50)
    p90 = _pct(samples_us, 90)
    p99 = _pct(samples_us, 99)
    mean = statistics.fmean(samples_us)

    print(f"iterations: {args.iters}")
    print(f"overhead_us: p50={p50:.2f} p90={p90:.2f} p99={p99:.2f} mean={mean:.2f}")
    print("target: <5000us (<5ms) overhead per call")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

