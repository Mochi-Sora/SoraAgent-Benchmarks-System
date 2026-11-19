"""Response-speed suite."""

from __future__ import annotations

import time

from ..config import TARGETS
from ..harness import BenchmarkHarness
from ..scenarios import scenarios_for_suite
from ..scoring import (
    mean,
    p95,
    safe_div,
    score_higher_is_better,
    score_lower_is_better,
    weighted_score,
)
from ..types import SuiteResult, metric


def run(harness: BenchmarkHarness) -> SuiteResult:
    started = time.perf_counter()
    result = SuiteResult(name="speed")
    runs = []
    # In simulated mode a small fixed "model" delay keeps end-to-end timings
    # realistic; the framework overhead is still isolated below.
    latency = harness.config.simulated_latency_ms
    if harness.config.simulated and latency <= 0:
        latency = 25.0
    for scenario in scenarios_for_suite("speed"):
        for run_index in range(harness.config.runs):
            artifacts = harness.run_task(
                scenario,
                run_index=run_index,
                latency_ms=latency,
            )
            runs.append(artifacts.run)
            artifacts.cleanup(keep=harness.config.keep_workdir)
    result.runs = runs
    if not runs:
        result.error = "no speed runs"
        return result

    wall = mean(run.wall_ms for run in runs)
    wall_p95 = p95([run.wall_ms for run in runs])
    provider = mean(run.provider_ms for run in runs)
    tool = mean(run.tool_ms for run in runs)
    overhead = mean(run.overhead_ms for run in runs)
    overhead_p95 = p95([run.overhead_ms for run in runs])
    overhead_share = mean(safe_div(run.overhead_ms, max(1.0, run.wall_ms)) for run in runs)
    step_latency = mean(safe_div(run.wall_ms, max(1, run.steps)) for run in runs)
    call_latency = mean(
        safe_div(
            sum(request.latency_ms for request in run.requests),
            max(1, len(run.requests)),
        )
        for run in runs
    )
    completion_per_second = mean(
        safe_div(
            run.reported_completion_tokens or run.completion_tokens,
            max(0.001, run.provider_ms / 1000.0),
        )
        for run in runs
    )
    tasks_per_second = mean(1.0 / max(0.001, run.wall_ms / 1000.0) for run in runs)
    score = weighted_score(
        [
            (
                score_lower_is_better(wall_p95, TARGETS["response_p95_ms"]),
                0.45,
            ),
            (
                score_lower_is_better(overhead_p95, TARGETS["overhead_ms"]),
                0.35,
            ),
            (score_higher_is_better(completion_per_second, 50.0), 0.20),
        ]
    )
    result.score = score
    result.metrics = [
        metric(
            "speed_e2e_mean",
            "End-to-end latency (mean)",
            wall,
            unit="ms",
            higher_is_better=False,
            score=score_lower_is_better(wall, TARGETS["response_mean_ms"]),
            n=len(runs),
        ),
        metric(
            "speed_e2e_p95",
            "End-to-end latency (p95)",
            wall_p95,
            unit="ms",
            higher_is_better=False,
            score=score_lower_is_better(wall_p95, TARGETS["response_p95_ms"]),
            n=len(runs),
        ),
        metric(
            "speed_provider_mean",
            "Provider time (mean)",
            provider,
            unit="ms",
            higher_is_better=False,
            score=score_lower_is_better(provider, TARGETS["response_mean_ms"]),
            n=len(runs),
        ),
        metric(
            "speed_tool_mean",
            "Tool time (mean)",
            tool,
            unit="ms",
            higher_is_better=False,
            score=score_lower_is_better(tool, 250.0),
            n=len(runs),
        ),
        metric(
            "speed_framework_overhead_mean",
            "Framework overhead (mean)",
            overhead,
            unit="ms",
            higher_is_better=False,
            score=score_lower_is_better(overhead, TARGETS["overhead_ms"]),
            n=len(runs),
        ),
        metric(
            "speed_framework_overhead_p95",
            "Framework overhead (p95)",
            overhead_p95,
            unit="ms",
            higher_is_better=False,
            score=score_lower_is_better(overhead_p95, TARGETS["overhead_ms"]),
            n=len(runs),
        ),
        metric(
            "speed_overhead_share",
            "Framework overhead share",
            overhead_share,
            unit="ratio",
            higher_is_better=False,
            score=score_lower_is_better(overhead_share, 0.25),
            n=len(runs),
        ),
        metric(
            "speed_step_latency",
            "Latency per step",
            step_latency,
            unit="ms/step",
            higher_is_better=False,
            score=score_lower_is_better(step_latency, 800.0),
            n=len(runs),
        ),
        metric(
            "speed_provider_call_latency",
            "Latency per provider call",
            call_latency,
            unit="ms/call",
            higher_is_better=False,
            score=score_lower_is_better(call_latency, 750.0),
            n=len(runs),
        ),
        metric(
            "speed_completion_tokens_per_second",
            "Completion tokens per second",
            completion_per_second,
            unit="tok/s",
            score=score_higher_is_better(completion_per_second, 50.0),
            n=len(runs),
        ),
        metric(
            "speed_tasks_per_second",
            "Tasks per second",
            tasks_per_second,
            unit="tasks/s",
            score=score_higher_is_better(tasks_per_second, 2.0),
            n=len(runs),
        ),
    ]
    result.diagnostics = {
        "simulated_provider_delay_ms": latency,
        "runs": [
            {
                "scenario_id": run.scenario_id,
                "run_index": run.run_index,
                "wall_ms": round(run.wall_ms, 3),
                "provider_ms": round(run.provider_ms, 3),
                "tool_ms": round(run.tool_ms, 3),
                "overhead_ms": round(run.overhead_ms, 3),
                "steps": run.steps,
                "provider_calls": run.provider_calls,
                "completion_tokens": run.reported_completion_tokens or run.completion_tokens,
                "passed": run.passed,
            }
            for run in runs
        ],
    }
    result.duration_ms = (time.perf_counter() - started) * 1000.0
    return result
