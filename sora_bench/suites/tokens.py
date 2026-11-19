"""Token-usage suite."""

from __future__ import annotations

import time
from typing import Any

from ..config import TARGETS
from ..harness import BenchmarkHarness
from ..scenarios import scenarios_for_suite
from ..scoring import (
    mean,
    safe_div,
    score_higher_is_better,
    score_lower_is_better,
    weighted_score,
)
from ..types import SuiteResult, metric


def _prompt(run: Any) -> int:
    return int(run.reported_prompt_tokens or run.prompt_tokens)


def _completion(run: Any) -> int:
    return int(run.reported_completion_tokens or run.completion_tokens)


def _cached(run: Any) -> int:
    if any(request.cache_usage_source for request in run.requests):
        return int(run.reported_cached_tokens)
    return int(run.cache_hit_tokens)


def _total(run: Any) -> int:
    return _prompt(run) + _completion(run)


def _effective(run: Any) -> int:
    return max(0, _prompt(run) - _cached(run)) + _completion(run)


def run(harness: BenchmarkHarness) -> SuiteResult:
    started = time.perf_counter()
    result = SuiteResult(name="tokens")
    runs = []
    for scenario in scenarios_for_suite("tokens"):
        for run_index in range(harness.config.runs):
            artifacts = harness.run_task(scenario, run_index=run_index)
            runs.append(artifacts.run)
            artifacts.cleanup(keep=harness.config.keep_workdir)
    result.runs = runs
    if not runs:
        result.error = "no token runs"
        return result

    successful = [run for run in runs if run.passed]
    total_tokens = sum(_total(run) for run in runs)
    effective_tokens = sum(_effective(run) for run in runs)
    prompt_tokens = sum(_prompt(run) for run in runs)
    completion_tokens = sum(_completion(run) for run in runs)
    tool_schema = sum(run.tool_schema_tokens for run in runs)
    system = sum(run.system_tokens for run in runs)
    identity = sum(run.identity_tokens for run in runs)
    clock = sum(run.clock_tokens for run in runs)
    overhead = tool_schema + system + identity + clock
    successful_count = max(1, len(successful))
    per_success = total_tokens / float(successful_count)
    efficiency = len(successful) / (total_tokens / 1000.0) if total_tokens else 0.0
    overhead_share = safe_div(overhead, prompt_tokens)
    tool_schema_share = safe_div(tool_schema, prompt_tokens)
    cache_savings = safe_div(sum(_cached(run) for run in runs), prompt_tokens)
    reported_available = any(run.reported_prompt_tokens for run in runs)

    score = weighted_score(
        [
            (
                score_higher_is_better(efficiency, TARGETS["token_efficiency_tasks_per_1k"]),
                0.35,
            ),
            (score_lower_is_better(per_success, TARGETS["token_per_success"]), 0.25),
            (score_lower_is_better(overhead_share, 0.55), 0.20),
            (score_higher_is_better(cache_savings, 0.35), 0.20),
        ]
    )
    result.score = score
    result.metrics = [
        metric(
            "token_prompt_mean",
            "Prompt tokens per run",
            mean(_prompt(run) for run in runs),
            unit="tokens",
            higher_is_better=False,
            score=score_lower_is_better(
                mean(_prompt(run) for run in runs), TARGETS["token_prompt_mean"]
            ),
            n=len(runs),
        ),
        metric(
            "token_completion_mean",
            "Completion tokens per run",
            mean(_completion(run) for run in runs),
            unit="tokens",
            higher_is_better=False,
            score=score_lower_is_better(mean(_completion(run) for run in runs), 500.0),
            n=len(runs),
        ),
        metric(
            "token_total_mean",
            "Total tokens per run",
            mean(_total(run) for run in runs),
            unit="tokens",
            higher_is_better=False,
            score=score_lower_is_better(
                mean(_total(run) for run in runs), TARGETS["token_total_mean"]
            ),
            n=len(runs),
        ),
        metric(
            "token_effective_mean",
            "Effective tokens per run (after prefix reuse)",
            effective_tokens / float(len(runs)),
            unit="tokens",
            higher_is_better=False,
            score=score_lower_is_better(effective_tokens / float(len(runs)), 4000.0),
            n=len(runs),
        ),
        metric(
            "token_per_successful_task",
            "Tokens per successful task",
            per_success,
            unit="tokens",
            higher_is_better=False,
            score=score_lower_is_better(per_success, TARGETS["token_per_success"]),
            n=len(successful),
        ),
        metric(
            "token_efficiency",
            "Successful tasks per 1k tokens",
            efficiency,
            unit="tasks/1k",
            score=score_higher_is_better(efficiency, TARGETS["token_efficiency_tasks_per_1k"]),
            n=len(runs),
        ),
        metric(
            "token_system_overhead_share",
            "Static prompt overhead share",
            overhead_share,
            unit="ratio",
            higher_is_better=False,
            score=score_lower_is_better(overhead_share, 0.55),
            n=len(runs),
        ),
        metric(
            "token_tool_schema_share",
            "Tool-schema token share",
            tool_schema_share,
            unit="ratio",
            higher_is_better=False,
            score=score_lower_is_better(tool_schema_share, 0.40),
            n=len(runs),
        ),
        metric(
            "token_cache_savings",
            "Prompt tokens reused by prefix cache",
            cache_savings,
            unit="ratio",
            score=score_higher_is_better(cache_savings, 0.35),
            n=len(runs),
        ),
        metric(
            "token_reported_usage_available",
            "Provider-reported usage available",
            1.0 if reported_available else 0.0,
            unit="bool",
            n=len(runs),
        ),
    ]
    result.diagnostics = {
        "scenarios": [
            {
                "scenario_id": run.scenario_id,
                "run_index": run.run_index,
                "prompt_tokens": _prompt(run),
                "completion_tokens": _completion(run),
                "total_tokens": _total(run),
                "cache_hit_tokens": _cached(run),
                "effective_tokens": _effective(run),
                "estimated_prompt_tokens": run.prompt_tokens,
                "reported_prompt_tokens": run.reported_prompt_tokens,
                "system_tokens": run.system_tokens,
                "identity_tokens": run.identity_tokens,
                "clock_tokens": run.clock_tokens,
                "tool_schema_tokens": run.tool_schema_tokens,
                "conversation_tokens": run.conversation_tokens,
                "steps": run.steps,
                "passed": run.passed,
            }
            for run in runs
        ],
        "breakdown_totals": {
            "system": system,
            "identity": identity,
            "clock": clock,
            "tool_schema": tool_schema,
            "conversation": sum(run.conversation_tokens for run in runs),
            "completion": completion_tokens,
        },
    }
    result.duration_ms = (time.perf_counter() - started) * 1000.0
    return result
