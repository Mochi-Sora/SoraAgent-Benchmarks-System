"""Prompt-caching rate and stability suite.

The benchmark snapshots the exact request prefix SoraAgent sends, then measures
how much of each request could be served by a prefix cache and how stable that
prefix is across turns and repeated runs.  A live provider can additionally
report vendor cache counters (``cached_tokens`` / ``prompt_cache_hit_tokens``);
the predicted and reported rates are both shown.

The ablation table isolates the usual cache breakers: the per-turn clock, the
identity section, and memory recall.  The tool catalog is deliberately included
in the request prefix because ``find_tools`` embeds a catalog that shrinks as
tools are loaded.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from ..config import TARGETS
from ..harness import BenchmarkHarness
from ..protocol import aggregate_run_cache
from ..scenarios import scenarios_for_suite
from ..scoring import (
    mean,
    score_higher_is_better,
    weighted_score,
)
from ..types import SuiteResult, metric

_VARIANTS = [
    ("full", {}),
    ("no_clock", {"clock": "off"}),
    ("fixed_clock", {"clock": "fixed"}),
    ("no_identity", {"identity": False}),
    ("no_memory", {"memory": "off"}),
    ("minimal", {"memory": "off", "identity": False, "clock": "off"}),
]


def run(harness: BenchmarkHarness) -> SuiteResult:
    started = time.perf_counter()
    result = SuiteResult(name="caching")
    scenarios = scenarios_for_suite("caching")
    simulate_ablation = bool(harness.config.simulated and harness.config.cache_ablation)
    variants = _VARIANTS if simulate_ablation else [("full", {})]
    repetitions = max(2, harness.config.runs) if simulate_ablation else harness.config.runs
    runs_by_variant: Dict[str, List[Any]] = {}
    all_runs: List[Any] = []
    for scenario in scenarios:
        for variant, overrides in variants:
            for repetition in range(repetitions):
                artifacts = harness.run_task(
                    scenario,
                    run_index=repetition,
                    variant=variant,
                    identity=overrides.get("identity"),
                    clock=overrides.get("clock"),
                    memory=overrides.get("memory"),
                )
                runs_by_variant.setdefault(variant, []).append(artifacts.run)
                all_runs.append(artifacts.run)
                artifacts.cleanup(keep=harness.config.keep_workdir)
    result.runs = all_runs
    full_runs = runs_by_variant.get("full", [])
    if not full_runs:
        result.error = "no full caching runs"
        return result

    aggregate = aggregate_run_cache(full_runs, min_cache_block=harness.config.min_cache_block)
    variant_stats = {
        variant: aggregate_run_cache(runs, min_cache_block=harness.config.min_cache_block)
        for variant, runs in runs_by_variant.items()
    }
    hit_rate = float(aggregate.get("cache_hit_rate", 0.0))
    message_hit_rate = float(aggregate.get("message_cache_hit_rate", 0.0))
    message_stability = float(aggregate.get("message_stability_mean", 0.0))
    stability = float(aggregate.get("cache_stability_mean", 0.0))
    clock_trailing_rate = float(aggregate.get("clock_trailing_rate_mean", 0.0))
    stable_prefix = mean(
        item.get("stable_prefix_tokens", 0) for item in aggregate.get("per_run", [])
    )
    cacheable = mean(
        item.get("cacheable_request_rate", 0.0) for item in aggregate.get("per_run", [])
    )
    retention_mean = mean(
        item.get("prefix_retention_mean", 0.0) for item in aggregate.get("per_run", [])
    )
    retention_min = min(
        (item.get("prefix_retention_min", 0.0) for item in aggregate.get("per_run", [])),
        default=0.0,
    )
    tool_churn = mean(
        item.get("tool_schema_churn_rate", 0.0) for item in aggregate.get("per_run", [])
    )
    clock_churn = mean(item.get("clock_churn_rate", 0.0) for item in aggregate.get("per_run", []))
    identity_churn = mean(
        item.get("identity_churn_rate", 0.0) for item in aggregate.get("per_run", [])
    )
    break_events = mean(item.get("break_events", 0) for item in aggregate.get("per_run", []))
    reported_available = bool(aggregate.get("reported_cache_available"))
    reported_rate = float(aggregate.get("reported_cache_hit_rate", 0.0))
    reported_error = abs(reported_rate - hit_rate) if reported_available else 0.0
    # The provider's own counters are the ground truth when available.  The
    # tools-first prefix model is deliberately conservative and undercounts
    # vendors that cache in a different order.
    effective_rate = reported_rate if reported_available else hit_rate
    rate_score = score_higher_is_better(effective_rate, TARGETS["cache_hit_rate"])
    predicted_score = score_higher_is_better(hit_rate, TARGETS["cache_hit_rate"])
    stability_score = stability * 100.0
    message_stability_score = message_stability * 100.0
    message_hit_score = score_higher_is_better(message_hit_rate, 0.70)
    prefix_score = score_higher_is_better(stable_prefix, TARGETS["cache_stable_prefix_tokens"])
    score = weighted_score(
        [
            (rate_score, 0.40),
            (message_stability_score, 0.20),
            (stability_score, 0.20),
            (prefix_score, 0.10),
            (message_hit_score, 0.10),
        ]
    )
    result.score = score
    result.metrics = [
        metric(
            "cache_hit_rate",
            "Prompt prefix cache hit rate",
            hit_rate,
            unit="ratio",
            score=predicted_score,
            detail=(
                "Predicted share of prompt tokens reusable as a byte-for-byte "
                "prefix; this is conservative when tools precede messages."
            ),
            n=len(full_runs),
        ),
        metric(
            "cache_effective_hit_rate",
            "Effective prompt cache hit rate",
            effective_rate,
            unit="ratio",
            score=rate_score,
            detail=(
                "Provider-reported cache hit rate when available, otherwise "
                "the predicted prefix hit rate."
            ),
            n=len(full_runs),
        ),
        metric(
            "cache_message_hit_rate",
            "Message prefix cache hit rate",
            message_hit_rate,
            unit="ratio",
            score=score_higher_is_better(message_hit_rate, 0.70),
            detail=(
                "Prompt-cache reuse after excluding tool schemas; isolates "
                "system-prompt/conversation ordering from tool-catalog churn."
            ),
            n=len(full_runs),
        ),
        metric(
            "cache_message_stability",
            "Message prefix stability",
            message_stability,
            unit="ratio",
            score=score_higher_is_better(message_stability, 0.70),
            detail=(
                "Prefix retention for system prompt and conversation only, excluding tool schemas."
            ),
            n=len(full_runs),
        ),
        metric(
            "cache_stability",
            "Prompt cache stability",
            stability,
            unit="ratio",
            score=stability_score,
            detail="Mean prefix retention penalized by its variation across turns.",
            n=len(full_runs),
        ),
        metric(
            "cache_stable_prefix_tokens",
            "Stable prefix tokens",
            stable_prefix,
            unit="tokens",
            score=prefix_score,
            n=len(full_runs),
        ),
        metric(
            "cache_cacheable_request_rate",
            "Cacheable request rate",
            cacheable,
            unit="ratio",
            score=score_higher_is_better(cacheable, 0.75),
            n=len(full_runs),
        ),
        metric(
            "cache_prefix_retention_mean",
            "Mean prefix retention",
            retention_mean,
            unit="ratio",
            score=score_higher_is_better(retention_mean, 0.60),
            n=len(full_runs),
        ),
        metric(
            "cache_prefix_retention_min",
            "Worst prefix retention",
            retention_min,
            unit="ratio",
            score=score_higher_is_better(retention_min, 0.25),
            n=len(full_runs),
        ),
        metric(
            "cache_tool_schema_churn",
            "Tool-schema churn per transition",
            tool_churn,
            unit="ratio",
            higher_is_better=False,
            score=(1.0 - min(1.0, tool_churn)) * 100.0,
            n=len(full_runs),
        ),
        metric(
            "cache_clock_churn",
            "Clock churn per transition",
            clock_churn,
            unit="ratio",
            higher_is_better=False,
            score=None,
            detail=(
                "Clock content changes are expected; what matters is whether "
                "the clock is trailing. See trailing rate."
            ),
            n=len(full_runs),
        ),
        metric(
            "cache_clock_trailing_rate",
            "Clock in trailing position",
            clock_trailing_rate,
            unit="ratio",
            score=score_higher_is_better(clock_trailing_rate, 1.0),
            detail=(
                "Fraction of clock-bearing requests where ## CURRENT TIME is "
                "the final message, after the conversation."
            ),
            n=len(full_runs),
        ),
        metric(
            "cache_identity_churn",
            "Identity churn per transition",
            identity_churn,
            unit="ratio",
            higher_is_better=False,
            score=(1.0 - min(1.0, identity_churn)) * 100.0,
            n=len(full_runs),
        ),
        metric(
            "cache_break_events_per_run",
            "Prefix breaks per run",
            break_events,
            unit="breaks",
            higher_is_better=False,
            score=max(0.0, 100.0 - 20.0 * break_events),
            n=len(full_runs),
        ),
        metric(
            "cache_cross_run_prefix_stable",
            "First prompt identical across runs",
            1.0 if aggregate.get("cross_run_prompt_prefix_identical") else 0.0,
            unit="bool",
            score=100.0 if aggregate.get("cross_run_prompt_prefix_identical") else 0.0,
            n=len(full_runs),
        ),
        metric(
            "cache_hit_rate_stdev",
            "Cache hit-rate variation across runs",
            aggregate.get("cache_hit_rate_stdev", 0.0),
            unit="ratio",
            higher_is_better=False,
            score=(1.0 - min(1.0, aggregate.get("cache_hit_rate_stdev", 0.0))) * 100.0,
            n=len(full_runs),
        ),
        metric(
            "cache_cached_tokens",
            "Reused prompt tokens",
            aggregate.get("cached_tokens", 0),
            unit="tokens",
            score=None,
            n=len(full_runs),
        ),
        metric(
            "cache_prompt_tokens",
            "Total prompt tokens",
            aggregate.get("prompt_tokens", 0),
            unit="tokens",
            higher_is_better=False,
            score=None,
            n=len(full_runs),
        ),
        metric(
            "cache_reported_hit_rate",
            "Provider-reported cache hit rate",
            reported_rate if reported_available else 0.0,
            unit="ratio",
            score=score_higher_is_better(reported_rate, TARGETS["cache_hit_rate"])
            if reported_available
            else None,
            detail="" if reported_available else "Provider did not report cache counters.",
            n=len(full_runs),
        ),
        metric(
            "cache_reported_available",
            "Provider-reported cache counters available",
            1.0 if reported_available else 0.0,
            unit="bool",
            n=len(full_runs),
        ),
        metric(
            "cache_prediction_error",
            "Predicted vs reported cache-rate error",
            reported_error,
            unit="ratio",
            higher_is_better=False,
            score=(1.0 - min(1.0, reported_error)) * 100.0 if reported_available else None,
            n=len(full_runs),
        ),
    ]
    result.diagnostics = {
        "variants": {
            variant: {
                "run_count": stats.get("run_count", 0),
                "request_count": stats.get("request_count", 0),
                "cache_hit_rate": round(stats.get("cache_hit_rate", 0.0), 4),
                "message_cache_hit_rate": round(stats.get("message_cache_hit_rate", 0.0), 4),
                "message_stability": round(stats.get("message_stability_mean", 0.0), 4),
                "cache_stability": round(stats.get("cache_stability_mean", 0.0), 4),
                "clock_trailing_rate": round(stats.get("clock_trailing_rate_mean", 0.0), 4),
                "cached_tokens": stats.get("cached_tokens", 0),
                "prompt_tokens": stats.get("prompt_tokens", 0),
                "cross_run_prefix_identical": bool(stats.get("cross_run_prompt_prefix_identical")),
                "divergence_roles": stats.get("divergence_roles"),
                "per_run": [
                    {
                        "cache_hit_rate": round(item.get("cache_hit_rate", 0.0), 4),
                        "cache_stability": round(item.get("cache_stability", 0.0), 4),
                        "stable_prefix_tokens": item.get("stable_prefix_tokens", 0),
                        "clock_churn_rate": round(item.get("clock_churn_rate", 0.0), 4),
                        "tool_schema_churn_rate": round(item.get("tool_schema_churn_rate", 0.0), 4),
                        "break_events": item.get("break_events", 0),
                        "divergence_roles": item.get("divergence_roles"),
                    }
                    for item in stats.get("per_run", [])
                ],
            }
            for variant, stats in variant_stats.items()
        },
        "ablation_delta_vs_full": {
            variant: {
                "cache_hit_rate_delta": round(stats.get("cache_hit_rate", 0.0) - hit_rate, 4),
                "cache_stability_delta": round(
                    stats.get("cache_stability_mean", 0.0) - stability, 4
                ),
                "cached_tokens_delta": stats.get("cached_tokens", 0)
                - aggregate.get("cached_tokens", 0),
            }
            for variant, stats in variant_stats.items()
            if variant != "full"
        },
        "cache_breakers": {
            "clock_churn": round(clock_churn, 4),
            "clock_trailing_rate": round(clock_trailing_rate, 4),
            "tool_schema_churn": round(tool_churn, 4),
            "recommendation": _recommendation(clock_churn, tool_churn, clock_trailing_rate),
        },
        "mode": "simulated" if harness.config.simulated else "live",
        "cache_ablation_enabled": simulate_ablation,
    }
    result.duration_ms = (time.perf_counter() - started) * 1000.0
    return result


def _recommendation(clock_churn: float, tool_churn: float, clock_trailing_rate: float) -> str:
    messages = []
    if clock_churn >= 0.5 and clock_trailing_rate < 0.9:
        messages.append(
            "clock churn is high and the clock is not trailing; append the "
            "per-turn clock after the conversation history"
        )
    elif clock_churn >= 0.5:
        messages.append(
            "the clock changes every turn but is trailing, so it no longer "
            "invalidates the cached prefix"
        )
    if tool_churn >= 0.5:
        messages.append(
            "tool-schema churn remains high; the find_tools catalog changes on "
            "every load, so the tool definition prefix is rebuilt"
        )
    if not messages:
        return "prefix is stable; no dominant cache breaker detected"
    return "; ".join(messages)
