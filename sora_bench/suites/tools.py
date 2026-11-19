"""Tool-use accuracy suite."""

from __future__ import annotations

from typing import Any, Dict

from ..config import TARGETS
from ..harness import BenchmarkHarness
from ..scenarios import scenarios_for_suite
from ..scoring import (
    mean,
    percentile,
    safe_div,
    score_higher_is_better,
    score_lower_is_better,
    weighted_score,
)
from ..types import SuiteResult, metric


def run(harness: BenchmarkHarness) -> SuiteResult:
    started = __import__("time").perf_counter()
    result = SuiteResult(name="tools")
    runs = []
    for scenario in scenarios_for_suite("tools"):
        for run_index in range(harness.config.runs):
            artifacts = harness.run_task(
                scenario,
                run_index=run_index,
                policy=harness.config.autonomy_policy,
            )
            runs.append(artifacts.run)
            artifacts.cleanup(keep=harness.config.keep_workdir)
    result.runs = runs
    if not runs:
        result.error = "no tool runs"
        return result

    matches = [run.evaluation.get("calls", {}) for run in runs]
    f1 = mean(item.get("f1", 0.0) for item in matches)
    precision = mean(item.get("precision", 0.0) for item in matches)
    recall = mean(item.get("recall", 0.0) for item in matches)
    exact = mean(1.0 if item.get("exact_trajectory") else 0.0 for item in matches)
    forbidden = mean(1.0 if item.get("forbidden_violations") else 0.0 for item in matches)
    unnecessary = mean(item.get("unnecessary_call_rate", 0.0) for item in matches)
    argument_accuracy = mean(_argument_accuracy(item) for item in matches)
    success = mean(1.0 if run.passed else 0.0 for run in runs)
    errors = mean(safe_div(run.tool_errors, max(1, run.tool_calls)) for run in runs)
    approvals_ok = mean(1.0 if run.approval_denials == 0 else 0.0 for run in runs)
    discovery_calls = mean(
        sum(1 for item in run.tool_call_log if item["name"] == "find_tools") for run in runs
    )
    business_calls = mean(
        sum(
            1
            for item in run.tool_call_log
            if item["name"] not in {"find_tools", "memory_recall"} and not item["skipped"]
        )
        for run in runs
    )
    first_tool_correct = mean(1.0 if _first_tool_correct(run) else 0.0 for run in runs)
    memory_first = [run for run in runs if run.metadata.get("memory") not in {None, "off"}]
    memory_first_rate = (
        mean(1.0 if _memory_recall_first(run) else 0.0 for run in runs) if memory_first else 0.0
    )

    score = weighted_score(
        [
            (score_higher_is_better(f1, TARGETS["tool_f1"]), 0.32),
            (score_higher_is_better(exact, TARGETS["tool_exact_rate"]), 0.18),
            (score_higher_is_better(success, 0.95), 0.18),
            (
                score_lower_is_better(forbidden, TARGETS["tool_unnecessary_rate"]),
                0.12,
            ),
            (
                score_lower_is_better(unnecessary, TARGETS["tool_unnecessary_rate"]),
                0.10,
            ),
            (score_higher_is_better(argument_accuracy, 0.95), 0.10),
        ]
    )
    result.score = score
    result.metrics = [
        metric(
            "tool_f1",
            "Tool selection F1",
            f1,
            unit="ratio",
            score=score_higher_is_better(f1, TARGETS["tool_f1"]),
            n=len(runs),
        ),
        metric(
            "tool_precision",
            "Tool selection precision",
            precision,
            unit="ratio",
            score=score_higher_is_better(precision, TARGETS["tool_f1"]),
            n=len(runs),
        ),
        metric(
            "tool_recall",
            "Tool selection recall",
            recall,
            unit="ratio",
            score=score_higher_is_better(recall, TARGETS["tool_f1"]),
            n=len(runs),
        ),
        metric(
            "tool_exact_trajectory",
            "Exact trajectory rate",
            exact,
            unit="ratio",
            score=score_higher_is_better(exact, TARGETS["tool_exact_rate"]),
            n=len(runs),
        ),
        metric(
            "tool_argument_accuracy",
            "Argument accuracy",
            argument_accuracy,
            unit="ratio",
            score=score_higher_is_better(argument_accuracy, 0.95),
            n=len(runs),
        ),
        metric(
            "tool_forbidden_rate",
            "Forbidden tool rate",
            forbidden,
            unit="ratio",
            higher_is_better=False,
            score=score_lower_is_better(forbidden, TARGETS["tool_unnecessary_rate"]),
            n=len(runs),
        ),
        metric(
            "tool_unnecessary_rate",
            "Unnecessary call rate",
            unnecessary,
            unit="ratio",
            higher_is_better=False,
            score=score_lower_is_better(unnecessary, TARGETS["tool_unnecessary_rate"]),
            n=len(runs),
        ),
        metric(
            "tool_task_success",
            "Task success rate",
            success,
            unit="ratio",
            score=score_higher_is_better(success, 0.95),
            n=len(runs),
        ),
        metric(
            "tool_error_rate",
            "Tool error rate",
            errors,
            unit="ratio",
            higher_is_better=False,
            score=score_lower_is_better(errors, 0.05),
            n=len(runs),
        ),
        metric(
            "tool_approval_compliance",
            "Approval compliance",
            approvals_ok,
            unit="ratio",
            score=score_higher_is_better(approvals_ok, 1.0),
            n=len(runs),
        ),
        metric(
            "tool_discovery_calls_per_run",
            "Discovery calls per run",
            discovery_calls,
            unit="calls",
            higher_is_better=False,
            score=score_lower_is_better(discovery_calls, 1.5),
            n=len(runs),
        ),
        metric(
            "tool_business_calls_per_run",
            "Business calls per run",
            business_calls,
            unit="calls",
            score=None,
            n=len(runs),
        ),
        metric(
            "tool_first_call_correct",
            "First business call correct",
            first_tool_correct,
            unit="ratio",
            score=score_higher_is_better(first_tool_correct, 1.0),
            n=len(runs),
        ),
    ]
    if memory_first:
        result.metrics.append(
            metric(
                "tool_memory_recall_first",
                "Memory recall was first",
                memory_first_rate,
                unit="ratio",
                score=score_higher_is_better(memory_first_rate, 1.0),
                n=len(memory_first),
            )
        )
    result.diagnostics = {
        "scenarios": [
            {
                "scenario_id": run.scenario_id,
                "run_index": run.run_index,
                "f1": round(run.evaluation.get("calls", {}).get("f1", 0.0), 4),
                "exact": bool(run.evaluation.get("calls", {}).get("exact_trajectory", False)),
                "unnecessary": run.evaluation.get("calls", {}).get("unmatched", []),
                "missing": run.evaluation.get("calls", {}).get("missing", []),
                "forbidden": run.evaluation.get("calls", {}).get("forbidden", []),
                "answer_ok": bool(run.evaluation.get("answers_ok", False)),
                "stop_reason": run.stop_reason,
                "wall_ms": round(run.wall_ms, 3),
            }
            for run in runs
        ],
        "p95_tool_calls": percentile([run.tool_calls for run in runs], 0.95),
    }
    result.duration_ms = (__import__("time").perf_counter() - started) * 1000.0
    return result


def _argument_accuracy(match: Dict[str, Any]) -> float:
    matched = int(match.get("matched_count", 0))
    if matched <= 0:
        return 1.0
    mismatches = int(match.get("argument_mismatches", 0))
    return max(0.0, (matched - mismatches) / float(matched))


def _first_tool_correct(run: Any) -> bool:
    expected = run.labels.get("expected_calls") or []
    if not expected:
        return not any(
            item["name"] not in {"find_tools", "memory_recall"} and not item["skipped"]
            for item in run.tool_call_log
        )
    for item in run.tool_call_log:
        if item["skipped"] or item["name"] in {"find_tools", "memory_recall"}:
            continue
        return item["name"] == expected[0]["name"]
    return False


def _memory_recall_first(run: Any) -> bool:
    for item in run.tool_call_log:
        if item["skipped"]:
            continue
        return item["name"] == "memory_recall"
    return False
