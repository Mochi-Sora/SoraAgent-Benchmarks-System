"""Scoring shared by the SoraAgent and portable command adapters."""

from __future__ import annotations

from .config import TARGETS
from .scoring import mean, score_higher_is_better, score_lower_is_better, weighted_score
from .types import SuiteResult, metric


def score_general_runs(runs: list) -> SuiteResult:
    result = SuiteResult(name="general")
    result.runs = list(runs)
    if not result.runs:
        result.error = "no general runs"
        return result
    answers = [run.evaluation.get("answers_ok", False) for run in result.runs]
    forbidden = [
        1.0
        if any(
            term.casefold() in str(run.evaluation.get("answer_text", "")).casefold()
            for term in run.evaluation.get("forbidden_answer_terms", [])
        )
        else 0.0
        for run in result.runs
    ]
    restraint = [
        1.0 if not run.evaluation.get("business_calls") else 0.0
        for run in result.runs
    ]
    success = [
        1.0 if passed and not run.error else 0.0
        for passed, run in zip(answers, result.runs)
    ]
    accuracy = mean(1.0 if item else 0.0 for item in answers)
    forbidden_rate = mean(forbidden)
    restraint_rate = mean(restraint)
    success_rate = mean(success)
    score = weighted_score(
        [
            (score_higher_is_better(accuracy, TARGETS["general_answer_accuracy"]), 0.55),
            (score_higher_is_better(success_rate, 0.90), 0.20),
            (score_lower_is_better(forbidden_rate, TARGETS["general_forbidden_rate"]), 0.15),
            (score_higher_is_better(restraint_rate, 0.90), 0.10),
        ]
    )
    result.score = score
    result.metrics = [
        metric(
            "general_answer_accuracy",
            "Instruction/answer accuracy",
            accuracy,
            unit="ratio",
            score=score_higher_is_better(accuracy, TARGETS["general_answer_accuracy"]),
            n=len(result.runs),
        ),
        metric(
            "general_task_success",
            "Task success rate",
            success_rate,
            unit="ratio",
            score=score_higher_is_better(success_rate, 0.90),
            n=len(result.runs),
        ),
        metric(
            "general_forbidden_answer_rate",
            "Forbidden-answer rate",
            forbidden_rate,
            unit="ratio",
            higher_is_better=False,
            score=score_lower_is_better(forbidden_rate, TARGETS["general_forbidden_rate"]),
            n=len(result.runs),
        ),
        metric(
            "general_tool_restraint",
            "No unnecessary tool calls",
            restraint_rate,
            unit="ratio",
            score=score_higher_is_better(restraint_rate, 0.90),
            n=len(result.runs),
        ),
    ]
    result.diagnostics = {
        "scenarios": [
            {
                "scenario_id": run.scenario_id,
                "run_index": run.run_index,
                "answers_ok": bool(run.evaluation.get("answers_ok")),
                "forbidden": item > 0,
                "business_calls": len(run.evaluation.get("business_calls") or []),
                "tokens": run.total_tokens,
                "wall_ms": run.wall_ms,
                "answer": run.final_answer or run.proposal,
            }
            for run, item in zip(result.runs, forbidden)
        ],
        "portable": True,
        "requires": [],
    }
    return result
