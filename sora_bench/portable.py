"""Standalone portable benchmark path.

This module intentionally imports no SoraAgent package. It lets the general suite
score another agent through the JSON command contract even when SoraAgent is not
installed in the same Python environment.
"""

from __future__ import annotations

from .adapters import CommandAdapter
from .config import BenchConfig
from .general_score import score_general_runs
from .scenarios import scenarios_for_suite
from .types import RunRecord, SuiteResult


def run_external_general(config: BenchConfig) -> SuiteResult:
    adapter = CommandAdapter(config.agent_command, timeout=config.timeout)
    runs = []
    for scenario in scenarios_for_suite("general"):
        for run_index in range(config.runs):
            external = adapter.run(scenario)
            answer = external.answer
            lower = answer.casefold()
            expected_terms = [str(item) for item in scenario.expected_answer_terms]
            forbidden_terms = [str(item) for item in scenario.forbidden_answer_terms]
            expected_ok = all(term.casefold() in lower for term in expected_terms)
            forbidden_ok = not any(term.casefold() in lower for term in forbidden_terms)
            answers_ok = expected_ok and forbidden_ok
            tool_calls = [
                {
                    "name": str(item.get("name", "")),
                    "arguments": dict(item.get("arguments") or {}),
                    "executed": bool(item.get("executed", True)),
                    "is_error": bool(item.get("is_error", False)),
                    "skipped": False,
                }
                for item in external.tool_calls
            ]
            error = external.error
            runs.append(
                RunRecord(
                    scenario_id=scenario.id,
                    suite=scenario.suite,
                    run_index=run_index,
                    mode=config.mode,
                    task=scenario.task,
                    final_answer=answer,
                    proposal=answer,
                    passed=bool(answers_ok and not error),
                    eval_score=1.0 if answers_ok and not error else 0.0,
                    eval_checks=["external_answer_terms"],
                    error=error,
                    wall_ms=external.latency_ms,
                    provider_ms=external.latency_ms,
                    steps=1 if answer else 0,
                    provider_calls=1 if not error else 0,
                    tool_calls=len(tool_calls),
                    executed_tool_calls=sum(1 for item in tool_calls if item["executed"]),
                    tool_errors=sum(1 for item in tool_calls if item["is_error"]),
                    prompt_tokens=external.prompt_tokens,
                    completion_tokens=external.completion_tokens,
                    total_tokens=external.total_tokens,
                    reported_prompt_tokens=external.prompt_tokens,
                    reported_completion_tokens=external.completion_tokens,
                    reported_cached_tokens=external.reported_cached_tokens,
                    evaluation={
                        "calls": {},
                        "answers_ok": answers_ok,
                        "expected_answer_terms": expected_terms,
                        "forbidden_answer_terms": forbidden_terms,
                        "answer_text": answer[:2000],
                        "business_calls": tool_calls,
                        "orchestration_calls": [],
                        "skipped_business_calls": 0,
                        "tool_errors": sum(1 for item in tool_calls if item["is_error"]),
                        "approval_denials": 0,
                        "eval_passed": bool(answers_ok and not error),
                        "eval_score": 1.0 if answers_ok and not error else 0.0,
                    },
                    tool_call_log=tool_calls,
                    labels={
                        "expected_calls": [
                            {"name": item.name, "args": item.args}
                            for item in scenario.expected_calls
                        ],
                        "forbidden_calls": [
                            {"name": item.name, "args": item.args}
                            for item in scenario.forbidden_calls
                        ],
                        "description": scenario.description,
                    },
                    tags=list(scenario.tags),
                    metadata={
                        "adapter": "command",
                        "external": external.metadata,
                        "memory": "off",
                        "identity": False,
                        "policy": "external",
                    },
                )
            )
    return score_general_runs(runs)
