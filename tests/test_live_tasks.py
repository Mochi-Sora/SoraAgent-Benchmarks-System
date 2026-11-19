from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sora_agent.providers import ModelTurn, TokenUsage

import live_tasks


def _run(answer: str, tools: list[str], executed: bool = True):
    results = [
        SimpleNamespace(
            call=SimpleNamespace(name=name),
            executed=executed,
        )
        for name in tools
    ]
    step = SimpleNamespace(results=results)
    return SimpleNamespace(
        final_answer=answer,
        proposal=answer,
        steps=[step],
    )


def test_web_evaluation_passes_on_expected_extract():
    root = Path("/tmp")
    state = live_tasks._prepare_web(root)
    result = _run(
        "The H1 is Example Domain. This domain is for use in illustrative examples.",
        ["web_extract"],
    )
    evaluation = live_tasks._evaluate_web(root, result, state)
    assert evaluation["passed"] is True


def test_code_fixture_and_evaluation_pass_when_tests_are_fixed(tmp_path):
    state = live_tasks._prepare_code(tmp_path)
    source = tmp_path / "demo" / "calc.py"
    source.write_text(
        source.read_text().replace("return a * b", "return a / b"),
        encoding="utf-8",
    )
    result = _run("Tests pass.", ["terminal"])
    evaluation = live_tasks._evaluate_code(tmp_path, result, state)
    assert evaluation["passed"] is True
    assert evaluation["checks"]["tests_pass"] is True
    assert evaluation["checks"]["tests_unchanged"] is True


def test_audit_evaluation_requires_read_only_and_correct_facts(tmp_path):
    state = live_tasks._prepare_audit(tmp_path)
    answer = "2 TODO occurrences. Largest: data.log. Deprecated: beta.py, notes.md."
    result = _run(answer, ["search_files"])
    evaluation = live_tasks._evaluate_audit(tmp_path, result, state)
    assert evaluation["passed"] is True
    assert evaluation["checks"]["read_only"] is True

    (tmp_path / "audit" / "small.txt").write_text("changed\n", encoding="utf-8")
    evaluation = live_tasks._evaluate_audit(tmp_path, result, state)
    assert evaluation["checks"]["read_only"] is False
    assert evaluation["passed"] is False


def test_budget_provider_accounts_usage_and_stops_at_cap():
    class Inner:
        def __init__(self):
            self.calls = 0
            self.last_cache_usage = {"cached_tokens": 5}

        def complete_turn(self, messages, tools):
            self.calls += 1
            return ModelTurn(
                text="ok",
                usage=TokenUsage(prompt_tokens=60, completion_tokens=10, total_tokens=70),
            )

    budget = live_tasks.Budget(limit=100)
    inner = Inner()
    provider = live_tasks.BudgetProvider(inner, budget, "test")
    provider.complete_turn([], [])
    assert budget.used == 70
    with pytest.raises(live_tasks.BudgetExceeded):
        provider.complete_turn([], [])
    assert budget.used == 140
    with pytest.raises(live_tasks.BudgetExceeded):
        provider.complete_turn([], [])
    assert inner.calls == 2


def test_budget_state_round_trip(tmp_path):
    path = tmp_path / "budget.json"
    budget = live_tasks.Budget(
        limit=100,
        prompt_tokens=10,
        completion_tokens=5,
        cached_tokens=3,
        calls=2,
    )
    live_tasks.save_budget_state(path, budget)
    loaded = live_tasks.load_budget_state(path)
    assert loaded["prompt_tokens"] == 10
    assert loaded["completion_tokens"] == 5
    assert loaded["cached_tokens"] == 3
    assert loaded["calls"] == 2


def test_aggregate_results_groups_repetitions():
    results = [
        live_tasks.TaskResult(
            key="web", title="Web", passed=True, prompt_tokens=10, completion_tokens=2
        ),
        live_tasks.TaskResult(
            key="web", title="Web", passed=False, prompt_tokens=20, completion_tokens=3
        ),
        live_tasks.TaskResult(
            key="code", title="Code", passed=True, prompt_tokens=30, completion_tokens=4
        ),
    ]
    aggregates = live_tasks.aggregate_results(results)
    assert aggregates["web"]["runs"] == 2
    assert aggregates["web"]["passed"] == 1
    assert aggregates["web"]["pass_rate"] == 0.5
    assert aggregates["web"]["total_tokens"] == 35
    assert aggregates["code"]["runs"] == 1


def test_write_report_includes_aggregates_and_run_totals(tmp_path):
    budget = live_tasks.Budget(
        limit=1000,
        prompt_tokens=120,
        completion_tokens=30,
        cached_tokens=60,
        calls=4,
    )
    results = [
        live_tasks.TaskResult(
            key="web",
            title="Web",
            repetition=1,
            status="passed",
            passed=True,
            prompt_tokens=20,
            completion_tokens=5,
            cached_tokens=10,
            duration_ms=100,
        )
    ]
    paths = live_tasks.write_report(
        tmp_path / "run",
        provider="deepseek",
        model="deepseek-flash",
        budget=budget,
        results=results,
        started="2026-11-12T00:00:00Z",
        run_start=(100, 25, 50),
        prior_used=100,
    )
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["run_used_tokens"] == 50
    assert payload["run_prompt_tokens"] == 20
    assert payload["aggregates"]["web"]["runs"] == 1
    assert "Aggregate" in paths["markdown"].read_text(encoding="utf-8")
