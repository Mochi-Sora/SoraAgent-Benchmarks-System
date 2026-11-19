from __future__ import annotations

from sora_bench.stats import bootstrap_mean_ci, summarize_runs, wilson_interval


def test_wilson_interval_is_bounded_and_narrows_with_samples():
    low, high = wilson_interval(5, 10)
    assert 0.0 <= low <= 0.5 <= high <= 1.0
    assert wilson_interval(0, 0) is None
    narrow = wilson_interval(50, 100)
    wide = wilson_interval(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_bootstrap_interval_needs_replicates():
    assert bootstrap_mean_ci([1.0]) is None
    low, high = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], seed=7)
    assert low <= high


def test_summarize_runs_reports_insufficient_replicates():
    class Run:
        passed = True
        eval_score = 1.0
        total_tokens = 12
        wall_ms = 4.0

    summary = summarize_runs([Run()])
    assert summary["n"] == 1
    assert summary["insufficient_replicates"] is True
    assert summary["success_ci95"] is not None
    assert summary["tokens_ci95"] is None
