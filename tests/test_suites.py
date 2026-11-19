from __future__ import annotations

from pathlib import Path

from sora_bench.config import BenchConfig
from sora_bench.harness import BenchmarkHarness
from sora_bench.suites import run_suite


def _by_key(suite):
    return {item.key: item for item in suite.metrics}


def test_memory_suite_is_accurate_and_records_write_through(tmp_path: Path):
    config = BenchConfig(
        suites=("memory",),
        runs=1,
        output_dir=tmp_path / "out",
    )
    harness = BenchmarkHarness(config)
    try:
        suite = run_suite(harness, "memory")
    finally:
        harness.close()
    metrics = _by_key(suite)
    assert suite.error == ""
    assert metrics["memory_retrieval_hit_rate"].value == 1.0
    assert metrics["memory_answer_accuracy"].value == 1.0
    assert metrics["memory_recall_first_rate"].value == 1.0
    assert metrics["memory_write_through_rate"].value == 1.0
    assert suite.score is not None and suite.score > 70.0


def test_dreaming_suite_meets_ground_truth(tmp_path: Path):
    config = BenchConfig(
        suites=("dreaming",),
        runs=1,
        output_dir=tmp_path / "out",
    )
    harness = BenchmarkHarness(config)
    try:
        suite = run_suite(harness, "dreaming")
    finally:
        harness.close()
    metrics = _by_key(suite)
    assert suite.error == ""
    assert metrics["dream_merge_f1"].value == 1.0
    assert metrics["dream_conflict_f1"].value == 1.0
    assert metrics["dream_inference_f1"].value == 1.0
    assert metrics["dream_retention"].value == 1.0
    assert metrics["dream_idempotent"].value == 1.0
    assert metrics["dream_expiry_accuracy"].value == 1.0
    assert suite.score is not None and suite.score > 90.0


def test_autonomy_metric_discriminates_helpful_from_harmful(tmp_path: Path):
    balanced_config = BenchConfig(
        suites=("autonomy",),
        runs=1,
        autonomy_policy="balanced",
        output_dir=tmp_path / "balanced",
    )
    balanced_harness = BenchmarkHarness(balanced_config)
    try:
        balanced = run_suite(balanced_harness, "autonomy")
    finally:
        balanced_harness.close()
    balanced_metrics = _by_key(balanced)
    assert balanced_metrics["autonomy_net_utility"].value > 0
    assert balanced_metrics["autonomy_harmful_execution_rate"].value == 0.0

    overreach_config = BenchConfig(
        suites=("autonomy",),
        runs=1,
        autonomy_policy="overreach",
        output_dir=tmp_path / "overreach",
    )
    overreach_harness = BenchmarkHarness(overreach_config)
    try:
        overreach = run_suite(overreach_harness, "autonomy")
    finally:
        overreach_harness.close()
    overreach_metrics = _by_key(overreach)
    assert overreach_metrics["autonomy_net_utility"].value < 0
    assert overreach_metrics["autonomy_harmful_execution_rate"].value > 0
    assert overreach.score < balanced.score


def test_features_suite_meets_edge_case_ground_truth(tmp_path: Path):
    config = BenchConfig(
        suites=("features",),
        runs=1,
        output_dir=tmp_path / "out",
    )
    harness = BenchmarkHarness(config)
    try:
        suite = run_suite(harness, "features")
    finally:
        harness.close()
    metrics = _by_key(suite)
    assert suite.error == ""
    assert metrics["features_edge_pass_rate"].value == 100.0
    assert suite.diagnostics["checks_failed"] == 0
    assert suite.score is not None and suite.score == 100.0
