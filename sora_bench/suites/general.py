"""Agent-agnostic instruction and answer-quality suite.

The suite deliberately uses no SoraAgent-specific tools, memory fixtures, or
orchestration assumptions. The same scenarios can be run by the command adapter.
"""

from __future__ import annotations

import time

from ..general_score import score_general_runs
from ..harness import BenchmarkHarness
from ..scenarios import scenarios_for_suite
from ..types import SuiteResult


def run(harness: BenchmarkHarness) -> SuiteResult:
    started = time.perf_counter()
    runs = []
    for scenario in scenarios_for_suite("general"):
        for run_index in range(harness.config.runs):
            if harness.config.adapter == "command":
                artifacts = harness.run_external_task(scenario, run_index=run_index)
            else:
                artifacts = harness.run_task(scenario, run_index=run_index)
            runs.append(artifacts.run)
            artifacts.cleanup(keep=harness.config.keep_workdir)
    result = score_general_runs(runs)
    result.duration_ms = (time.perf_counter() - started) * 1000.0
    return result
