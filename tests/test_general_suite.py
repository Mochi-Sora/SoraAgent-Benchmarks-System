from __future__ import annotations

import json
from pathlib import Path

from sora_bench.cli import main


def test_general_suite_is_portable_and_deterministic(tmp_path: Path):
    output = tmp_path / "general"
    code = main(["--suites", "general", "--runs", "2", "--out", str(output)])
    assert code == 0
    report = json.loads((output / "latest.json").read_text(encoding="utf-8"))
    suite = report["suites"]["general"]
    assert suite["diagnostics"]["portable"] is True
    assert suite["diagnostics"]["requires"] == []
    assert suite["run_count"] == 16
    metrics = {item["key"]: item["value"] for item in suite["metrics"]}
    assert metrics["general_answer_accuracy"] == 1.0
    assert report["summary"]["uncertainty"]["general"]["n"] == 16


def test_raw_score_mode_suppresses_the_aggregate(tmp_path: Path):
    output = tmp_path / "raw"
    code = main(["--suites", "general", "--score-mode", "raw", "--out", str(output)])
    assert code == 0
    report = json.loads((output / "latest.json").read_text(encoding="utf-8"))
    assert report["summary"]["overall_score"] is None
    assert report["summary"]["grade"] == "—"
    assert report["summary"]["score_mode"] == "raw"
