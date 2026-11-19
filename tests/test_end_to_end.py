from __future__ import annotations

import json
from pathlib import Path

from sora_bench.cli import main


def test_full_cli_run_writes_reports(tmp_path: Path):
    output = tmp_path / "results"
    code = main(["--out", str(output), "--runs", "1"])
    assert code == 0
    latest_json = output / "latest.json"
    latest_md = output / "latest.md"
    assert latest_json.exists()
    assert latest_md.exists()
    payload = json.loads(latest_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["mode"] == "simulated"
    for suite in (
        "tokens",
        "tools",
        "memory",
        "dreaming",
        "speed",
        "autonomy",
        "caching",
    ):
        assert suite in payload["suites"]
    assert payload["summary"]["overall_score"] > 0
    assert payload["summary"]["grade"] in {"A", "B", "C", "D", "F"}
    markdown = latest_md.read_text(encoding="utf-8")
    assert "# SoraAgent Benchmark" in markdown
    assert "## Prompt caching" in markdown
    assert "### Cache ablation" in markdown
    runs = list((output).glob("*/runs.jsonl"))
    assert runs
    assert runs[0].read_text(encoding="utf-8").strip()


def test_cli_include_requests_writes_request_dump(tmp_path: Path):
    output = tmp_path / "requests"
    code = main(
        [
            "--out",
            str(output),
            "--suites",
            "tools",
            "--include-requests",
        ]
    )
    assert code == 0
    dump = list(output.glob("*/cache_requests.jsonl"))
    assert dump
    first = json.loads(dump[0].read_text(encoding="utf-8").splitlines()[0])
    assert "prompt_hash" in first
    assert "lcp_tokens_prev" in first
