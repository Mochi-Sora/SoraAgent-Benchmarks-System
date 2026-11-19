from __future__ import annotations

import sys
from pathlib import Path

from sora_bench.adapters import CommandAdapter


def test_command_adapter_json_contract(tmp_path: Path):
    script = tmp_path / "agent.py"
    script.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "print(json.dumps({'answer': request['task'][::-1], "
        "'usage': {'prompt_tokens': 4, 'completion_tokens': 2}}))\n",
        encoding="utf-8",
    )
    adapter = CommandAdapter(f"{sys.executable} {script}")
    scenario = type("Scenario", (), {"id": "general/echo", "task": "abc"})()
    result = adapter.run(scenario)
    assert result.answer == "cba"
    assert result.total_tokens == 6
    assert not result.error


def test_command_adapter_rejects_non_json(tmp_path: Path):
    script = tmp_path / "bad.py"
    script.write_text("print('not json')\n", encoding="utf-8")
    adapter = CommandAdapter(f"{sys.executable} {script}")
    scenario = type("Scenario", (), {"id": "general/bad", "task": "x"})()
    result = adapter.run(scenario)
    assert result.answer == ""
    assert "JSON" in result.error
