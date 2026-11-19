"""Adapters for agents that are not the built-in SoraAgent harness.

The benchmark's portable ``general`` suite can score any agent that can answer a
task. A command adapter keeps that contract intentionally small and language
agnostic: one JSON request on stdin, one JSON response on stdout.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


class AdapterError(RuntimeError):
    """The external agent did not honor the benchmark protocol."""


@dataclass
class AdapterResult:
    answer: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reported_cached_tokens: int = 0
    latency_ms: float = 0.0
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)


class CommandAdapter:
    """Run an arbitrary agent command with a minimal JSON stdin/stdout contract.

    Request::

        {"scenario_id": "...", "task": "...", "requires_tools": false}

    Response (one JSON object, or the last line of stdout)::

        {"answer": "...", "tool_calls": [], "usage": {
            "prompt_tokens": 0, "completion_tokens": 0,
            "cached_tokens": 0
        }, "latency_ms": 0}

    ``answer`` is required. The other fields are optional. This is deliberately
    smaller than SoraAgent's native trace format so other harnesses can adopt it
    without importing SoraAgent.
    """

    def __init__(self, command: str, *, timeout: float = 120.0) -> None:
        self.command = str(command or "").strip()
        if not self.command:
            raise ValueError("agent command must not be empty")
        self.timeout = max(1.0, float(timeout))

    def run(self, scenario: Any) -> AdapterResult:
        request = {
            "scenario_id": str(getattr(scenario, "id", "")),
            "task": str(getattr(scenario, "task", "")),
            "requires_tools": False,
        }
        started = time.perf_counter()
        try:
            process = subprocess.run(
                shlex.split(self.command),
                input=json.dumps(request, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return AdapterResult(error="{}: {}".format(type(exc).__name__, exc))
        latency_ms = (time.perf_counter() - started) * 1000.0
        if process.returncode != 0:
            detail = (process.stderr or process.stdout or "").strip()[-800:]
            return AdapterResult(
                error="agent command exited {}: {}".format(process.returncode, detail),
                latency_ms=latency_ms,
            )
        try:
            payload = _parse_response(process.stdout)
        except AdapterError as exc:
            return AdapterResult(
                error=str(exc),
                latency_ms=latency_ms,
                metadata={"adapter": "command", "command": self.command},
            )
        answer = str(payload.get("answer", "") or "")
        if not answer.strip():
            return AdapterResult(
                error="agent response has no non-empty answer",
                latency_ms=latency_ms,
            )
        usage = payload.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        return AdapterResult(
            answer=answer,
            tool_calls=[
                dict(item) for item in (payload.get("tool_calls") or []) if isinstance(item, dict)
            ][:100],
            prompt_tokens=max(0, int(usage.get("prompt_tokens", 0) or 0)),
            completion_tokens=max(0, int(usage.get("completion_tokens", 0) or 0)),
            reported_cached_tokens=max(0, int(usage.get("cached_tokens", 0) or 0)),
            latency_ms=float(payload.get("latency_ms") or latency_ms),
            metadata={
                "adapter": "command",
                "command": self.command,
                "reported_usage": bool(usage),
            },
        )


def _parse_response(stdout: str) -> Dict[str, Any]:
    text = str(stdout or "").strip()
    if not text:
        raise AdapterError("agent produced no stdout")
    candidates: Sequence[str] = (text, text.splitlines()[-1].strip())
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise AdapterError("agent response is not a JSON object")
