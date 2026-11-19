"""Shared data structures for the SoraAgent benchmark.

The objects here are deliberately plain dataclasses with ``to_dict`` helpers so
the whole report can be serialized to JSON without a custom encoder.  Nothing in
this module imports SoraAgent, which keeps the report format independent from
the version under test.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.1"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Metric:
    """One reported number plus the context needed to interpret it."""

    key: str
    label: str
    value: float
    unit: str = ""
    higher_is_better: bool = True
    score: Optional[float] = None
    detail: str = ""
    n: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value": _round(self.value),
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
            "score": _round(self.score) if self.score is not None else None,
            "detail": self.detail,
            "n": self.n,
        }


@dataclass
class RequestRecord:
    """What one provider call actually looked like.

    ``prompt_tokens`` and ``completion_tokens`` are deterministic estimates
    (see ``tokens.py``); ``reported_*`` values come from the provider when it
    reports usage.  Cache measurements are computed on the canonical request
    text, independently of provider usage, so they work offline and online.
    """

    index: int
    step: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reported_prompt_tokens: int = 0
    reported_completion_tokens: int = 0
    reported_cached_tokens: int = 0
    reported_cache_write_tokens: int = 0
    cache_usage_source: str = ""
    latency_ms: float = 0.0
    prompt_chars: int = 0
    message_count: int = 0
    tool_count: int = 0
    tool_schema_tokens: int = 0
    system_tokens: int = 0
    identity_tokens: int = 0
    clock_tokens: int = 0
    state_tokens: int = 0
    user_tokens: int = 0
    assistant_tokens: int = 0
    tool_result_tokens: int = 0
    lcp_tokens_prev: int = 0
    lcp_tokens_first: int = 0
    lcp_message_tokens_prev: int = 0
    message_prefix_ratio: float = 0.0
    first_diff_kind: str = "none"
    first_diff_role: str = ""
    tool_schema_changed: bool = False
    clock_changed: bool = False
    identity_changed: bool = False
    system_changed: bool = False
    prompt_hash: str = ""
    message_hash: str = ""
    tool_schema_hash: str = ""
    system_hash: str = ""
    clock_hash: str = ""
    identity_hash: str = ""
    cache_eligible: bool = False
    clock_trailing: bool = False
    error: str = ""
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        for key, value in list(data.items()):
            if isinstance(value, float):
                data[key] = _round(value)
        return data


@dataclass
class RunRecord:
    """One scenario run (a full SoraAgent ``Agent.run`` call)."""

    scenario_id: str
    suite: str
    run_index: int
    mode: str
    task: str
    variant: str = ""
    stop_reason: str = ""
    final_answer: str = ""
    proposal: str = ""
    passed: bool = False
    eval_score: float = 0.0
    eval_checks: List[str] = field(default_factory=list)
    error: str = ""
    wall_ms: float = 0.0
    provider_ms: float = 0.0
    tool_ms: float = 0.0
    overhead_ms: float = 0.0
    steps: int = 0
    provider_calls: int = 0
    tool_calls: int = 0
    executed_tool_calls: int = 0
    skipped_tool_calls: int = 0
    tool_errors: int = 0
    approval_denials: int = 0
    redundant_tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reported_prompt_tokens: int = 0
    reported_completion_tokens: int = 0
    reported_cached_tokens: int = 0
    reported_cache_write_tokens: int = 0
    tool_schema_tokens: int = 0
    system_tokens: int = 0
    identity_tokens: int = 0
    clock_tokens: int = 0
    conversation_tokens: int = 0
    evaluation: Dict[str, Any] = field(default_factory=dict)
    tool_call_log: List[Dict[str, Any]] = field(default_factory=list)
    ledger: List[Dict[str, Any]] = field(default_factory=list)
    requests: List[RequestRecord] = field(default_factory=list)
    labels: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def cache_hit_tokens(self) -> int:
        total = 0
        for request in self.requests:
            total += request.lcp_tokens_prev
        return total

    def to_dict(self, include_requests: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "suite": self.suite,
            "run_index": self.run_index,
            "mode": self.mode,
            "task": self.task,
            "variant": self.variant,
            "stop_reason": self.stop_reason,
            "final_answer": self.final_answer,
            "proposal": self.proposal,
            "passed": self.passed,
            "eval_score": _round(self.eval_score),
            "eval_checks": list(self.eval_checks),
            "error": self.error,
            "wall_ms": _round(self.wall_ms),
            "provider_ms": _round(self.provider_ms),
            "tool_ms": _round(self.tool_ms),
            "overhead_ms": _round(self.overhead_ms),
            "steps": self.steps,
            "provider_calls": self.provider_calls,
            "tool_calls": self.tool_calls,
            "executed_tool_calls": self.executed_tool_calls,
            "skipped_tool_calls": self.skipped_tool_calls,
            "tool_errors": self.tool_errors,
            "approval_denials": self.approval_denials,
            "redundant_tool_calls": self.redundant_tool_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reported_prompt_tokens": self.reported_prompt_tokens,
            "reported_completion_tokens": self.reported_completion_tokens,
            "reported_cached_tokens": self.reported_cached_tokens,
            "reported_cache_write_tokens": self.reported_cache_write_tokens,
            "tool_schema_tokens": self.tool_schema_tokens,
            "system_tokens": self.system_tokens,
            "identity_tokens": self.identity_tokens,
            "clock_tokens": self.clock_tokens,
            "conversation_tokens": self.conversation_tokens,
            "evaluation": self.evaluation,
            "tool_call_log": self.tool_call_log,
            "ledger": self.ledger,
            "labels": self.labels,
            "tags": self.tags,
            "metadata": self.metadata,
        }
        if include_requests:
            data["requests"] = [request.to_dict() for request in self.requests]
        return data


@dataclass
class SuiteResult:
    name: str
    score: Optional[float] = None
    metrics: List[Metric] = field(default_factory=list)
    runs: List[RunRecord] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str = ""

    def add(self, metric: Metric) -> Metric:
        self.metrics.append(metric)
        return metric

    def to_dict(self, include_runs: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "name": self.name,
            "score": _round(self.score) if self.score is not None else None,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "diagnostics": self.diagnostics,
            "duration_ms": _round(self.duration_ms),
            "error": self.error,
            "run_count": len(self.runs),
        }
        if include_runs:
            data["runs"] = [run.to_dict() for run in self.runs]
        return data


@dataclass
class BenchmarkReport:
    mode: str
    provider: str
    model: str
    seed: int
    score_mode: str = "heuristic"
    adapter: str = "sora"
    suites: Dict[str, SuiteResult] = field(default_factory=dict)
    runs: List[RunRecord] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=utcnow_iso)
    finished_at: str = ""
    duration_ms: float = 0.0
    schema_version: str = SCHEMA_VERSION
    errors: List[str] = field(default_factory=list)

    def environment(self) -> Dict[str, Any]:
        try:
            # The benchmark always runs the sibling checkout.  Prefer the
            # source version over a stale installed distribution.
            from sora_agent import __version__ as sora_version
        except Exception:  # noqa: BLE001 - metadata is best effort
            try:
                from importlib.metadata import version

                sora_version = version("sora-agent")
            except Exception:  # noqa: BLE001
                sora_version = "unknown"
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "sora_agent_version": sora_version,
            "adapter": self.adapter,
        }

    def to_dict(self, include_requests: bool = False) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.finished_at or utcnow_iso(),
            "started_at": self.started_at,
            "duration_ms": _round(self.duration_ms),
            "mode": self.mode,
            "provider": self.provider,
            "model": self.model,
            "seed": self.seed,
            "score_mode": self.score_mode,
            "adapter": self.adapter,
            "environment": self.environment(),
            "summary": self.summary,
            "suites": {name: suite.to_dict() for name, suite in self.suites.items()},
            "diagnostics": self.diagnostics,
            "errors": self.errors,
            "runs": [run.to_dict(include_requests=include_requests) for run in self.runs],
        }


def _round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return round(float(value), digits)


def metric(
    key: str,
    label: str,
    value: float,
    *,
    unit: str = "",
    higher_is_better: bool = True,
    score: Optional[float] = None,
    detail: str = "",
    n: int = 0,
) -> Metric:
    return Metric(
        key=key,
        label=label,
        value=float(value),
        unit=unit,
        higher_is_better=higher_is_better,
        score=score,
        detail=detail,
        n=n,
    )
