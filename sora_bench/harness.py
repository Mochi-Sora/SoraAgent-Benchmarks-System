"""The benchmark harness: build a real SoraAgent run and record everything."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from sora_agent.agent import Agent
from sora_agent.memory.config import MemoryConfig
from sora_agent.memory.daily import DailyStore
from sora_agent.memory.hook import MemoryHook

from .backend import FakeMemoryBackend
from .config import BenchConfig
from .providers import (
    LatencyProvider,
    RecordingProvider,
    ScriptedProvider,
    build_live_provider,
)
from .scenarios import PlanStep, TaskScenario, build_plan
from .scoring import call_signature, match_expected_calls
from .support import (
    FixedClock,
    SteppingClock,
    auto_approval,
    build_identity,
    fixed_clock,
)
from .tools import ToolLedger, build_registry
from .types import RequestRecord, RunRecord

log = logging.getLogger(__name__)

ORCHESTRATION_TOOLS = frozenset({"find_tools", "memory_recall"})


@dataclass
class RunArtifacts:
    run: RunRecord
    result: Any
    ledger: ToolLedger
    backend: Optional[FakeMemoryBackend] = None
    daily: Optional[DailyStore] = None
    memory: Any = None
    identity: Any = None
    requests: List[RequestRecord] = field(default_factory=list)
    root: Optional[Path] = None
    _tempdir: Any = None

    def cleanup(self, keep: bool = False) -> None:
        try:
            if self.memory is not None:
                self.memory.close()
        except Exception:  # noqa: BLE001 - cleanup is best effort
            log.debug("memory close failed", exc_info=True)
        if self._tempdir is not None and not keep:
            try:
                self._tempdir.cleanup()
            except Exception:  # noqa: BLE001
                log.debug("temporary directory cleanup failed", exc_info=True)
            self._tempdir = None


def prepare_process_environment(base: Path) -> None:
    """Keep optional defaults away from the developer's real home directory."""

    os.environ.setdefault("SORA_MEMORY", "0")
    os.environ.setdefault("SORA_TOOLPACK", "1")
    os.environ.setdefault("SORA_DELEGATIONS_ROOT", str(base / "delegations"))
    os.environ.setdefault("SORA_MEMORY_DREAM_AUTO", "0")
    os.environ.setdefault("SORA_MEMORY_AUTO_DOWNLOAD", "0")
    os.environ.setdefault("SORA_IDENTITY", "0")
    os.environ.setdefault("SORA_CALENDAR_DB", str(base / "calendar.db"))
    os.environ.setdefault("SORA_CONFIG", str(base / "config.json"))
    os.environ.setdefault("SORA_SECRETS", str(base / "secrets.env"))


class BenchmarkHarness:
    def __init__(self, config: BenchConfig) -> None:
        self.config = config
        self.prepare_dir = Path(tempfile.mkdtemp(prefix="sora-bench-process-"))
        prepare_process_environment(self.prepare_dir)
        self._live_provider: Any = None

    # ------------------------------------------------------------------
    # Provider construction
    # ------------------------------------------------------------------
    def _provider(
        self,
        *,
        plan: Sequence[PlanStep],
        final_answer: str,
        recall_query: str,
        extract_recall: bool,
        latency_ms: Optional[float],
    ) -> Any:
        if self.config.simulated:
            inner: Any = ScriptedProvider(
                plan,
                final_answer,
                recall_query=recall_query,
                extract_recall=extract_recall,
            )
        else:
            if self._live_provider is None:
                self._live_provider = build_live_provider(
                    self.config.provider_kind,
                    model=self.config.model,
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                    temperature=self.config.temperature,
                )
            inner = self._live_provider
        delay = (
            self.config.simulated_latency_ms if latency_ms is None else max(0.0, float(latency_ms))
        )
        if delay > 0 or self.config.latency_jitter_ms > 0:
            inner = LatencyProvider(
                inner,
                delay_ms=delay,
                jitter_ms=self.config.latency_jitter_ms,
                seed=self.config.seed,
            )
        return RecordingProvider(
            inner,
            model=self.config.model or ("simulated" if self.config.simulated else "live"),
            min_cache_block=self.config.min_cache_block,
        )

    # ------------------------------------------------------------------
    # One task run
    # ------------------------------------------------------------------
    def run_task(
        self,
        scenario: TaskScenario,
        *,
        run_index: int = 0,
        policy: str = "balanced",
        variant: str = "",
        plan: Optional[Sequence[PlanStep]] = None,
        identity: Optional[bool] = None,
        clock: Any = None,
        memory: Optional[str] = None,
        prepare_memory: Optional[Callable[[Any, Any, Any], None]] = None,
        recall_query: str = "",
        extract_recall: bool = False,
        latency_ms: Optional[float] = None,
        extra_tags: Optional[Sequence[str]] = None,
    ) -> RunArtifacts:
        tempdir = tempfile.TemporaryDirectory(prefix="sora-bench-run-")
        root = Path(tempdir.name)
        ledger = ToolLedger()
        registry = build_registry(ledger)

        identity_enabled = scenario.identity if identity is None else bool(identity)
        identity_store = build_identity(root / "identity") if identity_enabled else None

        memory_mode = scenario.memory if memory is None else memory
        memory_hook = None
        backend = None
        daily = None
        if memory_mode != "off" or prepare_memory is not None:
            mem_config = MemoryConfig(
                root=root / "memory-root",
                auto_download=False,
                dream_auto=False,
            )
            backend = FakeMemoryBackend()
            daily = DailyStore(
                mem_config.daily_dir,
                session_id="benchmark",
            )
            memory_hook = MemoryHook(
                config=mem_config,
                client=backend,
                daily=daily,
                auto_dream=False,
                identity=identity_store,
            )
            if prepare_memory is not None:
                prepare_memory(backend, daily, mem_config)

        clock_override = scenario.clock if clock is None else clock
        if clock_override is False or clock_override == "off":
            clock_value: Any = False
        elif clock_override is True or clock_override == "fixed":
            clock_value = FixedClock(fixed_clock())
        elif clock_override == "stepping":
            clock_value = SteppingClock()
        else:
            clock_value = clock_override

        selected_plan = list(plan) if plan is not None else build_plan(scenario, policy)
        provider = self._provider(
            plan=selected_plan,
            final_answer=scenario.final_answer,
            recall_query=recall_query,
            extract_recall=extract_recall,
            latency_ms=latency_ms,
        )

        events: List[Any] = []
        agent = Agent(
            provider=provider,
            registry=registry,
            approval=auto_approval(),
            memory=memory_hook if memory_hook is not None else False,
            identity=identity_store if identity_store is not None else False,
            clock=clock_value,
            calendar=False,
            max_steps=scenario.max_steps or self.config.max_steps,
            provider_retries=0,
            on_event=events.append,
        )

        started = time.perf_counter()
        error = ""
        result = None
        try:
            result = agent.run(scenario.task)
        except Exception as exc:  # noqa: BLE001 - a benchmark run must not vanish
            error = "{}: {}".format(type(exc).__name__, exc)
            log.exception("scenario %s crashed", scenario.id)
        wall_ms = (time.perf_counter() - started) * 1000.0

        record = self._build_record(
            scenario=scenario,
            result=result,
            provider=provider,
            ledger=ledger,
            events=events,
            wall_ms=wall_ms,
            error=error,
            run_index=run_index,
            variant=variant,
            policy=policy,
            tags=list(scenario.tags) + list(extra_tags or []),
        )
        artifacts = RunArtifacts(
            run=record,
            result=result,
            ledger=ledger,
            backend=backend,
            daily=daily,
            memory=memory_hook,
            identity=identity_store,
            requests=list(provider.requests),
            root=root,
            _tempdir=tempdir,
        )
        return artifacts

    # ------------------------------------------------------------------
    # Record creation
    # ------------------------------------------------------------------
    def _build_record(
        self,
        *,
        scenario: TaskScenario,
        result: Any,
        provider: RecordingProvider,
        ledger: ToolLedger,
        events: Sequence[Any],
        wall_ms: float,
        error: str,
        run_index: int,
        variant: str,
        policy: str,
        tags: List[str],
    ) -> RunRecord:
        requests = list(provider.requests)
        steps = list(getattr(result, "steps", []) or []) if result is not None else []
        tool_log: List[Dict[str, Any]] = []
        seen_signatures = set()
        redundant = 0
        for step in steps:
            for item in getattr(step, "results", []) or []:
                call = getattr(item, "call", None)
                name = str(getattr(call, "name", "") or "")
                arguments = dict(getattr(call, "arguments", {}) or {})
                signature = call_signature({"name": name, "arguments": arguments})
                if not getattr(item, "skipped", False):
                    if signature in seen_signatures:
                        redundant += 1
                    seen_signatures.add(signature)
                tool_log.append(
                    {
                        "name": name,
                        "arguments": arguments,
                        "step": getattr(step, "index", -1),
                        "executed": bool(getattr(item, "executed", False)),
                        "is_error": bool(getattr(item, "is_error", False)),
                        "skipped": bool(getattr(item, "skipped", False)),
                        "approval_required": bool(getattr(item, "approval_required", False)),
                        "approved": bool(getattr(item, "approved", False)),
                        "stop_reason": str(getattr(item, "stop_reason", "") or ""),
                        "content": str(getattr(item, "content", "") or "")[:500],
                    }
                )

        business_calls = [
            item
            for item in tool_log
            if item["name"] not in ORCHESTRATION_TOOLS and not item["skipped"]
        ]
        match = match_expected_calls(
            business_calls,
            scenario.expected_calls,
            scenario.forbidden_calls,
        )
        final_answer = str(getattr(result, "final_answer", "") or "")
        proposal = str(getattr(result, "proposal", "") or "")
        answer_text = final_answer or proposal
        expected_terms = [str(item) for item in scenario.expected_answer_terms]
        forbidden_terms = [str(item) for item in scenario.forbidden_answer_terms]
        lower_answer = answer_text.casefold()
        expected_ok = all(term.casefold() in lower_answer for term in expected_terms)
        forbidden_ok = not any(term.casefold() in lower_answer for term in forbidden_terms)
        answers_ok = (
            expected_ok
            if scenario.allow_forbidden_with_expected
            else (expected_ok and forbidden_ok)
        )

        report = getattr(result, "report", None) if result is not None else None
        eval_passed = bool(getattr(report, "passed", False))
        eval_score = float(getattr(report, "score", 0.0) or 0.0)
        eval_checks = list(getattr(report, "checks", []) or [])

        prompt_tokens = sum(request.prompt_tokens for request in requests)
        completion_tokens = sum(request.completion_tokens for request in requests)
        reported_prompt = sum(request.reported_prompt_tokens for request in requests)
        reported_completion = sum(request.reported_completion_tokens for request in requests)
        reported_cached = sum(request.reported_cached_tokens for request in requests)
        reported_cache_write = sum(request.reported_cache_write_tokens for request in requests)
        tool_ms = sum(
            float(getattr(event, "duration", 0.0)) * 1000.0
            for event in events
            if getattr(event, "kind", "") == "tool_finished"
        )
        provider_ms = sum(request.latency_ms for request in requests)
        overhead_ms = max(0.0, wall_ms - provider_ms - tool_ms)
        steps_count = len(steps)
        provider_calls = len(requests)
        executed = sum(1 for item in tool_log if item["executed"])
        skipped = sum(1 for item in tool_log if item["skipped"])
        errors = sum(1 for item in tool_log if item["is_error"])
        denials = sum(1 for item in tool_log if item["approval_required"] and not item["approved"])
        skipped_business = sum(
            1 for item in tool_log if item["skipped"] and item["name"] not in ORCHESTRATION_TOOLS
        )

        evaluation = {
            "calls": match,
            "answers_ok": answers_ok,
            "expected_answer_terms": expected_terms,
            "forbidden_answer_terms": forbidden_terms,
            "answer_text": answer_text[:2000],
            "skipped_business_calls": skipped_business,
            "orchestration_calls": [
                item for item in tool_log if item["name"] in ORCHESTRATION_TOOLS
            ],
            "business_calls": business_calls,
            "unique_business_tools": sorted({item["name"] for item in business_calls}),
            "redundant_tool_calls": redundant,
            "tool_errors": errors,
            "approval_denials": denials,
            "eval_passed": eval_passed,
            "eval_score": eval_score,
        }

        record = RunRecord(
            scenario_id=scenario.id,
            suite=scenario.suite,
            run_index=run_index,
            mode=self.config.mode,
            task=scenario.task,
            variant=variant,
            stop_reason=str(getattr(result, "stop_reason", "") or ""),
            final_answer=final_answer,
            proposal=proposal,
            passed=bool(eval_passed and answers_ok and not error),
            eval_score=eval_score,
            eval_checks=eval_checks,
            error=error,
            wall_ms=wall_ms,
            provider_ms=provider_ms,
            tool_ms=tool_ms,
            overhead_ms=overhead_ms,
            steps=steps_count,
            provider_calls=provider_calls,
            tool_calls=len(tool_log),
            executed_tool_calls=executed,
            skipped_tool_calls=skipped,
            tool_errors=errors,
            approval_denials=denials,
            redundant_tool_calls=redundant,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            reported_prompt_tokens=reported_prompt,
            reported_completion_tokens=reported_completion,
            reported_cached_tokens=reported_cached,
            reported_cache_write_tokens=reported_cache_write,
            tool_schema_tokens=sum(r.tool_schema_tokens for r in requests),
            system_tokens=sum(r.system_tokens for r in requests),
            identity_tokens=sum(r.identity_tokens for r in requests),
            clock_tokens=sum(r.clock_tokens for r in requests),
            conversation_tokens=sum(
                r.user_tokens + r.assistant_tokens + r.tool_result_tokens for r in requests
            ),
            evaluation=evaluation,
            tool_call_log=tool_log,
            ledger=ledger.to_dicts(),
            requests=requests,
            labels={
                "expected_calls": [
                    {"name": item.name, "args": item.args} for item in scenario.expected_calls
                ],
                "forbidden_calls": [
                    {"name": item.name, "args": item.args} for item in scenario.forbidden_calls
                ],
                "autonomy_actions": [
                    {
                        "name": item.name,
                        "args": item.args,
                        "label": item.label,
                        "weight": item.weight,
                        "policies": list(item.policies),
                    }
                    for item in scenario.autonomy_actions
                ],
                "requested_calls": [
                    {"name": item.name, "args": item.args} for item in scenario.requested_calls
                ],
                "description": scenario.description,
                "plan": [
                    {
                        "kind": step.kind,
                        "name": step.name,
                        "args": step.args,
                        "discover_need": step.discover_need,
                    }
                    for step in (build_plan(scenario, policy) if self.config.simulated else [])
                ],
            },
            tags=tags,
            metadata={
                "policy": policy,
                "identity": bool(scenario.identity),
                "memory": scenario.memory,
                "clock": scenario.clock,
                "task_success": bool(eval_passed and answers_ok and not error),
            },
        )
        return record

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._live_provider is not None:
            close = getattr(self._live_provider, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    log.debug("live provider close failed", exc_info=True)
        try:
            shutil.rmtree(self.prepare_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
