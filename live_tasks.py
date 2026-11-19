#!/usr/bin/env python3
"""Live three-task benchmark: web scraping, codebase repair, read-only audit.

This runner is intentionally separate from the deterministic benchmark.  It
drives a real OpenAI-compatible model through SoraAgent's actual tool loop and
enforces a hard total token budget.

Example::

    DEEPSEEK_API_KEY=... python3 live_tasks.py \
        --provider deepseek --model deepseek-flash --budget-tokens 10000000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sora_agent.agent import Agent  # noqa: E402
from sora_agent.approval import AutoApprovalGate  # noqa: E402
from sora_agent.providers import provider_turn  # noqa: E402
from sora_agent.settings import load_secrets  # noqa: E402
from sora_agent.toolpack import ToolPackConfig, register_core_tools  # noqa: E402
from sora_agent.tools import ToolRegistry  # noqa: E402

from sora_bench.providers import build_live_provider  # noqa: E402

BUDGET_EXCEEDED = "budget_exceeded"


@dataclass
class Budget:
    limit: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 0
    tasks: dict[str, int] = field(default_factory=dict)

    @property
    def used(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def add(
        self,
        *,
        prompt: int,
        completion: int,
        cached: int = 0,
        task: str = "",
    ) -> None:
        self.prompt_tokens += max(0, prompt)
        self.completion_tokens += max(0, completion)
        self.cached_tokens += max(0, cached)
        self.calls += 1
        self.tasks[task] = self.tasks.get(task, 0) + max(0, prompt) + max(0, completion)

    def check(self) -> None:
        if self.used >= self.limit:
            raise BudgetExceeded(
                "token budget exhausted: {:,} / {:,}".format(self.used, self.limit)
            )


class BudgetExceeded(RuntimeError):
    pass


class BudgetProvider:
    """Count provider-reported usage and stop before spending past the cap."""

    def __init__(self, inner: Any, budget: Budget, task: str) -> None:
        self.inner = inner
        self.budget = budget
        self.task = task
        self.last_usage: Any = None
        self.last_cache_usage: dict[str, Any] = {}

    def complete_turn(self, messages: list[Any], tools: list[Any]) -> Any:
        self.budget.check()
        turn = provider_turn(self.inner, messages, tools)
        usage = getattr(turn, "usage", None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        cache = getattr(self.inner, "last_cache_usage", None)
        cache = dict(cache) if isinstance(cache, dict) else {}
        cached = int(cache.get("cached_tokens") or 0)
        if not prompt and not completion:
            # Conservative fallback for providers that omit usage.
            prompt = max(1, sum(len(str(getattr(m, "content", ""))) for m in messages) // 4)
            completion = max(1, len(str(getattr(turn, "text", ""))) // 4)
        self.budget.add(
            prompt=prompt,
            completion=completion,
            cached=cached,
            task=self.task,
        )
        self.last_usage = usage
        self.last_cache_usage = cache
        if self.budget.used > self.budget.limit:
            raise BudgetExceeded(
                "token budget exceeded during task {!r}: {:,} / {:,}".format(
                    self.task, self.budget.used, self.budget.limit
                )
            )
        return turn

    def complete(self, messages: list[Any], tools: list[Any]) -> str:
        return self.complete_turn(messages, tools).text


def load_budget_state(path: Path) -> dict[str, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "calls"):
        try:
            out[key] = max(0, int(payload.get(key, 0)))
        except (TypeError, ValueError):
            out[key] = 0
    return out


def save_budget_state(path: Path, budget: "Budget") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt_tokens": budget.prompt_tokens,
        "completion_tokens": budget.completion_tokens,
        "cached_tokens": budget.cached_tokens,
        "calls": budget.calls,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@dataclass
class TaskSpec:
    key: str
    title: str
    prompt: str
    prepare: Callable[[Path], dict[str, Any]]
    evaluate: Callable[[Path, Any, dict[str, Any]], dict[str, Any]]


@dataclass
class TaskResult:
    key: str
    title: str
    repetition: int = 0
    status: str = "not_run"
    passed: bool = False
    answer: str = ""
    stop_reason: str = ""
    error: str = ""
    duration_ms: float = 0.0
    steps: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    checks: dict[str, Any] = field(default_factory=dict)
    tools_used: list[str] = field(default_factory=list)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    artifacts: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "repetition": self.repetition,
            "status": self.status,
            "passed": self.passed,
            "answer": self.answer,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "checks": self.checks,
            "tools_used": self.tools_used,
            "tool_trace": self.tool_trace,
            "artifacts": self.artifacts,
        }


# ---------------------------------------------------------------------------
# Fixtures and evaluation
# ---------------------------------------------------------------------------


def _tree_digest(root: Path) -> dict[str, str]:
    digest: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _prepare_web(root: Path) -> dict[str, Any]:
    return {}


def _evaluate_web(root: Path, result: Any, state: dict[str, Any]) -> dict[str, Any]:
    answer = (result.final_answer or result.proposal or "").casefold()
    tools = {item.call.name for step in result.steps for item in step.results if item.executed}
    checks = {
        "answer_mentions_example_domain": "example domain" in answer,
        "answer_mentions_paragraph": (
            "illustrative examples" in answer
            or "documentation examples" in answer
            or "for use in" in answer
        ),
        "used_web_extract": "web_extract" in tools,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _prepare_code(root: Path) -> dict[str, Any]:
    package = root / "demo"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    source = (
        "def divide(a, b):\n"
        "    if b == 0:\n"
        "        raise ValueError('division by zero')\n"
        "    return a * b\n"
        "\n"
        "def average(values):\n"
        "    if not values:\n"
        "        raise ValueError('empty values')\n"
        "    return divide(sum(values), len(values))\n"
    )
    (package / "calc.py").write_text(source, encoding="utf-8")
    tests = (
        "import unittest\n"
        "\n"
        "from demo.calc import average, divide\n"
        "\n"
        "\n"
        "class CalcTests(unittest.TestCase):\n"
        "    def test_divide(self):\n"
        "        self.assertEqual(divide(6, 2), 3)\n"
        "\n"
        "    def test_average(self):\n"
        "        self.assertEqual(average([2, 4, 6]), 4)\n"
        "\n"
        "    def test_divide_by_zero(self):\n"
        "        with self.assertRaises(ValueError):\n"
        "            divide(1, 0)\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )
    (package / "test_calc.py").write_text(tests, encoding="utf-8")
    return {
        "test_hash": hashlib.sha256((package / "test_calc.py").read_bytes()).hexdigest(),
    }


def _evaluate_code(root: Path, result: Any, state: dict[str, Any]) -> dict[str, Any]:
    test_file = root / "demo" / "test_calc.py"
    test_hash = hashlib.sha256(test_file.read_bytes()).hexdigest()
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "demo", "-t", ".", "-v"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=60,
    )
    tools = {item.call.name for step in result.steps for item in step.results if item.executed}
    checks = {
        "tests_pass": completed.returncode == 0,
        "tests_unchanged": test_hash == state["test_hash"],
        "source_changed": (root / "demo" / "calc.py").read_text(encoding="utf-8")
        != (
            "def divide(a, b):\n"
            "    if b == 0:\n"
            "        raise ValueError('division by zero')\n"
            "    return a * b\n"
            "\n"
            "def average(values):\n"
            "    if not values:\n"
            "        raise ValueError('empty values')\n"
            "    return divide(sum(values), len(values))\n"
        ),
        "ran_tests": bool({"terminal", "execute_code"} & tools),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "test_output_tail": (completed.stdout + completed.stderr)[-1500:],
    }


def _prepare_audit(root: Path) -> dict[str, Any]:
    audit = root / "audit"
    audit.mkdir(parents=True)
    (audit / "alpha.py").write_text("# TODO: remove legacy path\nprint('ok')\n", encoding="utf-8")
    (audit / "beta.py").write_text("config = 'deprecated'\n# TODO: update docs\n", encoding="utf-8")
    (audit / "notes.md").write_text("This module is deprecated.\n", encoding="utf-8")
    (audit / "data.log").write_text("x" * 4000 + "\n", encoding="utf-8")
    (audit / "small.txt").write_text("hello\n", encoding="utf-8")
    return {"digest": _tree_digest(root)}


def _evaluate_audit(root: Path, result: Any, state: dict[str, Any]) -> dict[str, Any]:
    answer = (result.final_answer or result.proposal or "").casefold()
    after = _tree_digest(root)
    checks = {
        "todo_count_2": "2" in answer,
        "largest_data_log": "data.log" in answer,
        "deprecated_beta": "beta.py" in answer,
        "deprecated_notes": "notes.md" in answer,
        "read_only": after == state["digest"],
    }
    return {"passed": all(checks.values()), "checks": checks}


TASKS: list[TaskSpec] = [
    TaskSpec(
        key="web",
        title="Web scraping",
        prompt=(
            "Use web_extract on https://example.com. Report the page's H1 "
            "heading and the first sentence of its only paragraph. Use the "
            "tool; do not answer from prior knowledge."
        ),
        prepare=_prepare_web,
        evaluate=_evaluate_web,
    ),
    TaskSpec(
        key="code",
        title="Codebase repair",
        prompt=(
            "The demo codebase has a bug. Read demo/calc.py and its test file, "
            "fix the source bug without editing any test, then run "
            "`python3 -m unittest discover -s demo -t . -v`. Report the test "
            "result."
        ),
        prepare=_prepare_code,
        evaluate=_evaluate_code,
    ),
    TaskSpec(
        key="audit",
        title="Read-only filesystem audit",
        prompt=(
            "Perform a read-only audit of the audit/ directory. Report: "
            "(a) the number of TODO occurrences, (b) the largest file by "
            "bytes, and (c) every file containing the word deprecated. Do not "
            "modify, create, or delete any file."
        ),
        prepare=_prepare_audit,
        evaluate=_evaluate_audit,
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_task(
    spec: TaskSpec,
    provider: Any,
    budget: Budget,
    *,
    max_steps: int,
    repetition: int = 0,
    artifact_dir: Path | None = None,
    keep_artifacts: bool = False,
) -> TaskResult:
    tempdir = tempfile.TemporaryDirectory(prefix="sora-live-{}-".format(spec.key))
    root = Path(tempdir.name)
    state = spec.prepare(root)
    result = TaskResult(key=spec.key, title=spec.title, repetition=repetition)
    started = time.perf_counter()
    before = (
        budget.prompt_tokens,
        budget.completion_tokens,
        budget.cached_tokens,
    )
    try:
        config = ToolPackConfig(
            root=root,
            restrict_to_root=True,
            max_terminal_output=20_000,
            terminal_timeout=60.0,
            code_timeout=60.0,
            web_timeout=25.0,
        )
        registry = ToolRegistry()
        register_core_tools(registry, config)
        guarded = BudgetProvider(provider, budget, spec.key)
        agent = Agent(
            provider=guarded,
            registry=registry,
            approval=AutoApprovalGate(),
            memory=False,
            identity=False,
            clock=False,
            calendar=False,
            max_steps=max_steps,
            max_tool_result_chars=8000,
            tool_timeout=60.0,
            provider_retries=1,
            delegate=False,
        )
        run = agent.run(spec.prompt)
        result.answer = run.final_answer or run.proposal
        result.stop_reason = run.stop_reason
        result.steps = len(run.steps)
        result.tool_calls = sum(len(step.results) for step in run.steps)
        result.tools_used = sorted(
            {item.call.name for step in run.steps for item in step.results if item.executed}
        )
        result.tool_trace = [
            {
                "step": getattr(step, "index", -1),
                "tool": item.call.name,
                "arguments": dict(item.call.arguments or {}),
                "executed": bool(item.executed),
                "is_error": bool(item.is_error),
                "approval_required": bool(item.approval_required),
                "approved": bool(item.approved),
                "skipped": bool(item.skipped),
                "content_preview": str(item.content or "")[:400],
            }
            for step in run.steps
            for item in step.results
        ]
        evaluation = spec.evaluate(root, run, state)
        result.checks = evaluation.get("checks", {})
        result.passed = bool(evaluation.get("passed"))
        result.status = "passed" if result.passed else "failed"
        if evaluation.get("test_output_tail"):
            result.checks["test_output_tail"] = evaluation["test_output_tail"]
    except BudgetExceeded as exc:
        result.status = BUDGET_EXCEEDED
        result.error = str(exc)
    except Exception as exc:  # noqa: BLE001 - one task must not kill the rest
        result.status = "error"
        result.error = "{}: {}".format(type(exc).__name__, exc)
    finally:
        result.duration_ms = (time.perf_counter() - started) * 1000.0
        result.prompt_tokens = budget.prompt_tokens - before[0]
        result.completion_tokens = budget.completion_tokens - before[1]
        result.cached_tokens = budget.cached_tokens - before[2]
        if artifact_dir is not None and (keep_artifacts or not result.passed):
            try:
                if artifact_dir.exists():
                    shutil.rmtree(artifact_dir)
                artifact_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    root,
                    artifact_dir,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
                )
                result.artifacts = str(artifact_dir)
            except Exception as exc:  # noqa: BLE001 - artifact copy is best effort
                result.checks["artifact_copy_error"] = str(exc)
        tempdir.cleanup()
    return result


def aggregate_results(results: list[TaskResult]) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for item in results:
        entry = aggregates.setdefault(
            item.key,
            {
                "key": item.key,
                "title": item.title,
                "runs": 0,
                "passed": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "duration_ms": 0.0,
                "tool_calls": 0,
            },
        )
        entry["runs"] += 1
        entry["passed"] += 1 if item.passed else 0
        entry["prompt_tokens"] += item.prompt_tokens
        entry["completion_tokens"] += item.completion_tokens
        entry["cached_tokens"] += item.cached_tokens
        entry["duration_ms"] += item.duration_ms
        entry["tool_calls"] += item.tool_calls
    for entry in aggregates.values():
        runs = max(1, entry["runs"])
        entry["pass_rate"] = entry["passed"] / float(runs)
        entry["total_tokens"] = entry["prompt_tokens"] + entry["completion_tokens"]
        entry["mean_tokens"] = entry["total_tokens"] / float(runs)
        entry["mean_duration_ms"] = entry["duration_ms"] / float(runs)
        entry["cache_hit_rate"] = (
            entry["cached_tokens"] / float(entry["prompt_tokens"])
            if entry["prompt_tokens"]
            else 0.0
        )
    return aggregates


def write_report(
    run_dir: Path,
    *,
    provider: str,
    model: str,
    budget: Budget,
    results: list[TaskResult],
    started: str,
    run_start: tuple[int, int, int],
    prior_used: int,
) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    aggregates = aggregate_results(results)
    run_prompt = budget.prompt_tokens - run_start[0]
    run_completion = budget.completion_tokens - run_start[1]
    run_cached = budget.cached_tokens - run_start[2]
    payload = {
        "mode": "live",
        "provider": provider,
        "model": model,
        "started_at": started,
        "budget_tokens": budget.limit,
        "used_tokens": budget.used,
        "run_used_tokens": budget.used - prior_used,
        "prior_used_tokens": prior_used,
        "remaining_tokens": budget.remaining,
        "prompt_tokens": budget.prompt_tokens,
        "completion_tokens": budget.completion_tokens,
        "cached_tokens": budget.cached_tokens,
        "run_prompt_tokens": run_prompt,
        "run_completion_tokens": run_completion,
        "run_cached_tokens": run_cached,
        "run_cache_hit_rate": (run_cached / float(run_prompt) if run_prompt else 0.0),
        "provider_calls": budget.calls,
        "aggregates": aggregates,
        "tasks": [item.to_dict() for item in results],
    }
    report_path = run_dir / "live_report.json"
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# SoraAgent live three-task benchmark",
        "",
        "- Provider: `{}`".format(provider),
        "- Model: `{}`".format(model),
        "- Token budget: **{:,}**".format(budget.limit),
        "- Tokens used before this run: **{:,}**".format(prior_used),
        "- Tokens used in this run: **{:,}**".format(budget.used - prior_used),
        "- Tokens used in total: **{:,}**".format(budget.used),
        "- Remaining: **{:,}**".format(budget.remaining),
        "- Provider calls: **{}**".format(budget.calls),
        "",
        "## Aggregate",
        "",
        "| Task | Runs | Passed | Pass rate | Prompt | Completion | Cached | Avg tokens | Avg time |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in aggregates.values():
        lines.append(
            "| {} | {} | {} | {:.0%} | {:,} | {:,} | {:,} | {:,.0f} | {:.1f}s |".format(
                entry["title"],
                entry["runs"],
                entry["passed"],
                entry["pass_rate"],
                entry["prompt_tokens"],
                entry["completion_tokens"],
                entry["cached_tokens"],
                entry["mean_tokens"],
                entry["mean_duration_ms"] / 1000.0,
            )
        )
    lines += [
        "",
        "## Runs",
        "",
        "| Task | Rep | Status | Prompt | Completion | Cached | Time | Tools |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in results:
        lines.append(
            "| {} | {} | {} | {:,} | {:,} | {:,} | {:.1f}s | {} |".format(
                item.title,
                item.repetition,
                item.status,
                item.prompt_tokens,
                item.completion_tokens,
                item.cached_tokens,
                item.duration_ms / 1000.0,
                ", ".join(item.tools_used) or "—",
            )
        )
    failures = [item for item in results if not item.passed]
    if failures:
        lines += ["", "## Failures", ""]
        for item in failures:
            lines.append("### {} (rep {})".format(item.title, item.repetition))
            lines.append("")
            lines.append("**Status:** `{}`".format(item.status))
            if item.error:
                lines.append("")
                lines.append("**Error:** `{}`".format(item.error))
            if item.answer:
                lines.append("")
                lines.append("**Answer:**")
                lines.append("")
                lines.append("```text")
                lines.append(item.answer[:3000])
                lines.append("```")
            if item.checks:
                lines.append("")
                for key, value in item.checks.items():
                    if key == "test_output_tail":
                        continue
                    lines.append("- `{}`: {}".format(key, value))
            if item.artifacts:
                lines.append("")
                lines.append("Artifacts: `{}`".format(item.artifacts))
            lines.append("")
    # One representative passing answer per task keeps the report readable.
    seen: set[str] = set()
    representatives = [item for item in results if item.passed and item.key not in seen]
    if representatives:
        lines += ["", "## Representative passing answers", ""]
        for item in representatives:
            seen.add(item.key)
            lines.append("### {} (rep {})".format(item.title, item.repetition))
            lines.append("")
            lines.append("```text")
            lines.append(item.answer[:2500])
            lines.append("```")
            lines.append("")
    summary_path = run_dir / "live_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": report_path, "markdown": summary_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--budget-tokens", type=int, default=10_000_000)
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Maximum full cycles of the selected tasks",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=0,
        help="Run full cycles until this many new tokens are spent (0 = off)",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Copy every task workspace into the report directory",
    )
    parser.add_argument("--out", default="results/live-three-tasks")
    parser.add_argument(
        "--budget-state",
        default="results/live-three-tasks/budget.json",
        help="Persistent cumulative budget ledger",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=[spec.key for spec in TASKS],
        choices=[spec.key for spec in TASKS],
    )
    args = parser.parse_args(argv)
    key_name = {
        "deepseek": "DEEPSEEK_API_KEY",
        "ds": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "azure": "OPENAI_API_KEY",
        "qwen": "DASHSCOPE_API_KEY",
        "dashscope": "DASHSCOPE_API_KEY",
    }.get(args.provider.lower(), "")
    if not args.api_key and not os.getenv(key_name, "").strip():
        try:
            saved = load_secrets()
        except Exception:  # noqa: BLE001 - malformed saved secrets are reported below
            saved = {}
        args.api_key = saved.get(key_name, "").strip()
    if args.budget_tokens <= 0:
        print("error: --budget-tokens must be positive", file=sys.stderr)
        return 2
    if args.repetitions <= 0:
        print("error: --repetitions must be positive", file=sys.stderr)
        return 2
    if args.target_tokens < 0:
        print("error: --target-tokens must be >= 0", file=sys.stderr)
        return 2
    try:
        provider = build_live_provider(
            args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            timeout=120.0,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 - configuration error is user-facing
        print("error: {}".format(exc), file=sys.stderr)
        print(
            "Set DEEPSEEK_API_KEY (or pass --api-key) before running.",
            file=sys.stderr,
        )
        return 2
    state_path = Path(args.budget_state)
    prior = load_budget_state(state_path)
    prior_used = prior.get("prompt_tokens", 0) + prior.get("completion_tokens", 0)
    if prior_used >= args.budget_tokens:
        print(
            "error: cumulative token budget already exhausted: {:,} / {:,}".format(
                prior_used, args.budget_tokens
            ),
            file=sys.stderr,
        )
        return 2
    budget = Budget(
        limit=args.budget_tokens,
        prompt_tokens=prior.get("prompt_tokens", 0),
        completion_tokens=prior.get("completion_tokens", 0),
        cached_tokens=prior.get("cached_tokens", 0),
        calls=prior.get("calls", 0),
    )
    run_start = (
        budget.prompt_tokens,
        budget.completion_tokens,
        budget.cached_tokens,
    )
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    selected = [spec for spec in TASKS if spec.key in set(args.tasks)]
    run_dir = Path(args.out) / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = run_dir / "artifacts"
    results: list[TaskResult] = []
    stopped = False
    for repetition in range(1, args.repetitions + 1):
        for spec in selected:
            print(
                "[live] {} rep {} ... budget {:,}/{:,}".format(
                    spec.title, repetition, budget.used, budget.limit
                ),
                flush=True,
            )
            artifact_dir = artifact_root / "{}-r{}".format(spec.key, repetition)
            result = run_task(
                spec,
                provider,
                budget,
                max_steps=args.max_steps,
                repetition=repetition,
                artifact_dir=artifact_dir,
                keep_artifacts=args.keep_artifacts,
            )
            results.append(result)
            print(
                "[live] {} rep {} -> {} ({:,} run tokens)".format(
                    spec.title,
                    repetition,
                    result.status,
                    result.prompt_tokens + result.completion_tokens,
                ),
                flush=True,
            )
            if budget.used >= budget.limit:
                stopped = True
                break
        new_tokens = budget.used - prior_used
        if stopped:
            break
        if args.target_tokens and new_tokens >= args.target_tokens:
            break
    paths = write_report(
        run_dir,
        provider=args.provider,
        model=args.model,
        budget=budget,
        results=results,
        started=started,
        run_start=run_start,
        prior_used=prior_used,
    )
    save_budget_state(state_path, budget)
    print("")
    print(
        "Tokens: {:,} / {:,} (this run {:,}, remaining {:,})".format(
            budget.used,
            budget.limit,
            budget.used - prior_used,
            budget.remaining,
        )
    )
    print("Report: {}".format(paths["markdown"]))
    print("JSON:   {}".format(paths["json"]))
    return 0 if all(item.passed for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
