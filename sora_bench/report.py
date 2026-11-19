"""Report aggregation and output writers."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import SUITE_WEIGHTS
from .types import BenchmarkReport, Metric, SuiteResult, utcnow_iso

SUITE_LABELS = {
    "tokens": "Token usage",
    "tools": "Tool-use accuracy",
    "memory": "Memory retention",
    "dreaming": "Dreaming mechanism",
    "speed": "Response speed",
    "autonomy": "Autonomous task completion",
    "caching": "Prompt caching",
    "features": "Files, specialists, memory, graph & security",
}

HEADLINE_KEYS = {
    "tokens": ["token_total_mean", "token_efficiency"],
    "tools": ["tool_f1", "tool_exact_trajectory"],
    "memory": [
        "memory_retrieval_hit_rate",
        "memory_answer_accuracy",
        "memory_facts_per_1k_tokens",
    ],
    "dreaming": [
        "dream_merge_f1",
        "dream_conflict_f1",
        "dream_duration_per_100_memories",
    ],
    "speed": ["speed_e2e_p95", "speed_framework_overhead_mean"],
    "autonomy": [
        "autonomy_net_utility",
        "autonomy_harmful_execution_rate",
    ],
    "caching": [
        "cache_effective_hit_rate",
        "cache_stability",
        "cache_message_stability",
        "cache_stable_prefix_tokens",
    ],
    "features": ["features_edge_pass_rate"],
}


def build_summary(report: BenchmarkReport) -> Dict[str, Any]:
    suite_scores: Dict[str, Optional[float]] = {}
    headline: Dict[str, Any] = {}
    for name, suite in report.suites.items():
        suite_scores[name] = round(float(suite.score), 2) if suite.score is not None else None
        metrics = {item.key: item for item in suite.metrics}
        for key in HEADLINE_KEYS.get(name, []):
            item = metrics.get(key)
            if item is not None:
                headline[key] = {
                    "label": item.label,
                    "value": _round(item.value),
                    "unit": item.unit,
                    "score": _round(item.score) if item.score is not None else None,
                }
    weighted = 0.0
    weights = 0.0
    for name, score in suite_scores.items():
        if score is None:
            continue
        weight = SUITE_WEIGHTS.get(name, 1.0 / 8.0)
        weighted += score * weight
        weights += weight
    overall = weighted / weights if weights else 0.0
    return {
        "overall_score": round(overall, 2),
        "grade": _grade(overall),
        "suite_scores": suite_scores,
        "headline": headline,
        "run_count": len(report.runs),
        "suite_count": len(report.suites),
        "mode": report.mode,
        "provider": report.provider,
        "model": report.model,
        "errors": list(report.errors),
    }


def write_reports(
    report: BenchmarkReport,
    *,
    output_dir: Path,
    include_requests: bool = False,
    formats: Iterable[str] = ("json", "markdown", "csv", "jsonl"),
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    report.summary = build_summary(report)
    report.finished_at = utcnow_iso()
    formats = set(formats)
    paths: Dict[str, Path] = {}

    if "json" in formats:
        path = run_dir / "report.json"
        _write_json(path, report.to_dict(include_requests=include_requests))
        paths["json"] = path
        _copy(path, output_dir / "latest.json")
        paths["json_latest"] = output_dir / "latest.json"

    if "markdown" in formats:
        path = run_dir / "summary.md"
        path.write_text(render_markdown(report), encoding="utf-8")
        paths["markdown"] = path
        _copy(path, output_dir / "latest.md")
        paths["markdown_latest"] = output_dir / "latest.md"

    if "csv" in formats:
        path = run_dir / "metrics.csv"
        _write_metrics_csv(path, report)
        paths["csv"] = path

    if "jsonl" in formats:
        path = run_dir / "runs.jsonl"
        _write_runs_jsonl(path, report, include_requests=include_requests)
        paths["jsonl"] = path

    if include_requests:
        path = run_dir / "cache_requests.jsonl"
        _write_cache_requests(path, report)
        paths["cache_requests"] = path

    return paths


def render_markdown(report: BenchmarkReport) -> str:
    summary = report.summary or build_summary(report)
    lines: List[str] = []
    lines.append("# SoraAgent Benchmark")
    lines.append("")
    environment = report.environment()
    lines.append(
        "**Mode:** {mode} &nbsp;|&nbsp; **Provider:** {provider} "
        "&nbsp;|&nbsp; **Model:** {model}".format(
            mode=report.mode,
            provider=report.provider,
            model=report.model or "reference policy",
        )
    )
    lines.append(
        "**SoraAgent:** {version} &nbsp;|&nbsp; **Python:** {python} "
        "&nbsp;|&nbsp; **Seed:** {seed}".format(
            version=environment.get("sora_agent_version", "unknown"),
            python=environment.get("python", "unknown"),
            seed=report.seed,
        )
    )
    lines.append(
        "**Generated:** {time} &nbsp;|&nbsp; **Duration:** {duration:.2f} s".format(
            time=report.finished_at or utcnow_iso(),
            duration=report.duration_ms / 1000.0,
        )
    )
    lines.append("")
    lines.append(
        "## Overall: {score:.1f} / 100 ({grade})".format(
            score=summary.get("overall_score", 0.0),
            grade=summary.get("grade", "F"),
        )
    )
    lines.append("")
    lines.append("| Suite | Score | Runs | Status |")
    lines.append("| --- | ---: | ---: | --- |")
    for name, suite in report.suites.items():
        status = "error" if suite.error else "ok"
        score = "{:.1f}".format(suite.score) if suite.score is not None else "—"
        run_display = len(suite.runs) if suite.runs else "—"
        lines.append(
            "| {} | {} | {} | {} |".format(
                SUITE_LABELS.get(name, name),
                score,
                run_display,
                status,
            )
        )
    lines.append("")
    if summary.get("headline"):
        lines.append("## Headline metrics")
        lines.append("")
        lines.append("| Metric | Value | Score |")
        lines.append("| --- | ---: | ---: |")
        for key, item in summary["headline"].items():
            lines.append(
                "| {} | {} | {} |".format(
                    item["label"],
                    _format_value(item["value"], item["unit"]),
                    _format_score(item.get("score")),
                )
            )
        lines.append("")

    for name, suite in report.suites.items():
        lines.append("## {}".format(SUITE_LABELS.get(name, name)))
        lines.append("")
        if suite.error:
            lines.append("**Error:** `{}`".format(suite.error))
            lines.append("")
            continue
        lines.append("| Metric | Value | Target/Score | Interpretation |")
        lines.append("| --- | ---: | ---: | --- |")
        for item in suite.metrics:
            lines.append(
                "| {} | {} | {} | {} |".format(
                    item.label,
                    _format_value(item.value, item.unit),
                    _format_score(item.score),
                    _interpretation(item),
                )
            )
        lines.append("")
        extra = _suite_diagnostics_markdown(name, suite)
        if extra:
            lines.extend(extra)
            lines.append("")
    if report.errors:
        lines.append("## Errors")
        lines.append("")
        for error in report.errors:
            lines.append("- {}".format(error))
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "Scores are heuristic normalizations against explicit targets; inspect "
        "the raw metrics and diagnostics before drawing conclusions.  Simulated "
        "mode is a deterministic reference policy, not a model ranking."
    )
    lines.append("")
    return "\n".join(lines)


def _suite_diagnostics_markdown(name: str, suite: SuiteResult) -> List[str]:
    diagnostics = suite.diagnostics or {}
    lines: List[str] = []
    if name == "caching" and diagnostics.get("variants"):
        lines.append("### Cache ablation")
        lines.append("")
        lines.append("| Variant | Hit rate | Stability | Cached tokens | Prompt tokens |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for variant, item in diagnostics["variants"].items():
            lines.append(
                "| {} | {:.1%} | {:.1%} | {} | {} |".format(
                    variant,
                    item.get("cache_hit_rate", 0.0),
                    item.get("cache_stability", 0.0),
                    item.get("cached_tokens", 0),
                    item.get("prompt_tokens", 0),
                )
            )
        lines.append("")
        breakers = diagnostics.get("cache_breakers") or {}
        if breakers.get("recommendation"):
            lines.append("**Cache diagnosis:** {}".format(breakers["recommendation"]))
            lines.append("")
    if name == "memory" and diagnostics.get("queries"):
        lines.append("### Memory queries")
        lines.append("")
        lines.append("| Query | Hit | Precision | Tokens | MRR |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for item in diagnostics["queries"]:
            lines.append(
                "| {} | {} | {:.1%} | {} | {:.2f} |".format(
                    item.get("query_id", ""),
                    "yes"
                    if item.get("expected_hit")
                    else ("n/a" if item.get("no_answer") else "no"),
                    item.get("precision", 0.0),
                    item.get("recall_tokens", 0),
                    item.get("mrr", 0.0),
                )
            )
        lines.append("")
    if name == "autonomy" and diagnostics.get("actions"):
        lines.append("### Extra actions")
        lines.append("")
        lines.append("| Scenario | Tool | Label | Weight | Executed | Rationale |")
        lines.append("| --- | --- | --- | ---: | --- | --- |")
        for item in diagnostics["actions"]:
            lines.append(
                "| {} | {} | {} | {:+.2f} | {} | {} |".format(
                    item.get("scenario_id", ""),
                    item.get("tool", ""),
                    item.get("label", ""),
                    item.get("weight", 0.0),
                    "yes" if item.get("executed") else "no",
                    item.get("rationale", "")[:120],
                )
            )
        lines.append("")
    if name == "dreaming" and diagnostics.get("observed"):
        observed = diagnostics["observed"]
        expected = diagnostics.get("expected", {})
        lines.append("### Dream expectations vs. observed")
        lines.append("")
        lines.append(
            "- Merge groups: observed {} / expected {}".format(
                len(observed.get("merge_groups") or []),
                len(expected.get("merge_groups") or []),
            )
        )
        lines.append(
            "- Conflict pairs: observed {} / expected {}".format(
                len(observed.get("conflict_pairs") or []),
                len(expected.get("conflict_pairs") or []),
            )
        )
        lines.append(
            "- Inferred pairs: observed {} / expected {}".format(
                len(observed.get("inference_pairs") or []),
                len(expected.get("inference_pairs") or []),
            )
        )
        lines.append(
            "- Source retention: {} / {}".format(
                observed.get("retained", 0),
                observed.get("retained_expected", 0),
            )
        )
        lines.append("")
    return lines


def _format_value(value: Any, unit: str) -> str:
    if value is None:
        return "—"
    if unit == "ratio":
        return "{:.1%}".format(float(value))
    if unit == "bool":
        return "yes" if value else "no"
    if unit == "ms":
        if float(value) >= 1000:
            return "{:.2f} s".format(float(value) / 1000.0)
        return "{:.1f} ms".format(float(value))
    if unit == "seconds":
        return "{:.3f} s".format(float(value))
    if unit in {"tokens", "messages", "bytes"}:
        return "{:,.0f} {}".format(float(value), unit)
    if unit in {"calls", "actions", "breaks"}:
        return "{:,.2f} {}".format(float(value), unit)
    if unit in {"tasks/1k", "facts/1k", "tok/s", "tasks/s", "mem/s", "x"}:
        return "{:.2f} {}".format(float(value), unit)
    if unit in {"ms/step", "ms/call"}:
        return "{:.1f} {}".format(float(value), unit)
    if unit in {"utility/run", "actions/run"}:
        return "{:+.2f} {}".format(float(value), unit)
    if unit == "score":
        return "{:.1f}".format(float(value))
    if isinstance(value, float):
        return "{:.3f}".format(value)
    return str(value)


def _format_score(score: Optional[float]) -> str:
    if score is None:
        return "—"
    return "{:.1f} / 100".format(float(score))


def _interpretation(item: Metric) -> str:
    parts = []
    parts.append("higher is better" if item.higher_is_better else "lower is better")
    if item.detail:
        parts.append(item.detail)
    return "; ".join(parts)


def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), digits)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _copy(source: Path, target: Path) -> None:
    try:
        shutil.copyfile(source, target)
    except OSError:
        pass


def _write_metrics_csv(path: Path, report: BenchmarkReport) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "suite",
                "key",
                "label",
                "value",
                "unit",
                "higher_is_better",
                "score",
                "n",
                "detail",
            ]
        )
        for name, suite in report.suites.items():
            for item in suite.metrics:
                writer.writerow(
                    [
                        name,
                        item.key,
                        item.label,
                        item.value,
                        item.unit,
                        item.higher_is_better,
                        item.score if item.score is not None else "",
                        item.n,
                        item.detail,
                    ]
                )


def _write_runs_jsonl(path: Path, report: BenchmarkReport, *, include_requests: bool) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for run in report.runs:
            handle.write(
                json.dumps(
                    run.to_dict(include_requests=include_requests),
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )


def _write_cache_requests(path: Path, report: BenchmarkReport) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for run in report.runs:
            for request in run.requests:
                payload = request.to_dict()
                payload["scenario_id"] = run.scenario_id
                payload["suite"] = run.suite
                payload["run_index"] = run.run_index
                payload["variant"] = run.variant
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
