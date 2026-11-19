"""Command-line interface for the SoraAgent benchmark."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .config import SUITES, TARGETS, BenchConfig
from .harness import BenchmarkHarness
from .report import write_reports
from .scenarios import scenarios_for_suite
from .suites import run_suite
from .types import BenchmarkReport, SuiteResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sora-bench",
        description=(
            "Benchmark SoraAgent on tokens, tool use, memory, dreaming, "
            "speed, autonomy, and prompt caching."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("simulated", "live"),
        default="simulated",
        help="Simulated uses a deterministic reference policy; live calls a model.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Live provider kind: openai, deepseek, qwen, custom.",
    )
    parser.add_argument("--model", default="", help="Model id for live mode.")
    parser.add_argument("--api-key", default="", help="API key (prefer env vars).")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL.")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Repetitions per scenario (caching forces at least two in simulated mode).",
    )
    parser.add_argument("--seed", type=int, default=1729, help="Deterministic seed.")
    parser.add_argument(
        "--suites",
        nargs="*",
        default=["all"],
        help="Subset of: " + ", ".join(SUITES) + " (or all).",
    )
    parser.add_argument(
        "--out",
        default="results",
        help="Output directory for timestamped reports.",
    )
    parser.add_argument("--max-steps", type=int, default=16, help="Agent step budget per run.")
    parser.add_argument(
        "--min-cache-block",
        type=int,
        default=256,
        help="Minimum prefix size treated as a cacheable block.",
    )
    parser.add_argument(
        "--latency-ms",
        type=float,
        default=0.0,
        help="Simulated provider delay per call (speed suite defaults to 25 ms).",
    )
    parser.add_argument(
        "--jitter-ms",
        type=float,
        default=0.0,
        help="Additional deterministic jitter for simulated latency.",
    )
    parser.add_argument(
        "--autonomy-policy",
        choices=("minimal", "balanced", "overreach"),
        default="balanced",
        help="Simulated policy: requested only, helpful extras, or harmful overreach.",
    )
    parser.add_argument(
        "--cache-ablation",
        dest="cache_ablation",
        action="store_true",
        default=True,
        help="Run cache-breaker ablations in simulated mode.",
    )
    parser.add_argument(
        "--no-cache-ablation",
        dest="cache_ablation",
        action="store_false",
        help="Skip cache-breaker ablations.",
    )
    parser.add_argument(
        "--include-requests",
        action="store_true",
        help="Include every provider request in report.json and runs.jsonl.",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep per-run temporary directories for debugging.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="Live provider timeout (seconds)."
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Live sampling temperature.")
    parser.add_argument(
        "--format",
        nargs="*",
        default=["json", "markdown", "csv", "jsonl"],
        choices=("json", "markdown", "csv", "jsonl"),
        help="Output formats.",
    )
    parser.add_argument("--label", default="", help="Free-form label stored in the report.")
    parser.add_argument("--list-suites", action="store_true", help="List suite names and exit.")
    parser.add_argument("--list-scenarios", action="store_true", help="List scenarios and exit.")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.list_suites:
        for name in SUITES:
            print(name)
        return 0
    if args.list_scenarios:
        for suite in SUITES:
            try:
                scenarios = scenarios_for_suite(suite)
            except KeyError:
                continue
            for scenario in scenarios:
                print("{}\t{}\t{}".format(suite, scenario.id, scenario.description))
        return 0

    provider = args.provider
    if not provider:
        provider = "simulated" if args.mode == "simulated" else "openai"
    if args.mode == "simulated" and provider != "simulated":
        print(
            "warning: --mode simulated ignores --provider and uses the reference policy",
            file=sys.stderr,
        )
        provider = "simulated"

    try:
        config = BenchConfig(
            mode=args.mode,
            provider_kind=provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            runs=args.runs,
            seed=args.seed,
            suites=tuple(args.suites),
            output_dir=Path(args.out),
            max_steps=args.max_steps,
            min_cache_block=args.min_cache_block,
            simulated_latency_ms=args.latency_ms,
            latency_jitter_ms=args.jitter_ms,
            autonomy_policy=args.autonomy_policy,
            cache_ablation=args.cache_ablation,
            include_requests=args.include_requests,
            keep_workdir=args.keep_workdir,
            verbose=args.verbose,
            timeout=args.timeout,
            temperature=args.temperature,
            output_formats=tuple(args.format),
            label=args.label,
        )
    except ValueError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2

    started = time.perf_counter()
    harness = BenchmarkHarness(config)
    report = BenchmarkReport(
        mode=config.mode,
        provider=config.provider_kind,
        model=config.model or ("reference" if config.simulated else "default"),
        seed=config.seed,
    )
    report.diagnostics = {
        "config": config.to_dict(),
        "targets": dict(TARGETS),
        "label": config.label,
        "sora_bench_version": __version__,
        "cache_ablation": config.cache_ablation if config.simulated else False,
    }
    exit_code = 0
    try:
        for name in config.suites:
            print("[bench] running {} ...".format(name), flush=True)
            suite_started = time.perf_counter()
            try:
                suite = run_suite(harness, name)
            except Exception as exc:  # noqa: BLE001 - keep other suites running
                from sora_agent.providers import ProviderConfigError

                if isinstance(exc, ProviderConfigError):
                    logging.warning("suite %s cannot start: %s", name, exc)
                else:
                    logging.exception("suite %s failed", name)
                suite = SuiteResult(
                    name=name,
                    error="{}: {}".format(type(exc).__name__, exc),
                )
                exit_code = 1
            suite.duration_ms = (time.perf_counter() - suite_started) * 1000.0
            report.suites[name] = suite
            report.runs.extend(suite.runs)
            if suite.error:
                report.errors.append("{}: {}".format(name, suite.error))
            print(
                "[bench] {} done in {:.2f}s (score: {})".format(
                    name,
                    suite.duration_ms / 1000.0,
                    "{:.1f}".format(suite.score) if suite.score is not None else "n/a",
                ),
                flush=True,
            )
    finally:
        harness.close()
    report.duration_ms = (time.perf_counter() - started) * 1000.0
    paths = write_reports(
        report,
        output_dir=config.output_dir,
        include_requests=config.include_requests,
        formats=config.output_formats,
    )
    summary = report.summary
    print("")
    print(
        "Overall: {:.1f}/100 ({})".format(
            summary.get("overall_score", 0.0), summary.get("grade", "F")
        )
    )
    if paths.get("markdown"):
        print("Summary: {}".format(paths["markdown"]))
    if paths.get("json"):
        print("JSON:    {}".format(paths["json"]))
    if paths.get("csv"):
        print("CSV:     {}".format(paths["csv"]))
    if paths.get("jsonl"):
        print("Runs:    {}".format(paths["jsonl"]))
    if report.errors:
        print("Errors:", file=sys.stderr)
        for error in report.errors:
            print("  - {}".format(error), file=sys.stderr)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
