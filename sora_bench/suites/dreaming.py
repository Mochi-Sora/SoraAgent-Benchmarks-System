"""Dreaming (nightly consolidation) accuracy and efficiency suite."""

from __future__ import annotations

import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from sora_agent.memory.config import MemoryConfig
from sora_agent.memory.daily import DailyStore
from sora_agent.memory.dream import DreamRunner

from ..backend import FakeMemoryBackend
from ..config import TARGETS
from ..harness import BenchmarkHarness
from ..scoring import (
    safe_div,
    score_higher_is_better,
    score_lower_is_better,
    weighted_score,
)
from ..support import CountingWake, NoopBackup
from ..tokens import estimate_tokens
from ..types import SuiteResult, metric


def _memory(
    memory_id: str,
    content: str,
    *,
    importance: float = 0.9,
    created_at: str = "2026-11-01T00:00:00+00:00",
    expires_at: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "memory_id": memory_id,
        "content": content,
        "importance": importance,
        "metadata": {"sora_kind": "verbatim", "dream_round": 0},
        "created_at": created_at,
    }
    if expires_at:
        payload["expires_at"] = expires_at
    return payload


def fixture(now: datetime) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    seeds = [
        _memory(
            "dup-a",
            "Project ORCHID release owner is Alice and the launch date is 2026-12-03.",
        ),
        _memory(
            "dup-b",
            "Project ORCHID release owner is Alice and the launch date is 2026-12-03.",
        ),
        _memory(
            "dup-c",
            "Project ORCHID release owner is Alice and the launch date is 2026-12-03.",
        ),
        _memory(
            "dup-d",
            "The Northwind escalation owner is Priya Raman and the SLA is four hours.",
        ),
        _memory(
            "dup-e",
            "The Northwind escalation owner is Priya Raman and the SLA is four hours.",
        ),
        _memory(
            "conf-a",
            "The release date is 2026-11-01 and the owner is Alice.",
        ),
        _memory(
            "conf-b",
            "The release date is 2026-12-01 and the owner is Alice.",
        ),
        _memory(
            "same-a",
            "The APAC budget is 5000 and Dana is the approver.",
        ),
        _memory(
            "same-b",
            "The APAC budget is 5000 and Dana is the approver for Q3.",
        ),
        _memory("tri-a", "Project ORCHID release owner Alice."),
        _memory("tri-b", "Project ORCHID release QA Bob."),
        _memory("tri-c", "Project ORCHID owner Alice QA Bob."),
        _memory(
            "exp-a",
            "Temporary incident bridge code is INC-7781 and expires soon.",
            expires_at="2026-11-01T00:00:00+00:00",
            importance=0.4,
        ),
    ]
    expected = {
        "merge_groups": [
            frozenset({"dup-a", "dup-b", "dup-c"}),
            frozenset({"dup-d", "dup-e"}),
        ],
        "conflict_pairs": [frozenset({"conf-a", "conf-b"})],
        "inference_pairs": [frozenset({"tri-a", "tri-b"})],
        "expired": {"exp-a"},
        "retained": {item["memory_id"] for item in seeds if item["memory_id"] != "exp-a"},
        "source_count": len(seeds),
    }
    return seeds, expected


def _action_groups(report: Dict[str, Any], op: str, key: str = "sources") -> List[frozenset]:
    groups: List[frozenset] = []
    for action in report.get("actions") or []:
        if action.get("op") != op:
            continue
        values = action.get(key) or []
        if values:
            groups.append(frozenset(str(item) for item in values))
    return groups


def _set_prf(
    created: Sequence[frozenset], expected: Sequence[frozenset]
) -> Tuple[float, float, float]:
    created = list(created)
    expected = list(expected)
    matched_expected = 0
    matched_created = 0
    used_created: Set[int] = set()
    for expected_set in expected:
        for index, created_set in enumerate(created):
            if index in used_created:
                continue
            if created_set == expected_set:
                used_created.add(index)
                matched_expected += 1
                matched_created += 1
                break
    precision = safe_div(matched_created, len(created), 1.0 if not created else 0.0)
    recall = safe_div(matched_expected, len(expected), 1.0 if not expected else 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def run(harness: BenchmarkHarness) -> SuiteResult:
    started = time.perf_counter()
    result = SuiteResult(name="dreaming")
    now = datetime(2026, 11, 12, 12, 0, 0, tzinfo=timezone.utc)
    seeds, expected = fixture(now)
    tempdir = tempfile.TemporaryDirectory(prefix="sora-bench-dream-")
    root = Path(tempdir.name)
    config = MemoryConfig(
        root=root,
        auto_download=False,
        dream_auto=False,
        dream_rounds=2,
        dream_max_memories=500,
        dream_max_merges=20,
        dream_max_conflicts=20,
        dream_max_inferences=20,
        dream_max_updates=300,
        dream_edge_jaccard=0.45,
        dream_decay=0.98,
        dream_min_importance=0.05,
    )
    daily = DailyStore(config.daily_dir, clock=lambda: now)
    # Two transfer days, distinct from the consolidation fixture so they cannot
    # create extra merges or conflicts.
    daily.append(
        "user",
        "The cafeteria menu changes every Monday.",
        ts=now - timedelta(days=2),
    )
    daily.append(
        "assistant",
        "The shuttle bus leaves at 8:30.",
        ts=now - timedelta(days=2),
    )
    daily.append(
        "user",
        "The visitor badge pickup point is the west lobby.",
        ts=now - timedelta(days=1),
    )
    backend = FakeMemoryBackend(latency_ms=0.5)
    for payload in seeds:
        backend.save_memory(payload)
    backup = NoopBackup()
    wake = CountingWake()
    runner = DreamRunner(
        config,
        daily,
        backend,
        backup=backup,
        wake=wake,
        clock=lambda: now,
    )
    report = runner.run()
    backend_calls_first = backend.total_calls()
    backend_stats_first = backend.stats()
    snapshot_after_first = backend.snapshot()
    report_path = Path(str(report.get("report_path") or ""))
    report_bytes = report_path.stat().st_size if report_path.exists() else 0

    # Idempotency: rerunning at the same moment must not repeat destructive work.
    second_runner = DreamRunner(
        config,
        daily,
        backend,
        backup=backup,
        wake=wake,
        clock=lambda: now,
    )
    second_report = second_runner.run()
    idempotent = backend.snapshot() == snapshot_after_first
    destructive_second = [
        action
        for action in second_report.get("actions") or []
        if action.get("op") in {"merge", "conflict", "infer", "expire", "cleanup"}
    ]

    merge_groups = _action_groups(report, "merge", "sources")
    conflict_pairs = [
        frozenset(str(item) for item in item.get("sources") or [])
        for item in report.get("conflicts") or []
    ]
    inference_pairs = []
    for item in report.get("inferences") or []:
        sources = [str(value) for value in item.get("sources") or []]
        if len(sources) >= 2:
            inference_pairs.append(frozenset({sources[0], sources[-1]}))
    expired = {
        str(action.get("memory_id"))
        for action in report.get("actions") or []
        if action.get("op") == "expire"
    }
    merge_precision, merge_recall, merge_f1 = _set_prf(merge_groups, expected["merge_groups"])
    conflict_precision, conflict_recall, conflict_f1 = _set_prf(
        conflict_pairs, expected["conflict_pairs"]
    )
    inference_precision, inference_recall, inference_f1 = _set_prf(
        inference_pairs, expected["inference_pairs"]
    )

    retained_ok = 0
    retained_expected = 0
    for memory_id in expected["retained"]:
        retained_expected += 1
        record = backend.get_memory(memory_id)
        if record is None:
            continue
        original = next(item["content"] for item in seeds if item["memory_id"] == memory_id)
        if str(record.get("content") or "") == original:
            retained_ok += 1
    retention = safe_div(retained_ok, retained_expected)
    expired_ok = expected["expired"].issubset(expired)
    cleanup = report.get("artifacts_cleanup") or {}
    cleanup_rate = safe_div(cleanup.get("removed", 0), cleanup.get("planned", 0), 1.0)
    artifacts_left = [memory_id for memory_id in backend.memories if memory_id.startswith("dream:")]
    errors = list(report.get("errors") or []) + list(second_report.get("errors") or [])
    error_rate = safe_div(len(errors), 1 + len(seeds))
    transferred = sum(
        int(item.get("messages") or 0) for item in report.get("transferred_days") or []
    )

    elapsed = float(report.get("elapsed_seconds") or 0.0)
    throughput = safe_div(transferred + expected["source_count"], elapsed)
    calls_per_memory = safe_div(backend_calls_first, expected["source_count"])
    actions_per_memory = safe_div(len(report.get("actions") or []), expected["source_count"])
    duration_per_100 = elapsed * 100.0 / max(1, transferred + expected["source_count"])
    wake_tokens = safe_div(
        estimate_tokens(str(report.get("wake_response") or "")),
        max(1, wake.calls),
    )
    merge_yield = safe_div(len(merge_groups), len(expected["merge_groups"]))

    accuracy_score = 100.0 * (
        0.30 * merge_f1
        + 0.22 * conflict_f1
        + 0.16 * inference_f1
        + 0.16 * retention
        + 0.08 * (1.0 if expired_ok else 0.0)
        + 0.08 * cleanup_rate
    )
    if not idempotent or destructive_second:
        accuracy_score *= 0.85
    efficiency_score = weighted_score(
        [
            (
                score_lower_is_better(duration_per_100, TARGETS["dream_seconds_per_100"]),
                0.35,
            ),
            (
                score_lower_is_better(calls_per_memory, TARGETS["dream_calls_per_memory"]),
                0.30,
            ),
            (score_higher_is_better(throughput, 20.0), 0.20),
            (score_lower_is_better(wake_tokens, 100.0), 0.15),
        ]
    )
    result.score = 0.65 * accuracy_score + 0.35 * efficiency_score
    result.metrics = [
        metric(
            "dream_merge_f1",
            "Dream merge F1",
            merge_f1,
            unit="ratio",
            score=score_higher_is_better(merge_f1, TARGETS["dream_f1"]),
        ),
        metric(
            "dream_merge_precision",
            "Dream merge precision",
            merge_precision,
            unit="ratio",
            score=score_higher_is_better(merge_precision, TARGETS["dream_f1"]),
        ),
        metric(
            "dream_merge_recall",
            "Dream merge recall",
            merge_recall,
            unit="ratio",
            score=score_higher_is_better(merge_recall, TARGETS["dream_f1"]),
        ),
        metric(
            "dream_conflict_f1",
            "Dream conflict F1",
            conflict_f1,
            unit="ratio",
            score=score_higher_is_better(conflict_f1, TARGETS["dream_f1"]),
        ),
        metric(
            "dream_inference_f1",
            "Dream inference F1",
            inference_f1,
            unit="ratio",
            score=score_higher_is_better(inference_f1, TARGETS["dream_f1"]),
        ),
        metric(
            "dream_retention",
            "Source-memory retention",
            retention,
            unit="ratio",
            score=score_higher_is_better(retention, 1.0),
        ),
        metric(
            "dream_expiry_accuracy",
            "Expiry accuracy",
            1.0 if expired_ok else 0.0,
            unit="ratio",
            score=100.0 if expired_ok else 0.0,
        ),
        metric(
            "dream_artifact_cleanup",
            "Artifact cleanup rate",
            cleanup_rate,
            unit="ratio",
            score=score_higher_is_better(cleanup_rate, 1.0),
        ),
        metric(
            "dream_idempotent",
            "Idempotent on rerun",
            1.0 if idempotent and not destructive_second else 0.0,
            unit="bool",
            score=100.0 if idempotent and not destructive_second else 0.0,
        ),
        metric(
            "dream_error_rate",
            "Dream error rate",
            error_rate,
            unit="ratio",
            higher_is_better=False,
            score=score_lower_is_better(error_rate, 0.01),
        ),
        metric(
            "dream_duration_per_100_memories",
            "Duration per 100 memories",
            duration_per_100,
            unit="seconds",
            higher_is_better=False,
            score=score_lower_is_better(duration_per_100, TARGETS["dream_seconds_per_100"]),
        ),
        metric(
            "dream_calls_per_memory",
            "Backend calls per memory",
            calls_per_memory,
            unit="calls",
            higher_is_better=False,
            score=score_lower_is_better(calls_per_memory, TARGETS["dream_calls_per_memory"]),
        ),
        metric(
            "dream_throughput",
            "Memories consolidated per second",
            throughput,
            unit="mem/s",
            score=score_higher_is_better(throughput, 20.0),
        ),
        metric(
            "dream_actions_per_memory",
            "Actions per memory",
            actions_per_memory,
            unit="actions",
            higher_is_better=False,
            score=score_lower_is_better(actions_per_memory, 15.0),
        ),
        metric(
            "dream_wake_tokens",
            "Wake-model tokens per call",
            wake_tokens,
            unit="tokens",
            higher_is_better=False,
            score=score_lower_is_better(wake_tokens, 100.0),
        ),
        metric(
            "dream_wake_calls",
            "Wake-model calls",
            wake.calls,
            unit="calls",
            higher_is_better=False,
            score=None,
            n=1,
        ),
        metric(
            "dream_wake_tokens_total",
            "Wake-model tokens (total)",
            wake_tokens * wake.calls,
            unit="tokens",
            higher_is_better=False,
            score=None,
            n=1,
        ),
        metric(
            "dream_report_bytes",
            "Dream report size",
            report_bytes,
            unit="bytes",
            higher_is_better=False,
            score=None,
            n=1,
        ),
        metric(
            "dream_merge_yield",
            "Merge yield vs expected",
            merge_yield,
            unit="ratio",
            score=score_higher_is_better(merge_yield, 1.0),
        ),
        metric(
            "dream_transferred_messages",
            "Daily messages consolidated",
            transferred,
            unit="messages",
            n=1,
        ),
    ]
    result.diagnostics = {
        "now": now.isoformat(),
        "report": {
            "run_id": report.get("run_id"),
            "elapsed_seconds": elapsed,
            "transferred_days": report.get("transferred_days"),
            "conflicts": len(report.get("conflicts") or []),
            "inferences": len(report.get("inferences") or []),
            "communities": report.get("communities"),
            "caps_hit": report.get("caps_hit"),
            "errors": report.get("errors"),
            "artifacts_cleanup": cleanup,
            "wake_calls": wake.calls,
            "wake_response": report.get("wake_response"),
        },
        "expected": {
            "merge_groups": [sorted(group) for group in expected["merge_groups"]],
            "conflict_pairs": [sorted(pair) for pair in expected["conflict_pairs"]],
            "inference_pairs": [sorted(pair) for pair in expected["inference_pairs"]],
            "expired": sorted(expected["expired"]),
            "source_count": expected["source_count"],
        },
        "observed": {
            "merge_groups": [sorted(group) for group in merge_groups],
            "conflict_pairs": [sorted(pair) for pair in conflict_pairs],
            "inference_pairs": [sorted(pair) for pair in inference_pairs],
            "expired": sorted(expired),
            "retained": retained_ok,
            "retained_expected": retained_expected,
            "artifacts_left": artifacts_left,
        },
        "efficiency": {
            "backend_latency_ms": 0.5,
            "backend_calls_first_run": backend_calls_first,
            "backend_calls": backend_stats_first.get("calls"),
            "backend_bytes": backend_stats_first.get("bytes"),
            "backup": backup.stats(),
            "report_bytes": report_bytes,
            "actions": len(report.get("actions") or []),
            "second_run_actions": len(second_report.get("actions") or []),
            "second_run_destructive_actions": destructive_second,
        },
    }
    tempdir.cleanup()
    result.duration_ms = (time.perf_counter() - started) * 1000.0
    return result
