"""Memory-retention accuracy and efficiency suite."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from sora_agent.memory.config import MemoryConfig
from sora_agent.memory.daily import DailyStore
from sora_agent.memory.recall import MemoryRecallService

from ..backend import FakeMemoryBackend
from ..config import TARGETS
from ..harness import BenchmarkHarness
from ..scenarios import TaskScenario
from ..scoring import (
    mean,
    percentile,
    safe_div,
    score_higher_is_better,
    score_lower_is_better,
    weighted_score,
)
from ..tokens import estimate_tokens
from ..types import SuiteResult, metric


@dataclass
class MemoryFact:
    fact_id: str
    content: str
    day_offset: int
    scope: str = "daily"  # daily | longterm
    role: str = "user"
    topic: str = ""


@dataclass
class MemoryQuery:
    id: str
    query: str
    expected_markers: List[str] = field(default_factory=list)
    forbidden_markers: List[str] = field(default_factory=list)
    scope: str = "auto"
    no_answer: bool = False


def fixture() -> tuple:
    facts = [
        MemoryFact(
            "f-orchid",
            "The vendor migration project is codenamed ORCHID-2741.",
            1,
            topic="project",
        ),
        MemoryFact(
            "f-tulip",
            "The analytics refresh project is codenamed TULIP-1180.",
            2,
            topic="project",
        ),
        MemoryFact(
            "f-seat",
            "My flight preference is an aisle seat; reference PREF-AISLE-441.",
            0,
            topic="preference",
        ),
        MemoryFact(
            "f-window",
            "An older note said my flight preference was a window seat; reference PREF-WINDOW-882.",
            3,
            topic="preference",
        ),
        MemoryFact(
            "f-badge",
            "My badge number is BADGE-7781.",
            2,
            topic="identity",
        ),
        MemoryFact(
            "f-deploy",
            "The deploy window for project ORCHID is 2026-12-03 02:00 UTC; reference DEPLOY-1203.",
            3,
            topic="schedule",
        ),
        MemoryFact(
            "f-owner",
            "The Northwind escalation owner is Priya Raman; reference OWNER-PRIYA.",
            4,
            topic="ownership",
        ),
        MemoryFact(
            "f-budget",
            "We approved the Q3 budget on 2026-10-14; reference BUDGET-Q3-1014.",
            5,
            topic="finance",
        ),
        MemoryFact(
            "f-sla",
            "The enterprise support SLA is 4 hours; reference SLA-4H.",
            6,
            topic="support",
        ),
        MemoryFact(
            "f-acme",
            "The Acme contract decision was to renew for 12 months; reference ACME-RENEW-12.",
            10,
            scope="longterm",
            topic="contract",
        ),
        MemoryFact(
            "f-retention",
            "The data retention policy is 18 months; reference RETENTION-18M.",
            12,
            scope="longterm",
            topic="policy",
        ),
        MemoryFact(
            "f-dc",
            "The primary datacenter is in Singapore; reference DC-SIN-1.",
            15,
            scope="longterm",
            topic="infrastructure",
        ),
        MemoryFact(
            "f-portal",
            "The employee travel portal is TravelHub; reference PORTAL-TRAVELHUB.",
            20,
            scope="longterm",
            topic="tooling",
        ),
        MemoryFact(
            "f-cjk",
            "项目 ORCHID 的部署窗口是 2026-12-03 02:00 UTC；参考 DEPLOY-CJK-1203。"
            " Deployment window for project ORCHID is 2026-12-03 02:00 UTC.",
            4,
            topic="schedule",
        ),
    ]
    queries = [
        MemoryQuery(
            "codename",
            "What is the codename of the vendor migration project?",
            ["ORCHID-2741"],
            ["TULIP-1180"],
        ),
        MemoryQuery(
            "seat",
            "Which seat do I prefer on flights?",
            ["PREF-AISLE-441"],
            ["PREF-WINDOW-882"],
        ),
        MemoryQuery("badge", "What is my badge number?", ["BADGE-7781"]),
        MemoryQuery(
            "deploy",
            "When is the deploy window for project ORCHID?",
            ["DEPLOY-1203"],
        ),
        MemoryQuery("owner", "Who owns the Northwind escalation?", ["OWNER-PRIYA"]),
        MemoryQuery("budget", "When did we approve the Q3 budget?", ["BUDGET-Q3-1014"]),
        MemoryQuery("sla", "What is the enterprise support SLA?", ["SLA-4H"]),
        MemoryQuery(
            "acme",
            "What did we decide about the Acme contract?",
            ["ACME-RENEW-12"],
            scope="longterm",
        ),
        MemoryQuery(
            "retention",
            "How long is the data retention policy?",
            ["RETENTION-18M"],
        ),
        MemoryQuery("datacenter", "What is the primary datacenter?", ["DC-SIN-1"]),
        MemoryQuery("portal", "What is the employee travel portal?", ["PORTAL-TRAVELHUB"]),
        MemoryQuery("cjk", "部署窗口", ["DEPLOY-CJK-1203"], scope="daily"),
        MemoryQuery(
            "none",
            "What is the CEO's shoe size?",
            [],
            no_answer=True,
        ),
    ]
    return facts, queries


def _seed(
    backend: FakeMemoryBackend,
    daily: DailyStore,
    facts: Sequence[MemoryFact],
    now: datetime,
) -> None:
    for fact in facts:
        moment = now - timedelta(days=fact.day_offset)
        if fact.scope == "daily":
            daily.append(fact.role, fact.content, ts=moment)
        else:
            backend.save_memory(
                {
                    "memory_id": fact.fact_id,
                    "content": fact.content,
                    "importance": 0.6,
                    "role": fact.role,
                    "metadata": {"sora_kind": "seed", "topic": fact.topic},
                    "created_at": moment.isoformat(),
                }
            )


def _query_metrics(
    service: MemoryRecallService, query: MemoryQuery, all_markers: Sequence[str]
) -> Dict[str, Any]:
    started = time.perf_counter()
    items, note = service.collect(query.query, scope=query.scope, k=8)
    latency_ms = (time.perf_counter() - started) * 1000.0
    text = service.format(items, note=note, max_chars=service.max_chars)
    contents = [item.content for item in items]
    relevant = [
        content
        for content in contents
        if any(marker in content for marker in query.expected_markers)
    ]
    expected_hit = bool(
        not query.expected_markers
        or any(marker in content for content in contents for marker in query.expected_markers)
    )
    forbidden_hit = any(
        marker in content for content in contents for marker in query.forbidden_markers
    )
    rank = next(
        (
            index + 1
            for index, content in enumerate(contents)
            if any(marker in content for marker in query.expected_markers)
        ),
        None,
    )
    mrr = (1.0 / rank) if rank else (1.0 if query.no_answer else 0.0)
    precision = len(relevant) / float(len(items)) if items else (1.0 if query.no_answer else 0.0)
    known_marker_hit = any(marker in text for marker in all_markers)
    no_answer_false_positive = bool(query.no_answer and known_marker_hit)
    tokens = estimate_tokens(text)
    signal_chars = sum(len(content) for content in relevant)
    density = safe_div(signal_chars, len(text))
    corpus_chars = 0  # filled by caller
    return {
        "query_id": query.id,
        "query": query.query,
        "scope": query.scope,
        "expected_hit": expected_hit,
        "forbidden_hit": forbidden_hit,
        "no_answer": query.no_answer,
        "no_answer_false_positive": no_answer_false_positive,
        "precision": precision,
        "mrr": mrr,
        "recall_tokens": tokens,
        "recall_chars": len(text),
        "signal_chars": signal_chars,
        "signal_density": density,
        "latency_ms": latency_ms,
        "items": len(items),
        "text": text[:2000],
        "note": note,
        "corpus_chars": corpus_chars,
    }


def run(harness: BenchmarkHarness) -> SuiteResult:
    started = time.perf_counter()
    result = SuiteResult(name="memory")
    facts, queries = fixture()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    all_markers = [
        marker for query in queries for marker in query.expected_markers + query.forbidden_markers
    ]

    tempdir = tempfile.TemporaryDirectory(prefix="sora-bench-memory-")
    root = Path(tempdir.name)
    config = MemoryConfig(root=root, auto_download=False, dream_auto=False)
    daily = DailyStore(config.daily_dir, clock=lambda: now)
    backend = FakeMemoryBackend()
    _seed(backend, daily, facts, now)
    service = MemoryRecallService(daily, lambda: backend, top_k=8, max_chars=6000)
    corpus_chars = sum(len(fact.content) for fact in facts)
    query_metrics: List[Dict[str, Any]] = []
    for query in queries:
        item = _query_metrics(service, query, all_markers)
        item["corpus_chars"] = corpus_chars
        item["compression_ratio"] = safe_div(corpus_chars, item["recall_chars"])
        query_metrics.append(item)

    answerable = [item for item in query_metrics if not item["no_answer"]]
    hit_rate = mean(1.0 if item["expected_hit"] else 0.0 for item in answerable)
    forbidden_rate = mean(1.0 if item["forbidden_hit"] else 0.0 for item in answerable)
    precision = mean(item["precision"] for item in answerable)
    mrr = mean(item["mrr"] for item in answerable)
    no_answer_fp = (
        mean(
            1.0 if item["no_answer_false_positive"] else 0.0
            for item in query_metrics
            if item["no_answer"]
        )
        if any(item["no_answer"] for item in query_metrics)
        else 0.0
    )
    recall_tokens_mean = mean(item["recall_tokens"] for item in query_metrics)
    density_mean = mean(item["signal_density"] for item in query_metrics)
    compression_mean = mean(item["compression_ratio"] for item in query_metrics)
    facts_per_1k = mean(
        safe_div(
            len(query.expected_markers) * 1000.0,
            max(1, item["recall_tokens"]),
        )
        for query, item in zip(queries, query_metrics)
        if query.expected_markers
    )
    recall_latency = mean(item["latency_ms"] for item in query_metrics)

    # End-to-end agent runs for a representative subset.  The provider answers
    # extractively from the first recall bullet, so this measures the plumbing
    # (recall -> prompt -> answer) rather than a second model's reasoning.
    agent_queries = [
        query
        for query in queries
        if query.id in {"codename", "seat", "badge", "deploy", "acme", "cjk", "none"}
    ]
    agent_runs = []
    agent_answers = []
    recall_first = []
    grounded = []
    write_through = []
    recall_tokens_agent = []
    for query in agent_queries:
        for run_index in range(harness.config.runs):
            scenario = TaskScenario(
                id="memory/agent/{}".format(query.id),
                suite="memory",
                task=query.query,
                plan=[],
                expected_answer_terms=list(query.expected_markers),
                forbidden_answer_terms=list(query.forbidden_markers),
                memory="seeded",
                identity=False,
                clock="off",
                max_steps=8,
                tags=["memory_agent"],
                allow_forbidden_with_expected=True,
                description="Recall-first answer for {!r}".format(query.query),
            )

            def prepare(
                backend: FakeMemoryBackend,
                daily_store: DailyStore,
                _config: MemoryConfig,
                _facts=facts,
                _now=now,
            ) -> None:
                _seed(backend, daily_store, _facts, _now)

            artifacts = harness.run_task(
                scenario,
                run_index=run_index,
                prepare_memory=prepare,
                recall_query=query.query,
                extract_recall=True,
            )
            record = artifacts.run
            recall_content = _recall_content(artifacts.result)
            recall_tokens_agent.append(estimate_tokens(recall_content))
            is_grounded = bool(
                not query.expected_markers
                or all(marker in recall_content for marker in query.expected_markers)
            )
            agent_answers.append(1.0 if record.evaluation.get("answers_ok") else 0.0)
            recall_first.append(1.0 if _recall_first(record) else 0.0)
            grounded.append(1.0 if is_grounded else 0.0)
            seeded_daily = len([fact for fact in facts if fact.scope == "daily"])
            total_daily_messages = (
                sum(artifacts.daily.count(day) for day in artifacts.daily.all_days())
                if artifacts.daily is not None
                else 0
            )
            write_through.append(1.0 if total_daily_messages > seeded_daily else 0.0)
            agent_runs.append(record)
            artifacts.cleanup(keep=harness.config.keep_workdir)

    answer_accuracy = mean(agent_answers)
    recall_first_rate = mean(recall_first)
    grounding_rate = mean(grounded)
    write_through_rate = mean(write_through)
    accuracy_score = 100.0 * (
        0.45 * hit_rate
        + 0.30 * answer_accuracy
        + 0.10 * (1.0 - forbidden_rate)
        + 0.10 * (1.0 - no_answer_fp)
        + 0.05 * mrr
    )
    efficiency_score = weighted_score(
        [
            (
                score_higher_is_better(facts_per_1k, TARGETS["memory_facts_per_1k_tokens"]),
                0.45,
            ),
            (
                score_lower_is_better(recall_tokens_mean, TARGETS["memory_recall_tokens"]),
                0.30,
            ),
            (score_higher_is_better(density_mean, 0.55), 0.25),
        ]
    )
    result.score = 0.60 * accuracy_score + 0.40 * efficiency_score
    result.runs = agent_runs
    result.metrics = [
        metric(
            "memory_retrieval_hit_rate",
            "Retrieval hit rate",
            hit_rate,
            unit="ratio",
            score=score_higher_is_better(hit_rate, TARGETS["memory_hit_rate"]),
            n=len(answerable),
        ),
        metric(
            "memory_precision",
            "Retrieval precision",
            precision,
            unit="ratio",
            score=score_higher_is_better(precision, TARGETS["memory_precision"]),
            n=len(answerable),
        ),
        metric(
            "memory_mrr",
            "Mean reciprocal rank",
            mrr,
            unit="ratio",
            score=score_higher_is_better(mrr, 0.85),
            n=len(answerable),
        ),
        metric(
            "memory_forbidden_rate",
            "Contradicting/forbidden fact rate",
            forbidden_rate,
            unit="ratio",
            higher_is_better=False,
            score=score_lower_is_better(forbidden_rate, 0.05),
            n=len(answerable),
        ),
        metric(
            "memory_no_answer_false_positive_rate",
            "No-answer false-positive rate",
            no_answer_fp,
            unit="ratio",
            higher_is_better=False,
            score=score_lower_is_better(no_answer_fp, 0.05),
            n=1,
        ),
        metric(
            "memory_answer_accuracy",
            "End-to-end answer accuracy",
            answer_accuracy,
            unit="ratio",
            score=score_higher_is_better(answer_accuracy, TARGETS["memory_hit_rate"]),
            n=len(agent_answers),
        ),
        metric(
            "memory_recall_first_rate",
            "Recall-first compliance",
            recall_first_rate,
            unit="ratio",
            score=score_higher_is_better(recall_first_rate, 1.0),
            n=len(agent_answers),
        ),
        metric(
            "memory_grounding_rate",
            "Answer grounded in recall",
            grounding_rate,
            unit="ratio",
            score=score_higher_is_better(grounding_rate, 1.0),
            n=len(agent_answers),
        ),
        metric(
            "memory_write_through_rate",
            "Memory write-through rate",
            write_through_rate,
            unit="ratio",
            score=score_higher_is_better(write_through_rate, 1.0),
            n=len(agent_answers),
        ),
        metric(
            "memory_facts_per_1k_tokens",
            "Retained facts per 1k recall tokens",
            facts_per_1k,
            unit="facts/1k",
            score=score_higher_is_better(facts_per_1k, TARGETS["memory_facts_per_1k_tokens"]),
            n=len(answerable),
        ),
        metric(
            "memory_recall_tokens_mean",
            "Recall payload tokens per query",
            recall_tokens_mean,
            unit="tokens",
            higher_is_better=False,
            score=score_lower_is_better(recall_tokens_mean, TARGETS["memory_recall_tokens"]),
            n=len(query_metrics),
        ),
        metric(
            "memory_signal_density",
            "Relevant-character density",
            density_mean,
            unit="ratio",
            score=score_higher_is_better(density_mean, 0.55),
            n=len(query_metrics),
        ),
        metric(
            "memory_compression_ratio",
            "Corpus-to-recall compression",
            compression_mean,
            unit="x",
            score=None,
            detail="Corpus-size dependent; informational in this fixture.",
            n=len(query_metrics),
        ),
        metric(
            "memory_recall_latency_mean",
            "Recall latency",
            recall_latency,
            unit="ms",
            higher_is_better=False,
            score=score_lower_is_better(recall_latency, 50.0),
            n=len(query_metrics),
        ),
        metric(
            "memory_efficiency_score",
            "Memory efficiency sub-score",
            efficiency_score,
            unit="score",
            n=len(query_metrics),
        ),
    ]
    result.diagnostics = {
        "retrieval_note": (
            "Daily verbatim search treats query terms as OR matches; common "
            "words such as 'the' can surface unrelated memories.  The "
            "no-answer false-positive metric captures that behavior."
        ),
        "queries": query_metrics,
        "agent_runs": [
            {
                "scenario_id": run.scenario_id,
                "passed": run.passed,
                "answer_ok": bool(run.evaluation.get("answers_ok")),
                "answer": run.final_answer[:300],
                "recall_first": _recall_first(run),
                "recall_tokens": run.prompt_tokens,
                "stop_reason": run.stop_reason,
            }
            for run in agent_runs
        ],
        "corpus_facts": len(facts),
        "corpus_chars": corpus_chars,
        "p95_recall_tokens": percentile([item["recall_tokens"] for item in query_metrics], 0.95),
    }
    tempdir.cleanup()
    result.duration_ms = (time.perf_counter() - started) * 1000.0
    return result


def _recall_content(result: Any) -> str:
    if result is None:
        return ""
    for step in reversed(getattr(result, "steps", []) or []):
        for item in reversed(getattr(step, "results", []) or []):
            call = getattr(item, "call", None)
            if getattr(call, "name", "") == "memory_recall":
                return str(getattr(item, "content", "") or "")
    return ""


def _recall_first(record: Any) -> bool:
    for item in record.tool_call_log:
        if item["skipped"]:
            continue
        return item["name"] == "memory_recall"
    return False
