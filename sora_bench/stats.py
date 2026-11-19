"""Small, dependency-free uncertainty helpers for benchmark results.

Benchmarks with one run per task are descriptions, not measurements of a
population. These helpers make that distinction visible instead of hiding it in
a single normalized score.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Iterable, Optional, Sequence, Tuple


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Optional[Tuple[float, float]]:
    """95% Wilson score interval for a binomial success rate."""

    if n <= 0:
        return None
    successes = max(0, min(int(successes), int(n)))
    p = successes / float(n)
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    margin = (
        z
        * math.sqrt(max(0.0, p * (1.0 - p) / n + z * z / (4.0 * n * n)))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int = 1729,
    iterations: int = 2000,
    confidence: float = 0.95,
) -> Optional[Tuple[float, float]]:
    """Percentile bootstrap interval for a mean; ``None`` for fewer than 2 values."""

    items = [float(value) for value in values]
    if len(items) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(max(100, int(iterations))):
        sample = [rng.choice(items) for _ in items]
        means.append(statistics.fmean(sample))
    means.sort()
    alpha = max(0.0, min(0.5, (1.0 - float(confidence)) / 2.0))
    low_index = max(0, int(math.floor(alpha * len(means))))
    high_index = min(len(means) - 1, int(math.ceil((1.0 - alpha) * len(means))) - 1)
    return means[low_index], means[high_index]


def summarize_runs(runs: Iterable[object], *, seed: int = 1729) -> dict:
    """Produce transparent uncertainty diagnostics from a suite's runs."""

    records = list(runs)
    n = len(records)
    if n == 0:
        return {"n": 0}
    passed = [1 if bool(getattr(run, "passed", False)) else 0 for run in records]
    success = sum(passed)
    success_rate = success / float(n)
    success_ci = wilson_interval(success, n)
    eval_scores = [float(getattr(run, "eval_score", 0.0) or 0.0) for run in records]
    tokens = [int(getattr(run, "total_tokens", 0) or 0) for run in records]
    latency = [float(getattr(run, "wall_ms", 0.0) or 0.0) for run in records]
    return {
        "n": n,
        "passed": success,
        "success_rate": round(success_rate, 4),
        "success_ci95": [round(value, 4) for value in success_ci] if success_ci else None,
        "eval_score_mean": round(statistics.fmean(eval_scores), 4),
        "eval_score_ci95": _rounded_ci(bootstrap_mean_ci(eval_scores, seed=seed)),
        "tokens_mean": round(statistics.fmean(tokens), 2),
        "tokens_ci95": _rounded_ci(bootstrap_mean_ci(tokens, seed=seed + 1)),
        "latency_ms_mean": round(statistics.fmean(latency), 2),
        "latency_ci95": _rounded_ci(bootstrap_mean_ci(latency, seed=seed + 2)),
        "insufficient_replicates": n < 2,
    }


def _rounded_ci(value: Optional[Tuple[float, float]]) -> Optional[list]:
    if value is None:
        return None
    return [round(value[0], 4), round(value[1], 4)]
