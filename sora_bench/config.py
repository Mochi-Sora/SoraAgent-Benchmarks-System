"""Benchmark configuration, targets, and suite weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Tuple

SUITES: Tuple[str, ...] = (
    "tokens",
    "tools",
    "memory",
    "dreaming",
    "speed",
    "autonomy",
    "caching",
    "features",
)

DEFAULT_SUITES: Tuple[str, ...] = SUITES


@dataclass
class BenchConfig:
    mode: str = "simulated"
    provider_kind: str = "simulated"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    runs: int = 1
    seed: int = 1729
    suites: Tuple[str, ...] = DEFAULT_SUITES
    output_dir: Path = field(default_factory=lambda: Path("results"))
    max_steps: int = 16
    min_cache_block: int = 256
    simulated_latency_ms: float = 0.0
    latency_jitter_ms: float = 0.0
    autonomy_policy: str = "balanced"
    cache_ablation: bool = True
    include_requests: bool = False
    keep_workdir: bool = False
    verbose: bool = False
    timeout: float = 60.0
    temperature: float = 0.0
    output_formats: Tuple[str, ...] = ("json", "markdown", "csv", "jsonl")
    label: str = ""

    def __post_init__(self) -> None:
        self.mode = (self.mode or "simulated").strip().lower()
        if self.mode not in {"simulated", "live"}:
            raise ValueError("mode must be 'simulated' or 'live'")
        self.provider_kind = (
            (self.provider_kind or ("simulated" if self.mode == "simulated" else "openai"))
            .strip()
            .lower()
        )
        self.runs = max(1, int(self.runs))
        self.max_steps = max(1, int(self.max_steps))
        self.min_cache_block = max(0, int(self.min_cache_block))
        self.simulated_latency_ms = max(0.0, float(self.simulated_latency_ms))
        self.latency_jitter_ms = max(0.0, float(self.latency_jitter_ms))
        self.timeout = max(1.0, float(self.timeout))
        self.temperature = max(0.0, min(2.0, float(self.temperature)))
        policy = (self.autonomy_policy or "balanced").strip().lower()
        if policy not in {"minimal", "balanced", "overreach"}:
            raise ValueError("autonomy_policy must be minimal, balanced, or overreach")
        self.autonomy_policy = policy
        self.output_dir = Path(self.output_dir).expanduser()
        selected = []
        for name in self.suites:
            value = str(name).strip().lower()
            if value == "all":
                selected = list(SUITES)
                break
            if value not in SUITES:
                raise ValueError(
                    "unknown suite {!r}; choose from {}".format(value, ", ".join(SUITES))
                )
            if value not in selected:
                selected.append(value)
        self.suites = tuple(selected or DEFAULT_SUITES)
        formats = []
        for name in self.output_formats:
            value = str(name).strip().lower()
            if value in {"json", "markdown", "md", "csv", "jsonl"}:
                formats.append("markdown" if value == "md" else value)
        self.output_formats = tuple(dict.fromkeys(formats or ("json",)))

    @property
    def simulated(self) -> bool:
        return self.mode == "simulated"

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "provider_kind": self.provider_kind,
            "model": self.model,
            "runs": self.runs,
            "seed": self.seed,
            "suites": list(self.suites),
            "max_steps": self.max_steps,
            "min_cache_block": self.min_cache_block,
            "simulated_latency_ms": self.simulated_latency_ms,
            "latency_jitter_ms": self.latency_jitter_ms,
            "autonomy_policy": self.autonomy_policy,
            "cache_ablation": self.cache_ablation,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "label": self.label,
        }


#: Normalization targets for the 0..100 heuristic scores.  They are deliberately
#: explicit and conservative rather than hidden constants.
TARGETS = {
    "tool_f1": 0.95,
    "tool_exact_rate": 0.90,
    "tool_unnecessary_rate": 0.05,
    "token_efficiency_tasks_per_1k": 0.25,
    "token_prompt_mean": 5000.0,
    "token_total_mean": 5500.0,
    "token_per_success": 5000.0,
    "memory_hit_rate": 0.95,
    "memory_precision": 0.80,
    "memory_facts_per_1k_tokens": 8.0,
    "memory_recall_tokens": 400.0,
    "dream_f1": 0.95,
    "dream_retention": 1.0,
    "dream_seconds_per_100": 5.0,
    "dream_calls_per_memory": 8.0,
    "response_p95_ms": 3000.0,
    "response_mean_ms": 1500.0,
    "overhead_ms": 50.0,
    "autonomy_net_utility": 1.0,
    "autonomy_useful_precision": 0.90,
    "autonomy_harmful_rate": 0.0,
    "cache_hit_rate": 0.60,
    "cache_stability": 0.80,
    "cache_stable_prefix_tokens": 512.0,
}

#: Weights for the overall score.  Metric families carry the same weight;
#: within a family the suite decides how to combine sub-metrics.
SUITE_WEIGHTS = {
    "tokens": 1.0 / 8.0,
    "tools": 1.0 / 8.0,
    "memory": 1.0 / 8.0,
    "dreaming": 1.0 / 8.0,
    "speed": 1.0 / 8.0,
    "autonomy": 1.0 / 8.0,
    "caching": 1.0 / 8.0,
    "features": 1.0 / 8.0,
}


def parse_suites(values: Sequence[str]) -> Tuple[str, ...]:
    if not values:
        return DEFAULT_SUITES
    return BenchConfig(suites=tuple(values)).suites
