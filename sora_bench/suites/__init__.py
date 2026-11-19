"""Suite dispatch."""

from __future__ import annotations

from typing import Callable, Dict

from ..harness import BenchmarkHarness
from ..types import SuiteResult
from . import autonomy, caching, dreaming, features, general, memory, speed, tokens, tools

RUNNERS: Dict[str, Callable[[BenchmarkHarness], SuiteResult]] = {
    "tokens": tokens.run,
    "tools": tools.run,
    "memory": memory.run,
    "dreaming": dreaming.run,
    "speed": speed.run,
    "autonomy": autonomy.run,
    "caching": caching.run,
    "features": features.run,
    "general": general.run,
}


def run_suite(harness: BenchmarkHarness, name: str) -> SuiteResult:
    if name not in RUNNERS:
        raise KeyError("unknown suite {!r}".format(name))
    return RUNNERS[name](harness)
