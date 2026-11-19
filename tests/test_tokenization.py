from __future__ import annotations

from sora_bench.tokenization import HeuristicCounter, build_counter


def test_tokenizer_falls_back_without_optional_package():
    heuristic = HeuristicCounter()
    counter = build_counter("tiktoken", "unknown-model")
    assert counter.count("hello world") > 0
    assert heuristic.count("hello world") > 0
