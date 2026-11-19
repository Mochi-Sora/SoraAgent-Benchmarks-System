from __future__ import annotations

from sora_bench.tokens import estimate_tokens


def test_estimator_is_deterministic_and_nonzero():
    text = "The quick brown fox jumps over the lazy dog."
    assert estimate_tokens(text) == estimate_tokens(text)
    assert estimate_tokens(text) > 5
    assert estimate_tokens("") == 0


def test_estimator_handles_cjk_and_punctuation():
    assert estimate_tokens("部署窗口是什么时候？") >= 7
    assert estimate_tokens("Hello, world!") >= 3


def test_long_words_cost_more():
    assert estimate_tokens("a" * 60) > estimate_tokens("a" * 12)
