"""Optional tokenizer backends for token accounting.

The default estimator is deterministic and dependency-free. For live runs, a
model-specific tokenizer is more accurate; it is optional so the benchmark
continues to work offline and in CI without extra packages.
"""

from __future__ import annotations

from typing import Any

from .tokens import estimate_tokens


class HeuristicCounter:
    name = "heuristic"

    def count(self, text: Any) -> int:
        return estimate_tokens(text)


class TiktokenCounter:
    """Optional exact BPE counter; falls back to the heuristic if unavailable."""

    name = "tiktoken"

    def __init__(self, model: str = "") -> None:
        self._encoding = None
        self._fallback = True
        self.detail = ""
        try:
            import tiktoken
        except ImportError as exc:
            self.detail = "tiktoken unavailable: {}".format(exc)
            return
        try:
            self._encoding = (
                tiktoken.encoding_for_model(model)
                if model
                else tiktoken.get_encoding("cl100k_base")
            )
            self._fallback = False
        except Exception as exc:  # noqa: BLE001 - optional dependency
            self.detail = "tiktoken could not select an encoding: {}".format(exc)

    def count(self, text: Any) -> int:
        if self._encoding is None:
            return estimate_tokens(text)
        return max(1, len(self._encoding.encode(str(text or ""))))


def build_counter(kind: str = "heuristic", model: str = "") -> Any:
    choice = str(kind or "heuristic").strip().lower()
    if choice == "tiktoken":
        return TiktokenCounter(model)
    return HeuristicCounter()
