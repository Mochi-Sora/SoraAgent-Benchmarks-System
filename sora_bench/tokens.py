"""Deterministic, dependency-free token estimation.

Real providers report exact usage, and the benchmark always prefers those
numbers.  The simulated provider reports nothing, so we need a stable proxy for
prompt-size accounting and prompt-cache analysis.  This estimator is deliberately
simple and deterministic: it counts words, CJK characters, and punctuation with
fixed weights.  It is *not* a vendor tokenizer, and every report labels estimates
as such.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

_CJK_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\ucd00-\ud7af"
    r"\U00020000-\U0002fa1f]"
)
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def estimate_tokens(text: Any) -> int:
    """Estimate prompt/completion tokens for one string.

    Weights are intentionally conservative:

    * every CJK character is one token;
    * every ASCII word/number run is one token, plus one for very long runs;
    * every two punctuation characters are one token;
    * a non-empty string is at least one token.
    """

    if text is None:
        return 0
    value = str(text)
    if not value:
        return 0

    cjk = len(_CJK_RE.findall(value))
    without_cjk = _CJK_RE.sub(" ", value)
    words = _WORD_RE.findall(without_cjk)
    word_tokens = 0
    for word in words:
        length = len(word)
        if length <= 12:
            word_tokens += 1
        else:
            word_tokens += 1 + int(math.ceil((length - 12) / 12.0))
    punctuation = len(_PUNCT_RE.findall(without_cjk))
    punct_tokens = int(math.ceil(punctuation / 2.0))
    total = cjk + word_tokens + punct_tokens
    return max(1, int(total))


def estimate_chars_tokens(text: str) -> int:
    """Cheap fallback used only for coarse comparisons."""

    return int(math.ceil(len(text) / 4.0)) if text else 0


def estimate_messages_tokens(messages: Iterable[Any]) -> int:
    """Estimate the token footprint of a list of SoraAgent ``Message`` objects.

    Each message carries a small role/structural overhead, matching the fact
    that chat formats add role markers even for empty content.
    """

    total = 0
    for message in messages:
        total += 4
        total += estimate_tokens(getattr(message, "content", ""))
        for call in getattr(message, "tool_calls", ()) or ():
            total += 4
            total += estimate_tokens(getattr(call, "name", ""))
            total += estimate_tokens(getattr(call, "arguments", ""))
        if getattr(message, "name", ""):
            total += 1
        if getattr(message, "tool_call_id", ""):
            total += 2
    return total


def text_stats(text: str) -> dict:
    value = str(text or "")
    return {
        "chars": len(value),
        "lines": len(value.splitlines()) if value else 0,
        "words": len(_WORD_RE.findall(value)),
        "cjk_chars": len(_CJK_RE.findall(value)),
        "tokens": estimate_tokens(value),
    }


def normalized_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(text or "")).strip()


def truncate_preview(text: str, limit: int = 240) -> str:
    value = normalized_whitespace(text)
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."
