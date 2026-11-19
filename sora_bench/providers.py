"""Providers used by the benchmark.

There are three layers:

* ``ScriptedProvider`` — deterministic reference policy used in simulated mode.
  It drives the *real* SoraAgent loop, including lazy discovery, the required
  memory recall, approvals, and observation feedback.
* ``RecordingProvider`` — wraps any provider and snapshots every request for
  token and prompt-cache accounting.
* ``CacheAwareOpenAIProvider`` — optional live OpenAI-compatible provider that
  preserves vendor cache-usage fields (``cached_tokens``,
  ``prompt_cache_hit_tokens``, ...) that SoraAgent's normal turn object does not
  expose.

No network access happens in simulated mode.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from sora_agent.providers import (
    Message,
    ModelTurn,
    OpenAIChatProvider,
    ProviderConfigError,
    ToolCall,
    _tool_calls,
    _usage_from,
    provider_turn,
)

from .protocol import build_request_record, canonical_request
from .tokenization import HeuristicCounter

_BACKTICK_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")
_MEMORY_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_MEMORY_DATE_RE = re.compile(r"daily:(\d{4}-\d{2}-\d{2})")
_MEMORY_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "do",
        "did",
        "does",
        "my",
        "me",
        "we",
        "our",
        "i",
        "on",
        "in",
        "of",
        "to",
        "a",
        "an",
        "about",
        "that",
        "this",
        "it",
        "its",
        "be",
        "been",
        "have",
        "has",
        "had",
        "can",
        "could",
        "would",
        "should",
        "please",
    }
)


def clone_messages(messages: Sequence[Any]) -> List[Message]:
    """Copy the message list so later runner mutations cannot rewrite history."""

    cloned: List[Message] = []
    for message in messages:
        calls = [
            ToolCall(
                id=str(getattr(call, "id", "")),
                name=str(getattr(call, "name", "")),
                arguments=dict(getattr(call, "arguments", {}) or {}),
                raw_arguments=str(getattr(call, "raw_arguments", "") or ""),
                error=str(getattr(call, "error", "") or ""),
            )
            for call in (getattr(message, "tool_calls", ()) or ())
        ]
        cloned.append(
            Message(
                role=str(getattr(message, "role", "")),
                content=str(getattr(message, "content", "") or ""),
                tool_calls=calls,
                tool_call_id=str(getattr(message, "tool_call_id", "") or ""),
                name=str(getattr(message, "name", "") or ""),
            )
        )
    return cloned


def _completion_text(turn: ModelTurn) -> str:
    parts = [turn.text or ""]
    for call in turn.tool_calls:
        parts.append(call.name)
        parts.append(json.dumps(call.arguments, sort_keys=True, ensure_ascii=False, default=str))
    return "\n".join(parts)


class RecordingProvider:
    """Capture each provider request and return the wrapped provider's turn."""

    def __init__(
        self,
        inner: Any,
        *,
        model: str = "simulated",
        min_cache_block: int = 256,
        tokenizer: Any = None,
    ) -> None:
        self.inner = inner
        self.model = model
        self.min_cache_block = max(0, int(min_cache_block))
        self.tokenizer = tokenizer or HeuristicCounter()
        self.requests: List[Any] = []
        self._previous_messages: Optional[List[Message]] = None
        self._first_messages: Optional[List[Message]] = None
        self._previous_full: str = ""
        self._first_full: str = ""
        self.last_cache_usage: Dict[str, Any] = {}

    def complete_turn(self, messages: List[Message], tools: List[Any]) -> ModelTurn:
        snapshot = clone_messages(messages)
        full, _message_text, _tool_text = canonical_request(messages, tools)
        prompt_tokens = self.tokenizer.count(full)
        started = time.perf_counter()
        error = ""
        try:
            turn = provider_turn(self.inner, messages, tools)
        except Exception as exc:  # noqa: BLE001 - record failures before reraising
            error = "{}: {}".format(type(exc).__name__, exc)
            self._record(
                snapshot,
                tools,
                step=len(self.requests),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                turn=None,
                error=error,
            )
            raise
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = getattr(turn, "usage", None)
        cache_usage = self._cache_usage()
        completion_tokens = self.tokenizer.count(_completion_text(turn))
        self._record(
            snapshot,
            tools,
            step=len(self.requests),
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            turn=turn,
            error="",
            usage=usage,
            cache_usage=cache_usage,
        )
        return turn

    def complete(self, messages: List[Message], tools: List[Any]) -> str:
        return self.complete_turn(messages, tools).text

    def stream_turn(
        self,
        messages: List[Message],
        tools: List[Any],
        on_delta: Callable[[str], None],
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> ModelTurn:
        # SoraAgent's Agent facade does not stream today, but keeping this
        # method lets library callers exercise the provider edge unchanged.
        snapshot = clone_messages(messages)
        full, _message_text, _tool_text = canonical_request(messages, tools)
        prompt_tokens = self.tokenizer.count(full)
        started = time.perf_counter()
        stream = getattr(self.inner, "stream_turn", None)
        turn: ModelTurn
        if callable(stream):
            turn = stream(
                messages,
                tools,
                on_delta,
                should_stop=should_stop,
            )
        else:
            turn = provider_turn(self.inner, messages, tools)
            if turn.text and (should_stop is None or not should_stop()):
                on_delta(turn.text)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._record(
            snapshot,
            tools,
            step=len(self.requests),
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=self.tokenizer.count(_completion_text(turn)),
            turn=turn,
            error="",
            usage=getattr(turn, "usage", None),
            cache_usage=self._cache_usage(),
        )
        return turn

    def _cache_usage(self) -> Dict[str, Any]:
        value = getattr(self.inner, "last_cache_usage", None)
        return dict(value) if isinstance(value, dict) else {}

    def _record(
        self,
        snapshot: List[Message],
        tools: List[Any],
        *,
        step: int,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        turn: Optional[ModelTurn],
        error: str,
        usage: Any = None,
        cache_usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        reported_prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        reported_completion = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        record, full, _message_text = build_request_record(
            index=len(self.requests),
            step=step,
            messages=snapshot,
            tools=tools,
            previous_messages=self._previous_messages,
            first_messages=self._first_messages,
            previous_full=self._previous_full,
            first_full=self._first_full,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reported_prompt_tokens=reported_prompt,
            reported_completion_tokens=reported_completion,
            cache_usage=cache_usage,
            model=self.model,
            min_cache_block=self.min_cache_block,
        )
        record.error = error
        self.requests.append(record)
        if self._first_messages is None:
            self._first_messages = snapshot
            self._first_full = full
        self._previous_messages = snapshot
        self._previous_full = full


# ---------------------------------------------------------------------------
# Simulated reference policy
# ---------------------------------------------------------------------------


class ScriptedProvider:
    """Follow a plan through the real SoraAgent loop.

    The plan contains business actions only.  Lazy discovery and the required
    memory recall are handled automatically, exactly as a cooperative model
    would handle them.
    """

    def __init__(
        self,
        plan: Sequence[Any],
        final_answer: str,
        *,
        recall_query: str = "",
        extract_recall: bool = False,
    ) -> None:
        self.plan = list(plan)
        self.final_answer = final_answer
        self.recall_query = recall_query
        self.extract_recall = extract_recall
        self.plan_index = 0
        self._pending: Optional[Dict[str, Any]] = None
        self._forced_load: List[str] = []
        self._discovery_attempts = 0
        self._failed = ""
        self._sequence = 0
        self.last_cache_usage: Dict[str, Any] = {}

    def complete_turn(self, messages: List[Message], tools: List[Any]) -> ModelTurn:
        self.last_cache_usage = {}
        visible = {str(getattr(tool, "name", "")) for tool in tools}
        tool_messages = [message for message in messages if getattr(message, "role", "") == "tool"]

        if self._pending and self._observed(self._pending, tool_messages):
            kind = self._pending.get("kind")
            observation = self._last_observation(self._pending, tool_messages)
            self._pending = None
            if kind == "call":
                self.plan_index += 1
                self._discovery_attempts = 0
            elif kind == "preview":
                candidate = self._preview_candidate(observation, visible)
                if candidate:
                    self._forced_load = [candidate]
                else:
                    self._failed = "The tool preview did not identify a usable tool."
            # ``find`` and ``recall`` deliberately keep the plan index.

        if self._failed:
            return ModelTurn(text=self._failed)

        # Memory is a required first action whenever the tool is visible.
        if "memory_recall" in visible and not any(
            getattr(message, "name", "") == "memory_recall" for message in tool_messages
        ):
            query = self.recall_query
            if not query:
                query = next(
                    (
                        str(getattr(message, "content", "") or "")
                        for message in reversed(messages)
                        if getattr(message, "role", "") == "user"
                    ),
                    "task context",
                )
            call_id = self._next_id()
            self._pending = {
                "kind": "recall",
                "id": call_id,
                "name": "memory_recall",
            }
            return ModelTurn(
                text="I will recall relevant memory before acting.",
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        name="memory_recall",
                        arguments={"query": query, "scope": "auto", "k": 8},
                    )
                ],
            )

        if self._forced_load:
            names = [name for name in self._forced_load if name not in visible]
            self._forced_load = []
            if names:
                call_id = self._next_id()
                self._pending = {
                    "kind": "find",
                    "id": call_id,
                    "name": "find_tools",
                }
                return ModelTurn(
                    text="Loading the discovered tool.",
                    tool_calls=[
                        ToolCall(
                            id=call_id,
                            name="find_tools",
                            arguments={"load": names},
                        )
                    ],
                )

        if self.plan_index >= len(self.plan):
            return ModelTurn(text=self._final_text(tool_messages))

        step = self.plan[self.plan_index]
        kind = getattr(step, "kind", "tool")
        if kind == "say":
            self.plan_index += 1
            return ModelTurn(text=str(getattr(step, "text", "")))

        name = str(getattr(step, "name", ""))
        arguments = dict(getattr(step, "args", {}) or {})
        discover_need = str(getattr(step, "discover_need", "") or "")

        if name not in visible and discover_need and "find_tools" in visible:
            self._discovery_attempts += 1
            if self._discovery_attempts > 3:
                self._failed = "I could not discover a tool for this request."
                return ModelTurn(text=self._failed)
            call_id = self._next_id()
            self._pending = {
                "kind": "preview",
                "id": call_id,
                "name": "find_tools",
            }
            return ModelTurn(
                text="Searching the tool catalog.",
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        name="find_tools",
                        arguments={"need": discover_need},
                    )
                ],
            )

        if name not in visible and "find_tools" in visible:
            self._discovery_attempts += 1
            if self._discovery_attempts > 3:
                self._failed = "I could not load the tool needed for this request."
                return ModelTurn(text=self._failed)
            call_id = self._next_id()
            self._pending = {
                "kind": "find",
                "id": call_id,
                "name": "find_tools",
            }
            return ModelTurn(
                text="Loading the tool needed for this request.",
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        name="find_tools",
                        arguments={"load": [name]},
                    )
                ],
            )

        if name not in visible:
            self._failed = "The tool needed for this request is not available."
            return ModelTurn(text=self._failed)

        call_id = self._next_id()
        self._pending = {"kind": "call", "id": call_id, "name": name}
        return ModelTurn(
            text="",
            tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        )

    def complete(self, messages: List[Message], tools: List[Any]) -> str:
        return self.complete_turn(messages, tools).text

    def _final_text(self, tool_messages: Sequence[Message]) -> str:
        if self.extract_recall:
            for message in reversed(tool_messages):
                if getattr(message, "name", "") == "memory_recall":
                    content = str(getattr(message, "content", "") or "")
                    selected = self._select_memory_bullet(content)
                    if selected:
                        return "Based on memory: " + selected
                    if content.strip():
                        return "No relevant memory found."
        return self.final_answer

    def _select_memory_bullet(self, content: str) -> str:
        """Pick the recall bullet with the strongest lexical overlap.

        Ties prefer the more recent daily memory.  This is the deterministic
        reference policy's "read the recall result" behavior, not an oracle:
        it never sees the expected marker.
        """

        query_terms = self._memory_terms(self.recall_query)
        if not query_terms:
            return ""
        best = ""
        best_key = None
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped.startswith("- ["):
                continue
            terms = self._memory_terms(stripped)
            overlap = len(query_terms & terms)
            if overlap <= 0:
                continue
            date_match = _MEMORY_DATE_RE.search(stripped)
            recency = date_match.group(1) if date_match else ""
            key = (overlap, recency)
            if best_key is None or key > best_key:
                best = stripped
                best_key = key
        return best

    @staticmethod
    def _memory_terms(text: str) -> set:
        value = str(text or "").lower()
        terms = {term for term in _MEMORY_TOKEN_RE.findall(value) if term not in _MEMORY_STOPWORDS}
        terms.update(char for char in value if "\u3400" <= char <= "\u9fff")
        return terms

    def _next_id(self) -> str:
        self._sequence += 1
        return "bench_call_{}".format(self._sequence)

    @staticmethod
    def _observed(pending: Dict[str, Any], tool_messages: Sequence[Message]) -> bool:
        call_id = str(pending.get("id") or "")
        name = str(pending.get("name") or "")
        for message in tool_messages:
            message_id = str(getattr(message, "tool_call_id", "") or "")
            if message_id and message_id == call_id:
                return True
            if not message_id and getattr(message, "name", "") == name:
                return True
        return False

    @staticmethod
    def _last_observation(pending: Dict[str, Any], tool_messages: Sequence[Message]) -> str:
        call_id = str(pending.get("id") or "")
        name = str(pending.get("name") or "")
        for message in reversed(tool_messages):
            message_id = str(getattr(message, "tool_call_id", "") or "")
            if message_id == call_id or (not message_id and getattr(message, "name", "") == name):
                return str(getattr(message, "content", "") or "")
        return ""

    @staticmethod
    def _preview_candidate(observation: str, visible: set) -> str:
        for match in _BACKTICK_RE.findall(observation or ""):
            if match in {"find_tools", "memory_recall"}:
                continue
            return match
        return ""


class LatencyProvider:
    """Add a deterministic delay before delegating, for speed-sensitivity runs."""

    def __init__(
        self,
        inner: Any,
        delay_ms: float = 0.0,
        jitter_ms: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.inner = inner
        self.delay_ms = max(0.0, float(delay_ms))
        self.jitter_ms = max(0.0, float(jitter_ms))
        self.random = random.Random(seed)

    @property
    def last_cache_usage(self) -> Dict[str, Any]:
        value = getattr(self.inner, "last_cache_usage", None)
        return dict(value) if isinstance(value, dict) else {}

    def complete_turn(self, messages: List[Message], tools: List[Any]) -> ModelTurn:
        delay = self.delay_ms
        if self.jitter_ms:
            delay += self.random.uniform(0.0, self.jitter_ms)
        if delay > 0:
            time.sleep(delay / 1000.0)
        return provider_turn(self.inner, messages, tools)

    def complete(self, messages: List[Message], tools: List[Any]) -> str:
        return self.complete_turn(messages, tools).text


# ---------------------------------------------------------------------------
# Live provider with cache-usage capture
# ---------------------------------------------------------------------------


class CacheAwareOpenAIProvider(OpenAIChatProvider):
    """OpenAI-compatible provider that keeps vendor prompt-cache counters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_cache_usage: Dict[str, Any] = {}

    def complete_turn(self, messages: List[Message], tools: List[Any]) -> ModelTurn:
        client = self._client()
        response = client.chat.completions.create(**self._payload(messages, tools))
        message = response.choices[0].message
        text = message.content or ""
        if not isinstance(text, str):
            text = str(text)
        usage_raw = getattr(response, "usage", None)
        self.last_cache_usage = extract_cache_usage(usage_raw)
        return ModelTurn(
            text=text,
            tool_calls=_tool_calls(getattr(message, "tool_calls", None) or []),
            usage=_usage_from(usage_raw),
        )

    def complete(self, messages: List[Message], tools: List[Any]) -> str:
        return self.complete_turn(messages, tools).text


def _value(raw: Any, key: str) -> Any:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw.get(key)
    return getattr(raw, key, None)


def extract_cache_usage(raw: Any) -> Dict[str, Any]:
    """Extract cache counters from OpenAI/DeepSeek/Anthropic-style usage."""

    if raw is None:
        return {}
    cached = 0
    written = 0
    source = ""
    details = _value(raw, "prompt_tokens_details") or _value(raw, "input_tokens_details")
    detail_cached = _value(details, "cached_tokens")
    if isinstance(detail_cached, (int, float)):
        cached = int(detail_cached)
        source = source or "prompt_tokens_details.cached_tokens"
    deepseek_hit = _value(raw, "prompt_cache_hit_tokens")
    if isinstance(deepseek_hit, (int, float)):
        cached = max(cached, int(deepseek_hit))
        source = source or "prompt_cache_hit_tokens"
    anthropic_read = _value(raw, "cache_read_input_tokens")
    if isinstance(anthropic_read, (int, float)):
        cached = max(cached, int(anthropic_read))
        source = source or "cache_read_input_tokens"
    anthropic_write = _value(raw, "cache_creation_input_tokens")
    if isinstance(anthropic_write, (int, float)):
        written = int(anthropic_write)
        source = source or "cache_creation_input_tokens"
    if not source:
        return {}
    return {
        "cached_tokens": cached,
        "cache_write_tokens": written,
        "source": source,
    }


def _saved_api_key(kind: str) -> str:
    """Load a saved SoraAgent secret when the environment has no key."""

    env_name = {
        "openai": "OPENAI_API_KEY",
        "azure": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "ds": "DEEPSEEK_API_KEY",
        "qwen": "DASHSCOPE_API_KEY",
        "dashscope": "DASHSCOPE_API_KEY",
        "custom": "SORA_CUSTOM_API_KEY",
        "openai-compatible": "SORA_CUSTOM_API_KEY",
    }.get(kind, "")
    if not env_name:
        return ""
    try:
        from sora_agent.settings import load_secrets

        value = str(load_secrets().get(env_name, "")).strip()
        if value:
            return value
        fallback = Path.home() / ".sora_agent" / "secrets.env"
        return str(load_secrets(fallback).get(env_name, "")).strip()
    except Exception:  # noqa: BLE001 - saved secrets are optional
        return ""


def build_live_provider(
    kind: str,
    *,
    model: str = "",
    api_key: str = "",
    base_url: str = "",
    timeout: float = 60.0,
    temperature: float = 0.0,
) -> Any:
    """Build a live OpenAI-compatible provider from env/defaults."""

    kind = (kind or "").strip().lower()
    if kind in {"openai", "azure"}:
        resolved_model = model or os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
        resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
        resolved_url = base_url or os.getenv("OPENAI_BASE_URL", "")
    elif kind in {"deepseek", "ds"}:
        resolved_model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        resolved_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        resolved_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "") or "https://api.deepseek.com"
    elif kind in {"qwen", "dashscope"}:
        resolved_model = model or os.getenv("QWEN_MODEL", "qwen3.7-plus")
        resolved_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        resolved_url = (
            base_url
            or os.getenv("QWEN_BASE_URL", "")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    elif kind in {"custom", "openai-compatible"}:
        if not (base_url or os.getenv("SORA_BASE_URL")):
            raise ProviderConfigError("custom provider requires --base-url or SORA_BASE_URL")
        resolved_model = model or os.getenv("SORA_CUSTOM_MODEL", "gpt-5.4-mini")
        resolved_key = api_key or os.getenv("SORA_CUSTOM_API_KEY", "")
        resolved_url = base_url or os.getenv("SORA_BASE_URL", "")
    else:
        raise ValueError("unknown live provider kind: {}".format(kind))

    if not resolved_key:
        resolved_key = _saved_api_key(kind)
    if kind not in {"custom", "openai-compatible"} and not resolved_key:
        env_name = {
            "openai": "OPENAI_API_KEY",
            "azure": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "ds": "DEEPSEEK_API_KEY",
            "qwen": "DASHSCOPE_API_KEY",
            "dashscope": "DASHSCOPE_API_KEY",
        }.get(kind, "API_KEY")
        raise ProviderConfigError(
            "{} requires an API key: set {} or pass --api-key".format(kind, env_name)
        )

    return CacheAwareOpenAIProvider(
        model=resolved_model,
        api_key=resolved_key,
        base_url=resolved_url or None,
        timeout=timeout,
        temperature=temperature,
        require_api_key=kind not in {"custom", "openai-compatible"},
    )
