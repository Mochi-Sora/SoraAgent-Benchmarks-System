"""Canonical request rendering and prompt-cache accounting.

Prompt-cache behaviour is a property of the exact bytes sent to the model.  The
recorder snapshots every provider call, renders it deterministically, and this
module derives:

* how much of each request is a byte-for-byte prefix of the previous request
  (the part a provider's prefix cache can reuse);
* which component caused the first divergence (clock, identity, state, tools,
  a new message, ...);
* per-component token attribution;
* cache hit rate and a stability score that penalizes churn.

The model is provider-agnostic.  A real provider may have a minimum cache block
(1024 tokens for some vendors); ``block_tokens`` lets the caller model that.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .tokens import estimate_tokens

CLOCK_PREFIX = "## CURRENT TIME"
IDENTITY_PREFIX = "## PERSISTENT USER PROFILE"
STATE_PREFIX = "## SESSION STATE PROTOCOL"


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def message_repr(message: Any) -> str:
    """Stable textual representation of one message for hashing/LCP."""

    role = str(getattr(message, "role", "") or "")
    content = str(getattr(message, "content", "") or "")
    parts = [role, "\x1f", content]
    calls = getattr(message, "tool_calls", ()) or ()
    if calls:
        rendered = []
        for call in calls:
            rendered.append(
                "{}:{}".format(
                    getattr(call, "name", ""),
                    json.dumps(
                        getattr(call, "arguments", {}),
                        sort_keys=True,
                        ensure_ascii=False,
                        default=_json_default,
                    ),
                )
            )
        parts.extend(["\x1f", "|".join(sorted(rendered))])
    name = str(getattr(message, "name", "") or "")
    if name:
        parts.extend(["\x1f", name])
    call_id = str(getattr(message, "tool_call_id", "") or "")
    if call_id:
        parts.extend(["\x1e", call_id])
    return "".join(parts)


def messages_repr(messages: Sequence[Any]) -> List[str]:
    return [message_repr(message) for message in messages]


def tool_schema(tool: Any) -> Dict[str, Any]:
    try:
        schema = tool.schema
        if isinstance(schema, dict):
            return schema
    except Exception:  # noqa: BLE001 - a broken schema is still worth measuring
        pass
    return {"name": str(getattr(tool, "name", "")), "description": "", "parameters": {}}


def tools_repr(tools: Sequence[Any]) -> str:
    schemas = [tool_schema(tool) for tool in tools]
    schemas.sort(key=lambda item: str(item.get("name", "")))
    return json.dumps(schemas, sort_keys=True, ensure_ascii=False, default=_json_default)


def canonical_request(messages: Sequence[Any], tools: Sequence[Any]) -> Tuple[str, str, str]:
    """Return ``(full, message_text, tool_text)`` canonical strings."""

    message_text = "\x1e".join(messages_repr(messages))
    tool_text = tools_repr(tools)
    # Tools are part of the model's prompt and normally precede the
    # conversation.  Keeping them first also makes tool-schema churn visible in
    # the prefix, which is exactly the failure mode this benchmark tracks.
    full = "TOOLS\x1d" + tool_text + "\x1eMESSAGES\x1d" + message_text
    return full, message_text, tool_text


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def common_prefix_chars(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def common_prefix_tokens(left: str, right: str) -> int:
    if not left or not right:
        return 0
    chars = common_prefix_chars(left, right)
    if chars <= 0:
        return 0
    return estimate_tokens(left[:chars])


def _system_kind(content: str) -> str:
    text = str(content or "")
    if text.startswith(CLOCK_PREFIX):
        return "clock"
    if text.startswith(IDENTITY_PREFIX):
        return "identity"
    if text.startswith(STATE_PREFIX):
        return "state"
    return "system"


def segment_request(messages: Sequence[Any], tools: Sequence[Any]) -> Dict[str, Any]:
    """Token attribution and hashes for one provider request."""

    segments = {
        "tool_schema_tokens": 0,
        "system_tokens": 0,
        "identity_tokens": 0,
        "clock_tokens": 0,
        "state_tokens": 0,
        "user_tokens": 0,
        "assistant_tokens": 0,
        "tool_result_tokens": 0,
    }
    base_systems: List[str] = []
    identity_parts: List[str] = []
    clock_parts: List[str] = []
    state_parts: List[str] = []

    for message in messages:
        rendered = message_repr(message)
        tokens = estimate_tokens(rendered)
        role = str(getattr(message, "role", "") or "")
        content = str(getattr(message, "content", "") or "")
        if role == "system":
            kind = _system_kind(content)
            segments[kind + "_tokens"] += tokens
            if kind == "system":
                base_systems.append(content)
            elif kind == "identity":
                identity_parts.append(content)
            elif kind == "clock":
                clock_parts.append(content)
            elif kind == "state":
                state_parts.append(content)
        elif role == "user":
            segments["user_tokens"] += tokens
        elif role == "assistant":
            segments["assistant_tokens"] += tokens
        elif role == "tool":
            segments["tool_result_tokens"] += tokens

    tool_text = tools_repr(tools)
    segments["tool_schema_tokens"] = estimate_tokens(tool_text)
    segments["system_hash"] = content_hash("\x1e".join(base_systems))
    segments["identity_hash"] = content_hash("\x1e".join(identity_parts))
    segments["clock_hash"] = content_hash("\x1e".join(clock_parts))
    segments["state_hash"] = content_hash("\x1e".join(state_parts))
    segments["tool_schema_hash"] = content_hash(tool_text)
    segments["message_count"] = len(messages)
    segments["tool_count"] = len(tools)
    return segments


def _first_divergence(previous: Sequence[Any], current: Sequence[Any]) -> Tuple[int, str, str]:
    """Return ``(index, kind, role)`` of the first differing message.

    ``kind`` is ``append`` when the previous request is a strict prefix (the
    healthy case), otherwise it names the component that changed.
    """

    prev = messages_repr(previous)
    cur = messages_repr(current)
    limit = min(len(prev), len(cur))
    for index in range(limit):
        if prev[index] != cur[index]:
            role = str(getattr(current[index], "role", "") or "")
            content = str(getattr(current[index], "content", "") or "")
            previous_role = str(getattr(previous[index], "role", "") or "")
            previous_content = str(getattr(previous[index], "content", "") or "")
            previous_clock = previous_role == "system" and previous_content.startswith(CLOCK_PREFIX)
            current_clock = role == "system" and content.startswith(CLOCK_PREFIX)
            if previous_clock or current_clock:
                # A removed/replaced trailing clock is the root cause even when
                # the current message at that position is conversation history.
                return index, "clock", role
            if role == "system":
                kind = _system_kind(content)
            else:
                kind = role or "unknown"
            return index, kind, role
    if len(cur) > len(prev):
        return len(prev), "append", ""
    if len(prev) > len(cur):
        return len(cur), "truncated", ""
    return -1, "none", ""


def build_request_record(
    index: int,
    step: int,
    messages: Sequence[Any],
    tools: Sequence[Any],
    *,
    previous_messages: Optional[Sequence[Any]],
    first_messages: Optional[Sequence[Any]],
    previous_full: str = "",
    first_full: str = "",
    latency_ms: float = 0.0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    reported_prompt_tokens: int = 0,
    reported_completion_tokens: int = 0,
    cache_usage: Optional[Dict[str, Any]] = None,
    model: str = "",
    min_cache_block: int = 256,
):
    """Create a :class:`RequestRecord` plus the canonical strings for next time."""

    from .types import RequestRecord

    full, message_text, tool_text = canonical_request(messages, tools)
    segments = segment_request(messages, tools)
    previous_full = previous_full or ""
    first_full = first_full or ""
    lcp_prev = common_prefix_tokens(previous_full, full) if previous_full else 0
    lcp_first = common_prefix_tokens(first_full, full) if first_full else 0
    lcp_message_prev = 0
    first_diff_kind = "none"
    first_diff_role = ""
    if previous_messages is not None:
        prev_message_text = "\x1e".join(messages_repr(previous_messages))
        lcp_message_prev = common_prefix_tokens(prev_message_text, message_text)
        _, first_diff_kind, first_diff_role = _first_divergence(previous_messages, messages)
        prev_message_tokens = estimate_tokens(prev_message_text)
        cur_message_tokens = estimate_tokens(message_text)
        denominator = max(1, min(prev_message_tokens, cur_message_tokens))
        message_prefix_ratio = lcp_message_prev / float(denominator)
    else:
        message_prefix_ratio = 0.0

    prev_tools = ""
    if previous_messages is not None and previous_full:
        marker = previous_full.find("\x1eMESSAGES\x1d")
        if marker >= 0:
            prev_tools = previous_full[len("TOOLS\x1d") : marker]
    tool_schema_changed = bool(prev_tools) and prev_tools != tool_text
    clock_changed = False
    identity_changed = False
    system_changed = False
    if previous_messages is not None:
        prev_segments = segment_request(previous_messages, tools)
        clock_changed = bool(prev_segments.get("clock_hash")) and prev_segments.get(
            "clock_hash"
        ) != segments.get("clock_hash")
        identity_changed = bool(prev_segments.get("identity_hash")) and prev_segments.get(
            "identity_hash"
        ) != segments.get("identity_hash")
        system_changed = prev_segments.get("system_hash") != segments.get("system_hash")
    cache_usage = cache_usage or {}
    clock_trailing = False
    if messages:
        last = messages[-1]
        clock_trailing = str(getattr(last, "role", "") or "") == "system" and str(
            getattr(last, "content", "") or ""
        ).startswith(CLOCK_PREFIX)
    record = RequestRecord(
        index=index,
        step=step,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reported_prompt_tokens=reported_prompt_tokens,
        reported_completion_tokens=reported_completion_tokens,
        reported_cached_tokens=int(cache_usage.get("cached_tokens") or 0),
        reported_cache_write_tokens=int(cache_usage.get("cache_write_tokens") or 0),
        cache_usage_source=str(cache_usage.get("source") or ""),
        latency_ms=latency_ms,
        prompt_chars=len(full),
        message_count=segments["message_count"],
        tool_count=segments["tool_count"],
        tool_schema_tokens=segments["tool_schema_tokens"],
        system_tokens=segments["system_tokens"],
        identity_tokens=segments["identity_tokens"],
        clock_tokens=segments["clock_tokens"],
        state_tokens=segments["state_tokens"],
        user_tokens=segments["user_tokens"],
        assistant_tokens=segments["assistant_tokens"],
        tool_result_tokens=segments["tool_result_tokens"],
        lcp_tokens_prev=lcp_prev,
        lcp_tokens_first=lcp_first,
        lcp_message_tokens_prev=lcp_message_prev,
        message_prefix_ratio=message_prefix_ratio,
        first_diff_kind=first_diff_kind,
        first_diff_role=first_diff_role,
        tool_schema_changed=tool_schema_changed,
        clock_changed=clock_changed,
        identity_changed=identity_changed,
        system_changed=system_changed,
        prompt_hash=content_hash(full),
        message_hash=content_hash(message_text),
        tool_schema_hash=segments["tool_schema_hash"],
        system_hash=segments["system_hash"],
        clock_hash=segments["clock_hash"],
        identity_hash=segments["identity_hash"],
        cache_eligible=lcp_prev >= max(0, int(min_cache_block)),
        clock_trailing=clock_trailing,
        model=model,
    )
    return record, full, message_text


def _message_tokens(request: Any) -> int:
    return max(
        0,
        int(request.system_tokens)
        + int(request.identity_tokens)
        + int(request.clock_tokens)
        + int(request.state_tokens)
        + int(request.user_tokens)
        + int(request.assistant_tokens)
        + int(request.tool_result_tokens),
    )


def analyze_cache(requests: Sequence[Any], *, min_cache_block: int = 256) -> Dict[str, Any]:
    """Aggregate prompt-cache rate and stability for one run."""

    requests = list(requests)
    if not requests:
        return {
            "requests": 0,
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "cache_hit_rate": 0.0,
            "message_tokens": 0,
            "message_cached_tokens": 0,
            "message_cache_hit_rate": 0.0,
            "message_retention_mean": 0.0,
            "message_retention_stdev": 0.0,
            "message_stability": 0.0,
            "clock_trailing_rate": 0.0,
            "cacheable_request_rate": 0.0,
            "prefix_retention_mean": 0.0,
            "prefix_retention_stdev": 0.0,
            "prefix_retention_min": 0.0,
            "stable_prefix_tokens": 0,
            "break_events": 0,
            "tool_schema_churn_rate": 0.0,
            "clock_churn_rate": 0.0,
            "identity_churn_rate": 0.0,
            "divergence_roles": {},
            "reported_cache_hit_rate": 0.0,
            "reported_cache_available": False,
            "cache_stability": 0.0,
        }

    total_prompt = sum(max(0, int(r.prompt_tokens)) for r in requests)
    total_message_tokens = sum(_message_tokens(r) for r in requests)
    cached = 0
    message_cached = 0
    retentions: List[float] = []
    message_retentions: List[float] = []
    eligible = 0
    tool_churn = 0
    clock_churn = 0
    identity_churn = 0
    breaks = 0
    divergence: Dict[str, int] = {}
    for index, request in enumerate(requests):
        if index == 0:
            continue
        reusable = min(
            max(0, int(request.lcp_tokens_prev)),
            max(0, int(request.prompt_tokens)),
        )
        if reusable >= max(0, int(min_cache_block)):
            cached += reusable
            eligible += 1
        message_reusable = min(
            max(0, int(request.lcp_message_tokens_prev)),
            _message_tokens(request),
        )
        message_cached += message_reusable
        previous_message_tokens = max(1, _message_tokens(requests[index - 1]))
        current_message_tokens = max(1, _message_tokens(request))
        message_retentions.append(
            message_reusable / float(min(previous_message_tokens, current_message_tokens))
        )
        previous_prompt = max(1, int(requests[index - 1].prompt_tokens))
        current_prompt = max(1, int(request.prompt_tokens))
        denominator = max(1, min(previous_prompt, current_prompt))
        retention = reusable / float(denominator)
        retentions.append(retention)
        if retention < 0.5:
            breaks += 1
        if request.tool_schema_changed:
            tool_churn += 1
        if request.clock_changed:
            clock_churn += 1
        if request.identity_changed:
            identity_churn += 1
        key = request.first_diff_kind or "unknown"
        divergence[key] = divergence.get(key, 0) + 1

    transitions = max(1, len(requests) - 1)
    reported_prompt = sum(max(0, int(r.reported_prompt_tokens)) for r in requests)
    reported_cached = sum(max(0, int(r.reported_cached_tokens)) for r in requests)
    reported_available = any(r.cache_usage_source or r.reported_cached_tokens for r in requests)
    mean_retention = statistics.fmean(retentions) if retentions else 0.0
    stdev_retention = statistics.pstdev(retentions) if len(retentions) > 1 else 0.0
    cv = (stdev_retention / mean_retention) if mean_retention > 0 else 1.0
    stability = max(0.0, min(1.0, mean_retention * (1.0 - min(1.0, cv))))
    message_mean = statistics.fmean(message_retentions) if message_retentions else 0.0
    message_stdev = statistics.pstdev(message_retentions) if len(message_retentions) > 1 else 0.0
    message_cv = (message_stdev / message_mean) if message_mean > 0 else 1.0
    message_stability = max(0.0, min(1.0, message_mean * (1.0 - min(1.0, message_cv))))
    clock_requests = [r for r in requests if r.clock_hash]
    clock_trailing_rate = (
        sum(1 for r in clock_requests if r.clock_trailing) / float(len(clock_requests))
        if clock_requests
        else 0.0
    )
    return {
        "requests": len(requests),
        "prompt_tokens": total_prompt,
        "cached_tokens": cached,
        "cache_hit_rate": (cached / total_prompt) if total_prompt else 0.0,
        "message_tokens": total_message_tokens,
        "message_cached_tokens": message_cached,
        "message_cache_hit_rate": (
            message_cached / total_message_tokens if total_message_tokens else 0.0
        ),
        "message_retention_mean": message_mean,
        "message_retention_stdev": message_stdev,
        "message_stability": message_stability,
        "clock_trailing_rate": clock_trailing_rate,
        "cacheable_request_rate": eligible / transitions,
        "prefix_retention_mean": mean_retention,
        "prefix_retention_stdev": stdev_retention,
        "prefix_retention_min": min(retentions) if retentions else 0.0,
        "stable_prefix_tokens": int(statistics.median([r.lcp_tokens_prev for r in requests[1:]]))
        if len(requests) > 1
        else 0,
        "break_events": breaks,
        "tool_schema_churn_rate": tool_churn / transitions,
        "clock_churn_rate": clock_churn / transitions,
        "identity_churn_rate": identity_churn / transitions,
        "divergence_roles": divergence,
        "reported_cache_hit_rate": (reported_cached / reported_prompt if reported_prompt else 0.0),
        "reported_cache_available": reported_available,
        "cache_stability": stability,
    }


def aggregate_run_cache(runs: Iterable[Any], *, min_cache_block: int = 256) -> Dict[str, Any]:
    """Aggregate cache analysis across runs, including cross-run stability."""

    runs = list(runs)
    per_run = [analyze_cache(run.requests, min_cache_block=min_cache_block) for run in runs]
    total_prompt = sum(item["prompt_tokens"] for item in per_run)
    total_cached = sum(item["cached_tokens"] for item in per_run)
    total_message_tokens = sum(item.get("message_tokens", 0) for item in per_run)
    total_message_cached = sum(item.get("message_cached_tokens", 0) for item in per_run)
    message_stability_values = [item.get("message_stability", 0.0) for item in per_run]
    clock_trailing_values = [item.get("clock_trailing_rate", 0.0) for item in per_run]
    first_prompt_hashes = {run.requests[0].prompt_hash for run in runs if run.requests}
    first_system_hashes = {run.requests[0].system_hash for run in runs if run.requests}
    first_tool_hashes = {run.requests[0].tool_schema_hash for run in runs if run.requests}
    hit_rates = [item["cache_hit_rate"] for item in per_run]
    stabilities = [item["cache_stability"] for item in per_run]
    divergence: Dict[str, int] = {}
    for item in per_run:
        for key, count in item["divergence_roles"].items():
            divergence[key] = divergence.get(key, 0) + int(count)
    reported_prompt = sum(
        request.reported_prompt_tokens for run in runs for request in run.requests
    )
    reported_cached = sum(
        request.reported_cached_tokens for run in runs for request in run.requests
    )
    reported_available = any(
        request.cache_usage_source or request.reported_cached_tokens
        for run in runs
        for request in run.requests
    )
    return {
        "run_count": len(runs),
        "request_count": sum(item["requests"] for item in per_run),
        "prompt_tokens": total_prompt,
        "cached_tokens": total_cached,
        "cache_hit_rate": (total_cached / total_prompt) if total_prompt else 0.0,
        "message_tokens": total_message_tokens,
        "message_cached_tokens": total_message_cached,
        "message_cache_hit_rate": (
            total_message_cached / total_message_tokens if total_message_tokens else 0.0
        ),
        "message_stability_mean": (
            statistics.fmean(message_stability_values) if message_stability_values else 0.0
        ),
        "clock_trailing_rate_mean": (
            statistics.fmean(clock_trailing_values) if clock_trailing_values else 0.0
        ),
        "cache_hit_rate_mean": statistics.fmean(hit_rates) if hit_rates else 0.0,
        "cache_hit_rate_stdev": (statistics.pstdev(hit_rates) if len(hit_rates) > 1 else 0.0),
        "cache_stability_mean": (statistics.fmean(stabilities) if stabilities else 0.0),
        "cache_stability_min": min(stabilities) if stabilities else 0.0,
        "cross_run_prompt_prefix_identical": len(first_prompt_hashes) <= 1,
        "cross_run_system_prefix_identical": len(first_system_hashes) <= 1,
        "cross_run_tool_prefix_identical": len(first_tool_hashes) <= 1,
        "reported_cache_hit_rate": (reported_cached / reported_prompt if reported_prompt else 0.0),
        "reported_cache_available": reported_available,
        "divergence_roles": divergence,
        "per_run": per_run,
    }
