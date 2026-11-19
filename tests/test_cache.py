from __future__ import annotations

from sora_agent.providers import Message

from sora_bench.protocol import (
    analyze_cache,
    build_request_record,
    canonical_request,
)
from sora_bench.tokens import estimate_tokens


def _record(messages, tools, previous=None, first=None, previous_full="", first_full=""):
    full, _message_text, _tool_text = canonical_request(messages, tools)
    record, full, _ = build_request_record(
        index=0,
        step=0,
        messages=messages,
        tools=tools,
        previous_messages=previous,
        first_messages=first,
        previous_full=previous_full,
        first_full=first_full,
        prompt_tokens=estimate_tokens(full),
        completion_tokens=4,
        min_cache_block=0,
    )
    return record, full


def test_appended_conversation_reuses_prefix():
    messages1 = [Message("system", "stable system"), Message("user", "hello")]
    tools = []
    first, full_first = _record(messages1, tools)
    messages2 = messages1 + [
        Message("assistant", "working"),
        Message("tool", "result", tool_call_id="1", name="lookup"),
    ]
    second, _full_second = _record(
        messages2,
        tools,
        previous=messages1,
        first=messages1,
        previous_full=full_first,
        first_full=full_first,
    )
    analysis = analyze_cache([first, second], min_cache_block=0)
    assert second.first_diff_kind == "append"
    assert second.lcp_tokens_prev > 0
    assert analysis["cache_hit_rate"] > 0
    assert analysis["clock_churn_rate"] == 0


def test_changing_clock_breaks_prefix_early():
    stable = Message("system", "stable system")
    messages1 = [
        stable,
        Message("system", "## CURRENT TIME\n10:00"),
        Message("user", "hi"),
    ]
    first, full_first = _record(messages1, [])
    messages2 = [
        stable,
        Message("system", "## CURRENT TIME\n10:01"),
        Message("user", "hi"),
        Message("assistant", "ok"),
    ]
    second, _full_second = _record(
        messages2,
        [],
        previous=messages1,
        first=messages1,
        previous_full=full_first,
        first_full=full_first,
    )
    analysis = analyze_cache([first, second], min_cache_block=0)
    assert second.clock_changed is True
    assert second.first_diff_kind == "clock"
    assert analysis["clock_churn_rate"] == 1.0
    assert second.lcp_tokens_prev < estimate_tokens(full_first)


def test_tool_schema_churn_is_detected():
    class FakeTool:
        def __init__(self, name, description):
            self.name = name
            self.description = description

        @property
        def schema(self):
            return {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}},
            }

    messages = [Message("system", "stable"), Message("user", "go")]
    first, full_first = _record(messages, [FakeTool("a", "one")])
    second, _full_second = _record(
        messages,
        [FakeTool("a", "two")],
        previous=messages,
        first=messages,
        previous_full=full_first,
        first_full=full_first,
    )
    assert second.tool_schema_changed is True
    assert second.lcp_tokens_prev < estimate_tokens(full_first)


def test_extract_cache_usage_vendor_shapes():
    from sora_bench.providers import extract_cache_usage

    openai_shape = {
        "prompt_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 60},
    }
    assert extract_cache_usage(openai_shape)["cached_tokens"] == 60

    deepseek_shape = {
        "prompt_tokens": 100,
        "prompt_cache_hit_tokens": 40,
        "prompt_cache_miss_tokens": 60,
    }
    assert extract_cache_usage(deepseek_shape)["cached_tokens"] == 40

    anthropic_shape = {
        "cache_read_input_tokens": 30,
        "cache_creation_input_tokens": 70,
    }
    usage = extract_cache_usage(anthropic_shape)
    assert usage["cached_tokens"] == 30
    assert usage["cache_write_tokens"] == 70


def test_extract_cache_usage_absent_returns_empty():
    from sora_bench.providers import extract_cache_usage

    assert extract_cache_usage(None) == {}
    assert extract_cache_usage({"prompt_tokens": 10}) == {}


def test_clock_trailing_detection():
    base = [Message("system", "static"), Message("user", "task")]
    trailing = base + [Message("system", "## CURRENT TIME\n10:00")]
    record, _full = _record(trailing, [])
    assert record.clock_trailing is True

    leading = [
        Message("system", "static"),
        Message("system", "## CURRENT TIME\n10:00"),
        Message("user", "task"),
    ]
    record, _full = _record(leading, [])
    assert record.clock_trailing is False
