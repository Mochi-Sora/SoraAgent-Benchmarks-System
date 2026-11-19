from __future__ import annotations

from sora_bench.scoring import match_expected_calls


def test_exact_match():
    actual = [
        {"name": "lookup_order", "arguments": {"order_id": "PO-1"}},
        {"name": "expedite_order", "arguments": {"order_id": "PO-1"}},
    ]
    expected = [
        {"name": "lookup_order", "args": {"order_id": "PO-1"}},
        {"name": "expedite_order", "args": {"order_id": "PO-1"}},
    ]
    result = match_expected_calls(actual, expected)
    assert result["f1"] == 1.0
    assert result["exact_trajectory"] is True


def test_extra_and_missing_calls_reduce_precision_and_recall():
    actual = [
        {"name": "lookup_order", "arguments": {"order_id": "PO-1"}},
        {"name": "search_knowledge", "arguments": {"query": "noise"}},
    ]
    expected = [{"name": "lookup_order", "args": {"order_id": "PO-1"}}]
    result = match_expected_calls(actual, expected)
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0
    assert result["exact_trajectory"] is False
    assert result["unnecessary_call_rate"] == 0.5


def test_wrong_argument_is_name_match_but_argument_mismatch():
    actual = [{"name": "refund_order", "arguments": {"order_id": "PO-9", "amount": 10}}]
    expected = [{"name": "refund_order", "args": {"order_id": "PO-9", "amount": 240}}]
    result = match_expected_calls(actual, expected)
    assert result["argument_mismatches"] == 1
    assert result["exact_trajectory"] is False


def test_forbidden_call_is_reported():
    actual = [{"name": "refund_order", "arguments": {"order_id": "PO-1"}}]
    expected = []
    result = match_expected_calls(actual, expected, [{"name": "refund_order"}])
    assert result["forbidden_violations"] == 1
    assert result["exact_trajectory"] is False


def test_string_arguments_ignore_case_and_whitespace():
    actual = [
        {
            "name": "create_calendar_event",
            "arguments": {
                "title": "Project Review with Alice",
                "start": "2026-12-01T10:00:00",
            },
        }
    ]
    expected = [
        {
            "name": "create_calendar_event",
            "args": {
                "title": "Project review with Alice",
                "start": "2026-12-01T10:00:00",
            },
        }
    ]
    result = match_expected_calls(actual, expected)
    assert result["f1"] == 1.0
    assert result["exact_trajectory"] is True


def test_iso_timestamps_with_different_suffixes_match():
    actual = [
        {
            "name": "create_calendar_event",
            "arguments": {
                "title": "Project Review",
                "start": "2026-12-01T10:00",
            },
        }
    ]
    expected = [
        {
            "name": "create_calendar_event",
            "args": {
                "title": "Project review",
                "start": "2026-12-01T10:00:00Z",
            },
        }
    ]
    result = match_expected_calls(actual, expected)
    assert result["f1"] == 1.0
