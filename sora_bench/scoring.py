"""Scoring helpers shared by the benchmark suites."""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def canonical_args(arguments: Any) -> str:
    return json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False, default=str)


def call_signature(call: Any) -> str:
    if isinstance(call, dict):
        name = call.get("name", "")
        arguments = call.get("arguments", {})
    else:
        name = getattr(call, "name", "")
        arguments = getattr(call, "arguments", {})
    return "{}:{}".format(name, canonical_args(arguments))


def _name(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("name", ""))
    return str(getattr(call, "name", ""))


def _args(call: Any) -> Dict[str, Any]:
    if isinstance(call, dict):
        value = call.get("arguments", {})
    else:
        value = getattr(call, "arguments", {})
    return dict(value or {})


def _expected_name(expected: Any) -> str:
    if isinstance(expected, dict):
        return str(expected.get("name", ""))
    return str(getattr(expected, "name", ""))


def _expected_args(expected: Any) -> Dict[str, Any]:
    if isinstance(expected, dict):
        value = expected.get("args", expected.get("arguments", {}))
    else:
        value = getattr(expected, "args", getattr(expected, "arguments", {}))
    return dict(value or {})


def _iso(value: str) -> datetime | None:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _string_match(left: str, right: str) -> bool:
    first = " ".join(str(left).split()).casefold()
    second = " ".join(str(right).split()).casefold()
    if first == second:
        return True
    first_moment = _iso(left)
    second_moment = _iso(right)
    return bool(
        first_moment is not None and second_moment is not None and first_moment == second_moment
    )


def args_match(actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """True when every expected key is present with an equal value."""

    for key, value in expected.items():
        if key not in actual:
            return False
        if isinstance(value, float) and isinstance(actual[key], (int, float)):
            if not math.isclose(float(value), float(actual[key]), rel_tol=1e-9):
                return False
        elif isinstance(value, str) and isinstance(actual[key], str):
            if not _string_match(value, actual[key]):
                return False
        elif actual[key] != value:
            return False
    return True


def match_expected_calls(
    actual_calls: Sequence[Any],
    expected_calls: Sequence[Any],
    forbidden_calls: Sequence[Any] = (),
) -> Dict[str, Any]:
    """Match expected calls against actual calls, order-insensitively."""

    actual = list(actual_calls)
    expected = list(expected_calls)
    used = [False] * len(actual)
    matched: List[Tuple[int, int]] = []
    argument_mismatches = 0
    missing: List[Dict[str, Any]] = []

    for expected_index, item in enumerate(expected):
        name = _expected_name(item)
        args = _expected_args(item)
        found = -1
        name_only = -1
        for index, call in enumerate(actual):
            if used[index] or _name(call) != name:
                continue
            if args_match(_args(call), args):
                found = index
                break
            if name_only < 0:
                name_only = index
        if found >= 0:
            used[found] = True
            matched.append((expected_index, found))
        elif name_only >= 0:
            used[name_only] = True
            argument_mismatches += 1
            matched.append((expected_index, name_only))
        else:
            missing.append({"name": name, "args": args})

    unmatched = [
        {"name": _name(call), "arguments": _args(call)}
        for index, call in enumerate(actual)
        if not used[index]
    ]
    forbidden_hits: List[Dict[str, Any]] = []
    for item in forbidden_calls:
        name = _expected_name(item)
        args = _expected_args(item)
        for call in actual:
            if _name(call) == name and (not args or args_match(_args(call), args)):
                forbidden_hits.append({"name": name, "arguments": _args(call)})
                break

    matched_count = len(matched)
    precision = matched_count / float(len(actual)) if actual else 1.0
    recall = matched_count / float(len(expected)) if expected else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    exact = (
        len(actual) == len(expected)
        and len(matched) == len(expected)
        and argument_mismatches == 0
        and [actual_index for _exp_index, actual_index in matched] == list(range(len(expected)))
        and all(
            _name(actual[actual_index]) == _expected_name(expected[exp_index])
            and args_match(_args(actual[actual_index]), _expected_args(expected[exp_index]))
            for exp_index, actual_index in matched
        )
        and not forbidden_hits
    )
    return {
        "actual_count": len(actual),
        "expected_count": len(expected),
        "matched_count": matched_count,
        "missing": missing,
        "unmatched": unmatched,
        "forbidden": forbidden_hits,
        "argument_mismatches": argument_mismatches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_trajectory": bool(exact),
        "unnecessary_call_rate": (len(unmatched) / float(len(actual)) if actual else 0.0),
        "forbidden_violations": len(forbidden_hits),
    }


def summarize_calls(runs: Iterable[Any], tool_call_key: str = "tool_call_log") -> Dict[str, Any]:
    """Aggregate matching details from runs that carry ``evaluation['calls']``."""

    runs = list(runs)
    matches = [run.evaluation.get("calls", {}) for run in runs if isinstance(run.evaluation, dict)]
    if not matches:
        return {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "exact_rate": 0.0,
            "forbidden_rate": 0.0,
        }
    return {
        "f1": _mean(item.get("f1", 0.0) for item in matches),
        "precision": _mean(item.get("precision", 0.0) for item in matches),
        "recall": _mean(item.get("recall", 0.0) for item in matches),
        "exact_rate": _mean(1.0 if item.get("exact_trajectory") else 0.0 for item in matches),
        "forbidden_rate": _mean(
            1.0 if item.get("forbidden_violations") else 0.0 for item in matches
        ),
    }


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return statistics.fmean(items) if items else 0.0


def mean(values: Iterable[float]) -> float:
    return _mean(values)


def median(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return statistics.median(items) if items else 0.0


def stdev(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return statistics.pstdev(items) if len(items) > 1 else 0.0


def percentile(values: Iterable[float], fraction: float) -> float:
    items = sorted(float(value) for value in values)
    if not items:
        return 0.0
    if len(items) == 1:
        return items[0]
    position = max(0.0, min(1.0, float(fraction))) * (len(items) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return items[lower]
    weight = position - lower
    return items[lower] * (1.0 - weight) + items[upper] * weight


def p95(values: Iterable[float]) -> float:
    return percentile(values, 0.95)


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not denominator:
        return default
    return float(numerator) / float(denominator)


def score_higher_is_better(value: float, target: float) -> float:
    """Map ``value`` to 0..100; anything at or above target scores 100."""

    if target <= 0:
        return 100.0 if value >= 0 else 0.0
    return max(0.0, min(100.0, 100.0 * float(value) / float(target)))


def score_lower_is_better(value: float, target: float) -> float:
    """Map ``value`` to 0..100; anything at or below target scores 100."""

    value = max(0.0, float(value))
    if value == 0:
        return 100.0
    if target <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * float(target) / value))


def weighted_score(pairs: Sequence[Tuple[float, float]]) -> float:
    """Weighted mean of ``(score, weight)`` pairs, ignoring zero weights."""

    total_weight = sum(weight for _score, weight in pairs if weight > 0)
    if total_weight <= 0:
        return 0.0
    return sum(score * weight for score, weight in pairs if weight > 0) / total_weight


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))
