from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isclose
from typing import Any


def canonical_equal(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            canonical_equal(left[key], right[key], tolerance) for key in left
        )
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes))
        and not isinstance(right, (str, bytes))
    ):
        return len(left) == len(right) and all(
            canonical_equal(a, b, tolerance) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def flatten_leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_leaves(value[key], child))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            result.update(flatten_leaves(item, child))
        return result
    return {prefix: value}


def field_f1(predicted: Any, expected: Any, tolerance: float = 1e-6) -> float:
    pred = flatten_leaves(predicted)
    gold = flatten_leaves(expected)
    if not pred and not gold:
        return 1.0
    true_positive = sum(
        1 for path, value in pred.items()
        if path in gold and canonical_equal(value, gold[path], tolerance)
    )
    precision = true_positive / len(pred) if pred else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
