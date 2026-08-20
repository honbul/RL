from __future__ import annotations

from typing import Iterable


def set_f1(predicted: Iterable[str], expected: Iterable[str]) -> float:
    pred = set(predicted)
    gold = set(expected)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    true_positive = len(pred & gold)
    precision = true_positive / len(pred)
    recall = true_positive / len(gold)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def count_f1(matched: int, predicted: int, expected: int) -> float:
    if predicted == 0 and expected == 0:
        return 1.0
    if predicted == 0 or expected == 0:
        return 0.0
    precision = matched / predicted
    recall = matched / expected
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
