from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Callable, TypeVar

from src.clinical_ie.canonicalize import count_f1, set_f1
from src.clinical_ie.schema import Condition, Encounter, ExtractionOutput

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class ClinicalMetrics:
    condition_object_f1: float
    condition_attribute_f1: float
    encounter_object_f1: float
    relation_f1: float
    evidence_f1: float
    global_consistency: float
    unsupported_fact_rate: float
    missing_fact_rate: float
    graph_f1: float
    record_exact: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _maximum_pairs(predicted: list[T], expected: list[U], score: Callable[[T, U], float]) -> list[tuple[int, int]]:
    matrix = [[score(pred, gold) for gold in expected] for pred in predicted]

    @lru_cache(maxsize=None)
    def solve(pred_index: int, used_mask: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if pred_index == len(predicted):
            return 0.0, ()
        best_score, best_pairs = solve(pred_index + 1, used_mask)
        for gold_index in range(len(expected)):
            value = matrix[pred_index][gold_index]
            if value <= 0 or used_mask & (1 << gold_index):
                continue
            tail_score, tail_pairs = solve(pred_index + 1, used_mask | (1 << gold_index))
            candidate = (value + tail_score, ((pred_index, gold_index), *tail_pairs))
            if candidate[0] > best_score or (candidate[0] == best_score and candidate[1] < best_pairs):
                best_score, best_pairs = candidate
        return best_score, best_pairs

    return list(solve(0, 0)[1])


def _condition_score(predicted: Condition, expected: Condition) -> float:
    if predicted.concept_id != expected.concept_id:
        return 0.0
    return 1.0 + 0.5 * (predicted.assertion == expected.assertion) + 0.3 * (predicted.event_time == expected.event_time) + 0.2 * set_f1(predicted.evidence_sentence_ids, expected.evidence_sentence_ids)


def _encounter_score(predicted: Encounter, expected: Encounter) -> float:
    if predicted.facility != expected.facility:
        return 0.0
    attributes = [
        predicted.date == expected.date,
        predicted.encounter_type == expected.encounter_type,
        predicted.reason_concept_id == expected.reason_concept_id,
        predicted.outcome == expected.outcome,
    ]
    return 1.0 + sum(attributes) / len(attributes) + 0.2 * set_f1(predicted.evidence_sentence_ids, expected.evidence_sentence_ids)


def _acceptable_groups(
    acceptable: dict[str, Any], kind: str, gold_id: str, fallback: list[str]
) -> list[list[str]]:
    groups = acceptable.get(kind, {}).get(gold_id)
    if not groups:
        return [fallback]
    return groups


def evaluate_output(
    predicted: ExtractionOutput,
    expected: ExtractionOutput,
    acceptable_evidence_groups: dict[str, Any] | None = None,
) -> ClinicalMetrics:
    acceptable = acceptable_evidence_groups or {}
    condition_pairs = _maximum_pairs(predicted.conditions, expected.conditions, _condition_score)
    encounter_pairs = _maximum_pairs(predicted.encounters, expected.encounters, _encounter_score)
    condition_f1 = count_f1(len(condition_pairs), len(predicted.conditions), len(expected.conditions))
    attribute_parts = []
    for pred_index, gold_index in condition_pairs:
        pred = predicted.conditions[pred_index]
        gold = expected.conditions[gold_index]
        attribute_parts.extend([float(pred.assertion == gold.assertion), float(pred.event_time == gold.event_time)])
    condition_attribute_f1 = sum(attribute_parts) / len(attribute_parts) if attribute_parts else (1.0 if not predicted.conditions and not expected.conditions else 0.0)
    strict_encounters = sum(
        predicted.encounters[p].date == expected.encounters[g].date
        and predicted.encounters[p].encounter_type == expected.encounters[g].encounter_type
        and predicted.encounters[p].reason_concept_id == expected.encounters[g].reason_concept_id
        and predicted.encounters[p].outcome == expected.encounters[g].outcome
        for p, g in encounter_pairs
    )
    encounter_f1 = count_f1(strict_encounters, len(predicted.encounters), len(expected.encounters))

    condition_map = {predicted.conditions[p].id: expected.conditions[g].id for p, g in condition_pairs}
    encounter_map = {predicted.encounters[p].id: expected.encounters[g].id for p, g in encounter_pairs}
    predicted_relations = {
        (condition_map[r.condition_id], encounter_map[r.encounter_id], r.relation_type)
        for r in predicted.relations
        if r.condition_id in condition_map and r.encounter_id in encounter_map
    }
    expected_relations = {(r.condition_id, r.encounter_id, r.relation_type) for r in expected.relations}
    relation_f1 = set_f1(predicted_relations, expected_relations)

    evidence_scores = []
    for pred_index, gold_index in condition_pairs:
        pred = predicted.conditions[pred_index]
        gold = expected.conditions[gold_index]
        groups = _acceptable_groups(acceptable, "conditions", gold.id, gold.evidence_sentence_ids)
        evidence_scores.append(max(set_f1(pred.evidence_sentence_ids, group) for group in groups))
    for pred_index, gold_index in encounter_pairs:
        pred = predicted.encounters[pred_index]
        gold = expected.encounters[gold_index]
        groups = _acceptable_groups(acceptable, "encounters", gold.id, gold.evidence_sentence_ids)
        evidence_scores.append(max(set_f1(pred.evidence_sentence_ids, group) for group in groups))
    expected_objects = len(expected.conditions) + len(expected.encounters)
    evidence_f1 = sum(evidence_scores) / expected_objects if expected_objects else (1.0 if not predicted.conditions and not predicted.encounters else 0.0)

    unmatched_conditions = len(predicted.conditions) - len(condition_pairs)
    unmatched_encounters = len(predicted.encounters) - len(encounter_pairs)
    mapped_relation_count = len(predicted_relations)
    unsupported_relations = len(predicted.relations) - mapped_relation_count
    predicted_fact_count = len(predicted.conditions) + len(predicted.encounters) + len(predicted.relations)
    unsupported_count = unmatched_conditions + unmatched_encounters + unsupported_relations
    unsupported_rate = unsupported_count / predicted_fact_count if predicted_fact_count else 0.0
    missing_count = (len(expected.conditions) - len(condition_pairs)) + (len(expected.encounters) - len(encounter_pairs)) + len(expected_relations - predicted_relations)
    expected_fact_count = len(expected.conditions) + len(expected.encounters) + len(expected.relations)
    missing_rate = missing_count / expected_fact_count if expected_fact_count else 0.0
    global_consistency = 1.0
    graph_f1 = (
        0.30 * condition_f1
        + 0.15 * condition_attribute_f1
        + 0.20 * encounter_f1
        + 0.10 * relation_f1
        + 0.15 * evidence_f1
        + 0.10 * global_consistency
    )
    exact = float(
        condition_f1 == 1.0
        and condition_attribute_f1 == 1.0
        and encounter_f1 == 1.0
        and relation_f1 == 1.0
        and evidence_f1 == 1.0
        and unsupported_rate == 0.0
        and missing_rate == 0.0
    )
    if exact == 1.0:
        graph_f1 = 1.0
    return ClinicalMetrics(
        condition_object_f1=condition_f1,
        condition_attribute_f1=condition_attribute_f1,
        encounter_object_f1=encounter_f1,
        relation_f1=relation_f1,
        evidence_f1=evidence_f1,
        global_consistency=global_consistency,
        unsupported_fact_rate=unsupported_rate,
        missing_fact_rate=missing_rate,
        graph_f1=graph_f1,
        record_exact=exact,
    )
