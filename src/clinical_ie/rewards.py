from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import ValidationError

from src.clinical_ie.evaluator import evaluate_output
from src.clinical_ie.schema import ExtractionOutput, parse_completion, validate_evidence_ids


@dataclass
class RewardResult:
    reward: float
    components: dict[str, float]
    parsed_output: dict[str, Any] | None
    failure_type: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_completion(
    completion: str | dict[str, Any],
    gold_output: str | dict[str, Any],
    sentence_ids: set[str],
    acceptable_evidence_groups: dict[str, Any] | None = None,
) -> RewardResult:
    try:
        predicted = parse_completion(completion)
    except json.JSONDecodeError as exc:
        return RewardResult(0.0, {}, None, "json_parse_error", str(exc))
    except (ValueError, ValidationError) as exc:
        return RewardResult(0.0, {}, None, "schema_error", str(exc))
    try:
        validate_evidence_ids(predicted, sentence_ids)
    except ValueError as exc:
        return RewardResult(0.0, {}, predicted.model_dump(mode="json"), "invalid_evidence_id", str(exc))
    try:
        expected = parse_completion(gold_output)
        validate_evidence_ids(expected, sentence_ids)
    except (ValueError, ValidationError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid gold authority: {exc}") from exc
    metrics = evaluate_output(predicted, expected, acceptable_evidence_groups)
    components = {
        "condition_object_f1": metrics.condition_object_f1,
        "condition_attribute_f1": metrics.condition_attribute_f1,
        "encounter_object_f1": metrics.encounter_object_f1,
        "relation_f1": metrics.relation_f1,
        "evidence_f1": metrics.evidence_f1,
        "global_consistency": metrics.global_consistency,
        "unsupported_fact_rate": metrics.unsupported_fact_rate,
        "missing_fact_rate": metrics.missing_fact_rate,
        "graph_f1": metrics.graph_f1,
        "record_exact": metrics.record_exact,
    }
    reward = 1.0 if metrics.record_exact == 1.0 else max(
        0.0, min(1.0, metrics.graph_f1 - 0.20 * metrics.unsupported_fact_rate)
    )
    failure_type = None
    if metrics.unsupported_fact_rate > 0:
        unmatched_concepts = {item.concept_id for item in predicted.conditions} - {item.concept_id for item in expected.conditions}
        failure_type = "unsupported_condition" if unmatched_concepts else "unsupported_fact"
    elif metrics.record_exact < 1.0:
        failure_type = "semantic_mismatch"
    return RewardResult(reward, components, predicted.model_dump(mode="json"), failure_type, None)
