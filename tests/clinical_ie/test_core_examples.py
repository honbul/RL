from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.clinical_ie.rewards import score_completion


CORE_PATH = Path("data/clinical_ie/dev/core_examples_20.jsonl")


def load_rows() -> list[dict]:
    return [json.loads(line) for line in CORE_PATH.read_text(encoding="utf-8").splitlines()]


def test_twenty_gold_outputs_receive_full_reward() -> None:
    rows = load_rows()
    assert len(rows) == 20
    for row in rows:
        result = score_completion(
            json.dumps(row["gold_output"], ensure_ascii=False),
            row["gold_output"],
            set(row["sentence_ids"]),
            row["acceptable_evidence_groups"],
        )
        assert result.reward == 1.0, (row["sample_id"], result)
        assert result.components["graph_f1"] == 1.0


def test_invalid_json_is_gated_to_zero() -> None:
    row = load_rows()[0]
    result = score_completion("not-json", row["gold_output"], set(row["sentence_ids"]), row["acceptable_evidence_groups"])
    assert result.reward == 0.0
    assert result.failure_type == "json_parse_error"


def test_extra_field_is_gated_to_zero() -> None:
    row = load_rows()[0]
    bad = deepcopy(row["gold_output"])
    bad["unexpected"] = True
    result = score_completion(json.dumps(bad), row["gold_output"], set(row["sentence_ids"]), row["acceptable_evidence_groups"])
    assert result.reward == 0.0
    assert result.failure_type == "schema_error"


def test_invalid_evidence_id_is_gated_to_zero() -> None:
    row = load_rows()[0]
    bad = deepcopy(row["gold_output"])
    bad["conditions"][0]["evidence_sentence_ids"] = ["S999"]
    result = score_completion(json.dumps(bad), row["gold_output"], set(row["sentence_ids"]), row["acceptable_evidence_groups"])
    assert result.reward == 0.0
    assert result.failure_type == "invalid_evidence_id"


def test_dangling_relation_is_gated_to_zero() -> None:
    row = load_rows()[0]
    bad = deepcopy(row["gold_output"])
    bad["relations"] = [{"condition_id": "C999", "encounter_id": "E1", "relation_type": "diagnosed_at"}]
    result = score_completion(json.dumps(bad), row["gold_output"], set(row["sentence_ids"]), row["acceptable_evidence_groups"])
    assert result.reward == 0.0
    assert result.failure_type == "schema_error"


def test_family_condition_hallucination_is_penalized() -> None:
    row = load_rows()[0]
    bad = deepcopy(row["gold_output"])
    bad["conditions"].append({
        "id": "C9",
        "concept_id": "diabetes_mellitus",
        "assertion": "active",
        "event_time": "2024",
        "evidence_sentence_ids": [row["sentence_ids"][-1]],
    })
    result = score_completion(json.dumps(bad), row["gold_output"], set(row["sentence_ids"]), row["acceptable_evidence_groups"])
    assert 0.0 <= result.reward < 1.0
    assert result.components["unsupported_fact_rate"] > 0.0
    assert result.failure_type == "unsupported_condition"


def test_object_order_and_ids_do_not_change_semantic_score() -> None:
    row = next(row for row in load_rows() if len(row["gold_output"]["conditions"]) >= 2)
    predicted = deepcopy(row["gold_output"])
    predicted["conditions"] = list(reversed(predicted["conditions"]))
    id_map = {}
    for index, condition in enumerate(predicted["conditions"], start=1):
        old = condition["id"]
        condition["id"] = f"C{index + 10}"
        id_map[old] = condition["id"]
    for relation in predicted["relations"]:
        relation["condition_id"] = id_map[relation["condition_id"]]
    result = score_completion(json.dumps(predicted), row["gold_output"], set(row["sentence_ids"]), row["acceptable_evidence_groups"])
    assert result.reward == pytest.approx(1.0)
