from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from src.dsl.interpreter import DSLExecutionError, validate_program_paths
from src.dsl.schema import Program
from src.rewards import score_completion


def test_all_twenty_reference_examples_pass_both_rewards() -> None:
    path = Path("data/dev/core_examples_20.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 20
    for row in rows:
        for mode in ("binary", "dense"):
            result = score_completion(
                row["reference_program"],
                row["source_schema"],
                row["target_schema"],
                row["hidden_tests"],
                mode,
            )
            assert result.reward == 1.0, (row["sample_id"], mode, result)


def test_invalid_operation_is_rejected() -> None:
    try:
        Program.model_validate({"steps": [{"op": "python", "code": "pass"}]})
    except ValidationError:
        return
    raise AssertionError("unsupported operation was accepted")


def test_invalid_source_and_target_paths_are_rejected() -> None:
    source_schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    target_schema = {"type": "object", "properties": {"b": {"type": "string"}}}
    for payload in (
        {"steps": [{"op": "copy", "from": "missing", "to": "b"}]},
        {"steps": [{"op": "copy", "from": "a", "to": "missing"}]},
    ):
        try:
            validate_program_paths(Program.model_validate(payload), source_schema, target_schema)
        except DSLExecutionError:
            continue
        raise AssertionError(f"invalid path was accepted: {payload}")


def test_partial_dense_reward_is_bounded() -> None:
    row = json.loads(Path("data/dev/core_examples_20.jsonl").read_text(encoding="utf-8").splitlines()[0])
    wrong = {"steps": [{"op": "constant", "to": "b", "value": "wrong"}]}
    result = score_completion(wrong, row["source_schema"], row["target_schema"], row["hidden_tests"], "dense")
    assert 0.0 <= result.reward < 1.0
    assert result.failure_type == "partial_value_mismatch"
