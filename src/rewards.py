from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import ValidationError

from src.dsl.canonicalize import canonical_equal, field_f1
from src.dsl.interpreter import (
    DSLExecutionError,
    execute_program,
    validate_json_schema_instance,
    validate_program_paths,
)
from src.dsl.schema import Program


@dataclass
class RewardResult:
    reward: float
    components: dict[str, float]
    hidden_case_results: list[dict[str, Any]]
    parsed_program: dict[str, Any] | None
    failure_type: str | None
    parse_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_program(completion: str | dict[str, Any]) -> tuple[dict[str, Any], Program]:
    if isinstance(completion, str):
        payload = json.loads(completion, object_pairs_hook=_reject_duplicate_keys)
    else:
        payload = completion
    if not isinstance(payload, dict):
        raise ValueError("program must be a JSON object")
    return payload, Program.model_validate(payload)


def score_completion(
    completion: str | dict[str, Any],
    source_schema: dict[str, Any],
    target_schema: dict[str, Any],
    hidden_cases: list[dict[str, Any]],
    mode: str,
) -> RewardResult:
    try:
        payload, program = parse_program(completion)
    except json.JSONDecodeError as exc:
        return RewardResult(0.0, {}, [], None, "json_parse_error", str(exc))
    except (ValueError, ValidationError) as exc:
        parse_component = 0.1 if isinstance(completion, (str, dict)) else 0.0
        return RewardResult(parse_component if mode == "dense" else 0.0, {"parse_schema": parse_component}, [], None, "dsl_schema_error", str(exc))

    try:
        validate_program_paths(program, source_schema, target_schema)
    except DSLExecutionError as exc:
        components = {"parse_schema": 0.1, "reference_validity": 0.0}
        return RewardResult(0.1 if mode == "dense" else 0.0, components, [], payload, "invalid_source_path" if "source" in str(exc) else "invalid_target_path", str(exc))

    case_results: list[dict[str, Any]] = []
    exact_count = 0
    schema_count = 0
    f1_total = 0.0
    failure_type: str | None = None
    for index, case in enumerate(hidden_cases):
        try:
            predicted = execute_program(program, case["input"])
            validate_json_schema_instance(predicted, target_schema)
            schema_pass = True
            schema_count += 1
            exact = canonical_equal(predicted, case["expected"])
            exact_count += int(exact)
            score = field_f1(predicted, case["expected"])
            f1_total += score
            if not exact and failure_type is None:
                failure_type = "partial_value_mismatch"
            case_results.append({"index": index, "passed": exact, "schema_pass": True, "field_f1": score, "predicted": predicted})
        except DSLExecutionError as exc:
            message = str(exc)
            if failure_type is None:
                failure_type = "target_schema_error" if "target schema" in message else "runtime_error"
            case_results.append({"index": index, "passed": False, "schema_pass": False, "field_f1": 0.0, "error": message})

    count = len(hidden_cases)
    exact_rate = exact_count / count if count else 0.0
    schema_rate = schema_count / count if count else 0.0
    average_f1 = f1_total / count if count else 0.0
    if mode == "binary":
        reward = 1.0 if count > 0 and exact_count == count else 0.0
        components = {"all_hidden_cases": reward}
    elif mode == "dense":
        components = {
            "parse_schema": 0.10,
            "reference_validity": 0.10,
            "target_schema": 0.20 * schema_rate,
            "hidden_exact": 0.45 * exact_rate,
            "field_f1": 0.15 * average_f1,
        }
        reward = min(1.0, max(0.0, sum(components.values())))
    else:
        raise ValueError(f"unknown reward mode: {mode}")
    return RewardResult(reward, components, case_results, payload, failure_type, None)
