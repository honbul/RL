from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.dsl.canonicalize import canonical_equal
from src.dsl.interpreter import execute_program, validate_json_schema_instance, validate_program_paths
from src.dsl.schema import Program


def infer_schema(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        item_schema = infer_schema(value[0]) if value else {}
        return {"type": "array", "items": item_schema}
    if isinstance(value, dict):
        return {
            "type": "object",
            "properties": {key: infer_schema(item) for key, item in value.items()},
            "required": list(value),
            "additionalProperties": False,
        }
    raise TypeError(type(value))


def example_rows() -> list[dict[str, Any]]:
    specs = [
        ("copy a field", {"a": "x"}, {"b": "x"}, [{"op": "copy", "from": "a", "to": "b"}]),
        ("write a constant", {}, {"kind": "fixed"}, [{"op": "constant", "to": "kind", "value": "fixed"}]),
        ("cast text to integer", {"age": "42"}, {"age": 42}, [{"op": "cast", "from": "age", "to": "age", "target_type": "integer"}]),
        ("use a default for null", {"name": None}, {"name": "unknown"}, [{"op": "default", "from": "name", "to": "name", "value": "unknown"}]),
        ("take the first non-null value", {"a": None, "b": "B"}, {"value": "B"}, [{"op": "coalesce", "sources": ["a", "b"], "to": "value"}]),
        ("map an enum", {"sex": "F"}, {"gender": "female"}, [{"op": "map_enum", "from": "sex", "to": "gender", "mapping": {"M": "male", "F": "female"}, "default": "unknown"}]),
        ("scale pounds to kilograms", {"weight": 10.0}, {"weight_kg": 4.53592}, [{"op": "scale", "from": "weight", "to": "weight_kg", "factor": 0.453592}]),
        ("format an ISO date", {"date": "2024-02-29"}, {"date": "29/02/2024"}, [{"op": "format_date", "from": "date", "to": "date", "input_format": "%Y-%m-%d", "output_format": "%d/%m/%Y"}]),
        ("concatenate names", {"first": "Ada", "last": "Lovelace"}, {"full": "Ada Lovelace"}, [{"op": "concat", "sources": ["first", "last"], "to": "full", "separator": " "}]),
        ("split a code", {"code": "AA-19"}, {"suffix": "19"}, [{"op": "split", "from": "code", "to": "suffix", "separator": "-", "index": 1}]),
        ("nest fields", {"id": "p1", "sex": "F"}, {"subject": {"identifier": "p1", "sex": "F"}}, [{"op": "nest", "to": "subject", "fields": {"identifier": "id", "sex": "sex"}}]),
        ("flatten an object", {"profile": {"id": "p1", "age": 7}}, {"flat": {"id": "p1", "age": 7}}, [{"op": "flatten", "from": "profile", "to": "flat"}]),
        ("choose the adult branch", {"age": 22, "adult_label": "adult"}, {"group": "adult"}, [{"op": "conditional", "condition": {"path": "age", "operator": "ge", "value": 18}, "to": "group", "then": {"from": "adult_label"}, "otherwise": {"value": "minor"}}]),
        ("choose the minor branch", {"age": 12}, {"group": "minor"}, [{"op": "conditional", "condition": {"path": "age", "operator": "ge", "value": 18}, "to": "group", "then": {"value": "adult"}, "otherwise": {"value": "minor"}}]),
        ("map an array", {"items": [{"x": 2.0}, {"x": -3.0}]}, {"values": [{"y": 4.0}, {"y": -6.0}]}, [{"op": "map_array", "from": "items", "to": "values", "steps": [{"op": "scale", "from": "x", "to": "y", "factor": 2.0}]}]),
        ("filter an array", {"items": [{"x": 0}, {"x": 2}, {"x": -1}]}, {"positive": [{"x": 2}]}, [{"op": "filter_array", "from": "items", "to": "positive", "condition": {"path": "x", "operator": "gt", "value": 0}}]),
        ("compose enum copy and scale", {"id": "a", "status": "Y", "amount": 5.0}, {"id": "a", "active": True, "amount": 50.0}, [{"op": "copy", "from": "id", "to": "id"}, {"op": "map_enum", "from": "status", "to": "active", "mapping": {"Y": True, "N": False}, "default": False}, {"op": "scale", "from": "amount", "to": "amount", "factor": 10.0}]),
        ("coalesce three optional values", {"a": None, "b": None, "c": 0}, {"value": 0}, [{"op": "coalesce", "sources": ["a", "b", "c"], "to": "value"}]),
        ("scale a negative decimal", {"temperature_c": -40.0}, {"temperature_f": -40.0}, [{"op": "scale", "from": "temperature_c", "to": "temperature_f", "factor": 1.8, "offset": 32.0}]),
        ("copy a nested field", {"patient": {"id": "nested-1"}}, {"subject": {"identifier": "nested-1"}}, [{"op": "copy", "from": "patient.id", "to": "subject.identifier"}]),
    ]
    rows = []
    for index, (instruction, source, expected, steps) in enumerate(specs):
        program = Program.model_validate({"steps": steps})
        source_schema = infer_schema(source)
        target_schema = infer_schema(expected)
        validate_program_paths(program, source_schema, target_schema)
        actual = execute_program(program, source)
        validate_json_schema_instance(actual, target_schema)
        if not canonical_equal(actual, expected):
            raise AssertionError((instruction, actual, expected))
        rows.append({
            "sample_id": f"core_example_{index:02d}",
            "source_schema": source_schema,
            "target_schema": target_schema,
            "instruction": instruction,
            "visible_examples": [],
            "reference_program": program.model_dump(by_alias=True, exclude_unset=True),
            "hidden_tests": [{"input": source, "expected": expected}],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the 20 Stage 1 executable examples")
    parser.add_argument("--output", type=Path, default=Path("data/dev/core_examples_20.jsonl"))
    args = parser.parse_args()
    rows = example_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
