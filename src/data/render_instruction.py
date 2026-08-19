from __future__ import annotations

from typing import Any


def _describe_step(step: dict[str, Any]) -> str:
    op = step["op"]
    if op == "copy":
        return f"copy `{step['from']}` to `{step['to']}`"
    if op == "constant":
        return f"set `{step['to']}` to the constant {step['value']!r}"
    if op == "cast":
        return f"cast `{step['from']}` to {step['target_type']} and store it at `{step['to']}`"
    if op == "default":
        return f"copy `{step['from']}` to `{step['to']}`, using {step['value']!r} when it is missing or null"
    if op == "coalesce":
        return f"store the first non-null value among {step['sources']!r} at `{step['to']}`"
    if op == "map_enum":
        return f"map `{step['from']}` through {step['mapping']!r} into `{step['to']}`, defaulting to {step.get('default')!r}"
    if op == "scale":
        return f"store `{step['from']}` × {step['factor']} + {step.get('offset', 0.0)} at `{step['to']}`"
    if op == "format_date":
        return f"reformat `{step['from']}` from {step['input_format']!r} to {step['output_format']!r} at `{step['to']}`"
    if op == "concat":
        return f"join {step['sources']!r} with {step.get('separator', '')!r} into `{step['to']}`"
    if op == "split":
        return f"split `{step['from']}` on {step['separator']!r} and store item {step['index']} at `{step['to']}`"
    if op == "nest":
        return f"create `{step['to']}` with target-to-source fields {step['fields']!r}"
    if op == "flatten":
        return f"flatten object `{step['from']}` under `{step['to']}`"
    if op == "conditional":
        condition = step["condition"]
        return (
            f"set `{step['to']}` from {step['then']!r} when `{condition['path']}` "
            f"{condition['operator']} {condition.get('value')!r}, otherwise from {step['otherwise']!r}"
        )
    if op == "map_array":
        nested = "; ".join(_describe_step(child) for child in step["steps"])
        return f"map array `{step['from']}` to `{step['to']}` and for each item {nested}"
    if op == "filter_array":
        condition = step["condition"]
        return (
            f"filter array `{step['from']}` into `{step['to']}` keeping items where "
            f"`{condition['path']}` {condition['operator']} {condition.get('value')!r}"
        )
    raise ValueError(f"unsupported operation: {op}")


def render_instruction(program: dict[str, Any], family: str) -> str:
    clauses = [_describe_step(step) for step in program["steps"]]
    if family.startswith("ling_"):
        introductions = [
            "Produce the destination object by carrying out these rules in order: ",
            "Translate each source record into the requested target shape. Specifically, ",
            "Construct only the target JSON described here: ",
        ]
        separator = "; then "
    else:
        introductions = [
            "Transform the source JSON as follows: ",
            "Build the target JSON by applying these operations: ",
            "Apply the following ordered mapping rules: ",
            "Return a target object that will ",
        ]
        separator = "; "
    index = sum(ord(char) for char in family) % len(introductions)
    return introductions[index] + separator.join(clauses) + "."
