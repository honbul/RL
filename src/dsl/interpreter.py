from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from jsonschema import Draft202012Validator

from src.dsl.schema import (
    CastStep,
    CoalesceStep,
    ConcatStep,
    ConditionalStep,
    ConstantStep,
    CopyStep,
    DefaultStep,
    FilterArrayStep,
    FlattenStep,
    FormatDateStep,
    MapArrayStep,
    MapEnumStep,
    NestStep,
    Program,
    ScaleStep,
    SplitStep,
    ValueSource,
)


class DSLExecutionError(ValueError):
    pass


_MISSING = object()


def get_path(value: Any, path: str, default: Any = _MISSING) -> Any:
    if path in ("", "$"):
        return value
    current = value
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            if default is not _MISSING:
                return default
            raise DSLExecutionError(f"missing source path: {path}")
    return current


def set_path(target: dict[str, Any], path: str, value: Any) -> None:
    if not path:
        raise DSLExecutionError("target path cannot be empty")
    parts = path.split(".")
    current = target
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            existing = {}
            current[part] = existing
        if not isinstance(existing, dict):
            raise DSLExecutionError(f"target path collides with scalar: {path}")
        current = existing
    current[parts[-1]] = deepcopy(value)


def _cast(value: Any, target_type: str) -> Any:
    try:
        if target_type == "string":
            return str(value)
        if target_type == "integer":
            if isinstance(value, bool):
                raise ValueError("boolean is not an integer source")
            return int(value)
        if target_type == "number":
            if isinstance(value, bool):
                raise ValueError("boolean is not a number source")
            return float(value)
        if target_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in {"true", "1", "yes"}:
                return True
            if isinstance(value, str) and value.lower() in {"false", "0", "no"}:
                return False
            if value in (0, 1):
                return bool(value)
            raise ValueError(f"cannot cast {value!r} to boolean")
    except (TypeError, ValueError) as exc:
        raise DSLExecutionError(f"cast failed for {value!r} -> {target_type}") from exc
    raise DSLExecutionError(f"unsupported cast target: {target_type}")


def _condition_matches(condition: Any, source: Any) -> bool:
    value = get_path(source, condition.path, _MISSING)
    op = condition.operator
    if op == "exists":
        return value is not _MISSING
    if value is _MISSING:
        return False
    expected = condition.value
    try:
        if op == "eq":
            return value == expected
        if op == "ne":
            return value != expected
        if op == "gt":
            return value > expected
        if op == "ge":
            return value >= expected
        if op == "lt":
            return value < expected
        if op == "le":
            return value <= expected
        if op == "in":
            return value in expected
        if op == "contains":
            return expected in value
    except (TypeError, ValueError):
        return False
    raise DSLExecutionError(f"unsupported condition operator: {op}")


def _resolve_value(spec: ValueSource, source: Any) -> Any:
    if "from_" in spec.model_fields_set:
        return get_path(source, spec.from_ or "")
    return deepcopy(spec.value)


def _execute_scalar(step: Any, source: Any, target: dict[str, Any]) -> None:
    if isinstance(step, CopyStep):
        set_path(target, step.to, get_path(source, step.from_))
    elif isinstance(step, ConstantStep):
        set_path(target, step.to, step.value)
    elif isinstance(step, CastStep):
        set_path(target, step.to, _cast(get_path(source, step.from_), step.target_type))
    elif isinstance(step, DefaultStep):
        value = get_path(source, step.from_, None)
        set_path(target, step.to, step.value if value is None else value)
    elif isinstance(step, CoalesceStep):
        for path in step.sources:
            value = get_path(source, path, None)
            if value is not None:
                set_path(target, step.to, value)
                break
        else:
            raise DSLExecutionError(f"coalesce found no value: {step.sources}")
    elif isinstance(step, MapEnumStep):
        value = get_path(source, step.from_)
        set_path(target, step.to, step.mapping.get(str(value), step.default))
    elif isinstance(step, ScaleStep):
        value = get_path(source, step.from_)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DSLExecutionError(f"scale requires a number: {step.from_}")
        set_path(target, step.to, value * step.factor + step.offset)
    elif isinstance(step, FormatDateStep):
        try:
            value = datetime.strptime(str(get_path(source, step.from_)), step.input_format)
        except ValueError as exc:
            raise DSLExecutionError(f"date parse failed: {step.from_}") from exc
        set_path(target, step.to, value.strftime(step.output_format))
    elif isinstance(step, ConcatStep):
        set_path(target, step.to, step.separator.join(str(get_path(source, path)) for path in step.sources))
    elif isinstance(step, SplitStep):
        parts = str(get_path(source, step.from_)).split(step.separator)
        try:
            value = parts[step.index]
        except IndexError as exc:
            raise DSLExecutionError(f"split index out of range: {step.index}") from exc
        set_path(target, step.to, value)
    elif isinstance(step, NestStep):
        nested: dict[str, Any] = {}
        for relative_target, source_path in step.fields.items():
            set_path(nested, relative_target, get_path(source, source_path))
        set_path(target, step.to, nested)
    elif isinstance(step, FlattenStep):
        value = get_path(source, step.from_)
        if not isinstance(value, dict):
            raise DSLExecutionError(f"flatten requires an object: {step.from_}")
        if step.to:
            for key, item in value.items():
                set_path(target, f"{step.to}.{key}", item)
        else:
            for key, item in value.items():
                set_path(target, key, item)
    elif isinstance(step, ConditionalStep):
        branch = step.then if _condition_matches(step.condition, source) else step.otherwise
        set_path(target, step.to, _resolve_value(branch, source))
    else:
        raise DSLExecutionError(f"unsupported scalar operation: {step.op}")


def execute_program(program: Program, source: dict[str, Any]) -> dict[str, Any]:
    target: dict[str, Any] = {}
    for step in program.steps:
        if isinstance(step, MapArrayStep):
            values = get_path(source, step.from_)
            if not isinstance(values, list):
                raise DSLExecutionError(f"map_array requires an array: {step.from_}")
            mapped = []
            for item in values:
                child: dict[str, Any] = {}
                for child_step in step.steps:
                    _execute_scalar(child_step, item, child)
                mapped.append(child)
            set_path(target, step.to, mapped)
        elif isinstance(step, FilterArrayStep):
            values = get_path(source, step.from_)
            if not isinstance(values, list):
                raise DSLExecutionError(f"filter_array requires an array: {step.from_}")
            set_path(target, step.to, [item for item in values if _condition_matches(step.condition, item)])
        else:
            _execute_scalar(step, source, target)
    return target


def schema_path_exists(schema: dict[str, Any], path: str) -> bool:
    if path in ("", "$"):
        return True
    current = schema
    for part in path.split("."):
        if current.get("type") == "array":
            current = current.get("items", {})
        properties = current.get("properties", {})
        if part not in properties:
            return False
        current = properties[part]
    return True


def _target_child_schema(schema: dict[str, Any], path: str) -> dict[str, Any]:
    current = schema
    if not path:
        return current
    for part in path.split("."):
        if current.get("type") == "array":
            current = current.get("items", {})
        current = current.get("properties", {}).get(part, {})
    return current


def validate_program_paths(program: Program, source_schema: dict[str, Any], target_schema: dict[str, Any]) -> None:
    for step in program.steps:
        source_paths: list[str] = []
        target_paths: list[str] = []
        if hasattr(step, "from_"):
            source_paths.append(step.from_)
        if hasattr(step, "sources"):
            source_paths.extend(step.sources)
        if isinstance(step, NestStep):
            source_paths.extend(step.fields.values())
            target_paths.extend(f"{step.to}.{key}" for key in step.fields)
        elif hasattr(step, "to"):
            target_paths.append(step.to)
        if isinstance(step, ConditionalStep):
            source_paths.append(step.condition.path)
            for branch in (step.then, step.otherwise):
                if "from_" in branch.model_fields_set:
                    source_paths.append(branch.from_ or "")
        for path in source_paths:
            if not schema_path_exists(source_schema, path):
                raise DSLExecutionError(f"invalid source path: {path}")
        for path in target_paths:
            if not schema_path_exists(target_schema, path):
                raise DSLExecutionError(f"invalid target path: {path}")
        if isinstance(step, (MapArrayStep, FilterArrayStep)):
            source_item = _target_child_schema(source_schema, step.from_).get("items", {})
            target_item = _target_child_schema(target_schema, step.to).get("items", {})
            if isinstance(step, MapArrayStep):
                nested = Program(steps=step.steps)
                validate_program_paths(nested, source_item, target_item)
            elif not schema_path_exists(source_item, step.condition.path):
                raise DSLExecutionError(f"invalid array condition path: {step.condition.path}")


def validate_json_schema_instance(instance: Any, schema: dict[str, Any]) -> None:
    errors = sorted(Draft202012Validator(schema).iter_errors(instance), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.path) or "$"
        raise DSLExecutionError(f"target schema error at {location}: {first.message}")
