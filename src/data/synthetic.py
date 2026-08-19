from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from typing import Any

from src.data.render_instruction import render_instruction
from src.dsl.canonicalize import canonical_equal
from src.dsl.interpreter import execute_program, validate_json_schema_instance, validate_program_paths
from src.dsl.schema import Program

BASE_PATTERNS = [
    "copy_constant", "cast_default", "coalesce_enum", "scale_date", "concat_split",
    "nest_flatten", "conditional_copy", "map_array", "filter_array",
    "copy_enum_scale", "default_concat_conditional",
]
COMPOSITIONAL_PATTERNS = [
    "comp_copy_scale_conditional", "comp_coalesce_concat_nest",
    "comp_date_split_constant_copy", "comp_map_mixed", "comp_filter_copy_constant",
]
STRUCTURAL_PATTERNS = ["struct_long_nested", "struct_deep_nest", "struct_array_rich"]


def _schema_for_values(values: list[Any]) -> dict[str, Any]:
    non_null = [value for value in values if value is not None]
    if not non_null:
        return {"type": "null"}
    if any(isinstance(value, dict) for value in non_null):
        if not all(isinstance(value, dict) for value in non_null):
            return {}
        keys = sorted(set().union(*(value.keys() for value in non_null)))
        required = [key for key in keys if all(key in value for value in non_null)]
        schema = {
            "type": "object",
            "properties": {
                key: _schema_for_values([value[key] for value in non_null if key in value])
                for key in keys
            },
            "required": required,
            "additionalProperties": False,
        }
        return schema
    if any(isinstance(value, list) for value in non_null):
        if not all(isinstance(value, list) for value in non_null):
            return {}
        items = [item for value in non_null for item in value]
        return {"type": "array", "items": _schema_for_values(items) if items else {}}
    types = set()
    for value in values:
        if value is None:
            types.add("null")
        elif isinstance(value, bool):
            types.add("boolean")
        elif isinstance(value, int):
            types.add("integer")
        elif isinstance(value, float):
            types.add("number")
        elif isinstance(value, str):
            types.add("string")
    if types == {"integer", "number"}:
        types = {"number"}
    ordered = sorted(types)
    return {"type": ordered[0] if len(ordered) == 1 else ordered}


def _blueprint(pattern: str, rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    params: dict[str, Any] = {}
    if pattern == "copy_constant":
        params = {"kind": rng.choice(["person", "sample", "event"])}
        steps = [{"op": "copy", "from": "patient_id", "to": "subject_id"}, {"op": "constant", "to": "kind", "value": params["kind"]}]
    elif pattern == "cast_default":
        params = {"default": rng.choice(["unknown", "anonymous", "n/a"])}
        steps = [{"op": "cast", "from": "age_text", "to": "age", "target_type": "integer"}, {"op": "default", "from": "nickname", "to": "display_name", "value": params["default"]}]
    elif pattern == "coalesce_enum":
        steps = [{"op": "coalesce", "sources": ["primary_code", "secondary_code"], "to": "code"}, {"op": "map_enum", "from": "sex", "to": "gender", "mapping": {"M": "male", "F": "female", "U": "unknown"}, "default": "unknown"}]
    elif pattern == "scale_date":
        params = {"factor": rng.choice([0.453592, 2.54, 0.001])}
        steps = [{"op": "scale", "from": "measurement", "to": "scaled", "factor": params["factor"]}, {"op": "format_date", "from": "date", "to": "formatted_date", "input_format": "%Y-%m-%d", "output_format": "%d/%m/%Y"}]
    elif pattern == "concat_split":
        params = {"separator": rng.choice([" ", "-", "/"])}
        steps = [{"op": "concat", "sources": ["first", "last"], "to": "full", "separator": params["separator"]}, {"op": "split", "from": "compound", "to": "suffix", "separator": ":", "index": 1}]
    elif pattern == "nest_flatten":
        steps = [{"op": "nest", "to": "subject", "fields": {"identifier": "identifier", "location.city": "details.city"}}, {"op": "flatten", "from": "details", "to": "address"}]
    elif pattern == "conditional_copy":
        params = {"threshold": rng.choice([18, 50, 75])}
        steps = [{"op": "copy", "from": "identifier", "to": "identifier"}, {"op": "conditional", "condition": {"path": "score", "operator": "ge", "value": params["threshold"]}, "to": "band", "then": {"value": "high"}, "otherwise": {"value": "low"}}]
    elif pattern == "map_array":
        params = {"factor": rng.choice([0.1, 2.0, 10.0])}
        steps = [{"op": "map_array", "from": "measurements", "to": "normalized", "steps": [{"op": "scale", "from": "value", "to": "value", "factor": params["factor"]}, {"op": "copy", "from": "unit", "to": "unit"}]}]
    elif pattern == "filter_array":
        params = {"threshold": rng.choice([-1, 0, 5])}
        steps = [{"op": "filter_array", "from": "items", "to": "kept", "condition": {"path": "value", "operator": "gt", "value": params["threshold"]}}]
    elif pattern == "copy_enum_scale":
        params = {"factor": rng.choice([1.5, 10.0, 100.0])}
        steps = [{"op": "copy", "from": "identifier", "to": "identifier"}, {"op": "map_enum", "from": "status", "to": "active", "mapping": {"Y": True, "N": False}, "default": False}, {"op": "scale", "from": "amount", "to": "amount", "factor": params["factor"]}]
    elif pattern == "default_concat_conditional":
        params = {"fallback": rng.choice(["NA", "missing", "none"])}
        steps = [{"op": "default", "from": "middle", "to": "middle", "value": params["fallback"]}, {"op": "concat", "sources": ["first", "last"], "to": "label", "separator": ", "}, {"op": "conditional", "condition": {"path": "enabled", "operator": "eq", "value": True}, "to": "state", "then": {"value": "enabled"}, "otherwise": {"value": "disabled"}}]
    elif pattern == "comp_copy_scale_conditional":
        params = {"factor": rng.choice([0.5, 2.5]), "threshold": rng.choice([0, 10])}
        steps = [{"op": "copy", "from": "identifier", "to": "identifier"}, {"op": "scale", "from": "amount", "to": "scaled_amount", "factor": params["factor"]}, {"op": "conditional", "condition": {"path": "amount", "operator": "gt", "value": params["threshold"]}, "to": "class", "then": {"value": "positive"}, "otherwise": {"value": "non_positive"}}]
    elif pattern == "comp_coalesce_concat_nest":
        steps = [{"op": "coalesce", "sources": ["preferred", "fallback"], "to": "name"}, {"op": "concat", "sources": ["prefix", "code"], "to": "label", "separator": "-"}, {"op": "nest", "to": "meta", "fields": {"source": "source", "code": "code"}}]
    elif pattern == "comp_date_split_constant_copy":
        params = {"kind": rng.choice(["A", "B"])}
        steps = [{"op": "format_date", "from": "date", "to": "date", "input_format": "%Y-%m-%d", "output_format": "%Y%m%d"}, {"op": "split", "from": "code", "to": "prefix", "separator": "-", "index": 0}, {"op": "constant", "to": "kind", "value": params["kind"]}, {"op": "copy", "from": "value", "to": "value"}]
    elif pattern == "comp_map_mixed":
        params = {"factor": rng.choice([0.25, 4.0])}
        steps = [{"op": "map_array", "from": "items", "to": "items", "steps": [{"op": "copy", "from": "name", "to": "label"}, {"op": "map_enum", "from": "status", "to": "active", "mapping": {"on": True, "off": False}, "default": False}, {"op": "scale", "from": "value", "to": "scaled", "factor": params["factor"]}]}]
    elif pattern == "comp_filter_copy_constant":
        params = {"threshold": rng.choice([1, 3, 7])}
        steps = [{"op": "filter_array", "from": "items", "to": "selected", "condition": {"path": "value", "operator": "ge", "value": params["threshold"]}}, {"op": "copy", "from": "identifier", "to": "identifier"}, {"op": "constant", "to": "filtered", "value": True}]
    elif pattern == "struct_long_nested":
        params = {"factor": rng.choice([1.1, 3.0]), "threshold": rng.choice([5, 20])}
        steps = [{"op": "copy", "from": "identifier", "to": "record.identifier"}, {"op": "map_enum", "from": "status", "to": "record.active", "mapping": {"Y": True, "N": False}, "default": False}, {"op": "scale", "from": "amount", "to": "record.scaled", "factor": params["factor"]}, {"op": "format_date", "from": "date", "to": "record.date", "input_format": "%Y-%m-%d", "output_format": "%d-%b-%Y"}, {"op": "conditional", "condition": {"path": "amount", "operator": "ge", "value": params["threshold"]}, "to": "record.tier", "then": {"value": "upper"}, "otherwise": {"value": "lower"}}]
    elif pattern == "struct_deep_nest":
        steps = [{"op": "nest", "to": "envelope", "fields": {"identity.primary.value": "identifier", "location.address.city": "city", "location.address.postal": "postal"}}, {"op": "concat", "sources": ["first", "last"], "to": "envelope.display.full", "separator": " "}, {"op": "copy", "from": "category", "to": "envelope.classification.category"}]
    elif pattern == "struct_array_rich":
        params = {"factor": rng.choice([0.01, 10.0])}
        steps = [{"op": "map_array", "from": "items", "to": "payload.rows", "steps": [{"op": "copy", "from": "name", "to": "identity.name"}, {"op": "scale", "from": "value", "to": "measurement.value", "factor": params["factor"]}, {"op": "map_enum", "from": "status", "to": "state.active", "mapping": {"ok": True, "bad": False}, "default": False}, {"op": "conditional", "condition": {"path": "value", "operator": "lt", "value": 0}, "to": "measurement.sign", "then": {"value": "negative"}, "otherwise": {"value": "non_negative"}}]}, {"op": "copy", "from": "batch_id", "to": "payload.batch.identifier"}]
    else:
        raise ValueError(pattern)
    return {"pattern": pattern, "params": params}, {"steps": steps}


def _case(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    p = spec["pattern"]
    q = spec["params"]
    word = lambda: rng.choice(["alpha", "beta", "gamma", "delta", "omega"])
    if p == "copy_constant": return {"patient_id": f"p{rng.randrange(100000)}", "unused_id": f"x{rng.randrange(1000)}"}
    if p == "cast_default": return {"age_text": str(rng.randrange(0, 101)), "nickname": None if rng.random() < 0.5 else word()}
    if p == "coalesce_enum": return {"primary_code": None if rng.random() < 0.5 else word(), "secondary_code": word(), "sex": rng.choice(["M", "F", "U", "X"])}
    if p == "scale_date": return {"measurement": rng.uniform(-100, 10000), "date": f"{rng.randrange(2000, 2031):04d}-{rng.randrange(1,13):02d}-{rng.randrange(1,29):02d}"}
    if p == "concat_split": return {"first": word(), "last": word(), "compound": f"{word()}:{rng.randrange(1000)}"}
    if p == "nest_flatten": return {"identifier": f"p{rng.randrange(9999)}", "details": {"city": word(), "zip": f"{rng.randrange(10000,99999)}"}}
    if p == "conditional_copy": return {"identifier": f"i{rng.randrange(9999)}", "score": rng.randrange(0, 101)}
    if p == "map_array": return {"measurements": [{"value": rng.uniform(-10, 100), "unit": rng.choice(["mg", "kg", "cm"])} for _ in range(rng.randrange(1, 5))]}
    if p == "filter_array": return {"items": [{"value": rng.randrange(-10, 11), "name": word()} for _ in range(rng.randrange(1, 7))]}
    if p == "copy_enum_scale": return {"identifier": f"i{rng.randrange(9999)}", "status": rng.choice(["Y", "N", "U"]), "amount": rng.uniform(-20, 100)}
    if p == "default_concat_conditional": return {"middle": None if rng.random() < 0.5 else word(), "first": word(), "last": word(), "enabled": rng.choice([True, False])}
    if p == "comp_copy_scale_conditional": return {"identifier": f"c{rng.randrange(9999)}", "amount": rng.uniform(-20, 30)}
    if p == "comp_coalesce_concat_nest": return {"preferred": None if rng.random() < 0.5 else word(), "fallback": word(), "prefix": word(), "code": str(rng.randrange(1000)), "source": rng.choice(["A", "B"])}
    if p == "comp_date_split_constant_copy": return {"date": f"{rng.randrange(2000,2031):04d}-{rng.randrange(1,13):02d}-{rng.randrange(1,29):02d}", "code": f"{word()}-{rng.randrange(100)}", "value": rng.uniform(-1,1)}
    if p == "comp_map_mixed": return {"items": [{"name": word(), "status": rng.choice(["on", "off", "unknown"]), "value": rng.uniform(-50,50)} for _ in range(rng.randrange(1,5))]}
    if p == "comp_filter_copy_constant": return {"items": [{"value": rng.randrange(-2,12), "name": word()} for _ in range(rng.randrange(1,7))], "identifier": f"f{rng.randrange(9999)}"}
    if p == "struct_long_nested": return {"identifier": f"s{rng.randrange(9999)}", "status": rng.choice(["Y", "N", "X"]), "amount": rng.uniform(-50,100), "date": f"{rng.randrange(2000,2031):04d}-{rng.randrange(1,13):02d}-{rng.randrange(1,29):02d}"}
    if p == "struct_deep_nest": return {"identifier": f"d{rng.randrange(9999)}", "city": word(), "postal": f"{rng.randrange(10000,99999)}", "first": word(), "last": word(), "category": rng.choice(["A", "B", "C"])}
    if p == "struct_array_rich": return {"batch_id": f"b{rng.randrange(9999)}", "items": [{"name": word(), "value": rng.uniform(-100,100), "status": rng.choice(["ok", "bad", "unknown"])} for _ in range(rng.randrange(3,9))]}
    raise ValueError(p)


def _prefix_path(path: str, prefix: str) -> str:
    if path in ("", "$"):
        return path
    first, *rest = path.split(".")
    return ".".join([prefix + first, *rest])


def _prefix_program(program: dict[str, Any], prefix: str) -> dict[str, Any]:
    result = deepcopy(program)
    for step in result["steps"]:
        if "from" in step: step["from"] = _prefix_path(step["from"], prefix)
        if "sources" in step: step["sources"] = [_prefix_path(path, prefix) for path in step["sources"]]
        if "to" in step: step["to"] = _prefix_path(step["to"], prefix)
        if step["op"] == "nest": step["fields"] = {key: _prefix_path(path, prefix) for key, path in step["fields"].items()}
        if step["op"] == "conditional":
            step["condition"]["path"] = _prefix_path(step["condition"]["path"], prefix)
            for branch in (step["then"], step["otherwise"]):
                if "from" in branch: branch["from"] = _prefix_path(branch["from"], prefix)
        if step["op"] in {"map_array", "filter_array"}:
            # Child paths are relative to each array item and must stay unprefixed.
            if step["op"] == "filter_array":
                step["condition"]["path"] = step["condition"]["path"].removeprefix(prefix)
    return result


def _prefix_top_level(value: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {prefix + key: item for key, item in value.items()}


def _partition(index: int) -> str:
    if index < 15000: return "base"
    if index < 16000: return "linguistic"
    if index < 18500: return "compositional"
    return "structural"


def _enforce_declared_nullable_fields(
    source_schema: dict[str, Any], pattern: str, prefix: str
) -> None:
    nullable_by_pattern = {
        "cast_default": ["nickname"],
        "coalesce_enum": ["primary_code"],
        "default_concat_conditional": ["middle"],
        "comp_coalesce_concat_nest": ["preferred"],
    }
    for field in nullable_by_pattern.get(pattern, []):
        source_schema["properties"][prefix + field]["type"] = ["null", "string"]


def make_candidate(index: int, base_seed: int = 42) -> dict[str, Any]:
    seed = base_seed * 1_000_000 + index
    rng = random.Random(seed)
    partition = _partition(index)
    if partition in {"base", "linguistic"}:
        patterns = BASE_PATTERNS
    elif partition == "compositional":
        patterns = COMPOSITIONAL_PATTERNS
    else:
        patterns = STRUCTURAL_PATTERNS
    pattern = patterns[index % len(patterns)]
    spec, raw_program = _blueprint(pattern, rng)
    prefix = f"s{index:05d}_"
    program_payload = _prefix_program(raw_program, prefix)
    program = Program.model_validate(program_payload)
    schema_sources = [_prefix_top_level(_case(spec, seed + 1000 + offset), prefix) for offset in range(12)]
    schema_targets = [execute_program(program, source) for source in schema_sources]
    source_schema = _schema_for_values(schema_sources)
    target_schema = _schema_for_values(schema_targets)
    _enforce_declared_nullable_fields(source_schema, pattern, prefix)
    validate_program_paths(program, source_schema, target_schema)
    validation_cases = []
    for offset in range(2):
        source = _prefix_top_level(_case(spec, seed + 2000 + offset), prefix)
        expected = execute_program(program, source)
        validate_json_schema_instance(expected, target_schema)
        validation_cases.append({"input": source, "expected": expected})
    visible_count = index % 3
    visible_examples = []
    for offset in range(visible_count):
        source = _prefix_top_level(_case(spec, seed + 3000 + offset), prefix)
        visible_examples.append({"input": source, "output": execute_program(program, source)})
    family_prefix = "ling" if partition == "linguistic" else "train"
    template_family = f"{family_prefix}_{index % 4}"
    dumped_program = program.model_dump(by_alias=True, exclude_unset=True)
    fingerprint = hashlib.sha256(json.dumps({"program": dumped_program, "source_schema": source_schema, "target_schema": target_schema}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    serialized_schema = json.dumps(source_schema)
    difficulty = {
        "level": 4 if partition == "structural" else min(4, max(0, len(program.steps) - 1)),
        "num_source_fields": serialized_schema.count('"type"') - 1,
        "num_target_fields": json.dumps(target_schema).count('"type"') - 1,
        "num_steps": len(program.steps),
        "nesting_depth": 4 if partition == "structural" else (2 if any(step.op in {"nest", "map_array"} for step in program.steps) else 1),
        "has_null": '"null"' in serialized_schema,
        "has_array": '"array"' in serialized_schema,
        "has_condition": any(step.op in {"conditional", "filter_array"} for step in program.steps),
    }
    row = {
        "sample_id": f"sample_{index:06d}",
        "global_index": index,
        "partition": partition,
        "source_schema": source_schema,
        "target_schema": target_schema,
        "instruction": render_instruction(dumped_program, template_family),
        "visible_examples": visible_examples,
        "reference_program": dumped_program,
        "difficulty": difficulty,
        "graph_signature": pattern,
        "graph_fingerprint": fingerprint,
        "instruction_template_family": template_family,
        "seed": seed,
        "generator_spec": spec,
        "path_prefix": prefix,
        "validation_cases": validation_cases,
    }
    return row


def materialize_hidden_cases(row: dict[str, Any], count: int, offset_base: int) -> list[dict[str, Any]]:
    program = Program.model_validate(row["reference_program"])
    cases = []
    for offset in range(count):
        source = _prefix_top_level(_case(row["generator_spec"], row["seed"] + offset_base + offset), row["path_prefix"])
        expected = execute_program(program, source)
        validate_json_schema_instance(expected, row["target_schema"])
        cases.append({"input": source, "expected": expected})
    return cases


def public_projection(row: dict[str, Any]) -> dict[str, Any]:
    keys = ["sample_id", "source_schema", "target_schema", "instruction", "visible_examples", "difficulty", "graph_signature", "instruction_template_family", "seed"]
    return {key: deepcopy(row[key]) for key in keys}


def verify_candidate(row: dict[str, Any]) -> None:
    program = Program.model_validate(row["reference_program"])
    validate_program_paths(program, row["source_schema"], row["target_schema"])
    for case in row["validation_cases"]:
        predicted = execute_program(program, case["input"])
        validate_json_schema_instance(predicted, row["target_schema"])
        if not canonical_equal(predicted, case["expected"]):
            raise ValueError(f"reference mismatch: {row['sample_id']}")
