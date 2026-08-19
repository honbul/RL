from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from src.dsl.canonicalize import canonical_equal
from src.dsl.interpreter import execute_program, validate_json_schema_instance, validate_program_paths
from src.dsl.schema import Program


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validate_reference(row: dict[str, Any]) -> None:
    Draft202012Validator.check_schema(row["source_schema"])
    Draft202012Validator.check_schema(row["target_schema"])
    program = Program.model_validate(row["reference_program"])
    validate_program_paths(program, row["source_schema"], row["target_schema"])
    for case in row.get("hidden_tests", []):
        source_errors = list(Draft202012Validator(row["source_schema"]).iter_errors(case["input"]))
        if source_errors:
            raise ValueError(f"source schema failure for {row['sample_id']}: {source_errors[0].message}")
        predicted = execute_program(program, case["input"])
        validate_json_schema_instance(predicted, row["target_schema"])
        if not canonical_equal(predicted, case["expected"]):
            raise ValueError(f"hidden reference mismatch for {row['sample_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate required generated-data invariants")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    counts = config["counts"]
    expected = {
        "sft/sft_1k.jsonl": int(counts["sft_1k"]),
        "sft/sft_6k.jsonl": int(counts["sft_6k"]),
        "grpo/prompts.jsonl": int(counts["grpo_train"]),
        "grpo/hidden_tests.jsonl": int(counts["grpo_train"]),
        "grpo/references.jsonl": int(counts["grpo_train"]),
        "dev/dev.jsonl": int(counts["dev"]),
        "test/test_iid.jsonl": int(counts["test_iid"]),
        "test/test_linguistic_ood.jsonl": int(counts["test_linguistic_ood"]),
        "test/test_compositional_ood.jsonl": int(counts["test_compositional_ood"]),
        "test/test_structural_ood.jsonl": int(counts["test_structural_ood"]),
    }
    loaded = {}
    for relative, count in expected.items():
        rows = _rows(args.data_root / relative)
        if len(rows) != count:
            raise SystemExit(f"count mismatch {relative}: {len(rows)} != {count}")
        if len({row["sample_id"] for row in rows}) != count:
            raise SystemExit(f"duplicate sample_id in {relative}")
        loaded[relative] = rows
    if [row["sample_id"] for row in loaded["sft/sft_1k.jsonl"]] != [row["sample_id"] for row in loaded["sft/sft_6k.jsonl"][: int(counts["sft_1k"])]]:
        raise SystemExit("SFT 1K is not a prefix subset of SFT 6K")
    for row in loaded["sft/sft_6k.jsonl"]:
        _validate_reference(row)
        if "hidden_tests" in row["prompt"] or "reference_program" in row["prompt"]:
            raise SystemExit(f"SFT prompt leakage: {row['sample_id']}")
    grpo_prompts = loaded["grpo/prompts.jsonl"]
    hidden_by_id = {row["sample_id"]: row["hidden_tests"] for row in loaded["grpo/hidden_tests.jsonl"]}
    ref_by_id = {row["sample_id"]: row["reference_program"] for row in loaded["grpo/references.jsonl"]}
    if set(hidden_by_id) != {row["sample_id"] for row in grpo_prompts} or set(ref_by_id) != set(hidden_by_id):
        raise SystemExit("GRPO join keys do not match")
    for prompt in grpo_prompts:
        if "hidden_tests" in prompt["prompt"] or "reference_program" in prompt["prompt"]:
            raise SystemExit(f"GRPO prompt leakage: {prompt['sample_id']}")
        joined = {**prompt, "reference_program": ref_by_id[prompt["sample_id"]], "hidden_tests": hidden_by_id[prompt["sample_id"]]}
        if len(joined["hidden_tests"]) != int(config["hidden_cases"]["grpo"]):
            raise SystemExit(f"GRPO hidden count mismatch: {prompt['sample_id']}")
        _validate_reference(joined)
    for relative in ("dev/dev.jsonl", "test/test_iid.jsonl", "test/test_linguistic_ood.jsonl", "test/test_compositional_ood.jsonl", "test/test_structural_ood.jsonl"):
        for row in loaded[relative]:
            if len(row["hidden_tests"]) != int(config["hidden_cases"]["evaluation"]):
                raise SystemExit(f"evaluation hidden count mismatch: {row['sample_id']}")
            _validate_reference(row)
    manifest = json.loads((args.data_root / "manifests/data_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "PASS" or any(manifest["leakage_checks"].values()):
        raise SystemExit("manifest leakage/status failure")
    summary = {"status": "PASS", "candidate_count": int(config["candidate_count"]), "files": expected, "references_validated": sum(expected.values()) - expected["grpo/hidden_tests.jsonl"] - expected["grpo/references.jsonl"]}
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
