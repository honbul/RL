from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.clinical_ie.schema import ExtractionOutput, validate_evidence_ids

FORBIDDEN_PROMPT_TERMS = ("gold_output", "latent_timeline", "acceptable_evidence_groups")


def rows_from_files(files: list[dict[str, Any]]) -> Iterable[tuple[dict[str, Any], str]]:
    for entry in files:
        path = Path(entry["path"])
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line), line


def validate_full_row(row: dict[str, Any]) -> None:
    parsed = ExtractionOutput.model_validate(row["gold_output"])
    sentence_ids = {
        sentence["sentence_id"]
        for document in row["documents"]
        for sentence in document["sentences"]
    }
    validate_evidence_ids(parsed, sentence_ids)
    if any(term in row["prompt"] for term in FORBIDDEN_PROMPT_TERMS):
        raise ValueError(f"prompt leakage: {row['sample_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the four required clinical IE data invariants")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/clinical_ie"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = json.loads((args.data_root / "manifests/data_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "PASS" or manifest["candidate_count"] != int(config["candidate_count"]):
        raise SystemExit("candidate manifest count/status mismatch")
    max_bytes = 50 * 1024 * 1024
    candidate_rows = 0
    candidate_ids: set[str] = set()
    for entry in manifest["candidate_files"]:
        if entry["bytes"] > max_bytes:
            raise SystemExit(f"candidate shard exceeds 50MB: {entry['path']}")
        for row, _ in rows_from_files([entry]):
            validate_full_row(row)
            candidate_rows += 1
            if row["sample_id"] in candidate_ids:
                raise SystemExit(f"duplicate candidate ID: {row['sample_id']}")
            candidate_ids.add(row["sample_id"])
    if candidate_rows != int(config["candidate_count"]):
        raise SystemExit(f"candidate rows mismatch: {candidate_rows}")

    expected = {
        "sft_8k": 8000,
        "sft_24k": 24000,
        "hard_candidates": 8000,
        "hard_prompts": 8000,
        "hard_references": 8000,
        "dev": 2000,
        "test_iid_hard": 1000,
        "test_linguistic_ood": 1000,
        "test_compositional_ood": 1000,
        "test_long_context_ood": 1000,
        "test_counterfactual_ood": 1000,
    }
    inventories: dict[str, list[dict[str, Any]]] = manifest["split_files"]
    observed_counts = {}
    ids_by_group: dict[str, set[str]] = {}
    topology_by_group: dict[str, set[str]] = {}
    for group, expected_count in expected.items():
        files = inventories[group]
        if any(int(entry["bytes"]) > max_bytes for entry in files):
            raise SystemExit(f"split shard exceeds 50MB: {group}")
        count = 0
        ids: set[str] = set()
        topologies: set[str] = set()
        for row, _ in rows_from_files(files):
            count += 1
            sample_id = row["sample_id"]
            if sample_id in ids:
                raise SystemExit(f"duplicate ID in {group}: {sample_id}")
            ids.add(sample_id)
            if group == "hard_prompts":
                if any(key in row for key in ("gold_output", "latent_timeline", "acceptable_evidence_groups")):
                    raise SystemExit(f"hard prompt leakage: {sample_id}")
                if any(term in row["prompt"] for term in FORBIDDEN_PROMPT_TERMS):
                    raise SystemExit(f"hard prompt text leakage: {sample_id}")
            elif group == "hard_references":
                ExtractionOutput.model_validate(row["gold_output"])
            else:
                validate_full_row(row)
                topologies.add(row["topology_fingerprint"])
                if group.startswith("sft_"):
                    if json.loads(row["completion"]) != row["gold_output"]:
                        raise SystemExit(f"SFT completion mismatch: {sample_id}")
        if count != expected_count:
            raise SystemExit(f"{group} rows mismatch: {count} != {expected_count}")
        observed_counts[group] = count
        ids_by_group[group] = ids
        topology_by_group[group] = topologies
    if ids_by_group["hard_candidates"] != ids_by_group["hard_prompts"] or ids_by_group["hard_candidates"] != ids_by_group["hard_references"]:
        raise SystemExit("hard candidate/prompt/reference join mismatch")

    sft8_lines = [line for _, line in rows_from_files(inventories["sft_8k"])]
    sft24_prefix = []
    for _, line in rows_from_files(inventories["sft_24k"]):
        if len(sft24_prefix) == 8000:
            break
        sft24_prefix.append(line)
    if sft8_lines != sft24_prefix:
        raise SystemExit("SFT-8K is not a byte-identical prefix of SFT-24K")

    train_ids = ids_by_group["sft_24k"] | ids_by_group["hard_candidates"]
    dev_ids = ids_by_group["dev"]
    test_groups = [name for name in expected if name.startswith("test_")]
    test_ids = set().union(*(ids_by_group[name] for name in test_groups))
    id_overlap = {
        "train_dev": len(train_ids & dev_ids),
        "train_test": len(train_ids & test_ids),
        "dev_test": len(dev_ids & test_ids),
    }
    train_topologies = topology_by_group["sft_24k"] | topology_by_group["hard_candidates"]
    dev_topologies = topology_by_group["dev"]
    test_topologies = set().union(*(topology_by_group[name] for name in test_groups))
    topology_overlap = {
        "train_dev": len(train_topologies & dev_topologies),
        "train_test": len(train_topologies & test_topologies),
        "dev_test": len(dev_topologies & test_topologies),
    }
    if any(id_overlap.values()) or any(topology_overlap.values()):
        raise SystemExit(f"split leakage: ids={id_overlap}, topology={topology_overlap}")

    counterfactual = {}
    for row, _ in rows_from_files(inventories["test_counterfactual_ood"]):
        counterfactual.setdefault(row["counterfactual_pair_id"], {})[row["counterfactual_role"]] = row
    if len(counterfactual) != 500 or any(set(pair) != {"base", "variant"} for pair in counterfactual.values()):
        raise SystemExit("counterfactual pair leakage/inventory mismatch")
    for pair in counterfactual.values():
        base = json.loads(json.dumps(pair["base"]["gold_output"]))
        variant = json.loads(json.dumps(pair["variant"]["gold_output"]))
        if base["conditions"][0].pop("assertion") == variant["conditions"][0].pop("assertion") or base != variant:
            raise SystemExit("counterfactual pair changed more than one gold assertion")

    summary = {
        "status": "PASS",
        "candidate_rows": candidate_rows,
        "split_rows": observed_counts,
        "strict_reference_rows_validated": candidate_rows + sum(value for key, value in observed_counts.items() if key not in {"hard_prompts"}),
        "evidence_ids_valid": True,
        "sample_id_overlap": id_overlap,
        "topology_overlap": topology_overlap,
        "counterfactual_pairs": len(counterfactual),
        "gpu_work_started": False,
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
