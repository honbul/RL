from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.data.synthetic import materialize_hidden_cases, public_projection


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    rows = []
    for shard in sorted(path.glob("candidate_shard_*.jsonl")):
        with shard.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    rows.sort(key=lambda row: row["global_index"])
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def render_prompt(row: dict[str, Any]) -> str:
    visible = json.dumps(row["visible_examples"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (
        "Source JSON schema:\n" + json.dumps(row["source_schema"], ensure_ascii=False, sort_keys=True, separators=(",", ":")) +
        "\nTarget JSON schema:\n" + json.dumps(row["target_schema"], ensure_ascii=False, sort_keys=True, separators=(",", ":")) +
        "\nTransformation requirement:\n" + row["instruction"] +
        "\nVisible examples:\n" + visible +
        "\nReturn only one JSON object conforming to the restricted DSL schema. Do not include prose or Markdown fences."
    )


def _sft_row(row: dict[str, Any]) -> dict[str, Any]:
    result = public_projection(row)
    result["reference_program"] = row["reference_program"]
    result["prompt"] = render_prompt(row)
    result["completion"] = json.dumps(row["reference_program"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return result


def _public_prompt(row: dict[str, Any]) -> dict[str, Any]:
    result = public_projection(row)
    result["prompt"] = render_prompt(row)
    return result


def _evaluation_row(row: dict[str, Any], split: str, hidden_count: int) -> dict[str, Any]:
    result = _public_prompt(row)
    result["split"] = split
    result["reference_program"] = row["reference_program"]
    result["hidden_tests"] = materialize_hidden_cases(row, hidden_count, 5000)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic SFT, GRPO, dev, IID, and OOD splits")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, default=Path("data/candidates"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    counts = config["counts"]
    candidates = _load_candidates(args.candidate_dir)
    expected_candidates = int(config["candidate_count"])
    if len(candidates) != expected_candidates:
        raise SystemExit(f"candidate count mismatch: {len(candidates)} != {expected_candidates}")
    if [row["global_index"] for row in candidates] != list(range(expected_candidates)):
        raise SystemExit("candidate global indices are incomplete or duplicated")
    base = [row for row in candidates if row["partition"] == "base"]
    linguistic = [row for row in candidates if row["partition"] == "linguistic"]
    compositional = [row for row in candidates if row["partition"] == "compositional"]
    structural = [row for row in candidates if row["partition"] == "structural"]

    sft_6k = base[: int(counts["sft_6k"])]
    sft_1k = sft_6k[: int(counts["sft_1k"])]
    cursor = int(counts["sft_6k"])
    grpo = base[cursor: cursor + int(counts["grpo_train"])]
    cursor += int(counts["grpo_train"])
    dev = base[cursor: cursor + int(counts["dev"])]
    cursor += int(counts["dev"])
    iid = base[cursor: cursor + int(counts["test_iid"])]
    ling = linguistic[: int(counts["test_linguistic_ood"])]
    comp = compositional[: int(counts["test_compositional_ood"])]
    struct = structural[: int(counts["test_structural_ood"])]

    written = {}
    written["sft_1k"] = _write_jsonl(args.data_root / "sft/sft_1k.jsonl", (_sft_row(row) for row in sft_1k))
    written["sft_6k"] = _write_jsonl(args.data_root / "sft/sft_6k.jsonl", (_sft_row(row) for row in sft_6k))
    written["grpo_prompts"] = _write_jsonl(args.data_root / "grpo/prompts.jsonl", (_public_prompt(row) for row in grpo))
    hidden_count = int(config["hidden_cases"]["grpo"])
    written["grpo_hidden_tests"] = _write_jsonl(args.data_root / "grpo/hidden_tests.jsonl", ({"sample_id": row["sample_id"], "hidden_tests": materialize_hidden_cases(row, hidden_count, 4000)} for row in grpo))
    written["grpo_references"] = _write_jsonl(args.data_root / "grpo/references.jsonl", ({"sample_id": row["sample_id"], "reference_program": row["reference_program"]} for row in grpo))
    eval_hidden = int(config["hidden_cases"]["evaluation"])
    written["dev"] = _write_jsonl(args.data_root / "dev/dev.jsonl", (_evaluation_row(row, "dev", eval_hidden) for row in dev))
    for key, split_rows, name in (
        ("test_iid", iid, "test_iid"),
        ("test_linguistic_ood", ling, "test_linguistic_ood"),
        ("test_compositional_ood", comp, "test_compositional_ood"),
        ("test_structural_ood", struct, "test_structural_ood"),
    ):
        written[key] = _write_jsonl(args.data_root / f"test/{name}.jsonl", (_evaluation_row(row, name, eval_hidden) for row in split_rows))

    split_rows = {"sft_6k": sft_6k, "grpo": grpo, "dev": dev, "test_iid": iid, "test_linguistic_ood": ling, "test_compositional_ood": comp, "test_structural_ood": struct}
    train_rows = sft_6k + grpo
    train_ids = {row["sample_id"] for row in train_rows}
    test_rows = iid + ling + comp + struct
    leakage = {
        "sample_id_overlap_train_test": len(train_ids & {row["sample_id"] for row in test_rows}),
        "graph_fingerprint_overlap_train_test": len({row["graph_fingerprint"] for row in train_rows} & {row["graph_fingerprint"] for row in test_rows}),
        "compositional_signature_overlap_train": len({row["graph_signature"] for row in train_rows} & {row["graph_signature"] for row in comp}),
        "linguistic_template_overlap_train": len({row["instruction_template_family"] for row in train_rows} & {row["instruction_template_family"] for row in ling}),
    }
    if any(leakage.values()):
        raise SystemExit(f"split leakage detected: {leakage}")
    manifest = {
        "seed": int(config["seed"]),
        "candidate_count": len(candidates),
        "counts": written,
        "hidden_cases": config["hidden_cases"],
        "leakage_checks": leakage,
        "sft_1k_is_prefix_subset_of_sft_6k": [row["sample_id"] for row in sft_1k] == [row["sample_id"] for row in sft_6k[: len(sft_1k)]],
        "status": "PASS",
    }
    manifests = args.data_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "data_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    quality = {
        "partition_counts": dict(sorted(Counter(row["partition"] for row in candidates).items())),
        "pattern_counts": dict(sorted(Counter(row["graph_signature"] for row in candidates).items())),
        "split_pattern_counts": {name: dict(sorted(Counter(row["graph_signature"] for row in rows).items())) for name, rows in split_rows.items()},
        "difficulty_level_counts": dict(sorted(Counter(str(row["difficulty"]["level"]) for row in candidates).items())),
        "has_array": sum(bool(row["difficulty"]["has_array"]) for row in candidates),
        "has_null": sum(bool(row["difficulty"]["has_null"]) for row in candidates),
        "has_condition": sum(bool(row["difficulty"]["has_condition"]) for row in candidates),
        "status": "PASS",
    }
    (manifests / "data_quality_summary.json").write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
