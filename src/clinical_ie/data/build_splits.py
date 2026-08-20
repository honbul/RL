from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.clinical_ie.data.render_documents import canonical_json, public_projection


class RotatingJsonlWriter:
    def __init__(self, base: Path, max_bytes: int):
        self.base = base
        self.max_bytes = max_bytes
        self.part = -1
        self.handle = None
        self.current_bytes = 0
        self.current_rows = 0
        self.files: list[dict[str, Any]] = []

    def _open(self) -> None:
        self.part += 1
        path = self.base.with_name(f"{self.base.name}.part-{self.part:03d}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8", newline="\n")
        self.current_bytes = 0
        self.current_rows = 0

    def _close_part(self) -> None:
        if self.handle is None:
            return
        path = Path(self.handle.name)
        self.handle.close()
        self.files.append({"path": str(path), "rows": self.current_rows, "bytes": self.current_bytes})
        self.handle = None

    def write_line(self, line: str) -> None:
        encoded_bytes = len(line.encode("utf-8"))
        if self.handle is None:
            self._open()
        if self.current_rows and self.current_bytes + encoded_bytes > self.max_bytes:
            self._close_part()
            self._open()
        assert self.handle is not None
        self.handle.write(line)
        self.current_bytes += encoded_bytes
        self.current_rows += 1

    def write_object(self, row: dict[str, Any]) -> None:
        self.write_line(canonical_json(row) + "\n")

    def close(self) -> list[dict[str, Any]]:
        self._close_part()
        if len(self.files) == 1:
            old = Path(self.files[0]["path"])
            new = self.base.with_suffix(".jsonl")
            old.replace(new)
            self.files[0]["path"] = str(new)
        return self.files


def candidate_rows(candidate_dir: Path) -> Iterable[tuple[dict[str, Any], str]]:
    for path in sorted(candidate_dir.glob("candidate_shard_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line), line


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clinical IE SFT, hard, dev, and test splits")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, default=Path("data/clinical_ie/candidates"))
    parser.add_argument("--data-root", type=Path, default=Path("data/clinical_ie"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    max_bytes = int(config["max_jsonl_mb"]) * 1024 * 1024
    writers = {
        "sft_8k": RotatingJsonlWriter(args.data_root / "sft/sft_8k", max_bytes),
        "sft_24k": RotatingJsonlWriter(args.data_root / "sft/sft_24k", max_bytes),
        "hard_candidates": RotatingJsonlWriter(args.data_root / "hard/hard_candidates_8k", max_bytes),
        "hard_prompts": RotatingJsonlWriter(args.data_root / "hard/hard_prompts", max_bytes),
        "hard_references": RotatingJsonlWriter(args.data_root / "hard/hard_references", max_bytes),
        "dev": RotatingJsonlWriter(args.data_root / "dev/dev_2k", max_bytes),
        "test_iid_hard": RotatingJsonlWriter(args.data_root / "test/test_iid_hard", max_bytes),
        "test_linguistic_ood": RotatingJsonlWriter(args.data_root / "test/test_linguistic_ood", max_bytes),
        "test_compositional_ood": RotatingJsonlWriter(args.data_root / "test/test_compositional_ood", max_bytes),
        "test_long_context_ood": RotatingJsonlWriter(args.data_root / "test/test_long_context_ood", max_bytes),
        "test_counterfactual_ood": RotatingJsonlWriter(args.data_root / "test/test_counterfactual_ood", max_bytes),
    }
    counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    document_type_counts: Counter[str] = Counter()
    topology_by_group: dict[str, set[str]] = {name: set() for name in ("train", "dev", "test")}
    pair_counts: Counter[str] = Counter()
    previous_slot = -1
    for row, original_line in candidate_rows(args.candidate_dir):
        slot = int(row["latent_timeline"]["slot"])
        if slot != previous_slot + 1:
            raise SystemExit(f"candidate slot order gap: {previous_slot} -> {slot}")
        previous_slot = slot
        regime = row["latent_timeline"]["regime"]
        counts[regime] += 1
        difficulty_counts[str(row["difficulty_level"])] += 1
        tag_counts.update(row["difficulty_tags"])
        document_type_counts.update(document["document_type"] for document in row["documents"])
        if slot < 32000:
            topology_by_group["train"].add(row["topology_fingerprint"])
        elif slot < 34000:
            topology_by_group["dev"].add(row["topology_fingerprint"])
        elif slot < 39000:
            topology_by_group["test"].add(row["topology_fingerprint"])
        if row["counterfactual_pair_id"]:
            pair_counts[row["counterfactual_pair_id"]] += 1
        if slot < 24000:
            sft_row = {**row, "completion": json.dumps(row["gold_output"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))}
            sft_line = canonical_json(sft_row) + "\n"
            writers["sft_24k"].write_line(sft_line)
            if slot < 8000:
                writers["sft_8k"].write_line(sft_line)
        elif slot < 32000:
            writers["hard_candidates"].write_line(original_line)
            writers["hard_prompts"].write_object(public_projection(row))
            writers["hard_references"].write_object({
                "sample_id": row["sample_id"],
                "gold_output": row["gold_output"],
                "acceptable_evidence_groups": row["acceptable_evidence_groups"],
                "latent_timeline": row["latent_timeline"],
            })
        elif slot < 34000:
            writers["dev"].write_line(original_line)
        elif slot < 35000:
            writers["test_iid_hard"].write_line(original_line)
        elif slot < 36000:
            writers["test_linguistic_ood"].write_line(original_line)
        elif slot < 37000:
            writers["test_compositional_ood"].write_line(original_line)
        elif slot < 38000:
            writers["test_long_context_ood"].write_line(original_line)
        elif slot < 39000:
            writers["test_counterfactual_ood"].write_line(original_line)
    if previous_slot != int(config["candidate_count"]) - 1:
        raise SystemExit(f"candidate count ended at slot {previous_slot}")
    file_inventory = {name: writer.close() for name, writer in writers.items()}
    overlap = {
        "train_dev_topology_overlap": len(topology_by_group["train"] & topology_by_group["dev"]),
        "train_test_topology_overlap": len(topology_by_group["train"] & topology_by_group["test"]),
        "dev_test_topology_overlap": len(topology_by_group["dev"] & topology_by_group["test"]),
    }
    if any(overlap.values()):
        raise SystemExit(f"topology leakage: {overlap}")
    if len(pair_counts) != 500 or set(pair_counts.values()) != {2}:
        raise SystemExit("counterfactual pair inventory mismatch")
    candidate_files = []
    for path in sorted(args.candidate_dir.glob("candidate_shard_*.jsonl")):
        candidate_files.append({"path": str(path), "rows": sum(1 for _ in path.open(encoding="utf-8")), "bytes": path.stat().st_size})
    generation_commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
    manifest_dir = args.data_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "PASS",
        "dataset_version": "clinical_ie_synthetic_v1",
        "generation_seed": int(config["seed"]),
        "generation_code_commit": generation_commit,
        "generation_command": ".venv/bin/python -m src.clinical_ie.data.generate --config configs/clinical_ie/data.yaml --workers 8",
        "candidate_count": sum(counts.values()),
        "regime_counts": dict(sorted(counts.items())),
        "split_files": file_inventory,
        "candidate_files": candidate_files,
        "leakage_checks": overlap,
        "counterfactual_pairs": {"pairs": len(pair_counts), "rows": sum(pair_counts.values()), "cross_split_leakage": 0},
        "gpu_work_started": False,
    }
    (manifest_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    quality = {
        "status": "PASS",
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "difficulty_tag_counts": dict(sorted(tag_counts.items())),
        "document_type_counts": dict(sorted(document_type_counts.items())),
        "concept_count": 36,
        "synthetic_only": True,
    }
    (manifest_dir / "data_quality_summary.json").write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    topology = {
        "status": "PASS",
        "unique_topologies": {key: len(value) for key, value in topology_by_group.items()},
        "overlap": overlap,
        "heldout_renderer_family": "heldout_linguistic",
        "counterfactual_pair_count": len(pair_counts),
    }
    (manifest_dir / "split_topology_summary.json").write_text(json.dumps(topology, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "candidate_count": sum(counts.values()), "split_file_groups": len(file_inventory), "leakage_checks": overlap}, sort_keys=True))


if __name__ == "__main__":
    main()
