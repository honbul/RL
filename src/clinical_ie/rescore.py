from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from src.clinical_ie.common import iter_jsonl, sentence_ids, write_jsonl
from src.clinical_ie.evaluate import SPLITS, aggregate_rows
from src.clinical_ie.rewards import score_completion
from src.common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute clinical IE metrics from persisted raw generations")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    aggregate = {}
    for split, paths in SPLITS.items():
        raw_path = args.run_dir / split / "raw_generations.jsonl"
        if not raw_path.exists():
            continue
        source = {row["sample_id"]: row for row in iter_jsonl(paths)}
        raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line]
        failures: Counter[str] = Counter()
        difficulty: dict[str, list[float]] = defaultdict(list)
        rescored = []
        for raw in raw_rows:
            row = source[raw["sample_id"]]
            scored = score_completion(raw["completion"], row["gold_output"], sentence_ids(row), row["acceptable_evidence_groups"])
            failure = "truncated_output" if raw.get("failure_type") == "truncated_output" and scored.components.get("record_exact") != 1.0 else scored.failure_type
            if failure:
                failures[failure] += 1
            updated = {**raw, "parsed_output": scored.parsed_output, "reward": scored.reward, "reward_components": scored.components, "failure_type": failure, "counterfactual_pair_id": row["counterfactual_pair_id"]}
            rescored.append(updated)
            for tag in row["difficulty_tags"]:
                difficulty[tag].append(scored.components.get("graph_f1", 0.0))
        if split == "test_counterfactual_ood":
            pairs: dict[str, list[dict]] = defaultdict(list)
            for row in rescored:
                pairs[row["counterfactual_pair_id"]].append(row)
            for pair_rows in pairs.values():
                consistency = float(len(pair_rows) == 2 and all(row["reward_components"].get("record_exact") == 1.0 for row in pair_rows))
                for row in pair_rows:
                    row["counterfactual_consistency"] = consistency
        old = json.loads((args.run_dir / split / "eval_metrics.json").read_text(encoding="utf-8"))
        metrics = aggregate_rows(rescored, float(old["peak_vram_mb"]), float(old["generation_wall_seconds"]))
        metrics["rescored_from_raw"] = True
        write_json(args.run_dir / split / "eval_metrics.json", metrics)
        write_json(args.run_dir / split / "failure_summary.json", dict(sorted(failures.items())))
        write_json(args.run_dir / split / "difficulty_summary.json", {tag: {"samples": len(values), "graph_f1": statistics.fmean(values)} for tag, values in sorted(difficulty.items())})
        aggregate[split] = metrics
    write_json(args.run_dir / "eval_metrics.json", aggregate)
    print(json.dumps({"status": "PASS", "run_dir": str(args.run_dir), "splits": len(aggregate)}, sort_keys=True))


if __name__ == "__main__":
    main()
