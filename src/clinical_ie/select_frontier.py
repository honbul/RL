from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.clinical_ie.common import iter_jsonl, write_jsonl
from src.common import write_json


def select_ids(rollouts: list[dict[str, Any]], target_count: int) -> tuple[list[str], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollouts:
        grouped[row["sample_id"]].append(row)
    ranked = []
    stats = {"mixed_success_failure": 0, "reward_020_080": 0, "partial_nonzero": 0, "all_zero_partial": 0, "excluded_all_correct": 0, "excluded_all_schema_invalid": 0}
    for sample_id, rows in grouped.items():
        rewards = [float(row["reward"]) for row in rows]
        schema_valid = [bool(row.get("schema_valid", row.get("parsed_output") is not None)) for row in rows]
        success = [reward >= 0.999999 for reward in rewards]
        mixed = any(success) and not all(success)
        mid = any(0.20 <= reward <= 0.80 for reward in rewards)
        partial = any(0.0 < reward < 1.0 for reward in rewards)
        field_partial = any(float(row.get("field_partial_score", 0.0)) > 0 for row in rows)
        if all(success):
            stats["excluded_all_correct"] += 1
            continue
        if not any(schema_valid):
            stats["excluded_all_schema_invalid"] += 1
            continue
        if mixed:
            priority = 0
            stats["mixed_success_failure"] += 1
        elif mid:
            priority = 1
            stats["reward_020_080"] += 1
        elif partial:
            priority = 2
            stats["partial_nonzero"] += 1
        elif field_partial:
            priority = 3
            stats["all_zero_partial"] += 1
        else:
            priority = 4
        variance = max(rewards) - min(rewards)
        ranked.append((priority, -variance, abs(sum(rewards) / len(rewards) - 0.5), sample_id, rewards))
    ranked.sort()
    selected = [row[3] for row in ranked[:target_count]]
    if len(selected) < target_count:
        raise ValueError(f"frontier candidates insufficient: {len(selected)} < {target_count}")
    stats.update({"prompt_groups": len(grouped), "eligible_groups": len(ranked), "selected": len(selected)})
    return selected, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the matched 4K hard frontier from four rollouts per prompt")
    parser.add_argument("--rollouts", type=Path)
    parser.add_argument("--hard-files", nargs="*", default=["data/clinical_ie/hard/hard_candidates_8k.part-000.jsonl", "data/clinical_ie/hard/hard_candidates_8k.part-001.jsonl"])
    parser.add_argument("--output", type=Path, default=Path("data/clinical_ie/hard/frontier_4k.jsonl"))
    parser.add_argument("--stats", type=Path, default=Path("data/clinical_ie/manifests/frontier_selection.json"))
    parser.add_argument("--target-count", type=int, default=4000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        rollouts = []
        for index in range(12):
            for generation in range(4):
                reward = (generation % 2) if index < 6 else 0.5
                rollouts.append({"sample_id": f"dummy_{index:03d}", "reward": reward, "schema_valid": True, "field_partial_score": reward})
        selected, stats = select_ids(rollouts, 8)
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(output, ({"sample_id": sample_id, "prompt": "dummy", "completion": "{}"} for sample_id in selected))
        write_json(args.stats, {**stats, "status": "PASS", "mode": "CPU_DRY_RUN"})
        print(json.dumps({"status": "PASS", "mode": "CPU_DRY_RUN", "selected": len(selected), "output": str(output)}, sort_keys=True))
        return
    if args.rollouts is None or not args.rollouts.exists():
        raise SystemExit("--rollouts is required for formal selection")
    rollouts = [json.loads(line) for line in args.rollouts.read_text(encoding="utf-8").splitlines() if line]
    selected_ids, stats = select_ids(rollouts, args.target_count)
    selected_set = set(selected_ids)
    rank = {sample_id: index for index, sample_id in enumerate(selected_ids)}
    selected_rows = []
    for row in iter_jsonl(args.hard_files):
        if row["sample_id"] in selected_set:
            selected_rows.append({**row, "completion": json.dumps(row["gold_output"], ensure_ascii=False, sort_keys=True, separators=(",", ":")), "frontier_rank": rank[row["sample_id"]]})
    selected_rows.sort(key=lambda row: row["frontier_rank"])
    if len(selected_rows) != args.target_count:
        raise SystemExit("selected hard rows did not join to target count")
    write_jsonl(args.output, selected_rows)
    write_json(args.stats, {**stats, "status": "PASS", "rollout_rows": len(rollouts), "target_count": args.target_count})
    print(json.dumps({"status": "PASS", "selected": len(selected_rows), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
