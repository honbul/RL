from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from src.data.synthetic import make_candidate, verify_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and validate one deterministic candidate shard")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/candidates"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    expected_shards = int(config["num_shards"])
    if args.num_shards != expected_shards:
        raise SystemExit(f"num_shards must equal config value {expected_shards}")
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("shard-id out of range")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"candidate_shard_{args.shard_id:03d}.jsonl"
    summary_path = args.output_dir / f"candidate_shard_{args.shard_id:03d}.summary.json"
    count = int(config["candidate_count"])
    base_seed = int(config["seed"])
    partition_counts: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()
    written = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(args.shard_id, count, args.num_shards):
            row = make_candidate(index, base_seed)
            verify_candidate(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            partition_counts[row["partition"]] += 1
            pattern_counts[row["graph_signature"]] += 1
            written += 1
    summary = {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "rows": written,
        "first_index": args.shard_id,
        "last_index": max(range(args.shard_id, count, args.num_shards)),
        "partition_counts": dict(sorted(partition_counts.items())),
        "pattern_counts": dict(sorted(pattern_counts.items())),
        "status": "PASS",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
