from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the complete 40K clinical IE candidate pool")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--candidate-dir", type=Path, default=Path("data/clinical_ie/candidates"))
    parser.add_argument("--log", type=Path, default=Path("outputs/clinical_ie/pre_gpu/data_generation.log"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    total = int(config["candidate_count"])
    shards = int(config["candidate_shards"])
    workers = min(int(args.workers or config["workers"]), 8)
    if total % shards:
        raise SystemExit("candidate_count must divide evenly by candidate_shards")
    args.candidate_dir.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    existing = list(args.candidate_dir.glob("candidate_shard_*.jsonl"))
    if existing:
        raise SystemExit(f"refusing to overwrite {len(existing)} existing candidate shards")
    per_shard = total // shards

    def run_shard(shard_id: int) -> dict:
        start = shard_id * per_shard
        end = start + per_shard
        output = args.candidate_dir / f"candidate_shard_{shard_id:03d}.jsonl"
        command = [
            sys.executable,
            "-m",
            "src.clinical_ie.data.generate_shard",
            "--start",
            str(start),
            "--end",
            str(end),
            "--output",
            str(output),
            "--seed",
            str(config["seed"]),
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f"shard {shard_id} failed: {result.stderr}")
        return json.loads(result.stdout.strip().splitlines()[-1])

    summaries = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_shard, shard_id): shard_id for shard_id in range(shards)}
        for future in as_completed(futures):
            summaries.append(future.result())
    summaries.sort(key=lambda row: row["start"])
    if sum(row["rows"] for row in summaries) != total:
        raise SystemExit("generated row count mismatch")
    log_payload = {
        "status": "PASS",
        "workers": workers,
        "candidate_shards": shards,
        "candidate_count": total,
        "shards": summaries,
    }
    args.log.write_text(json.dumps(log_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: log_payload[key] for key in ("status", "workers", "candidate_shards", "candidate_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
