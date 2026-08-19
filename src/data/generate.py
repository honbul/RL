from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and validate the complete synthetic schema-RLVR dataset")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, default=Path("data/candidates"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    num_shards = int(config["num_shards"])
    processes = []
    for shard_id in range(num_shards):
        command = [sys.executable, "-m", "src.data.generate_shard", "--config", str(args.config), "--shard-id", str(shard_id), "--num-shards", str(num_shards), "--output-dir", str(args.candidate_dir)]
        processes.append((shard_id, command, subprocess.Popen(command)))
    failures = []
    for shard_id, command, process in processes:
        code = process.wait()
        if code:
            failures.append({"shard_id": shard_id, "exit_code": code, "command": command})
    if failures:
        raise SystemExit(json.dumps({"status": "FAIL", "failures": failures}, sort_keys=True))
    subprocess.run([sys.executable, "-m", "src.data.build_splits", "--config", str(args.config), "--candidate-dir", str(args.candidate_dir), "--data-root", str(args.data_root)], check=True)
    subprocess.run([sys.executable, "-m", "src.data.validate_generated", "--config", str(args.config), "--data-root", str(args.data_root)], check=True)
    print(json.dumps({"status": "PASS", "num_shards": num_shards, "candidate_count": int(config["candidate_count"])}, sort_keys=True))


if __name__ == "__main__":
    main()
