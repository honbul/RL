from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_RAW_FIELDS = {
    "run_id", "model_id", "sample_id", "split", "prompt", "completion",
    "parsed_program", "parse_error", "reward_components", "hidden_case_results",
    "metrics", "failure_type", "prompt_tokens", "completion_tokens", "latency_ms",
    "seed", "config_path", "git_commit",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the Stage 3 local GPU smoke artifacts")
    parser.add_argument("--root", type=Path, default=Path("outputs/smoke"))
    args = parser.parse_args()
    sft = args.root / "sft_10_steps"
    grpo = args.root / "grpo_binary_10_steps_attempt2"
    e0 = args.root / "e0_eval_50_attempt2/test_iid/raw_generations.jsonl"
    sft_eval = args.root / "sft_eval_50/test_iid/raw_generations.jsonl"
    sft_metrics = _json(sft / "train_metrics.json")
    grpo_metrics = _json(grpo / "train_metrics.json")
    if sft_metrics["train_rows"] != 100 or grpo_metrics["train_rows"] != 100:
        raise SystemExit("smoke train row count mismatch")
    if not (sft / "adapter/adapter_model.safetensors").exists():
        raise SystemExit("SFT adapter missing")
    if not (grpo / "adapter/adapter_model.safetensors").exists():
        raise SystemExit("GRPO adapter missing")
    e0_rows = [json.loads(line) for line in e0.read_text(encoding="utf-8").splitlines()]
    sft_eval_rows = [json.loads(line) for line in sft_eval.read_text(encoding="utf-8").splitlines()]
    rollout_rows = [json.loads(line) for line in (grpo / "raw_rollouts.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(e0_rows) != 50 or len(sft_eval_rows) != 50 or len(rollout_rows) != 40:
        raise SystemExit("smoke raw row count mismatch")
    for row in e0_rows + sft_eval_rows:
        missing = REQUIRED_RAW_FIELDS - set(row)
        if missing:
            raise SystemExit(f"raw generation fields missing: {sorted(missing)}")
    summary = {
        "status": "PASS",
        "sft_steps": 10,
        "grpo_steps": 10,
        "e0_eval_rows": len(e0_rows),
        "sft_adapter_eval_rows": len(sft_eval_rows),
        "grpo_rollout_rows": len(rollout_rows),
        "sft_peak_vram_mb": sft_metrics["peak_vram_mb"],
        "grpo_policy_peak_vram_mb": grpo_metrics["policy_peak_vram_mb"],
        "grpo_rollout_peak_vram_mb": grpo_metrics["rollout_peak_vram_mb"],
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
