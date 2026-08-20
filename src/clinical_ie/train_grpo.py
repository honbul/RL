from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from src.clinical_ie.common import iter_jsonl, sentence_ids, write_jsonl
from src.clinical_ie.rewards import score_completion
from src.common import configure_compiler, format_model_prompt, git_commit, load_yaml, package_versions, write_json
from src.gpu import GPUMemoryMonitor, require_idle_gpu
from src.train_grpo import _wait_for_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Train clinical IE verifier-only GRPO on the matched hard frontier")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="CPU-only data/reward/output contract check")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--policy-adapter", type=Path)
    parser.add_argument("--no-vllm", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    output_dir = args.output_dir or Path(config["output_dir"])
    if args.dry_run:
        source_files = [config["frontier_path"]] if Path(config["frontier_path"]).exists() else ["data/clinical_ie/hard/hard_candidates_8k.part-000.jsonl"]
        row = next(iter(iter_jsonl(source_files)))
        gold_text = json.dumps(row["gold_output"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        good = score_completion(gold_text, row["gold_output"], sentence_ids(row), row["acceptable_evidence_groups"])
        bad = score_completion("not-json", row["gold_output"], sentence_ids(row), row["acceptable_evidence_groups"])
        if good.reward != 1.0 or bad.reward != 0.0:
            raise SystemExit("reward callback dry-run failed")
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_dir / "raw_rollouts.jsonl", [
            {"sample_id": row["sample_id"], "global_step": 0, "completion": gold_text, "reward": good.reward, "reward_components": good.components, "failure_type": good.failure_type, "seed": config["seed"]},
            {"sample_id": row["sample_id"], "global_step": 0, "completion": "not-json", "reward": bad.reward, "reward_components": bad.components, "failure_type": bad.failure_type, "seed": config["seed"]},
        ])
        result = {"status": "PASS", "mode": "CPU_DRY_RUN", "model_loaded": False, "gpu_initialized": False, "gold_reward": good.reward, "error_reward": bad.reward, "output_dir": str(output_dir)}
        write_json(output_dir / "cpu_contract.json", result)
        print(json.dumps(result, sort_keys=True))
        return

    frontier = Path(config["frontier_path"])
    if not frontier.exists():
        raise SystemExit("frontier_4k.jsonl does not exist; run select_frontier first")
    if (output_dir / "train_metrics.json").exists():
        raise SystemExit(f"refusing to overwrite completed run: {output_dir}")
    rows = list(iter_jsonl([frontier]))
    if args.limit is not None:
        rows = rows[: args.limit]
    model_config = load_yaml(Path(config["model_config"]))
    policy_adapter = args.policy_adapter or Path(config["policy_adapter"])
    policy_gpu = int(config["policy_gpu"])
    rollout_gpu = int(config["rollout_gpu"])
    use_vllm = not args.no_vllm
    require_idle_gpu(policy_gpu)
    if use_vllm:
        require_idle_gpu(rollout_gpu)
        configure_compiler()
    output_dir.mkdir(parents=True, exist_ok=True)
    server = None
    server_log = None
    rollout_monitor = None
    port = int(config["vllm_server_port"])
    server_url = f"http://127.0.0.1:{port}"
    if use_vllm:
        server_log = (output_dir / "vllm_server.log").open("w", encoding="utf-8")
        command = [str(Path(os.sys.executable).parent / "trl"), "vllm-serve", "--model", model_config["model_id"], "--revision", model_config.get("revision", "main"), "--host", "127.0.0.1", "--port", str(port), "--tensor-parallel-size", "1", "--gpu-memory-utilization", str(config["vllm_gpu_memory_utilization"]), "--dtype", "bfloat16", "--max-model-len", str(int(model_config["max_prompt_length"]) + int(model_config["max_completion_length"])), "--enforce-eager"]
        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = str(rollout_gpu)
        server = subprocess.Popen(command, env=server_env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        rollout_monitor = GPUMemoryMonitor(rollout_gpu).start()
        _wait_for_server(server, server_url)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(policy_gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    set_seed(int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_id"], revision=model_config.get("revision", "main"), padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset_rows = []
    max_prompt_tokens = 0
    for row in rows:
        prompt = format_model_prompt(tokenizer, row["prompt"])
        token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if token_count > int(model_config["max_prompt_length"]):
            raise SystemExit(f"prompt token budget exceeded: {row['sample_id']}={token_count}")
        max_prompt_tokens = max(max_prompt_tokens, token_count)
        dataset_rows.append({
            "prompt": prompt,
            "sample_id": row["sample_id"],
            "gold_json": json.dumps(row["gold_output"], sort_keys=True, separators=(",", ":")),
            "evidence_json": json.dumps(row["acceptable_evidence_groups"], sort_keys=True, separators=(",", ":")),
            "sentence_ids_json": json.dumps(sorted(sentence_ids(row))),
        })
    dataset = Dataset.from_list(dataset_rows)
    base = AutoModelForCausalLM.from_pretrained(model_config["model_id"], revision=model_config.get("revision", "main"), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, policy_adapter, is_trainable=True)
    rollout_path = output_dir / "raw_rollouts.jsonl"
    reward_counter: Counter[str] = Counter()
    reward_sum = 0.0

    def clinical_reward(prompts: list[str], completions: list[str], sample_id: list[str], gold_json: list[str], evidence_json: list[str], sentence_ids_json: list[str], trainer_state: Any, **_: Any) -> list[float]:
        nonlocal reward_sum
        rewards = []
        with rollout_path.open("a", encoding="utf-8", newline="\n") as handle:
            for index, completion in enumerate(completions):
                scored = score_completion(completion, json.loads(gold_json[index]), set(json.loads(sentence_ids_json[index])), json.loads(evidence_json[index]))
                rewards.append(scored.reward)
                reward_sum += scored.reward
                reward_counter[scored.failure_type or "success"] += 1
                handle.write(json.dumps({"sample_id": sample_id[index], "global_step": int(trainer_state.global_step), "completion": completion, "reward": scored.reward, "reward_components": scored.components, "failure_type": scored.failure_type, "seed": config["seed"]}, ensure_ascii=False, sort_keys=True) + "\n")
        return rewards

    clinical_reward.__name__ = "clinical_semantic_reward"
    trainer_args = GRPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=float(config["num_train_epochs"]),
        max_steps=args.max_steps if args.max_steps is not None else -1,
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        bf16=True,
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        num_generations=int(config["num_generations"]),
        generation_batch_size=int(config["generation_batch_size"]),
        temperature=float(config["temperature"]),
        top_p=float(config["top_p"]),
        max_completion_length=int(config["max_completion_length"]),
        beta=float(config["beta"]),
        loss_type=config["loss_type"],
        mask_truncated_completions=bool(config["mask_truncated_completions"]),
        use_vllm=use_vllm,
        vllm_mode="server",
        vllm_server_base_url=server_url if use_vllm else None,
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=int(config["save_steps"]),
        save_total_limit=8,
        logging_steps=1,
        report_to=[],
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        shuffle_dataset=False,
    )
    resolved = {**config, "config_path": str(args.config), "git_commit": git_commit(), "rows": len(rows), "policy_adapter_resolved": str(policy_adapter), "use_vllm_resolved": use_vllm, "max_prompt_tokens_observed": max_prompt_tokens, "checkpoint_dev_evaluation_required": True}
    (output_dir / "run_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    policy_monitor = GPUMemoryMonitor(policy_gpu).start()
    try:
        trainer = GRPOTrainer(model=model, reward_funcs=clinical_reward, args=trainer_args, train_dataset=dataset, processing_class=tokenizer)
        result = trainer.train()
        trainer.save_model(str(output_dir / "adapter"))
    finally:
        policy_peak = policy_monitor.stop()
        rollout_peak = rollout_monitor.stop() if rollout_monitor is not None else 0
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
        if server_log is not None:
            server_log.close()
    metrics = {**result.metrics, "train_rows": len(rows), "policy_peak_vram_mb": policy_peak / (1024 * 1024), "rollout_peak_vram_mb": rollout_peak / (1024 * 1024), "max_prompt_tokens": max_prompt_tokens}
    write_json(output_dir / "train_metrics.json", metrics)
    write_jsonl(output_dir / "training_log.jsonl", trainer.state.log_history)
    rollout_count = sum(reward_counter.values())
    write_json(output_dir / "reward_summary.json", {"rollouts": rollout_count, "reward_mean": reward_sum / rollout_count if rollout_count else 0.0, "failure_counts": dict(sorted(reward_counter.items()))})
    (output_dir / "checkpoint_eval.csv").write_text("checkpoint,dev_graph_f1,status\n", encoding="utf-8")
    versions = package_versions(["torch", "transformers", "trl", "peft", "datasets", "vllm"])
    lines = [f"platform={platform.platform()}", f"policy_gpu=NVIDIA RTX 4090 physical:{policy_gpu}", f"rollout_gpu=NVIDIA RTX 4090 physical:{rollout_gpu}", *[f"{key}={value}" for key, value in versions.items()]]
    (output_dir / "system_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), **metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
