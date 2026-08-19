from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from src.common import (
    configure_compiler,
    config_parser,
    format_model_prompt,
    git_commit,
    load_yaml,
    package_versions,
    write_json,
)
from src.gpu import GPUMemoryMonitor, require_idle_gpu
from src.rewards import score_completion


def _wait_for_server(process: subprocess.Popen[Any], url: str, timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited before readiness: {process.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"vLLM server did not become ready within {timeout} seconds")


def main() -> None:
    parser = config_parser("Train an execution-rewarded policy with GRPO")
    parser.add_argument("--max-steps", type=int, default=None, help="Bounded smoke override")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N prompts")
    parser.add_argument("--output-dir", type=Path, default=None, help="Unique output-root override")
    parser.add_argument("--policy-adapter", type=Path, default=None, help="Smoke/start-policy override")
    parser.add_argument("--no-vllm", action="store_true", help="Use Transformers generation fallback")
    args = parser.parse_args()
    config = load_yaml(args.config)
    model_config = load_yaml(Path(config["model_config"]))
    output_dir = args.output_dir or Path(config["output_dir"])
    policy_adapter = args.policy_adapter or Path(config["policy_adapter"])
    if (output_dir / "train_metrics.json").exists():
        raise SystemExit(f"refusing to overwrite completed run: {output_dir}")
    policy_gpu = int(config.get("policy_gpu", 0))
    rollout_gpu = int(config.get("rollout_gpu", 1))
    use_vllm = bool(config.get("use_vllm", True)) and not args.no_vllm
    require_idle_gpu(policy_gpu)
    if use_vllm:
        if policy_gpu == rollout_gpu:
            raise SystemExit("policy_gpu and rollout_gpu must differ in server mode")
        require_idle_gpu(rollout_gpu)
        configure_compiler()
    output_dir.mkdir(parents=True, exist_ok=True)
    port = int(config.get("vllm_server_port", 8000))
    server_url = f"http://127.0.0.1:{port}"
    server: subprocess.Popen[Any] | None = None
    server_log_handle = None
    rollout_monitor: GPUMemoryMonitor | None = None
    if use_vllm:
        server_log_handle = (output_dir / "vllm_server.log").open("w", encoding="utf-8")
        trl_executable = Path(os.sys.executable).parent / "trl"
        server_command = [
            str(trl_executable), "vllm-serve", "--model", model_config["model_id"],
            "--revision", model_config.get("revision", "main"),
            "--host", "127.0.0.1", "--port", str(port),
            "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", str(config.get("vllm_gpu_memory_utilization", 0.85)),
            "--dtype", "bfloat16", "--max-model-len", "2304", "--enforce-eager",
        ]
        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = str(rollout_gpu)
        server = subprocess.Popen(
            server_command,
            env=server_env,
            stdout=server_log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_id"], revision=model_config.get("revision", "main"), padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = []
    with Path(config["data_path"]).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                prompts.append(json.loads(line))
                if args.limit is not None and len(prompts) >= args.limit:
                    break
    hidden_by_id = {}
    with Path("data/grpo/hidden_tests.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            hidden_by_id[row["sample_id"]] = row["hidden_tests"]
    dataset_rows = []
    max_prompt_tokens = 0
    for row in prompts:
        model_prompt = format_model_prompt(tokenizer, row["prompt"])
        token_count = len(tokenizer(model_prompt, add_special_tokens=False)["input_ids"])
        max_prompt_tokens = max(max_prompt_tokens, token_count)
        if token_count > int(model_config["max_prompt_length"]):
            raise SystemExit(f"prompt token budget exceeded: {row['sample_id']}={token_count}")
        dataset_rows.append({
            "prompt": model_prompt,
            "sample_id": row["sample_id"],
            "source_schema": row["source_schema"],
            "target_schema": row["target_schema"],
            "hidden_tests": hidden_by_id[row["sample_id"]],
        })
    dataset = Dataset.from_list(dataset_rows)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config["model_id"],
        revision=model_config.get("revision", "main"),
        torch_dtype=torch.bfloat16,
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
    model = PeftModel.from_pretrained(base_model, policy_adapter, is_trainable=True)
    rollout_path = output_dir / "raw_rollouts.jsonl"

    def execution_reward(
        prompts: list[str],
        completions: list[str],
        sample_id: list[str],
        source_schema: list[dict[str, Any]],
        target_schema: list[dict[str, Any]],
        hidden_tests: list[list[dict[str, Any]]],
        trainer_state: Any,
        **_: Any,
    ) -> list[float]:
        rewards = []
        with rollout_path.open("a", encoding="utf-8", newline="\n") as handle:
            for index, completion in enumerate(completions):
                scored = score_completion(
                    completion,
                    source_schema[index],
                    target_schema[index],
                    hidden_tests[index],
                    config["reward"],
                )
                rewards.append(scored.reward)
                handle.write(json.dumps({
                    "sample_id": sample_id[index],
                    "completion": completion,
                    "reward": scored.reward,
                    "reward_components": scored.components,
                    "failure_type": scored.failure_type,
                    "global_step": int(trainer_state.global_step),
                    "seed": int(config["seed"]),
                }, ensure_ascii=False, sort_keys=True) + "\n")
        return rewards

    execution_reward.__name__ = f"{config['reward']}_execution_reward"
    trainer_args = GRPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        overwrite_output_dir=False,
        num_train_epochs=float(config["num_train_epochs"]),
        max_steps=args.max_steps if args.max_steps is not None else -1,
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        bf16=True,
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        num_generations=int(config["num_generations"]),
        generation_batch_size=int(config["num_generations"]),
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
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        shuffle_dataset=False,
    )
    resolved = {
        **config,
        "config_path": str(args.config),
        "output_dir": str(output_dir),
        "limit": args.limit,
        "max_steps_override": args.max_steps,
        "policy_adapter_resolved": str(policy_adapter),
        "use_vllm_resolved": use_vllm,
        "git_commit": git_commit(),
        "rows": len(dataset_rows),
        "max_prompt_tokens_observed": max_prompt_tokens,
    }
    (output_dir / "run_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    policy_monitor = GPUMemoryMonitor(policy_gpu).start()
    try:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=execution_reward,
            args=trainer_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
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
        if server_log_handle is not None:
            server_log_handle.close()
    metrics = dict(result.metrics)
    metrics.update({
        "policy_peak_vram_mb": policy_peak / (1024 * 1024),
        "rollout_peak_vram_mb": rollout_peak / (1024 * 1024),
        "train_rows": len(dataset_rows),
        "max_prompt_tokens": max_prompt_tokens,
    })
    write_json(output_dir / "train_metrics.json", metrics)
    with (output_dir / "training_log.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in trainer.state.log_history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    versions = package_versions(["torch", "transformers", "trl", "peft", "datasets", "vllm"])
    system_lines = [f"platform={platform.platform()}", f"policy_gpu=NVIDIA RTX 4090 physical:{policy_gpu}", f"rollout_gpu=NVIDIA RTX 4090 physical:{rollout_gpu}"]
    system_lines.extend(f"{key}={value}" for key, value in versions.items())
    system_lines.extend([f"policy_peak_vram_mb={metrics['policy_peak_vram_mb']:.2f}", f"rollout_peak_vram_mb={metrics['rollout_peak_vram_mb']:.2f}"])
    (output_dir / "system_info.txt").write_text("\n".join(system_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), **metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
