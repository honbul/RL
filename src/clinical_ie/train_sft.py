from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import yaml

from src.clinical_ie.common import load_sft_rows, write_jsonl
from src.common import format_model_prompt, git_commit, load_yaml, package_versions, write_json
from src.gpu import GPUMemoryMonitor, require_idle_gpu


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a clinical IE completion-only SFT LoRA adapter")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="CPU-only config/data/output contract check")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_yaml(args.config)
    if config.get("blocked_pre_gpu"):
        raise SystemExit(f"blocked config: {config['block_reason']}")
    required = ["model_config", "data_files", "output_dir", "seed", "lora"]
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"missing config keys: {missing}")
    row_limit = args.limit if args.limit is not None else int(config.get("max_rows", 0)) or None
    rows = load_sft_rows(config["data_files"], int(config.get("skip_rows", 0)), row_limit)
    output_dir = args.output_dir or Path(config["output_dir"])
    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe_rows = [{"sample_id": row["sample_id"], "completion": row["completion"]} for row in rows[:2]]
        write_jsonl(output_dir / "raw_writer_probe.jsonl", probe_rows)
        result = {
            "status": "PASS",
            "mode": "CPU_DRY_RUN",
            "config": str(args.config),
            "rows_loaded": len(rows),
            "first_sample_id": rows[0]["sample_id"] if rows else None,
            "output_dir": str(output_dir),
            "model_loaded": False,
            "gpu_initialized": False,
        }
        write_json(output_dir / "cpu_contract.json", result)
        print(json.dumps(result, sort_keys=True))
        return

    if (output_dir / "train_metrics.json").exists():
        raise SystemExit(f"refusing to overwrite completed run: {output_dir}")
    model_config = load_yaml(Path(config["model_config"]))
    physical_gpu = int(config["physical_gpu"])
    require_idle_gpu(physical_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_id"], revision=model_config.get("revision", "main"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    formatted = []
    max_prompt_tokens = 0
    max_completion_tokens = 0
    for row in rows:
        prompt = format_model_prompt(tokenizer, row["prompt"])
        completion = row["completion"] + tokenizer.eos_token
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        completion_tokens = len(tokenizer(completion, add_special_tokens=False)["input_ids"])
        if prompt_tokens > int(model_config["max_prompt_length"]) or completion_tokens > int(model_config["max_completion_length"]):
            raise SystemExit(f"token budget exceeded: {row['sample_id']} prompt={prompt_tokens} completion={completion_tokens}")
        max_prompt_tokens = max(max_prompt_tokens, prompt_tokens)
        max_completion_tokens = max(max_completion_tokens, completion_tokens)
        formatted.append({"prompt": prompt, "completion": completion})
    dataset = Dataset.from_list(formatted)
    lora = config["lora"]
    parent_adapter = config.get("parent_adapter")
    if parent_adapter:
        base = AutoModelForCausalLM.from_pretrained(model_config["model_id"], revision=model_config.get("revision", "main"), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        base.config.use_cache = False
        model = PeftModel.from_pretrained(base, parent_adapter, is_trainable=True)
        peft_config = None
    else:
        model = model_config["model_id"]
        peft_config = LoraConfig(r=int(lora["r"]), lora_alpha=int(lora["alpha"]), lora_dropout=float(lora["dropout"]), target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM")
    trainer_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=float(config["num_train_epochs"]),
        max_steps=args.max_steps if args.max_steps is not None else -1,
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        bf16=bool(config["bf16"]),
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        max_length=int(config["max_length"]),
        completion_only_loss=True,
        save_strategy="no",
        logging_steps=1,
        report_to=[],
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        model_init_kwargs=None if parent_adapter else {"revision": model_config.get("revision", "main"), "torch_dtype": torch.bfloat16, "use_cache": False},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {**config, "config_path": str(args.config), "git_commit": git_commit(), "rows": len(rows), "max_prompt_tokens_observed": max_prompt_tokens, "max_completion_tokens_observed": max_completion_tokens}
    (output_dir / "run_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    monitor = GPUMemoryMonitor(physical_gpu).start()
    try:
        trainer = SFTTrainer(model=model, args=trainer_args, train_dataset=dataset, processing_class=tokenizer, peft_config=peft_config)
        result = trainer.train()
        trainer.save_model(str(output_dir / "adapter"))
    finally:
        peak_bytes = monitor.stop()
    metrics = {**result.metrics, "train_rows": len(rows), "peak_vram_mb": peak_bytes / (1024 * 1024), "max_prompt_tokens": max_prompt_tokens, "max_completion_tokens": max_completion_tokens}
    write_json(output_dir / "train_metrics.json", metrics)
    write_jsonl(output_dir / "training_log.jsonl", trainer.state.log_history)
    versions = package_versions(["torch", "transformers", "trl", "peft", "datasets", "vllm"])
    lines = [f"platform={platform.platform()}", f"gpu=NVIDIA RTX 4090 physical:{physical_gpu}", *[f"{key}={value}" for key, value in versions.items()], f"peak_vram_mb={metrics['peak_vram_mb']:.2f}"]
    (output_dir / "system_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), **metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
