from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import yaml

from src.common import (
    config_parser,
    format_model_prompt,
    git_commit,
    load_yaml,
    package_versions,
    write_json,
)
from src.gpu import GPUMemoryMonitor, require_idle_gpu


def main() -> None:
    parser = config_parser("Train a LoRA adapter with supervised DSL examples")
    parser.add_argument("--max-steps", type=int, default=None, help="Bounded smoke override")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N training rows")
    parser.add_argument("--output-dir", type=Path, default=None, help="Unique output-root override")
    args = parser.parse_args()
    config = load_yaml(args.config)
    model_config = load_yaml(Path(config["model_config"]))
    output_dir = args.output_dir or Path(config["output_dir"])
    if (output_dir / "train_metrics.json").exists():
        raise SystemExit(f"refusing to overwrite completed run: {output_dir}")
    physical_gpu = int(config.get("physical_gpu", 0))
    require_idle_gpu(physical_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_id"], revision=model_config.get("revision", "main")
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = []
    with Path(config["data_path"]).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if args.limit is not None and len(rows) >= args.limit:
                    break
    max_prompt_tokens = 0
    max_completion_tokens = 0
    formatted = []
    for row in rows:
        prompt = format_model_prompt(tokenizer, row["prompt"])
        completion = row["completion"] + tokenizer.eos_token
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        completion_tokens = len(tokenizer(completion, add_special_tokens=False)["input_ids"])
        max_prompt_tokens = max(max_prompt_tokens, prompt_tokens)
        max_completion_tokens = max(max_completion_tokens, completion_tokens)
        if prompt_tokens > int(model_config["max_prompt_length"]):
            raise SystemExit(f"prompt token budget exceeded: {row['sample_id']}={prompt_tokens}")
        if completion_tokens > int(model_config["max_completion_length"]):
            raise SystemExit(f"completion token budget exceeded: {row['sample_id']}={completion_tokens}")
        formatted.append({"prompt": prompt, "completion": completion})
    dataset = Dataset.from_list(formatted)
    lora = config["lora"]
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    trainer_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        overwrite_output_dir=False,
        num_train_epochs=float(config["num_train_epochs"]),
        max_steps=args.max_steps if args.max_steps is not None else -1,
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        bf16=bool(config["bf16"]),
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        max_length=int(config["max_length"]),
        completion_only_loss=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        model_init_kwargs={
            "revision": model_config.get("revision", "main"),
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
            "use_cache": False,
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        **config,
        "config_path": str(args.config),
        "output_dir": str(output_dir),
        "limit": args.limit,
        "max_steps_override": args.max_steps,
        "git_commit": git_commit(),
        "rows": len(rows),
        "max_prompt_tokens_observed": max_prompt_tokens,
        "max_completion_tokens_observed": max_completion_tokens,
    }
    (output_dir / "run_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8"
    )
    monitor = GPUMemoryMonitor(physical_gpu).start()
    try:
        trainer = SFTTrainer(
            model=model_config["model_id"],
            args=trainer_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        result = trainer.train()
        trainer.save_model(str(output_dir / "adapter"))
    finally:
        peak_bytes = monitor.stop()
    metrics = dict(result.metrics)
    metrics.update(
        {
            "peak_vram_mb": peak_bytes / (1024 * 1024),
            "train_rows": len(rows),
            "max_prompt_tokens": max_prompt_tokens,
            "max_completion_tokens": max_completion_tokens,
        }
    )
    write_json(output_dir / "train_metrics.json", metrics)
    with (output_dir / "training_log.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in trainer.state.log_history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    versions = package_versions(["torch", "transformers", "trl", "peft", "datasets", "vllm"])
    system_lines = [f"platform={platform.platform()}", f"gpu=NVIDIA RTX 4090 physical:{physical_gpu}"]
    system_lines.extend(f"{key}={value}" for key, value in versions.items())
    system_lines.append(f"peak_vram_mb={metrics['peak_vram_mb']:.2f}")
    (output_dir / "system_info.txt").write_text("\n".join(system_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), **metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
