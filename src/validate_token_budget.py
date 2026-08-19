from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from src.common import format_model_prompt, load_yaml


FILES = [
    Path("data/sft/sft_6k.jsonl"),
    Path("data/grpo/prompts.jsonl"),
    Path("data/dev/dev.jsonl"),
    Path("data/test/test_iid.jsonl"),
    Path("data/test/test_linguistic_ood.jsonl"),
    Path("data/test/test_compositional_ood.jsonl"),
    Path("data/test/test_structural_ood.jsonl"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate tokenizer-exact prompt and completion budgets")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    args = parser.parse_args()
    config = load_yaml(args.model_config)
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config.get("revision", "main"))
    max_prompt = 0
    max_completion = 0
    prompt_row = None
    completion_row = None
    total = 0
    for path in FILES:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = format_model_prompt(tokenizer, row["prompt"])
                prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                if prompt_tokens > max_prompt:
                    max_prompt = prompt_tokens
                    prompt_row = row["sample_id"]
                if "completion" in row:
                    completion_tokens = len(tokenizer(row["completion"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"])
                    if completion_tokens > max_completion:
                        max_completion = completion_tokens
                        completion_row = row["sample_id"]
                total += 1
    status = "PASS" if max_prompt <= int(config["max_prompt_length"]) and max_completion <= int(config["max_completion_length"]) else "FAIL"
    result = {
        "status": status,
        "rows": total,
        "max_prompt_tokens": max_prompt,
        "max_prompt_sample_id": prompt_row,
        "max_completion_tokens": max_completion,
        "max_completion_sample_id": completion_row,
        "prompt_limit": int(config["max_prompt_length"]),
        "completion_limit": int(config["max_completion_length"]),
    }
    print(json.dumps(result, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
