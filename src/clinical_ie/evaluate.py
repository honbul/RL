from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from src.clinical_ie.common import iter_jsonl, sentence_ids, write_jsonl
from src.clinical_ie.rewards import score_completion
from src.common import configure_compiler, format_model_prompt, git_commit, load_yaml, write_json
from src.evaluate import _percentile
from src.gpu import GPUMemoryMonitor, require_idle_gpu

SPLITS = {
    "dev_2k": ["data/clinical_ie/dev/dev_2k.jsonl"],
    "test_iid_hard": ["data/clinical_ie/test/test_iid_hard.jsonl"],
    "test_linguistic_ood": ["data/clinical_ie/test/test_linguistic_ood.jsonl"],
    "test_compositional_ood": ["data/clinical_ie/test/test_compositional_ood.jsonl"],
    "test_long_context_ood": ["data/clinical_ie/test/test_long_context_ood.jsonl"],
    "test_counterfactual_ood": ["data/clinical_ie/test/test_counterfactual_ood.jsonl"],
}


def aggregate_rows(rows: list[dict[str, Any]], peak_vram_mb: float, wall_seconds: float) -> dict[str, Any]:
    count = len(rows)
    component_keys = ["graph_f1", "condition_object_f1", "condition_attribute_f1", "encounter_object_f1", "relation_f1", "evidence_f1", "unsupported_fact_rate", "missing_fact_rate", "record_exact"]
    result = {key: statistics.fmean(row["reward_components"].get(key, 0.0) for row in rows) for key in component_keys}
    result.update({
        "samples": count,
        "json_parse_rate": sum(row["failure_type"] != "json_parse_error" for row in rows) / count,
        "strict_schema_conformance_rate": sum(row["parsed_output"] is not None for row in rows) / count,
        "assertion_accuracy": result["condition_attribute_f1"],
        "event_time_f1": result["condition_attribute_f1"],
        "patient_family_attribution_error_rate": sum(row["failure_type"] == "unsupported_condition" for row in rows) / count,
        "counterfactual_consistency": statistics.fmean(row.get("counterfactual_consistency", 0.0) for row in rows if row.get("counterfactual_pair_id")) if any(row.get("counterfactual_pair_id") for row in rows) else None,
        "avg_completion_tokens": statistics.fmean(row["completion_tokens"] for row in rows),
        "p50_latency_ms": _percentile([row["latency_ms"] for row in rows], 0.5),
        "p95_latency_ms": _percentile([row["latency_ms"] for row in rows], 0.95),
        "generation_wall_seconds": wall_seconds,
        "peak_vram_mb": peak_vram_mb,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base or adapter clinical IE extraction")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--all-splits", action="store_true")
    parser.add_argument("--split", choices=sorted(SPLITS))
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.all_splits and args.split is None:
        parser.error("one of --all-splits or --split is required")
    selected = list(SPLITS) if args.all_splits else [args.split]
    if args.dry_run:
        rows = list(iter_jsonl(SPLITS[selected[0]]))[: args.limit or 2]
        raw = []
        for row in rows:
            completion = json.dumps(row["gold_output"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            scored = score_completion(completion, row["gold_output"], sentence_ids(row), row["acceptable_evidence_groups"])
            raw.append({"sample_id": row["sample_id"], "completion": completion, "parsed_output": scored.parsed_output, "reward_components": scored.components, "failure_type": scored.failure_type, "completion_tokens": 0, "latency_ms": 0.0, "difficulty_tags": row["difficulty_tags"], "counterfactual_pair_id": row["counterfactual_pair_id"]})
        args.run_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.run_dir / "raw_generations.jsonl", raw)
        metrics = aggregate_rows(raw, 0.0, 0.0)
        write_json(args.run_dir / "eval_metrics.json", metrics)
        result = {"status": "PASS", "mode": "CPU_DRY_RUN", "rows": len(raw), "graph_f1": metrics["graph_f1"], "model_loaded": False, "gpu_initialized": False}
        write_json(args.run_dir / "cpu_contract.json", result)
        print(json.dumps(result, sort_keys=True))
        return

    config_path = args.config or (args.run_dir / "run_config.yaml")
    config = load_yaml(config_path)
    model_config = load_yaml(Path(config["model_config"]))
    physical_gpu = args.physical_gpu if args.physical_gpu is not None else int(config.get("physical_gpu", 1))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        os.execvpe(sys.executable, [sys.executable, "-m", "src.clinical_ie.evaluate", *sys.argv[1:]], env)
    require_idle_gpu(physical_gpu)
    configure_compiler()
    adapter_path = args.adapter_path or (Path(config["adapter_path"]) if config.get("adapter_path") else (args.run_dir / "adapter" if (args.run_dir / "adapter").exists() else None))
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(model_config["model_id"], revision=model_config.get("revision", "main"))
    monitor = GPUMemoryMonitor(physical_gpu).start()
    started = time.perf_counter()
    try:
        llm = LLM(model=model_config["model_id"], revision=model_config.get("revision", "main"), dtype="bfloat16", max_model_len=int(model_config["max_prompt_length"]) + int(model_config["max_completion_length"]), gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.85)), enable_lora=adapter_path is not None, max_lora_rank=16, seed=int(config.get("seed", 42)), enforce_eager=True)
        lora_request = LoRARequest("clinical_adapter", 1, str(adapter_path.resolve())) if adapter_path else None
        sampling = SamplingParams(temperature=0.0, max_tokens=int(model_config["max_completion_length"]), seed=int(config.get("seed", 42)))
        aggregate = {}
        current_commit = git_commit()
        for split in selected:
            data_rows = list(iter_jsonl(SPLITS[split]))
            if args.limit is not None:
                data_rows = data_rows[: args.limit]
            prompts = [format_model_prompt(tokenizer, row["prompt"]) for row in data_rows]
            split_started = time.perf_counter()
            outputs = llm.generate(prompts, sampling, lora_request=lora_request)
            wall = time.perf_counter() - split_started
            raw = []
            failures: Counter[str] = Counter()
            difficulty: dict[str, list[float]] = defaultdict(list)
            for row, request_output in zip(data_rows, outputs, strict=True):
                generated = request_output.outputs[0]
                scored = score_completion(generated.text, row["gold_output"], sentence_ids(row), row["acceptable_evidence_groups"])
                failure = scored.failure_type
                if generated.finish_reason == "length" and scored.components.get("record_exact") != 1.0:
                    failure = "truncated_output"
                if failure:
                    failures[failure] += 1
                metrics = getattr(request_output, "metrics", None)
                latency = (metrics.finished_time - metrics.arrival_time) * 1000 if metrics and metrics.finished_time and metrics.arrival_time else wall * 1000 / len(data_rows)
                item = {"run_id": config["run_id"], "model_id": model_config["model_id"], "sample_id": row["sample_id"], "split": split, "prompt": prompts[len(raw)], "completion": generated.text, "parsed_output": scored.parsed_output, "reward": scored.reward, "reward_components": scored.components, "failure_type": failure, "prompt_tokens": len(request_output.prompt_token_ids), "completion_tokens": len(generated.token_ids), "latency_ms": latency, "difficulty_tags": row["difficulty_tags"], "counterfactual_pair_id": row["counterfactual_pair_id"], "seed": config.get("seed", 42), "git_commit": current_commit}
                raw.append(item)
                for tag in row["difficulty_tags"]:
                    difficulty[tag].append(scored.components.get("graph_f1", 0.0))
            split_dir = args.run_dir / split
            write_jsonl(split_dir / "raw_generations.jsonl", raw)
            write_jsonl(split_dir / "failures.jsonl", (row for row in raw if row["failure_type"]))
            write_json(split_dir / "failure_summary.json", dict(sorted(failures.items())))
            write_json(split_dir / "difficulty_summary.json", {tag: {"samples": len(values), "graph_f1": statistics.fmean(values)} for tag, values in sorted(difficulty.items())})
            aggregate[split] = aggregate_rows(raw, 0.0, wall)
        peak = monitor.stop() / (1024 * 1024)
    except BaseException:
        monitor.stop(); raise
    for split, metrics in aggregate.items():
        metrics["peak_vram_mb"] = peak
        write_json(args.run_dir / split / "eval_metrics.json", metrics)
    write_json(args.run_dir / "eval_metrics.json", aggregate)
    (args.run_dir / "system_info_eval.txt").write_text(f"gpu=NVIDIA RTX 4090 physical:{physical_gpu}\npeak_vram_mb={peak:.2f}\nwall_seconds={time.perf_counter()-started:.3f}\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "run_dir": str(args.run_dir), "splits": aggregate}, sort_keys=True))


if __name__ == "__main__":
    main()
