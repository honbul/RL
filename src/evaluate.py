from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from src.common import (
    configure_compiler,
    format_model_prompt,
    git_commit,
    load_yaml,
    package_versions,
    write_json,
)
from src.dsl.canonicalize import canonical_equal, flatten_leaves
from src.gpu import GPUMemoryMonitor, require_idle_gpu
from src.rewards import score_completion

SPLITS = {
    "test_iid": Path("data/test/test_iid.jsonl"),
    "test_linguistic_ood": Path("data/test/test_linguistic_ood.jsonl"),
    "test_compositional_ood": Path("data/test/test_compositional_ood.jsonl"),
    "test_structural_ood": Path("data/test/test_structural_ood.jsonl"),
}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return float(ordered[index])


def _load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _write_raw(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a base model or adapter on benchmark splits")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--all-splits", action="store_true")
    parser.add_argument("--split", choices=sorted(SPLITS))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--limit", type=int, default=None, help="Bounded smoke override")
    args = parser.parse_args()
    if not args.all_splits and args.split is None:
        parser.error("one of --all-splits or --split is required")
    config_path = args.config or (args.run_dir / "run_config.yaml")
    if not config_path.exists():
        raise SystemExit(f"missing evaluation/run config: {config_path}")
    config = load_yaml(config_path)
    model_config = load_yaml(Path(config["model_config"]))
    physical_gpu = args.physical_gpu if args.physical_gpu is not None else int(config.get("physical_gpu", 1))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        os.execvpe(
            sys.executable,
            [sys.executable, "-m", "src.evaluate", *sys.argv[1:]],
            environment,
        )
    require_idle_gpu(physical_gpu)
    configure_compiler()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    adapter_path = args.adapter_path
    if adapter_path is None:
        configured_adapter = config.get("adapter_path")
        if configured_adapter:
            adapter_path = Path(configured_adapter)
        elif (args.run_dir / "adapter").exists():
            adapter_path = args.run_dir / "adapter"
    selected_splits = list(SPLITS) if args.all_splits else [args.split]
    for split in selected_splits:
        if (args.run_dir / split / "eval_metrics.json").exists():
            raise SystemExit(f"refusing to overwrite completed evaluation: {args.run_dir / split}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_id"], revision=model_config.get("revision", "main")
    )
    monitor = GPUMemoryMonitor(physical_gpu).start()
    started = time.perf_counter()
    current_commit = git_commit()
    try:
        llm = LLM(
            model=model_config["model_id"],
            revision=model_config.get("revision", "main"),
            dtype="bfloat16",
            max_model_len=int(model_config["max_prompt_length"]) + int(model_config["max_completion_length"]),
            gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.85)),
            trust_remote_code=bool(model_config.get("trust_remote_code", False)),
            enable_lora=adapter_path is not None,
            max_lora_rank=16,
            seed=int(config.get("seed", 42)),
            enforce_eager=True,
        )
        lora_request = None
        if adapter_path is not None:
            lora_request = LoRARequest("evaluation_adapter", 1, str(adapter_path.resolve()))
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=int(config.get("max_completion_length", model_config["max_completion_length"])),
            seed=int(config.get("seed", 42)),
        )
        aggregate = {}
        for split in selected_splits:
            rows = _load_rows(SPLITS[split], args.limit)
            model_prompts = [format_model_prompt(tokenizer, row["prompt"]) for row in rows]
            prompt_lengths = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in model_prompts]
            if max(prompt_lengths, default=0) > int(model_config["max_prompt_length"]):
                bad_index = next(index for index, length in enumerate(prompt_lengths) if length > int(model_config["max_prompt_length"]))
                raise SystemExit(f"prompt token budget exceeded: {rows[bad_index]['sample_id']}={prompt_lengths[bad_index]}")
            generation_started = time.perf_counter()
            outputs = llm.generate(model_prompts, sampling, lora_request=lora_request)
            batch_elapsed = time.perf_counter() - generation_started
            raw_rows = []
            problem_exact = 0
            json_valid = 0
            parse_valid = 0
            executable = 0
            hidden_passed = 0
            hidden_total = 0
            target_schema_passed = 0
            field_f1_total = 0.0
            field_true_positive = 0
            field_predicted = 0
            field_expected = 0
            edge_passed = 0
            edge_total = 0
            completion_tokens = []
            latencies = []
            failures: Counter[str] = Counter()
            for index, (row, request_output) in enumerate(zip(rows, outputs, strict=True)):
                generated = request_output.outputs[0]
                completion = generated.text
                scored = score_completion(
                    completion,
                    row["source_schema"],
                    row["target_schema"],
                    row["hidden_tests"],
                    "dense",
                )
                try:
                    json_valid += int(isinstance(json.loads(completion), dict))
                except (json.JSONDecodeError, TypeError):
                    pass
                exact = bool(scored.hidden_case_results) and all(case.get("passed", False) for case in scored.hidden_case_results)
                schema_pass_count = sum(bool(case.get("schema_pass")) for case in scored.hidden_case_results)
                passed_count = sum(bool(case.get("passed")) for case in scored.hidden_case_results)
                row_f1 = statistics.fmean(case.get("field_f1", 0.0) for case in scored.hidden_case_results) if scored.hidden_case_results else 0.0
                parse_valid += int(scored.parsed_program is not None)
                problem_exact += int(exact)
                hidden_passed += passed_count
                hidden_total += len(row["hidden_tests"])
                target_schema_passed += schema_pass_count
                field_f1_total += row_f1
                field_expected += sum(
                    len(flatten_leaves(hidden_case["expected"]))
                    for hidden_case in row["hidden_tests"]
                )
                for case_result, hidden_case in zip(scored.hidden_case_results, row["hidden_tests"]):
                    if "predicted" not in case_result:
                        continue
                    predicted_leaves = flatten_leaves(case_result["predicted"])
                    expected_leaves = flatten_leaves(hidden_case["expected"])
                    field_predicted += len(predicted_leaves)
                    field_true_positive += sum(
                        1
                        for path, value in predicted_leaves.items()
                        if path in expected_leaves and canonical_equal(value, expected_leaves[path])
                    )
                executable += int(bool(scored.hidden_case_results) and all("error" not in case for case in scored.hidden_case_results))
                is_edge = any(bool(row["difficulty"][key]) for key in ("has_null", "has_array", "has_condition"))
                edge_total += int(is_edge)
                edge_passed += int(is_edge and exact)
                failure_type = scored.failure_type
                if getattr(generated, "finish_reason", None) == "length" and not exact:
                    failure_type = "truncated_output"
                if failure_type:
                    failures[failure_type] += 1
                token_count = len(generated.token_ids)
                completion_tokens.append(token_count)
                metrics = getattr(request_output, "metrics", None)
                if metrics is not None and getattr(metrics, "finished_time", None) is not None and getattr(metrics, "arrival_time", None) is not None:
                    latency_ms = (metrics.finished_time - metrics.arrival_time) * 1000
                else:
                    latency_ms = batch_elapsed * 1000 / max(1, len(rows))
                latencies.append(latency_ms)
                raw_rows.append({
                    "run_id": config["run_id"],
                    "model_id": model_config["model_id"],
                    "sample_id": row["sample_id"],
                    "split": split,
                    "prompt": model_prompts[index],
                    "completion": completion,
                    "parsed_program": scored.parsed_program,
                    "parse_error": scored.parse_error,
                    "reward_components": scored.components,
                    "hidden_case_results": scored.hidden_case_results,
                    "metrics": {"problem_exact": exact, "hidden_case_pass_rate": passed_count / len(row["hidden_tests"]), "target_schema_pass_rate": schema_pass_count / len(row["hidden_tests"]), "field_f1": row_f1},
                    "failure_type": failure_type,
                    "prompt_tokens": len(request_output.prompt_token_ids),
                    "completion_tokens": token_count,
                    "latency_ms": latency_ms,
                    "seed": int(config.get("seed", 42)),
                    "config_path": str(config_path),
                    "git_commit": current_commit,
                })
            count = len(rows)
            field_precision = field_true_positive / field_predicted if field_predicted else 0.0
            field_recall = field_true_positive / field_expected if field_expected else 0.0
            field_micro_f1 = 2 * field_precision * field_recall / (field_precision + field_recall) if field_precision + field_recall else 0.0
            split_metrics = {
                "split": split,
                "samples": count,
                "json_parse_rate": json_valid / count,
                "schema_conformance_rate": parse_valid / count,
                "executable_rate": executable / count,
                "target_schema_pass_rate": target_schema_passed / hidden_total,
                "hidden_case_pass_rate": hidden_passed / hidden_total,
                "problem_exact_pass_rate": problem_exact / count,
                "field_micro_f1": field_micro_f1,
                "field_macro_f1": field_f1_total / count,
                "edge_case_pass_rate": edge_passed / edge_total if edge_total else 0.0,
                "avg_completion_tokens": statistics.fmean(completion_tokens),
                "p50_latency_ms": _percentile(latencies, 0.50),
                "p95_latency_ms": _percentile(latencies, 0.95),
                "generation_wall_seconds": batch_elapsed,
            }
            split_dir = args.run_dir / split
            _write_raw(split_dir / "raw_generations.jsonl", raw_rows)
            _write_raw(split_dir / "failures.jsonl", [row for row in raw_rows if row["failure_type"]])
            write_json(split_dir / "eval_metrics.json", split_metrics)
            write_json(split_dir / "failure_summary.json", dict(sorted(failures.items())))
            aggregate[split] = split_metrics
        peak_bytes = monitor.stop()
    except BaseException:
        monitor.stop()
        raise
    for split_metrics in aggregate.values():
        split_metrics["peak_vram_mb"] = peak_bytes / (1024 * 1024)
        write_json(args.run_dir / split_metrics["split"] / "eval_metrics.json", split_metrics)
    write_json(args.run_dir / "eval_metrics.json", aggregate)
    if not (args.run_dir / "run_config.yaml").exists():
        resolved = {**config, "config_path": str(config_path), "git_commit": git_commit(), "adapter_path": str(adapter_path) if adapter_path else None}
        (args.run_dir / "run_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    versions = package_versions(["torch", "transformers", "trl", "peft", "datasets", "vllm"])
    lines = [f"platform={platform.platform()}", f"gpu=NVIDIA RTX 4090 physical:{physical_gpu}"]
    lines.extend(f"{key}={value}" for key, value in versions.items())
    lines.extend([f"peak_vram_mb={peak_bytes / (1024 * 1024):.2f}", f"wall_seconds={time.perf_counter() - started:.3f}"])
    (args.run_dir / "system_info_eval.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "run_dir": str(args.run_dir), "splits": aggregate}, sort_keys=True))


if __name__ == "__main__":
    main()
