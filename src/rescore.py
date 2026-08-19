from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from src.common import write_json
from src.dsl.canonicalize import canonical_equal, flatten_leaves
from src.evaluate import SPLITS, _percentile


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute evaluation metrics from persisted raw generations")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    aggregate = {}
    for split, data_path in SPLITS.items():
        raw_path = args.run_dir / split / "raw_generations.jsonl"
        if not raw_path.exists():
            continue
        raw_rows = _load_jsonl(raw_path)
        dataset = {row["sample_id"]: row for row in _load_jsonl(data_path)}
        json_valid = 0
        schema_valid = 0
        executable = 0
        hidden_passed = 0
        hidden_total = 0
        target_schema_passed = 0
        problem_exact = 0
        field_macro_total = 0.0
        field_true_positive = 0
        field_predicted = 0
        field_expected = 0
        edge_passed = 0
        edge_total = 0
        completion_tokens = []
        latencies = []
        failures: Counter[str] = Counter()
        for raw in raw_rows:
            source = dataset[raw["sample_id"]]
            try:
                json_valid += int(isinstance(json.loads(raw["completion"]), dict))
            except (json.JSONDecodeError, TypeError):
                pass
            schema_valid += int(raw["parsed_program"] is not None)
            cases = raw["hidden_case_results"]
            exact = bool(cases) and len(cases) == len(source["hidden_tests"]) and all(case.get("passed", False) for case in cases)
            problem_exact += int(exact)
            hidden_passed += sum(bool(case.get("passed")) for case in cases)
            target_schema_passed += sum(bool(case.get("schema_pass")) for case in cases)
            hidden_total += len(source["hidden_tests"])
            executable += int(bool(cases) and len(cases) == len(source["hidden_tests"]) and all("error" not in case for case in cases))
            field_macro_total += float(raw["metrics"]["field_f1"])
            field_expected += sum(len(flatten_leaves(case["expected"])) for case in source["hidden_tests"])
            for result, hidden in zip(cases, source["hidden_tests"]):
                if "predicted" not in result:
                    continue
                predicted = flatten_leaves(result["predicted"])
                expected = flatten_leaves(hidden["expected"])
                field_predicted += len(predicted)
                field_true_positive += sum(
                    1 for path, value in predicted.items()
                    if path in expected and canonical_equal(value, expected[path])
                )
            is_edge = any(bool(source["difficulty"][key]) for key in ("has_null", "has_array", "has_condition"))
            edge_total += int(is_edge)
            edge_passed += int(is_edge and exact)
            completion_tokens.append(int(raw["completion_tokens"]))
            latencies.append(float(raw["latency_ms"]))
            if raw["failure_type"]:
                failures[raw["failure_type"]] += 1
        count = len(raw_rows)
        field_precision = field_true_positive / field_predicted if field_predicted else 0.0
        field_recall = field_true_positive / field_expected if field_expected else 0.0
        field_micro = 2 * field_precision * field_recall / (field_precision + field_recall) if field_precision + field_recall else 0.0
        old_metrics_path = args.run_dir / split / "eval_metrics.json"
        old = json.loads(old_metrics_path.read_text(encoding="utf-8"))
        metrics = {
            "split": split,
            "samples": count,
            "json_parse_rate": json_valid / count,
            "schema_conformance_rate": schema_valid / count,
            "executable_rate": executable / count,
            "target_schema_pass_rate": target_schema_passed / hidden_total,
            "hidden_case_pass_rate": hidden_passed / hidden_total,
            "problem_exact_pass_rate": problem_exact / count,
            "field_micro_f1": field_micro,
            "field_macro_f1": field_macro_total / count,
            "edge_case_pass_rate": edge_passed / edge_total if edge_total else 0.0,
            "avg_completion_tokens": statistics.fmean(completion_tokens),
            "p50_latency_ms": _percentile(latencies, 0.50),
            "p95_latency_ms": _percentile(latencies, 0.95),
            "generation_wall_seconds": old["generation_wall_seconds"],
            "peak_vram_mb": old["peak_vram_mb"],
            "rescored_from_raw": True,
        }
        write_json(old_metrics_path, metrics)
        write_json(args.run_dir / split / "failure_summary.json", dict(sorted(failures.items())))
        aggregate[split] = metrics
    write_json(args.run_dir / "eval_metrics.json", aggregate)
    print(json.dumps({"status": "PASS", "run_dir": str(args.run_dir), "splits": aggregate}, sort_keys=True))


if __name__ == "__main__":
    main()
