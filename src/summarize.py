from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

RUNS = {
    "E0": Path("outputs/e0_zero_shot"),
    "E1": Path("outputs/e1_sft_1k"),
    "E2": Path("outputs/e2_sft_6k"),
    "E3": Path("outputs/e3_grpo_binary"),
    "E4": Path("outputs/e4_grpo_dense"),
}
SPLITS = [
    "test_iid",
    "test_linguistic_ood",
    "test_compositional_ood",
    "test_structural_ood",
]
LABELS = {
    "test_iid": "IID",
    "test_linguistic_ood": "Linguistic OOD",
    "test_compositional_ood": "Compositional OOD",
    "test_structural_ood": "Structural OOD",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted(metrics: dict[str, dict[str, Any]], key: str) -> float:
    total = sum(metrics[split]["samples"] for split in SPLITS)
    return sum(metrics[split][key] * metrics[split]["samples"] for split in SPLITS) / total


def peak_vram(root: Path, metrics: dict[str, dict[str, Any]]) -> float:
    values = [float(metrics[split]["peak_vram_mb"]) for split in SPLITS]
    train_path = root / "train_metrics.json"
    if train_path.exists():
        train = load_json(train_path)
        for key in ("peak_vram_mb", "policy_peak_vram_mb", "rollout_peak_vram_mb"):
            if key in train:
                values.append(float(train[key]))
    return max(values)


def failure_counts(root: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for split in SPLITS:
        counts.update(load_json(root / split / "failure_summary.json"))
    return counts


def select_examples(root: Path, split: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    success = None
    failure = None
    with (root / split / "raw_generations.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            entry = {
                "sample_id": row["sample_id"],
                "completion": row["completion"],
                "failure_type": row["failure_type"],
                "path": f"{root / split / 'raw_generations.jsonl'}:{line_number}",
            }
            if row["metrics"]["problem_exact"] and success is None:
                success = entry
            if not row["metrics"]["problem_exact"] and failure is None:
                failure = entry
            if success is not None and failure is not None:
                break
    return success, failure


def excerpt(text: str, limit: int = 500) -> str:
    compact = text.replace("\n", "\\n")
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def main() -> None:
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {run: load_json(root / "eval_metrics.json") for run, root in RUNS.items()}

    summary_rows = []
    for run, root in RUNS.items():
        metrics = all_metrics[run]
        summary_rows.append({
            "run": run,
            "iid_exact": metrics["test_iid"]["problem_exact_pass_rate"],
            "linguistic_ood_exact": metrics["test_linguistic_ood"]["problem_exact_pass_rate"],
            "compositional_ood_exact": metrics["test_compositional_ood"]["problem_exact_pass_rate"],
            "structural_ood_exact": metrics["test_structural_ood"]["problem_exact_pass_rate"],
            "overall_hidden_case_pass": weighted(metrics, "hidden_case_pass_rate"),
            "overall_field_macro_f1": weighted(metrics, "field_macro_f1"),
            "overall_json_parse_rate": weighted(metrics, "json_parse_rate"),
            "avg_completion_tokens": weighted(metrics, "avg_completion_tokens"),
            "peak_vram_mb": peak_vram(root, metrics),
        })
    with (report_dir / "benchmark_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    failure_rows = []
    for run, root in RUNS.items():
        for split in SPLITS:
            counts = load_json(root / split / "failure_summary.json")
            samples = all_metrics[run][split]["samples"]
            for failure_type, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                failure_rows.append({
                    "run": run,
                    "split": split,
                    "failure_type": failure_type,
                    "count": count,
                    "sample_rate": count / samples,
                })
    with (report_dir / "failure_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run", "split", "failure_type", "count", "sample_rate"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(failure_rows)

    manifest = load_json(Path("data/manifests/data_manifest.json"))
    table_lines = [
        "| Run | IID exact | Linguistic OOD exact | Compositional OOD exact | Structural OOD exact | Hidden case pass | Field macro F1 | JSON valid | Avg tokens | Peak VRAM MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        table_lines.append(
            f"| {row['run']} | {row['iid_exact']:.3f} | {row['linguistic_ood_exact']:.3f} | "
            f"{row['compositional_ood_exact']:.3f} | {row['structural_ood_exact']:.3f} | "
            f"{row['overall_hidden_case_pass']:.3f} | {row['overall_field_macro_f1']:.3f} | "
            f"{row['overall_json_parse_rate']:.3f} | {row['avg_completion_tokens']:.1f} | {row['peak_vram_mb']:.1f} |"
        )

    failure_table = ["| Run | Top failure types across all splits |", "|---|---|"]
    for run, root in RUNS.items():
        counts = failure_counts(root)
        top = ", ".join(f"`{name}` {count}" for name, count in counts.most_common(4)) or "none"
        failure_table.append(f"| {run} | {top} |")

    example_lines = []
    for run in ("E0", "E2", "E3", "E4"):
        success, failure = select_examples(RUNS[run], "test_compositional_ood")
        if success is not None:
            example_lines.extend([
                f"- **{run} success** `{success['sample_id']}` — `{success['path']}`",
                f"  - completion: `{excerpt(success['completion'])}`",
            ])
        if failure is not None:
            example_lines.extend([
                f"- **{run} failure** `{failure['sample_id']}` (`{failure['failure_type']}`) — `{failure['path']}`",
                f"  - completion: `{excerpt(failure['completion'])}`",
            ])

    report = f"""# Synthetic Schema RLVR 최종 보고서

## 1. 실험 목적

4B instruct 모델이 source/target JSON schema와 자연어 요구사항으로부터 실행 가능한 제한형 JSON DSL을 생성하도록 학습했다. 핵심 질문은 **SFT-1K 이후 verifier 기반 GRPO가 SFT-6K보다 Compositional-OOD Problem Pass@1을 높이는가**이다.

## 2. 데이터 생성 방식과 수량

정답 transformation graph → source/target schema → canonical reference DSL → hidden input → expected output → instruction 순서로 파생했다. 4개 shard worker가 후보 20,000건을 생성했고 reference interpreter 및 schema 검증을 통과한 데이터만 사용했다.

- SFT-1K: {manifest['counts']['sft_1k']:,} (SFT-6K의 prefix subset)
- SFT-6K: {manifest['counts']['sft_6k']:,}
- GRPO train: {manifest['counts']['grpo_prompts']:,}, prompt당 hidden case {manifest['hidden_cases']['grpo']}
- Development: {manifest['counts']['dev']:,}
- Test: IID {manifest['counts']['test_iid']:,}, Linguistic OOD {manifest['counts']['test_linguistic_ood']:,}, Compositional OOD {manifest['counts']['test_compositional_ood']:,}, Structural OOD {manifest['counts']['test_structural_ood']:,}; prompt당 hidden case {manifest['hidden_cases']['evaluation']}
- 누출 검사: sample ID, graph fingerprint, compositional signature, linguistic template overlap 모두 0

## 3. 모델 및 학습 설정

- 모델: `Qwen/Qwen3-4B-Instruct-2507`, seed 42
- SFT: BF16 LoRA r=16, alpha=32, dropout=0.05, LR 2e-5, 1 epoch
- GRPO: E1 adapter 시작, group size 4, temperature 0.9, top-p 0.95, LR 5e-6, beta 0, DAPO loss, 1 epoch
- GRPO 실행 batch: 8 completions = 2 prompt groups/step, 2,000 optimizer steps, GPU 0 policy / GPU 1 vLLM
- 평가: temperature 0, max completion 256, 동일 evaluator와 동일 4개 split

## 4. E0–E4 결과

{chr(10).join(table_lines)}

주 지표인 Compositional OOD exact는 E2 0.568, E3 0.084, E4 0.074였다. E3/E4가 E2를 넘지 않았으므로 preregistered 조건에 따라 seed 43/44 반복은 실행하지 않았다.

## 5. IID와 OOD 비교

- E2는 IID와 Linguistic OOD에서 1.000을 달성했지만 Compositional OOD는 0.568, Structural OOD는 0이었다.
- E3/E4는 JSON parse rate를 대체로 높게 유지했지만 IID, Linguistic OOD, Compositional OOD 실행 정확도가 E2보다 크게 낮았다.
- 모든 조건이 Structural OOD 0이므로 더 긴 프로그램, 깊은 nesting, 복합 array edge case 일반화는 해결되지 않았다.

## 6. Binary와 Dense reward 비교

Binary E3가 Dense E4보다 Compositional OOD에서 0.084 대 0.074, IID에서 0.461 대 0.449로 소폭 높았다. Dense E4는 rollout reward가 거의 항상 nonzero였지만(15,806/16,000), 이것이 problem-level exact 개선으로 이어지지 않았다. Binary E3도 E2보다 낮아 reward 형태 선택보다 SFT-1K 시작 policy와 on-policy update에 따른 성능 저하가 더 큰 결과였다.

## 7. 주요 raw output 사례

{chr(10).join(example_lines)}

## 8. 실패 유형

{chr(10).join(failure_table)}

세부 run/split/failure count는 `reports/failure_summary.csv`에 저장했다. E0는 target JSON을 직접 출력하는 DSL schema 오류가 지배적이었다. E3/E4는 parse 가능한 출력 비율은 높았지만 invalid path/schema 또는 partial mismatch가 남아 실행 exact가 낮았다.

## 9. 결론

### **GRPO 이점 없음**

이 실험에서는 추가 SFT 데이터가 verifier 기반 GRPO보다 효과적이었다. E2의 Compositional-OOD Problem Pass@1 0.568에 비해 E3는 0.084, E4는 0.074였고 hidden execution 성능도 같은 방향이었다. 따라서 개선이 단순 형식 준수에 그치지 않았는지를 묻기 전에, GRPO가 강한 SFT-6K 기준선의 실행 정확도를 보존하지 못했다.

인과 해석은 단일 4B 모델, 단일 seed 42, 고정 LoRA/GRPO 설정, 합성 DSL benchmark에 한정된다. 추가 seed는 사전 정의된 개선 조건이 충족되지 않아 생략했다. 실제 데이터·임상 유효성·production readiness에 대한 주장은 하지 않는다.

## 10. 재현 명령

```bash
.venv/bin/python -m src.data.generate --config configs/data.yaml
.venv/bin/python -m src.evaluate --run-dir outputs/e0_zero_shot --config configs/e0_zero_shot.yaml --all-splits
PYTHONHASHSEED=42 .venv/bin/python -m src.train_sft --config configs/sft_1k.yaml
.venv/bin/python -m src.evaluate --run-dir outputs/e1_sft_1k --all-splits --physical-gpu 1
PYTHONHASHSEED=42 .venv/bin/python -m src.train_sft --config configs/sft_6k.yaml
.venv/bin/python -m src.evaluate --run-dir outputs/e2_sft_6k --all-splits --physical-gpu 1
PYTHONHASHSEED=42 .venv/bin/python -m src.train_grpo --config configs/grpo_binary.yaml
.venv/bin/python -m src.evaluate --run-dir outputs/e3_grpo_binary --all-splits --physical-gpu 1
PYTHONHASHSEED=42 .venv/bin/python -m src.train_grpo --config configs/grpo_dense.yaml
.venv/bin/python -m src.evaluate --run-dir outputs/e4_grpo_dense --all-splits --physical-gpu 1
.venv/bin/python -m src.rescore --run-dir outputs/e4_grpo_dense
```

환경의 정확한 Python 패키지는 `requirements-lock.txt`, 실제 hardware/runtime은 `environment.txt`에 기록했다.
"""
    (report_dir / "final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "PASS", "summary_rows": len(summary_rows), "failure_rows": len(failure_rows), "report": str(report_dir / 'final_report.md')}, sort_keys=True))


if __name__ == "__main__":
    main()
