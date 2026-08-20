from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

RUNS = {
    "B0": "b0_base",
    "S1": "s1_sft_8k",
    "S2": "s2_sft_24k",
    "S2X": "s2x_sft_32k",
    "S3": "s3_matched_hard_sft",
    "R1": "r1_grpo",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize clinical IE benchmark metrics")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/clinical_ie"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports/clinical_ie"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.reports_root.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.dry_run:
        for index, run in enumerate(("B0", "S1", "S2", "S3", "R1")):
            rows.append({"run": run, "dev_graph_f1": index / 10, "iid_graph_f1": index / 10, "ood_graph_f1": index / 20, "evidence_f1": index / 10, "unsupported_fact_rate": 0.1, "status": "DUMMY"})
    else:
        for run, directory in RUNS.items():
            path = args.outputs_root / directory / "eval_metrics.json"
            if not path.exists():
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "run": run,
                "dev_graph_f1": metrics.get("dev_2k", {}).get("graph_f1"),
                "iid_graph_f1": metrics.get("test_iid_hard", {}).get("graph_f1"),
                "ood_graph_f1": sum(metrics.get(split, {}).get("graph_f1", 0.0) for split in ("test_linguistic_ood", "test_compositional_ood", "test_long_context_ood", "test_counterfactual_ood")) / 4,
                "evidence_f1": metrics.get("test_iid_hard", {}).get("evidence_f1"),
                "unsupported_fact_rate": metrics.get("test_iid_hard", {}).get("unsupported_fact_rate"),
                "status": "PASS",
            })
    if not rows:
        raise SystemExit("no metric rows available")
    summary = args.reports_root / "benchmark_summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    report = ["# Clinical IE RLVR Summary", "", "| Run | Dev Graph F1 | IID Graph F1 | OOD Graph F1 | Evidence F1 | Unsupported fact rate |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        report.append(f"| {row['run']} | {row['dev_graph_f1']} | {row['iid_graph_f1']} | {row['ood_graph_f1']} | {row['evidence_f1']} | {row['unsupported_fact_rate']} |")
    (args.reports_root / "summary_preview.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "mode": "CPU_DRY_RUN" if args.dry_run else "FORMAL", "rows": len(rows), "summary": str(summary)}, sort_keys=True))


if __name__ == "__main__":
    main()
