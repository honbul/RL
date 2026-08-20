from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.clinical_ie.data.latent_timeline import generate_latent
from src.clinical_ie.data.render_documents import canonical_json, render_sample
from src.clinical_ie.schema import ExtractionOutput, validate_evidence_ids

FORBIDDEN_PROMPT_TERMS = ("gold_output", "latent_timeline", "acceptable_evidence_groups")


def validate_generated_row(row: dict[str, Any]) -> None:
    parsed = ExtractionOutput.model_validate(row["gold_output"])
    sentence_ids = {
        sentence["sentence_id"]
        for document in row["documents"]
        for sentence in document["sentences"]
    }
    validate_evidence_ids(parsed, sentence_ids)
    if len(sentence_ids) != len({sentence["sentence_id"] for document in row["documents"] for sentence in document["sentences"]}):
        raise ValueError("duplicate sentence IDs")
    if any(term in row["prompt"] for term in FORBIDDEN_PROMPT_TERMS):
        raise ValueError("private authority leaked into prompt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one deterministic clinical IE candidate shard")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.start < args.end <= 40000:
        raise SystemExit("expected 0 <= start < end <= 40000")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for slot in range(args.start, args.end):
            row = render_sample(generate_latent(slot, args.seed))
            validate_generated_row(row)
            handle.write(canonical_json(row) + "\n")
            regime = row["latent_timeline"]["regime"]
            counts[regime] = counts.get(regime, 0) + 1
    summary = {
        "status": "PASS",
        "start": args.start,
        "end": args.end,
        "rows": args.end - args.start,
        "regime_counts": dict(sorted(counts.items())),
        "output": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
