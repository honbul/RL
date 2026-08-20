from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.clinical_ie.schema import ExtractionOutput


def iter_jsonl(paths: Iterable[str | Path]) -> Iterable[dict[str, Any]]:
    for value in paths:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def load_sft_rows(paths: list[str], skip_rows: int, max_rows: int | None) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(iter_jsonl(paths)):
        if index < skip_rows:
            continue
        if "prompt" not in row or "completion" not in row:
            raise ValueError(f"SFT row missing prompt/completion: {row.get('sample_id')}")
        if json.loads(row["completion"]) != row["gold_output"]:
            raise ValueError(f"SFT completion differs from gold: {row['sample_id']}")
        ExtractionOutput.model_validate(row["gold_output"])
        rows.append(row)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def sentence_ids(row: dict[str, Any]) -> set[str]:
    return {
        sentence["sentence_id"]
        for document in row["documents"]
        for sentence in document["sentences"]
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count
