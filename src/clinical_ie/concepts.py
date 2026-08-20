from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def load_concepts(path: str = "configs/clinical_ie/concepts.json") -> list[dict[str, Any]]:
    concepts = json.loads(Path(path).read_text(encoding="utf-8"))
    if not 30 <= len(concepts) <= 50:
        raise ValueError("clinical concept vocabulary must contain 30-50 concepts")
    ids = [item["concept_id"] for item in concepts]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate concept IDs")
    return concepts


def surface_name(concept: dict[str, Any], rng: random.Random, linguistic_ood: bool = False) -> str:
    choices = list(concept["ko_names"])
    if linguistic_ood:
        choices.extend(concept["en_names"])
        choices.extend(concept["abbreviations"])
    elif concept["abbreviations"] and rng.random() < 0.25:
        choices.extend(concept["abbreviations"])
    return rng.choice(choices)


def concept_by_id() -> dict[str, dict[str, Any]]:
    return {item["concept_id"]: item for item in load_concepts()}
