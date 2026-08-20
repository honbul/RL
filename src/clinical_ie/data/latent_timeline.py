from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from typing import Any

from src.clinical_ie.concepts import load_concepts

DOCUMENT_TYPES = [
    "outpatient_initial",
    "outpatient_followup",
    "emergency_note",
    "inpatient_note",
    "discharge_summary",
    "referral_note",
    "screening_result",
    "problem_list",
]
FACILITY_PREFIXES = ["가온", "누리", "다솜", "라온", "마루", "바른", "새론", "아람", "이든", "푸른", "해온", "희원"]
FACILITY_SUFFIXES = ["병원", "의원", "메디컬센터", "진료센터", "건강센터"]


def slot_regime(slot: int) -> str:
    if not 0 <= slot < 40000:
        raise ValueError("slot must be in [0, 40000)")
    if slot < 24000:
        return "sft"
    if slot < 32000:
        return "hard"
    if slot < 34000:
        return "dev"
    if slot < 35000:
        return "test_iid_hard"
    if slot < 36000:
        return "test_linguistic_ood"
    if slot < 37000:
        return "test_compositional_ood"
    if slot < 38000:
        return "test_long_context_ood"
    if slot < 39000:
        return "test_counterfactual"
    return "reserve"


def difficulty_for_slot(slot: int, regime: str) -> int:
    local = slot % 100
    if regime == "sft":
        return 1 if local < 35 else 2 if local < 70 else 3 if local < 90 else 4
    if regime == "hard":
        return 3 if local < 35 else 4 if local < 75 else 5
    if regime == "dev":
        return 2 if local < 15 else 3 if local < 45 else 4 if local < 80 else 5
    if regime == "test_iid_hard":
        return 3 if local < 35 else 4 if local < 75 else 5
    if regime == "test_linguistic_ood":
        return 3 if local < 30 else 4 if local < 75 else 5
    if regime == "test_compositional_ood":
        return 4 if local < 60 else 5
    if regime == "test_long_context_ood":
        return 5
    if regime == "test_counterfactual":
        return 4 if local < 60 else 5
    return 3 if local < 50 else 4


def _facility(rng: random.Random, used: set[str]) -> str:
    for _ in range(100):
        value = rng.choice(FACILITY_PREFIXES) + rng.choice(FACILITY_SUFFIXES)
        if value not in used:
            used.add(value)
            return value
    raise RuntimeError("facility namespace exhausted")


def _base_latent(slot: int, base_seed: int) -> dict[str, Any]:
    regime = slot_regime(slot)
    seed = base_seed * 1_000_000 + slot
    rng = random.Random(seed)
    difficulty = difficulty_for_slot(slot, regime)
    condition_ranges = {1: (1, 1), 2: (1, 3), 3: (2, 4), 4: (3, 6), 5: (4, 8)}
    encounter_ranges = {1: (0, 1), 2: (0, 2), 3: (1, 3), 4: (2, 4), 5: (3, 6)}
    document_ranges = {1: (1, 1), 2: (1, 2), 3: (2, 3), 4: (3, 5), 5: (4, 6)}
    sentence_ranges = {1: (3, 5), 2: (5, 8), 3: (8, 14), 4: (12, 24), 5: (20, 40)}
    condition_count = rng.randint(*condition_ranges[difficulty])
    encounter_count = rng.randint(*encounter_ranges[difficulty])
    document_count = rng.randint(*document_ranges[difficulty])
    target_sentence_count = rng.randint(*sentence_ranges[difficulty])
    concepts = rng.sample(load_concepts(), condition_count + (1 if difficulty >= 4 else 0))
    assertions = ["active"] if difficulty == 1 else ["active", "resolved", "suspected", "ruled_out"]
    conditions = []
    for index, concept in enumerate(concepts[:condition_count], start=1):
        assertion = rng.choice(assertions)
        year = rng.randint(2016, 2025)
        event_time: str | None = str(year)
        if difficulty >= 5 and rng.random() < 0.2:
            event_time = f"relative:{rng.randint(1,5)}_years_ago"
        conditions.append({
            "latent_id": f"LC{index}",
            "concept_id": concept["concept_id"],
            "assertion": assertion,
            "event_time": event_time,
        })
    used_facilities: set[str] = set()
    encounters = []
    relations = []
    encounter_types = ["outpatient", "emergency", "inpatient", "screening"]
    relation_types = {"outpatient": "diagnosed_at", "emergency": "observed_at", "inpatient": "treated_at", "screening": "observed_at"}
    outcomes = {
        "outpatient": ["ongoing", "unknown"],
        "emergency": ["admitted", "discharged", "transferred"],
        "inpatient": ["discharged", "transferred", "ongoing"],
        "screening": ["unknown"],
    }
    for index in range(1, encounter_count + 1):
        kind = rng.choice(encounter_types)
        reason = conditions[(index - 1) % len(conditions)]["concept_id"] if conditions and rng.random() < 0.85 else None
        encounter = {
            "latent_id": f"LE{index}",
            "facility": _facility(rng, used_facilities),
            "date": str(rng.randint(2016, 2025)),
            "encounter_type": kind,
            "reason_concept_id": reason,
            "outcome": rng.choice(outcomes[kind]),
        }
        encounters.append(encounter)
        if reason is not None:
            condition = next(item for item in conditions if item["concept_id"] == reason)
            relations.append({
                "condition_latent_id": condition["latent_id"],
                "encounter_latent_id": encounter["latent_id"],
                "relation_type": relation_types[kind],
            })
    tags = []
    assertion_tags = {"suspected": "suspected", "ruled_out": "ruled_out", "resolved": "resolved"}
    tags.extend(sorted({assertion_tags[c["assertion"]] for c in conditions if c["assertion"] in assertion_tags}))
    if any(c["event_time"] and c["event_time"].startswith("relative:") for c in conditions):
        tags.append("relative_time")
    if difficulty >= 3:
        tags.extend(["status_transition", "long_distance_evidence"])
    family_distractors = []
    if difficulty >= 4:
        family_concept = concepts[-1]
        family_distractors.append({"subject": rng.choice(["부친", "모친", "형제"]), "concept_id": family_concept["concept_id"]})
        tags.extend(["family_distractor", "copy_forward_conflict", "multi_hospital"])
    if difficulty == 5:
        tags.append("long_context")
    if regime == "test_compositional_ood":
        tags.extend(["heldout_composition", "negation"])
    renderer_family = "heldout_linguistic" if regime == "test_linguistic_ood" else "standard"
    if regime == "test_long_context_ood":
        renderer_family = "long_context"
    document_types = rng.sample(DOCUMENT_TYPES, document_count)
    latent = {
        "sample_id": f"clinical_{slot:06d}",
        "slot": slot,
        "regime": regime,
        "difficulty_level": difficulty,
        "difficulty_tags": sorted(set(tags)),
        "conditions": conditions,
        "encounters": encounters,
        "relations": relations,
        "family_distractors": family_distractors,
        "document_types": document_types,
        "renderer_family": renderer_family,
        "target_sentence_count": target_sentence_count,
        "seed": seed,
        "render_seed": seed,
        "render_slot": slot,
        "counterfactual_pair_id": None,
        "counterfactual_role": None,
    }
    return latent


def generate_latent(slot: int, base_seed: int = 42) -> dict[str, Any]:
    if 38500 <= slot < 39000:
        base_slot = slot - 500
        latent = deepcopy(_base_latent(base_slot, base_seed))
        latent["sample_id"] = f"clinical_{slot:06d}"
        latent["slot"] = slot
        latent["seed"] = base_seed * 1_000_000 + slot
        latent["counterfactual_pair_id"] = f"cf_{base_slot:06d}"
        latent["counterfactual_role"] = "variant"
        condition = latent["conditions"][0]
        condition["assertion"] = "ruled_out" if condition["assertion"] != "ruled_out" else "active"
        latent["difficulty_tags"] = sorted(set([*latent["difficulty_tags"], "counterfactual_assertion_flip"]))
    else:
        latent = _base_latent(slot, base_seed)
        if 38000 <= slot < 38500:
            latent["counterfactual_pair_id"] = f"cf_{slot:06d}"
            latent["counterfactual_role"] = "base"
    normalized = {
        "regime": latent["regime"],
        "difficulty": latent["difficulty_level"],
        "assertions": [item["assertion"] for item in latent["conditions"]],
        "encounter_types": [item["encounter_type"] for item in latent["encounters"]],
        "outcomes": [item["outcome"] for item in latent["encounters"]],
        "relations": [item["relation_type"] for item in latent["relations"]],
        "tags": latent["difficulty_tags"],
        "documents": latent["document_types"],
        "renderer_family": latent["renderer_family"],
        "sentence_plan": latent["target_sentence_count"],
        "structural_variant": latent["seed"] % 1_000_003,
    }
    latent["topology_fingerprint"] = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return latent
