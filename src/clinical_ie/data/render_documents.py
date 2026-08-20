from __future__ import annotations

import json
import random
from typing import Any, Callable

from src.clinical_ie.concepts import concept_by_id, surface_name
from src.clinical_ie.data.latent_timeline import DOCUMENT_TYPES
from src.clinical_ie.schema import ExtractionOutput, validate_evidence_ids


def _outpatient_initial(date: str, sentences: list[str], family: str) -> tuple[str, list[str]]:
    return "외래 초진 기록", sentences


def _outpatient_followup(date: str, sentences: list[str], family: str) -> tuple[str, list[str]]:
    return "외래 경과 기록", sentences


def _emergency_note(date: str, sentences: list[str], family: str) -> tuple[str, list[str]]:
    return "응급실 기록", sentences


def _inpatient_note(date: str, sentences: list[str], family: str) -> tuple[str, list[str]]:
    return "입원 기록", sentences


def _discharge_summary(date: str, sentences: list[str], family: str) -> tuple[str, list[str]]:
    return "퇴원 요약", sentences


def _referral_note(date: str, sentences: list[str], family: str) -> tuple[str, list[str]]:
    return "진료 의뢰서", sentences


def _screening_result(date: str, sentences: list[str], family: str) -> tuple[str, list[str]]:
    return "건강검진 결과", sentences


def _problem_list(date: str, sentences: list[str], family: str) -> tuple[str, list[str]]:
    return "과거력 및 문제 목록", sentences


RENDERERS: dict[str, Callable[[str, list[str], str], tuple[str, list[str]]]] = {
    "outpatient_initial": _outpatient_initial,
    "outpatient_followup": _outpatient_followup,
    "emergency_note": _emergency_note,
    "inpatient_note": _inpatient_note,
    "discharge_summary": _discharge_summary,
    "referral_note": _referral_note,
    "screening_result": _screening_result,
    "problem_list": _problem_list,
}
assert set(RENDERERS) == set(DOCUMENT_TYPES)


def _time_text(value: str | None) -> str:
    if value is None:
        return "시점 미상의 기록에서"
    if value.startswith("relative:"):
        return value.removeprefix("relative:").replace("_", " ") + "에"
    return value + "년에"


def _condition_sentence(condition: dict[str, Any], encounter: dict[str, Any] | None, rng: random.Random, linguistic: bool) -> str:
    concept = concept_by_id()[condition["concept_id"]]
    name = surface_name(concept, rng, linguistic)
    time = _time_text(condition["event_time"])
    location = f" {encounter['facility']}에서" if encounter else ""
    assertion = condition["assertion"]
    if assertion == "active":
        return f"환자는 {time}{location} {name} 진단을 받았으며 현재도 치료 또는 추적 중이다."
    if assertion == "resolved":
        return f"환자는 {time}{location} {name} 치료를 받았고 이후 호전되어 해결된 상태로 기록되었다."
    if assertion == "suspected":
        return f"환자에게 {time}{location} {name} 가능성이 의심되어 추가 확인이 필요하다고 판단했다."
    return f"환자에게 {time}{location} {name} 가능성이 제기되었으나 최종 평가에서 배제되었다."


def _encounter_sentence(encounter: dict[str, Any], rng: random.Random) -> str:
    type_text = {"outpatient": "외래 진료", "emergency": "응급실 진료", "inpatient": "입원 치료", "screening": "건강검진"}[encounter["encounter_type"]]
    outcome_text = {"admitted": "입원하였다", "discharged": "진료 후 귀가 또는 퇴원하였다", "transferred": "다른 기관으로 전원되었다", "ongoing": "현재 치료가 진행 중이다", "unknown": "이후 경과는 명확하지 않다"}[encounter["outcome"]]
    return f"환자는 {encounter['date']}년에 {encounter['facility']}에서 {type_text}를 받았고 {outcome_text}."


def render_sample(latent: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(latent.get("render_seed", latent["seed"]) + 91_337)
    linguistic = latent["renderer_family"] == "heldout_linguistic"
    sentence_specs: list[dict[str, Any]] = []
    condition_main: dict[str, int] = {}
    condition_repeat: dict[str, int] = {}
    encounter_main: dict[str, int] = {}
    relation_by_condition = {item["condition_latent_id"]: item for item in latent["relations"]}
    encounter_by_id = {item["latent_id"]: item for item in latent["encounters"]}
    for condition in latent["conditions"]:
        relation = relation_by_condition.get(condition["latent_id"])
        encounter = encounter_by_id.get(relation["encounter_latent_id"]) if relation else None
        condition_main[condition["latent_id"]] = len(sentence_specs)
        sentence_specs.append({"text": _condition_sentence(condition, encounter, rng, linguistic), "kind": "condition", "latent_id": condition["latent_id"]})
        if encounter is not None:
            encounter_main.setdefault(encounter["latent_id"], len(sentence_specs) - 1)
        if latent["difficulty_level"] >= 3 and rng.random() < 0.75:
            concept = concept_by_id()[condition["concept_id"]]
            name = surface_name(concept, rng, linguistic)
            condition_repeat[condition["latent_id"]] = len(sentence_specs)
            sentence_specs.append({"text": f"후속 기록에서도 환자의 {name} 상태는 이전 판단과 연결하여 확인되었다.", "kind": "condition_repeat", "latent_id": condition["latent_id"]})
    for encounter in latent["encounters"]:
        if encounter["latent_id"] not in encounter_main:
            encounter_main[encounter["latent_id"]] = len(sentence_specs)
            sentence_specs.append({"text": _encounter_sentence(encounter, rng), "kind": "encounter", "latent_id": encounter["latent_id"]})
    for family in latent["family_distractors"]:
        concept = concept_by_id()[family["concept_id"]]
        name = surface_name(concept, rng, linguistic)
        sentence_specs.append({"text": f"가족력으로 {family['subject']}이 {name} 치료를 받고 있으나 환자 본인의 진단은 아니다.", "kind": "family", "latent_id": None})
    if "copy_forward_conflict" in latent["difficulty_tags"]:
        first = latent["conditions"][0]
        name = surface_name(concept_by_id()[first["concept_id"]], rng, linguistic)
        sentence_specs.append({"text": f"과거 문제 목록에는 {name} 항목이 복사되어 있으나 최신 담당의 판단을 우선해야 한다.", "kind": "conflict", "latent_id": None})
    filler_templates = [
        "활력징후는 전반적으로 안정적이었고 급성 악화 소견은 없었다.",
        "일상생활과 식사 상태에 관한 일반 상담을 시행하였다.",
        "관련 없는 경미한 피로와 수면 변화가 함께 기록되었다.",
        "기본 신체진찰에서 즉시 처치가 필요한 새로운 소견은 없었다.",
        "검사 일정과 추후 연락 방법을 안내하였다.",
        "복약 여부와 생활 습관에 대한 일반적인 교육을 제공하였다.",
        "보호자가 동행했으며 문진 내용은 환자 진술을 중심으로 정리했다.",
        "이전 기록 일부는 반복 기재되어 최신 판단과 구분할 필요가 있다.",
    ]
    while len(sentence_specs) < latent["target_sentence_count"]:
        index = len(sentence_specs)
        sentence_specs.append({"text": filler_templates[index % len(filler_templates)], "kind": "filler", "latent_id": None})
    document_count = len(latent["document_types"])
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(document_count)]
    for index, spec in enumerate(sentence_specs):
        buckets[index % document_count].append(spec)
    documents = []
    spec_to_sentence: dict[int, str] = {}
    sentence_counter = 1
    for doc_index, (doc_type, specs) in enumerate(zip(latent["document_types"], buckets, strict=True), start=1):
        date = str(2018 + ((latent.get("render_slot", latent["slot"]) + doc_index) % 8))
        rendered_title, _ = RENDERERS[doc_type](date, [item["text"] for item in specs], latent["renderer_family"])
        sentences = []
        for spec in specs:
            sid = f"S{sentence_counter:03d}"
            spec_to_sentence[id(spec)] = sid
            sentences.append({"sentence_id": sid, "text": spec["text"]})
            sentence_counter += 1
        documents.append({"document_id": f"D{doc_index}", "document_date": date, "document_type": doc_type, "title": rendered_title, "sentences": sentences})
    def sid_for_spec(index: int) -> str:
        return spec_to_sentence[id(sentence_specs[index])]
    condition_ids = {item["latent_id"]: f"C{index}" for index, item in enumerate(latent["conditions"], start=1)}
    encounter_ids = {item["latent_id"]: f"E{index}" for index, item in enumerate(latent["encounters"], start=1)}
    gold_conditions = []
    acceptable_conditions: dict[str, list[list[str]]] = {}
    for condition in latent["conditions"]:
        cid = condition_ids[condition["latent_id"]]
        main = sid_for_spec(condition_main[condition["latent_id"]])
        evidence = [main]
        groups = [[main]]
        if condition["latent_id"] in condition_repeat:
            repeat = sid_for_spec(condition_repeat[condition["latent_id"]])
            groups.append([main, repeat])
        acceptable_conditions[cid] = groups
        gold_conditions.append({"id": cid, "concept_id": condition["concept_id"], "assertion": condition["assertion"], "event_time": condition["event_time"], "evidence_sentence_ids": evidence})
    gold_encounters = []
    acceptable_encounters: dict[str, list[list[str]]] = {}
    for encounter in latent["encounters"]:
        eid = encounter_ids[encounter["latent_id"]]
        main = sid_for_spec(encounter_main[encounter["latent_id"]])
        acceptable_encounters[eid] = [[main]]
        gold_encounters.append({"id": eid, "facility": encounter["facility"], "date": encounter["date"], "encounter_type": encounter["encounter_type"], "reason_concept_id": encounter["reason_concept_id"], "outcome": encounter["outcome"], "evidence_sentence_ids": [main]})
    gold_relations = [{"condition_id": condition_ids[item["condition_latent_id"]], "encounter_id": encounter_ids[item["encounter_latent_id"]], "relation_type": item["relation_type"]} for item in latent["relations"]]
    gold_output = {"conditions": gold_conditions, "encounters": gold_encounters, "relations": gold_relations}
    sentence_ids = {sentence["sentence_id"] for document in documents for sentence in document["sentences"]}
    parsed = ExtractionOutput.model_validate(gold_output)
    validate_evidence_ids(parsed, sentence_ids)
    document_text = []
    for document in documents:
        document_text.extend([
            f"[DOCUMENT {document['document_id']}]",
            f"document_date: {document['document_date']}",
            f"document_type: {document['document_type']}",
            "",
            *[f"[{sentence['sentence_id']}] {sentence['text']}" for sentence in document["sentences"]],
            "",
        ])
    schema_instruction = (
        "Output schema: conditions[{id,concept_id,assertion,event_time,evidence_sentence_ids}], "
        "encounters[{id,facility,date,encounter_type,reason_concept_id,outcome,evidence_sentence_ids}], "
        "relations[{condition_id,encounter_id,relation_type}]."
    )
    prompt = "\n".join([
        *document_text,
        schema_instruction,
        "Return only one JSON object conforming to the provided extraction schema.",
        "Do not include explanations, Markdown fences, or thinking text.",
    ])
    return {
        "sample_id": latent["sample_id"],
        "difficulty_level": latent["difficulty_level"],
        "difficulty_tags": latent["difficulty_tags"],
        "documents": documents,
        "prompt": prompt,
        "gold_output": gold_output,
        "acceptable_evidence_groups": {"conditions": acceptable_conditions, "encounters": acceptable_encounters},
        "latent_timeline": latent,
        "topology_fingerprint": latent["topology_fingerprint"],
        "renderer_family": latent["renderer_family"],
        "counterfactual_pair_id": latent["counterfactual_pair_id"],
        "counterfactual_role": latent["counterfactual_role"],
        "seed": latent["seed"],
    }


def public_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("sample_id", "difficulty_level", "difficulty_tags", "documents", "prompt", "topology_fingerprint", "renderer_family", "counterfactual_pair_id", "counterfactual_role", "seed")}


def canonical_json(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
