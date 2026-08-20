from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ID_CONDITION = re.compile(r"^C[1-9][0-9]*$")
ID_ENCOUNTER = re.compile(r"^E[1-9][0-9]*$")
EVENT_TIME = re.compile(r"^(?:[0-9]{4}(?:-[0-9]{2}(?:-[0-9]{2})?)?|relative:[a-z0-9_+-]+)$")
SENTENCE_ID = re.compile(r"^S[0-9]{3,}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", populate_by_name=True)


class Condition(StrictModel):
    id: str = Field(pattern=ID_CONDITION.pattern)
    concept_id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    assertion: Literal["active", "resolved", "suspected", "ruled_out"]
    event_time: str | None = None
    evidence_sentence_ids: list[str] = Field(min_length=1)

    @field_validator("event_time")
    @classmethod
    def valid_event_time(cls, value: str | None) -> str | None:
        if value is not None and not EVENT_TIME.fullmatch(value):
            raise ValueError("invalid event_time")
        return value

    @field_validator("evidence_sentence_ids")
    @classmethod
    def unique_evidence(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(not SENTENCE_ID.fullmatch(item) for item in value):
            raise ValueError("evidence IDs must be unique sentence IDs")
        return value


class Encounter(StrictModel):
    id: str = Field(pattern=ID_ENCOUNTER.pattern)
    facility: str = Field(min_length=1)
    date: str | None = None
    encounter_type: Literal["outpatient", "emergency", "inpatient", "screening"]
    reason_concept_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]*$")
    outcome: Literal["admitted", "discharged", "transferred", "ongoing", "unknown"]
    evidence_sentence_ids: list[str] = Field(min_length=1)

    @field_validator("date")
    @classmethod
    def valid_date(cls, value: str | None) -> str | None:
        if value is not None and not EVENT_TIME.fullmatch(value):
            raise ValueError("invalid encounter date")
        return value

    @field_validator("evidence_sentence_ids")
    @classmethod
    def unique_evidence(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(not SENTENCE_ID.fullmatch(item) for item in value):
            raise ValueError("evidence IDs must be unique sentence IDs")
        return value


class Relation(StrictModel):
    condition_id: str = Field(pattern=ID_CONDITION.pattern)
    encounter_id: str = Field(pattern=ID_ENCOUNTER.pattern)
    relation_type: Literal["diagnosed_at", "treated_at", "observed_at"]


class ExtractionOutput(StrictModel):
    conditions: list[Condition]
    encounters: list[Encounter]
    relations: list[Relation]

    @model_validator(mode="after")
    def validate_graph(self) -> "ExtractionOutput":
        condition_ids = [item.id for item in self.conditions]
        encounter_ids = [item.id for item in self.encounters]
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("duplicate condition ID")
        if len(encounter_ids) != len(set(encounter_ids)):
            raise ValueError("duplicate encounter ID")
        condition_set = set(condition_ids)
        encounter_set = set(encounter_ids)
        for relation in self.relations:
            if relation.condition_id not in condition_set or relation.encounter_id not in encounter_set:
                raise ValueError("dangling relation reference")
        relation_keys = [(r.condition_id, r.encounter_id, r.relation_type) for r in self.relations]
        if len(relation_keys) != len(set(relation_keys)):
            raise ValueError("duplicate relation")
        condition_semantics = [
            (c.concept_id, c.assertion, c.event_time, tuple(sorted(c.evidence_sentence_ids)))
            for c in self.conditions
        ]
        encounter_semantics = [
            (e.facility, e.date, e.encounter_type, e.reason_concept_id, e.outcome, tuple(sorted(e.evidence_sentence_ids)))
            for e in self.encounters
        ]
        if len(condition_semantics) != len(set(condition_semantics)):
            raise ValueError("duplicate condition object")
        if len(encounter_semantics) != len(set(encounter_semantics)):
            raise ValueError("duplicate encounter object")
        return self


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def parse_completion(completion: str | dict[str, Any]) -> ExtractionOutput:
    if isinstance(completion, str):
        payload = json.loads(
            completion,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    else:
        payload = completion
    if not isinstance(payload, dict):
        raise ValueError("completion must be one JSON object")
    return ExtractionOutput.model_validate(payload)


def validate_evidence_ids(output: ExtractionOutput, sentence_ids: set[str]) -> None:
    for item in [*output.conditions, *output.encounters]:
        unknown = set(item.evidence_sentence_ids) - sentence_ids
        if unknown:
            raise ValueError(f"unknown evidence sentence IDs: {sorted(unknown)}")
