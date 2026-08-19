from __future__ import annotations

from jsonschema import Draft202012Validator

from src.data.synthetic import make_candidate, materialize_hidden_cases, verify_candidate


def test_every_pattern_accepts_deterministic_hidden_source_variants() -> None:
    indices = list(range(11)) + list(range(16000, 16005)) + [18501, 18502, 18503]
    signatures = set()
    for index in indices:
        row = make_candidate(index)
        verify_candidate(row)
        signatures.add(row["graph_signature"])
        validator = Draft202012Validator(row["source_schema"])
        for case in materialize_hidden_cases(row, 100, 4000):
            assert not list(validator.iter_errors(case["input"])), row["sample_id"]
    assert len(signatures) == 19
