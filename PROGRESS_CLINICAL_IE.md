# Clinical IE RLVR Progress

## P0 — Branch and directory initialization

- Status: implemented and verified
- Commands: `git switch -c experiment/clinical-ie-rlvr b8cd57b5d9c0ab7f1fb70eb267d4ac75d08d42fd`; `git push -u origin experiment/clinical-ie-rlvr`
- Major files: `src/clinical_ie/`, `configs/clinical_ie/`, `data/clinical_ie/`, `outputs/clinical_ie/pre_gpu/`
- Result: branch created from the requested base; prior DSL code, data, outputs, and reports preserved
- Commit: `dfc17550c23d440b94d67e41391c6a5e537ed884`
- Next: P1 strict schema, canonical evaluator, rewards, and 20 core examples

## P1 — Strict schema, canonical evaluator, and rewards

- Status: implemented and verified
- Commands: `.venv/bin/python -m pytest -q tests/clinical_ie/test_core_examples.py`
- Major files: `src/clinical_ie/schema.py`, `src/clinical_ie/canonicalize.py`, `src/clinical_ie/evaluator.py`, `src/clinical_ie/rewards.py`, `data/clinical_ie/dev/core_examples_20.jsonl`
- Result: 20/20 gold outputs received reward 1.0; invalid JSON, extra field, invalid evidence ID, dangling relation were gated to 0; family-condition hallucination was penalized; 7 focused tests passed
- Commit: `be2d63a051e0382328d2ca805f4130f293563750`
- Next: P2 latent timeline, concept vocabulary, eight document renderers, and counterfactual pairs

## P2 — Latent timeline and document renderers

- Status: implemented and verified
- Commands: `.venv/bin/python -m src.clinical_ie.data.generate_shard --start 0 --end 100 --output data/clinical_ie/dev/p2_samples_100.jsonl --seed 42`
- Major files: `configs/clinical_ie/concepts.json`, `src/clinical_ie/concepts.py`, `src/clinical_ie/data/latent_timeline.py`, `src/clinical_ie/data/render_documents.py`, `src/clinical_ie/data/generate_shard.py`
- Result: 36-concept vocabulary; all 8 renderers observed; 100/100 strict schema and evidence validation PASS; repeated seed output byte-identical; counterfactual pair changed exactly one assertion fact and one sentence
- Commit: recorded in the next stage update after publication
- Next: P3 generate the full 40,000-candidate pool with 8 CPU workers and build all splits/manifests
