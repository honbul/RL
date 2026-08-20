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
- Commit: `f21c3f706d945c0d470625439454e26905375991`
- Next: P3 generate the full 40,000-candidate pool with 8 CPU workers and build all splits/manifests

## P3 — Full 40K dataset generation

- Status: generated and verified
- Commands: `.venv/bin/python -m src.clinical_ie.data.generate --config configs/clinical_ie/data.yaml --workers 8`; `.venv/bin/python -m src.clinical_ie.data.build_splits --config configs/clinical_ie/data.yaml`; `.venv/bin/python -m src.clinical_ie.data.validate_generated --config configs/clinical_ie/data.yaml`
- Major files: `data/clinical_ie/candidates/`, `data/clinical_ie/sft/`, `data/clinical_ie/hard/`, `data/clinical_ie/dev/`, `data/clinical_ie/test/`, `data/clinical_ie/manifests/`, `outputs/clinical_ie/pre_gpu/data_generation.log`
- Result: 40,000 candidates; SFT 8K/24K, hard 8K, dev 2K, five test splits × 1K; 95,000 strict reference rows validated; 500 counterfactual pairs; all sample-ID/topology overlaps 0; all 30 JSONL artifacts under 50MiB
- GPU work started: false
- Commit: `1cb9000dc04bcf7eee64816b2046b44bea636fd0`
- Next: P4 prepare SFT/GRPO/frontier/evaluation runners and execute CPU-only contract checks

## P4 — Training, frontier, evaluation, and summary runners

- Status: implemented and CPU-verified
- Commands: `.venv/bin/python -m src.clinical_ie.train_sft --config configs/clinical_ie/sft_8k.yaml --dry-run --limit 2`; `.venv/bin/python -m src.clinical_ie.select_frontier --dry-run`; `.venv/bin/python -m src.clinical_ie.train_grpo --config configs/clinical_ie/grpo.yaml --dry-run`; `.venv/bin/python -m src.clinical_ie.evaluate --run-dir outputs/clinical_ie/pre_gpu/cpu_contract/evaluate --split dev_2k --dry-run --limit 2`; `.venv/bin/python -m src.clinical_ie.summarize --dry-run`
- Major files: `src/clinical_ie/train_sft.py`, `src/clinical_ie/select_frontier.py`, `src/clinical_ie/train_grpo.py`, `src/clinical_ie/evaluate.py`, `src/clinical_ie/rescore.py`, `src/clinical_ie/summarize.py`, `configs/clinical_ie/*.yaml`
- Result: all CLI help/config parse/data row/reward/output/raw writer/dummy summary contracts PASS; S1 first row `clinical_000000`, S2 continuation first row `clinical_008000`; gold/error rewards 1.0/0.0; model_loaded=false; gpu_initialized=false
- S2X note: the fixed 40K plan provides only 1K reserve rows, so the conditional 8K S2X extension is explicitly blocked rather than contaminating the hard frontier pool
- Commit: recorded in the next stage update after publication
- Next: P5 write WAITING_FOR_GPU status, README marker, final clean push/readback, and stop
