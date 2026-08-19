# Progress

## Stage 0 — Minimal project initialization

- Status: implemented and verified
- Completed: repository skeleton, minimal configs, CLI entry points
- Commands: `python -m src.data.generate --help`; `python -m src.train_sft --help`; `python -m src.train_grpo --help`; `python -m src.evaluate --help`
- Major files: `configs/*.yaml`, `src/`, `requirements.txt`, `.gitignore`
- Result: all four CLIs returned help, all Stage 0 modules imported, and `compileall` passed
- Commit: `8b9aab338310704ddf264d4fb3a7e4811db36b85`
- Next: Stage 1 executable DSL and rewards

## Stage 1 — Executable DSL and rewards

- Status: implemented and verified
- Completed: strict Pydantic DSL, deterministic interpreter, canonical comparison, binary reward, dense reward, 20 executable reference examples
- Commands: `.venv/bin/python -m src.data.generate_examples --output data/dev/core_examples_20.jsonl`; `.venv/bin/python -m pytest -q tests/test_core_examples.py`
- Major files: `src/dsl/*.py`, `src/rewards.py`, `data/dev/core_examples_20.jsonl`, `tests/test_core_examples.py`
- Result: 20/20 reference programs passed binary and dense reward; 4 focused tests passed; invalid op/source path/target path rejected
- Commit: `908c0e7830908ecf651e6f32775c18fe7e1d2ec3`
- Next: generate and validate the 20,000-candidate dataset with four shard workers

## Stage 2 — Synthetic dataset generation

- Status: implemented, generated, and verified
- Completed: 20,000-candidate pool with four parallel workers; SFT, GRPO, dev, IID, linguistic OOD, compositional OOD, and structural OOD splits
- Commands: `.venv/bin/python -m src.data.generate --config configs/data.yaml`; `.venv/bin/python -m src.data.validate_generated --config configs/data.yaml`
- Major files: `data/candidates/*.jsonl`, `data/sft/*.jsonl`, `data/grpo/*.jsonl`, `data/dev/dev.jsonl`, `data/test/*.jsonl`, `data/manifests/*.json`
- Result: 20,000 candidates; 14,500 reference-bearing rows re-executed; exact requested counts; all four leakage checks zero; tokenizer audit max prompt 1,665/2,048 and completion 115/256
- Commit: `805b2366e823a04e7e0c288e71262d21161a64ed`
- Next: complete and smoke-test the SFT, GRPO, vLLM, evaluation, and raw-output runners

## Stage 3 — SFT, GRPO, vLLM, and evaluation runners

- Status: implemented and verified
- Completed: reproducible SFT/GRPO CLIs, self-managed two-GPU vLLM server, adapter-aware evaluation, raw logging, exact token-budget audit, environment lock
- Commands: `.venv/bin/python -m src.train_sft --config configs/sft_1k.yaml --max-steps 10 --limit 100 --output-dir outputs/smoke/sft_10_steps`; `.venv/bin/python -m src.train_grpo --config configs/grpo_binary.yaml --max-steps 10 --limit 100 --output-dir outputs/smoke/grpo_binary_10_steps_attempt2 --policy-adapter outputs/smoke/sft_10_steps/adapter`; `.venv/bin/python -m src.verify_smoke`
- Major files: `src/train_sft.py`, `src/train_grpo.py`, `src/evaluate.py`, `src/gpu.py`, `requirements-lock.txt`, `environment.txt`, `outputs/smoke/`
- Result: SFT 10/10 steps PASS (peak 20,622.56MB); E0 vLLM 50 rows PASS; SFT adapter vLLM 50 rows PASS; binary GRPO 10/10 steps and 40 rollouts PASS (policy 9,905.25MB, rollout 22,350.25MB); all smoke rewards were 0.0, so smoke establishes execution only
- Runtime fallback: vLLM required a local C compiler and Python headers; compiler closure was fixed before the successful attempts; server mode then passed and no Transformers-generation fallback was used
- Commit: recorded in the next progress update after publication
- Next: run E0, E1, and E2 completely and publish their raw outputs
