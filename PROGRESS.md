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
- Commit: `e2b150c960fdb9054214d2971d7cdc912c7b7431`
- Next: run E0, E1, and E2 completely and publish their raw outputs

## Stage 4 — E0/E1/E2 baselines

- Status: completed and raw-rescored
- Completed: E0 zero-shot full evaluation; E1 SFT-1K training and evaluation; E2 SFT-6K training and evaluation; all success/failure raw rows retained
- Commands: `.venv/bin/python -m src.evaluate --run-dir outputs/e0_zero_shot --config configs/e0_zero_shot.yaml --all-splits --physical-gpu 1`; `PYTHONHASHSEED=42 .venv/bin/python -m src.train_sft --config configs/sft_1k.yaml`; `PYTHONHASHSEED=42 .venv/bin/python -m src.train_sft --config configs/sft_6k.yaml`; `.venv/bin/python -m src.rescore --run-dir <run-dir>`
- Major files: `outputs/e0_zero_shot/`, `outputs/e1_sft_1k/`, `outputs/e2_sft_6k/`
- Result: E0 exact IID/Linguistic/Compositional/Structural = 0/0/0/0; E1 = 0.171/0.158/0.001/0; E2 = 1.000/1.000/0.568/0
- Training: E1 loss 0.539731, peak 24,373.25MB, 207.24s; E2 loss 0.078594, peak 24,525.25MB, 1,239.40s
- Metric correction: field-micro originally omitted expected leaves on parse failures; raw generations were unchanged and E0/E1/E2 metrics were deterministically recomputed by `src.rescore`
- Commit: recorded in the next progress update after publication
- Next: run E3 binary and E4 dense GRPO from the E1 adapter, then evaluate both on all four splits
