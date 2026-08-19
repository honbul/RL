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
- Commit: `f4e0cca34dfb79f988e8486c4b84165466c2f57e`
- Next: run E3 binary and E4 dense GRPO from the E1 adapter, then evaluate both on all four splits

## Stage 5 preflight — GRPO batch topology

- Status: PASS; formal setting frozen
- Result: batch 8 processed two prompt groups per step at 1.747 samples/s with policy peak 11,989.25MB; batch 16 reached 1.888 samples/s but policy peak 22,901.25MB, so batch 8 was selected for safe headroom
- Final setting: `per_device_train_batch_size=8`, `gradient_accumulation_steps=1`, `generation_batch_size=8`, `num_generations=4`, `steps_per_generation=1`
- Dataset contract correction: first E3 launch reached 1/2,000 steps before nested Arrow expansion was stopped and archived; schema/hidden columns were changed to JSON strings, reducing 4,000-row dataset construction to 0.238s, and a fresh 1-step two-GPU smoke passed
- Next: E3 binary GRPO

## Stage 5 — E3/E4 GRPO

- Status: completed and raw-rescored
- Formal source: `98c09a9de65785caebf625722e52c9698581fbde`
- Completed: E3 binary GRPO and E4 dense GRPO, 2,000 steps each; 16,000 rollout rows and 3,000 evaluation rows per run; all four evaluation splits
- Commands: `PYTHONHASHSEED=42 .venv/bin/python -m src.train_grpo --config configs/grpo_binary.yaml`; `PYTHONHASHSEED=42 .venv/bin/python -m src.train_grpo --config configs/grpo_dense.yaml`; `.venv/bin/python -m src.evaluate --run-dir <run-dir> --all-splits --physical-gpu 1`; `.venv/bin/python -m src.rescore --run-dir <run-dir>`
- Result: E3 exact IID/Linguistic/Compositional/Structural = 0.461/0.482/0.084/0; E4 = 0.449/0.448/0.074/0
- Training: E3 loss -0.032575, policy/rollout peak 19,629.25/22,741.56MB, 8,553.93s; E4 loss -0.048465, policy/rollout peak 24,135.25/22,350.25MB, 8,475.62s
- Reward: E3 rollout mean 0.365563 with 5,849/16,000 nonzero; E4 rollout mean 0.386244 with 15,806/16,000 nonzero
- Additional seeds: skipped because neither E3 nor E4 exceeded E2 Compositional OOD Problem Pass@1 of 0.568
- Commit: `2aa7aba7224b89493aca7fc86e6ba34cb03101d7`
- Next: build E0–E4 summary, failure table, final report, and remote readback

## Stage 6 — Final comparison and report

- Status: generated and verified
- Completed: E0–E4 benchmark summary, run/split failure summary, cited raw examples, final report, reproduction commands
- Commands: `.venv/bin/python -m src.summarize`
- Major files: `reports/benchmark_summary.csv`, `reports/failure_summary.csv`, `reports/final_report.md`, `src/summarize.py`
- Result: E2 Compositional OOD exact 0.568 exceeded E3 0.084 and E4 0.074; final decision is **GRPO 이점 없음**
- Additional seeds: not run because the preregistered improvement condition was false
- Commit: recorded in the publication follow-up after the report commit
- Next: final verification, report commit/push, remote SHA readback
