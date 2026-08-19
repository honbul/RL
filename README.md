# Synthetic Schema RLVR

Train a 4B instruct model to emit an executable, restricted JSON transformation DSL from source/target JSON schemas and a natural-language requirement.

## Status

The experiment is executed stage by stage on `experiment/synthetic-schema-rlvr`. See `PROGRESS.md` for exact commands and results.

## Environment

```bash
uv venv --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

vLLM/Triton needs a C compiler and Python headers. On this machine they were installed without root access with:

```bash
conda create -y -p ~/.cache/rlvr-compiler -c conda-forge c-compiler python=3.12
```

The runners use system `gcc`/`cc` first, then the compiler path above. Set `CC` and `CPATH` explicitly for another local compiler layout.

## Reproduction

```bash
# 1. Generate and validate all data with four shard workers
.venv/bin/python -m src.data.generate --config configs/data.yaml

# 2. Zero-shot evaluation
.venv/bin/python -m src.evaluate --run-dir outputs/e0_zero_shot --config configs/e0_zero_shot.yaml --all-splits

# 3. SFT 1K and evaluation
PYTHONHASHSEED=42 .venv/bin/python -m src.train_sft --config configs/sft_1k.yaml
.venv/bin/python -m src.evaluate --run-dir outputs/e1_sft_1k --all-splits --physical-gpu 1

# 4. SFT 6K and evaluation
PYTHONHASHSEED=42 .venv/bin/python -m src.train_sft --config configs/sft_6k.yaml
.venv/bin/python -m src.evaluate --run-dir outputs/e2_sft_6k --all-splits --physical-gpu 1

# 5. Binary and dense GRPO; each command owns GPU 0 policy and GPU 1 vLLM
PYTHONHASHSEED=42 .venv/bin/python -m src.train_grpo --config configs/grpo_binary.yaml
.venv/bin/python -m src.evaluate --run-dir outputs/e3_grpo_binary --all-splits --physical-gpu 1

PYTHONHASHSEED=42 .venv/bin/python -m src.train_grpo --config configs/grpo_dense.yaml
.venv/bin/python -m src.evaluate --run-dir outputs/e4_grpo_dense --all-splits --physical-gpu 1
```

Model checkpoints, optimizer state, and adapters remain local and ignored. Synthetic datasets, configs, metrics, logs, evaluation generations, and failure rows are tracked.
