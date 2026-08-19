# Synthetic Schema RLVR

Train a 4B instruct model to emit an executable, restricted JSON transformation DSL from source/target JSON schemas and a natural-language requirement.

## Status

The experiment is executed stage by stage on `experiment/synthetic-schema-rlvr`. See `PROGRESS.md` for exact commands and results.

## Entrypoints

```bash
python -m src.data.generate --help
python -m src.train_sft --help
python -m src.train_grpo --help
python -m src.evaluate --help
```

Complete reproduction commands are added after the runners have passed their smoke tests.
