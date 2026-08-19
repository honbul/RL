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
- Commit: recorded in the next progress update after publication
- Next: generate and validate the 20,000-candidate dataset with four shard workers
