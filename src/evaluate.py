from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a base model or adapter on all benchmark splits")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--all-splits", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.parse_args()
    raise SystemExit("evaluation implementation is added in Stage 3")


if __name__ == "__main__":
    main()
