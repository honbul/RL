from __future__ import annotations

import argparse
from pathlib import Path


def config_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration path")
    return parser
