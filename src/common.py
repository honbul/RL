from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml


def config_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration path")
    return parser


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be an object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def format_model_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def package_versions(names: list[str]) -> dict[str, str]:
    result = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def configure_compiler() -> str:
    configured = os.environ.get("CC")
    if configured:
        return configured
    discovered = shutil.which("gcc") or shutil.which("cc")
    if discovered:
        os.environ["CC"] = discovered
        return discovered
    local_fallback = Path.home() / ".cache/rlvr-compiler/bin/x86_64-conda-linux-gnu-gcc"
    if local_fallback.exists():
        os.environ["CC"] = str(local_fallback)
        python_headers = Path.home() / ".cache/rlvr-compiler/include/python3.12"
        if python_headers.exists():
            existing = os.environ.get("CPATH")
            os.environ["CPATH"] = f"{python_headers}:{existing}" if existing else str(python_headers)
        return str(local_fallback)
    raise RuntimeError("vLLM/Triton requires a C compiler; set CC to a working compiler")
