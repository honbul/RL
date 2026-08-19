from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one deterministic candidate shard")
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.parse_args()
    raise SystemExit("shard generator implementation is added in Stage 2")


if __name__ == "__main__":
    main()
