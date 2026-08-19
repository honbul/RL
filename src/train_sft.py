from __future__ import annotations

from src.common import config_parser


def main() -> None:
    parser = config_parser("Train a LoRA adapter with supervised DSL examples")
    parser.parse_args()
    raise SystemExit("SFT implementation is added in Stage 3")


if __name__ == "__main__":
    main()
