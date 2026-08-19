from __future__ import annotations

from src.common import config_parser


def main() -> None:
    parser = config_parser("Generate and validate the synthetic schema-RLVR dataset")
    parser.parse_args()
    raise SystemExit("generator implementation is added in Stage 2")


if __name__ == "__main__":
    main()
