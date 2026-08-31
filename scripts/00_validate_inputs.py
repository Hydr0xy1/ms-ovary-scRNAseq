from __future__ import annotations

import argparse

from ms_ovary_scrna.io import validate_inputs
from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-memory audit of 10x MTX inputs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--full-gzip-test",
        action="store_true",
        help="Read every compressed byte and verify CRC (safe but slow).",
    )
    args = parser.parse_args()
    validate_inputs(load_config(args.config), full_gzip_test=args.full_gzip_test)


if __name__ == "__main__":
    main()
