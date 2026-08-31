from __future__ import annotations

import argparse

from ms_ovary_scrna.io import build_merged
from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a merged counts AnnData object.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=None)
    parser.add_argument("--allow-low-memory", action="store_true")
    args = parser.parse_args()
    build_merged(
        load_config(args.config),
        output=args.output,
        allow_low_memory=args.allow_low_memory,
    )


if __name__ == "__main__":
    main()
