from __future__ import annotations

import argparse

from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config
from ms_ovary_scrna.stage1_exploratory import run_stage1_exploratory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conservative stage-1 QC followed by unintegrated exploratory analysis."
    )
    parser.add_argument("input")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--allow-low-memory", action="store_true")
    args = parser.parse_args()
    run_stage1_exploratory(
        load_config(args.config),
        args.input,
        allow_low_memory=args.allow_low_memory,
    )


if __name__ == "__main__":
    main()
