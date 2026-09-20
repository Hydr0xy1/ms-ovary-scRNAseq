#!/usr/bin/env python
from __future__ import annotations

import argparse

from ms_ovary_scrna.project import load_config
from ms_ovary_scrna.stage23_incremental import run_stage23_incremental


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 23 isolated incremental validation and deep-dive"
    )
    parser.add_argument("--config", default="config/analysis_config.yaml")
    parser.add_argument("--stages", nargs="+", default=None)
    parser.add_argument("--force", nargs="+", default=None)
    args = parser.parse_args()
    run_stage23_incremental(
        load_config(args.config),
        selected_stages=args.stages,
        force_stages=args.force,
    )


if __name__ == "__main__":
    main()
