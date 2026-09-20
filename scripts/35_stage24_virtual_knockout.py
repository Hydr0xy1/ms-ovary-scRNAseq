#!/usr/bin/env python
from __future__ import annotations

import argparse

from ms_ovary_scrna.project import load_config
from ms_ovary_scrna.stage24_virtual_knockout import run_stage24_virtual_knockout


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 24 library-aware scTenifoldKnk virtual-knockout pilot"
    )
    parser.add_argument("--config", default="config/analysis_config.yaml")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["prepare", "r_crosscheck", "network_queue", "summarize", "figures"],
        default=None,
    )
    args = parser.parse_args()
    run_stage24_virtual_knockout(load_config(args.config), selected_steps=args.steps)


if __name__ == "__main__":
    main()
