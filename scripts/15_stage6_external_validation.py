#!/usr/bin/env python
from __future__ import annotations

import argparse

from ms_ovary_scrna.project import load_config
from ms_ovary_scrna.stage6_external_validation import run_stage6


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6 external ovarian-aging validation")
    parser.add_argument("--config", default="config/analysis_config.yaml")
    parser.add_argument("--rscript", default="/root/autodl-tmp/envs/stage6_r/bin/Rscript")
    args = parser.parse_args()
    run_stage6(load_config(args.config), rscript=args.rscript)


if __name__ == "__main__":
    main()
