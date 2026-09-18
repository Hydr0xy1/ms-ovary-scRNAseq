#!/usr/bin/env python
from __future__ import annotations

import argparse

from ms_ovary_scrna.project import load_config
from ms_ovary_scrna.stage15_program_stability import run_program_stability


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 15 balanced latent gene-program stability audit")
    parser.add_argument("--config", default="config/analysis_config.yaml")
    args = parser.parse_args()
    run_program_stability(load_config(args.config))


if __name__ == "__main__":
    main()

