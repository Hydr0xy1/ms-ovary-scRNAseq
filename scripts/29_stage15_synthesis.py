#!/usr/bin/env python
from __future__ import annotations

import argparse

from ms_ovary_scrna.project import load_config
from ms_ovary_scrna.stage15_synthesis import run_stage15_synthesis


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 15 file-grounded synthesis")
    parser.add_argument("--config", default="config/analysis_config.yaml")
    args = parser.parse_args()
    run_stage15_synthesis(load_config(args.config))


if __name__ == "__main__":
    main()

