#!/usr/bin/env python
# ruff: noqa: E501
from __future__ import annotations

import argparse

from ms_ovary_scrna.project import load_config
from ms_ovary_scrna.stage12_synthesis import run_stage12


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 12 integrated biological synthesis")
    parser.add_argument("--config", default="config/analysis_config.yaml")
    args = parser.parse_args()
    run_stage12(load_config(args.config))


if __name__ == "__main__":
    main()
