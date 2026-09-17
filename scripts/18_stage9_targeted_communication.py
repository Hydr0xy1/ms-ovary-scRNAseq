#!/usr/bin/env python
# ruff: noqa: E501
from __future__ import annotations

import argparse

from ms_ovary_scrna.project import load_config
from ms_ovary_scrna.stage9_communication import run_stage9


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 9 targeted cell-cell communication")
    parser.add_argument("--config", default="config/analysis_config.yaml")
    parser.add_argument("--refresh-resources", action="store_true")
    args = parser.parse_args()
    run_stage9(load_config(args.config), refresh_resources=args.refresh_resources)


if __name__ == "__main__":
    main()
