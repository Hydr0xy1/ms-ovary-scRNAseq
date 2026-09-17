#!/usr/bin/env python
# ruff: noqa: E501
from __future__ import annotations

import argparse

from ms_ovary_scrna.project import load_config
from ms_ovary_scrna.stage7_regulatory_activity import run_stage7


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7 regulatory activity inference")
    parser.add_argument("--config", default="config/analysis_config.yaml")
    parser.add_argument("--refresh-resources", action="store_true")
    args = parser.parse_args()
    run_stage7(load_config(args.config), refresh_resources=args.refresh_resources)


if __name__ == "__main__":
    main()
