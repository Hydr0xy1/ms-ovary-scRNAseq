#!/usr/bin/env python
from __future__ import annotations

import argparse

from ms_ovary_scrna.deep_dive_audit import run_stage15_audit
from ms_ovary_scrna.project import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 15 project-state and metadata audit")
    parser.add_argument("--config", default="config/analysis_config.yaml")
    args = parser.parse_args()
    run_stage15_audit(load_config(args.config))


if __name__ == "__main__":
    main()
