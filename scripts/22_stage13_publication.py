#!/usr/bin/env python
from __future__ import annotations

import argparse

from ms_ovary_scrna.project import load_config
from ms_ovary_scrna.stage13_publication import run_stage13


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 13 publication evidence consolidation and standalone figure rendering"
    )
    parser.add_argument("--config", default="config/analysis_config.yaml")
    args = parser.parse_args()
    run_stage13(load_config(args.config))


if __name__ == "__main__":
    main()

