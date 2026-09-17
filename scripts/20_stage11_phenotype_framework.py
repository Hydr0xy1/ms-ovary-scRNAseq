#!/usr/bin/env python
from __future__ import annotations

import argparse

from ms_ovary_scrna.project import load_config
from ms_ovary_scrna.stage11_phenotype import run_stage11


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 11 transcriptome-to-phenotype framework")
    parser.add_argument("--config", default="config/analysis_config.yaml")
    args = parser.parse_args()
    run_stage11(load_config(args.config))


if __name__ == "__main__":
    main()
