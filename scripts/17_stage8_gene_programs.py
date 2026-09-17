#!/usr/bin/env python
from __future__ import annotations

import argparse

from ms_ovary_scrna.project import load_config
from ms_ovary_scrna.stage8_gene_programs import run_stage8


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 8 data-driven gene programs")
    parser.add_argument("--config", default="config/analysis_config.yaml")
    parser.add_argument("--max-cells-per-library", type=int, default=1000)
    parser.add_argument("--n-hvg", type=int, default=2000)
    args = parser.parse_args()
    run_stage8(
        load_config(args.config),
        max_cells_per_library=args.max_cells_per_library,
        n_hvg=args.n_hvg,
    )


if __name__ == "__main__":
    main()
