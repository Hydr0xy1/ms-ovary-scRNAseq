#!/usr/bin/env python
from __future__ import annotations

import argparse

from ms_ovary_scrna.project import load_config
from ms_ovary_scrna.stage26_all_cell_cycle import run_all_cell_cycle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Low-memory all-cell Scanpy-compatible cell-cycle scoring"
    )
    parser.add_argument("--config", default="config/analysis_config.yaml")
    parser.add_argument("--chunk-nnz", type=int, default=2_000_000)
    parser.add_argument("--cells-per-chunk", type=int, default=500)
    parser.add_argument("--no-reuse-gene-means", action="store_true")
    args = parser.parse_args()
    run_all_cell_cycle(
        load_config(args.config),
        chunk_nnz=args.chunk_nnz,
        cells_per_chunk=args.cells_per_chunk,
        reuse_gene_means=not args.no_reuse_gene_means,
    )


if __name__ == "__main__":
    main()

