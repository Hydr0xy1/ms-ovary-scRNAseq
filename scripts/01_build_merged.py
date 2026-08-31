from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc
from scipy import sparse

from _project import (
    DEFAULT_CONFIG,
    load_config,
    project_paths,
    read_metadata,
    require_compute_resources,
    setup_logging,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a merged counts AnnData object.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=None)
    parser.add_argument("--allow-low-memory", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    require_compute_resources(config, allow_low_memory=args.allow_low_memory)
    paths = project_paths(config)
    metadata = read_metadata(config)
    logger = setup_logging("01_build_merged", config)
    ingest = config["ingest"]

    objects: dict[str, ad.AnnData] = {}
    for library_id, row in metadata.iterrows():
        sample_dir = paths["input"] / library_id
        logger.info("Reading %s", sample_dir)
        adata = sc.read_10x_mtx(
            sample_dir,
            var_names=ingest["var_names"],
            make_unique=bool(ingest["make_unique"]),
            gex_only=bool(ingest["gene_expression_only"]),
            cache=False,
        )
        adata.X = sparse.csr_matrix(adata.X)
        adata.obs_names = [
            f"{library_id}{ingest['barcode_separator']}{barcode}"
            for barcode in adata.obs_names
        ]
        for column, value in row.items():
            adata.obs[column] = str(value)
        objects[library_id] = adata
        logger.info("%s shape=%s nnz=%d", library_id, adata.shape, adata.X.nnz)

    merged = ad.concat(
        objects,
        axis=0,
        join="inner",
        merge="same",
        label=None,
        index_unique=None,
    )
    merged.X = sparse.csr_matrix(merged.X)
    merged.layers[ingest["counts_layer"]] = merged.X.copy()
    for categorical in ("library_id", "group", "treatment", "batch", "estrous_stage"):
        if categorical in merged.obs:
            merged.obs[categorical] = merged.obs[categorical].astype("category")
    merged.uns["analysis_groups"] = {
        "aging": "OC_vs_Y",
        "treatment": "OT_vs_OC",
        "residual": "OT_vs_Y",
    }

    output = Path(args.output) if args.output else paths["results"] / "01_merged_counts.h5ad"
    merged.write_h5ad(output, compression="gzip")
    pd.DataFrame(
        {
            "cells": merged.obs.groupby("library_id", observed=True).size(),
        }
    ).to_csv(paths["results"] / "01_cells_by_library.tsv", sep="\t")
    logger.info("MERGE_OK: %s shape=%s", output, merged.shape)


if __name__ == "__main__":
    main()
