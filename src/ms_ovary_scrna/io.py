from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc
from scipy import sparse

from .project import project_paths, read_metadata, require_compute_resources, setup_logging


def count_gzip_lines(path: Path) -> int:
    count = 0
    with gzip.open(path, "rb") as handle:
        for count, _ in enumerate(handle, start=1):
            pass
    return count


def matrix_market_shape(path: Path) -> tuple[int, int, int]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("%"):
                rows, columns, nnz = map(int, line.split())
                return rows, columns, nnz
    raise ValueError(f"Matrix Market dimension line not found: {path}")


def uncompressed_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def gzip_integrity(path: Path) -> None:
    with gzip.open(path, "rb") as handle:
        while handle.read(8 * 1024 * 1024):
            pass


def validate_inputs(config: dict, *, full_gzip_test: bool = False) -> Path:
    paths = project_paths(config)
    metadata = read_metadata(config)
    logger = setup_logging("00_validate_inputs", config)
    records: list[dict[str, object]] = []

    for library_id in metadata.index:
        sample_dir = paths["input"] / library_id
        matrix = sample_dir / "matrix.mtx.gz"
        features = sample_dir / "features.tsv.gz"
        barcodes = sample_dir / "barcodes.tsv.gz"
        missing = [str(path) for path in (matrix, features, barcodes) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{library_id}: missing files: {missing}")

        n_features = count_gzip_lines(features)
        n_cells = count_gzip_lines(barcodes)
        matrix_rows, matrix_cols, nnz = matrix_market_shape(matrix)
        if (matrix_rows, matrix_cols) != (n_features, n_cells):
            raise ValueError(
                f"{library_id}: matrix={matrix_rows}x{matrix_cols}, "
                f"features={n_features}, barcodes={n_cells}"
            )
        if full_gzip_test:
            for path in (matrix, features, barcodes):
                gzip_integrity(path)

        records.append(
            {
                "library_id": library_id,
                "group": metadata.loc[library_id, "group"],
                "features": n_features,
                "cells": n_cells,
                "nnz": nnz,
                "feature_sha256_uncompressed": uncompressed_sha256(features),
                "full_gzip_test": full_gzip_test,
            }
        )
        logger.info("%s: %d cells, %d features, %d nnz", library_id, n_cells, n_features, nnz)

    report = pd.DataFrame(records)
    if report["feature_sha256_uncompressed"].nunique() != 1:
        raise ValueError("features.tsv.gz content differs among libraries")
    output_dir = paths["results"] / "00_preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "input_validation.tsv"
    report.to_csv(output, sep="\t", index=False)
    logger.info("INPUT_VALIDATION_OK: %s", output)
    return output


def build_merged(
    config: dict,
    *,
    output: str | Path | None = None,
    allow_low_memory: bool = False,
) -> Path:
    require_compute_resources(config, allow_low_memory=allow_low_memory)
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

    output_path = Path(output).resolve() if output else paths["results"] / "01_merged_counts.h5ad"
    merged.write_h5ad(output_path, compression="gzip")
    pd.DataFrame(
        {"cells": merged.obs.groupby("library_id", observed=True).size()}
    ).to_csv(paths["results"] / "01_cells_by_library.tsv", sep="\t")
    logger.info("MERGE_OK: %s shape=%s", output_path, merged.shape)
    return output_path
