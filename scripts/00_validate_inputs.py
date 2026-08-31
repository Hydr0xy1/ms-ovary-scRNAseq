from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path

import pandas as pd

from _project import DEFAULT_CONFIG, load_config, project_paths, read_metadata, setup_logging


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-memory audit of 10x MTX inputs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--full-gzip-test",
        action="store_true",
        help="Read every compressed byte and verify CRC (safe but slow in no-card mode).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    paths = project_paths(config)
    metadata = read_metadata(config)
    logger = setup_logging("00_validate_inputs", config)
    records: list[dict[str, object]] = []

    for library_id in metadata.index:
        sample_dir = paths["input"] / library_id
        matrix = sample_dir / "matrix.mtx.gz"
        features = sample_dir / "features.tsv.gz"
        barcodes = sample_dir / "barcodes.tsv.gz"
        missing = [str(p) for p in (matrix, features, barcodes) if not p.exists()]
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
        if args.full_gzip_test:
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
                "full_gzip_test": args.full_gzip_test,
            }
        )
        logger.info("%s: %d cells, %d features, %d nnz", library_id, n_cells, n_features, nnz)

    report = pd.DataFrame(records)
    if report["feature_sha256_uncompressed"].nunique() != 1:
        raise ValueError("features.tsv.gz content differs among libraries")
    output_dir = paths["results"] / "00_preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "input_validation.tsv", sep="\t", index=False)
    logger.info("INPUT_VALIDATION_OK: %s", output_dir / "input_validation.tsv")


if __name__ == "__main__":
    main()
