from __future__ import annotations

import gc
import gzip
import hashlib
from pathlib import Path

import anndata as ad
import numpy as np
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


def read_10x_sample(
    config: dict,
    library_id: str,
    *,
    make_unique: bool | None = None,
) -> ad.AnnData:
    """Read one Cell Ranger filtered MTX directory without downstream processing."""
    paths = project_paths(config)
    ingest = config["ingest"]
    sample_dir = paths["input"] / library_id
    if not sample_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {sample_dir}")

    # read_10x_mtx 只负责读取，不会 normalization 或 log1p；X 仍是 Cell Ranger UMI count。
    adata = sc.read_10x_mtx(
        sample_dir,
        var_names=ingest["var_names"],
        make_unique=bool(ingest["make_unique"]) if make_unique is None else make_unique,
        gex_only=bool(ingest["gene_expression_only"]),
        cache=False,
    )
    adata.X = sparse.csr_matrix(adata.X)
    return adata


def _values_are_integer(values: np.ndarray, *, chunk_size: int = 1_000_000) -> bool:
    """Check integer-valued counts in chunks to avoid one large temporary array."""
    for start in range(0, values.size, chunk_size):
        chunk = values[start : start + chunk_size]
        if not np.equal(chunk, np.floor(chunk)).all():
            return False
    return True


def _matrix_statistics(matrix: object) -> dict[str, object]:
    is_sparse = sparse.issparse(matrix)
    values = matrix.data if is_sparse else np.asarray(matrix).ravel()
    nnz = int(matrix.nnz) if is_sparse else int(np.count_nonzero(values))
    has_implicit_zeros = is_sparse and nnz < int(np.prod(matrix.shape))

    if values.size:
        stored_min = values.min().item()
        maximum = values.max().item()
        minimum = min(0, stored_min) if has_implicit_zeros else stored_min
        min_nonzero = stored_min
        has_negative = bool(stored_min < 0)
    else:
        minimum = maximum = min_nonzero = 0
        has_negative = False

    return {
        "matrix_type": type(matrix).__name__,
        "matrix_type_full": f"{type(matrix).__module__}.{type(matrix).__name__}",
        "dtype": str(matrix.dtype),
        "is_sparse": is_sparse,
        "nnz": nnz,
        # 稀疏矩阵没有显式存储零，因此全矩阵最小值通常是 0；同时单列最小非零值。
        "min_count": minimum,
        "min_nonzero_count": min_nonzero,
        "max_count": maximum,
        "has_negative": has_negative,
        "integer_valued": _values_are_integer(values),
    }


def _feature_information(adata: ad.AnnData) -> dict[str, object]:
    feature_id_column = next(
        (column for column in ("gene_ids", "feature_ids") if column in adata.var.columns),
        None,
    )
    if feature_id_column is None:
        raise KeyError(f"No feature ID column found in adata.var: {adata.var.columns.tolist()}")

    feature_ids = pd.Index(adata.var[feature_id_column].astype(str), name="feature_id")
    gene_symbols = pd.Index(adata.var_names.astype(str), name="gene_symbol")
    symbol_counts = pd.Series(gene_symbols, dtype="string").value_counts(dropna=False)
    duplicated = symbol_counts[symbol_counts > 1]
    examples = []
    for symbol, occurrences in duplicated.head(10).items():
        matching_ids = feature_ids[gene_symbols == symbol].tolist()
        examples.append(
            {
                "gene_symbol": symbol,
                "occurrences": int(occurrences),
                "feature_ids": ", ".join(matching_ids),
            }
        )

    return {
        "feature_id_column": feature_id_column,
        "feature_ids": feature_ids,
        "gene_symbols": gene_symbols,
        "feature_id_unique": bool(feature_ids.is_unique),
        "gene_symbols_present": bool((gene_symbols != "").all()),
        # duplicated_gene_symbols 指出现次数大于 1 的不同 symbol 数，而不是重复行数。
        "duplicated_gene_symbols": int(duplicated.size),
        "duplicated_gene_symbol_rows": int(duplicated.sum()),
        "duplicate_gene_symbol_extra_occurrences": int((duplicated - 1).sum()),
        "duplicate_examples": pd.DataFrame(examples),
    }


def _read_raw_feature_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", header=None, dtype=str, compression="gzip")
    names = ["feature_id", "gene_symbol", "feature_type"]
    table.columns = names[: table.shape[1]]
    return table


def inspect_filtered_inputs(
    config: dict,
    *,
    expected_cells: dict[str, int],
    expected_features: int,
    expected_total_cells: int,
    focus_sample: str = "Y_1",
) -> tuple[Path, Path]:
    """Read all filtered matrices sequentially and report structure without modifying them."""
    paths = project_paths(config)
    metadata = read_metadata(config)
    records: list[dict[str, object]] = []
    report_lines: list[str] = []
    anomalies: list[str] = []
    reference_feature_ids: np.ndarray | None = None
    feature_id_sets_identical = True
    feature_order_identical = True

    expected_samples = set(expected_cells)
    observed_samples = set(metadata.index)
    if observed_samples != expected_samples:
        anomalies.append(
            f"sample set mismatch: observed={sorted(observed_samples)}, "
            f"expected={sorted(expected_samples)}"
        )

    for library_id in metadata.index:
        # 本轮显式禁止 make_unique：重复 gene symbol 必须先被看见和报告。
        adata = read_10x_sample(config, library_id, make_unique=False)
        matrix_info = _matrix_statistics(adata.X)
        feature_info = _feature_information(adata)
        feature_ids = feature_info["feature_ids"].to_numpy(copy=True)

        if reference_feature_ids is None:
            reference_feature_ids = feature_ids
        else:
            feature_id_sets_identical &= set(feature_ids) == set(reference_feature_ids)
            feature_order_identical &= np.array_equal(feature_ids, reference_feature_ids)

        expected_n_cells = expected_cells.get(library_id)
        cell_count_matches = expected_n_cells is not None and adata.n_obs == expected_n_cells
        feature_count_matches = adata.n_vars == expected_features
        if not cell_count_matches:
            anomalies.append(f"{library_id}: cells={adata.n_obs}, expected={expected_n_cells}")
        if not feature_count_matches:
            anomalies.append(f"{library_id}: features={adata.n_vars}, expected={expected_features}")
        if matrix_info["has_negative"]:
            anomalies.append(f"{library_id}: negative count detected")
        if not matrix_info["integer_valued"]:
            anomalies.append(f"{library_id}: expression values are not integer-valued")
        if not adata.obs_names.is_unique:
            anomalies.append(f"{library_id}: duplicated barcodes detected")
        if not feature_info["feature_id_unique"]:
            anomalies.append(f"{library_id}: duplicated feature IDs detected")

        records.append(
            {
                "sample_id": library_id,
                "group": metadata.loc[library_id, "group"],
                "n_cells": adata.n_obs,
                "n_features": adata.n_vars,
                "matrix_type": matrix_info["matrix_type"],
                "dtype": matrix_info["dtype"],
                "nnz": matrix_info["nnz"],
                "min_count": matrix_info["min_count"],
                "min_nonzero_count": matrix_info["min_nonzero_count"],
                "max_count": matrix_info["max_count"],
                "has_negative": matrix_info["has_negative"],
                "integer_valued": matrix_info["integer_valued"],
                "barcode_unique": bool(adata.obs_names.is_unique),
                "feature_id_unique": feature_info["feature_id_unique"],
                "duplicated_gene_symbols": feature_info["duplicated_gene_symbols"],
                "cell_count_matches_expected": cell_count_matches,
                "feature_count_matches_expected": feature_count_matches,
            }
        )

        if library_id == focus_sample:
            # AnnData 按 observations x variables 存储，因此 shape 是 cells x genes。
            # X 是表达矩阵；obs 是以 barcode 为索引的细胞表；var 是 feature 注释表。
            # scRNA-seq 零值极多，稀疏矩阵只存非零 count，避免浪费内存。
            # 本轮不做 normalization/log1p，否则会改变原始 UMI 尺度并干扰输入核验。
            raw_features = _read_raw_feature_table(paths["input"] / library_id / "features.tsv.gz")
            raw_ids_match = np.array_equal(raw_features["feature_id"].to_numpy(), feature_ids)
            raw_symbols_match = np.array_equal(
                raw_features["gene_symbol"].to_numpy(),
                feature_info["gene_symbols"].to_numpy(),
            )
            report_lines.extend(
                [
                    f"=== {library_id}: AnnData basic structure ===",
                    "1. adata",
                    str(adata),
                    "",
                    "2. adata.shape",
                    str(adata.shape),
                    "",
                    "3. type(adata.X)",
                    str(type(adata.X)),
                    "",
                    "4. adata.X.dtype",
                    str(adata.X.dtype),
                    "",
                    "5. adata.obs.head()",
                    adata.obs.head().to_string(),
                    "",
                    "6. adata.var.head()",
                    adata.var.head().to_string(),
                    "",
                    "7. adata.obs_names[:5]",
                    repr(adata.obs_names[:5].tolist()),
                    "",
                    "8. adata.var_names[:10]",
                    repr(adata.var_names[:10].tolist()),
                    "",
                    "=== Matrix checks ===",
                    f"cells={adata.n_obs}",
                    f"features={adata.n_vars}",
                    f"is_sparse={matrix_info['is_sparse']}",
                    f"matrix_type={matrix_info['matrix_type_full']}",
                    f"dtype={matrix_info['dtype']}",
                    f"min_count={matrix_info['min_count']}",
                    f"min_nonzero_count={matrix_info['min_nonzero_count']}",
                    f"max_count={matrix_info['max_count']}",
                    f"has_negative={matrix_info['has_negative']}",
                    f"integer_valued={matrix_info['integer_valued']}",
                    f"nnz={matrix_info['nnz']}",
                    f"barcode_unique={adata.obs_names.is_unique}",
                    f"feature_id_unique={feature_info['feature_id_unique']}",
                    f"gene_symbols_present={feature_info['gene_symbols_present']}",
                    f"duplicated_gene_symbols={feature_info['duplicated_gene_symbols']}",
                    "duplicate_gene_symbol_extra_occurrences="
                    f"{feature_info['duplicate_gene_symbol_extra_occurrences']}",
                    "",
                    "=== Raw features.tsv.gz head ===",
                    raw_features.head().to_string(index=False),
                    "",
                    "=== Feature storage mapping ===",
                    f"Scanpy var_names source={config['ingest']['var_names']}",
                    f"feature ID column in adata.var={feature_info['feature_id_column']}",
                    f"raw feature IDs match adata.var IDs={raw_ids_match}",
                    f"raw gene symbols match adata.var_names={raw_symbols_match}",
                    "",
                    "=== Duplicated gene symbol examples (up to 10) ===",
                    feature_info["duplicate_examples"].to_string(index=False)
                    if not feature_info["duplicate_examples"].empty
                    else "None",
                    "",
                ]
            )

        # 每次只保留一个样本，避免把 9 个大稀疏矩阵同时放入内存。
        del adata
        gc.collect()

    summary = pd.DataFrame(records)
    total_cells = int(summary["n_cells"].sum())
    all_feature_counts_match = bool(summary["feature_count_matches_expected"].all())
    all_cell_counts_match = bool(summary["cell_count_matches_expected"].all())
    total_cells_match = total_cells == expected_total_cells
    if not feature_id_sets_identical:
        anomalies.append("feature ID sets differ among libraries")
    if not feature_order_identical:
        anomalies.append("feature ID order differs among libraries")
    if not total_cells_match:
        anomalies.append(f"total cells={total_cells}, expected={expected_total_cells}")

    report_lines.extend(
        [
            "=== Nine-library summary ===",
            summary.to_string(index=False),
            "",
            "=== Cross-library consistency ===",
            f"all_feature_counts_equal_{expected_features}={all_feature_counts_match}",
            f"feature_id_sets_identical={feature_id_sets_identical}",
            f"feature_order_identical={feature_order_identical}",
            f"all_cell_counts_match_expected={all_cell_counts_match}",
            f"total_cells={total_cells}",
            f"total_cells_match_{expected_total_cells}={total_cells_match}",
            "",
            "=== Anomalies ===",
            "None" if not anomalies else "\n".join(f"- {item}" for item in anomalies),
            "",
            "=== Chinese explanation ===",
            "adata.X 是 cells x genes 的表达矩阵；每一行对应一个 barcode/cell，"
            "每一列对应一个 feature/gene。",
            "adata.obs 是细胞注释表，行索引是 barcode；刚读取的 10x 矩阵通常还没有额外细胞列。",
            "adata.var 是 feature 注释表；本项目以 gene symbol 作为 var_names，"
            "Ensembl/feature ID 保存在 gene_ids 列。",
            "adata.shape 因而按 AnnData 约定表示 n_cells x n_genes。",
            "scRNA-seq 中绝大多数 cell-gene 组合为 0；稀疏矩阵只存非零值，可显著节省内存。",
            "Cell Ranger filtered count matrix 的 matrix.mtx 保存原始 UMI 计数；"
            "Scanpy 可能用 float32 承载它们，但这里额外检查数值是否仍为非负整数值。",
            "本轮必须保留输入原貌；normalization 或 log1p 会改变计数尺度，"
            "使输入完整性、整数性和后续原始 counts 保存无法被清楚核验。",
        ]
    )

    output_dir = paths["results"] / "00_preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "scanpy_input_summary.tsv"
    report_path = output_dir / "input_structure_report.txt"
    summary.to_csv(summary_path, sep="\t", index=False)
    report_text = "\n".join(report_lines)
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"SUMMARY_TABLE={summary_path}")
    print(f"STRUCTURE_REPORT={report_path}")
    return summary_path, report_path


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
        logger.info("Reading %s", paths["input"] / library_id)
        adata = read_10x_sample(config, library_id)
        adata.obs_names = [
            f"{library_id}{ingest['barcode_separator']}{barcode}" for barcode in adata.obs_names
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
    pd.DataFrame({"cells": merged.obs.groupby("library_id", observed=True).size()}).to_csv(
        paths["results"] / "01_cells_by_library.tsv", sep="\t"
    )
    logger.info("MERGE_OK: %s shape=%s", output_path, merged.shape)
    return output_path
