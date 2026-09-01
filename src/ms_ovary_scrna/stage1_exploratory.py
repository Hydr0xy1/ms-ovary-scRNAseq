from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.metrics import normalized_mutual_info_score

from .preprocessing import (
    compute_hvg_pca,
    compute_unintegrated_neighbors_umap,
    normalize_log1p_preserving_counts,
    select_batch_aware_hvgs,
)
from .project import project_paths, require_compute_resources, setup_logging
from .qc_sensitivity import _markdown_table, mad_masks_and_thresholds

EXPECTED_LIBRARIES = 9
EXPECTED_GROUPS = 3
STRESS_GENES = {
    "Atf3",
    "Ddit3",
    "Egr1",
    "Fos",
    "Fosb",
    "Gadd45a",
    "Gadd45b",
    "Hmox1",
    "Hspa1a",
    "Hspa1b",
    "Hsp90aa1",
    "Ier2",
    "Ier3",
    "Jun",
    "Junb",
    "Jund",
    "Nfkbia",
}


def _percentage(count: int | float, total: int) -> float:
    return 100.0 * float(count) / total if total else float("nan")


def _bool_array(values: pd.Series) -> np.ndarray:
    return values.fillna(False).astype(bool).to_numpy()


def _entity_frames(obs: pd.DataFrame):
    for library_id, frame in obs.groupby("library_id", observed=True, sort=False):
        yield "library", str(library_id), str(frame["group"].iloc[0]), frame
    for group, frame in obs.groupby("group", observed=True, sort=False):
        yield "group", str(group), str(group), frame
    yield "overall", "all", None, obs


def annotate_stage1_qc_flags(
    adata: ad.AnnData,
    config: dict,
) -> pd.DataFrame:
    """Add the agreed stage-1 flags without overwriting existing QC columns."""
    stage = config["stage1_exploratory"]
    min_genes = int(stage["min_genes_absolute"])
    low_gene_mads = float(stage["low_genes_n_mads"])
    moderate_lower = float(stage["mt_moderate_lower_pct"])
    extreme_mt = float(stage["mt_extreme_pct"])

    adata.obs["qc_low_genes_absolute"] = adata.obs["n_genes_by_counts"] < min_genes
    adata.obs["qc_low_genes_5mad"] = False
    threshold_records: list[dict[str, Any]] = []
    library_values = adata.obs["library_id"].astype(str)
    for library_id in library_values.unique():
        mask = (library_values == library_id).to_numpy()
        frame = adata.obs.loc[mask]
        result = mad_masks_and_thresholds(
            frame["n_genes_by_counts"],
            low_gene_mads,
            log1p=True,
        )
        adata.obs.loc[mask, "qc_low_genes_5mad"] = result["low_mask"]
        existing = (
            _bool_array(frame["qc_genes_outlier"])
            if "qc_genes_outlier" in frame
            else np.zeros(len(frame), dtype=bool)
        )
        threshold_records.append(
            {
                "library_id": library_id,
                "group": str(frame["group"].iloc[0]),
                "n_cells": len(frame),
                "low_genes_n_mads": low_gene_mads,
                "low_gene_threshold": result["low_threshold"],
                "qc_low_genes_5mad_count": int(result["low_mask"].sum()),
                "existing_qc_genes_outlier_count": int(existing.sum()),
                "mismatch_with_existing_qc_genes_outlier": int(
                    np.logical_xor(result["low_mask"], existing).sum()
                ),
            }
        )

    mt = adata.obs["pct_counts_mt"].astype(float)
    adata.obs["qc_mt_extreme"] = mt > extreme_mt
    adata.obs["qc_mt_moderate"] = (mt > moderate_lower) & (mt <= extreme_mt)
    adata.obs["qc_doublet_auto"] = adata.obs["predicted_doublet"].fillna(False).astype(bool)
    adata.obs["qc_low_quality_strong"] = adata.obs["qc_low_genes_absolute"] | (
        adata.obs["qc_mt_extreme"] & adata.obs["qc_low_genes_5mad"]
    )
    adata.obs["retained_extreme_mt"] = (
        adata.obs["qc_mt_extreme"] & ~adata.obs["qc_low_quality_strong"]
    )
    return pd.DataFrame(threshold_records)


def mark_high_doublet_score_top_fraction(
    obs: pd.DataFrame,
    fraction: float,
) -> pd.DataFrame:
    """Mark an exact ceil(fraction*n) highest-score set within each library."""
    if not 0 < fraction < 1:
        raise ValueError("Doublet-score fraction must be between zero and one")
    obs["qc_high_doublet_score_top1pct"] = False
    records: list[dict[str, Any]] = []
    for library_id, frame in obs.groupby("library_id", observed=True, sort=False):
        n_select = max(1, math.ceil(len(frame) * fraction))
        ordering = pd.DataFrame(
            {
                "doublet_score": frame["doublet_score"].astype(float),
                "cell_barcode": frame.index.astype(str),
            },
            index=frame.index,
        ).sort_values(
            ["doublet_score", "cell_barcode"],
            ascending=[False, True],
            kind="mergesort",
        )
        selected = ordering.index[:n_select]
        obs.loc[selected, "qc_high_doublet_score_top1pct"] = True
        records.append(
            {
                "library_id": str(library_id),
                "group": str(frame["group"].iloc[0]),
                "n_cells": len(frame),
                "requested_fraction": fraction,
                "selected_cells": n_select,
                "selected_pct": _percentage(n_select, len(frame)),
                "minimum_selected_score": float(ordering.iloc[n_select - 1]["doublet_score"]),
                "maximum_unselected_score": (
                    float(ordering.iloc[n_select]["doublet_score"])
                    if n_select < len(ordering)
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(records)


def stage1_filter_summary(obs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for entity_type, entity_id, group, frame in _entity_frames(obs):
        removed = _bool_array(frame["qc_low_quality_strong"])
        retained = frame.loc[~removed]
        record: dict[str, Any] = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "group": group,
            "cells_before": len(frame),
            "would_remove": int(removed.sum()),
            "cells_after": len(retained),
            "removed_pct": _percentage(int(removed.sum()), len(frame)),
            "retained_pct": _percentage(len(retained), len(frame)),
        }
        for metric in ("total_counts", "n_genes_by_counts", "pct_counts_mt"):
            record[f"median_{metric}_before"] = float(frame[metric].median())
            record[f"median_{metric}_after"] = float(retained[metric].median())
        records.append(record)
    return pd.DataFrame(records)


def stage1_filter_reason_overlap(obs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for entity_type, entity_id, group, frame in _entity_frames(obs):
        absolute = _bool_array(frame["qc_low_genes_absolute"])
        low_5mad = _bool_array(frame["qc_low_genes_5mad"])
        extreme = _bool_array(frame["qc_mt_extreme"])
        extreme_and_low = extreme & low_5mad
        strong = _bool_array(frame["qc_low_quality_strong"])
        masks = {
            "qc_low_genes_absolute": absolute,
            "qc_low_genes_5mad_diagnostic": low_5mad,
            "qc_mt_extreme_diagnostic": extreme,
            "qc_mt_moderate_diagnostic": _bool_array(frame["qc_mt_moderate"]),
            "qc_doublet_auto_diagnostic": _bool_array(frame["qc_doublet_auto"]),
            "extreme_mt_and_low_genes_5mad": extreme_and_low,
            "absolute_low_genes_and_extreme_low5mad": absolute & extreme_and_low,
            "absolute_low_genes_only_filter_reason": absolute & ~extreme_and_low,
            "extreme_low5mad_only_filter_reason": extreme_and_low & ~absolute,
            "qc_low_quality_strong": strong,
        }
        for reason, mask in masks.items():
            count = int(mask.sum())
            records.append(
                {
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "group": group,
                    "reason": reason,
                    "n_cells": len(frame),
                    "flagged_cells": count,
                    "flagged_pct": _percentage(count, len(frame)),
                }
            )
    return pd.DataFrame(records)


def sparse_matrix_audit(matrix: object) -> dict[str, Any]:
    if not sparse.issparse(matrix):
        raise TypeError("Raw counts must remain a sparse matrix")
    csr = sparse.csr_matrix(matrix)
    data = csr.data
    digest = hashlib.sha256()
    for array in (csr.data, csr.indices, csr.indptr):
        contiguous = np.ascontiguousarray(array)
        digest.update(memoryview(contiguous).cast("B"))
    integer_values = bool(np.isfinite(data).all() and np.equal(data, np.rint(data)).all())
    return {
        "matrix_type": type(matrix).__name__,
        "is_csr": sparse.isspmatrix_csr(matrix),
        "dtype": str(matrix.dtype),
        "shape": [int(csr.shape[0]), int(csr.shape[1])],
        "nnz": int(csr.nnz),
        "minimum_nonzero": float(data.min()) if data.size else 0.0,
        "maximum": float(data.max()) if data.size else 0.0,
        "sum": float(data.sum(dtype=np.float64)),
        "integer_values": integer_values,
        "sha256_sparse_arrays": digest.hexdigest(),
    }


def validate_stage1_object(
    adata: ad.AnnData,
    *,
    counts_layer: str,
    expected_shape: tuple[int, int],
    expected_counts_audit: dict[str, Any],
) -> dict[str, Any]:
    if adata.shape != expected_shape:
        raise ValueError(f"Unexpected stage-1 shape: {adata.shape} != {expected_shape}")
    if adata.obs["library_id"].nunique() != EXPECTED_LIBRARIES:
        raise ValueError("Stage-1 object does not contain nine libraries")
    if adata.obs["group"].nunique() != EXPECTED_GROUPS:
        raise ValueError("Stage-1 object does not contain three groups")
    audit = sparse_matrix_audit(adata.layers[counts_layer])
    if not audit["is_csr"]:
        raise TypeError("The saved counts layer is not CSR")
    if not audit["integer_values"]:
        raise ValueError("The saved counts layer contains non-integer values")
    for key in ("shape", "nnz", "sum", "sha256_sparse_arrays"):
        if audit[key] != expected_counts_audit[key]:
            raise ValueError(f"Counts validation mismatch for {key}")
    return {
        "n_cells": adata.n_obs,
        "n_features": adata.n_vars,
        "n_samples": int(adata.obs["library_id"].nunique()),
        "n_libraries": int(adata.obs["library_id"].nunique()),
        "n_groups": int(adata.obs["group"].nunique()),
        **{f"counts_{key}": value for key, value in audit.items()},
    }


def _atomic_write_h5ad(adata: ad.AnnData, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.h5ad")
    adata.write_h5ad(temporary, compression="gzip")
    temporary.replace(output)


def hvg_summary_table(adata: ad.AnnData) -> pd.DataFrame:
    selected = adata.var[adata.var["highly_variable"].astype(bool)].copy()
    selected.insert(0, "gene", selected.index.astype(str))
    wanted = [
        "gene",
        "gene_ids",
        "feature_types",
        "mt",
        "ribo",
        "highly_variable_rank",
        "highly_variable_nbatches",
        "highly_variable_intersection",
        "means",
        "variances",
        "variances_norm",
    ]
    columns = [column for column in wanted if column in selected]
    result = selected[columns].copy()
    result["stress_gene"] = result["gene"].isin(STRESS_GENES)
    return result.sort_values(
        ["highly_variable_nbatches", "highly_variable_rank", "gene"],
        ascending=[False, True, True],
        kind="mergesort",
    )


def pca_tables(
    adata: ad.AnnData,
    hvg: ad.AnnData,
    *,
    top_n: int = 30,
    loading_pcs: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variance = np.asarray(adata.uns["pca"]["variance"], dtype=float)
    ratios = np.asarray(adata.uns["pca"]["variance_ratio"], dtype=float)
    cumulative = np.cumsum(ratios)
    loadings = np.asarray(hvg.varm["PCs"], dtype=float)
    genes = hvg.var_names.astype(str).to_numpy()
    mt = hvg.var["mt"].astype(bool).to_numpy() if "mt" in hvg.var else np.zeros(hvg.n_vars, bool)
    ribo = (
        hvg.var["ribo"].astype(bool).to_numpy() if "ribo" in hvg.var else np.zeros(hvg.n_vars, bool)
    )

    loading_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    for pc_index, (variance_value, ratio, cumulative_ratio) in enumerate(
        zip(variance, ratios, cumulative, strict=True),
        start=1,
    ):
        values = loadings[:, pc_index - 1]
        absolute_order = np.argsort(np.abs(values))[-top_n:][::-1]
        positive_order = np.argsort(values)[-10:][::-1]
        negative_order = np.argsort(values)[:10]
        summary_records.append(
            {
                "pc": f"PC{pc_index}",
                "explained_variance": variance_value,
                "explained_variance_ratio": ratio,
                "cumulative_variance_ratio": cumulative_ratio,
                "top_positive_genes": ";".join(genes[positive_order]),
                "top_negative_genes": ";".join(genes[negative_order]),
                "top_absolute_genes": ";".join(genes[absolute_order[:10]]),
            }
        )
        if pc_index <= loading_pcs:
            for rank, gene_index in enumerate(absolute_order, start=1):
                gene = genes[gene_index]
                if mt[gene_index]:
                    hint = "mitochondrial"
                elif ribo[gene_index]:
                    hint = "ribosomal"
                elif gene in STRESS_GENES:
                    hint = "stress_immediate_early"
                else:
                    hint = "other"
                loading_records.append(
                    {
                        "pc": f"PC{pc_index}",
                        "absolute_rank": rank,
                        "gene": gene,
                        "loading": values[gene_index],
                        "absolute_loading": abs(values[gene_index]),
                        "direction": "positive" if values[gene_index] >= 0 else "negative",
                        "mt": bool(mt[gene_index]),
                        "ribo": bool(ribo[gene_index]),
                        "stress_gene": gene in STRESS_GENES,
                        "program_hint": hint,
                    }
                )
    return pd.DataFrame(summary_records), pd.DataFrame(loading_records)


def run_leiden_resolutions(
    adata: ad.AnnData,
    *,
    resolutions: list[float],
    primary_resolution: float,
    neighbors_key: str,
    random_state: int,
) -> None:
    for resolution in resolutions:
        key = f"leiden_{resolution:g}"
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=key,
            neighbors_key=neighbors_key,
            random_state=random_state,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
    primary_key = f"leiden_{primary_resolution:g}"
    adata.obs["leiden"] = adata.obs[primary_key].copy()


def _composition_json(frame: pd.DataFrame, column: str) -> str:
    counts = frame[column].astype(str).value_counts().sort_index()
    return json.dumps({key: int(value) for key, value in counts.items()}, sort_keys=True)


def _entropy(values: pd.Series) -> float:
    proportions = values.astype(str).value_counts(normalize=True).to_numpy()
    if len(proportions) <= 1:
        return 0.0
    return float(-(proportions * np.log(proportions)).sum() / np.log(len(proportions)))


def leiden_qc_tables(
    obs: pd.DataFrame,
    resolutions: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    composition: list[dict[str, Any]] = []
    for resolution in resolutions:
        key = f"leiden_{resolution:g}"
        for cluster, frame in obs.groupby(key, observed=True, sort=False):
            libraries = frame["library_id"].astype(str).value_counts()
            records.append(
                {
                    "cluster_key": key,
                    "resolution": resolution,
                    "cluster_id": str(cluster),
                    "n_cells": len(frame),
                    "pct_all_cells": _percentage(len(frame), len(obs)),
                    "n_libraries": int(libraries.size),
                    "library_entropy_normalized": _entropy(frame["library_id"]),
                    "maximum_library_pct": _percentage(int(libraries.max()), len(frame)),
                    "library_counts_json": _composition_json(frame, "library_id"),
                    "group_counts_json": _composition_json(frame, "group"),
                    "median_total_counts": float(frame["total_counts"].median()),
                    "median_n_genes_by_counts": float(frame["n_genes_by_counts"].median()),
                    "median_pct_counts_mt": float(frame["pct_counts_mt"].median()),
                    "median_doublet_score": float(frame["doublet_score"].median()),
                    "predicted_doublet_pct": 100 * float(frame["qc_doublet_auto"].mean()),
                    "mt_extreme_pct": 100 * float(frame["qc_mt_extreme"].mean()),
                    "mt_moderate_pct": 100 * float(frame["qc_mt_moderate"].mean()),
                    "low_genes_5mad_pct": 100 * float(frame["qc_low_genes_5mad"].mean()),
                    "high_doublet_top1pct_pct": 100
                    * float(frame["qc_high_doublet_score_top1pct"].mean()),
                }
            )
            for dimension in ("library_id", "group"):
                counts = frame[dimension].astype(str).value_counts().sort_index()
                for value, count in counts.items():
                    composition.append(
                        {
                            "cluster_key": key,
                            "resolution": resolution,
                            "cluster_id": str(cluster),
                            "dimension": dimension,
                            "value": value,
                            "count": int(count),
                            "pct_cluster": _percentage(int(count), len(frame)),
                        }
                    )
    return pd.DataFrame(records), pd.DataFrame(composition)


def qc_candidate_cluster_distribution(
    obs: pd.DataFrame,
    *,
    cluster_key: str,
) -> pd.DataFrame:
    candidates = {
        "retained_extreme_mt": _bool_array(obs["retained_extreme_mt"]),
        "scrublet_auto_doublet": _bool_array(obs["qc_doublet_auto"]),
        "high_doublet_score": _bool_array(obs["qc_high_doublet_score_top1pct"]),
    }
    records: list[dict[str, Any]] = []
    cluster_values = obs[cluster_key].astype(str)
    for candidate_name, candidate_mask in candidates.items():
        total_candidate = int(candidate_mask.sum())
        global_fraction = total_candidate / len(obs)
        for cluster in ["ALL", *cluster_values.drop_duplicates().tolist()]:
            cluster_mask = (
                np.ones(len(obs), dtype=bool)
                if cluster == "ALL"
                else (cluster_values == cluster).to_numpy()
            )
            selected_mask = candidate_mask & cluster_mask
            selected = obs.loc[selected_mask]
            cluster_n = int(cluster_mask.sum())
            selected_n = len(selected)
            cluster_fraction = selected_n / cluster_n if cluster_n else float("nan")
            records.append(
                {
                    "candidate_set": candidate_name,
                    "cluster_key": cluster_key,
                    "cluster_id": cluster,
                    "candidate_cells_all": total_candidate,
                    "cluster_cells": cluster_n,
                    "candidate_cells_in_cluster": selected_n,
                    "pct_of_candidate_set": _percentage(selected_n, total_candidate),
                    "pct_cluster_candidate": 100 * cluster_fraction,
                    "enrichment_vs_global": (
                        cluster_fraction / global_fraction if global_fraction else float("nan")
                    ),
                    "library_counts_json": (
                        _composition_json(selected, "library_id") if selected_n else "{}"
                    ),
                    "group_counts_json": (
                        _composition_json(selected, "group") if selected_n else "{}"
                    ),
                    "median_total_counts": (
                        float(selected["total_counts"].median()) if selected_n else float("nan")
                    ),
                    "median_n_genes_by_counts": (
                        float(selected["n_genes_by_counts"].median())
                        if selected_n
                        else float("nan")
                    ),
                    "median_pct_counts_mt": (
                        float(selected["pct_counts_mt"].median()) if selected_n else float("nan")
                    ),
                    "median_doublet_score": (
                        float(selected["doublet_score"].median()) if selected_n else float("nan")
                    ),
                }
            )
    return pd.DataFrame(records)


def neighbor_mixing_summary(
    adata: ad.AnnData,
    *,
    neighbors_key: str,
    resolutions: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    distance_key = f"{neighbors_key}_distances"
    distances = sparse.csr_matrix(adata.obsp[distance_key])
    records: list[dict[str, Any]] = []
    for dimension in ("library_id", "group"):
        labels = adata.obs[dimension].astype(str).to_numpy()
        same_fraction = np.zeros(adata.n_obs, dtype=float)
        for row in range(adata.n_obs):
            start, end = distances.indptr[row], distances.indptr[row + 1]
            neighbors = distances.indices[start:end]
            same_fraction[row] = (
                float(np.mean(labels[neighbors] == labels[row])) if len(neighbors) else np.nan
            )
        for entity_id in ["all", *pd.unique(labels).tolist()]:
            mask = np.ones(adata.n_obs, dtype=bool) if entity_id == "all" else labels == entity_id
            values = same_fraction[mask]
            expected = (
                float(np.square(pd.Series(labels).value_counts(normalize=True)).sum())
                if entity_id == "all"
                else float(np.mean(labels == entity_id))
            )
            records.append(
                {
                    "dimension": dimension,
                    "entity_id": entity_id,
                    "n_cells": int(mask.sum()),
                    "mean_same_neighbor_fraction": float(np.nanmean(values)),
                    "median_same_neighbor_fraction": float(np.nanmedian(values)),
                    "p90_same_neighbor_fraction": float(np.nanquantile(values, 0.9)),
                    "expected_random_fraction": expected,
                    "excess_over_random": float(np.nanmean(values) - expected),
                }
            )

    association: list[dict[str, Any]] = []
    for resolution in resolutions:
        cluster_key = f"leiden_{resolution:g}"
        clusters = adata.obs[cluster_key].astype(str)
        for dimension in ("library_id", "group"):
            association.append(
                {
                    "cluster_key": cluster_key,
                    "resolution": resolution,
                    "metadata_dimension": dimension,
                    "normalized_mutual_information": float(
                        normalized_mutual_info_score(
                            adata.obs[dimension].astype(str),
                            clusters,
                        )
                    ),
                }
            )
    return pd.DataFrame(records), pd.DataFrame(association)


def _plot_pca_scree(pca_summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(1, len(pca_summary) + 1)
    axes[0].plot(x, 100 * pca_summary["explained_variance_ratio"], marker="o", markersize=3)
    axes[0].set(xlabel="Principal component", ylabel="Explained variance (%)", title="PCA scree")
    axes[1].plot(x, 100 * pca_summary["cumulative_variance_ratio"], marker="o", markersize=3)
    axes[1].set(
        xlabel="Principal component",
        ylabel="Cumulative explained variance (%)",
        title="Cumulative variance",
    )
    for axis in axes:
        axis.set_xlim(1, len(pca_summary))
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_embedding_panels(
    adata: ad.AnnData,
    panels: list[tuple[str, str, str | None]],
    output: Path,
) -> None:
    ncols = 2 if len(panels) == 4 else min(3, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    point_size = max(1.0, 120000 / adata.n_obs)
    for axis, (key, title, vmax) in zip(axes.flat, panels, strict=False):
        kwargs: dict[str, Any] = {}
        if vmax is not None:
            kwargs.update({"vmin": "p1", "vmax": vmax, "color_map": "viridis"})
        sc.pl.embedding(
            adata,
            basis="umap_unintegrated",
            color=key,
            title=title,
            frameon=False,
            size=point_size,
            sort_order=True,
            show=False,
            ax=axis,
            **kwargs,
        )
    for axis in axes.flat[len(panels) :]:
        axis.set_visible(False)
    figure.suptitle("Unintegrated / pre-integration UMAP", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _file_signature(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def build_stage1_report(
    *,
    filter_summary: pd.DataFrame,
    reason_summary: pd.DataFrame,
    validation: pd.DataFrame,
    hvg_summary: pd.DataFrame,
    pca_summary: pd.DataFrame,
    mixing: pd.DataFrame,
    association: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    candidates: pd.DataFrame,
) -> str:
    overall = filter_summary[
        (filter_summary["entity_type"] == "overall") & (filter_summary["entity_id"] == "all")
    ].iloc[0]
    group_view = filter_summary[filter_summary["entity_type"] == "group"]
    reasons = reason_summary[
        (reason_summary["entity_type"] == "overall")
        & reason_summary["reason"].isin(
            [
                "qc_low_genes_absolute",
                "extreme_mt_and_low_genes_5mad",
                "absolute_low_genes_and_extreme_low5mad",
                "qc_low_quality_strong",
            ]
        )
    ]
    hvg_counts = pd.DataFrame(
        {
            "metric": ["selected_hvg", "mt_hvg", "ribo_hvg", "stress_hvg"],
            "count": [
                len(hvg_summary),
                int(hvg_summary.get("mt", pd.Series(False, index=hvg_summary.index)).sum()),
                int(hvg_summary.get("ribo", pd.Series(False, index=hvg_summary.index)).sum()),
                int(hvg_summary["stress_gene"].sum()),
            ],
        }
    )
    validation_view = validation[
        [
            "object",
            "X_semantics",
            "counts_layer_semantics",
            "n_cells",
            "n_features",
            "n_libraries",
            "n_groups",
            "counts_matrix_type",
            "counts_is_csr",
            "counts_dtype",
            "counts_nnz",
            "counts_integer_values",
        ]
    ]
    candidate_all = candidates[candidates["cluster_id"] == "ALL"]
    primary_clusters = cluster_summary[cluster_summary["cluster_key"] == "leiden_0.5"]
    lines = [
        "# Stage-1 QC and unintegrated exploratory report",
        "",
        "本报告只记录保守一级QC和未整合探索结果；没有执行Harmony、正式注释、cluster删除或差异分析。",
        "",
        "## Stage-1 filter",
        "",
        f"一级QC删除 {int(overall['would_remove'])}/{int(overall['cells_before'])} "
        f"({overall['removed_pct']:.3f}%)，保留 {int(overall['cells_after'])} 个细胞。",
        "",
        _markdown_table(
            group_view[
                ["entity_id", "cells_before", "would_remove", "cells_after", "retained_pct"]
            ].rename(columns={"entity_id": "group"})
        ),
        "",
        "### Filter-reason overlap",
        "",
        _markdown_table(reasons[["reason", "flagged_cells", "flagged_pct"]]),
        "",
        "## Saved-object validation",
        "",
        _markdown_table(validation_view),
        "",
        "`X` in the exploratory object is log1p library-size normalized expression; "
        "`layers['counts']` remains CSR raw integer UMI counts.",
        "",
        "## Batch-aware HVG",
        "",
        _markdown_table(hvg_counts),
        "",
        "## PCA",
        "",
        _markdown_table(
            pca_summary.head(10)[
                [
                    "pc",
                    "explained_variance_ratio",
                    "cumulative_variance_ratio",
                    "top_absolute_genes",
                ]
            ]
        ),
        "",
        "## Library/group structure before integration",
        "",
        _markdown_table(mixing[mixing["entity_id"] == "all"]),
        "",
        _markdown_table(association),
        "",
        "## Leiden 0.5 QC range",
        "",
        _markdown_table(
            primary_clusters[
                [
                    "cluster_id",
                    "n_cells",
                    "n_libraries",
                    "maximum_library_pct",
                    "median_total_counts",
                    "median_n_genes_by_counts",
                    "median_pct_counts_mt",
                    "median_doublet_score",
                    "predicted_doublet_pct",
                    "mt_extreme_pct",
                    "low_genes_5mad_pct",
                ]
            ]
        ),
        "",
        "## QC candidate sets",
        "",
        _markdown_table(
            candidate_all[
                [
                    "candidate_set",
                    "candidate_cells_all",
                    "median_total_counts",
                    "median_n_genes_by_counts",
                    "median_pct_counts_mt",
                    "median_doublet_score",
                ]
            ]
        ),
        "",
        "## Scope guard",
        "",
        "本流程没有执行Harmony、BBKNN、CellTypist、正式细胞注释、cluster删除、DEG、"
        "pseudobulk、GSEA、trajectory、CellChat或SCENIC。",
    ]
    return "\n".join(lines) + "\n"


def run_stage1_exploratory(
    config: dict,
    input_path: str | Path,
    *,
    allow_low_memory: bool = False,
) -> Path:
    require_compute_resources(config, allow_low_memory=allow_low_memory)
    paths = project_paths(config)
    logger = setup_logging("02_stage1_exploratory", config)
    stage = config["stage1_exploratory"]
    preprocess = config["preprocess"]
    counts_layer = config["ingest"]["counts_layer"]
    seed = int(config["project"]["random_seed"])
    input_path = Path(input_path)
    stage1_output = paths["results"] / "02_qc_filtered_stage1.h5ad"
    exploratory_dir = paths["results"] / "exploratory"
    figure_dir = paths["figures"] / "exploratory"
    exploratory_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    exploratory_output = exploratory_dir / "03_stage1_unintegrated_exploratory.h5ad"
    forbidden_filtered = paths["results"] / "02_qc_filtered.h5ad"
    input_signature_before = _file_signature(input_path)
    forbidden_signature_before = _file_signature(forbidden_filtered)

    logger.info("Reading annotated QC object: %s", input_path)
    annotated = sc.read_h5ad(input_path)
    required_obs = {
        "library_id",
        "group",
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
        "doublet_score",
        "predicted_doublet",
    }
    missing = sorted(required_obs - set(annotated.obs.columns))
    if missing:
        raise KeyError(f"Annotated object is missing obs columns: {missing}")
    if counts_layer not in annotated.layers:
        raise KeyError(f"Missing counts layer: {counts_layer}")

    logger.info("Annotating the agreed stage-1 QC flags")
    low_gene_thresholds = annotate_stage1_qc_flags(annotated, config)
    filter_summary = stage1_filter_summary(annotated.obs)
    reason_summary = stage1_filter_reason_overlap(annotated.obs)
    keep = ~_bool_array(annotated.obs["qc_low_quality_strong"])
    filtered = annotated[keep].copy()
    filtered.layers[counts_layer] = sparse.csr_matrix(filtered.layers[counts_layer])
    filtered.X = filtered.layers[counts_layer].copy()
    high_score_thresholds = mark_high_doublet_score_top_fraction(
        filtered.obs,
        float(stage["high_doublet_score_fraction"]),
    )
    expected_stage1_shape = filtered.shape
    expected_counts_audit = sparse_matrix_audit(filtered.layers[counts_layer])
    filtered.uns["stage1_qc"] = {
        "input": str(input_path),
        "rule": (
            f"n_genes_by_counts < {int(stage['min_genes_absolute'])} OR "
            f"(pct_counts_mt > {float(stage['mt_extreme_pct']):g} AND "
            f"low genes at {float(stage['low_genes_n_mads']):g} MAD)"
        ),
        "scrublet_used_for_filtering": False,
        "mt_3mad_used_for_filtering": False,
        "n_cells_before": int(annotated.n_obs),
        "n_cells_after": int(filtered.n_obs),
        "counts_layer": counts_layer,
    }
    logger.info("Writing stage-1 raw-count object: %s shape=%s", stage1_output, filtered.shape)
    _atomic_write_h5ad(filtered, stage1_output)
    del annotated, filtered
    gc.collect()

    logger.info("Reloading and validating stage-1 object")
    adata = sc.read_h5ad(stage1_output)
    stage1_validation = validate_stage1_object(
        adata,
        counts_layer=counts_layer,
        expected_shape=expected_stage1_shape,
        expected_counts_audit=expected_counts_audit,
    )
    counts_before_normalization = sparse_matrix_audit(adata.layers[counts_layer])

    logger.info("Normalizing X and log1p-transforming X; raw counts layer remains untouched")
    normalize_log1p_preserving_counts(
        adata,
        counts_layer=counts_layer,
        target_sum=float(preprocess["target_sum"]),
    )
    counts_after_normalization = sparse_matrix_audit(adata.layers[counts_layer])
    if counts_after_normalization != counts_before_normalization:
        raise RuntimeError("Raw counts layer changed during normalization")

    logger.info("Selecting batch-aware HVGs from raw counts")
    select_batch_aware_hvgs(
        adata,
        counts_layer=counts_layer,
        flavor=str(preprocess["hvg_flavor"]),
        n_top_genes=int(preprocess["n_top_hvg"]),
        batch_key=str(preprocess["hvg_batch_key"]),
    )
    hvg_summary = hvg_summary_table(adata)
    logger.info("Running PCA after scaling only %d HVGs", len(hvg_summary))
    hvg = compute_hvg_pca(
        adata,
        n_comps=int(preprocess["n_pcs"]),
        scale_max_value=float(preprocess["scale_max_value"]),
        random_state=seed,
    )
    pca_summary, pca_loadings = pca_tables(adata, hvg)
    del hvg
    gc.collect()

    neighbors_key = "neighbors_unintegrated_stage1"
    logger.info("Building unintegrated neighbors and UMAP")
    compute_unintegrated_neighbors_umap(
        adata,
        n_neighbors=int(stage["n_neighbors"]),
        n_pcs=int(stage["use_n_pcs"]),
        random_state=seed,
        neighbors_key=neighbors_key,
    )
    resolutions = [float(value) for value in stage["leiden_resolutions"]]
    primary_resolution = float(stage["primary_resolution"])
    logger.info("Running QC-only Leiden resolutions: %s", resolutions)
    run_leiden_resolutions(
        adata,
        resolutions=resolutions,
        primary_resolution=primary_resolution,
        neighbors_key=neighbors_key,
        random_state=seed,
    )

    cluster_summary, cluster_composition = leiden_qc_tables(adata.obs, resolutions)
    candidate_summary = qc_candidate_cluster_distribution(
        adata.obs,
        cluster_key=f"leiden_{primary_resolution:g}",
    )
    mixing_summary, association_summary = neighbor_mixing_summary(
        adata,
        neighbors_key=neighbors_key,
        resolutions=resolutions,
    )

    logger.info("Writing diagnostic tables and figures")
    filter_summary.to_csv(exploratory_dir / "stage1_filter_summary.tsv", sep="\t", index=False)
    reason_summary.to_csv(
        exploratory_dir / "stage1_filter_reason_overlap.tsv", sep="\t", index=False
    )
    low_gene_thresholds.to_csv(
        exploratory_dir / "stage1_low_gene_thresholds.tsv", sep="\t", index=False
    )
    high_score_thresholds.to_csv(
        exploratory_dir / "high_doublet_score_top1pct_thresholds.tsv", sep="\t", index=False
    )
    hvg_summary.to_csv(exploratory_dir / "hvg_summary.tsv", sep="\t", index=False)
    pca_summary.to_csv(exploratory_dir / "pca_summary.tsv", sep="\t", index=False)
    pca_loadings.to_csv(exploratory_dir / "pca_top_loadings.tsv", sep="\t", index=False)
    cluster_summary.to_csv(exploratory_dir / "leiden_qc_summary.tsv", sep="\t", index=False)
    cluster_composition.to_csv(
        exploratory_dir / "leiden_cluster_composition.tsv", sep="\t", index=False
    )
    candidate_summary.to_csv(
        exploratory_dir / "qc_candidate_cluster_distribution.tsv", sep="\t", index=False
    )
    mixing_summary.to_csv(exploratory_dir / "neighbor_mixing_summary.tsv", sep="\t", index=False)
    association_summary.to_csv(
        exploratory_dir / "cluster_metadata_association.tsv", sep="\t", index=False
    )

    _plot_pca_scree(pca_summary, figure_dir / "pca_scree.png")
    _plot_embedding_panels(
        adata,
        [("library_id", "Library", None)],
        figure_dir / "umap_library.png",
    )
    _plot_embedding_panels(
        adata,
        [("group", "Group", None)],
        figure_dir / "umap_group.png",
    )
    _plot_embedding_panels(
        adata,
        [
            ("total_counts", "Total UMI counts", "p99"),
            ("n_genes_by_counts", "Detected genes", "p99"),
            ("qc_low_genes_5mad", "Low genes: 5 MAD diagnostic", None),
        ],
        figure_dir / "umap_qc_metrics.png",
    )
    _plot_embedding_panels(
        adata,
        [
            ("doublet_score", "Scrublet score", "p99.5"),
            ("qc_doublet_auto", "Scrublet automatic doublet", None),
            ("qc_high_doublet_score_top1pct", "Top 1% score within library", None),
        ],
        figure_dir / "umap_doublet.png",
    )
    _plot_embedding_panels(
        adata,
        [
            ("pct_counts_mt", "Mitochondrial percentage", "p99"),
            ("qc_mt_moderate", "Moderate mt: 5% < mt <= 25%", None),
            ("qc_mt_extreme", "Extreme mt: >25%", None),
            ("retained_extreme_mt", "Retained extreme mt", None),
        ],
        figure_dir / "umap_mt.png",
    )
    _plot_embedding_panels(
        adata,
        [(f"leiden_{primary_resolution:g}", "Leiden 0.5 (QC-only)", None)],
        figure_dir / "leiden_qc.png",
    )
    _plot_embedding_panels(
        adata,
        [(f"leiden_{resolution:g}", f"Leiden {resolution:g}", None) for resolution in resolutions],
        figure_dir / "leiden_resolution_comparison.png",
    )

    adata.uns["stage1_exploratory"] = {
        "integration": "none",
        "purpose": "QC, library structure, mitochondrial and doublet candidate review only",
        "neighbors_key": neighbors_key,
        "n_neighbors": int(stage["n_neighbors"]),
        "n_pcs_neighbors": int(stage["use_n_pcs"]),
        "leiden_resolutions": resolutions,
        "primary_leiden": f"leiden_{primary_resolution:g}",
        "counts_layer": counts_layer,
        "counts_sha256_sparse_arrays": counts_before_normalization["sha256_sparse_arrays"],
    }
    logger.info("Writing exploratory unintegrated object: %s", exploratory_output)
    _atomic_write_h5ad(adata, exploratory_output)
    del adata
    gc.collect()

    logger.info("Reloading final exploratory object for counts/normalization validation")
    checked = sc.read_h5ad(exploratory_output)
    final_counts_audit = sparse_matrix_audit(checked.layers[counts_layer])
    if final_counts_audit != counts_before_normalization:
        raise RuntimeError("Raw counts layer changed in the saved exploratory object")
    if not sparse.isspmatrix_csr(checked.X):
        raise TypeError("Exploratory X is not CSR sparse")
    sample_n = min(100, checked.n_obs)
    normalized_sums = np.expm1(checked.X[:sample_n].toarray()).sum(axis=1)
    nonzero = normalized_sums > 0
    if nonzero.any() and not np.allclose(
        normalized_sums[nonzero],
        float(preprocess["target_sum"]),
        rtol=2e-4,
        atol=1e-2,
    ):
        raise ValueError("Exploratory X failed the normalize_total/log1p validation")
    normalization_validation = pd.DataFrame(
        [
            {
                "object": "02_qc_filtered_stage1.h5ad",
                "X_semantics": "raw UMI counts",
                "counts_layer_semantics": "raw UMI counts",
                **stage1_validation,
            },
            {
                "object": "03_stage1_unintegrated_exploratory.h5ad",
                "X_semantics": "log1p library-size normalized expression",
                "counts_layer_semantics": "raw UMI counts",
                "n_cells": checked.n_obs,
                "n_features": checked.n_vars,
                "n_samples": int(checked.obs["library_id"].nunique()),
                "n_libraries": int(checked.obs["library_id"].nunique()),
                "n_groups": int(checked.obs["group"].nunique()),
                **{f"counts_{key}": value for key, value in final_counts_audit.items()},
            },
        ]
    )
    normalization_validation.to_csv(
        exploratory_dir / "object_validation.tsv", sep="\t", index=False
    )
    del checked
    gc.collect()

    report = build_stage1_report(
        filter_summary=filter_summary,
        reason_summary=reason_summary,
        validation=normalization_validation,
        hvg_summary=hvg_summary,
        pca_summary=pca_summary,
        mixing=mixing_summary,
        association=association_summary,
        cluster_summary=cluster_summary,
        candidates=candidate_summary,
    )
    report_path = exploratory_dir / "STAGE1_EXPLORATORY_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    if _file_signature(input_path) != input_signature_before:
        raise RuntimeError("Forbidden side effect: annotated input H5AD changed")
    if _file_signature(forbidden_filtered) != forbidden_signature_before:
        raise RuntimeError("Forbidden side effect: legacy 02_qc_filtered.h5ad changed")
    logger.info("STAGE1_EXPLORATORY_OK: %s", report_path)
    return report_path
