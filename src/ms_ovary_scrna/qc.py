from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from .plotting import plot_qc
from .project import project_paths, require_compute_resources, setup_logging


def mad_outlier(values: pd.Series, n_mads: float, side: str = "both") -> pd.Series:
    values = values.astype(float)
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    if mad == 0:
        return pd.Series(False, index=values.index)
    low = values < median - n_mads * mad
    high = values > median + n_mads * mad
    if side == "low":
        return low
    if side == "high":
        return high
    if side != "both":
        raise ValueError(f"Unsupported side: {side}")
    return low | high


def mark_sample_outliers(adata: ad.AnnData, config: dict) -> None:
    qc = config["qc"]
    adata.obs["qc_counts_outlier"] = False
    adata.obs["qc_genes_outlier"] = False
    adata.obs["qc_mt_outlier"] = False
    for library_id in adata.obs["library_id"].astype(str).unique():
        mask = adata.obs["library_id"].astype(str) == library_id
        obs = adata.obs.loc[mask]
        adata.obs.loc[mask, "qc_counts_outlier"] = mad_outlier(
            np.log1p(obs["total_counts"]), qc["counts_n_mads"], "both"
        ).to_numpy()
        adata.obs.loc[mask, "qc_genes_outlier"] = mad_outlier(
            np.log1p(obs["n_genes_by_counts"]), qc["genes_n_mads"], "both"
        ).to_numpy()
        adata.obs.loc[mask, "qc_mt_outlier"] = mad_outlier(
            obs["pct_counts_mt"], qc["mt_n_mads"], "high"
        ).to_numpy()


def run_scrublet_per_library(adata: ad.AnnData, config: dict, logger) -> None:
    qc = config["qc"]
    counts_layer = config["ingest"]["counts_layer"]
    adata.obs["doublet_score"] = np.nan
    adata.obs["predicted_doublet"] = False
    for library_id in adata.obs["library_id"].astype(str).unique():
        mask = adata.obs["library_id"].astype(str) == library_id
        subset = ad.AnnData(
            X=sparse.csr_matrix(adata.layers[counts_layer][mask, :]),
            obs=adata.obs.loc[mask, []].copy(),
            var=adata.var.copy(),
        )
        logger.info("Scrublet: %s (%d cells)", library_id, subset.n_obs)
        sc.pp.scrublet(
            subset,
            expected_doublet_rate=float(qc["expected_doublet_rate"]),
            random_state=int(config["project"]["random_seed"]),
        )
        adata.obs.loc[mask, "doublet_score"] = subset.obs["doublet_score"].to_numpy()
        adata.obs.loc[mask, "predicted_doublet"] = subset.obs[
            "predicted_doublet"
        ].to_numpy()


def run_qc(
    config: dict,
    input_path: str | Path,
    *,
    apply_filter: bool = False,
    skip_scrublet: bool = False,
    allow_low_memory: bool = False,
) -> Path:
    require_compute_resources(config, allow_low_memory=allow_low_memory)
    paths = project_paths(config)
    logger = setup_logging("02_qc", config)
    qc = config["qc"]
    adata = sc.read_h5ad(input_path)

    adata.var["mt"] = adata.var_names.str.startswith(qc["mouse_mt_prefix"])
    adata.var["ribo"] = adata.var_names.str.match(qc["mouse_ribo_regex"])
    adata.var["hb"] = adata.var_names.str.match(qc["mouse_hb_regex"])
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], percent_top=None, inplace=True
    )
    mark_sample_outliers(adata, config)
    if qc["apply_scrublet"] and not skip_scrublet:
        run_scrublet_per_library(adata, config, logger)
    else:
        adata.obs["doublet_score"] = np.nan
        adata.obs["predicted_doublet"] = False

    adata.obs["qc_fail"] = (
        adata.obs["qc_counts_outlier"]
        | adata.obs["qc_genes_outlier"]
        | adata.obs["qc_mt_outlier"]
        | (adata.obs["n_genes_by_counts"] < int(qc["min_genes_initial"]))
    )
    if qc["hard_mt_pct"] is not None:
        adata.obs["qc_fail"] |= adata.obs["pct_counts_mt"] > float(qc["hard_mt_pct"])
    if qc["filter_predicted_doublets"]:
        adata.obs["qc_fail"] |= adata.obs["predicted_doublet"].fillna(False)

    summary = (
        adata.obs.groupby(["library_id", "group"], observed=True)
        .agg(
            cells_before=("qc_fail", "size"),
            cells_flagged=("qc_fail", "sum"),
            median_counts=("total_counts", "median"),
            median_genes=("n_genes_by_counts", "median"),
            median_mt_pct=("pct_counts_mt", "median"),
            predicted_doublets=("predicted_doublet", "sum"),
        )
        .reset_index()
    )
    summary["retained_fraction"] = 1 - summary["cells_flagged"] / summary["cells_before"]
    summary.to_csv(paths["results"] / "02_qc_summary.tsv", sep="\t", index=False)
    plot_qc(adata, paths["figures"] / "02_qc_before_filtering.png")
    annotated_output = paths["results"] / "02_qc_annotated.h5ad"
    adata.write_h5ad(annotated_output, compression="gzip")

    if not apply_filter:
        logger.info("QC_AUDIT_OK: filters annotated but not applied")
        return annotated_output

    filtered = adata[~adata.obs["qc_fail"]].copy()
    detected = np.asarray(
        (filtered.layers[config["ingest"]["counts_layer"]] > 0).sum(axis=0)
    ).ravel()
    filtered = filtered[:, detected >= int(qc["min_cells_per_gene"])].copy()
    filtered_output = paths["results"] / "02_qc_filtered.h5ad"
    filtered.write_h5ad(filtered_output, compression="gzip")
    logger.info("QC_FILTER_OK: %s", filtered.shape)
    return filtered_output
