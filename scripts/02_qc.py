from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse

from _project import (
    DEFAULT_CONFIG,
    load_config,
    project_paths,
    require_compute_resources,
    setup_logging,
)


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


def plot_qc(adata: ad.AnnData, output: Path) -> None:
    frame = adata.obs[
        ["library_id", "group", "total_counts", "n_genes_by_counts", "pct_counts_mt"]
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.violinplot(frame, x="library_id", y="n_genes_by_counts", hue="group", ax=axes[0, 0])
    sns.violinplot(frame, x="library_id", y="total_counts", hue="group", ax=axes[0, 1])
    sns.violinplot(frame, x="library_id", y="pct_counts_mt", hue="group", ax=axes[1, 0])
    sns.scatterplot(
        frame.sample(min(len(frame), 30000), random_state=0),
        x="total_counts",
        y="n_genes_by_counts",
        hue="pct_counts_mt",
        palette="viridis",
        s=5,
        linewidth=0,
        ax=axes[1, 1],
    )
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-library adaptive QC and Scrublet.")
    parser.add_argument("input")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--apply-filter", action="store_true")
    parser.add_argument("--skip-scrublet", action="store_true")
    parser.add_argument("--allow-low-memory", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    require_compute_resources(config, allow_low_memory=args.allow_low_memory)
    paths = project_paths(config)
    logger = setup_logging("02_qc", config)
    qc = config["qc"]
    adata = sc.read_h5ad(args.input)

    adata.var["mt"] = adata.var_names.str.startswith(qc["mouse_mt_prefix"])
    adata.var["ribo"] = adata.var_names.str.match(qc["mouse_ribo_regex"])
    adata.var["hb"] = adata.var_names.str.match(qc["mouse_hb_regex"])
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], percent_top=None, inplace=True
    )
    mark_sample_outliers(adata, config)
    if qc["apply_scrublet"] and not args.skip_scrublet:
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
    adata.write_h5ad(paths["results"] / "02_qc_annotated.h5ad", compression="gzip")

    if args.apply_filter:
        filtered = adata[~adata.obs["qc_fail"]].copy()
        detected = np.asarray(
            (filtered.layers[config["ingest"]["counts_layer"]] > 0).sum(axis=0)
        ).ravel()
        filtered = filtered[:, detected >= int(qc["min_cells_per_gene"])].copy()
        filtered.write_h5ad(paths["results"] / "02_qc_filtered.h5ad", compression="gzip")
        logger.info("QC_FILTER_OK: %s", filtered.shape)
    else:
        logger.info("QC_AUDIT_OK: filters annotated but not applied")


if __name__ == "__main__":
    main()
