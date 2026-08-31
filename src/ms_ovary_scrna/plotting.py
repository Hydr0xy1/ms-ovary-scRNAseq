from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns


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
    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
