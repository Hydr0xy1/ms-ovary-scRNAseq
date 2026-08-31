from __future__ import annotations

from pathlib import Path

import anndata as ad
import celltypist
import numpy as np
import pandas as pd
import scanpy as sc

from .project import load_yaml, project_paths, require_compute_resources, setup_logging


def available_genes(adata: ad.AnnData, genes: list[str]) -> list[str]:
    names = set(adata.raw.var_names if adata.raw is not None else adata.var_names)
    return [gene for gene in genes if gene in names]


def score_marker_panels(adata: ad.AnnData, markers: dict, logger) -> list[str]:
    score_columns: list[str] = []
    for section, panels in markers.items():
        for label, definition in panels.items():
            genes = available_genes(adata, definition.get("positive", []))
            if len(genes) < 2:
                logger.warning(
                    "Skipping %s/%s; only %d marker genes found",
                    section,
                    label,
                    len(genes),
                )
                continue
            score_name = f"score__{section}__{label}"
            sc.tl.score_genes(
                adata,
                gene_list=genes,
                score_name=score_name,
                use_raw=adata.raw is not None,
                random_state=0,
            )
            score_columns.append(score_name)
    return score_columns


def apply_manual_labels(
    adata: ad.AnnData,
    mapping_path: Path,
    cluster_key: str,
    final_key: str,
) -> None:
    mapping = pd.read_csv(mapping_path, sep="\t", dtype=str)
    mapping = mapping.dropna(subset=["cluster", "cell_type_final"])
    cluster_to_label = dict(zip(mapping["cluster"], mapping["cell_type_final"]))
    adata.obs[final_key] = (
        adata.obs[cluster_key]
        .astype(str)
        .map(cluster_to_label)
        .fillna("Uncertain")
        .astype("category")
    )


def run_celltypist(adata: ad.AnnData, config: dict, logger) -> None:
    annotation = config["annotation"]
    expression = adata.raw.to_adata() if adata.raw is not None else adata.copy()
    if annotation["celltypist_immune_only"]:
        mask = adata.obs["cell_type_marker_provisional"].astype(str) == "Immune"
        expression = expression[mask].copy()
        logger.info(
            "CellTypist restricted to %d marker-provisional immune cells",
            expression.n_obs,
        )
    if expression.n_obs == 0:
        logger.warning("No cells selected for CellTypist")
        return
    prediction = celltypist.annotate(
        expression,
        model=annotation["celltypist_model"],
        majority_voting=True,
    ).to_adata()
    label_col = (
        "majority_voting" if "majority_voting" in prediction.obs else "predicted_labels"
    )
    adata.obs["celltypist_label"] = pd.Series(
        "Not_assessed", index=adata.obs_names, dtype="string"
    )
    adata.obs["celltypist_confidence"] = np.nan
    adata.obs.loc[prediction.obs_names, "celltypist_label"] = prediction.obs[
        label_col
    ].astype(str)
    adata.obs.loc[prediction.obs_names, "celltypist_confidence"] = prediction.obs[
        "conf_score"
    ].to_numpy()
    adata.obs["celltypist_label"] = adata.obs["celltypist_label"].astype("category")


def run_annotation(
    config: dict,
    input_path: str | Path,
    *,
    use_celltypist: bool = False,
    allow_low_memory: bool = False,
) -> Path:
    require_compute_resources(config, allow_low_memory=allow_low_memory)
    paths = project_paths(config)
    logger = setup_logging("04_annotate", config)
    annotation = config["annotation"]
    cluster_key = annotation["cluster_key"]
    final_key = annotation["final_label_key"]
    adata = sc.read_h5ad(input_path)
    if cluster_key not in adata.obs:
        raise KeyError(f"Cluster key missing: {cluster_key}")

    markers = load_yaml(paths["markers"])
    score_columns = score_marker_panels(adata, markers, logger)
    broad_scores = [column for column in score_columns if column.startswith("score__broad__")]
    if not broad_scores:
        raise ValueError("No broad marker panels contain at least two genes in the dataset")
    score_frame = adata.obs.groupby(cluster_key, observed=True)[score_columns].mean()
    score_frame.to_csv(paths["results"] / "04_cluster_marker_scores.tsv", sep="\t")
    broad_winner = score_frame[broad_scores].idxmax(axis=1).str.replace(
        "score__broad__", "", regex=False
    )
    adata.obs["cell_type_marker_provisional"] = (
        adata.obs[cluster_key].map(broad_winner).astype("category")
    )

    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        method="wilcoxon",
        use_raw=adata.raw is not None,
        pts=True,
    )
    marker_tables = []
    for cluster in adata.obs[cluster_key].cat.categories:
        table = sc.get.rank_genes_groups_df(adata, group=cluster).head(100)
        table.insert(0, "cluster", str(cluster))
        marker_tables.append(table)
    pd.concat(marker_tables, ignore_index=True).to_csv(
        paths["results"] / "04_cluster_top_markers.tsv", sep="\t", index=False
    )

    if use_celltypist or annotation["run_celltypist"]:
        run_celltypist(adata, config, logger)
    else:
        adata.obs["celltypist_label"] = pd.Categorical(
            ["Not_run"] * adata.n_obs,
            categories=["Not_run"],
        )
        adata.obs["celltypist_confidence"] = np.nan

    apply_manual_labels(adata, paths["cluster_labels"], cluster_key, final_key)
    output = paths["results"] / "04_annotated.h5ad"
    adata.write_h5ad(output, compression="gzip")
    adata.obs[
        [
            "library_id",
            "group",
            cluster_key,
            "cell_type_marker_provisional",
            "celltypist_label",
            "celltypist_confidence",
            final_key,
        ]
    ].to_csv(paths["results"] / "04_cell_labels.tsv", sep="\t")

    dotplot_genes = {
        label: available_genes(adata, definition["positive"][:5])
        for label, definition in markers["broad"].items()
    }
    dotplot = sc.pl.dotplot(
        adata,
        var_names=dotplot_genes,
        groupby=cluster_key,
        use_raw=adata.raw is not None,
        show=False,
        return_fig=True,
    )
    dotplot.savefig(paths["figures"] / "04_broad_marker_review.pdf")
    logger.info(
        "ANNOTATION_DRAFT_OK: final labels remain Uncertain until "
        "metadata/cluster_labels.tsv is curated"
    )
    return output
