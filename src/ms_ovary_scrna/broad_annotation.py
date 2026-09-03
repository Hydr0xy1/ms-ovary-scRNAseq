from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from .project import load_yaml, project_paths, require_compute_resources, setup_logging

ALLOWED_BROAD_LABELS = {
    "Granulosa",
    "Theca_steroidogenic",
    "Stromal_fibroblast",
    "Smooth_muscle_pericyte",
    "Vascular_endothelial",
    "Lymphatic_endothelial",
    "Ovarian_epithelial",
    "Ciliated_epithelial",
    "Immune",
    "Oocyte",
    "Luteal_candidate",
    "Erythroid",
    "Uncertain",
}
ALLOWED_STATES = {"Cycling", "Non_cycling", "High_mt_candidate", "Doublet_suspicious", "None"}
ALLOWED_CONFIDENCE = {"High", "Medium", "Low", "Uncertain"}


def _natural_cluster_order(values: pd.Series | pd.Index) -> list[str]:
    unique = {str(value) for value in values}
    return sorted(
        unique, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)
    )


def read_cluster_mapping(path: str | Path, expected_clusters: list[str]) -> pd.DataFrame:
    """Read and strictly validate the manually curated broad-lineage mapping."""
    mapping = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {
        "cluster",
        "candidate_1",
        "candidate_2",
        "cell_type_broad",
        "cell_state_provisional",
        "annotation_confidence",
        "doublet_concern",
        "qc_concern",
        "annotation_rationale",
    }
    missing = sorted(required - set(mapping.columns))
    if missing:
        raise KeyError(f"Broad annotation mapping is missing columns: {missing}")
    if mapping["cluster"].duplicated().any():
        duplicated = mapping.loc[mapping["cluster"].duplicated(), "cluster"].tolist()
        raise ValueError(f"Duplicated clusters in broad annotation mapping: {duplicated}")
    observed = set(mapping["cluster"])
    expected = set(expected_clusters)
    if observed != expected:
        raise ValueError(
            "Broad annotation mapping cluster mismatch: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    invalid_labels = sorted(set(mapping["cell_type_broad"]) - ALLOWED_BROAD_LABELS)
    invalid_states = sorted(set(mapping["cell_state_provisional"]) - ALLOWED_STATES)
    invalid_confidence = sorted(set(mapping["annotation_confidence"]) - ALLOWED_CONFIDENCE)
    if invalid_labels or invalid_states or invalid_confidence:
        raise ValueError(
            "Invalid annotation values: "
            f"labels={invalid_labels}, states={invalid_states}, confidence={invalid_confidence}"
        )
    empty_rationale = mapping["annotation_rationale"].str.strip().eq("")
    if empty_rationale.any():
        raise ValueError(
            "Every cluster needs an evidence-based annotation_rationale; missing for "
            f"{mapping.loc[empty_rationale, 'cluster'].tolist()}"
        )
    mapping["doublet_concern"] = (
        mapping["doublet_concern"].str.lower().map({"true": True, "false": False})
    )
    mapping["qc_concern"] = mapping["qc_concern"].str.lower().map({"true": True, "false": False})
    if mapping[["doublet_concern", "qc_concern"]].isna().any().any():
        raise ValueError("doublet_concern and qc_concern must be True or False")
    return mapping.sort_values(
        "cluster",
        key=lambda column: column.map({value: i for i, value in enumerate(expected_clusters)}),
    ).reset_index(drop=True)


def _expression_matrix(adata: ad.AnnData, genes: list[str]) -> sparse.csr_matrix:
    matrix = adata[:, genes].X
    if sparse.issparse(matrix):
        return sparse.csr_matrix(matrix)
    return sparse.csr_matrix(np.asarray(matrix))


def cluster_marker_evidence(
    adata: ad.AnnData,
    panels: dict[str, dict[str, list[str]]],
    marker_table: pd.DataFrame,
    *,
    cluster_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate transparent cluster-level expression and panel evidence.

    Means use the existing log-normalized ``adata.X``. Fractions are the fraction
    of cells with a positive value. Existing Wilcoxon ranks/logFC values are joined
    without recalculating differential expression.
    """
    clusters = _natural_cluster_order(adata.obs[cluster_key].astype(str))
    all_markers = sorted(
        {
            gene
            for definition in panels.values()
            for field in ("positive", "exclude")
            for gene in definition.get(field, [])
            if gene in adata.var_names
        }
    )
    if not all_markers:
        raise ValueError("None of the broad-annotation markers are present in adata.var_names")

    cluster_values = adata.obs[cluster_key].astype(str).to_numpy()
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    codes = np.array([cluster_index[value] for value in cluster_values], dtype=np.int32)
    sizes = np.bincount(codes, minlength=len(clusters)).astype(np.float64)
    membership = sparse.csr_matrix(
        (np.ones(adata.n_obs, dtype=np.float32), (codes, np.arange(adata.n_obs))),
        shape=(len(clusters), adata.n_obs),
    )
    expression = _expression_matrix(adata, all_markers)
    means = (membership @ expression).multiply(1.0 / sizes[:, None]).toarray()
    detected = expression.copy()
    detected.data = np.ones_like(detected.data)
    fractions = (membership @ detected).multiply(1.0 / sizes[:, None]).toarray()

    marker_table = marker_table.copy()
    marker_table["cluster"] = marker_table["cluster"].astype(str)
    marker_table["marker_rank"] = marker_table.groupby("cluster", sort=False).cumcount() + 1
    rank_lookup = marker_table.set_index(["cluster", "names"])["marker_rank"].to_dict()
    lfc_lookup = marker_table.set_index(["cluster", "names"])["logfoldchanges"].to_dict()
    gene_index = {gene: index for index, gene in enumerate(all_markers)}

    rows: list[dict[str, Any]] = []
    for panel, definition in panels.items():
        for role in ("positive", "exclude"):
            for gene in definition.get(role, []):
                if gene not in gene_index:
                    continue
                gene_column = gene_index[gene]
                for cluster_row, cluster in enumerate(clusters):
                    rows.append(
                        {
                            "cluster": cluster,
                            "panel": panel,
                            "marker_role": role,
                            "marker": gene,
                            "mean_log_normalized_expression": float(
                                means[cluster_row, gene_column]
                            ),
                            "fraction_expressing": float(fractions[cluster_row, gene_column]),
                            "marker_rank": rank_lookup.get((cluster, gene), np.nan),
                            "cluster_vs_rest_logFC": lfc_lookup.get((cluster, gene), np.nan),
                        }
                    )
    evidence = pd.DataFrame(rows)

    positive = evidence[evidence["marker_role"] == "positive"].copy()
    score_records: list[pd.DataFrame] = []
    for panel, panel_evidence in positive.groupby("panel", sort=False):
        pivot = panel_evidence.pivot(
            index="cluster",
            columns="marker",
            values="mean_log_normalized_expression",
        ).fillna(0.0)
        z_values = pivot.copy()
        for gene in z_values.columns:
            values = z_values[gene].to_numpy(dtype=float)
            standard_deviation = values.std()
            z_values[gene] = (values - values.mean()) / (
                standard_deviation if standard_deviation > 0 else 1.0
            )
        panel_scores = z_values.mean(axis=1).rename("relative_panel_score").reset_index()
        panel_scores.insert(1, "panel", panel)
        score_records.append(panel_scores)
    score = pd.concat(score_records, ignore_index=True)
    fraction_summary = (
        positive.groupby(["cluster", "panel"], observed=True)
        .agg(
            mean_positive_fraction=("fraction_expressing", "mean"),
            n_positive_markers_available=("marker", "nunique"),
            n_positive_markers_fraction_ge_0_10=(
                "fraction_expressing",
                lambda values: int((values >= 0.10).sum()),
            ),
        )
        .reset_index()
    )
    panel_summary = score.merge(fraction_summary, on=["cluster", "panel"], how="left")
    panel_summary["panel_rank"] = panel_summary.groupby("cluster", observed=True)[
        "relative_panel_score"
    ].rank(method="first", ascending=False)
    return evidence, panel_summary.sort_values(["cluster", "panel_rank"]).reset_index(drop=True)


def _top_marker_strings(marker_table: pd.DataFrame) -> pd.DataFrame:
    marker_table = marker_table.copy()
    marker_table["cluster"] = marker_table["cluster"].astype(str)
    records = []
    for cluster, table in marker_table.groupby("cluster", sort=False):
        records.append(
            {
                "cluster": cluster,
                "top_20_markers": ";".join(table.head(20)["names"].astype(str)),
                "top_50_markers": ";".join(table.head(50)["names"].astype(str)),
                "top_markers": ";".join(table.head(20)["names"].astype(str)),
            }
        )
    return pd.DataFrame(records)


def _format_positive_evidence(table: pd.DataFrame, panel: str) -> str:
    view = table[(table["panel"] == panel) & (table["marker_role"] == "positive")].copy()
    view = view.sort_values(
        ["fraction_expressing", "mean_log_normalized_expression"], ascending=False
    ).head(8)
    return (
        ";".join(
            f"{row.marker}(frac={row.fraction_expressing:.2f},mean={row.mean_log_normalized_expression:.2f})"
            for row in view.itertuples(index=False)
            if row.fraction_expressing > 0
        )
        or "none_detected"
    )


def _format_negative_conflicts(table: pd.DataFrame, panel: str) -> str:
    view = table[(table["panel"] == panel) & (table["marker_role"] == "exclude")].copy()
    view = view[view["fraction_expressing"] >= 0.10].sort_values(
        "fraction_expressing", ascending=False
    )
    return (
        ";".join(
            f"{row.marker}(frac={row.fraction_expressing:.2f})"
            for row in view.itertuples(index=False)
        )
        or "none_above_10pct"
    )


def build_broad_annotation_review(
    adata: ad.AnnData,
    mapping: pd.DataFrame,
    evidence: pd.DataFrame,
    panel_summary: pd.DataFrame,
    marker_table: pd.DataFrame,
    *,
    cluster_key: str,
    integration_review: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the requested 26-cluster audit table without forcing uncertain labels."""
    counts = (
        adata.obs[cluster_key]
        .astype(str)
        .value_counts()
        .rename_axis("cluster")
        .rename("n_cells")
        .reset_index()
    )
    review = mapping.merge(counts, on="cluster", validate="one_to_one")
    review = review.merge(_top_marker_strings(marker_table), on="cluster", validate="one_to_one")

    computed = panel_summary[panel_summary["panel_rank"] <= 2].copy()
    computed = computed.pivot(index="cluster", columns="panel_rank", values="panel").reset_index()
    computed = computed.rename(columns={1.0: "computed_candidate_1", 2.0: "computed_candidate_2"})
    review = review.merge(computed, on="cluster", how="left", validate="one_to_one")

    positives = []
    conflicts = []
    for row in review.itertuples(index=False):
        panel = row.cell_type_broad if row.cell_type_broad != "Uncertain" else row.candidate_1
        cluster_evidence = evidence[evidence["cluster"] == row.cluster]
        positives.append(_format_positive_evidence(cluster_evidence, panel))
        conflicts.append(_format_negative_conflicts(cluster_evidence, panel))
    review["positive_marker_evidence"] = positives
    review["negative_marker_conflicts"] = conflicts

    if integration_review is not None:
        qc_columns = [
            "cluster",
            "median_total_counts",
            "median_n_genes_by_counts",
            "median_pct_counts_mt",
            "median_doublet_score",
            "predicted_doublet_fraction",
            "max_incompatible_coexpression_fraction",
            "qc_comment",
        ]
        available = [column for column in qc_columns if column in integration_review.columns]
        integration_view = integration_review[available].copy()
        integration_view["cluster"] = integration_view["cluster"].astype(str)
        review = review.merge(integration_view, on="cluster", how="left", validate="one_to_one")

    order = [
        "cluster",
        "n_cells",
        "top_markers",
        "top_20_markers",
        "top_50_markers",
        "positive_marker_evidence",
        "negative_marker_conflicts",
        "candidate_1",
        "candidate_2",
        "computed_candidate_1",
        "computed_candidate_2",
        "cell_type_broad",
        "cell_state_provisional",
        "annotation_confidence",
        "doublet_concern",
        "qc_concern",
        "annotation_rationale",
    ]
    order.extend(column for column in review.columns if column not in order)
    return (
        review[order]
        .sort_values(
            "cluster",
            key=lambda column: column.map(
                {value: i for i, value in enumerate(_natural_cluster_order(column))}
            ),
        )
        .reset_index(drop=True)
    )


def incompatible_codetection(
    adata: ad.AnnData,
    panels: dict[str, dict[str, list[str]]],
    *,
    cluster_key: str,
    cluster: str,
    focal_panel: str,
    comparison_panels: list[str],
    counts_layer: str = "counts",
) -> pd.DataFrame:
    """Check single-cell co-detection of at least two genes from incompatible panels."""
    if counts_layer not in adata.layers:
        raise KeyError(f"Missing raw-count layer: {counts_layer}")
    mask = adata.obs[cluster_key].astype(str).to_numpy() == cluster
    records = []
    focal_genes = [gene for gene in panels[focal_panel]["positive"] if gene in adata.var_names]
    focal = adata[mask, focal_genes].layers[counts_layer]
    focal_detected = np.asarray((focal > 0).sum(axis=1)).ravel() >= min(2, len(focal_genes))
    scores = adata.obs.loc[mask, "doublet_score"].to_numpy(dtype=float)
    for panel in comparison_panels:
        genes = [gene for gene in panels[panel]["positive"] if gene in adata.var_names]
        comparison = adata[mask, genes].layers[counts_layer]
        comparison_detected = np.asarray((comparison > 0).sum(axis=1)).ravel() >= min(2, len(genes))
        both = focal_detected & comparison_detected
        records.append(
            {
                "cluster": cluster,
                "focal_panel": focal_panel,
                "comparison_panel": panel,
                "n_cells": int(mask.sum()),
                "focal_two_marker_fraction": float(focal_detected.mean()),
                "comparison_two_marker_fraction": float(comparison_detected.mean()),
                "co_detection_fraction": float(both.mean()),
                "median_doublet_score_co_detected": (
                    float(np.median(scores[both])) if both.any() else np.nan
                ),
                "median_doublet_score_not_co_detected": (
                    float(np.median(scores[~both])) if (~both).any() else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def noncycling_marker_table(
    marker_table: pd.DataFrame,
    *,
    cluster: str,
    cell_cycle_genes: list[str],
    top_n: int = 50,
) -> pd.DataFrame:
    view = marker_table[marker_table["cluster"].astype(str) == cluster].copy()
    view["excluded_as_cell_cycle"] = view["names"].isin(set(cell_cycle_genes))
    return view[~view["excluded_as_cell_cycle"]].head(top_n).reset_index(drop=True)


def cluster24_lineage_comparison(
    panel_summary: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Find cluster 24's closest curated cluster inside each requested major lineage."""
    score_matrix = panel_summary.pivot(
        index="cluster", columns="panel", values="relative_panel_score"
    ).fillna(0.0)
    if "24" not in score_matrix.index:
        raise KeyError("Cluster 24 is missing from marker panel summary")
    focal = score_matrix.loc["24"].to_numpy(dtype=float)
    records = []
    for lineage in ["Granulosa", "Theca_steroidogenic", "Stromal_fibroblast", "Immune"]:
        candidates = mapping.loc[
            mapping["cell_type_broad"].eq(lineage) & mapping["cluster"].ne("24"), "cluster"
        ]
        candidates = [cluster for cluster in candidates if cluster in score_matrix.index]
        if not candidates:
            continue
        distances = {
            cluster: float(np.linalg.norm(focal - score_matrix.loc[cluster].to_numpy(dtype=float)))
            for cluster in candidates
        }
        closest = min(distances, key=distances.get)
        records.append(
            {
                "candidate_lineage": lineage,
                "nearest_cluster": closest,
                "euclidean_distance_all_major_panels": distances[closest],
                "cluster24_lineage_score": float(
                    score_matrix.loc["24", lineage] if lineage in score_matrix.columns else np.nan
                ),
                "nearest_cluster_lineage_score": float(
                    score_matrix.loc[closest, lineage]
                    if lineage in score_matrix.columns
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(records).sort_values("euclidean_distance_all_major_panels")


def celltypist_model_audit() -> pd.DataFrame:
    """List available CellTypist models and explicitly assess whole-ovary suitability."""
    try:
        import celltypist

        models = list(celltypist.models.get_all_models())
        error = ""
    except Exception as exception:  # pragma: no cover - network/cache dependent
        models = []
        error = f"model_catalog_unavailable: {exception}"
    rows = []
    broadly_suitable_names = {
        "Mouse_Cell_Atlas.pkl",
        "Tabula_Muris.pkl",
        "Mouse_All_Tissues.pkl",
    }
    for model in models:
        if model in broadly_suitable_names:
            assessment = "potential_general_mouse_support_requires_manual_validation"
            suitable = True
        elif "Immune" in model:
            assessment = "immune_only_not_suitable_for_whole_ovary"
            suitable = False
        elif "Mouse" in model:
            assessment = "mouse_but_organ_or_development_stage_mismatch"
            suitable = False
        else:
            assessment = "human_or_other_reference_not_suitable_for_whole_mouse_ovary"
            suitable = False
        rows.append(
            {
                "model": model,
                "available": True,
                "suitable_for_whole_adult_mouse_ovary": suitable,
                "assessment": assessment,
                "catalog_error": error,
            }
        )
    if not rows:
        rows.append(
            {
                "model": "not_available",
                "available": False,
                "suitable_for_whole_adult_mouse_ovary": False,
                "assessment": (
                    "CellTypist was not run because no auditable suitable model was found"
                ),
                "catalog_error": error,
            }
        )
    return pd.DataFrame(rows)


def _hash_sparse_arrays(matrix: Any) -> str:
    matrix = sparse.csr_matrix(matrix)
    digest = hashlib.sha256()
    for array in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def _save_figure(figure: plt.Figure, base: Path) -> None:
    figure.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate dependency."""
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _plot_categorical_umap(
    adata: ad.AnnData,
    column: str,
    title: str,
    output_base: Path,
) -> None:
    coordinates = np.asarray(adata.obsm["X_umap"])
    values = adata.obs[column].astype(str)
    categories = sorted(values.unique())
    palette = plt.get_cmap("tab20", max(len(categories), 1))
    figure, axis = plt.subplots(figsize=(9.5, 7.5))
    for index, category in enumerate(categories):
        mask = values.to_numpy() == category
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=1.2,
            alpha=0.65,
            color=palette(index),
            label=f"{category} (n={int(mask.sum()):,})",
            linewidths=0,
            rasterized=True,
        )
    axis.set(title=title, xlabel="UMAP1", ylabel="UMAP2")
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, markerscale=5)
    axis.spines[["top", "right"]].set_visible(False)
    _save_figure(figure, output_base)


def _plot_marker_dotplot(
    evidence: pd.DataFrame,
    mapping: pd.DataFrame,
    output_base: Path,
) -> None:
    selected: list[tuple[str, str]] = []
    for panel, table in evidence[evidence["marker_role"] == "positive"].groupby(
        "panel", sort=False
    ):
        genes = (
            table.groupby("marker", observed=True)["fraction_expressing"]
            .max()
            .sort_values(ascending=False)
            .head(4)
            .index
        )
        selected.extend((panel, gene) for gene in genes)
    selected_keys = {(panel, gene) for panel, gene in selected}
    view = evidence[
        evidence.apply(lambda row: (row["panel"], row["marker"]) in selected_keys, axis=1)
        & evidence["marker_role"].eq("positive")
    ].copy()
    clusters = _natural_cluster_order(mapping["cluster"])
    gene_labels = [f"{panel}: {gene}" for panel, gene in selected]
    gene_positions = {(panel, gene): index for index, (panel, gene) in enumerate(selected)}
    cluster_positions = {cluster: index for index, cluster in enumerate(clusters)}
    figure, axis = plt.subplots(figsize=(max(16, len(selected) * 0.38), 10))
    x = [gene_positions[(row.panel, row.marker)] for row in view.itertuples(index=False)]
    y = [cluster_positions[row.cluster] for row in view.itertuples(index=False)]
    scatter = axis.scatter(
        x,
        y,
        s=np.maximum(view["fraction_expressing"].to_numpy() * 150, 2),
        c=view["mean_log_normalized_expression"],
        cmap="viridis",
        linewidths=0,
    )
    axis.set_xticks(range(len(gene_labels)), gene_labels, rotation=90, fontsize=7)
    axis.set_yticks(range(len(clusters)), clusters)
    axis.set(xlabel="Canonical major-lineage marker", ylabel="Leiden 0.5 cluster")
    axis.invert_yaxis()
    figure.colorbar(scatter, ax=axis, label="Mean log-normalized expression")
    axis.set_title("Broad-lineage canonical markers (dot size = fraction expressing)")
    _save_figure(figure, output_base)


def _plot_feature_grid(
    adata: ad.AnnData,
    features: list[str],
    title: str,
    output_base: Path,
    *,
    highlight_cluster: str | None = None,
    cluster_key: str = "leiden_0.5",
) -> None:
    available = [
        feature for feature in features if feature in adata.var_names or feature in adata.obs
    ]
    columns = 4
    rows = int(np.ceil(len(available) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.1 * columns, 3.6 * rows), squeeze=False)
    coordinates = np.asarray(adata.obsm["X_umap"])
    if highlight_cluster is None:
        mask = np.ones(adata.n_obs, dtype=bool)
    else:
        mask = adata.obs[cluster_key].astype(str).to_numpy() == highlight_cluster
    for axis, feature in zip(axes.ravel(), available, strict=False):
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=0.4,
            color="#d9d9d9",
            linewidths=0,
            rasterized=True,
        )
        if feature in adata.obs:
            values = adata.obs[feature].to_numpy(dtype=float)
        else:
            vector = adata[:, feature].X
            values = (
                vector.toarray().ravel() if sparse.issparse(vector) else np.asarray(vector).ravel()
            )
        shown_values = values[mask]
        upper = float(np.quantile(shown_values, 0.99)) if shown_values.size else 1.0
        order = np.argsort(shown_values)
        scatter = axis.scatter(
            coordinates[mask, 0][order],
            coordinates[mask, 1][order],
            c=shown_values[order],
            s=5 if highlight_cluster is not None else 0.8,
            cmap="magma",
            vmin=0,
            vmax=upper if upper > 0 else 1.0,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(feature)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.02)
    for axis in axes.ravel()[len(available) :]:
        axis.axis("off")
    figure.suptitle(title)
    _save_figure(figure, output_base)


def _write_report(
    review: pd.DataFrame,
    model_audit: pd.DataFrame,
    cluster24_codetection: pd.DataFrame,
) -> str:
    label_counts = review.groupby("cell_type_broad", observed=True)["n_cells"].sum().reset_index()
    confidence = (
        review.groupby("annotation_confidence", observed=True)
        .size()
        .rename("n_clusters")
        .reset_index()
    )
    suitable_models = model_audit[model_audit["suitable_for_whole_adult_mouse_ovary"]]
    cluster24 = review[review["cluster"] == "24"].iloc[0]
    cluster25 = review[review["cluster"] == "25"].iloc[0]
    lines = [
        "# Major/broad ovarian cell-type annotation review",
        "",
        "本轮只进行major lineage注释；没有删除细胞、subclustering、组间比较或下游机制分析。",
        "",
        "## Broad-lineage cell totals",
        "",
        _markdown_table(label_counts),
        "",
        "## Cluster confidence",
        "",
        _markdown_table(confidence),
        "",
        "## Cluster 24",
        "",
        f"- broad label: {cluster24.cell_type_broad}",
        f"- state: {cluster24.cell_state_provisional}",
        f"- confidence: {cluster24.annotation_confidence}",
        f"- rationale: {cluster24.annotation_rationale}",
        "",
        _markdown_table(cluster24_codetection),
        "",
        "## Cluster 25",
        "",
        f"- broad label: {cluster25.cell_type_broad}",
        f"- confidence: {cluster25.annotation_confidence}",
        f"- rationale: {cluster25.annotation_rationale}",
        "",
        "## CellTypist decision",
        "",
        (
            "A potentially suitable whole-mouse reference was found but was not allowed "
            "to overwrite "
            "manual broad labels."
            if not suitable_models.empty
            else (
                "No suitable whole-adult-mouse-ovary CellTypist model was found; "
                "CellTypist was not run."
            )
        ),
    ]
    return "\n".join(lines) + "\n"


def run_broad_annotation_review(
    config: dict[str, Any],
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    allow_low_memory: bool = False,
) -> Path:
    """Run the major-lineage review and atomically update the annotated H5AD."""
    require_compute_resources(config, allow_low_memory=allow_low_memory)
    paths = project_paths(config)
    settings = config["broad_annotation_review"]
    logger = setup_logging("04_broad_annotation_review", config)
    cluster_key = settings["cluster_key"]
    input_path = Path(input_path).resolve()
    output_path = Path(output_path or input_path).resolve()
    output_dir = (paths["root"] / settings["output_dir"]).resolve()
    figure_dir = (paths["root"] / settings["figure_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading annotation draft: %s", input_path)
    adata = sc.read_h5ad(input_path)
    if cluster_key not in adata.obs:
        raise KeyError(f"Missing cluster key: {cluster_key}")
    for required in ("counts",):
        if required not in adata.layers:
            raise KeyError(f"Missing required layer: {required}")
    if "X_umap" not in adata.obsm:
        raise KeyError("Missing Harmony UMAP: X_umap")

    clusters = _natural_cluster_order(adata.obs[cluster_key].astype(str))
    mapping_path = (paths["root"] / settings["mapping_file"]).resolve()
    marker_path = (paths["root"] / settings["top_marker_file"]).resolve()
    integration_path = (paths["root"] / settings["integration_review_file"]).resolve()
    mapping = read_cluster_mapping(mapping_path, clusters)
    markers = load_yaml(paths["markers"])[settings["marker_section"]]
    marker_table = pd.read_csv(marker_path, sep="\t", dtype={"cluster": str})
    integration_review = (
        pd.read_csv(integration_path, sep="\t", dtype={"cluster": str})
        if integration_path.exists()
        else None
    )

    logger.info("Calculating marker means, detection fractions, ranks and existing logFC")
    evidence, panel_summary = cluster_marker_evidence(
        adata,
        markers,
        marker_table,
        cluster_key=cluster_key,
    )
    review = build_broad_annotation_review(
        adata,
        mapping,
        evidence,
        panel_summary,
        marker_table,
        cluster_key=cluster_key,
        integration_review=integration_review,
    )

    cell_cycle_genes = list(settings["cell_cycle_genes"])
    cluster24_noncycling = noncycling_marker_table(
        marker_table,
        cluster="24",
        cell_cycle_genes=cell_cycle_genes,
    )
    cluster24_codetection = incompatible_codetection(
        adata,
        markers,
        cluster_key=cluster_key,
        cluster="24",
        focal_panel="Theca_steroidogenic",
        comparison_panels=[
            "Granulosa",
            "Stromal_fibroblast",
            "Immune",
            "Ovarian_epithelial",
            "Vascular_endothelial",
        ],
    )
    cluster24_comparison = cluster24_lineage_comparison(panel_summary, mapping)
    cluster25_focus = evidence[
        (evidence["cluster"] == "25")
        & evidence["marker"].isin(
            [
                "Cyp17a1",
                "Cyp11a1",
                "Hsd3b1",
                "Srd5a1",
                "Lhcgr",
                "Ptgfr",
                "Sfrp4",
                "Star",
                "Scarb1",
                "Foxl2",
                "Amh",
                "Inha",
                "Inhbb",
            ]
        )
    ].drop_duplicates(["cluster", "marker"])

    logger.info("Auditing available CellTypist models without running an unsuitable model")
    model_audit = celltypist_model_audit()
    references = pd.read_csv(paths["root"] / "metadata" / "annotation_references.tsv", sep="\t")

    evidence.to_csv(output_dir / "marker_evidence_long.tsv", sep="\t", index=False)
    panel_summary.to_csv(output_dir / "marker_panel_summary.tsv", sep="\t", index=False)
    review.to_csv(output_dir / "broad_annotation_review.tsv", sep="\t", index=False)
    cluster24_noncycling.to_csv(
        output_dir / "cluster24_non_cell_cycle_top_markers.tsv", sep="\t", index=False
    )
    cluster24_codetection.to_csv(
        output_dir / "cluster24_incompatible_codetection.tsv", sep="\t", index=False
    )
    cluster24_comparison.to_csv(
        output_dir / "cluster24_lineage_comparison.tsv", sep="\t", index=False
    )
    cluster25_focus.to_csv(
        output_dir / "cluster25_steroidogenic_evidence.tsv", sep="\t", index=False
    )
    model_audit.to_csv(output_dir / "celltypist_model_audit.tsv", sep="\t", index=False)
    references.to_csv(output_dir / "reference_crosscheck.tsv", sep="\t", index=False)

    cluster_to_mapping = mapping.set_index("cluster")
    cluster_values = adata.obs[cluster_key].astype(str)
    adata.obs["cell_type_broad"] = pd.Categorical(
        cluster_values.map(cluster_to_mapping["cell_type_broad"]),
        categories=sorted(ALLOWED_BROAD_LABELS),
    )
    adata.obs["cell_state_provisional"] = pd.Categorical(
        cluster_values.map(cluster_to_mapping["cell_state_provisional"]),
        categories=sorted(ALLOWED_STATES),
    )
    adata.obs["annotation_confidence"] = pd.Categorical(
        cluster_values.map(cluster_to_mapping["annotation_confidence"]),
        categories=["High", "Medium", "Low", "Uncertain"],
        ordered=True,
    )
    adata.uns["broad_annotation_review"] = {
        "cluster_key": cluster_key,
        "mapping_file": str(mapping_path.relative_to(paths["root"])),
        "marker_section": settings["marker_section"],
        "celltypist_decision": "not_run_no_suitable_whole_adult_mouse_ovary_model",
        "reference_ids": references["reference_id"].astype(str).tolist(),
        "scope": "major_lineage_only_no_subclustering_no_group_comparison",
    }
    adata.obs[
        [
            "library_id",
            "group",
            cluster_key,
            "cell_type_broad",
            "cell_state_provisional",
            "annotation_confidence",
            "celltypist_label",
            "celltypist_confidence",
            "cell_type_final",
        ]
    ].to_csv(output_dir / "cell_annotation_metadata.tsv.gz", sep="\t", compression="gzip")

    logger.info("Rendering major-lineage annotation figures")
    _plot_categorical_umap(
        adata,
        "cell_type_broad",
        "Harmony UMAP: curated major ovarian lineages",
        figure_dir / "umap_cell_type_broad",
    )
    _plot_categorical_umap(
        adata,
        "annotation_confidence",
        "Harmony UMAP: broad-annotation confidence",
        figure_dir / "umap_annotation_confidence",
    )
    _plot_marker_dotplot(evidence, mapping, figure_dir / "broad_lineage_marker_dotplot")
    _plot_feature_grid(
        adata,
        [
            "Foxl2",
            "Cyp11a1",
            "Dcn",
            "Rgs5",
            "Pecam1",
            "Prox1",
            "Epcam",
            "Foxj1",
            "Ptprc",
            "Ddx4",
            "Ptgfr",
            "Alas2",
        ],
        "Selected canonical major-lineage markers",
        figure_dir / "selected_marker_feature_plots",
    )
    _plot_feature_grid(
        adata,
        ["Mki67", "Fdx1", "Cyp11a1", "Hsd3b1", "Inha", "Epcam", "Ptprc", "doublet_score"],
        "Cluster 24: cycling steroidogenic evidence and doublet checks",
        figure_dir / "cluster24_special_review",
        highlight_cluster="24",
        cluster_key=cluster_key,
    )
    _plot_feature_grid(
        adata,
        ["Star", "Cyp11a1", "Scarb1", "Ptgfr", "Sfrp4", "Lhcgr", "Hsd3b1", "Foxl2"],
        "Cluster 25: rare steroidogenic identity review",
        figure_dir / "cluster25_special_review",
        highlight_cluster="25",
        cluster_key=cluster_key,
    )

    report = _write_report(review, model_audit, cluster24_codetection)
    (output_dir / "BROAD_ANNOTATION_REPORT.md").write_text(report, encoding="utf-8")

    baseline = {
        "X": _hash_sparse_arrays(adata.X),
        "counts": _hash_sparse_arrays(adata.layers["counts"]),
    }
    temporary = output_path.with_name(f".{output_path.name}.broad-annotation.tmp")
    logger.info("Writing annotated object atomically: %s", output_path)
    adata.write_h5ad(temporary, compression="gzip")
    verification = sc.read_h5ad(temporary)
    verified = {
        "X": _hash_sparse_arrays(verification.X),
        "counts": _hash_sparse_arrays(verification.layers["counts"]),
    }
    integrity = pd.DataFrame(
        [
            {
                "matrix": matrix,
                "sha256_before": baseline[matrix],
                "sha256_after": verified[matrix],
                "identical": baseline[matrix] == verified[matrix],
                "shape": json.dumps(list(verification.shape)),
                "n_cells": int(verification.n_obs),
            }
            for matrix in ("X", "counts")
        ]
    )
    integrity.to_csv(output_dir / "matrix_integrity.tsv", sep="\t", index=False)
    if not integrity["identical"].all():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Expression/count integrity failed; original annotated object was not replaced"
        )
    if verification.n_obs != adata.n_obs:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Unexpected cell count after annotation: {verification.n_obs}")
    del verification
    os.replace(temporary, output_path)
    logger.info("BROAD_ANNOTATION_OK: %s", output_path)
    return output_path
