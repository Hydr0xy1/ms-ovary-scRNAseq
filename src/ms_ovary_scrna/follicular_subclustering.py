from __future__ import annotations

import gc
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

from .integration import harmony_integrate_compatible
from .preprocessing import (
    compute_hvg_pca,
    normalize_log1p_preserving_counts,
    select_batch_aware_hvgs,
)
from .project import load_yaml, project_paths, require_compute_resources, setup_logging
from .stage1_exploratory import sparse_matrix_audit


def resolution_key(resolution: float) -> str:
    return f"follicular_leiden_{resolution:g}"


def build_follicular_subset(
    adata: ad.AnnData,
    *,
    source_cluster_key: str,
    source_clusters: list[str],
    counts_layer: str,
) -> ad.AnnData:
    """Copy the requested source clusters into a separate, count-preserving object."""
    required_obs = {
        "library_id",
        "group",
        "age_months",
        "treatment",
        source_cluster_key,
        "cell_type_broad",
        "annotation_confidence",
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
        "doublet_score",
        "predicted_doublet",
    }
    missing = sorted(required_obs - set(adata.obs.columns))
    if missing:
        raise KeyError(f"Input object is missing required cell metadata: {missing}")
    if counts_layer not in adata.layers:
        raise KeyError(f"Input object is missing raw-count layer: {counts_layer}")

    source = adata.obs[source_cluster_key].astype(str)
    observed = set(source.unique())
    missing_clusters = sorted(set(source_clusters) - observed)
    if missing_clusters:
        raise ValueError(f"Requested source clusters are absent: {missing_clusters}")
    subset = adata[source.isin(source_clusters).to_numpy()].copy()
    subset.obs["original_leiden_cluster"] = pd.Categorical(
        subset.obs[source_cluster_key].astype(str), categories=source_clusters
    )
    subset.layers[counts_layer] = sparse.csr_matrix(subset.layers[counts_layer])
    subset.raw = None

    # Remove inherited whole-atlas reductions/graphs so they cannot be mistaken for
    # subset-specific results. Original Leiden labels and all cell metadata are retained.
    for key in list(subset.obsm.keys()):
        del subset.obsm[key]
    for key in list(subset.obsp.keys()):
        del subset.obsp[key]
    for key in list(subset.varm.keys()):
        del subset.varm[key]
    subset.uns.clear()
    return subset


def subset_inventory(obs: pd.DataFrame) -> pd.DataFrame:
    counts = obs["original_leiden_cluster"].astype(str).value_counts(sort=False)
    records = [
        {
            "entity": "ALL",
            "n_cells": len(obs),
            "pct_subset": 100.0,
        }
    ]
    records.extend(
        {
            "entity": str(cluster),
            "n_cells": int(count),
            "pct_subset": 100.0 * int(count) / len(obs),
        }
        for cluster, count in counts.items()
    )
    return pd.DataFrame(records)


def hvg_diagnostics(
    adata: ad.AnnData,
    *,
    cell_cycle_genes: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = adata.var[adata.var["highly_variable"].astype(bool)].copy()
    selected.insert(0, "gene", selected.index.astype(str))
    selected["mt_gene"] = (
        selected["mt"].astype(bool) if "mt" in selected else selected.index.str.startswith("mt-")
    )
    selected["ribosomal_gene"] = (
        selected["ribo"].astype(bool)
        if "ribo" in selected
        else selected.index.str.match(r"^Rp[sl]")
    )
    selected["cell_cycle_gene"] = selected.index.isin(cell_cycle_genes)
    selected = selected.sort_values(
        ["highly_variable_nbatches", "highly_variable_rank", "gene"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    wanted = [
        "gene",
        "gene_ids",
        "feature_types",
        "highly_variable_rank",
        "highly_variable_nbatches",
        "highly_variable_intersection",
        "means",
        "variances",
        "variances_norm",
        "mt_gene",
        "ribosomal_gene",
        "cell_cycle_gene",
    ]
    selected = selected[[column for column in wanted if column in selected]].reset_index(drop=True)
    n_hvg = len(selected)
    summary = pd.DataFrame(
        [
            {"metric": "n_hvg", "count": n_hvg, "pct_hvg": 100.0},
            {
                "metric": "mitochondrial_hvg",
                "count": int(selected["mt_gene"].sum()),
                "pct_hvg": 100.0 * float(selected["mt_gene"].mean()),
            },
            {
                "metric": "ribosomal_hvg",
                "count": int(selected["ribosomal_gene"].sum()),
                "pct_hvg": 100.0 * float(selected["ribosomal_gene"].mean()),
            },
            {
                "metric": "cell_cycle_hvg",
                "count": int(selected["cell_cycle_gene"].sum()),
                "pct_hvg": 100.0 * float(selected["cell_cycle_gene"].mean()),
            },
        ]
    )
    return selected, summary


def pca_diagnostics(
    adata: ad.AnnData,
    *,
    marker_programs: dict[str, list[str]],
    top_n: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratios = np.asarray(adata.uns["pca"]["variance_ratio"], dtype=float)
    variance = np.asarray(adata.uns["pca"]["variance"], dtype=float)
    variance_table = pd.DataFrame(
        {
            "pc": np.arange(1, len(ratios) + 1),
            "explained_variance": variance,
            "explained_variance_ratio": ratios,
            "cumulative_variance_ratio": np.cumsum(ratios),
        }
    )
    loadings = np.asarray(adata.varm["PCs"], dtype=float)
    genes = adata.var_names.astype(str).to_numpy()
    lookup: dict[str, list[str]] = {}
    for program, program_genes in marker_programs.items():
        for gene in program_genes:
            lookup.setdefault(gene, []).append(program)
    records: list[dict[str, Any]] = []
    for pc_index in range(min(10, loadings.shape[1])):
        values = loadings[:, pc_index]
        order = np.argsort(np.abs(values))[-top_n:][::-1]
        for rank, index in enumerate(order, start=1):
            gene = genes[index]
            records.append(
                {
                    "pc": pc_index + 1,
                    "absolute_rank": rank,
                    "gene": gene,
                    "loading": float(values[index]),
                    "absolute_loading": float(abs(values[index])),
                    "direction": "positive" if values[index] >= 0 else "negative",
                    "program_hint": ";".join(lookup.get(gene, ["unassigned"])),
                }
            )
    return variance_table, pd.DataFrame(records)


def graph_mixing_summary(
    obs: pd.DataFrame,
    graph: sparse.spmatrix,
    *,
    representation: str,
) -> pd.DataFrame:
    distances = sparse.csr_matrix(graph)
    libraries = obs["library_id"].astype(str).to_numpy()
    original = obs["original_leiden_cluster"].astype(str).to_numpy()
    library_mixing = np.full(len(obs), np.nan)
    original_purity = np.full(len(obs), np.nan)
    for row in range(len(obs)):
        neighbors = distances.indices[distances.indptr[row] : distances.indptr[row + 1]]
        neighbors = neighbors[neighbors != row]
        if len(neighbors):
            library_mixing[row] = np.mean(libraries[neighbors] != libraries[row])
            original_purity[row] = np.mean(original[neighbors] == original[row])
    records: list[dict[str, Any]] = []
    for entity_type, values in (
        ("ALL", np.array(["ALL"] * len(obs))),
        ("library_id", libraries),
        ("original_leiden_cluster", original),
    ):
        for entity in pd.unique(values):
            mask = values == entity
            records.append(
                {
                    "representation": representation,
                    "entity_type": entity_type,
                    "entity": str(entity),
                    "n_cells": int(mask.sum()),
                    "mean_different_library_neighbor_fraction": float(
                        np.nanmean(library_mixing[mask])
                    ),
                    "median_different_library_neighbor_fraction": float(
                        np.nanmedian(library_mixing[mask])
                    ),
                    "mean_original_cluster_neighbor_purity": float(
                        np.nanmean(original_purity[mask])
                    ),
                }
            )
    return pd.DataFrame(records)


def run_local_leiden(
    adata: ad.AnnData,
    *,
    resolutions: list[float],
    primary_resolution: float,
    neighbors_key: str,
    random_state: int,
) -> None:
    for resolution in resolutions:
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=resolution_key(resolution),
            neighbors_key=neighbors_key,
            random_state=random_state,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
    adata.obs["follicular_leiden"] = adata.obs[resolution_key(primary_resolution)].copy()


def resolution_tables(
    obs: pd.DataFrame,
    resolutions: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary: list[dict[str, Any]] = []
    composition: list[dict[str, Any]] = []
    for resolution in resolutions:
        key = resolution_key(resolution)
        for cluster, frame in obs.groupby(key, observed=True, sort=True):
            summary.append(
                {
                    "resolution": resolution,
                    "cluster_key": key,
                    "cluster": str(cluster),
                    "n_cells": len(frame),
                    "pct_subset": 100.0 * len(frame) / len(obs),
                    "n_libraries": int(frame["library_id"].nunique()),
                    "dominant_original_cluster": str(
                        frame["original_leiden_cluster"].astype(str).value_counts().index[0]
                    ),
                    "dominant_original_cluster_fraction": float(
                        frame["original_leiden_cluster"].astype(str).value_counts(normalize=True).iloc[0]
                    ),
                    "dominant_library_fraction": float(
                        frame["library_id"].astype(str).value_counts(normalize=True).iloc[0]
                    ),
                }
            )
            for dimension in ("original_leiden_cluster", "library_id", "group"):
                counts = frame[dimension].astype(str).value_counts().sort_index()
                for value, count in counts.items():
                    composition.append(
                        {
                            "resolution": resolution,
                            "cluster_key": key,
                            "cluster": str(cluster),
                            "dimension": dimension,
                            "value": str(value),
                            "count": int(count),
                            "pct_cluster": 100.0 * int(count) / len(frame),
                        }
                    )
    return pd.DataFrame(summary), pd.DataFrame(composition)


def resolution_transition_table(
    obs: pd.DataFrame,
    resolutions: list[float],
) -> pd.DataFrame:
    """Describe how cells split between every pair of adjacent resolutions."""
    columns = [
        "from_resolution",
        "to_resolution",
        "from_cluster",
        "to_cluster",
        "n_cells",
        "from_cluster_n_cells",
        "to_cluster_n_cells",
        "pct_from_cluster",
        "pct_to_cluster",
        "n_destinations_from_cluster",
        "dominant_destination_fraction",
        "is_dominant_destination",
    ]
    records: list[dict[str, Any]] = []
    for from_resolution, to_resolution in zip(resolutions[:-1], resolutions[1:], strict=True):
        from_key = resolution_key(from_resolution)
        to_key = resolution_key(to_resolution)
        pair_counts = (
            obs.groupby([from_key, to_key], observed=True, sort=True)
            .size()
            .rename("n_cells")
            .reset_index()
        )
        from_sizes = pair_counts.groupby(from_key, observed=True)["n_cells"].transform("sum")
        to_sizes = pair_counts.groupby(to_key, observed=True)["n_cells"].transform("sum")
        n_destinations = pair_counts.groupby(from_key, observed=True)[to_key].transform("nunique")
        dominant_counts = pair_counts.groupby(from_key, observed=True)["n_cells"].transform("max")
        for index, row in pair_counts.iterrows():
            records.append(
                {
                    "from_resolution": from_resolution,
                    "to_resolution": to_resolution,
                    "from_cluster": str(row[from_key]),
                    "to_cluster": str(row[to_key]),
                    "n_cells": int(row["n_cells"]),
                    "from_cluster_n_cells": int(from_sizes.iloc[index]),
                    "to_cluster_n_cells": int(to_sizes.iloc[index]),
                    "pct_from_cluster": 100.0 * float(row["n_cells"] / from_sizes.iloc[index]),
                    "pct_to_cluster": 100.0 * float(row["n_cells"] / to_sizes.iloc[index]),
                    "n_destinations_from_cluster": int(n_destinations.iloc[index]),
                    "dominant_destination_fraction": float(
                        dominant_counts.iloc[index] / from_sizes.iloc[index]
                    ),
                    "is_dominant_destination": bool(
                        row["n_cells"] == dominant_counts.iloc[index]
                    ),
                }
            )
    return pd.DataFrame(records, columns=columns)


def exploratory_markers(
    adata: ad.AnnData,
    *,
    resolutions: list[float],
    top_n: int,
    logger: Any,
) -> pd.DataFrame:
    results: list[pd.DataFrame] = []
    for resolution in resolutions:
        key = resolution_key(resolution)
        result_key = f"_rank_{key}"
        logger.info("Wilcoxon markers for %s", key)
        sc.tl.rank_genes_groups(
            adata,
            groupby=key,
            method="wilcoxon",
            use_raw=False,
            pts=True,
            key_added=result_key,
        )
        for cluster in adata.obs[key].cat.categories:
            table = sc.get.rank_genes_groups_df(adata, group=cluster, key=result_key).head(top_n)
            table.insert(0, "cluster", str(cluster))
            table.insert(0, "cluster_key", key)
            table.insert(0, "resolution", resolution)
            results.append(table)
        del adata.uns[result_key]
        gc.collect()
    return pd.concat(results, ignore_index=True)


def origin_markers(
    adata: ad.AnnData,
    *,
    top_n: int,
) -> pd.DataFrame:
    result_key = "_rank_original_clusters_within_follicular"
    sc.tl.rank_genes_groups(
        adata,
        groupby="original_leiden_cluster",
        method="wilcoxon",
        use_raw=False,
        pts=True,
        key_added=result_key,
    )
    tables = []
    for cluster in adata.obs["original_leiden_cluster"].cat.categories:
        table = sc.get.rank_genes_groups_df(adata, group=cluster, key=result_key).head(top_n)
        table.insert(0, "original_leiden_cluster", str(cluster))
        tables.append(table)
    del adata.uns[result_key]
    return pd.concat(tables, ignore_index=True)


def marker_program_tables(
    adata: ad.AnnData,
    programs: dict[str, list[str]],
    *,
    cluster_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clusters = adata.obs[cluster_key].astype(str)
    marker_records: list[dict[str, Any]] = []
    program_records: list[dict[str, Any]] = []
    program_means: dict[str, dict[str, float]] = {}
    for program, configured_genes in programs.items():
        genes = [gene for gene in configured_genes if gene in adata.var_names]
        if not genes:
            continue
        matrix = sparse.csr_matrix(adata[:, genes].X)
        for cluster in pd.unique(clusters):
            mask = clusters.eq(cluster).to_numpy()
            view = matrix[mask]
            means = np.asarray(view.mean(axis=0)).ravel()
            fractions = np.asarray((view > 0).mean(axis=0)).ravel()
            program_means.setdefault(str(cluster), {})[program] = float(means.mean())
            for gene, mean, fraction in zip(genes, means, fractions, strict=True):
                marker_records.append(
                    {
                        "cluster": str(cluster),
                        "program": program,
                        "marker": gene,
                        "mean_log_normalized_expression": float(mean),
                        "fraction_expressing": float(fraction),
                    }
                )
            program_records.append(
                {
                    "cluster": str(cluster),
                    "program": program,
                    "n_markers_available": len(genes),
                    "mean_marker_expression": float(means.mean()),
                    "mean_marker_fraction": float(fractions.mean()),
                    "n_markers_fraction_ge_0_10": int((fractions >= 0.10).sum()),
                }
            )
    program_table = pd.DataFrame(program_records)
    for program, frame in program_table.groupby("program", sort=False):
        values = frame["mean_marker_expression"].to_numpy()
        scale = values.std()
        program_table.loc[frame.index, "relative_program_score"] = (
            (values - values.mean()) / scale if scale > 0 else 0.0
        )
    program_table["program_rank"] = program_table.groupby("cluster", observed=True)[
        "relative_program_score"
    ].rank(ascending=False, method="first")
    return pd.DataFrame(marker_records), program_table


def cluster_qc_summary(obs: pd.DataFrame, *, cluster_key: str) -> pd.DataFrame:
    global_doublet = float(obs["predicted_doublet"].fillna(False).astype(bool).mean())
    records = []
    for cluster, frame in obs.groupby(cluster_key, observed=True, sort=True):
        predicted_fraction = float(frame["predicted_doublet"].fillna(False).astype(bool).mean())
        records.append(
            {
                "cluster": str(cluster),
                "n_cells": len(frame),
                "median_total_counts": float(frame["total_counts"].median()),
                "median_n_genes_by_counts": float(frame["n_genes_by_counts"].median()),
                "median_pct_counts_mt": float(frame["pct_counts_mt"].median()),
                "median_doublet_score": float(frame["doublet_score"].median()),
                "predicted_doublet_fraction": predicted_fraction,
                "predicted_doublet_enrichment": (
                    predicted_fraction / global_doublet if global_doublet > 0 else math.nan
                ),
                "source_cluster6_fraction": float(
                    frame["original_leiden_cluster"].astype(str).eq("6").mean()
                ),
                "source_cluster24_fraction": float(
                    frame["original_leiden_cluster"].astype(str).eq("24").mean()
                ),
                "source_cluster25_fraction": float(
                    frame["original_leiden_cluster"].astype(str).eq("25").mean()
                ),
            }
        )
    return pd.DataFrame(records)


def annotation_review_table(
    obs: pd.DataFrame,
    markers: pd.DataFrame,
    programs: pd.DataFrame,
    qc: pd.DataFrame,
    *,
    cluster_key: str,
) -> pd.DataFrame:
    records = []
    for cluster, frame in obs.groupby(cluster_key, observed=True, sort=True):
        cluster = str(cluster)
        marker_names = markers.loc[
            (markers["cluster_key"] == cluster_key) & (markers["cluster"] == cluster), "names"
        ].astype(str)
        ranked = programs[programs["cluster"] == cluster].sort_values("program_rank")
        source_counts = frame["original_leiden_cluster"].astype(str).value_counts()
        records.append(
            {
                "cluster": cluster,
                "n_cells": len(frame),
                "top_20_markers": ";".join(marker_names.head(20)),
                "top_50_markers": ";".join(marker_names.head(50)),
                "candidate_program_1": ranked.iloc[0]["program"] if len(ranked) else "Uncertain",
                "candidate_program_2": (
                    ranked.iloc[1]["program"] if len(ranked) > 1 else "Uncertain"
                ),
                "candidate_1_relative_score": (
                    float(ranked.iloc[0]["relative_program_score"]) if len(ranked) else math.nan
                ),
                "candidate_2_relative_score": (
                    float(ranked.iloc[1]["relative_program_score"])
                    if len(ranked) > 1
                    else math.nan
                ),
                "dominant_original_cluster": str(source_counts.index[0]),
                "dominant_original_cluster_fraction": float(source_counts.iloc[0] / len(frame)),
                "original_cluster_counts_json": json.dumps(
                    {str(key): int(value) for key, value in source_counts.sort_index().items()},
                    sort_keys=True,
                ),
                "library_counts_json": json.dumps(
                    {
                        str(key): int(value)
                        for key, value in (
                            frame["library_id"].astype(str).value_counts().sort_index().items()
                        )
                    },
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(records).merge(qc, on=["cluster", "n_cells"], validate="one_to_one")


def incompatible_granulosa_theca_codetection(
    adata: ad.AnnData,
    *,
    cluster_key: str,
    granulosa_genes: list[str],
    theca_genes: list[str],
    counts_layer: str,
) -> pd.DataFrame:
    granulosa = [gene for gene in granulosa_genes if gene in adata.var_names]
    theca = [gene for gene in theca_genes if gene in adata.var_names]
    g_detected = np.asarray((adata[:, granulosa].layers[counts_layer] > 0).sum(axis=1)).ravel()
    t_detected = np.asarray((adata[:, theca].layers[counts_layer] > 0).sum(axis=1)).ravel()
    records = []
    for cluster, frame in adata.obs.groupby(cluster_key, observed=True, sort=True):
        positions = adata.obs_names.get_indexer(frame.index)
        both = (g_detected[positions] >= 2) & (t_detected[positions] >= 2)
        records.append(
            {
                "cluster": str(cluster),
                "n_cells": len(frame),
                "granulosa_two_marker_fraction": float((g_detected[positions] >= 2).mean()),
                "theca_two_marker_fraction": float((t_detected[positions] >= 2).mean()),
                "granulosa_theca_codetection_fraction": float(both.mean()),
                "median_doublet_score_codetected": (
                    float(frame.iloc[np.flatnonzero(both)]["doublet_score"].median())
                    if both.any()
                    else math.nan
                ),
                "median_doublet_score_not_codetected": (
                    float(frame.iloc[np.flatnonzero(~both)]["doublet_score"].median())
                    if (~both).any()
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(records)


def origin_neighborhood_summary(
    obs: pd.DataFrame,
    graph: sparse.spmatrix,
    *,
    origins: tuple[str, ...] = ("6", "24", "25"),
) -> pd.DataFrame:
    distances = sparse.csr_matrix(graph)
    source = obs["original_leiden_cluster"].astype(str).to_numpy()
    local = obs["follicular_leiden"].astype(str).to_numpy()
    records = []
    for origin in origins:
        rows = np.flatnonzero(source == origin)
        neighbor_source: list[str] = []
        neighbor_local: list[str] = []
        for row in rows:
            neighbors = distances.indices[distances.indptr[row] : distances.indptr[row + 1]]
            neighbors = neighbors[neighbors != row]
            neighbor_source.extend(source[neighbors])
            neighbor_local.extend(local[neighbors])
        source_counts = pd.Series(neighbor_source, dtype=str).value_counts()
        local_counts = pd.Series(local[rows], dtype=str).value_counts()
        records.append(
            {
                "original_cluster": origin,
                "n_cells": len(rows),
                "neighbor_edges": len(neighbor_source),
                "neighbor_original_cluster_counts_json": json.dumps(
                    {str(key): int(value) for key, value in source_counts.sort_index().items()},
                    sort_keys=True,
                ),
                "neighbor_original_cluster_fractions_json": json.dumps(
                    {
                        str(key): float(value / source_counts.sum())
                        for key, value in source_counts.sort_index().items()
                    },
                    sort_keys=True,
                ),
                "local_cluster_counts_json": json.dumps(
                    {str(key): int(value) for key, value in local_counts.sort_index().items()},
                    sort_keys=True,
                ),
                "dominant_local_cluster": str(local_counts.index[0]),
                "dominant_local_cluster_fraction": float(local_counts.iloc[0] / len(rows)),
                "median_total_counts": float(obs.iloc[rows]["total_counts"].median()),
                "median_n_genes_by_counts": float(
                    obs.iloc[rows]["n_genes_by_counts"].median()
                ),
                "median_pct_counts_mt": float(obs.iloc[rows]["pct_counts_mt"].median()),
                "median_doublet_score": float(obs.iloc[rows]["doublet_score"].median()),
                "predicted_doublet_fraction": float(
                    obs.iloc[rows]["predicted_doublet"].fillna(False).astype(bool).mean()
                ),
            }
        )
    return pd.DataFrame(records)


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 250,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
        }
    )


def _save_figure(figure: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".png"), bbox_inches="tight")
    plt.close(figure)


def _categorical_umap(
    adata: ad.AnnData,
    column: str,
    output: Path,
    *,
    title: str,
) -> None:
    coordinates = np.asarray(adata.obsm["X_umap_follicular_harmony"])
    values = adata.obs[column].astype(str)
    categories = sorted(
        values.unique(),
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )
    cmap = plt.get_cmap("tab20")
    figure, axis = plt.subplots(figsize=(10, 7))
    for index, category in enumerate(categories):
        mask = values.eq(category).to_numpy()
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=2,
            alpha=0.65,
            linewidths=0,
            color=cmap(index % 20),
            label=f"{category} (n={mask.sum():,})",
        )
    axis.set(title=title, xlabel="UMAP1", ylabel="UMAP2")
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, markerscale=3)
    _save_figure(figure, output)


def _continuous_panel(
    axis: plt.Axes,
    coordinates: np.ndarray,
    values: np.ndarray,
    title: str,
    highlight: np.ndarray | None = None,
) -> None:
    if highlight is None:
        highlight = np.ones(len(values), dtype=bool)
    axis.scatter(coordinates[:, 0], coordinates[:, 1], s=1, c="#d9d9d9", linewidths=0)
    plot = axis.scatter(
        coordinates[highlight, 0],
        coordinates[highlight, 1],
        c=values[highlight],
        s=4,
        cmap="magma",
        linewidths=0,
    )
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    plt.colorbar(plot, ax=axis, fraction=0.046, pad=0.03)


def _special_origin_plot(
    adata: ad.AnnData,
    origin: str,
    genes: list[str],
    output: Path,
) -> None:
    coordinates = np.asarray(adata.obsm["X_umap_follicular_harmony"])
    origin_mask = adata.obs["original_leiden_cluster"].astype(str).eq(origin).to_numpy()
    panels: list[tuple[str, np.ndarray]] = [
        ("total_counts", adata.obs["total_counts"].to_numpy(dtype=float)),
        ("n_genes_by_counts", adata.obs["n_genes_by_counts"].to_numpy(dtype=float)),
        ("doublet_score", adata.obs["doublet_score"].to_numpy(dtype=float)),
    ]
    for gene in genes:
        if gene in adata.var_names:
            panels.append((gene, np.asarray(adata[:, gene].X.toarray()).ravel()))
    ncols = 4
    nrows = math.ceil(len(panels) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
    for axis, (title, values) in zip(axes.flat, panels, strict=False):
        _continuous_panel(axis, coordinates, values, title, origin_mask)
    for axis in axes.flat[len(panels) :]:
        axis.axis("off")
    figure.suptitle(f"Original cluster {origin}: local follicular review")
    _save_figure(figure, output)


def plot_scree(variance: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(variance["pc"], variance["explained_variance_ratio"], marker="o", ms=3)
    axes[0].axvline(30, color="#666666", linestyle="--")
    axes[0].axvline(40, color="#666666", linestyle=":")
    axes[0].set(xlabel="PC", ylabel="Explained variance ratio", title="Subset PCA scree")
    axes[1].plot(variance["pc"], variance["cumulative_variance_ratio"], marker="o", ms=3)
    axes[1].set(xlabel="PC", ylabel="Cumulative variance ratio", title="Cumulative variance")
    _save_figure(figure, output)


def plot_marker_dotplot(
    marker_evidence: pd.DataFrame,
    output: Path,
) -> None:
    order = marker_evidence[["program", "marker"]].drop_duplicates()
    marker_order = order["marker"].drop_duplicates().tolist()
    clusters = sorted(marker_evidence["cluster"].unique(), key=lambda value: int(value))
    gene_x = {gene: index for index, gene in enumerate(marker_order)}
    cluster_y = {cluster: index for index, cluster in enumerate(clusters)}
    x = marker_evidence["marker"].map(gene_x).to_numpy()
    y = marker_evidence["cluster"].map(cluster_y).to_numpy()
    sizes = 3 + 90 * marker_evidence["fraction_expressing"].to_numpy()
    colors = marker_evidence["mean_log_normalized_expression"].to_numpy()
    figure, axis = plt.subplots(
        figsize=(max(13, len(marker_order) * 0.27), max(5, len(clusters) * 0.45))
    )
    plot = axis.scatter(x, y, s=sizes, c=colors, cmap="viridis", linewidths=0)
    axis.set_xticks(range(len(marker_order)), marker_order, rotation=90)
    axis.set_yticks(range(len(clusters)), clusters)
    axis.invert_yaxis()
    axis.set(xlabel="Follicular/steroidogenic marker", ylabel="Local Leiden cluster")
    plt.colorbar(plot, ax=axis, label="Mean log-normalized expression")
    _save_figure(figure, output)


def plot_cell_cycle(adata: ad.AnnData, output: Path) -> None:
    coordinates = np.asarray(adata.obsm["X_umap_follicular_harmony"])
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    _continuous_panel(axes[0], coordinates, adata.obs["S_score"].to_numpy(), "S score")
    _continuous_panel(axes[1], coordinates, adata.obs["G2M_score"].to_numpy(), "G2M score")
    phase = adata.obs["phase"].astype(str)
    cmap = {"G1": "#999999", "S": "#377eb8", "G2M": "#e41a1c"}
    for name, color in cmap.items():
        mask = phase.eq(name).to_numpy()
        axes[2].scatter(coordinates[mask, 0], coordinates[mask, 1], s=2, c=color, label=name)
    axes[2].set_title("Cell-cycle phase (not regressed)")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].legend(frameon=False)
    _save_figure(figure, output)


def _atomic_write(adata: ad.AnnData, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.h5ad")
    adata.write_h5ad(temporary, compression="gzip")
    temporary.replace(output)


def run_follicular_subclustering(
    config: dict[str, Any],
    input_path: str | Path,
    *,
    primary_resolution: float | None = None,
    use_n_pcs: int | None = None,
    allow_low_memory: bool = False,
) -> Path:
    """Run a self-contained follicular/steroidogenic subclustering review."""
    require_compute_resources(config, allow_low_memory=allow_low_memory)
    paths = project_paths(config)
    logger = setup_logging("04_follicular_subclustering", config)
    settings = config["follicular_subclustering"]
    input_path = Path(input_path).resolve()
    input_stat = input_path.stat()
    output_dir = paths["root"] / settings["output_dir"]
    figure_dir = paths["root"] / settings["figure_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    counts_layer = str(settings["counts_layer"])
    source_key = str(settings["source_cluster_key"])
    source_clusters = [str(value) for value in settings["source_clusters"]]
    seed = int(config["project"]["random_seed"])
    resolutions = [float(value) for value in settings["resolutions"]]
    primary_resolution = float(
        settings["primary_resolution"] if primary_resolution is None else primary_resolution
    )
    use_n_pcs = int(settings["use_n_pcs"] if use_n_pcs is None else use_n_pcs)
    if primary_resolution not in resolutions:
        raise ValueError("Primary resolution must be one of the configured resolutions")
    if use_n_pcs not in [int(value) for value in settings["neighbor_pc_candidates"]]:
        raise ValueError("use_n_pcs must be one of the configured 30-40 PC candidates")

    logger.info("Reading whole-ovary object without modifying it: %s", input_path)
    atlas = sc.read_h5ad(input_path)
    subset = build_follicular_subset(
        atlas,
        source_cluster_key=source_key,
        source_clusters=source_clusters,
        counts_layer=counts_layer,
    )
    del atlas
    gc.collect()
    inventory = subset_inventory(subset.obs)
    logger.info("Follicular/steroidogenic subset shape: %s", subset.shape)
    counts_before = sparse_matrix_audit(subset.layers[counts_layer])

    normalize_log1p_preserving_counts(
        subset,
        counts_layer=counts_layer,
        target_sum=float(config["preprocess"]["target_sum"]),
    )
    logger.info("Selecting 3,000 subset-specific batch-aware HVGs")
    select_batch_aware_hvgs(
        subset,
        counts_layer=counts_layer,
        flavor=str(settings["hvg_flavor"]),
        n_top_genes=int(settings["n_top_hvg"]),
        batch_key=str(settings["hvg_batch_key"]),
    )
    s_genes = [str(gene) for gene in settings["cell_cycle_genes"]["s_phase"]]
    g2m_genes = [str(gene) for gene in settings["cell_cycle_genes"]["g2m_phase"]]
    review_cell_cycle = [
        str(gene)
        for gene in config.get("broad_annotation_review", {}).get("cell_cycle_genes", [])
    ]
    # Use the union of the standard S/G2M lists and the broader ovarian review
    # list so cluster-24 markers such as Cenpa, Prc1 and Ccnb1 are recognized.
    cell_cycle = set(s_genes + g2m_genes + review_cell_cycle)
    hvg_table, hvg_summary = hvg_diagnostics(subset, cell_cycle_genes=cell_cycle)

    logger.info("Computing 50 subset-specific PCs")
    hvg = compute_hvg_pca(
        subset,
        n_comps=int(settings["n_pcs"]),
        scale_max_value=float(config["preprocess"]["scale_max_value"]),
        random_state=seed,
    )
    subset.obsm["X_pca_follicular_unintegrated"] = subset.obsm["X_pca"].copy()
    marker_config = load_yaml(paths["markers"])["follicular_subclustering"]
    marker_programs = {
        name: [str(gene) for gene in definition["positive"]]
        for name, definition in marker_config.items()
    }
    variance, loadings = pca_diagnostics(subset, marker_programs=marker_programs)
    del hvg
    gc.collect()

    logger.info("Scoring cell cycle without regression")
    sc.tl.score_genes_cell_cycle(
        subset,
        s_genes=[gene for gene in s_genes if gene in subset.var_names],
        g2m_genes=[gene for gene in g2m_genes if gene in subset.var_names],
    )

    logger.info("Building unintegrated PCA graph")
    unintegrated_key = "neighbors_follicular_unintegrated"
    sc.pp.neighbors(
        subset,
        n_neighbors=int(settings["n_neighbors"]),
        n_pcs=use_n_pcs,
        use_rep="X_pca_follicular_unintegrated",
        key_added=unintegrated_key,
        random_state=seed,
    )
    sc.tl.umap(subset, neighbors_key=unintegrated_key, random_state=seed)
    subset.obsm["X_umap_follicular_unintegrated"] = subset.obsm["X_umap"].copy()
    mixing_tables = [
        graph_mixing_summary(
            subset.obs,
            subset.obsp[f"{unintegrated_key}_distances"],
            representation=f"unintegrated_{use_n_pcs}PC",
        )
    ]

    logger.info("Harmony integration using library_id")
    harmony_integrate_compatible(
        subset,
        batch_key="library_id",
        seed=seed,
        basis="X_pca_follicular_unintegrated",
        adjusted_basis="X_pca_follicular_harmony",
    )
    candidate_keys: dict[int, str] = {}
    for n_pcs in [int(value) for value in settings["neighbor_pc_candidates"]]:
        key = f"neighbors_follicular_harmony_{n_pcs}pc"
        candidate_keys[n_pcs] = key
        sc.pp.neighbors(
            subset,
            n_neighbors=int(settings["n_neighbors"]),
            n_pcs=n_pcs,
            use_rep="X_pca_follicular_harmony",
            key_added=key,
            random_state=seed,
        )
        mixing_tables.append(
            graph_mixing_summary(
                subset.obs,
                subset.obsp[f"{key}_distances"],
                representation=f"harmony_{n_pcs}PC",
            )
        )
    harmony_key = candidate_keys[use_n_pcs]
    sc.tl.umap(subset, neighbors_key=harmony_key, random_state=seed)
    subset.obsm["X_umap_follicular_harmony"] = subset.obsm["X_umap"].copy()
    del subset.obsm["X_umap"]

    logger.info("Running multi-resolution local Leiden")
    run_local_leiden(
        subset,
        resolutions=resolutions,
        primary_resolution=primary_resolution,
        neighbors_key=harmony_key,
        random_state=seed,
    )
    resolution_summary, resolution_composition = resolution_tables(subset.obs, resolutions)
    resolution_transitions = resolution_transition_table(subset.obs, resolutions)
    markers = exploratory_markers(
        subset,
        resolutions=resolutions,
        top_n=int(settings["marker_top_n"]),
        logger=logger,
    )
    logger.info("Calculating original-cluster markers within the local compartment")
    source_markers = origin_markers(subset, top_n=200)
    source_markers["cell_cycle_gene"] = source_markers["names"].isin(cell_cycle)
    primary_key = resolution_key(primary_resolution)
    marker_evidence, program_summary = marker_program_tables(
        subset,
        marker_programs,
        cluster_key=primary_key,
    )
    qc_summary = cluster_qc_summary(subset.obs, cluster_key=primary_key)
    review = annotation_review_table(
        subset.obs,
        markers,
        program_summary,
        qc_summary,
        cluster_key=primary_key,
    )
    codetection = incompatible_granulosa_theca_codetection(
        subset,
        cluster_key=primary_key,
        granulosa_genes=marker_programs["Granulosa_identity"],
        theca_genes=marker_programs["Theca_steroidogenic"],
        counts_layer=counts_layer,
    )
    review = review.merge(codetection, on=["cluster", "n_cells"], validate="one_to_one")
    neighborhood = origin_neighborhood_summary(
        subset.obs,
        subset.obsp[f"{harmony_key}_distances"],
    )

    tables = {
        "follicular_subset_inventory.tsv": inventory,
        "follicular_hvg.tsv": hvg_table,
        "follicular_hvg_summary.tsv": hvg_summary,
        "follicular_pca_variance.tsv": variance,
        "follicular_pca_top_loadings.tsv": loadings,
        "follicular_neighbor_mixing.tsv": pd.concat(mixing_tables, ignore_index=True),
        "follicular_resolution_summary.tsv": resolution_summary,
        "follicular_resolution_composition.tsv": resolution_composition,
        "follicular_resolution_transitions.tsv": resolution_transitions,
        "follicular_cluster_markers.tsv": markers,
        "follicular_origin_markers.tsv": source_markers,
        "follicular_marker_evidence.tsv": marker_evidence,
        "follicular_program_summary.tsv": program_summary,
        "follicular_cluster_qc.tsv": qc_summary,
        "follicular_incompatible_codetection.tsv": codetection,
        "follicular_origin_neighborhood.tsv": neighborhood,
        "follicular_annotation_review.tsv": review,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, sep="\t", index=False)

    subset.uns["follicular_subclustering"] = {
        "source_cluster_key": source_key,
        "source_clusters": source_clusters,
        "hvg_flavor": str(settings["hvg_flavor"]),
        "hvg_batch_key": str(settings["hvg_batch_key"]),
        "n_top_hvg": int(settings["n_top_hvg"]),
        "n_pcs": int(settings["n_pcs"]),
        "use_n_pcs": use_n_pcs,
        "n_neighbors": int(settings["n_neighbors"]),
        "integration_batch_key": "library_id",
        "resolutions": resolutions,
        "primary_resolution": primary_resolution,
        "cell_cycle_regressed": False,
    }
    logger.info("Rendering local diagnostic figures")
    _set_style()
    plot_scree(variance, figure_dir / "follicular_pca_scree")
    _categorical_umap(
        subset,
        "follicular_leiden",
        figure_dir / "follicular_umap_cluster",
        title=f"Follicular/steroidogenic Harmony UMAP: Leiden {primary_resolution:g}",
    )
    _categorical_umap(
        subset,
        "original_leiden_cluster",
        figure_dir / "follicular_umap_original_cluster",
        title="Follicular/steroidogenic UMAP: original whole-ovary cluster",
    )
    _categorical_umap(
        subset,
        "library_id",
        figure_dir / "follicular_umap_library",
        title="Follicular/steroidogenic UMAP: library",
    )
    _categorical_umap(
        subset,
        "group",
        figure_dir / "follicular_umap_group",
        title="Follicular/steroidogenic UMAP: group (descriptive only)",
    )
    plot_marker_dotplot(marker_evidence, figure_dir / "follicular_marker_dotplot")
    plot_cell_cycle(subset, figure_dir / "follicular_cell_cycle")
    _special_origin_plot(
        subset,
        "6",
        ["Inha", "Hsd17b1", "Pik3ip1", "Itih5", "Hsd3b1", "Fdx1", "Cyp11a1"],
        figure_dir / "follicular_cluster6_review",
    )
    _special_origin_plot(
        subset,
        "24",
        ["Mki67", "Top2a", "Fdx1", "Cyp11a1", "Hsd3b1", "Inha", "Foxl2"],
        figure_dir / "follicular_cluster24_review",
    )
    _special_origin_plot(
        subset,
        "25",
        ["Ptgfr", "Sfrp4", "Lhcgr", "Star", "Cyp11a1", "Foxl2", "Inha"],
        figure_dir / "follicular_cluster25_review",
    )

    counts_after = sparse_matrix_audit(subset.layers[counts_layer])
    if counts_before != counts_after:
        raise RuntimeError("Raw counts changed during follicular subclustering")
    output = output_dir / "follicular_steroidogenic.h5ad"
    logger.info("Writing separate subcluster object atomically: %s", output)
    _atomic_write(subset, output)
    saved = sc.read_h5ad(output, backed="r")
    if saved.n_obs != subset.n_obs or saved.n_vars != subset.n_vars:
        raise RuntimeError("Saved follicular object has an unexpected shape")
    if "counts" not in saved.layers:
        raise RuntimeError("Saved follicular object lost the raw-count layer")
    saved_counts = saved.layers[counts_layer]
    if hasattr(saved_counts, "to_memory"):
        saved_counts = saved_counts.to_memory()
    if sparse_matrix_audit(saved_counts) != counts_before:
        raise RuntimeError("Saved follicular object changed the raw-count sparse arrays")
    saved.file.close()
    stat_after = input_path.stat()
    if (input_stat.st_size, input_stat.st_mtime_ns) != (
        stat_after.st_size,
        stat_after.st_mtime_ns,
    ):
        raise RuntimeError("Forbidden side effect: the whole-ovary input H5AD changed")
    logger.info("FOLLICULAR_SUBCLUSTERING_OK: %s", output)
    return output
