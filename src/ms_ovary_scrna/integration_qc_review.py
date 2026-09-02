from __future__ import annotations

import gc
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse

from .project import load_yaml, project_paths, require_compute_resources, setup_logging
from .qc_sensitivity import _markdown_table
from .stage1_exploratory import sparse_matrix_audit

REVIEW_LINEAGES = (
    "Stromal_fibroblast",
    "Granulosa",
    "Theca_steroidogenic",
    "Immune",
    "Ovarian_epithelium",
    "Smooth_muscle",
    "Endothelial",
)

INCOMPATIBLE_LINEAGE_PAIRS = (
    ("Granulosa", "Immune"),
    ("Granulosa", "Endothelial"),
    ("Granulosa", "Ovarian_epithelium"),
    ("Theca_steroidogenic", "Immune"),
    ("Theca_steroidogenic", "Endothelial"),
    ("Stromal_fibroblast", "Immune"),
    ("Endothelial", "Immune"),
    ("Endothelial", "Ovarian_epithelium"),
    ("Smooth_muscle", "Immune"),
    ("Smooth_muscle", "Ovarian_epithelium"),
    ("Immune", "Ovarian_epithelium"),
)

GROUP_PALETTE = {"Y": "#4C78A8", "OC": "#E45756", "OT": "#59A14F"}
LINEAGE_PALETTE = {
    "Stromal_fibroblast": "#A0CBE8",
    "Granulosa": "#F28E2B",
    "Theca_steroidogenic": "#FFBE7D",
    "Immune": "#E15759",
    "Ovarian_epithelium": "#B07AA1",
    "Smooth_muscle": "#9C755F",
    "Endothelial": "#76B7B2",
}


def _percentage(count: int | float, total: int) -> float:
    return 100.0 * float(count) / total if total else float("nan")


def _bool_array(values: pd.Series) -> np.ndarray:
    return values.fillna(False).astype(bool).to_numpy()


def _json_counts(values: pd.Series) -> str:
    counts = values.astype(str).value_counts().sort_index()
    return json.dumps({key: int(value) for key, value in counts.items()}, sort_keys=True)


def _quantiles(values: np.ndarray) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan"), float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(finite, [0.25, 0.5, 0.75]))


def _category_order(values: Iterable[str]) -> list[str]:
    def key(value: str) -> tuple[int, float | str]:
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)

    return sorted({str(value) for value in values}, key=key)


def resolution_cluster_key(resolution: float) -> str:
    """Match the one-decimal Leiden keys written by stage 3 (including 1.0)."""
    return f"leiden_{resolution:.1f}"


def load_primary_cluster_programs(
    path: str | Path,
    *,
    cluster_key: str,
    lineages: tuple[str, ...] = REVIEW_LINEAGES,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read the existing broad marker scores and derive provisional winners."""
    table = pd.read_csv(path, sep="\t", dtype={cluster_key: str})
    if cluster_key not in table:
        raise KeyError(f"Cluster marker score table is missing {cluster_key}")
    columns = [f"score__broad__{lineage}" for lineage in lineages]
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise KeyError(f"Cluster marker score table is missing columns: {missing}")
    table[cluster_key] = table[cluster_key].astype(str)
    table = table.set_index(cluster_key, drop=False)
    values = table[columns]
    top_column = values.idxmax(axis=1)
    ordered_values = np.sort(values.to_numpy(dtype=float), axis=1)
    table["broad_program_1"] = top_column.str.replace("score__broad__", "", regex=False)
    table["broad_program_1_score"] = values.max(axis=1)
    table["broad_program_2"] = values.apply(
        lambda row: row.nlargest(2).index[-1].replace("score__broad__", ""),
        axis=1,
    )
    table["broad_program_2_score"] = ordered_values[:, -2]
    table["broad_program_margin"] = ordered_values[:, -1] - ordered_values[:, -2]
    mapping = table["broad_program_1"].astype(str).to_dict()
    return table.reset_index(drop=True), mapping


def attach_provisional_lineage(
    adata: ad.AnnData,
    *,
    cluster_key: str,
    mapping: dict[str, str],
) -> None:
    labels = adata.obs[cluster_key].astype(str).map(mapping)
    if labels.isna().any():
        missing = sorted(adata.obs.loc[labels.isna(), cluster_key].astype(str).unique())
        raise ValueError(f"No provisional broad program for clusters: {missing}")
    adata.obs["provisional_broad_lineage"] = pd.Categorical(
        labels,
        categories=[label for label in REVIEW_LINEAGES if label in set(labels)],
    )


def compute_marker_program_means(
    adata: ad.AnnData,
    markers: dict[str, Any],
    *,
    lineages: tuple[str, ...] = REVIEW_LINEAGES,
) -> pd.DataFrame:
    """Compute transparent per-cell mean log-expression for broad marker panels."""
    records: list[dict[str, Any]] = []
    names = set(adata.var_names.astype(str))
    for lineage in lineages:
        genes = [gene for gene in markers["broad"][lineage]["positive"] if gene in names]
        if not genes:
            raise ValueError(f"No marker genes available for {lineage}")
        values = np.asarray(adata[:, genes].X.mean(axis=1)).ravel()
        column = f"review_program_mean__{lineage}"
        adata.obs[column] = values.astype(np.float32, copy=False)
        records.append(
            {
                "lineage": lineage,
                "score_column": column,
                "genes": ";".join(genes),
                "n_genes": len(genes),
            }
        )
    return pd.DataFrame(records)


def _normalized_entropy(labels: np.ndarray, possible_categories: int) -> float:
    if labels.size == 0:
        return float("nan")
    if possible_categories <= 1:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(possible_categories))


def per_cell_library_mixing(
    distances: sparse.spmatrix,
    library_labels: np.ndarray,
    *,
    conditioning_labels: np.ndarray | None = None,
) -> pd.DataFrame:
    """Measure same-library fraction and normalized entropy in a stored kNN graph.

    If ``conditioning_labels`` is supplied, only neighbors in the same stratum as
    the focal cell are eligible. This is how group- and lineage-conditioned mixing
    avoids treating cross-group mixing as the batch-correction target.
    """
    graph = sparse.csr_matrix(distances)
    n_cells = graph.shape[0]
    if len(library_labels) != n_cells:
        raise ValueError("Library labels do not align with the neighbor graph")
    if conditioning_labels is not None and len(conditioning_labels) != n_cells:
        raise ValueError("Conditioning labels do not align with the neighbor graph")

    if conditioning_labels is None:
        possible = {"__all__": int(pd.Series(library_labels).nunique())}
    else:
        frame = pd.DataFrame({"condition": conditioning_labels, "library": library_labels})
        possible = frame.groupby("condition", observed=True)["library"].nunique().to_dict()

    same_library = np.full(n_cells, np.nan, dtype=np.float32)
    entropy = np.full(n_cells, np.nan, dtype=np.float32)
    eligible_count = np.zeros(n_cells, dtype=np.int16)
    for cell_index in range(n_cells):
        start, end = graph.indptr[cell_index], graph.indptr[cell_index + 1]
        neighbors = graph.indices[start:end]
        if conditioning_labels is not None:
            neighbors = neighbors[
                conditioning_labels[neighbors] == conditioning_labels[cell_index]
            ]
            condition = conditioning_labels[cell_index]
            n_possible = int(possible[condition])
        else:
            n_possible = possible["__all__"]
        eligible_count[cell_index] = len(neighbors)
        if not len(neighbors):
            continue
        neighbor_libraries = library_labels[neighbors]
        same_library[cell_index] = np.mean(
            neighbor_libraries == library_labels[cell_index]
        )
        entropy[cell_index] = _normalized_entropy(neighbor_libraries, n_possible)
    return pd.DataFrame(
        {
            "same_library_fraction": same_library,
            "library_entropy_normalized": entropy,
            "eligible_neighbor_count": eligible_count,
        }
    )


def _mixing_entities(obs: pd.DataFrame) -> list[tuple[str, str, np.ndarray, str]]:
    group = obs["group"].astype(str).to_numpy()
    library = obs["library_id"].astype(str).to_numpy()
    lineage = obs["provisional_broad_lineage"].astype(str).to_numpy()
    entities: list[tuple[str, str, np.ndarray, str]] = [
        ("overall", "all", np.ones(len(obs), dtype=bool), "overall")
    ]
    for value in pd.unique(group):
        entities.append(("group", value, group == value, "group"))
    for value in pd.unique(lineage):
        entities.append(("lineage", value, lineage == value, "lineage"))
    for value in pd.unique(library):
        entities.append(("library", value, library == value, "group"))
    lineage_group = np.asarray(
        [f"{lineage_value}|{group_value}" for lineage_value, group_value in zip(lineage, group)],
        dtype=str,
    )
    combinations_seen = pd.unique(lineage_group)
    for value in combinations_seen:
        lineage_value, group_value = value.split("|", 1)
        mask = (lineage == lineage_value) & (group == group_value)
        entities.append(("lineage_group", value, mask, "lineage_group"))
    return entities


def neighbor_mixing_tables(
    obs: pd.DataFrame,
    graphs: dict[str, sparse.spmatrix],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize paired pre-/post-Harmony library mixing at requested strata."""
    library = obs["library_id"].astype(str).to_numpy()
    group = obs["group"].astype(str).to_numpy()
    lineage = obs["provisional_broad_lineage"].astype(str).to_numpy()
    lineage_group = np.asarray(
        [f"{lineage_value}|{group_value}" for lineage_value, group_value in zip(lineage, group)],
        dtype=str,
    )
    conditioning = {
        "overall": None,
        "group": group,
        "lineage": lineage,
        "lineage_group": lineage_group,
    }
    entities = _mixing_entities(obs)
    cell_metrics: dict[str, dict[str, pd.DataFrame]] = {}
    summary_records: list[dict[str, Any]] = []

    for state, graph in graphs.items():
        cell_metrics[state] = {
            key: per_cell_library_mixing(
                graph,
                library,
                conditioning_labels=labels,
            )
            for key, labels in conditioning.items()
        }
        for entity_type, entity_id, mask, condition_key in entities:
            frame = cell_metrics[state][condition_key].loc[mask]
            q1_same, median_same, q3_same = _quantiles(
                frame["same_library_fraction"].to_numpy(dtype=float)
            )
            q1_entropy, median_entropy, q3_entropy = _quantiles(
                frame["library_entropy_normalized"].to_numpy(dtype=float)
            )
            summary_records.append(
                {
                    "embedding_state": state,
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "conditioning": condition_key,
                    "n_cells": int(mask.sum()),
                    "valid_cells": int(frame["same_library_fraction"].notna().sum()),
                    "median_eligible_neighbors": float(
                        frame["eligible_neighbor_count"].median()
                    ),
                    "same_library_fraction_q1": q1_same,
                    "same_library_fraction_median": median_same,
                    "same_library_fraction_q3": q3_same,
                    "library_entropy_q1": q1_entropy,
                    "library_entropy_median": median_entropy,
                    "library_entropy_q3": q3_entropy,
                }
            )

    change_records: list[dict[str, Any]] = []
    before_state, after_state = "unintegrated", "harmony"
    for entity_type, entity_id, mask, condition_key in entities:
        before = cell_metrics[before_state][condition_key]
        after = cell_metrics[after_state][condition_key]
        same_delta = (
            after["same_library_fraction"].to_numpy(dtype=float)
            - before["same_library_fraction"].to_numpy(dtype=float)
        )[mask]
        entropy_delta = (
            after["library_entropy_normalized"].to_numpy(dtype=float)
            - before["library_entropy_normalized"].to_numpy(dtype=float)
        )[mask]
        valid = np.isfinite(same_delta) & np.isfinite(entropy_delta)
        change_records.append(
            {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "conditioning": condition_key,
                "n_cells": int(mask.sum()),
                "paired_valid_cells": int(valid.sum()),
                "median_delta_same_library_fraction": (
                    float(np.median(same_delta[valid])) if valid.any() else float("nan")
                ),
                "median_delta_library_entropy": (
                    float(np.median(entropy_delta[valid])) if valid.any() else float("nan")
                ),
                "pct_cells_same_library_decreased": (
                    _percentage(int((same_delta[valid] < 0).sum()), int(valid.sum()))
                    if valid.any()
                    else float("nan")
                ),
                "pct_cells_entropy_increased": (
                    _percentage(int((entropy_delta[valid] > 0).sum()), int(valid.sum()))
                    if valid.any()
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(summary_records), pd.DataFrame(change_records)


def _same_label_neighbor_fraction(
    distances: sparse.spmatrix,
    labels: np.ndarray,
    *,
    conditioning_labels: np.ndarray | None = None,
) -> np.ndarray:
    graph = sparse.csr_matrix(distances)
    values = np.full(graph.shape[0], np.nan, dtype=np.float32)
    for cell_index in range(graph.shape[0]):
        start, end = graph.indptr[cell_index], graph.indptr[cell_index + 1]
        neighbors = graph.indices[start:end]
        if conditioning_labels is not None:
            neighbors = neighbors[
                conditioning_labels[neighbors] == conditioning_labels[cell_index]
            ]
        if len(neighbors):
            values[cell_index] = np.mean(labels[neighbors] == labels[cell_index])
    return values


def biological_structure_tables(
    obs: pd.DataFrame,
    graphs: dict[str, sparse.spmatrix],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify lineage preservation and descriptive group locality."""
    lineage = obs["provisional_broad_lineage"].astype(str).to_numpy()
    group = obs["group"].astype(str).to_numpy()
    summary_records: list[dict[str, Any]] = []
    confusion_records: list[dict[str, Any]] = []

    for state, distances in graphs.items():
        for dimension, labels, condition in (
            ("broad_lineage", lineage, None),
            ("group_global", group, None),
            ("group_within_lineage", group, lineage),
        ):
            fractions = _same_label_neighbor_fraction(
                distances,
                labels,
                conditioning_labels=condition,
            )
            for entity in ["all", *pd.unique(labels).tolist()]:
                mask = np.ones(len(obs), dtype=bool) if entity == "all" else labels == entity
                q1, median, q3 = _quantiles(fractions[mask].astype(float))
                if condition is None:
                    global_proportions = pd.Series(labels).value_counts(normalize=True)
                    expected = (
                        float(np.square(global_proportions).sum())
                        if entity == "all"
                        else float(global_proportions.get(entity, np.nan))
                    )
                else:
                    frame = pd.DataFrame({"lineage": lineage, "group": group})
                    proportions = (
                        frame.groupby("lineage", observed=True)["group"]
                        .value_counts(normalize=True)
                        .to_dict()
                    )
                    cell_expectations = np.asarray(
                        [
                            proportions[(lineage_value, group_value)]
                            for lineage_value, group_value in zip(lineage, group)
                        ],
                        dtype=float,
                    )
                    expected = float(np.mean(cell_expectations[mask]))
                summary_records.append(
                    {
                        "embedding_state": state,
                        "dimension": dimension,
                        "entity_id": entity,
                        "n_cells": int(mask.sum()),
                        "same_neighbor_fraction_q1": q1,
                        "same_neighbor_fraction_median": median,
                        "same_neighbor_fraction_q3": q3,
                        "descriptive_random_expectation": expected,
                    }
                )

        graph = sparse.csr_matrix(distances)
        for source in pd.unique(lineage):
            source_cells = np.flatnonzero(lineage == source)
            target_counts: dict[str, int] = {target: 0 for target in pd.unique(lineage)}
            total = 0
            for cell_index in source_cells:
                start, end = graph.indptr[cell_index], graph.indptr[cell_index + 1]
                neighbors = graph.indices[start:end]
                total += len(neighbors)
                counts = pd.Series(lineage[neighbors]).value_counts()
                for target, count in counts.items():
                    target_counts[str(target)] += int(count)
            for target, count in target_counts.items():
                confusion_records.append(
                    {
                        "embedding_state": state,
                        "source_lineage": source,
                        "target_lineage": target,
                        "neighbor_edges": count,
                        "pct_source_neighbor_edges": _percentage(count, total),
                    }
                )
    return pd.DataFrame(summary_records), pd.DataFrame(confusion_records)


def cluster_size_and_transition_tables(
    obs: pd.DataFrame,
    resolutions: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    size_records: list[dict[str, Any]] = []
    transition_records: list[dict[str, Any]] = []
    for resolution in resolutions:
        key = resolution_cluster_key(resolution)
        counts = obs[key].astype(str).value_counts()
        for cluster, count in counts.items():
            size_records.append(
                {
                    "resolution": resolution,
                    "cluster_key": key,
                    "cluster": str(cluster),
                    "n_cells": int(count),
                    "pct_all_cells": _percentage(int(count), len(obs)),
                }
            )
    for parent_resolution, child_resolution in zip(resolutions[:-1], resolutions[1:]):
        parent_key = resolution_cluster_key(parent_resolution)
        child_key = resolution_cluster_key(child_resolution)
        cross = pd.crosstab(
            obs[parent_key].astype(str),
            obs[child_key].astype(str),
        )
        parent_sizes = cross.sum(axis=1)
        child_sizes = cross.sum(axis=0)
        for parent, child in zip(*np.nonzero(cross.to_numpy()), strict=True):
            count = int(cross.iloc[parent, child])
            transition_records.append(
                {
                    "parent_resolution": parent_resolution,
                    "child_resolution": child_resolution,
                    "parent_cluster": str(cross.index[parent]),
                    "child_cluster": str(cross.columns[child]),
                    "n_cells": count,
                    "pct_parent": _percentage(count, int(parent_sizes.iloc[parent])),
                    "pct_child": _percentage(count, int(child_sizes.iloc[child])),
                }
            )
    return pd.DataFrame(size_records), pd.DataFrame(transition_records)


def exploratory_marker_tables(
    adata: ad.AnnData,
    *,
    resolutions: list[float],
    top_n: int,
    existing_primary_path: str | Path | None,
    primary_resolution: float,
    logger: Any,
) -> pd.DataFrame:
    """Collect exploratory cluster markers at each resolution, reusing 0.5 markers."""
    records: list[pd.DataFrame] = []
    for resolution in resolutions:
        cluster_key = resolution_cluster_key(resolution)
        if (
            math.isclose(resolution, primary_resolution)
            and existing_primary_path is not None
            and Path(existing_primary_path).exists()
        ):
            table = pd.read_csv(existing_primary_path, sep="\t", dtype={"cluster": str})
            table = table.groupby("cluster", observed=True, sort=False).head(top_n).copy()
            table.insert(0, "cluster_key", cluster_key)
            table.insert(0, "resolution", resolution)
            records.append(table)
            continue

        logger.info("Exploratory Wilcoxon cluster markers: %s", cluster_key)
        result_key = "_integration_qc_review_rank_genes"
        sc.tl.rank_genes_groups(
            adata,
            groupby=cluster_key,
            method="wilcoxon",
            use_raw=adata.raw is not None,
            pts=True,
            key_added=result_key,
        )
        for cluster in adata.obs[cluster_key].cat.categories:
            table = sc.get.rank_genes_groups_df(
                adata,
                group=cluster,
                key=result_key,
            ).head(top_n)
            table.insert(0, "cluster", str(cluster))
            table.insert(0, "cluster_key", cluster_key)
            table.insert(0, "resolution", resolution)
            records.append(table)
        del adata.uns[result_key]
        gc.collect()
    return pd.concat(records, ignore_index=True)


def resolution_split_summary(
    transitions: pd.DataFrame,
    markers: pd.DataFrame,
    obs: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize whether each parent is stable or splits into marker-distinct children."""
    records: list[dict[str, Any]] = []
    lineage = obs["provisional_broad_lineage"].astype(str)
    for (parent_resolution, child_resolution, parent), frame in transitions.groupby(
        ["parent_resolution", "child_resolution", "parent_cluster"],
        observed=True,
        sort=False,
    ):
        substantial = frame[(frame["pct_parent"] >= 5) & (frame["n_cells"] >= 50)]
        child_sets: list[set[str]] = []
        child_programs: list[str] = []
        child_key = resolution_cluster_key(child_resolution)
        for child in substantial["child_cluster"].astype(str):
            marker_frame = markers[
                (markers["cluster_key"] == child_key)
                & (markers["cluster"].astype(str) == child)
            ]
            child_sets.append(set(marker_frame.head(20)["names"].astype(str)))
            child_mask = obs[child_key].astype(str) == child
            if child_mask.any():
                child_programs.append(str(lineage.loc[child_mask].value_counts().idxmax()))
        jaccards = []
        for left, right in combinations(child_sets, 2):
            union = left | right
            jaccards.append(len(left & right) / len(union) if union else float("nan"))
        dominant = float(frame["pct_parent"].max())
        if dominant >= 90:
            interpretation = "stable_single_child"
        elif len(substantial) >= 2 and (
            len(set(child_programs)) >= 2
            or (jaccards and float(np.nanmean(jaccards)) < 0.25)
        ):
            interpretation = "candidate_marker_distinct_split"
        elif len(substantial) >= 2:
            interpretation = "possible_overpartition_review"
        else:
            interpretation = "minor_fragmentation"
        records.append(
            {
                "parent_resolution": parent_resolution,
                "child_resolution": child_resolution,
                "parent_cluster": str(parent),
                "parent_n_cells": int(frame["n_cells"].sum()),
                "n_child_clusters_any": len(frame),
                "n_substantial_children": len(substantial),
                "dominant_child_pct_parent": dominant,
                "substantial_child_clusters": ";".join(
                    substantial["child_cluster"].astype(str)
                ),
                "substantial_child_broad_programs": ";".join(child_programs),
                "mean_pairwise_top20_marker_jaccard": (
                    float(np.nanmean(jaccards)) if jaccards else float("nan")
                ),
                "split_interpretation": interpretation,
            }
        )
    return pd.DataFrame(records)


def cluster_qc_tables(
    obs: pd.DataFrame,
    *,
    cluster_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    composition: list[dict[str, Any]] = []
    for cluster, frame in obs.groupby(cluster_key, observed=True, sort=False):
        libraries = frame["library_id"].astype(str).value_counts()
        groups = frame["group"].astype(str).value_counts()
        records.append(
            {
                "cluster": str(cluster),
                "n_cells": len(frame),
                "library_max": str(libraries.idxmax()),
                "library_max_fraction": float(libraries.max() / len(frame)),
                "library_counts_json": _json_counts(frame["library_id"]),
                "group_counts_json": _json_counts(frame["group"]),
                "Y_fraction": float(groups.get("Y", 0) / len(frame)),
                "OC_fraction": float(groups.get("OC", 0) / len(frame)),
                "OT_fraction": float(groups.get("OT", 0) / len(frame)),
                "median_total_counts": float(frame["total_counts"].median()),
                "median_n_genes_by_counts": float(frame["n_genes_by_counts"].median()),
                "median_pct_counts_mt": float(frame["pct_counts_mt"].median()),
                "pct_mt_gt_25": 100 * float((frame["pct_counts_mt"] > 25).mean()),
                "pct_mt_gt_15": 100 * float((frame["pct_counts_mt"] > 15).mean()),
                "median_doublet_score": float(frame["doublet_score"].median()),
                "max_doublet_score": float(frame["doublet_score"].max()),
                "predicted_doublet_fraction": float(
                    _bool_array(frame["qc_doublet_auto"]).mean()
                ),
                "top1pct_doublet_score_fraction": float(
                    _bool_array(frame["qc_high_doublet_score_top1pct"]).mean()
                ),
                "low_genes_5mad_fraction": float(
                    _bool_array(frame["qc_low_genes_5mad"]).mean()
                ),
                "low_genes_absolute_fraction": float(
                    _bool_array(frame["qc_low_genes_absolute"]).mean()
                ),
            }
        )
        for dimension in ("library_id", "group"):
            counts = frame[dimension].astype(str).value_counts().sort_index()
            for value, count in counts.items():
                composition.append(
                    {
                        "cluster": str(cluster),
                        "dimension": dimension,
                        "value": str(value),
                        "count": int(count),
                        "fraction_cluster": float(count / len(frame)),
                    }
                )
    return pd.DataFrame(records), pd.DataFrame(composition)


def incompatible_lineage_coexpression(
    adata: ad.AnnData,
    markers: dict[str, Any],
    *,
    cluster_key: str,
    counts_layer: str,
) -> pd.DataFrame:
    """Screen raw-count co-detection of at least two markers from incompatible panels."""
    names = set(adata.var_names.astype(str))
    active: dict[str, np.ndarray] = {}
    genes_used: dict[str, list[str]] = {}
    for lineage in REVIEW_LINEAGES:
        genes = [gene for gene in markers["broad"][lineage]["positive"] if gene in names]
        genes_used[lineage] = genes
        matrix = adata[:, genes].layers[counts_layer]
        detected = np.asarray((matrix > 0).sum(axis=1)).ravel()
        active[lineage] = detected >= min(2, len(genes))

    records: list[dict[str, Any]] = []
    clusters = adata.obs[cluster_key].astype(str).to_numpy()
    for lineage_a, lineage_b in INCOMPATIBLE_LINEAGE_PAIRS:
        both = active[lineage_a] & active[lineage_b]
        global_fraction = float(both.mean())
        for cluster in pd.unique(clusters):
            mask = clusters == cluster
            count = int((both & mask).sum())
            fraction = count / int(mask.sum())
            records.append(
                {
                    "cluster": str(cluster),
                    "lineage_a": lineage_a,
                    "lineage_b": lineage_b,
                    "genes_a": ";".join(genes_used[lineage_a]),
                    "genes_b": ";".join(genes_used[lineage_b]),
                    "coexpressing_cells": count,
                    "cluster_cells": int(mask.sum()),
                    "coexpression_fraction": fraction,
                    "global_coexpression_fraction": global_fraction,
                    "enrichment_vs_global": (
                        fraction / global_fraction if global_fraction else float("nan")
                    ),
                }
            )
    return pd.DataFrame(records)


def build_cluster_review(
    obs: pd.DataFrame,
    cluster_qc: pd.DataFrame,
    program_scores: pd.DataFrame,
    markers: pd.DataFrame,
    coexpression: pd.DataFrame,
    *,
    cluster_key: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build conservative doublet/high-mt/cluster review tables without filtering."""
    review = cluster_qc.copy()
    program_columns = [
        "broad_program_1",
        "broad_program_2",
        "broad_program_1_score",
        "broad_program_2_score",
        "broad_program_margin",
    ]
    scores = program_scores[[cluster_key, *program_columns]].rename(
        columns={cluster_key: "cluster"}
    )
    scores["cluster"] = scores["cluster"].astype(str)
    review = review.merge(scores, on="cluster", how="left", validate="one_to_one")

    marker_view = markers[markers["cluster_key"] == cluster_key].copy()
    top_markers = (
        marker_view.groupby("cluster", observed=True, sort=False)["names"]
        .apply(lambda values: ";".join(values.astype(str).head(10)))
        .rename("top_markers")
        .reset_index()
    )
    top_markers["cluster"] = top_markers["cluster"].astype(str)
    review = review.merge(top_markers, on="cluster", how="left", validate="one_to_one")

    incompatible = (
        coexpression.sort_values(
            ["cluster", "coexpression_fraction", "enrichment_vs_global"],
            ascending=[True, False, False],
        )
        .groupby("cluster", observed=True, sort=False)
        .first()
        .reset_index()
    )
    incompatible["incompatible_pair_max"] = (
        incompatible["lineage_a"] + "+" + incompatible["lineage_b"]
    )
    review = review.merge(
        incompatible[
            [
                "cluster",
                "incompatible_pair_max",
                "coexpression_fraction",
                "enrichment_vs_global",
            ]
        ].rename(
            columns={
                "coexpression_fraction": "max_incompatible_coexpression_fraction",
                "enrichment_vs_global": "max_incompatible_coexpression_enrichment",
            }
        ),
        on="cluster",
        how="left",
        validate="one_to_one",
    )

    settings = config["integration_qc_review"]
    doublet_settings = settings["doublet_review"]
    low_quality_settings = settings["low_quality_review"]
    global_auto = float(_bool_array(obs["qc_doublet_auto"]).mean())
    global_mt25 = float((obs["pct_counts_mt"] > 25).mean())
    thresholds = {
        "doublet_score_median_threshold": float(
            obs["doublet_score"].quantile(doublet_settings["metric_quantile"])
        ),
        "total_counts_median_threshold": float(
            obs["total_counts"].quantile(doublet_settings["metric_quantile"])
        ),
        "n_genes_median_threshold": float(
            obs["n_genes_by_counts"].quantile(doublet_settings["metric_quantile"])
        ),
        "predicted_doublet_fraction_threshold": max(
            global_auto * float(doublet_settings["predicted_enrichment"]),
            float(doublet_settings["predicted_min_fraction"]),
        ),
        "incompatible_coexpression_fraction_threshold": float(
            doublet_settings["incompatible_min_fraction"]
        ),
        "low_umi_threshold": float(
            obs["total_counts"].quantile(low_quality_settings["low_rna_quantile"])
        ),
        "low_genes_threshold": float(
            obs["n_genes_by_counts"].quantile(low_quality_settings["low_rna_quantile"])
        ),
        "mt25_cluster_fraction_threshold": max(
            global_mt25 * float(low_quality_settings["mt25_enrichment"]),
            float(low_quality_settings["mt25_min_fraction"]),
        ),
        "low_genes_5mad_fraction_threshold": float(
            low_quality_settings["low_genes_5mad_min_fraction"]
        ),
        "library_max_fraction_threshold": float(settings["library_specific_fraction"]),
        "small_cluster_max_cells": int(settings["small_cluster_max_cells"]),
    }

    review["high_doublet_score_median"] = (
        review["median_doublet_score"] >= thresholds["doublet_score_median_threshold"]
    )
    review["predicted_doublet_enriched"] = (
        review["predicted_doublet_fraction"]
        >= thresholds["predicted_doublet_fraction_threshold"]
    )
    review["high_umi_median"] = (
        review["median_total_counts"] >= thresholds["total_counts_median_threshold"]
    )
    review["high_genes_median"] = (
        review["median_n_genes_by_counts"] >= thresholds["n_genes_median_threshold"]
    )
    review["incompatible_program_evidence"] = (
        review["max_incompatible_coexpression_fraction"]
        >= thresholds["incompatible_coexpression_fraction_threshold"]
    ) & (review["max_incompatible_coexpression_enrichment"] >= 1.5)
    review["doublet_suspicious"] = review[
        [
            "high_doublet_score_median",
            "predicted_doublet_enriched",
            "high_umi_median",
            "high_genes_median",
            "incompatible_program_evidence",
        ]
    ].all(axis=1)

    review["mt25_enriched"] = (
        review["pct_mt_gt_25"] / 100 >= thresholds["mt25_cluster_fraction_threshold"]
    )
    review["low_umi_median"] = (
        review["median_total_counts"] <= thresholds["low_umi_threshold"]
    )
    review["low_genes_median"] = (
        review["median_n_genes_by_counts"] <= thresholds["low_genes_threshold"]
    )
    review["low_gene_flag_enriched"] = (
        review["low_genes_5mad_fraction"]
        >= thresholds["low_genes_5mad_fraction_threshold"]
    )
    review["low_quality_suspicious"] = review["mt25_enriched"] & (
        review["low_umi_median"] | review["low_genes_median"]
    ) & review["low_gene_flag_enriched"]
    review["library_specific_suspicious"] = (
        review["library_max_fraction"] >= thresholds["library_max_fraction_threshold"]
    )

    comments = []
    for row in review.itertuples(index=False):
        if row.doublet_suspicious:
            comment = "doublet_suspicious"
        elif row.low_quality_suspicious:
            comment = "low_quality_suspicious"
        elif row.n_cells <= thresholds["small_cluster_max_cells"]:
            if (
                pd.notna(row.broad_program_1)
                and row.broad_program_margin > 0.1
                and not row.incompatible_program_evidence
            ):
                comment = "rare_population_candidate"
            else:
                comment = "uncertain"
        elif row.library_specific_suspicious:
            comment = "library_specific_suspicious"
        else:
            comment = "no_obvious_qc_issue"
        comments.append(comment)
    review["qc_comment"] = comments

    threshold_table = pd.DataFrame(
        [{"threshold": key, "value": value} for key, value in thresholds.items()]
    )
    doublet_review = review[
        [
            "cluster",
            "n_cells",
            "median_doublet_score",
            "predicted_doublet_fraction",
            "median_total_counts",
            "median_n_genes_by_counts",
            "incompatible_pair_max",
            "max_incompatible_coexpression_fraction",
            "max_incompatible_coexpression_enrichment",
            "high_doublet_score_median",
            "predicted_doublet_enriched",
            "high_umi_median",
            "high_genes_median",
            "incompatible_program_evidence",
            "doublet_suspicious",
        ]
    ].copy()
    high_mt_review = review[
        [
            "cluster",
            "n_cells",
            "broad_program_1",
            "median_total_counts",
            "median_n_genes_by_counts",
            "median_pct_counts_mt",
            "pct_mt_gt_15",
            "pct_mt_gt_25",
            "low_genes_5mad_fraction",
            "mt25_enriched",
            "low_umi_median",
            "low_genes_median",
            "low_gene_flag_enriched",
            "low_quality_suspicious",
        ]
    ].copy()
    high_mt_review["high_mt_interpretation"] = "not_cluster_enriched"
    high_mt_review.loc[
        high_mt_review["mt25_enriched"],
        "high_mt_interpretation",
    ] = "uncertain_high_mt_review"
    biological_high_mt = (
        high_mt_review["mt25_enriched"]
        & ~high_mt_review["low_umi_median"]
        & ~high_mt_review["low_genes_median"]
        & high_mt_review["broad_program_1"].notna()
    )
    high_mt_review.loc[
        biological_high_mt,
        "high_mt_interpretation",
    ] = "biological_high_mt_candidate"
    high_mt_review.loc[
        high_mt_review["low_quality_suspicious"],
        "high_mt_interpretation",
    ] = "low_quality_suspicious"
    return review, threshold_table, doublet_review, high_mt_review


def integrity_tables(
    adata: ad.AnnData,
    reference_path: str | Path,
    *,
    counts_layer: str,
    marker_genes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare exact sparse arrays with the saved unintegrated reference object."""
    current_counts = sparse_matrix_audit(adata.layers[counts_layer])
    current_x = sparse_matrix_audit(adata.X)
    reference = sc.read_h5ad(reference_path)
    if not adata.obs_names.equals(reference.obs_names):
        raise ValueError("Unintegrated reference cell order differs from clustered object")
    if not adata.var_names.equals(reference.var_names):
        raise ValueError("Unintegrated reference feature order differs from clustered object")
    reference_counts = sparse_matrix_audit(reference.layers[counts_layer])
    reference_x = sparse_matrix_audit(reference.X)
    rows = []
    for object_name, matrix_name, audit in (
        ("unintegrated_reference", "X", reference_x),
        ("harmony_clustered", "X", current_x),
        ("unintegrated_reference", counts_layer, reference_counts),
        ("harmony_clustered", counts_layer, current_counts),
    ):
        rows.append({"object": object_name, "matrix": matrix_name, **audit})
    integrity = pd.DataFrame(rows)
    integrity["matches_reference"] = [
        True,
        current_x["sha256_sparse_arrays"] == reference_x["sha256_sparse_arrays"],
        True,
        current_counts["sha256_sparse_arrays"]
        == reference_counts["sha256_sparse_arrays"],
    ]

    genes = [gene for gene in marker_genes if gene in set(adata.var_names)]
    current_matrix = adata[:, genes].X
    reference_matrix = reference[:, genes].X
    current_means = np.asarray(current_matrix.mean(axis=0)).ravel()
    reference_means = np.asarray(reference_matrix.mean(axis=0)).ravel()
    current_detected = np.asarray((current_matrix > 0).mean(axis=0)).ravel()
    reference_detected = np.asarray((reference_matrix > 0).mean(axis=0)).ravel()
    expression = pd.DataFrame(
        {
            "gene": genes,
            "unintegrated_mean_log_expression": reference_means,
            "harmony_object_mean_log_expression": current_means,
            "absolute_mean_difference": np.abs(current_means - reference_means),
            "unintegrated_detected_fraction": reference_detected,
            "harmony_object_detected_fraction": current_detected,
            "absolute_detected_fraction_difference": np.abs(
                current_detected - reference_detected
            ),
        }
    )
    del reference
    gc.collect()
    return integrity, expression


def _set_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def _save_figure(figure: plt.Figure, base: Path) -> None:
    figure.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _scatter_categorical(
    axis: plt.Axes,
    coordinates: np.ndarray,
    labels: np.ndarray,
    palette: dict[str, str],
    *,
    title: str,
    legend: bool,
) -> None:
    for label in palette:
        mask = labels == label
        if mask.any():
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=0.35,
                color=palette[label],
                label=label,
                linewidths=0,
                alpha=0.8,
                rasterized=True,
            )
    axis.set_title(title)
    axis.set(xticks=[], yticks=[], xlabel="UMAP1", ylabel="UMAP2")
    if legend:
        axis.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            markerscale=8,
            fontsize=6,
        )


def plot_umap_library_group(adata: ad.AnnData, output_base: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    library_labels = adata.obs["library_id"].astype(str).to_numpy()
    group_labels = adata.obs["group"].astype(str).to_numpy()
    libraries = _category_order(library_labels)
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(libraries)))
    library_palette = dict(zip(libraries, colors, strict=True))
    for column, (basis, state) in enumerate(
        (("X_umap_unintegrated", "Unintegrated"), ("X_umap", "Harmony"))
    ):
        coordinates = np.asarray(adata.obsm[basis])
        _scatter_categorical(
            axes[0, column],
            coordinates,
            library_labels,
            library_palette,
            title=f"{state}: library",
            legend=column == 1,
        )
        _scatter_categorical(
            axes[1, column],
            coordinates,
            group_labels,
            GROUP_PALETTE,
            title=f"{state}: group (descriptive, not mixing target)",
            legend=column == 1,
        )
    figure.suptitle("Library/group structure before and after Harmony", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(figure, output_base)


def plot_umap_lineage(adata: ad.AnnData, output_base: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    labels = adata.obs["provisional_broad_lineage"].astype(str).to_numpy()
    for axis, (basis, state) in zip(
        axes,
        (("X_umap_unintegrated", "Unintegrated"), ("X_umap", "Harmony")),
        strict=True,
    ):
        _scatter_categorical(
            axis,
            np.asarray(adata.obsm[basis]),
            labels,
            LINEAGE_PALETTE,
            title=f"{state}: provisional broad programs",
            legend=state == "Harmony",
        )
    figure.suptitle(
        "Broad programs are provisional marker references, not final labels",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(figure, output_base)


def plot_marker_program_pairs(adata: ad.AnnData, output_base: Path) -> None:
    figure, axes = plt.subplots(len(REVIEW_LINEAGES), 2, figsize=(9, 3.1 * len(REVIEW_LINEAGES)))
    for row, lineage in enumerate(REVIEW_LINEAGES):
        values = adata.obs[f"review_program_mean__{lineage}"].to_numpy(dtype=float)
        vmax = float(np.quantile(values, 0.99))
        for column, (basis, state) in enumerate(
            (("X_umap_unintegrated", "Unintegrated"), ("X_umap", "Harmony"))
        ):
            coordinates = np.asarray(adata.obsm[basis])
            order = np.argsort(values)
            scatter = axes[row, column].scatter(
                coordinates[order, 0],
                coordinates[order, 1],
                c=values[order],
                s=0.3,
                linewidths=0,
                cmap="viridis",
                vmin=0,
                vmax=vmax,
                rasterized=True,
            )
            axes[row, column].set_title(f"{lineage} — {state}")
            axes[row, column].set(xticks=[], yticks=[])
            figure.colorbar(scatter, ax=axes[row, column], fraction=0.035, pad=0.02)
    figure.suptitle("Identical marker-program expression projected on both embeddings", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    _save_figure(figure, output_base)


def plot_mixing_change(change: pd.DataFrame, output_base: Path) -> None:
    view = change[change["entity_type"].isin(["group", "lineage_group"])].copy()
    view = view.sort_values(["entity_type", "entity_id"])
    figure, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.22 * len(view))))
    colors = view["entity_type"].map({"group": "#4C78A8", "lineage_group": "#B0B0B0"})
    y = np.arange(len(view))
    axes[0].barh(y, view["median_delta_same_library_fraction"], color=colors)
    axes[0].axvline(0, color="black", lw=0.6)
    axes[0].set_title("Harmony − unintegrated\nsame-library fraction")
    axes[0].set_xlabel("Median paired change (lower = improved mixing)")
    axes[1].barh(y, view["median_delta_library_entropy"], color=colors)
    axes[1].axvline(0, color="black", lw=0.6)
    axes[1].set_title("Harmony − unintegrated\nnormalized library entropy")
    axes[1].set_xlabel("Median paired change (higher = improved mixing)")
    labels = view["entity_id"].str.replace("|", " / ", regex=False)
    for axis in axes:
        axis.set_yticks(y, labels if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Within-group and lineage×group library mixing", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(figure, output_base)


def plot_biological_structure(summary: pd.DataFrame, output_base: Path) -> None:
    view = summary[
        (summary["dimension"].isin(["broad_lineage", "group_within_lineage"]))
        & (summary["entity_id"] != "all")
    ].copy()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, dimension, title in (
        (axes[0], "broad_lineage", "Broad-lineage neighbor purity"),
        (axes[1], "group_within_lineage", "Group locality within lineage"),
    ):
        subset = view[view["dimension"] == dimension]
        sns.pointplot(
            data=subset,
            x="same_neighbor_fraction_median",
            y="entity_id",
            hue="embedding_state",
            hue_order=["unintegrated", "harmony"],
            palette=["#9D9D9D", "#4C78A8"],
            markers=["o", "s"],
            linestyles="none",
            ax=axis,
        )
        axis.set(xlim=(0, 1), xlabel="Median same-label neighbor fraction", ylabel="")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Biological-structure diagnostics (descriptive only)", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(figure, output_base)


def plot_transition_heatmaps(transitions: pd.DataFrame, output_base: Path) -> None:
    pairs = transitions[["parent_resolution", "child_resolution"]].drop_duplicates()
    figure, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5), squeeze=False)
    for axis, pair in zip(axes.flat, pairs.itertuples(index=False), strict=True):
        frame = transitions[
            (transitions["parent_resolution"] == pair.parent_resolution)
            & (transitions["child_resolution"] == pair.child_resolution)
        ]
        matrix = frame.pivot(
            index="parent_cluster",
            columns="child_cluster",
            values="pct_parent",
        ).fillna(0)
        matrix = matrix.reindex(
            index=_category_order(matrix.index),
            columns=_category_order(matrix.columns),
        )
        sns.heatmap(matrix, cmap="Blues", vmin=0, vmax=100, ax=axis, cbar=False)
        axis.set_title(f"{pair.parent_resolution:g} → {pair.child_resolution:g}")
        axis.set(xlabel="Child cluster", ylabel="Parent cluster")
    figure.suptitle("Leiden transitions (% of each parent cluster)", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(figure, output_base)


def plot_resolution_comparison(
    adata: ad.AnnData,
    resolutions: list[float],
    output_base: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 9))
    coordinates = np.asarray(adata.obsm["X_umap"])
    for axis, resolution in zip(axes.flat, resolutions, strict=True):
        labels = adata.obs[resolution_cluster_key(resolution)].astype(str).to_numpy()
        categories = _category_order(labels)
        colors = plt.get_cmap("turbo")(np.linspace(0, 1, len(categories)))
        palette = dict(zip(categories, colors, strict=True))
        _scatter_categorical(
            axis,
            coordinates,
            labels,
            palette,
            title=f"Leiden {resolution:g}: {len(categories)} clusters",
            legend=False,
        )
    figure.suptitle("Harmony UMAP with candidate Leiden resolutions", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(figure, output_base)


def plot_qc_candidates(adata: ad.AnnData, output_base: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 9))
    coordinates = np.asarray(adata.obsm["X_umap"])
    panels = (
        ("pct_counts_mt", "Mitochondrial percentage", "viridis", 0.99),
        ("doublet_score", "Scrublet score", "magma", 0.995),
        ("qc_high_doublet_score_top1pct", "Top 1% score within library", None, None),
        ("qc_low_genes_5mad", "Low genes: 5-MAD diagnostic", None, None),
    )
    for axis, (column, title, cmap, quantile) in zip(axes.flat, panels, strict=True):
        values = adata.obs[column]
        if cmap is None:
            labels = values.fillna(False).astype(bool).map({False: "No", True: "Yes"}).to_numpy()
            _scatter_categorical(
                axis,
                coordinates,
                labels,
                {"No": "#D9D9D9", "Yes": "#D62728"},
                title=title,
                legend=True,
            )
        else:
            numeric = values.to_numpy(dtype=float)
            order = np.argsort(numeric)
            scatter = axis.scatter(
                coordinates[order, 0],
                coordinates[order, 1],
                c=numeric[order],
                s=0.35,
                linewidths=0,
                cmap=cmap,
                vmin=float(np.quantile(numeric, 0.01)),
                vmax=float(np.quantile(numeric, quantile)),
                rasterized=True,
            )
            axis.set_title(title)
            axis.set(xticks=[], yticks=[])
            figure.colorbar(scatter, ax=axis, fraction=0.035, pad=0.02)
    figure.suptitle("Stage-1 QC candidates on Harmony UMAP", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(figure, output_base)


def plot_cluster_qc_heatmap(review: pd.DataFrame, output_base: Path) -> None:
    metrics = [
        "median_total_counts",
        "median_n_genes_by_counts",
        "median_pct_counts_mt",
        "pct_mt_gt_25",
        "median_doublet_score",
        "predicted_doublet_fraction",
        "top1pct_doublet_score_fraction",
        "low_genes_5mad_fraction",
        "library_max_fraction",
    ]
    frame = review.set_index("cluster")[metrics].astype(float)
    z = (frame - frame.mean(axis=0)) / frame.std(axis=0).replace(0, np.nan)
    z = z.reindex(_category_order(z.index))
    figure, axis = plt.subplots(figsize=(9, max(5, 0.25 * len(z))))
    sns.heatmap(z, cmap="vlag", center=0, linewidths=0.2, ax=axis)
    axis.set_title("Cluster QC metrics (column-wise z scores; diagnostic only)")
    axis.set(xlabel="QC metric", ylabel="Leiden 0.5 cluster")
    figure.tight_layout()
    _save_figure(figure, output_base)


def plot_small_cluster_review(
    adata: ad.AnnData,
    small_clusters: list[str],
    output_base: Path,
    *,
    cluster_key: str,
) -> None:
    figure, axes = plt.subplots(1, len(small_clusters), figsize=(5 * len(small_clusters), 4.5))
    axes_array = np.atleast_1d(axes)
    coordinates = np.asarray(adata.obsm["X_umap"])
    labels = adata.obs[cluster_key].astype(str).to_numpy()
    for axis, cluster in zip(axes_array, small_clusters, strict=True):
        mask = labels == cluster
        axis.scatter(
            coordinates[~mask, 0],
            coordinates[~mask, 1],
            s=0.25,
            color="#D9D9D9",
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=8,
            color="#D62728",
            linewidths=0,
            label=f"cluster {cluster} (n={mask.sum()})",
        )
        axis.legend(loc="best", fontsize=7)
        axis.set(xticks=[], yticks=[], title=f"Small cluster {cluster}")
    figure.suptitle("Small clusters are highlighted for review, not deletion", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(figure, output_base)


def build_review_report(
    mixing: pd.DataFrame,
    mixing_change: pd.DataFrame,
    biology: pd.DataFrame,
    cluster_sizes: pd.DataFrame,
    split_summary: pd.DataFrame,
    cluster_review: pd.DataFrame,
    integrity: pd.DataFrame,
) -> str:
    group_change = mixing_change[mixing_change["entity_type"] == "group"]
    lineage_group_change = mixing_change[
        mixing_change["entity_type"] == "lineage_group"
    ]
    lineage_biology = biology[
        (biology["dimension"] == "broad_lineage") & (biology["entity_id"] != "all")
    ]
    small = cluster_review.nsmallest(2, "n_cells")
    suspicious = cluster_review[cluster_review["qc_comment"] != "no_obvious_qc_issue"]
    integrity_view = integrity[
        ["object", "matrix", "shape", "nnz", "sha256_sparse_arrays", "matches_reference"]
    ].copy()
    integrity_view["shape"] = integrity_view["shape"].map(json.dumps)
    lines = [
        "# Harmony integration and secondary-QC review",
        "",
        "本报告只进行整合质量、分辨率和QC候选审核；没有删除任何细胞或cluster，也没有执行正式注释或差异分析。",
        "",
        "## Exact expression/count integrity",
        "",
        _markdown_table(integrity_view),
        "",
        "## Within-group library mixing change",
        "",
        _markdown_table(group_change),
        "",
        "## Lineage × group mixing change",
        "",
        _markdown_table(lineage_group_change),
        "",
        "## Broad-lineage neighbor structure",
        "",
        _markdown_table(lineage_biology),
        "",
        "## Cluster counts by resolution",
        "",
        _markdown_table(
            cluster_sizes.groupby("resolution", observed=True).agg(
                n_clusters=("cluster", "nunique"),
                minimum_cluster_size=("n_cells", "min"),
                median_cluster_size=("n_cells", "median"),
                maximum_cluster_size=("n_cells", "max"),
            ).reset_index()
        ),
        "",
        "## Resolution split triage",
        "",
        _markdown_table(split_summary),
        "",
        "## Two smallest Leiden 0.5 clusters",
        "",
        _markdown_table(small),
        "",
        "## QC comments requiring review",
        "",
        _markdown_table(suspicious),
        "",
        "## Scope guard",
        "",
        "Harmony后的Y/OC/OT完全混合不是成功标准。所有broad program均为临时marker参考；"
        "本轮没有运行CellTypist、正式细胞注释、cluster/cell删除、DEG、pseudobulk、GSEA、"
        "trajectory、CellChat或SCENIC。",
    ]
    return "\n".join(lines) + "\n"


def run_integration_qc_review(
    config: dict[str, Any],
    input_path: str | Path,
    *,
    unintegrated_reference: str | Path | None = None,
    cluster_marker_scores: str | Path | None = None,
    primary_markers: str | Path | None = None,
    allow_low_memory: bool = False,
) -> Path:
    """Run read-only Harmony, resolution, marker, and secondary-QC diagnostics."""
    require_compute_resources(config, allow_low_memory=allow_low_memory)
    paths = project_paths(config)
    logger = setup_logging("03_integration_qc_review", config)
    review_config = config["integration_qc_review"]
    counts_layer = config["ingest"]["counts_layer"]
    resolutions = [float(value) for value in config["clustering"]["resolutions"]]
    primary_resolution = float(config["clustering"]["primary_resolution"])
    primary_key = resolution_cluster_key(primary_resolution)
    input_path = Path(input_path)
    unintegrated_reference = Path(
        unintegrated_reference
        or paths["results"] / "exploratory" / "03_stage1_unintegrated_exploratory.h5ad"
    )
    cluster_marker_scores = Path(
        cluster_marker_scores or paths["results"] / "04_cluster_marker_scores.tsv"
    )
    primary_markers = Path(primary_markers or paths["results"] / "04_cluster_top_markers.tsv")
    output_dir = paths["results"] / "integration_qc_review"
    figure_dir = paths["figures"] / "integration_qc_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    input_stat = input_path.stat()
    logger.info("Reading clustered object: %s", input_path)
    adata = sc.read_h5ad(input_path)
    required_obs = {
        "library_id",
        "group",
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
        "doublet_score",
        "qc_doublet_auto",
        "qc_high_doublet_score_top1pct",
        "qc_low_genes_5mad",
        "qc_low_genes_absolute",
        *[resolution_cluster_key(value) for value in resolutions],
    }
    missing_obs = sorted(required_obs - set(adata.obs.columns))
    if missing_obs:
        raise KeyError(f"Clustered object is missing obs columns: {missing_obs}")
    for key in ("X_pca", "X_pca_harmony", "X_umap_unintegrated", "X_umap"):
        if key not in adata.obsm:
            raise KeyError(f"Clustered object is missing embedding: {key}")
    for key in ("neighbors_unintegrated_distances", "distances"):
        if key not in adata.obsp:
            raise KeyError(f"Clustered object is missing graph: {key}")
    if counts_layer not in adata.layers:
        raise KeyError(f"Clustered object is missing layer: {counts_layer}")

    score_table, lineage_mapping = load_primary_cluster_programs(
        cluster_marker_scores,
        cluster_key=primary_key,
    )
    attach_provisional_lineage(adata, cluster_key=primary_key, mapping=lineage_mapping)
    marker_config = load_yaml(paths["markers"])
    marker_programs = compute_marker_program_means(adata, marker_config)

    graphs = {
        "unintegrated": sparse.csr_matrix(adata.obsp["neighbors_unintegrated_distances"]),
        "harmony": sparse.csr_matrix(adata.obsp["distances"]),
    }
    logger.info("Computing within-stratum library mixing metrics")
    mixing, mixing_change = neighbor_mixing_tables(adata.obs, graphs)
    biology, lineage_confusion = biological_structure_tables(adata.obs, graphs)

    logger.info("Building Leiden size/transition tables")
    cluster_sizes, transitions = cluster_size_and_transition_tables(adata.obs, resolutions)
    marker_table = exploratory_marker_tables(
        adata,
        resolutions=resolutions,
        top_n=int(review_config["top_marker_n"]),
        existing_primary_path=primary_markers,
        primary_resolution=primary_resolution,
        logger=logger,
    )
    split_summary = resolution_split_summary(transitions, marker_table, adata.obs)

    logger.info("Mapping stage-1 QC candidates to Leiden %s", primary_resolution)
    cluster_qc, cluster_composition = cluster_qc_tables(
        adata.obs,
        cluster_key=primary_key,
    )
    coexpression = incompatible_lineage_coexpression(
        adata,
        marker_config,
        cluster_key=primary_key,
        counts_layer=counts_layer,
    )
    cluster_review, review_thresholds, doublet_review, high_mt_review = build_cluster_review(
        adata.obs,
        cluster_qc,
        score_table,
        marker_table,
        coexpression,
        cluster_key=primary_key,
        config=config,
    )
    small_clusters = cluster_review.nsmallest(2, "n_cells")["cluster"].astype(str).tolist()
    small_review = cluster_review[cluster_review["cluster"].isin(small_clusters)].copy()
    small_markers = marker_table[
        (marker_table["cluster_key"] == primary_key)
        & marker_table["cluster"].astype(str).isin(small_clusters)
    ].copy()
    small_scores = score_table[score_table[primary_key].astype(str).isin(small_clusters)].copy()

    logger.info("Validating exact expression and counts integrity")
    marker_genes = sorted(
        {
            gene
            for lineage in REVIEW_LINEAGES
            for gene in marker_config["broad"][lineage]["positive"]
        }
    )
    integrity, expression_integrity = integrity_tables(
        adata,
        unintegrated_reference,
        counts_layer=counts_layer,
        marker_genes=marker_genes,
    )
    if not integrity["matches_reference"].all():
        raise RuntimeError("Expression/count arrays changed between reference and Harmony object")
    if expression_integrity[
        ["absolute_mean_difference", "absolute_detected_fraction_difference"]
    ].to_numpy().max() != 0:
        raise RuntimeError("Marker expression changed between reference and Harmony object")

    logger.info("Writing review tables")
    tables = {
        "neighbor_mixing_summary.tsv": mixing,
        "neighbor_mixing_change.tsv": mixing_change,
        "biological_structure_summary.tsv": biology,
        "lineage_neighbor_confusion.tsv": lineage_confusion,
        "leiden_cluster_sizes.tsv": cluster_sizes,
        "leiden_cluster_transitions.tsv": transitions,
        "leiden_split_summary.tsv": split_summary,
        "exploratory_markers_all_resolutions.tsv": marker_table,
        "cluster_qc_summary_leiden_0.5.tsv": cluster_qc,
        "cluster_composition_long.tsv": cluster_composition,
        "incompatible_lineage_coexpression.tsv": coexpression,
        "cluster_marker_review.tsv": cluster_review,
        "review_thresholds.tsv": review_thresholds,
        "doublet_cluster_review.tsv": doublet_review,
        "high_mt_cluster_review.tsv": high_mt_review,
        "small_cluster_review.tsv": small_review,
        "small_cluster_markers.tsv": small_markers,
        "small_cluster_broad_scores.tsv": small_scores,
        "matrix_integrity.tsv": integrity,
        "marker_expression_integrity.tsv": expression_integrity,
        "marker_program_definitions.tsv": marker_programs,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, sep="\t", index=False)

    logger.info("Rendering Python diagnostic figures")
    _set_figure_style()
    plot_umap_library_group(adata, figure_dir / "umap_library_group_pre_post")
    plot_umap_lineage(adata, figure_dir / "umap_broad_lineage_pre_post")
    plot_marker_program_pairs(adata, figure_dir / "umap_marker_programs_pre_post")
    plot_mixing_change(mixing_change, figure_dir / "neighbor_mixing_change")
    plot_biological_structure(biology, figure_dir / "biological_structure_preservation")
    plot_transition_heatmaps(transitions, figure_dir / "leiden_transition_heatmaps")
    plot_resolution_comparison(adata, resolutions, figure_dir / "leiden_resolution_comparison")
    plot_qc_candidates(adata, figure_dir / "umap_stage1_qc_candidates")
    plot_cluster_qc_heatmap(cluster_review, figure_dir / "cluster_qc_heatmap")
    plot_small_cluster_review(
        adata,
        small_clusters,
        figure_dir / "small_cluster_review",
        cluster_key=primary_key,
    )

    report = build_review_report(
        mixing,
        mixing_change,
        biology,
        cluster_sizes,
        split_summary,
        cluster_review,
        integrity,
    )
    report_path = output_dir / "INTEGRATION_QC_REVIEW_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    stat_after = input_path.stat()
    if (stat_after.st_size, stat_after.st_mtime_ns) != (
        input_stat.st_size,
        input_stat.st_mtime_ns,
    ):
        raise RuntimeError("Forbidden side effect: input H5AD changed")
    logger.info("INTEGRATION_QC_REVIEW_OK: %s", report_path)
    return report_path
