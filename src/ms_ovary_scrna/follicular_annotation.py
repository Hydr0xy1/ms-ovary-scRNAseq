from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.cluster.hierarchy import leaves_list, linkage

from .follicular_subclustering import _atomic_write, _save_figure, _set_style
from .project import load_yaml, project_paths, require_compute_resources, setup_logging
from .stage1_exploratory import sparse_matrix_audit

CLUSTER_ANNOTATIONS: dict[str, dict[str, str]] = {
    "0": {
        "broad": "Granulosa",
        "subtype": "Granulosa_atretic_like",
        "state": "Atretic_like",
        "confidence": "High",
        "positive": "Pik3ip1/Itih5/Ghr with granulosa context",
        "negative": "No dominant lineage conflict",
        "qc": "None",
        "doublet": "None",
        "rationale": "Stable atretic-like program and dominant original granulosa source.",
    },
    "1": {
        "broad": "Granulosa",
        "subtype": "Granulosa_preantral_like",
        "state": "None",
        "confidence": "Medium",
        "positive": "Amh and preantral program",
        "negative": "Mixed original sources",
        "qc": "None",
        "doublet": "None",
        "rationale": (
            "Preantral-like granulosa candidate supported by Amh and the preantral program."
        ),
    },
    "2": {
        "broad": "Theca_steroidogenic",
        "subtype": "Theca",
        "state": "None",
        "confidence": "High",
        "positive": "Aldh1a1/Hsd3b1/Cyp11a1/Mgarp steroidogenic program",
        "negative": "No dominant lineage conflict",
        "qc": "None",
        "doublet": "None",
        "rationale": (
            "Large, stable Theca/steroidogenic population with coherent marker combination."
        ),
    },
    "3": {
        "broad": "Theca_steroidogenic",
        "subtype": "Theca_luteal_transition",
        "state": "None",
        "confidence": "Medium",
        "positive": "Theca program with substantial luteal program",
        "negative": "Luteal and granulosa markers co-occur at population level",
        "qc": "None",
        "doublet": "None",
        "rationale": (
            "Theca score remains above luteal score, while luteal markers "
            "indicate a transition-like state."
        ),
    },
    "4": {
        "broad": "Granulosa",
        "subtype": "Granulosa_preantral_like",
        "state": "None",
        "confidence": "High",
        "positive": "Gatm/Igfbp5/Kitl preantral program",
        "negative": "No dominant lineage conflict",
        "qc": "None",
        "doublet": "None",
        "rationale": "Coherent preantral-like granulosa program and highly pure original source.",
    },
    "5": {
        "broad": "Granulosa",
        "subtype": "Granulosa_atretic_like",
        "state": "Atretic_like",
        "confidence": "Low",
        "positive": "Atretic program is present",
        "negative": "Low RNA complexity and weak subtype separation",
        "qc": "Low-complexity/low-RNA concern",
        "doublet": "None",
        "rationale": "Atretic-like candidate, but low RNA complexity limits confidence.",
    },
    "6": {
        "broad": "Granulosa",
        "subtype": "Granulosa_low_complexity_candidate",
        "state": "Low_complexity",
        "confidence": "Medium",
        "positive": "Inha/Gja1 and granulosa-neighbor structure",
        "negative": "Mixed steroidogenic signal; housekeeping/ribosomal dominance",
        "qc": "Very low median UMI/genes",
        "doublet": "Not enriched",
        "rationale": (
            "Granulosa-like low-complexity candidate; retain for review rather than delete."
        ),
    },
    "7": {
        "broad": "Granulosa",
        "subtype": "Granulosa_antral_like",
        "state": "None",
        "confidence": "High",
        "positive": "Inhba/Inha/Hsd17b1/Gja1 antral program",
        "negative": "No dominant lineage conflict",
        "qc": "None",
        "doublet": "None",
        "rationale": "Stable antral-like granulosa program with strong granulosa identity.",
    },
    "8": {
        "broad": "Uncertain",
        "subtype": "Mixed_doublet_suspicious",
        "state": "Doublet_suspicious",
        "confidence": "Low",
        "positive": "Stromal matrix program with lineage co-detection in a subset",
        "negative": "Dcn/Col1a1/Col1a2 and strong doublet enrichment",
        "qc": "None",
        "doublet": "Strongly enriched predicted doublets",
        "rationale": (
            "Mixed stromal/doublet-enriched population; resolve at single-cell level before use."
        ),
    },
    "9": {
        "broad": "Theca_steroidogenic",
        "subtype": "Steroidogenic_uncertain",
        "state": "Low_complexity",
        "confidence": "Low",
        "positive": "Luteal/steroidogenic neighborhood",
        "negative": "Low RNA and elevated mitochondrial percentage",
        "qc": "Low-complexity and mt concern",
        "doublet": "None",
        "rationale": (
            "Steroidogenic-like low-complexity population requiring conservative interpretation."
        ),
    },
    "10": {
        "broad": "Luteal",
        "subtype": "Luteal_like",
        "state": "Cycling_subpopulation",
        "confidence": "High",
        "positive": "Ptgfr/Sfrp4/Lhcgr/Star/Cyp11a1 luteal program",
        "negative": "Small cycling subpopulation from original cluster 24",
        "qc": "None",
        "doublet": "Review original cluster-24 predicted doublets separately",
        "rationale": "Most explicit luteal-like program in the local compartment.",
    },
    "11": {
        "broad": "Granulosa",
        "subtype": "Granulosa_cycling",
        "state": "Cycling",
        "confidence": "High",
        "positive": "Top2a/Mki67/Cdk1/Ube2c/Cenpf",
        "negative": "Identity is partly masked by cell-cycle program",
        "qc": "None",
        "doublet": "None",
        "rationale": "Stable mitotic/cycling granulosa population.",
    },
    "12": {
        "broad": "Immune_mixed",
        "subtype": "Mixed_doublet_suspicious",
        "state": "Doublet_suspicious",
        "confidence": "Low",
        "positive": "Ptprc/Cd74/Lyz2 immune program",
        "negative": "Mixed source and local-compartment mismatch",
        "qc": "None",
        "doublet": "Predicted-doublet enrichment",
        "rationale": (
            "Immune contamination/mixed population retained for explicit exclusion review."
        ),
    },
    "13": {
        "broad": "Granulosa",
        "subtype": "Granulosa_preantral_like",
        "state": "Stress_high",
        "confidence": "Medium",
        "positive": "Kitl/Gatm-associated preantral signal",
        "negative": "Stress program dominates the local markers",
        "qc": "Stress-high",
        "doublet": "None",
        "rationale": "Preantral-like granulosa candidate with a prominent stress state.",
    },
    "14": {
        "broad": "Oocyte",
        "subtype": "Oocyte",
        "state": "None",
        "confidence": "Medium",
        "positive": "Mov10l1/Mael/Lhx8/Gtsf1/Figla/Ooep/Zp2 germ-cell program",
        "negative": "Only six cells",
        "qc": "No major QC abnormality",
        "doublet": "None",
        "rationale": (
            "Rare but coherent oocyte/germ-cell population; preserve with low abundance warning."
        ),
    },
}


def _dense_matrix(adata: ad.AnnData, genes: list[str], *, layer: str | None = None) -> np.ndarray:
    if not genes:
        return np.zeros((adata.n_obs, 0), dtype=float)
    view = adata[:, genes]
    matrix = view.layers[layer] if layer is not None else view.X
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    if sparse.issparse(matrix):
        return matrix.toarray().astype(float, copy=False)
    return np.asarray(matrix, dtype=float)


def score_marker_programs(
    adata: ad.AnnData,
    programs: dict[str, list[str]],
    *,
    counts_layer: str,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Score programs from log-normalized X and retain raw-count detection counts."""
    scores = pd.DataFrame(index=adata.obs_names)
    available: dict[str, list[str]] = {}
    for program, configured in programs.items():
        genes = [str(gene) for gene in configured if str(gene) in adata.var_names]
        available[program] = genes
        expression = _dense_matrix(adata, genes)
        counts = _dense_matrix(adata, genes, layer=counts_layer)
        if expression.shape[1]:
            mean = expression.mean(axis=0)
            std = expression.std(axis=0)
            std[std == 0] = 1.0
            scores[f"{program}_program_score"] = ((expression - mean) / std).mean(axis=1)
            scores[f"{program}_n_detected"] = (counts > 0).sum(axis=1).astype(int)
        else:
            scores[f"{program}_program_score"] = 0.0
            scores[f"{program}_n_detected"] = 0
    return scores, available


def marker_evidence_table(
    adata: ad.AnnData,
    programs: dict[str, list[str]],
    *,
    counts_layer: str,
    top_markers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Emit cluster-level expression and detection evidence for every configured marker."""
    rows: list[dict[str, Any]] = []
    cluster_metadata: dict[str, dict[str, Any]] = {}
    for cluster, frame in adata.obs.groupby("follicular_leiden", observed=True, sort=True):
        cluster = str(cluster)
        metadata = {
            "cluster": cluster,
            "n_cells": len(frame),
            "original_global_clusters": json.dumps(
                {
                    str(k): int(v)
                    for k, v in frame["original_leiden_cluster"]
                    .astype(str)
                    .value_counts()
                    .sort_index()
                    .items()
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "library_distribution": json.dumps(
                {
                    str(k): int(v)
                    for k, v in frame["library_id"].astype(str).value_counts().sort_index().items()
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "group_distribution": json.dumps(
                {
                    str(k): int(v)
                    for k, v in frame["group"].astype(str).value_counts().sort_index().items()
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        if top_markers is not None:
            marker_rows = top_markers[
                (top_markers["resolution"].eq(0.5))
                & (top_markers["cluster"].astype(str).eq(cluster))
            ]
            metadata["top_30_markers"] = ";".join(marker_rows["names"].astype(str).head(30))
        else:
            metadata["top_30_markers"] = ""
        cluster_metadata[cluster] = metadata
    clusters = adata.obs["follicular_leiden"].astype(str).to_numpy()
    for program, configured in programs.items():
        genes = [str(gene) for gene in configured if str(gene) in adata.var_names]
        expression = _dense_matrix(adata, genes)
        counts = _dense_matrix(adata, genes, layer=counts_layer)
        for cluster, metadata in cluster_metadata.items():
            mask = clusters == cluster
            cluster_expression = expression[mask]
            cluster_counts = counts[mask]
            for gene, mean, fraction in zip(
                genes,
                cluster_expression.mean(axis=0) if genes else [],
                (cluster_counts > 0).mean(axis=0) if genes else [],
                strict=True,
            ):
                rows.append(
                    {
                        **metadata,
                        "marker_program": program,
                        "marker": gene,
                        "mean_expression": float(mean),
                        "fraction_expressing": float(fraction),
                    }
                )
    return pd.DataFrame(rows)


def _within_library_flags(
    obs: pd.DataFrame,
    *,
    doublet_quantile: float,
    high_qc_quantile: float,
) -> pd.DataFrame:
    result = pd.DataFrame(index=obs.index)
    result["doublet_score_within_library_high"] = False
    result["high_umi_within_library"] = False
    result["high_genes_within_library"] = False
    for _, frame in obs.groupby("library_id", observed=True, sort=False):
        q_doublet = frame["doublet_score"].quantile(doublet_quantile)
        q_umi = frame["total_counts"].quantile(high_qc_quantile)
        q_genes = frame["n_genes_by_counts"].quantile(high_qc_quantile)
        result.loc[frame.index, "doublet_score_within_library_high"] = (
            frame["doublet_score"] >= q_doublet
        ).to_numpy()
        result.loc[frame.index, "high_umi_within_library"] = (
            frame["total_counts"] >= q_umi
        ).to_numpy()
        result.loc[frame.index, "high_genes_within_library"] = (
            frame["n_genes_by_counts"] >= q_genes
        ).to_numpy()
    result["doublet_qc_support"] = (
        result["doublet_score_within_library_high"]
        & result["high_umi_within_library"]
        & result["high_genes_within_library"]
    )
    return result


def classify_cluster8(
    scores: pd.DataFrame,
    obs: pd.DataFrame,
    *,
    lineage_percentile: float,
    doublet_quantile: float,
    high_qc_quantile: float,
) -> pd.DataFrame:
    """Classify local cluster-8 cells using simple multi-marker evidence."""
    required = [
        "stromal_program_score",
        "granulosa_program_score",
        "theca_program_score",
        "stromal_n_detected",
        "granulosa_n_detected",
        "theca_n_detected",
    ]
    missing = sorted(set(required) - set(scores.columns))
    if missing:
        raise KeyError(f"Missing cluster-8 score columns: {missing}")
    thresholds = {
        lineage: float(scores[f"{lineage}_program_score"].quantile(lineage_percentile / 100.0))
        for lineage in ("stromal", "granulosa", "theca")
    }
    result = pd.DataFrame(index=scores.index)
    stromal_high = (scores["stromal_program_score"] >= thresholds["stromal"]) & (
        scores["stromal_n_detected"] >= 2
    )
    granulosa_high = (scores["granulosa_program_score"] >= thresholds["granulosa"]) & (
        scores["granulosa_n_detected"] >= 2
    )
    theca_high = (scores["theca_program_score"] >= thresholds["theca"]) & (
        scores["theca_n_detected"] >= 2
    )
    result["stromal_high"] = stromal_high
    result["granulosa_high"] = granulosa_high
    result["theca_high"] = theca_high
    result["cluster8_single_cell_class"] = "uncertain"
    result.loc[stromal_high & ~granulosa_high & ~theca_high, "cluster8_single_cell_class"] = (
        "stromal_only_like"
    )
    prefer_granulosa = scores["granulosa_program_score"] >= scores["theca_program_score"]
    result.loc[stromal_high & granulosa_high & prefer_granulosa, "cluster8_single_cell_class"] = (
        "stromal_plus_granulosa"
    )
    result.loc[stromal_high & theca_high & ~prefer_granulosa, "cluster8_single_cell_class"] = (
        "stromal_plus_theca"
    )
    support = _within_library_flags(
        obs,
        doublet_quantile=doublet_quantile,
        high_qc_quantile=high_qc_quantile,
    )
    result = result.join(support)
    result["heterotypic_doublet_supported"] = (
        result["cluster8_single_cell_class"].isin(["stromal_plus_granulosa", "stromal_plus_theca"])
        & result["doublet_qc_support"]
    )
    result["predicted_doublet"] = obs["predicted_doublet"].fillna(False).astype(bool).to_numpy()
    result["stromal_threshold"] = thresholds["stromal"]
    result["granulosa_threshold"] = thresholds["granulosa"]
    result["theca_threshold"] = thresholds["theca"]
    return result


def cluster8_summary(
    adata: ad.AnnData,
    cluster8: pd.DataFrame,
) -> pd.DataFrame:
    mask = adata.obs["follicular_leiden"].astype(str).eq("8")
    rows: list[dict[str, Any]] = []
    for category, frame in cluster8.loc[mask].groupby("cluster8_single_cell_class", observed=True):
        rows.append(
            {
                "cluster": "8",
                "category": str(category),
                "n_cells": len(frame),
                "pct_cluster8": 100.0 * len(frame) / int(mask.sum()),
                "n_doublet_qc_support": int(frame["doublet_qc_support"].sum()),
                "n_heterotypic_doublet_supported": int(
                    frame["heterotypic_doublet_supported"].sum()
                ),
                "n_predicted_doublet": int(frame["predicted_doublet"].sum()),
                "median_total_counts": float(adata.obs.loc[frame.index, "total_counts"].median()),
                "median_n_genes_by_counts": float(
                    adata.obs.loc[frame.index, "n_genes_by_counts"].median()
                ),
                "median_doublet_score": float(adata.obs.loc[frame.index, "doublet_score"].median()),
            }
        )
    return pd.DataFrame(rows)


def cluster8_global_composition(adata: ad.AnnData) -> pd.DataFrame:
    mask = adata.obs["follicular_leiden"].astype(str).eq("8")
    frame = adata.obs.loc[mask]
    counts = frame["original_leiden_cluster"].astype(str).value_counts().sort_index()
    return pd.DataFrame(
        {
            "global_cluster": counts.index.astype(str),
            "local_cluster": "8",
            "n_cells": counts.to_numpy(dtype=int),
            "pct_cluster8": 100.0 * counts.to_numpy(dtype=float) / len(frame),
        }
    )


def cluster6_comparison(adata: ad.AnnData, scores: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "local_cluster6": ["6"],
        "preantral_reference": ["1", "4", "13"],
        "antral_reference": ["7"],
        "atretic_reference": ["0", "5"],
    }
    # ``_apply_cell_labels`` may already have attached these score columns to
    # ``adata.obs``.  Drop overlapping names before the index-aligned join so
    # reruns remain idempotent and pandas does not raise a duplicate-column
    # error.
    frame = adata.obs.drop(columns=scores.columns, errors="ignore").join(scores)
    rows: list[dict[str, Any]] = []
    for name, clusters in groups.items():
        mask = frame["follicular_leiden"].astype(str).isin(clusters)
        selected = frame.loc[mask]
        rows.append(
            {
                "comparison_group": name,
                "local_clusters": ";".join(clusters),
                "n_cells": len(selected),
                "median_granulosa_score": float(selected["granulosa_program_score"].median()),
                "median_preantral_score": float(selected["preantral_program_score"].median()),
                "median_antral_score": float(selected["antral_program_score"].median()),
                "median_atretic_score": float(selected["atretic_program_score"].median()),
                "median_total_counts": float(selected["total_counts"].median()),
                "median_n_genes_by_counts": float(selected["n_genes_by_counts"].median()),
                "median_pct_counts_mt": float(selected["pct_counts_mt"].median()),
                "library_distribution": json.dumps(
                    {
                        str(k): int(v)
                        for k, v in selected["library_id"]
                        .astype(str)
                        .value_counts()
                        .sort_index()
                        .items()
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def _cluster_score_summary(scores: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    joined = obs[["follicular_leiden"]].join(scores)
    return joined.groupby("follicular_leiden", observed=True).mean(numeric_only=True)


def _build_cluster_annotations(
    adata: ad.AnnData,
    scores: pd.DataFrame,
    cluster8: pd.DataFrame,
) -> pd.DataFrame:
    summary = _cluster_score_summary(scores, adata.obs)
    annotations = {key: value.copy() for key, value in CLUSTER_ANNOTATIONS.items()}
    cluster8_counts = cluster8.loc[
        adata.obs["follicular_leiden"].astype(str).eq("8"), "cluster8_single_cell_class"
    ].value_counts()
    stromal_only = int(cluster8_counts.get("stromal_only_like", 0))
    plus = int(
        cluster8_counts.get("stromal_plus_granulosa", 0)
        + cluster8_counts.get("stromal_plus_theca", 0)
    )
    if stromal_only > plus:
        annotations["8"].update(
            {
                "broad": "Stromal_fibroblast",
                "subtype": "Stromal_fibroblast",
                "confidence": "Medium",
                "positive": "Stromal-only-like cells are the largest interpretable category",
                "rationale": (
                    "Cluster 8 is stromal-rich but retains a doublet-suspicious mixed fraction."
                ),
            }
        )
    cluster3 = summary.loc["3"]
    if cluster3["luteal_program_score"] > cluster3["theca_program_score"]:
        annotations["3"].update(
            {
                "broad": "Luteal",
                "subtype": "Luteal_like",
                "rationale": (
                    "Luteal score exceeds Theca score, with residual "
                    "steroidogenic transition signal."
                ),
            }
        )
    cluster10 = summary.loc["10"]
    if cluster10["luteal_program_score"] <= cluster10["theca_program_score"]:
        annotations["10"].update(
            {
                "broad": "Theca_steroidogenic",
                "subtype": "Steroidogenic_uncertain",
                "confidence": "Medium",
                "rationale": "Luteal and Theca programs do not separate clearly in this cluster.",
            }
        )
    rows: list[dict[str, Any]] = []
    for cluster, frame in adata.obs.groupby("follicular_leiden", observed=True, sort=True):
        cluster = str(cluster)
        annotation = annotations[cluster]
        row = {
            "cluster": cluster,
            "n_cells": len(frame),
            "cell_type_broad_local": annotation["broad"],
            "cell_type_subtype_local": annotation["subtype"],
            "cell_state_local": annotation["state"],
            "annotation_confidence": annotation["confidence"],
            "positive_evidence": annotation["positive"],
            "negative_conflict": annotation["negative"],
            "qc_concern": annotation["qc"],
            "doublet_concern": annotation["doublet"],
            "annotation_rationale": annotation["rationale"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _apply_cell_labels(
    adata: ad.AnnData,
    cluster_annotations: pd.DataFrame,
    scores: pd.DataFrame,
    cluster8: pd.DataFrame,
) -> pd.DataFrame:
    lookup = cluster_annotations.set_index("cluster")
    clusters = adata.obs["follicular_leiden"].astype(str)
    for column in ["cell_type_broad_local", "cell_type_subtype_local", "cell_state_local"]:
        adata.obs[column] = clusters.map(lookup[column]).astype("string")
    adata.obs["annotation_confidence_local"] = clusters.map(lookup["annotation_confidence"]).astype(
        "string"
    )
    adata.obs = adata.obs.join(scores)
    cluster8_for_join = cluster8.rename(
        columns={
            column: f"cluster8_{column}"
            for column in cluster8.columns
            if not column.startswith("cluster8_")
        }
    )
    keep = [
        "cluster8_single_cell_class",
        "cluster8_stromal_high",
        "cluster8_granulosa_high",
        "cluster8_theca_high",
        "cluster8_doublet_score_within_library_high",
        "cluster8_high_umi_within_library",
        "cluster8_high_genes_within_library",
        "cluster8_doublet_qc_support",
        "cluster8_heterotypic_doublet_supported",
    ]
    adata.obs = adata.obs.join(cluster8_for_join[keep])
    original24 = adata.obs["original_leiden_cluster"].astype(str).eq("24")
    adata.obs["global24_cycling"] = original24
    original25 = adata.obs["original_leiden_cluster"].astype(str).eq("25")
    adata.obs["rare_global25_luteal_candidate"] = original25
    # Source-cluster-specific evidence is more precise than the surrounding
    # local-cluster label for these rare populations.
    adata.obs.loc[original24, "cell_type_broad_local"] = "Theca_steroidogenic"
    adata.obs.loc[original24, "cell_type_subtype_local"] = "Theca_cycling"
    adata.obs.loc[original24, "cell_state_local"] = "Cycling"
    adata.obs.loc[original24, "annotation_confidence_local"] = "Medium"
    adata.obs.loc[original25, "cell_type_broad_local"] = "Luteal"
    adata.obs.loc[original25, "cell_type_subtype_local"] = "Luteal_like"
    adata.obs.loc[original25, "cell_state_local"] = "None"
    adata.obs.loc[original25, "annotation_confidence_local"] = "Low"

    cluster8_mask = clusters.eq("8")
    stromal_only = cluster8_mask & adata.obs["cluster8_single_cell_class"].eq("stromal_only_like")
    mixed = cluster8_mask & adata.obs["cluster8_single_cell_class"].isin(
        ["stromal_plus_granulosa", "stromal_plus_theca"]
    )
    uncertain = cluster8_mask & adata.obs["cluster8_single_cell_class"].eq("uncertain")
    adata.obs.loc[stromal_only, "cell_type_broad_local"] = "Stromal_fibroblast"
    adata.obs.loc[stromal_only, "cell_type_subtype_local"] = "Stromal_fibroblast"
    adata.obs.loc[stromal_only, "cell_state_local"] = "None"
    adata.obs.loc[stromal_only, "annotation_confidence_local"] = "Medium"
    adata.obs.loc[mixed, "cell_type_broad_local"] = "Uncertain"
    adata.obs.loc[mixed, "cell_type_subtype_local"] = "Mixed_doublet_suspicious"
    adata.obs.loc[mixed, "cell_state_local"] = "Doublet_suspicious"
    adata.obs.loc[mixed, "annotation_confidence_local"] = "Low"
    adata.obs.loc[uncertain, "cell_type_broad_local"] = "Uncertain"
    adata.obs.loc[uncertain, "cell_type_subtype_local"] = "Uncertain"
    adata.obs.loc[uncertain, "cell_state_local"] = "None"
    adata.obs.loc[uncertain, "annotation_confidence_local"] = "Low"
    adata.obs["doublet_suspicious"] = adata.obs["predicted_doublet"].fillna(False).astype(
        bool
    ) | adata.obs["cluster8_heterotypic_doublet_supported"].fillna(False).astype(bool)
    label_columns = [
        "library_id",
        "group",
        "original_leiden_cluster",
        "follicular_leiden",
        "cell_type_broad_local",
        "cell_type_subtype_local",
        "cell_state_local",
        "annotation_confidence_local",
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
        "doublet_score",
        "predicted_doublet",
        "doublet_suspicious",
        "global24_cycling",
        "rare_global25_luteal_candidate",
    ]
    label_columns.extend(scores.columns.tolist())
    label_columns.extend(keep)
    labels = adata.obs[label_columns].copy()
    labels.insert(0, "cell_barcode", adata.obs_names.astype(str))
    return labels


def _marker_matrix_for_cells(
    adata: ad.AnnData, cell_mask: np.ndarray, genes: list[str], counts_layer: str
) -> pd.DataFrame:
    selected = adata[cell_mask, :]
    expression = _dense_matrix(selected, [gene for gene in genes if gene in selected.var_names])
    columns = [gene for gene in genes if gene in selected.var_names]
    return pd.DataFrame(expression, index=selected.obs_names, columns=columns)


def _json_counts(frame: pd.DataFrame, column: str) -> str:
    return json.dumps(
        {str(k): int(v) for k, v in frame[column].astype(str).value_counts().sort_index().items()},
        ensure_ascii=False,
        sort_keys=True,
    )


def _plot_labeled_umap(adata: ad.AnnData, column: str, output: Path, title: str) -> None:
    coords = np.asarray(adata.obsm["X_umap_follicular_harmony"])
    values = adata.obs[column].astype(str)
    categories = sorted(values.unique())
    colors = plt.get_cmap("tab20")
    figure, axis = plt.subplots(figsize=(10, 7))
    for i, category in enumerate(categories):
        mask = values.eq(category).to_numpy()
        axis.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=2,
            alpha=0.65,
            linewidths=0,
            color=colors(i % 20),
            label=f"{category} (n={mask.sum():,})",
        )
    axis.set(title=title, xlabel="UMAP1", ylabel="UMAP2")
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, markerscale=3)
    _save_figure(figure, output)


def _plot_score_heatmap(
    evidence: pd.DataFrame,
    programs: list[str],
    output: Path,
    title: str,
    clusters: list[str] | None = None,
) -> None:
    table = evidence[evidence["marker_program"].isin(programs)].pivot_table(
        index="cluster", columns="marker", values="mean_expression", aggfunc="mean"
    )
    table = table.reindex(sorted(table.index, key=int))
    if clusters is not None:
        table = table.reindex(clusters)
    figure, axis = plt.subplots(figsize=(max(10, table.shape[1] * 0.35), 7))
    image = axis.imshow(table.to_numpy(), aspect="auto", cmap="viridis")
    axis.set(title=title, xlabel="Marker", ylabel="Local cluster")
    axis.set_xticks(range(table.shape[1]), table.columns, rotation=90)
    axis.set_yticks(range(table.shape[0]), table.index)
    figure.colorbar(image, ax=axis, label="Mean log-normalized expression")
    _save_figure(figure, output)


def _plot_cluster8_review(adata: ad.AnnData, cluster8: pd.DataFrame, output: Path) -> None:
    coords = np.asarray(adata.obsm["X_umap_follicular_harmony"])
    mask = adata.obs["follicular_leiden"].astype(str).eq("8").to_numpy()
    values = cluster8.loc[mask, "cluster8_single_cell_class"].astype(str).to_numpy()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(coords[:, 0], coords[:, 1], s=1, color="#d9d9d9", linewidths=0)
    palette = {
        "stromal_only_like": "#1b9e77",
        "stromal_plus_granulosa": "#d95f02",
        "stromal_plus_theca": "#7570b3",
        "uncertain": "#e7298a",
    }
    for category, color in palette.items():
        selected = mask.copy()
        selected[mask] = values == category
        axes[0].scatter(
            coords[selected, 0],
            coords[selected, 1],
            s=6,
            color=color,
            label=f"{category} (n={selected.sum():,})",
        )
    axes[0].set_title("Local cluster 8 single-cell classes")
    axes[0].legend(frameon=False, markerscale=2, fontsize=8)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].scatter(coords[:, 0], coords[:, 1], s=1, color="#d9d9d9", linewidths=0)
    support = cluster8["heterotypic_doublet_supported"].to_numpy()
    selected = mask.copy()
    selected[mask] = support
    axes[1].scatter(
        coords[selected, 0],
        coords[selected, 1],
        s=7,
        color="#e41a1c",
        label=f"heterotypic doublet QC-supported (n={selected.sum():,})",
    )
    axes[1].set_title("Cluster 8 heterotypic-doublet support")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    _save_figure(figure, output)


def _plot_cell_marker_heatmap(
    adata: ad.AnnData,
    mask: np.ndarray,
    genes: list[str],
    output: Path,
    title: str,
) -> None:
    available = [gene for gene in genes if gene in adata.var_names]
    matrix = _dense_matrix(adata[mask, :], available)
    names = adata.obs_names[mask].astype(str)
    if matrix.shape[0] > 2 and np.any(np.std(matrix, axis=0) > 0):
        # Hierarchical linkage is quadratic in cell number.  Cluster 6 has
        # thousands of cells, so sort large groups deterministically instead
        # of attempting an O(n^2) linkage that can exhaust memory overnight.
        if matrix.shape[0] <= 500:
            order = leaves_list(linkage(matrix, method="average", metric="euclidean"))
        else:
            order = np.argsort(adata.obs.loc[mask, "total_counts"].to_numpy())
        matrix = matrix[order]
        names = names[order]
    # Keep the figure readable for large clusters; the complete per-cell table
    # remains available as TSV for detailed inspection.
    figure_height = min(40, max(4, int(mask.sum()) * 0.015))
    figure, axis = plt.subplots(figsize=(max(8, len(available) * 0.45), figure_height))
    image = axis.imshow(matrix, aspect="auto", cmap="magma")
    axis.set(title=title, xlabel="Marker", ylabel="Cell")
    axis.set_xticks(range(len(available)), available, rotation=90)
    axis.set_yticks(range(int(mask.sum())), names, fontsize=6)
    figure.colorbar(image, ax=axis, label="Log-normalized expression")
    _save_figure(figure, output)


def _annotation_report(
    cluster_annotations: pd.DataFrame,
    cluster8: pd.DataFrame,
    cluster6: pd.DataFrame,
    cluster25: pd.DataFrame,
    cluster14: pd.DataFrame,
    output: Path,
) -> None:
    lines = [
        "# Follicular/steroidogenic local annotation review",
        "",
        (
            "This report is descriptive and marker-guided. No cells were "
            "deleted and no Y/OC/OT comparison was performed."
        ),
        "",
        "## Recommended working labels",
        "",
        "| cluster | n_cells | broad | subtype | state | confidence |",
        "|---:|---:|---|---|---|---|",
    ]
    for row in cluster_annotations.itertuples(index=False):
        lines.append(
            f"| {row.cluster} | {row.n_cells:,} | {row.cell_type_broad_local} | "
            f"{row.cell_type_subtype_local} | {row.cell_state_local} | "
            f"{row.annotation_confidence} |"
        )
    lines.extend(["", "## Cluster 8 single-cell review", ""])
    lines.append(
        (
            "The four classes use a transparent rule: each lineage is high "
            "when its standardized program score is above the global 75th "
            "percentile and at least two markers are detected. "
            "`doublet_qc_support` additionally requires within-library "
            "doublet score at or above the 75th percentile and UMI/genes at "
            "or above the within-library median."
        )
    )
    lines.append("")
    lines.append(cluster8.to_markdown(index=False))
    lines.extend(["", "## Cluster 6 comparison", "", cluster6.to_markdown(index=False)])
    lines.extend(["", "## Rare populations", ""])
    lines.append(
        (
            f"Global cluster 25 contains {len(cluster25):,} cells and global "
            f"cluster 14 contains {len(cluster14):,} cells. Their per-cell "
            "marker tables are retained for manual review."
        )
    )
    lines.extend(
        [
            "",
            "## Scope boundary",
            "",
            (
                "No normalization, clustering, differential expression, "
                "pseudobulk, GSEA, trajectory, CellChat, SCENIC, or "
                "condition-level interpretation was performed in this "
                "stage."
            ),
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_follicular_annotation_review(
    config: dict[str, Any],
    input_path: str | Path,
    *,
    allow_low_memory: bool = False,
) -> Path:
    """Add local hierarchical labels and evidence tables to a separate subset object."""
    require_compute_resources(config, allow_low_memory=allow_low_memory)
    paths = project_paths(config)
    logger = setup_logging("05_follicular_annotation_review", config)
    settings = config["follicular_annotation_review"]
    input_path = Path(input_path).resolve()
    input_stat = input_path.stat()
    output_dir = paths["root"] / settings["output_dir"]
    figure_dir = paths["root"] / settings["figure_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading local follicular object: %s", input_path)
    adata = sc.read_h5ad(input_path)
    counts_layer = str(config["follicular_subclustering"]["counts_layer"])
    if counts_layer not in adata.layers:
        raise KeyError(f"Missing raw-count layer: {counts_layer}")
    if "follicular_leiden" not in adata.obs or "X_umap_follicular_harmony" not in adata.obsm:
        raise KeyError("Input must be the completed follicular subclustering object")
    counts_before = sparse_matrix_audit(adata.layers[counts_layer])

    marker_yaml = load_yaml(paths["markers"])
    configured = marker_yaml["follicular_annotation_review"]
    programs = {
        str(name): [str(gene) for gene in definition["positive"]]
        for name, definition in configured.items()
    }
    top_marker_path = output_dir / "follicular_cluster_markers.tsv"
    top_markers = pd.read_csv(top_marker_path, sep="\t") if top_marker_path.exists() else None
    logger.info("Scoring local annotation marker programs")
    scores, _ = score_marker_programs(adata, programs, counts_layer=counts_layer)
    cluster8 = classify_cluster8(
        scores,
        adata.obs,
        lineage_percentile=float(settings["lineage_high_percentile"]),
        doublet_quantile=float(settings["within_library_doublet_quantile"]),
        high_qc_quantile=float(settings["within_library_high_qc_quantile"]),
    )
    cluster8_mask = adata.obs["follicular_leiden"].astype(str).eq("8").to_numpy()
    cluster8_all = cluster8.copy()
    cluster8_all.loc[~cluster8_mask, "cluster8_single_cell_class"] = "not_cluster8"
    cluster8_summary_table = cluster8_summary(adata, cluster8)
    evidence = marker_evidence_table(
        adata,
        programs,
        counts_layer=counts_layer,
        top_markers=top_markers,
    )
    cluster_annotations = _build_cluster_annotations(adata, scores, cluster8)
    labels = _apply_cell_labels(adata, cluster_annotations, scores, cluster8_all)

    cluster6 = cluster6_comparison(adata, scores)
    cluster8_composition = cluster8_global_composition(adata)
    key_genes = {
        "granulosa": programs["granulosa"],
        "preantral": programs["preantral"],
        "antral": programs["antral"],
        "atretic": programs["atretic"],
        "cycling": programs["cycling"],
        "theca": programs["theca"],
        "luteal": programs["luteal"],
        "stromal": programs["stromal"],
        "immune": programs["immune"],
        "oocyte": programs["oocyte"],
    }

    rare25_mask = adata.obs["original_leiden_cluster"].astype(str).eq("25").to_numpy()
    rare14_mask = adata.obs["follicular_leiden"].astype(str).eq("14").to_numpy()
    rare_genes = key_genes["theca"] + key_genes["luteal"] + key_genes["granulosa"]
    rare25_expr = _marker_matrix_for_cells(
        adata, rare25_mask, list(dict.fromkeys(rare_genes)), counts_layer
    )
    rare25 = adata.obs.loc[
        rare25_mask,
        [
            "library_id",
            "group",
            "follicular_leiden",
            "total_counts",
            "n_genes_by_counts",
            "pct_counts_mt",
            "doublet_score",
            "predicted_doublet",
        ],
    ].copy()
    rare25.insert(0, "cell_barcode", rare25.index.astype(str))
    rare25 = rare25.join(
        scores.loc[
            rare25.index, ["theca_program_score", "luteal_program_score", "granulosa_program_score"]
        ]
    )
    rare25 = rare25.join(rare25_expr)
    rare14_expr = _marker_matrix_for_cells(adata, rare14_mask, key_genes["oocyte"], counts_layer)
    rare14 = adata.obs.loc[
        rare14_mask,
        [
            "library_id",
            "group",
            "original_leiden_cluster",
            "follicular_leiden",
            "total_counts",
            "n_genes_by_counts",
            "pct_counts_mt",
            "doublet_score",
            "predicted_doublet",
        ],
    ].copy()
    rare14.insert(0, "cell_barcode", rare14.index.astype(str))
    rare14 = rare14.join(scores.loc[rare14.index, ["oocyte_program_score", "oocyte_n_detected"]])
    rare14 = rare14.join(rare14_expr)

    theca_luteal = (
        evidence[evidence["marker_program"].isin(["theca", "luteal", "granulosa"])]
        .groupby(["cluster", "marker_program"], observed=True)
        .agg(
            n_cells=("n_cells", "first"),
            mean_expression=("mean_expression", "mean"),
            mean_fraction_expressing=("fraction_expressing", "mean"),
        )
        .reset_index()
    )
    theca_luteal = theca_luteal[theca_luteal["cluster"].astype(str).isin(["2", "3", "10"])].copy()
    rare25_summary = pd.DataFrame(
        [
            {
                "entity": "global_cluster_25",
                "n_cells": len(rare25),
                "mean_theca_score": float(rare25["theca_program_score"].mean()),
                "mean_luteal_score": float(rare25["luteal_program_score"].mean()),
                "mean_granulosa_score": float(rare25["granulosa_program_score"].mean()),
                "luteal_marker_consistency": float(
                    (
                        rare25_expr[[g for g in key_genes["luteal"] if g in rare25_expr]]
                        .gt(0)
                        .mean(axis=1)
                        >= 0.6
                    ).mean()
                ),
            }
        ]
    )

    # Required single-cell audit for local cluster 8.  The table is deliberately
    # one row per cell (rather than a four-row summary) so mixed stromal/
    # granulosa/theca evidence can be inspected without altering the object.
    cluster8_cells = adata.obs.loc[
        cluster8_mask,
        [
            "library_id",
            "group",
            "original_leiden_cluster",
            "follicular_leiden",
            "total_counts",
            "n_genes_by_counts",
            "pct_counts_mt",
            "doublet_score",
            "predicted_doublet",
        ],
    ].copy()
    cluster8_cells.insert(0, "cell_barcode", cluster8_cells.index.astype(str))
    cluster8_score_columns = [
        "stromal_program_score",
        "granulosa_program_score",
        "theca_program_score",
        "stromal_n_detected",
        "granulosa_n_detected",
        "theca_n_detected",
    ]
    cluster8_cells = cluster8_cells.join(scores.loc[cluster8_cells.index, cluster8_score_columns])
    cluster8_cells = cluster8_cells.join(
        cluster8.loc[cluster8_cells.index].drop(columns=["predicted_doublet"])
    )

    tables = {
        "follicular_local_cluster_evidence.tsv": evidence,
        "follicular_annotation_final_review.tsv": cluster_annotations,
        "follicular_cell_labels.tsv": labels,
        "cluster8_single_cell_review.tsv": cluster8_cells,
        "follicular_cluster8_summary.tsv": cluster8_summary_table,
        "follicular_cluster8_global_composition.tsv": cluster8_composition,
        "follicular_cluster6_comparison.tsv": cluster6,
        "follicular_theca_luteal_review.tsv": theca_luteal,
        "follicular_cluster25_rare_cells.tsv": rare25,
        "follicular_cluster25_summary.tsv": rare25_summary,
        "follicular_cluster14_oocyte_review.tsv": rare14,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, sep="\t", index=False)
    thresholds = pd.DataFrame(
        [
            {
                "lineage_high_percentile": float(settings["lineage_high_percentile"]),
                "within_library_doublet_quantile": float(
                    settings["within_library_doublet_quantile"]
                ),
                "within_library_high_qc_quantile": float(
                    settings["within_library_high_qc_quantile"]
                ),
                "counts_layer": counts_layer,
                "input_object": str(input_path),
            }
        ]
    )
    thresholds.to_csv(
        output_dir / "follicular_annotation_score_thresholds.tsv", sep="\t", index=False
    )

    _set_style()
    _plot_labeled_umap(
        adata,
        "cell_type_broad_local",
        figure_dir / "follicular_umap_broad_local",
        "Follicular/steroidogenic broad local annotation",
    )
    _plot_labeled_umap(
        adata,
        "cell_type_subtype_local",
        figure_dir / "follicular_umap_subtype_local",
        "Follicular/steroidogenic subtype local annotation",
    )
    sc_evidence = evidence.rename(columns={"mean_expression": "mean_log_normalized_expression"})
    from .follicular_subclustering import plot_marker_dotplot

    plot_marker_dotplot(
        sc_evidence.rename(columns={"marker_program": "program"}),
        figure_dir / "follicular_canonical_marker_dotplot",
    )
    _plot_score_heatmap(
        evidence,
        ["granulosa", "preantral", "antral", "atretic", "cycling"],
        figure_dir / "follicular_granulosa_marker_heatmap",
        "Granulosa subtype marker evidence",
    )
    _plot_score_heatmap(
        evidence,
        ["theca", "luteal"],
        figure_dir / "follicular_theca_luteal_marker_heatmap",
        "Theca/luteal marker evidence",
    )
    _plot_cluster8_review(adata, cluster8, figure_dir / "follicular_cluster8_annotation_review")
    _plot_cell_marker_heatmap(
        adata,
        adata.obs["follicular_leiden"].astype(str).eq("6").to_numpy(),
        key_genes["granulosa"] + key_genes["atretic"] + key_genes["theca"],
        figure_dir / "follicular_cluster6_annotation_review",
        "Local cluster 6 marker review",
    )
    _plot_cell_marker_heatmap(
        adata,
        rare14_mask,
        key_genes["oocyte"],
        figure_dir / "follicular_cluster14_oocyte_review",
        "Rare oocyte/germ-cell review",
    )
    _plot_cell_marker_heatmap(
        adata,
        rare25_mask,
        list(dict.fromkeys(rare_genes)),
        figure_dir / "follicular_cluster25_rare_cell_review",
        "Global cluster 25 rare-cell review",
    )
    _annotation_report(
        cluster_annotations,
        cluster8_summary_table,
        cluster6,
        rare25,
        rare14,
        output_dir / "FOLLICULAR_ANNOTATION_REVIEW.md",
    )

    adata.uns["follicular_annotation_review"] = {
        "source_object": str(input_path),
        "no_cells_deleted": True,
        "lineage_high_percentile": float(settings["lineage_high_percentile"]),
        "within_library_doublet_quantile": float(settings["within_library_doublet_quantile"]),
        "within_library_high_qc_quantile": float(settings["within_library_high_qc_quantile"]),
        "scope": "local follicular/steroidogenic annotation only; no condition comparison",
    }
    output = paths["root"] / settings["annotated_object"]
    logger.info("Writing annotated subset separately: %s", output)
    _atomic_write(adata, output)
    saved = sc.read_h5ad(output, backed="r")
    if saved.shape != adata.shape or counts_layer not in saved.layers:
        raise RuntimeError("Annotated subset failed shape/layer validation")
    saved_counts = saved.layers[counts_layer]
    if hasattr(saved_counts, "to_memory"):
        saved_counts = saved_counts.to_memory()
    if sparse_matrix_audit(saved_counts) != counts_before:
        raise RuntimeError("Annotated subset changed raw-count sparse arrays")
    for column in ("cell_type_broad_local", "cell_type_subtype_local", "cell_state_local"):
        if column not in saved.obs:
            raise RuntimeError(f"Annotated subset missing label column: {column}")
    saved.file.close()
    stat_after = input_path.stat()
    if (input_stat.st_size, input_stat.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns):
        raise RuntimeError("Input local subset changed during annotation review")
    logger.info("FOLLICULAR_ANNOTATION_REVIEW_OK: %s", output)
    return output
