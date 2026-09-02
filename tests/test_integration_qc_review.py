import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from ms_ovary_scrna.integration_qc_review import (
    build_cluster_review,
    cluster_size_and_transition_tables,
    incompatible_lineage_coexpression,
    neighbor_mixing_tables,
    resolution_cluster_key,
)


def _obs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "library_id": ["Y_1", "Y_1", "Y_2", "Y_2", "OC_1", "OC_1"],
            "group": ["Y", "Y", "Y", "Y", "OC", "OC"],
            "provisional_broad_lineage": [
                "Immune",
                "Immune",
                "Immune",
                "Ovarian_epithelium",
                "Granulosa",
                "Granulosa",
            ],
            "leiden_0.3": ["0", "0", "0", "0", "1", "1"],
            "leiden_0.5": ["0", "0", "0", "1", "2", "2"],
            "leiden_0.8": ["0", "0", "1", "2", "3", "3"],
            "total_counts": [100, 110, 120, 900, 300, 320],
            "n_genes_by_counts": [50, 60, 70, 500, 200, 210],
            "pct_counts_mt": [30, 32, 28, 2, 3, 4],
            "doublet_score": [0.1, 0.2, 0.3, 0.9, 0.05, 0.06],
            "qc_doublet_auto": [False, False, False, True, False, False],
            "qc_high_doublet_score_top1pct": [False, False, False, True, False, False],
            "qc_low_genes_5mad": [True, True, True, False, False, False],
            "qc_low_genes_absolute": [True, True, True, False, False, False],
        },
        index=[f"cell_{index}" for index in range(6)],
    )


def _directed_graph(neighbors: list[list[int]]) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    for row, values in enumerate(neighbors):
        rows.extend([row] * len(values))
        columns.extend(values)
    return sparse.csr_matrix(
        (np.ones(len(rows)), (rows, columns)),
        shape=(len(neighbors), len(neighbors)),
    )


def test_resolution_key_preserves_one_decimal_stage3_convention() -> None:
    assert resolution_cluster_key(0.5) == "leiden_0.5"
    assert resolution_cluster_key(1.0) == "leiden_1.0"


def test_group_conditioned_mixing_detects_improvement() -> None:
    obs = _obs()
    before = _directed_graph([[1], [0], [3], [2], [5], [4]])
    after = _directed_graph([[1, 2], [0, 2], [0, 3], [0, 2], [5, 4], [4, 5]])

    summary, change = neighbor_mixing_tables(
        obs,
        {"unintegrated": before, "harmony": after},
    )

    y_change = change[
        (change["entity_type"] == "group") & (change["entity_id"] == "Y")
    ].iloc[0]
    assert y_change["median_delta_same_library_fraction"] < 0
    assert y_change["median_delta_library_entropy"] > 0
    assert set(summary["embedding_state"]) == {"unintegrated", "harmony"}


def test_cluster_transitions_preserve_exact_counts() -> None:
    sizes, transitions = cluster_size_and_transition_tables(_obs(), [0.3, 0.5, 0.8])

    assert sizes[sizes["resolution"] == 0.5]["n_cells"].sum() == 6
    for pair in [(0.3, 0.5), (0.5, 0.8)]:
        view = transitions[
            (transitions["parent_resolution"] == pair[0])
            & (transitions["child_resolution"] == pair[1])
        ]
        assert view["n_cells"].sum() == 6


def test_incompatible_marker_codetection_uses_raw_counts() -> None:
    obs = _obs()
    genes = ["Ptprc", "Tyrobp", "Krt18", "Epcam", "Foxl2", "Fshr"]
    counts = sparse.csr_matrix(
        np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1],
            ],
            dtype=np.float32,
        )
    )
    adata = ad.AnnData(X=counts.copy(), obs=obs, var=pd.DataFrame(index=genes))
    adata.layers["counts"] = counts
    markers = {
        "broad": {
            "Granulosa": {"positive": ["Foxl2", "Fshr"]},
            "Theca_steroidogenic": {"positive": ["Foxl2", "Fshr"]},
            "Stromal_fibroblast": {"positive": ["Foxl2", "Fshr"]},
            "Endothelial": {"positive": ["Foxl2", "Fshr"]},
            "Smooth_muscle": {"positive": ["Foxl2", "Fshr"]},
            "Immune": {"positive": ["Ptprc", "Tyrobp"]},
            "Ovarian_epithelium": {"positive": ["Krt18", "Epcam"]},
        }
    }

    table = incompatible_lineage_coexpression(
        adata,
        markers,
        cluster_key="leiden_0.5",
        counts_layer="counts",
    )
    row = table[
        (table["cluster"] == "1")
        & (table["lineage_a"] == "Immune")
        & (table["lineage_b"] == "Ovarian_epithelium")
    ].iloc[0]
    assert row["coexpression_fraction"] == 1.0


def test_cluster_review_requires_concordant_doublet_evidence() -> None:
    obs = _obs()
    cluster_qc = pd.DataFrame(
        {
            "cluster": ["0", "1"],
            "n_cells": [3, 3],
            "library_max": ["Y_1", "Y_2"],
            "library_max_fraction": [0.67, 0.67],
            "library_counts_json": ["{}", "{}"],
            "group_counts_json": ["{}", "{}"],
            "Y_fraction": [1.0, 0.33],
            "OC_fraction": [0.0, 0.67],
            "OT_fraction": [0.0, 0.0],
            "median_total_counts": [110, 900],
            "median_n_genes_by_counts": [60, 500],
            "median_pct_counts_mt": [30, 3],
            "pct_mt_gt_25": [100, 0],
            "pct_mt_gt_15": [100, 0],
            "median_doublet_score": [0.2, 0.9],
            "max_doublet_score": [0.3, 0.9],
            "predicted_doublet_fraction": [0.0, 1.0],
            "top1pct_doublet_score_fraction": [0.0, 1.0],
            "low_genes_5mad_fraction": [1.0, 0.0],
            "low_genes_absolute_fraction": [1.0, 0.0],
        }
    )
    program_scores = pd.DataFrame(
        {
            "leiden_0.5": ["0", "1"],
            "broad_program_1": ["Immune", "Immune"],
            "broad_program_2": ["Granulosa", "Ovarian_epithelium"],
            "broad_program_1_score": [1.0, 1.0],
            "broad_program_2_score": [0.0, 0.8],
            "broad_program_margin": [1.0, 0.2],
        }
    )
    markers = pd.DataFrame(
        {
            "cluster_key": ["leiden_0.5"] * 4,
            "cluster": ["0", "0", "1", "1"],
            "names": ["Ptprc", "Tyrobp", "Ptprc", "Krt18"],
        }
    )
    coexpression = pd.DataFrame(
        {
            "cluster": ["0", "1"],
            "lineage_a": ["Immune", "Immune"],
            "lineage_b": ["Ovarian_epithelium", "Ovarian_epithelium"],
            "coexpression_fraction": [0.0, 0.5],
            "global_coexpression_fraction": [0.1, 0.1],
            "enrichment_vs_global": [0.0, 5.0],
        }
    )
    config = {
        "integration_qc_review": {
            "small_cluster_max_cells": 1,
            "library_specific_fraction": 0.9,
            "doublet_review": {
                "metric_quantile": 0.5,
                "predicted_enrichment": 1.0,
                "predicted_min_fraction": 0.1,
                "incompatible_min_fraction": 0.1,
            },
            "low_quality_review": {
                "low_rna_quantile": 0.25,
                "mt25_enrichment": 2.0,
                "mt25_min_fraction": 0.1,
                "low_genes_5mad_min_fraction": 0.1,
            },
        }
    }

    review, _, _, _ = build_cluster_review(
        obs,
        cluster_qc,
        program_scores,
        markers,
        coexpression,
        cluster_key="leiden_0.5",
        config=config,
    )

    assert review.set_index("cluster").loc["1", "qc_comment"] == "doublet_suspicious"
    assert review.set_index("cluster").loc["0", "qc_comment"] == "low_quality_suspicious"
