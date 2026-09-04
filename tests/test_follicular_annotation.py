import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from ms_ovary_scrna.follicular_annotation import (
    CLUSTER_ANNOTATIONS,
    classify_cluster8,
    cluster6_comparison,
    score_marker_programs,
)


def test_score_marker_programs_keeps_score_and_detection_columns() -> None:
    adata = ad.AnnData(
        X=sparse.csr_matrix([[1, 0, 2], [0, 3, 0]]),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(index=["Foxl2", "Amh", "Dcn"]),
    )
    adata.layers["counts"] = adata.X.copy()
    scores, available = score_marker_programs(
        adata,
        {"granulosa": ["Foxl2", "Amh"], "stromal": ["Dcn", "Missing"]},
        counts_layer="counts",
    )
    assert available == {"granulosa": ["Foxl2", "Amh"], "stromal": ["Dcn"]}
    assert "granulosa_program_score" in scores
    assert "granulosa_n_detected" in scores
    assert scores.loc["a", "granulosa_n_detected"] == 1


def test_classify_cluster8_exposes_four_interpretable_classes() -> None:
    index = pd.Index([f"c{i}" for i in range(8)])
    scores = pd.DataFrame(
        {
            "stromal_program_score": [10, 10, 10, 0, 1, 2, 3, 4],
            "granulosa_program_score": [0, 9, 0, 0, 1, 2, 3, 4],
            "theca_program_score": [0, 0, 12, 0, 1, 2, 3, 4],
            "stromal_n_detected": [3, 3, 3, 0, 0, 0, 0, 0],
            "granulosa_n_detected": [0, 3, 0, 0, 0, 0, 0, 0],
            "theca_n_detected": [0, 0, 3, 0, 0, 0, 0, 0],
        },
        index=index,
    )
    obs = pd.DataFrame(
        {
            "library_id": ["Y_1"] * 4 + ["OC_1"] * 4,
            "doublet_score": np.arange(8, dtype=float),
            "total_counts": np.arange(100, 900, 100),
            "n_genes_by_counts": np.arange(10, 90, 10),
            "predicted_doublet": [False] * 7 + [True],
        },
        index=index,
    )
    result = classify_cluster8(
        scores,
        obs,
        lineage_percentile=75,
        doublet_quantile=0.75,
        high_qc_quantile=0.50,
    )
    assert set(result["cluster8_single_cell_class"]) >= {
        "stromal_only_like",
        "stromal_plus_granulosa",
        "stromal_plus_theca",
        "uncertain",
    }
    assert result.index.equals(index)
    assert result["heterotypic_doublet_supported"].dtype == bool


def test_cluster6_comparison_has_expected_reference_groups() -> None:
    clusters = ["0", "1", "4", "13", "7", "5", "6"]
    obs = pd.DataFrame(
        {
            "follicular_leiden": clusters,
            "library_id": ["Y_1"] * len(clusters),
            "total_counts": [100] * len(clusters),
            "n_genes_by_counts": [50] * len(clusters),
            "pct_counts_mt": [2.0] * len(clusters),
        },
        index=[f"c{i}" for i in range(len(clusters))],
    )
    scores = pd.DataFrame(
        {
            "granulosa_program_score": [1.0] * len(clusters),
            "preantral_program_score": [0.5] * len(clusters),
            "antral_program_score": [0.4] * len(clusters),
            "atretic_program_score": [0.3] * len(clusters),
        },
        index=obs.index,
    )
    result = cluster6_comparison(ad.AnnData(X=sparse.csr_matrix((len(obs), 1)), obs=obs), scores)
    assert set(result["comparison_group"]) == {
        "local_cluster6",
        "preantral_reference",
        "antral_reference",
        "atretic_reference",
    }
    assert result.loc[result["comparison_group"].eq("local_cluster6"), "n_cells"].iat[0] == 1
    assert set(CLUSTER_ANNOTATIONS) >= set(clusters)
