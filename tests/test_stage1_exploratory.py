import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from ms_ovary_scrna.preprocessing import normalize_log1p_preserving_counts
from ms_ovary_scrna.stage1_exploratory import (
    annotate_stage1_qc_flags,
    mark_high_doublet_score_top_fraction,
    qc_candidate_cluster_distribution,
    sparse_matrix_audit,
    stage1_filter_reason_overlap,
    stage1_filter_summary,
)


def _config() -> dict:
    return {
        "stage1_exploratory": {
            "min_genes_absolute": 200,
            "low_genes_n_mads": 5.0,
            "mt_moderate_lower_pct": 5.0,
            "mt_extreme_pct": 25.0,
        }
    }


def _adata() -> ad.AnnData:
    counts = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 0],
                [1, 1, 0],
                [2, 1, 1],
                [4, 2, 1],
                [8, 4, 2],
                [1, 0, 0],
                [1, 1, 0],
                [2, 1, 1],
                [4, 2, 1],
                [8, 4, 2],
            ],
            dtype=np.float32,
        )
    )
    obs = pd.DataFrame(
        {
            "library_id": ["Y_1"] * 5 + ["OC_1"] * 5,
            "group": ["Y"] * 5 + ["OC"] * 5,
            "total_counts": [100, 200, 300, 400, 500] * 2,
            "n_genes_by_counts": [10, 200, 210, 220, 230] * 2,
            "pct_counts_mt": [30, 25, 10, 5, 1, 30, 25, 10, 5, 1],
            "predicted_doublet": [False, True, False, False, False] * 2,
            "doublet_score": np.linspace(0, 0.9, 10),
            "qc_genes_outlier": [True, False, False, False, False] * 2,
        },
        index=[f"cell_{index}" for index in range(10)],
    )
    adata = ad.AnnData(X=counts.copy(), obs=obs, var=pd.DataFrame(index=["a", "b", "c"]))
    adata.layers["counts"] = counts.copy()
    return adata


def test_stage1_flags_apply_only_the_agreed_strong_rule() -> None:
    adata = _adata()
    thresholds = annotate_stage1_qc_flags(adata, _config())

    assert (
        adata.obs["qc_low_genes_absolute"].tolist()
        == [
            True,
            False,
            False,
            False,
            False,
        ]
        * 2
    )
    assert adata.obs["qc_mt_extreme"].tolist() == [True, False, False, False, False] * 2
    assert adata.obs["qc_mt_moderate"].tolist() == [False, True, True, False, False] * 2
    assert adata.obs["qc_doublet_auto"].tolist() == [False, True, False, False, False] * 2
    assert adata.obs["qc_low_quality_strong"].tolist() == [True, False, False, False, False] * 2
    assert len(thresholds) == 2


def test_filter_tables_report_retention_and_reason_overlap() -> None:
    adata = _adata()
    annotate_stage1_qc_flags(adata, _config())
    summary = stage1_filter_summary(adata.obs)
    overlap = stage1_filter_reason_overlap(adata.obs)

    overall = summary[summary["entity_type"] == "overall"].iloc[0]
    assert overall["cells_before"] == 10
    assert overall["would_remove"] == 2
    assert overall["cells_after"] == 8
    strong = overlap[
        (overlap["entity_type"] == "overall") & (overlap["reason"] == "qc_low_quality_strong")
    ].iloc[0]
    assert strong["flagged_cells"] == 2


def test_extreme_mt_and_low_5mad_filters_without_absolute_low_gene_flag() -> None:
    adata = _adata()
    adata.obs.loc[adata.obs["library_id"] == "Y_1", "n_genes_by_counts"] = [
        200,
        1000,
        1100,
        1200,
        1300,
    ]

    annotate_stage1_qc_flags(adata, _config())

    first = adata.obs.iloc[0]
    assert not first["qc_low_genes_absolute"]
    assert first["qc_low_genes_5mad"]
    assert first["qc_mt_extreme"]
    assert first["qc_low_quality_strong"]


def test_normalization_keeps_counts_sparse_and_unchanged() -> None:
    adata = _adata()
    before = sparse_matrix_audit(adata.layers["counts"])

    normalize_log1p_preserving_counts(adata, counts_layer="counts", target_sum=1e4)

    after = sparse_matrix_audit(adata.layers["counts"])
    assert before == after
    assert sparse.isspmatrix_csr(adata.X)
    normalized_sums = np.expm1(adata.X.toarray()).sum(axis=1)
    np.testing.assert_allclose(normalized_sums, 1e4, rtol=1e-5)


def test_top_fraction_is_exact_and_candidate_table_uses_primary_cluster() -> None:
    obs = pd.DataFrame(
        {
            "library_id": ["Y_1"] * 100 + ["OC_1"] * 100,
            "group": ["Y"] * 100 + ["OC"] * 100,
            "doublet_score": np.arange(200, dtype=float),
            "qc_doublet_auto": [False] * 198 + [True, True],
            "retained_extreme_mt": [False] * 199 + [True],
            "total_counts": np.arange(1, 201),
            "n_genes_by_counts": np.arange(201, 401),
            "pct_counts_mt": np.linspace(0, 30, 200),
            "leiden_0.5": ["0"] * 100 + ["1"] * 100,
        },
        index=[f"cell_{index}" for index in range(200)],
    )
    thresholds = mark_high_doublet_score_top_fraction(obs, 0.01)
    table = qc_candidate_cluster_distribution(obs, cluster_key="leiden_0.5")

    assert obs["qc_high_doublet_score_top1pct"].sum() == 2
    assert thresholds["selected_cells"].tolist() == [1, 1]
    top_all = table[
        (table["candidate_set"] == "high_doublet_score") & (table["cluster_id"] == "ALL")
    ].iloc[0]
    assert top_all["candidate_cells_all"] == 2
