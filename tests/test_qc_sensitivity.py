import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from ms_ovary_scrna.qc_sensitivity import (
    build_markdown_report,
    count_gene_mad_sensitivity,
    ensure_top20_metric,
    flag_overlap_summary,
    mad_masks_and_thresholds,
    metric_distribution,
    mt_threshold_sensitivity,
    retention_sensitivity,
)


def test_metric_distribution_uses_requested_quantile_names() -> None:
    result = metric_distribution(pd.Series(np.arange(1, 101)), "metric")

    assert {
        "metric_p90",
        "metric_p95",
        "metric_p97_5",
        "metric_p99",
        "metric_p99_5",
    }.issubset(result)


def test_top20_metric_is_calculated_from_counts_layer() -> None:
    counts = sparse.csr_matrix(
        np.vstack(
            [
                np.ones(25, dtype=np.float32),
                np.concatenate(([10.0], np.zeros(24, dtype=np.float32))),
            ]
        )
    )
    adata = ad.AnnData(X=sparse.csr_matrix(counts.shape, dtype=np.float32))
    adata.layers["counts"] = counts

    ensure_top20_metric(adata, "counts")

    np.testing.assert_allclose(
        adata.obs["pct_counts_in_top_20_genes"].to_numpy(),
        [80.0, 100.0],
    )


def test_mad_masks_keep_low_and_high_sides_separate() -> None:
    values = pd.Series([1.0, 10.0, 11.0, 12.0, 100.0])
    result = mad_masks_and_thresholds(values, 2, log1p=False)

    assert result["low_mask"].tolist() == [True, False, False, False, False]
    assert result["high_mask"].tolist() == [False, False, False, False, True]
    assert result["low_threshold"] == 9.0
    assert result["high_threshold"] == 13.0


def test_mt_sensitivity_reports_absolute_and_existing_mad_flags() -> None:
    obs = pd.DataFrame(
        {
            "library_id": ["Y_1"] * 5,
            "group": ["Y"] * 5,
            "pct_counts_mt": [1.0, 2.0, 3.0, 6.0, 12.0],
            "total_counts": [100, 200, 300, 400, 500],
            "n_genes_by_counts": [50, 100, 150, 200, 250],
            "qc_mt_outlier": [False, False, False, True, True],
        }
    )
    table = mt_threshold_sensitivity(obs, mt_n_mads=3)
    library = table[table["entity_type"] == "library"]
    five = library[library["threshold_label"] == "absolute_5_pct"].iloc[0]
    ten = library[library["threshold_label"] == "absolute_10_pct"].iloc[0]

    assert five["cells_above_threshold"] == 2
    assert ten["cells_above_threshold"] == 1
    assert five["median_total_counts_above"] == 450

    retention = retention_sensitivity(table)
    five_retention = retention[
        (retention["entity_type"] == "library") & (retention["threshold_label"] == "absolute_5_pct")
    ].iloc[0]
    assert five_retention["would_retain"] == 3
    assert five_retention["retained_pct"] == 60


def test_count_gene_table_contains_separate_candidates_and_top20() -> None:
    obs = pd.DataFrame(
        {
            "library_id": ["Y_1"] * 7,
            "group": ["Y"] * 7,
            "total_counts": [10, 100, 110, 120, 130, 140, 5000],
            "n_genes_by_counts": [10, 200, 210, 220, 230, 240, 1000],
            "pct_counts_in_top_20_genes": np.linspace(10, 70, 7),
        }
    )
    table = count_gene_mad_sensitivity(obs)

    assert table["n_mads"].tolist() == [3.0, 4.0, 5.0]
    assert {
        "low_count_count",
        "high_count_count",
        "low_gene_count",
        "high_gene_count",
        "top20_pct_median",
    }.issubset(table.columns)


def test_markdown_report_builds_from_sensitivity_tables() -> None:
    obs = pd.DataFrame(
        {
            "library_id": ["Y_1"] * 5,
            "group": ["Y"] * 5,
            "pct_counts_mt": [1.0, 2.0, 3.0, 6.0, 12.0],
            "total_counts": [100, 200, 300, 400, 500],
            "n_genes_by_counts": [50, 100, 150, 200, 250],
            "pct_counts_in_top_20_genes": [80, 70, 60, 50, 40],
            "qc_mt_outlier": [False, False, False, True, True],
            "predicted_doublet": [False, False, False, False, True],
        },
        index=[f"cell_{index}" for index in range(5)],
    )
    adata = ad.AnnData(X=sparse.csr_matrix((5, 25)), obs=obs)
    mt_table = mt_threshold_sensitivity(obs, mt_n_mads=3)
    count_gene_table = count_gene_mad_sensitivity(obs)
    overlap_table = flag_overlap_summary(obs)
    retention_table = retention_sensitivity(mt_table)
    scrublet_table = pd.DataFrame(
        {
            "library_id": ["Y_1"],
            "automatic_threshold": [0.2],
            "predicted_doublet_pct": [20.0],
            "observed_simulated_histogram_overlap": [0.5],
            "score_spearman_total_counts": [0.8],
            "score_spearman_n_genes": [0.7],
            "top100_total_counts_ratio_to_all": [1.5],
            "top100_n_genes_ratio_to_all": [1.4],
        }
    )

    report = build_markdown_report(
        adata,
        mt_table,
        count_gene_table,
        scrublet_table,
        overlap_table,
        retention_table,
    )

    assert "# QC sensitivity report" in report
    assert "当前 3-MAD mt 标记总数：2" in report
    assert "没有写出 filtered H5AD" in report
