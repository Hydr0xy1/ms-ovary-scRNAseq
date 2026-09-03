from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from ms_ovary_scrna.broad_annotation import (
    cluster24_lineage_comparison,
    cluster_marker_evidence,
    incompatible_codetection,
    noncycling_marker_table,
    read_cluster_mapping,
)


def _adata() -> ad.AnnData:
    genes = ["Foxl2", "Amh", "Cyp11a1", "Fdx1", "Ptprc", "Tyrobp"]
    matrix = sparse.csr_matrix(
        np.array(
            [
                [2, 1, 0, 0, 0, 0],
                [3, 1, 0, 0, 0, 0],
                [0, 0, 2, 1, 0, 0],
                [1, 0, 2, 1, 1, 1],
            ],
            dtype=np.float32,
        )
    )
    obs = pd.DataFrame(
        {
            "leiden_0.5": pd.Categorical(["0", "0", "1", "1"]),
            "doublet_score": [0.01, 0.02, 0.03, 0.9],
        },
        index=[f"cell_{index}" for index in range(4)],
    )
    adata = ad.AnnData(X=matrix.copy(), obs=obs, var=pd.DataFrame(index=genes))
    adata.layers["counts"] = matrix.copy()
    return adata


def _panels() -> dict:
    return {
        "Granulosa": {"positive": ["Foxl2", "Amh"], "exclude": ["Ptprc"]},
        "Theca_steroidogenic": {
            "positive": ["Cyp11a1", "Fdx1"],
            "exclude": ["Ptprc"],
        },
        "Immune": {"positive": ["Ptprc", "Tyrobp"], "exclude": ["Foxl2"]},
    }


def _markers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster": ["0", "0", "1", "1"],
            "names": ["Foxl2", "Amh", "Cyp11a1", "Fdx1"],
            "logfoldchanges": [3.0, 2.0, 4.0, 3.0],
        }
    )


def test_marker_evidence_uses_log_expression_and_existing_marker_statistics() -> None:
    evidence, panel_summary = cluster_marker_evidence(
        _adata(), _panels(), _markers(), cluster_key="leiden_0.5"
    )
    row = evidence[
        (evidence["cluster"] == "0")
        & (evidence["panel"] == "Granulosa")
        & (evidence["marker"] == "Foxl2")
    ].iloc[0]
    assert row["mean_log_normalized_expression"] == pytest.approx(2.5)
    assert row["fraction_expressing"] == 1.0
    assert row["marker_rank"] == 1
    assert row["cluster_vs_rest_logFC"] == 3.0
    winner = panel_summary[
        (panel_summary["cluster"] == "0") & (panel_summary["panel_rank"] == 1)
    ].iloc[0]
    assert winner["panel"] == "Granulosa"


def test_incompatible_codetection_is_single_cell_and_count_based() -> None:
    table = incompatible_codetection(
        _adata(),
        _panels(),
        cluster_key="leiden_0.5",
        cluster="1",
        focal_panel="Theca_steroidogenic",
        comparison_panels=["Immune"],
    )
    assert table.iloc[0]["focal_two_marker_fraction"] == 1.0
    assert table.iloc[0]["co_detection_fraction"] == 0.5
    assert table.iloc[0]["median_doublet_score_co_detected"] == 0.9


def test_noncycling_markers_do_not_modify_original_table() -> None:
    markers = _markers()
    result = noncycling_marker_table(markers, cluster="1", cell_cycle_genes=["Cyp11a1"], top_n=50)
    assert result["names"].tolist() == ["Fdx1"]
    assert "excluded_as_cell_cycle" not in markers.columns


def test_cluster24_comparison_reports_nearest_cluster_per_lineage() -> None:
    summary = pd.DataFrame(
        {
            "cluster": ["24", "24", "4", "4", "3", "3"],
            "panel": ["Granulosa", "Theca_steroidogenic"] * 3,
            "relative_panel_score": [0.0, 2.0, 0.1, 1.9, 2.0, 0.0],
        }
    )
    mapping = pd.DataFrame(
        {
            "cluster": ["24", "4", "3"],
            "cell_type_broad": [
                "Theca_steroidogenic",
                "Theca_steroidogenic",
                "Granulosa",
            ],
        }
    )
    result = cluster24_lineage_comparison(summary, mapping)
    theca = result[result["candidate_lineage"] == "Theca_steroidogenic"].iloc[0]
    assert theca["nearest_cluster"] == "4"
    assert theca["euclidean_distance_all_major_panels"] < 0.2


def test_cluster_mapping_requires_complete_cluster_coverage(tmp_path: Path) -> None:
    path = tmp_path / "mapping.tsv"
    pd.DataFrame(
        {
            "cluster": ["0"],
            "candidate_1": ["Granulosa"],
            "candidate_2": ["Theca_steroidogenic"],
            "cell_type_broad": ["Granulosa"],
            "cell_state_provisional": ["None"],
            "annotation_confidence": ["High"],
            "doublet_concern": ["False"],
            "qc_concern": ["False"],
            "annotation_rationale": ["Two positive markers and no exclusion conflict."],
        }
    ).to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="cluster mismatch"):
        read_cluster_mapping(path, ["0", "1"])
