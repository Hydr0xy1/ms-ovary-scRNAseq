import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from ms_ovary_scrna.follicular_subclustering import (
    build_follicular_subset,
    incompatible_granulosa_theca_codetection,
    marker_program_tables,
    origin_neighborhood_summary,
    resolution_key,
    resolution_tables,
    resolution_transition_table,
    subset_inventory,
)


def _adata() -> ad.AnnData:
    genes = ["Foxl2", "Inha", "Cyp11a1", "Fdx1", "Mki67"]
    counts = sparse.csr_matrix(
        np.array(
            [
                [2, 1, 0, 0, 0],
                [1, 1, 1, 1, 0],
                [0, 0, 2, 2, 1],
                [0, 0, 1, 1, 2],
                [1, 1, 1, 1, 2],
                [1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
    )
    obs = pd.DataFrame(
        {
            "library_id": ["Y_1", "Y_1", "OC_1", "OC_1", "OT_1", "OT_1"],
            "group": ["Y", "Y", "OC", "OC", "OT", "OT"],
            "age_months": [4, 4, 10, 10, 10, 10],
            "treatment": ["vehicle"] * 4 + ["MRJP1"] * 2,
            "leiden_0.5": pd.Categorical(["3", "6", "24", "24", "25", "0"]),
            "cell_type_broad": [
                "Granulosa",
                "Uncertain",
                "Theca_steroidogenic",
                "Theca_steroidogenic",
                "Luteal_candidate",
                "Stromal_fibroblast",
            ],
            "annotation_confidence": ["High", "Uncertain", "Low", "Low", "Low", "High"],
            "total_counts": [3, 4, 5, 6, 6, 1],
            "n_genes_by_counts": [2, 4, 3, 3, 5, 1],
            "pct_counts_mt": [1, 1, 1, 1, 1, 1],
            "doublet_score": [0.01, 0.1, 0.2, 0.3, 0.05, 0.01],
            "predicted_doublet": [False, False, True, False, False, False],
        },
        index=[f"cell_{index}" for index in range(6)],
    )
    var = pd.DataFrame(index=genes)
    adata = ad.AnnData(X=counts.copy(), obs=obs, var=var)
    adata.layers["counts"] = counts.copy()
    adata.obsm["X_umap"] = np.zeros((6, 2))
    adata.uns["neighbors"] = {"params": {}}
    return adata


def test_build_subset_preserves_counts_metadata_and_removes_atlas_embedding() -> None:
    source = _adata()
    original_counts = source.layers["counts"].copy()
    subset = build_follicular_subset(
        source,
        source_cluster_key="leiden_0.5",
        source_clusters=["3", "6", "24", "25"],
        counts_layer="counts",
    )

    assert subset.n_obs == 5
    assert subset.obs["original_leiden_cluster"].astype(str).tolist() == [
        "3",
        "6",
        "24",
        "24",
        "25",
    ]
    assert sparse.isspmatrix_csr(subset.layers["counts"])
    assert not subset.obsm
    assert not subset.uns
    np.testing.assert_array_equal(source.layers["counts"].toarray(), original_counts.toarray())


def test_subset_inventory_has_total_and_cluster_counts() -> None:
    subset = build_follicular_subset(
        _adata(),
        source_cluster_key="leiden_0.5",
        source_clusters=["3", "6", "24", "25"],
        counts_layer="counts",
    )
    inventory = subset_inventory(subset.obs).set_index("entity")
    assert inventory.loc["ALL", "n_cells"] == 5
    assert inventory.loc["24", "n_cells"] == 2


def test_resolution_table_reports_all_three_composition_dimensions() -> None:
    subset = build_follicular_subset(
        _adata(),
        source_cluster_key="leiden_0.5",
        source_clusters=["3", "6", "24", "25"],
        counts_layer="counts",
    )
    subset.obs[resolution_key(0.5)] = pd.Categorical(["0", "0", "1", "1", "1"])
    summary, composition = resolution_tables(subset.obs, [0.5])
    assert summary["n_cells"].sum() == 5
    assert set(composition["dimension"]) == {
        "original_leiden_cluster",
        "library_id",
        "group",
    }


def test_resolution_transition_table_quantifies_adjacent_splits() -> None:
    subset = build_follicular_subset(
        _adata(),
        source_cluster_key="leiden_0.5",
        source_clusters=["3", "6", "24", "25"],
        counts_layer="counts",
    )
    subset.obs[resolution_key(0.2)] = pd.Categorical(["0", "0", "0", "1", "1"])
    subset.obs[resolution_key(0.5)] = pd.Categorical(["0", "0", "1", "2", "2"])
    table = resolution_transition_table(subset.obs, [0.2, 0.5])

    parent_zero = table[table["from_cluster"] == "0"].set_index("to_cluster")
    assert parent_zero.loc["0", "n_cells"] == 2
    assert parent_zero.loc["1", "n_cells"] == 1
    assert parent_zero.loc["0", "pct_from_cluster"] == pytest.approx(200 / 3)
    assert parent_zero.loc["0", "n_destinations_from_cluster"] == 2
    assert bool(parent_zero.loc["0", "is_dominant_destination"])
    assert table["n_cells"].sum() == subset.n_obs


def test_marker_program_tables_keep_multi_marker_evidence() -> None:
    subset = build_follicular_subset(
        _adata(),
        source_cluster_key="leiden_0.5",
        source_clusters=["3", "6", "24", "25"],
        counts_layer="counts",
    )
    subset.obs[resolution_key(0.5)] = pd.Categorical(["0", "0", "1", "1", "1"])
    evidence, programs = marker_program_tables(
        subset,
        {
            "Granulosa": ["Foxl2", "Inha"],
            "Theca": ["Cyp11a1", "Fdx1"],
        },
        cluster_key=resolution_key(0.5),
    )
    assert len(evidence) == 8
    winner = programs[(programs["cluster"] == "0") & (programs["program_rank"] == 1)].iloc[0]
    assert winner["program"] == "Granulosa"


def test_codetection_uses_raw_counts_at_single_cell_level() -> None:
    subset = build_follicular_subset(
        _adata(),
        source_cluster_key="leiden_0.5",
        source_clusters=["3", "6", "24", "25"],
        counts_layer="counts",
    )
    subset.obs[resolution_key(0.5)] = pd.Categorical(["0", "0", "1", "1", "1"])
    table = incompatible_granulosa_theca_codetection(
        subset,
        cluster_key=resolution_key(0.5),
        granulosa_genes=["Foxl2", "Inha"],
        theca_genes=["Cyp11a1", "Fdx1"],
        counts_layer="counts",
    ).set_index("cluster")
    assert table.loc["0", "granulosa_theca_codetection_fraction"] == pytest.approx(0.5)
    assert table.loc["1", "granulosa_theca_codetection_fraction"] == pytest.approx(1 / 3)


def test_origin_neighborhood_summary_reports_local_and_source_context() -> None:
    subset = build_follicular_subset(
        _adata(),
        source_cluster_key="leiden_0.5",
        source_clusters=["3", "6", "24", "25"],
        counts_layer="counts",
    )
    subset.obs["follicular_leiden"] = pd.Categorical(["0", "0", "1", "1", "1"])
    rows = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    cols = np.array([1, 2, 0, 2, 0, 1, 2, 4, 2, 3])
    graph = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(5, 5))
    table = origin_neighborhood_summary(subset.obs, graph).set_index("original_cluster")
    assert table.loc["24", "n_cells"] == 2
    assert table.loc["25", "dominant_local_cluster"] == "1"
