from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from ms_ovary_scrna.stage8_gene_programs import (
    TIER1_LABEL,
    _balanced_counts,
    choose_rank,
    component_stability,
    estimate_dense_gib,
    select_hvgs,
)


def test_memory_estimate_is_bounded() -> None:
    assert estimate_dense_gib(9000, 2000) < 1.0


def test_component_stability_identical_permuted_components() -> None:
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = a[::-1]
    assert np.isclose(component_stability([a, b]), 1.0)


def test_choose_rank_uses_stable_elbow() -> None:
    table = pd.DataFrame(
        {
            "k": [5, 8, 10],
            "mean_component_stability": [0.9, 0.85, 0.7],
            "next_k_relative_error_improvement": [0.04, 0.03, np.nan],
        }
    )
    assert choose_rank(table) == 5


def test_sparse_hvg_selection() -> None:
    matrix = sparse.csr_matrix([[10, 0, 1], [0, 10, 1], [8, 0, 1], [0, 8, 1]] * 5)
    genes = pd.Index(["A", "B", "mt-C"])
    selected, names, audit = select_hvgs(matrix, genes, 2)
    assert selected.shape == (20, 2)
    assert set(names) == {"A", "B"}
    assert not audit.loc[audit["gene"].eq("mt-C"), "selected_hvg"].iloc[0]


def test_balanced_counts_uses_annotation_v2_tier1_label(tmp_path) -> None:
    libraries = ["Y_1", "Y_2", "Y_3", "OC_1", "OC_2", "OC_3", "OT_1", "OT_2", "OT_3"]
    n_per_library = 200
    obs = pd.DataFrame(
        {
            "library_id": np.repeat(libraries, n_per_library),
            "cell_type_broad_v2": "Granulosa",
            "analysis_tier_v2": TIER1_LABEL,
        },
        index=[f"cell_{i}" for i in range(len(libraries) * n_per_library)],
    )
    adata = ad.AnnData(
        X=sparse.csr_matrix((len(obs), 2)),
        obs=obs,
        var=pd.DataFrame(index=["GeneA", "GeneB"]),
        layers={"counts": sparse.csr_matrix(np.ones((len(obs), 2), dtype=np.int32))},
    )
    path = tmp_path / "tiny.h5ad"
    adata.write_h5ad(path)
    matrix, metadata, genes, sampling = _balanced_counts(
        path,
        "Granulosa",
        max_cells_per_library=n_per_library,
        seed=7,
    )
    assert matrix.shape == (len(obs), 2)
    assert len(metadata) == len(obs)
    assert genes.tolist() == ["GeneA", "GeneB"]
    assert sampling["sampled_cells"].eq(n_per_library).all()
