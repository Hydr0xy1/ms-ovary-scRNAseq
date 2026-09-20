from __future__ import annotations

import numpy as np
import pandas as pd

from ms_ovary_scrna.stage24_virtual_knockout import (
    balanced_sample,
    choose_expression_matched_references,
    cpm_normalize,
    pairwise_jaccard,
    select_gene_universe,
)


def test_balanced_sample_uses_equal_library_counts_and_is_reproducible() -> None:
    obs = pd.DataFrame(
        {"library_id": ["OC_1"] * 5 + ["OC_2"] * 6},
        index=[f"cell_{i}" for i in range(11)],
    )
    first = balanced_sample(
        obs, library_key="library_id", cells_per_library=3, seed=7
    )
    second = balanced_sample(
        obs, library_key="library_id", cells_per_library=3, seed=7
    )
    assert first.index.tolist() == second.index.tolist()
    assert first["library_id"].value_counts().to_dict() == {"OC_1": 3, "OC_2": 3}


def test_gene_universe_forces_candidates_without_exceeding_budget() -> None:
    stats = pd.DataFrame(
        {
            "gene": [f"g{i}" for i in range(10)],
            "detection_fraction": np.linspace(0.1, 1.0, 10),
            "mean_umi": np.linspace(1, 10, 10),
            "hvg_rank": np.arange(10),
        }
    )
    selected = select_gene_universe(
        stats,
        hvg_rank_column="hvg_rank",
        forced_genes=["g9"],
        n_genes=5,
        forced_min_detection_fraction=0.01,
    )
    assert len(selected) == 5
    assert "g9" in set(selected["gene"])
    assert selected.loc[selected["gene"].eq("g9"), "forced_gene"].item()


def test_expression_matched_references_are_unique() -> None:
    stats = pd.DataFrame(
        {
            "gene": ["A", "B", "C", "D", "E", "F"],
            "mean_umi": [1.0, 5.0, 1.1, 4.9, 2.0, 3.0],
            "detection_fraction": [0.2, 0.8, 0.21, 0.79, 0.4, 0.6],
        }
    )
    matched = choose_expression_matched_references(stats, targets=["A", "B"])
    assert matched["reference_gene"].is_unique
    assert set(matched["reference_gene"]).isdisjoint({"A", "B"})


def test_pairwise_jaccard() -> None:
    value = pairwise_jaccard([{"a", "b"}, {"b", "c"}, {"a", "b"}])
    assert np.isclose(value, 1 / 3)


def test_cpm_normalize_scales_each_cell() -> None:
    counts = pd.DataFrame({"c1": [1, 3], "c2": [2, 2]}, index=["g1", "g2"])
    normalized = cpm_normalize(counts)
    assert np.allclose(normalized.sum(axis=0), 1_000_000)
