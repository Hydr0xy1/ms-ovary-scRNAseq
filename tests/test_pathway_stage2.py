from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ms_ovary_scrna.de_stage1_6 import enumerate_permutation_assignments
from ms_ovary_scrna.pathway_stage2 import (
    classify_pathway_evidence,
    deduplicate_rank_table,
    exact_permutation_calibration,
    load_stage1_6_assignments,
    pathway_effect_geometry,
    pathway_member_overlap,
)


def _mapping() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_id": ["f1", "f2", "f3", "f4"],
            "canonical_mouse_symbol": ["A", "A", "B", "C"],
            "ensembl_gene_id": ["e1", "e2", "e3", "e4"],
            "valid_symbol": [True, True, True, True],
        }
    )


def test_symbol_dedup_is_deterministic_and_uses_base_mean_not_abs_stat() -> None:
    table = pd.DataFrame(
        {
            "gene": ["f2", "f1", "f4", "f3"],
            "baseMean": [50.0, 100.0, 20.0, 20.0],
            "stat": [99.0, 1.0, -2.0, 3.0],
        }
    )
    first, resolution = deduplicate_rank_table(table, _mapping())
    second, _ = deduplicate_rank_table(table.sample(frac=1, random_state=4), _mapping())
    pd.testing.assert_frame_equal(
        first[["feature_id", "canonical_mouse_symbol", "stat"]],
        second[["feature_id", "canonical_mouse_symbol", "stat"]],
    )
    assert first.set_index("canonical_mouse_symbol").loc["A", "feature_id"] == "f1"
    assert resolution.loc[resolution["selected"], "feature_id"].tolist() == ["f1"]


def test_ranking_direction_and_symbol_secondary_key() -> None:
    table = pd.DataFrame(
        {
            "gene": ["f4", "f3", "f1"],
            "baseMean": [1, 1, 1],
            "stat": [2.0, 2.0, -1.0],
        }
    )
    selected, _ = deduplicate_rank_table(table, _mapping())
    assert selected["canonical_mouse_symbol"].tolist() == ["B", "C", "A"]
    assert selected["stat"].tolist() == [2.0, 2.0, -1.0]


def test_pathway_member_overlap() -> None:
    overlap = pathway_member_overlap({"H1": ["A", "B", "X"], "H2": ["Z"]}, ["A", "B"])
    assert overlap.set_index("pathway").loc["H1", "n_matched_genes"] == 2
    assert overlap.set_index("pathway").loc["H2", "n_matched_genes"] == 0


def test_spearman_sign_cosine_and_residual_norm() -> None:
    genes = [f"G{i}" for i in range(20)]
    aging = np.arange(1, 21, dtype=float)
    treatment = -aging
    effects = pd.DataFrame(
        {
            "canonical_mouse_symbol": genes,
            "aging_effect": aging,
            "treatment_effect": treatment,
            "residual_effect": np.zeros(20),
        }
    )
    geometry = pathway_effect_geometry(effects, genes)
    assert np.isclose(geometry["rho_aging_treatment"], -1.0)
    assert np.isclose(geometry["cosine"], -1.0)
    assert np.isclose(geometry["residual_norm_ratio"], 0.0)
    assert np.isclose(geometry["vector_distance_reduction"], 1.0)
    assert np.isclose(geometry["directional_fraction"], 1.0)


def test_exact_permutation_p_and_observed_rank() -> None:
    frame = pd.DataFrame(
        {
            "permutation_id": [f"P{i:03d}" for i in range(1, 21)],
            "is_observed": [True] + [False] * 19,
            "rho": [-1.0] + list(np.linspace(-0.9, 0.9, 19)),
        }
    )
    result = exact_permutation_calibration(frame, "rho", higher_is_more_extreme=False)
    assert result["observed_rank"] == 1
    assert result["empirical_p"] == 0.05


def test_nes_is_rejected_by_lfc_additive_geometry() -> None:
    effects = pd.DataFrame(
        {
            "canonical_mouse_symbol": [f"G{i}" for i in range(15)],
            "aging_NES": np.ones(15),
            "treatment_effect": -np.ones(15),
            "residual_effect": np.zeros(15),
        }
    )
    with pytest.raises(ValueError, match="NES cannot"):
        pathway_effect_geometry(effects, effects["canonical_mouse_symbol"], aging_col="aging_NES")


def test_stage1_6_assignments_are_reused_exactly(tmp_path: Path) -> None:
    expected = enumerate_permutation_assignments()
    path = tmp_path / "permutation_assignments.tsv"
    expected.to_csv(path, sep="\t", index=False)
    loaded = load_stage1_6_assignments(path)
    pd.testing.assert_frame_equal(loaded, expected, check_dtype=False)
    altered = expected.copy()
    altered.loc[(altered["permutation_id"] == "P002") & (altered["library_id"] == "OC_1"), "model_group"] = "OT"
    altered.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError):
        load_stage1_6_assignments(path)


def test_evidence_levels_are_nested() -> None:
    base = {
        "aging_GSEA_supported": True,
        "direction_opposite": True,
        "treatment_GSEA_supported": True,
        "rho_perm_rank": 1,
        "cosine_perm_rank": 1,
        "residual_norm_ratio": 0.6,
    }
    assert classify_pathway_evidence(base) == "Strong_multi_metric_reversal"
    assert (
        classify_pathway_evidence({**base, "cosine_perm_rank": 2})
        == "Permutation_supported_reversal"
    )
    assert (
        classify_pathway_evidence({**base, "rho_perm_rank": 2, "cosine_perm_rank": 2})
        == "GSEA_supported_reversal_candidate"
    )
    assert (
        classify_pathway_evidence(
            {**base, "rho_perm_rank": 2, "cosine_perm_rank": 2, "treatment_GSEA_supported": False}
        )
        == "Directional_reversal_candidate"
    )
    assert (
        classify_pathway_evidence({**base, "direction_opposite": False}) == "Aging_only"
    )
