from __future__ import annotations

import numpy as np
import pandas as pd

from ms_ovary_scrna.de_stage1_6 import (
    _add_observed_gene_levels,
    build_population_evidence,
    cosine_similarity,
    effect_magnitude_summary,
    empirical_null_comparison,
    enumerate_permutation_assignments,
    whole_signature_metrics,
)
from ms_ovary_scrna.de_stage1_5 import build_rescue_ready_effects


def _rescue() -> pd.DataFrame:
    wide = pd.DataFrame(
        {
            "gene": ["a", "b", "c", "d"],
            "OC_vs_Y_log2FoldChange": [1.0, -1.0, 0.6, 0.1],
            "OT_vs_OC_log2FoldChange": [-0.8, 0.4, -0.1, -0.1],
            "OT_vs_Y_log2FoldChange": [0.2, -0.6, 0.5, 0.0],
            "OC_vs_Y_padj": [0.01, 0.02, 0.03, 0.9],
            "OT_vs_OC_padj": [0.01, 0.01, 0.8, 0.9],
            "OT_vs_Y_padj": [0.5, 0.03, 0.2, 0.9],
        }
    )
    return build_rescue_ready_effects(wide)


def test_exact_assignments_include_one_observed_and_keep_libraries_whole() -> None:
    assignments = enumerate_permutation_assignments()
    assert assignments["permutation_id"].nunique() == 20
    assert len(assignments) == 20 * 9
    observed = assignments[assignments["is_observed"]]
    assert observed["permutation_id"].nunique() == 1
    for _, sub in assignments.groupby("permutation_id"):
        assert sub["model_group"].value_counts().to_dict() == {"Y": 3, "OC": 3, "OT": 3}
        assert (sub.loc[sub["library_id"].str.startswith("Y_"), "model_group"] == "Y").all()


def test_whole_signature_and_effect_magnitude_are_scope_explicit() -> None:
    rescue = _rescue()
    whole = whole_signature_metrics(
        rescue, population="test", permutation_id="P001", is_observed=True
    )
    assert set(whole["scope"]) == {"all_tested_genes", "aging_primary_only"}
    assert int(whole.loc[whole["scope"] == "all_tested_genes", "n_genes"].iloc[0]) == 4
    assert int(whole.loc[whole["scope"] == "aging_primary_only", "n_genes"].iloc[0]) == 3
    summary, detail = effect_magnitude_summary(rescue, population="test")
    assert summary.loc[0, "n_aging_primary_with_finite_ratio"] == 3
    assert np.isclose(summary.loc[0, "ratio_median"], 0.4)
    assert set(detail["gene"]) == {"a", "b", "c"}


def test_empirical_p_uses_19_nulls_and_has_minimum_point_zero_five() -> None:
    frame = pd.DataFrame(
        {
            "is_observed": [True] + [False] * 19,
            "score": [1.0] + list(np.linspace(0.0, 0.9, 19)),
        }
    )
    high = empirical_null_comparison(frame, "score", higher_is_more_extreme=True)
    assert high["observed_rank"] == 1
    assert np.isclose(high["p_empirical"], 0.05)
    frame["score"] = [-1.0] + list(np.linspace(-0.9, 0.0, 19))
    low = empirical_null_comparison(frame, "score", higher_is_more_extreme=False)
    assert low["observed_rank"] == 1
    assert np.isclose(low["p_empirical"], 0.05)


def test_population_level_three_is_not_a_gene_level_claim() -> None:
    rows = []
    for population in ["A"]:
        for index in range(20):
            observed = index == 0
            rows.append(
                {
                    "population": population,
                    "is_observed": observed,
                    "n_aging_primary": 10,
                    "FDR_supported_rescue_fraction": 1.0 if observed else 0.2,
                    "directional_rescue_fraction": 0.9 if observed else 0.3,
                    "median_recovery_fraction": 0.8 if observed else 0.1,
                    "whole_all_spearman": -0.9 if observed else -0.2,
                    "whole_all_cosine_similarity": -0.8 if observed else -0.1,
                }
            )
    evidence = build_population_evidence(pd.DataFrame(rows))
    assert evidence.loc[0, "FDR_supported_rescue_fraction_p_empirical"] == 0.05
    assert bool(evidence.loc[0, "permutation_supported_signature"])
    assert evidence.loc[0, "population_evidence_level"].startswith("Level_3")


def test_gene_evidence_levels_are_strictly_nested() -> None:
    levels = _add_observed_gene_levels(
        _rescue(), population="test", population_level3=True
    )
    assert int(levels["Level_1_directional_candidate"].sum()) == 3
    assert int(levels["Level_2_DE_supported_candidate"].sum()) == 2
    assert (
        ~levels["Level_2_DE_supported_candidate"]
        | levels["Level_1_directional_candidate"]
    ).all()


def test_cosine_similarity_handles_zero_vectors() -> None:
    assert np.isclose(cosine_similarity([1, 0], [-1, 0]), -1.0)
    assert np.isnan(cosine_similarity([0, 0], [1, 2]))
