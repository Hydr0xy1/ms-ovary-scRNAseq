from __future__ import annotations

import pandas as pd

from ms_ovary_scrna.stage3_subtype_localization import (
    ALL_LIBRARIES,
    build_localization_table,
    classify_subtype_eligibility,
)


def _coverage(subtype: str, n1: int, n12: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "broad_population": "Granulosa",
            "subtype": subtype,
            "library_id": ALL_LIBRARIES,
            "n_cells_all": n12,
            "n_cells_tier1": n1,
            "n_cells_tier1_or_tier2": n12,
        }
    )


def _inventory(subtypes: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "broad_population": "Granulosa",
            "subtype": subtypes,
            "confident_identity_fraction": 0.95,
            "doublet_concern_fraction": 0.01,
        }
    )


def test_fixed_eligibility_tiers() -> None:
    coverage = pd.concat(
        [
            _coverage("Granulosa_antral_like", 60, 60),
            _coverage("Granulosa_cycling", 45, 45),
            _coverage("Granulosa_low_complexity_candidate", 60, 60),
        ],
        ignore_index=True,
    )
    result = classify_subtype_eligibility(
        coverage,
        _inventory(
            [
                "Granulosa_antral_like",
                "Granulosa_cycling",
                "Granulosa_low_complexity_candidate",
            ]
        ),
    ).set_index("subtype")
    assert result.loc["Granulosa_antral_like", "eligibility"] == "Primary_DE_ready"
    assert result.loc["Granulosa_cycling", "eligibility"] == "Sensitivity_only"
    assert (
        result.loc["Granulosa_low_complexity_candidate", "eligibility"]
        == "Descriptive_only"
    )


def test_localization_requires_bilateral_gsea_and_opposite_direction() -> None:
    hallmark = pd.DataFrame(
        {
            "broad_population": ["Granulosa"] * 3,
            "subtype": ["Granulosa_antral_like"] * 3,
            "contrast": ["OC_vs_Y", "OT_vs_OC", "OT_vs_Y"],
            "pathway": ["HALLMARK_TEST"] * 3,
            "NES": [2.0, -1.5, 0.5],
            "FDR_q": [0.01, 0.02, 0.4],
        }
    )
    broad = pd.DataFrame(
        {
            "population": ["Granulosa"],
            "pathway": ["HALLMARK_TEST"],
            "pathway_evidence_level": ["Strong_multi_metric_reversal"],
        }
    )
    result = build_localization_table(hallmark, broad)
    assert len(result) == 1
    assert bool(result.loc[0, "localized_GSEA_support"])
