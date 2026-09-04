import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from ms_ovary_scrna.annotation_consolidation import (
    NEW_COLUMNS,
    _apply_local_labels,
    _census_table,
    _tier_summary,
)


def _local_inputs() -> tuple[ad.AnnData, ad.AnnData, pd.DataFrame, pd.DataFrame]:
    main_index = pd.Index(list("abcdefg"))
    main_obs = pd.DataFrame(
        {
            "library_id": ["Y_1", "Y_1", "OC_1", "OC_1", "OT_1", "OT_1", "Y_2"],
            "group": ["Y", "Y", "OC", "OC", "OT", "OT", "Y"],
            "cell_type_broad": [
                "Immune",
                "Granulosa",
                "Granulosa",
                "Granulosa",
                "Granulosa",
                "Oocyte",
                "Immune",
            ],
            "cell_state_provisional": ["None"] * 7,
            "annotation_confidence": ["High"] * 7,
            "predicted_doublet": [False] * 7,
            "qc_low_quality_strong": [False] * 7,
            "qc_mt_extreme": [False] * 7,
            "qc_mt_moderate": [False] * 7,
            "qc_low_genes_absolute": [False] * 7,
            "qc_low_genes_5mad": [False] * 7,
        },
        index=main_index,
    )
    main = ad.AnnData(
        X=sparse.csr_matrix(np.arange(14).reshape(7, 2)),
        obs=main_obs,
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    main.layers["counts"] = main.X.copy()

    local_index = pd.Index(list("bcdef"))
    local_obs = pd.DataFrame(
        {
            "follicular_leiden": ["8", "8", "10", "14", "12"],
            "original_leiden_cluster": ["3", "24", "3", "14", "12"],
        },
        index=local_index,
    )
    local = ad.AnnData(
        X=sparse.csr_matrix(np.arange(10).reshape(5, 2)),
        obs=local_obs,
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    local.layers["counts"] = local.X.copy()
    labels = pd.DataFrame(
        {
            "cell_barcode": list("bcdef"),
            "cell_type_broad_local": [
                "Uncertain",
                "Theca_steroidogenic",
                "Luteal",
                "Oocyte",
                "Immune_mixed",
            ],
            "cell_type_subtype_local": [
                "Mixed_doublet_suspicious",
                "Theca_cycling",
                "Luteal_like",
                "Oocyte",
                "Mixed_doublet_suspicious",
            ],
            "cell_state_local": [
                "Doublet_suspicious",
                "Cycling",
                "Cycling_subpopulation",
                "None",
                "Doublet_suspicious",
            ],
            "annotation_confidence_local": ["Low", "Medium", "High", "Medium", "Low"],
            "predicted_doublet": [False, True, False, False, False],
        },
    ).set_index("cell_barcode", drop=False)
    cluster8 = pd.DataFrame(
        {
            "cell_barcode": list("bcdef"),
            "cluster8_single_cell_class": [
                "stromal_only_like",
                "stromal_plus_theca",
                "not_cluster8",
                "not_cluster8",
                "not_cluster8",
            ],
            "heterotypic_doublet_supported": [False, False, False, False, False],
            "predicted_doublet": [False, True, False, False, False],
        },
    ).set_index("cell_barcode", drop=False)
    return main, local, labels, cluster8


def test_apply_local_labels_uses_barcode_alignment_and_explicit_overrides() -> None:
    main, local, labels, cluster8 = _local_inputs()
    result, audit = _apply_local_labels(main, local, labels, cluster8)
    assert audit.shape[0] == 5
    assert set(NEW_COLUMNS).issubset(result.columns)
    assert result.loc["a", "cell_type_broad_v1"] == "Immune"
    assert result.loc["a", "cell_type_subtype_v1"] == "Unresolved"
    assert result.loc["b", "cell_type_broad_v1"] == "Stromal_fibroblast"
    assert result.loc["b", "analysis_tier_v1"] == "Tier1_primary"
    # Global cluster 24 overrides local cluster-8 mixed evidence.
    assert result.loc["c", "cell_type_subtype_v1"] == "Theca_cycling"
    assert result.loc["c", "cell_state_v1"] == "Cycling"
    assert result.loc["c", "analysis_tier_v1"] == "Tier2_sensitivity"
    assert result.loc["d", "cell_state_v1"] == "None"
    assert result.loc["e", "analysis_tier_v1"] == "Tier3_descriptive_only"
    assert result.loc["f", "cell_type_broad_v1"] == "Uncertain_immune_related"


def test_census_and_tier_summary_include_all_library_and_group_scopes() -> None:
    main, local, labels, cluster8 = _local_inputs()
    result, _ = _apply_local_labels(main, local, labels, cluster8)
    obs = main.obs.copy()
    for column in NEW_COLUMNS:
        obs[column] = result[column]
    census = _census_table(obs)
    tiers = _tier_summary(obs)
    assert set(census["scope"]) == {"all", "library", "group"}
    assert set(tiers["scope"]) == {"all", "library", "group"}
    assert int(
        census.loc[
            (census.scope == "all") & (census.field == "cell_type_broad_v1"), "n_cells"
        ].sum()
    ) == len(obs)
    assert int(tiers.loc[tiers.scope.eq("all"), "n_cells"].sum()) == len(obs)
