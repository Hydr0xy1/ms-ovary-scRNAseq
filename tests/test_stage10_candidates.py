from __future__ import annotations

import pandas as pd

from ms_ovary_scrna.stage10_candidates import assign_evidence_tier


def test_predefined_evidence_tiers() -> None:
    table = pd.DataFrame(
        {
            "stage1_6_level1_directional": [True, True, True],
            "stage1_6_level2_de_supported": [True, False, False],
            "stage1_6_population_signature_supported": [True, True, False],
            "stage2_leading_edge": [True, True, False],
            "stage3_subtype_localized": [True, True, False],
            "external_aging_supported": [False, False, False],
            "stage7_supported_tf_target": [False, False, False],
            "stage9_communication_member": [False, False, False],
        }
    )
    tiers = assign_evidence_tier(table).tolist()
    assert tiers == ["Tier A multi_layer", "Tier B DE_pathway_subtype", "Tier C exploratory"]
