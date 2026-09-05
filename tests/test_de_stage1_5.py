import numpy as np
import pandas as pd

from ms_ovary_scrna.de_stage1_5 import build_rescue_ready_effects


def _wide() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene": ["rescue", "same", "persistent", "near"],
            "OC_vs_Y_log2FoldChange": [1.0, 1.0, 1.0, -0.8],
            "OT_vs_OC_log2FoldChange": [-0.6, 0.4, 0.1, 0.3],
            "OT_vs_Y_log2FoldChange": [0.4, 1.4, 1.1, -0.5],
            "OC_vs_Y_padj": [0.01, 0.01, 0.01, 0.01],
            "OT_vs_OC_padj": [0.01, 0.01, 0.50, 0.50],
            "OT_vs_Y_padj": [0.50, 0.50, 0.50, 0.50],
        }
    )


def test_effect_geometry_is_algebraic_and_directional() -> None:
    result = build_rescue_ready_effects(_wide()).set_index("gene")
    np.testing.assert_allclose(
        result["residual_effect"], result["aging_effect"] + result["treatment_effect"]
    )
    assert result.loc["rescue", "directional_rescue_candidate"]
    assert result.loc["rescue", "FDR_supported_rescue_candidate"]
    assert result.loc["same", "same_direction_treatment"]
    assert result.loc["persistent", "persistent_aging"]
    assert result.loc["near", "opposite_direction"]


def test_primary_classification_is_single_label_while_flags_can_overlap() -> None:
    wide = _wide()
    wide.loc[wide["gene"] == "rescue", "OT_vs_Y_log2FoldChange"] = -0.2
    result = build_rescue_ready_effects(wide).set_index("gene")
    assert result.loc["rescue", "overshoot_candidate"]
    assert result.loc["rescue", "primary_classification"] == "overshoot_candidate"
    assert result.loc["rescue", "directional_rescue_candidate"]

