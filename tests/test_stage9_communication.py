from __future__ import annotations

import pandas as pd

from ms_ovary_scrna.stage9_communication import (
    communication_gate,
    complex_expression_support,
)


def test_gate_requires_both_sender_and_receiver_evidence() -> None:
    evidence = pd.DataFrame(
        {
            "population": ["Stromal_fibroblast", "Granulosa"],
            "pathway_evidence_level": ["Strong_multi_metric_reversal", "Supported"],
            "direction_opposite": [True, True],
            "aging_GSEA_supported": [True, True],
        }
    )
    assert communication_gate(evidence)[0]


def test_complex_requires_all_subunits() -> None:
    cpm = pd.DataFrame({"A": [2.0, 3.0], "B": [1.5, 2.0]}, index=["OC_1", "OT_1"])
    supported, minimum, missing = complex_expression_support(cpm, "A_B")
    assert supported and minimum >= 1.0 and not missing
    supported, _, missing = complex_expression_support(cpm, "A_C")
    assert not supported and missing == "C"
