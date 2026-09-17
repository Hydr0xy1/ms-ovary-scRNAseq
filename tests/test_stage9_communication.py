from __future__ import annotations

import pandas as pd

from ms_ovary_scrna.stage9_communication import (
    _abundance_inventory,
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


def test_abundance_inventory_maps_existing_qc_schema(tmp_path) -> None:
    qc_root = tmp_path / "pseudobulk_ready"
    qc_root.mkdir()
    pd.DataFrame(
        {
            "population": ["Stromal_fibroblast", "Granulosa"],
            "library": ["Y_1", "Y_1"],
            "n_cells": [60, 40],
            "total_umi": [6000, 4000],
            "n_expressed_genes": [1000, 900],
        }
    ).to_csv(qc_root / "pseudobulk_qc.tsv", sep="\t", index=False)
    inventory = _abundance_inventory({"results": tmp_path})
    assert inventory["library_id"].tolist() == ["Y_1", "Y_1"]
    assert inventory["group"].tolist() == ["Y", "Y"]
    assert inventory["detected_genes"].tolist() == [1000, 900]
    assert inventory["fraction_within_sender_receiver"].tolist() == [0.6, 0.4]
