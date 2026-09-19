from __future__ import annotations

import pandas as pd

from ms_ovary_scrna.stage16_ml import (
    _cosine,
    _extract_one_to_one_orthologs,
    _safe_float,
)


def test_stage16_safe_float_and_cosine() -> None:
    assert _safe_float("1.5") == 1.5
    assert _safe_float("not-a-number") != _safe_float("not-a-number")
    assert _cosine(pd.Series([1.0, 0.0]).to_numpy(), pd.Series([1.0, 0.0]).to_numpy()) == 1.0


def test_extract_one_to_one_orthologs_filters_non_one_to_one() -> None:
    payload = {
        "data": [
            {
                "homologies": [
                    {
                        "type": "ortholog_one2one",
                        "confidence": 1,
                        "source": {"id": "ENSMUSG1", "protein_id": "ENSMUSP1", "perc_id": 90.0},
                        "target": {
                            "species": "homo_sapiens",
                            "display_id": "FOXL2",
                            "id": "ENSG1",
                            "protein_id": "ENSP1",
                            "perc_id": 88.0,
                        },
                    },
                    {
                        "type": "ortholog_one2many",
                        "source": {"id": "ENSMUSG1"},
                        "target": {"species": "homo_sapiens", "display_id": "DROP"},
                    },
                ]
            }
        ]
    }
    rows = _extract_one_to_one_orthologs(payload, "Foxl2", "115")
    assert len(rows) == 1
    assert rows[0]["human_gene"] == "FOXL2"
    assert rows[0]["mapping_status"] == "ensembl_one_to_one"
    assert rows[0]["ensembl_release"] == "115"
