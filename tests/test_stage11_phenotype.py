from __future__ import annotations

from ms_ovary_scrna.stage11_phenotype import map_pathway_to_assays


def test_oxidative_program_maps_to_functional_assays() -> None:
    rows = map_pathway_to_assays("HALLMARK_OXIDATIVE_PHOSPHORYLATION")
    assert rows
    assert "ATP" in rows[0]["recommended_assays"]


def test_unmapped_program_does_not_invent_assay() -> None:
    assert map_pathway_to_assays("UNKNOWN_PROGRAM") == []
