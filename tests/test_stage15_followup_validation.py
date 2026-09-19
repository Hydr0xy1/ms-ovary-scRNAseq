from __future__ import annotations

import numpy as np
import pandas as pd

from ms_ovary_scrna.stage15_followup_validation import (
    _permutation_assignments,
    _project_internal,
)


def test_exact_three_vs_three_assignments_are_complete_and_unique() -> None:
    assignments = _permutation_assignments()
    assert len(assignments) == 20
    assert len({(ot, oc) for ot, oc in assignments}) == 20
    for ot, oc in assignments:
        assert len(ot) == len(oc) == 3
        assert set(ot).isdisjoint(oc)
        assert set(ot) | set(oc) == {"OC_1", "OC_2", "OC_3", "OT_1", "OT_2", "OT_3"}


def test_project_internal_returns_finite_axis_and_distance() -> None:
    genes = [f"G{i}" for i in range(30)]
    counts = pd.DataFrame(
        np.arange(9 * len(genes), dtype=float).reshape(9, len(genes)) + 1,
        index=["Y_1", "Y_2", "Y_3", "OC_1", "OC_2", "OC_3", "OT_1", "OT_2", "OT_3"],
        columns=genes,
    )
    selected = pd.DataFrame({"gene": genes, "external_age_log2fc": np.linspace(-1, 1, len(genes))})
    result = _project_internal(counts, selected)
    assert result["n_genes"] == len(genes)
    assert np.isfinite(result["delta_axis_OT_minus_OC"])
    assert np.isfinite(result["distance_change_OT_minus_OC"])
