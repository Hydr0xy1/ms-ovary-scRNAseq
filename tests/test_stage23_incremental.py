from __future__ import annotations

import numpy as np
import pandas as pd

from ms_ovary_scrna.stage23_incremental import (
    _axis_decomposition,
    _distribution_direction,
    _exact_permutation,
    _frozen_program_score,
)


def test_exact_permutation_3v3_uses_plus_one_convention() -> None:
    effect, pvalue, allocations = _exact_permutation([10, 11, 12], [0, 1, 2])
    assert effect == 10.0
    assert allocations == 20
    assert pvalue == 3 / 21


def test_distribution_direction_detects_center_tail_opposition() -> None:
    assert _distribution_direction(0.2, -0.1) == "center_tail_opposed"
    assert _distribution_direction(-0.2, -0.1) == "center_tail_concordant"


def test_axis_decomposition_has_orthogonal_residual() -> None:
    aging = pd.Series([1.0, 0.0], index=["a", "b"])
    treatment = pd.Series([-2.0, 3.0], index=["a", "b"])
    result = _axis_decomposition(aging, treatment).set_index("gene")
    assert result.loc["a", "parallel_treatment_component"] == -2.0
    assert result.loc["b", "orthogonal_treatment_residual"] == 3.0
    dot = np.dot(result["aging_effect_OC_minus_Y"], result["orthogonal_treatment_residual"])
    assert abs(dot) < 1e-12


def test_frozen_program_score_preserves_frozen_direction() -> None:
    counts = pd.DataFrame(
        {
            **{f"up{i}": [1, 2, 4, 8] for i in range(10)},
            **{f"down{i}": [8, 4, 2, 1] for i in range(10)},
            "background": [100, 100, 100, 100],
        },
        index=["a", "b", "c", "d"],
    )
    program = pd.DataFrame(
        {
            "gene": [f"up{i}" for i in range(10)] + [f"down{i}" for i in range(10)],
            "external_age_log2fc": [1.0] * 10 + [-1.0] * 10,
        }
    )
    score, genes = _frozen_program_score(counts, program, n_genes=20)
    assert len(genes) == 20
    assert score.loc["d"] > score.loc["a"]
