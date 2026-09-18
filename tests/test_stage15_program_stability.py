from __future__ import annotations

import numpy as np
import pandas as pd

from ms_ovary_scrna.stage15_program_stability import (
    _balanced_positions,
    _log_cpm_dense,
    _match_components,
)


def test_balanced_positions_equalizes_each_library() -> None:
    rows = []
    for library, n in [("Y_1", 4), ("Y_2", 3), ("Y_3", 5)]:
        rows.extend(
            {"library_id": library, "cell_type_broad_v2": "Granulosa", "analysis_tier_v2": "Tier1_primary"}
            for _ in range(n)
        )
    # The production function expects the complete nine-library design; fill
    # the remaining libraries with the same small eligible population.
    for library in ["OC_1", "OC_2", "OC_3", "OT_1", "OT_2", "OT_3"]:
        rows.extend(
            {"library_id": library, "cell_type_broad_v2": "Granulosa", "analysis_tier_v2": "Tier1_primary"}
            for _ in range(3)
        )
    obs = pd.DataFrame(rows)
    positions, meta = _balanced_positions(
        obs,
        "Granulosa",
        cells_per_library=10,
        seed=1,
        broad_key="cell_type_broad_v2",
        tier_key="analysis_tier_v2",
    )
    assert len(positions) == 9 * 3
    assert set(meta["n_sampled_balanced"]) == {3}


def test_log_cpm_is_nonnegative_and_row_scaled() -> None:
    from scipy.sparse import csr_matrix

    result = _log_cpm_dense(csr_matrix([[10, 0], [0, 5]]))
    assert result.shape == (2, 2)
    assert np.isfinite(result).all()
    assert (result >= 0).all()
    assert result[0, 0] > result[0, 1]


def test_component_matching_recovers_permutation() -> None:
    reference = np.array([[1.0, 0.0], [0.0, 1.0]])
    other = reference[[1, 0]]
    mean, minimum, assignment = _match_components(reference, other)
    assert mean > 0.99
    assert minimum > 0.99
    assert assignment.shape == (2, 2)

