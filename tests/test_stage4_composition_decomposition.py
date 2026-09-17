from __future__ import annotations

import numpy as np
import pandas as pd

from ms_ovary_scrna.stage4_composition_decomposition import (
    clr_transform_counts,
    shapley_two_factor_decomposition,
)


def test_clr_rows_are_closed_and_centered() -> None:
    counts = pd.DataFrame([[10, 20, 0], [4, 4, 4]], columns=["a", "b", "c"])
    proportions, clr = clr_transform_counts(counts, pseudocount=0.5)
    np.testing.assert_allclose(proportions.sum(axis=1), 1.0)
    np.testing.assert_allclose(clr.mean(axis=1), 0.0, atol=1e-12)
    assert np.isfinite(clr.to_numpy()).all()


def test_shapley_decomposition_exactly_reconstructs_total() -> None:
    p_y = np.array([0.6, 0.4])
    p_oc = np.array([0.3, 0.7])
    mu_y = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    mu_oc = np.array([[2.0, 2.5, 3.5], [4.0, 2.5, 0.5]])
    composition, intrinsic, total = shapley_two_factor_decomposition(
        p_y, p_oc, mu_y, mu_oc
    )
    np.testing.assert_allclose(composition + intrinsic, total)
    np.testing.assert_allclose(total, p_oc @ mu_oc - p_y @ mu_y)
