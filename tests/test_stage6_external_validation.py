from __future__ import annotations

import numpy as np
import pandas as pd

from ms_ovary_scrna.stage6_external_validation import signature_concordance


def test_signature_concordance_identical_vectors() -> None:
    a = pd.Series([1.0, -2.0, 3.0], index=["a", "b", "c"])
    metrics = signature_concordance(a, a)
    assert metrics["n_genes"] == 3
    assert np.isclose(metrics["spearman"], 1.0)
    assert np.isclose(metrics["cosine"], 1.0)
    assert metrics["same_direction_fraction"] == 1.0


def test_signature_concordance_aligns_by_gene() -> None:
    internal = pd.Series([1.0, 2.0], index=["a", "b"])
    external = pd.Series([2.0, 1.0], index=["b", "a"])
    metrics = signature_concordance(internal, external)
    assert metrics["n_genes"] == 2
