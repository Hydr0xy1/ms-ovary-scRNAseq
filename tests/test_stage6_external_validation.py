from __future__ import annotations

import numpy as np
import pandas as pd

from ms_ovary_scrna.stage6_external_validation import (
    prepare_external_pseudobulk_counts,
    set_population_column,
    signature_concordance,
)


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


def test_prepare_external_pseudobulk_rounds_only_aggregated_values() -> None:
    counts = pd.DataFrame(
        [[0.2, 1.5, 2.8], [10.49, 10.51, 0.0]],
        index=["young_1", "aged_1"],
        columns=["a", "b", "c"],
    )
    rounded, audit = prepare_external_pseudobulk_counts(counts)
    expected = pd.DataFrame(
        [[0, 2, 3], [10, 11, 0]],
        index=counts.index,
        columns=counts.columns,
    )
    pd.testing.assert_frame_equal(rounded, expected)
    assert audit["n_fractional_pseudobulk_entries"] == 5
    assert audit["rounding_stage"] == "after biological-sample pseudobulk aggregation"


def test_set_population_column_is_idempotent() -> None:
    table = pd.DataFrame({"gene": ["A", "B"], "population": ["old", "old"]})
    updated = set_population_column(table, "Granulosa")
    assert updated.columns.tolist() == ["population", "gene"]
    assert updated["population"].tolist() == ["Granulosa", "Granulosa"]
