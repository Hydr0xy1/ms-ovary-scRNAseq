from __future__ import annotations

import numpy as np
import pandas as pd

from ms_ovary_scrna.stage27_granulosa_stromal import (
    LIBRARIES,
    compare_library_scores,
    library_summary,
    standardize_subtypes,
    symmetric_decomposition,
)


def test_exact_library_test_has_correct_resolution() -> None:
    table = pd.DataFrame(
        {
            "family": "test",
            "population": "Granulosa",
            "view": "all",
            "metric": "cycle",
            "library_id": LIBRARIES,
            "group": [x.split("_")[0] for x in LIBRARIES],
            "value": [1, 2, 3, 10, 11, 12, 20, 21, 22],
        }
    )
    tests, loo = compare_library_scores(table)
    assert tests.n_permutations.eq(20).all()
    assert tests.p_exact_two_sided.eq(0.1).all()
    assert len(loo) == 18
    assert tests.loo_direction_fraction.eq(1).all()


def fixture_subtypes() -> pd.DataFrame:
    rows = []
    for i, library in enumerate(LIBRARIES):
        for subtype, counts, rate in [("A", 100 + i * 10, 10 + i), ("B", 200 - i * 10, 20 - i)]:
            rows.append(
                dict(
                    population="Granulosa",
                    view=f"subtype::{subtype}",
                    library_id=library,
                    group=library.split("_")[0],
                    n_cells=counts,
                    cycling_percent=rate,
                )
            )
    return pd.DataFrame(rows)


def test_standardization_preserves_fixed_weights_and_drops_rare() -> None:
    sub = fixture_subtypes()
    extra = sub[sub.view.eq("subtype::A")].copy()
    extra["view"] = "subtype::rare"
    extra["n_cells"] = 5
    standardized, audit = standardize_subtypes(pd.concat([sub, extra]))
    assert not audit.loc[audit.subtype.eq("rare"), "common_subtype"].iloc[0]
    assert np.isclose(audit.fixed_weight.sum(), 1)
    assert standardized.common_subtype_fraction.lt(1).all()
    # A/B 平均权重不强制相等；直接按已输出权重计算。
    weights = audit.set_index("subtype").fixed_weight.dropna()
    for row in standardized.itertuples():
        source = sub[sub.library_id.eq(row.library_id)].set_index("view")
        expected = sum(
            weights[s] * source.loc[f"subtype::{s}", "cycling_percent"] for s in weights.index
        )
        assert np.isclose(row.cycling_percent, expected)


def test_symmetric_decomposition_sums_to_equal_library_group_change() -> None:
    sub = fixture_subtypes()
    result = symmetric_decomposition(sub)
    sub["weighted"] = sub.n_cells * sub.cycling_percent
    rates = sub.groupby(["group", "library_id"]).agg({"weighted": "sum", "n_cells": "sum"})
    rates["rate"] = rates.weighted / rates.n_cells
    means = rates.groupby("group").rate.mean()
    total = result.groupby("contrast").total_component_pp.sum()
    for test, reference in [("OC", "Y"), ("OT", "OC"), ("OT", "Y")]:
        assert np.isclose(total[f"{test}_vs_{reference}"], means[test] - means[reference])


def test_cycle_summary_keeps_observed_zero_and_weak_scores_distinct() -> None:
    table = pd.DataFrame(
        {
            "library_id": ["Y_1"] * 3,
            "group": "Y",
            "phase": ["G1", "S", "S"],
            "S_score": [-0.2, 0.001, 0.2],
            "G2M_score": [-0.3, -0.1, -0.1],
            "cycle_score_max": [-0.2, 0.001, 0.2],
            "phase_margin": [0.1, 0.101, 0.3],
            "total_counts": [500, 1000, 2000],
            "n_genes_by_counts": [200, 400, 800],
            "pct_counts_mt": [2, 2, 2],
            "doublet_score": [0.1, 0.1, 0.1],
        }
    )
    summary = library_summary(table, "Granulosa", "test").iloc[0]
    assert summary.G2M_cells == 0
    assert np.isclose(summary.cycling_percent, 200 / 3)
    assert np.isclose(summary["cycling_score_ge_0.10_percent"], 100 / 3)
