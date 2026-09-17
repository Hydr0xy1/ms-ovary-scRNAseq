from __future__ import annotations

import numpy as np
import pandas as pd

from ms_ovary_scrna.de_stage1_5 import ALL_LIBRARIES
from ms_ovary_scrna.stage5_rejuvenation_geometry import (
    cross_validated_aging_projection,
    log_cpm,
    projection_onto_axis,
)


def test_projection_scale() -> None:
    young = np.array([0.0, 0.0])
    aged = np.array([2.0, 0.0])
    assert projection_onto_axis(young, young, aged) == 0.0
    assert projection_onto_axis(aged, young, aged) == 1.0
    assert projection_onto_axis(np.array([1.0, 1.0]), young, aged) == 0.5


def test_cross_validation_does_not_use_held_y_or_oc() -> None:
    rows = []
    for library in ALL_LIBRARIES:
        group = library.split("_", 1)[0]
        value = {"Y": 0.0, "OC": 2.0, "OT": 0.5}[group]
        rows.append([value, value + 0.1, value + 0.2])
    expression = pd.DataFrame(rows, index=ALL_LIBRARIES, columns=["a", "b", "c"])
    result = cross_validated_aging_projection(expression, n_variable_genes=3).set_index(
        "library_id"
    )
    assert result.loc["Y_1", "n_training_Y"] == 2
    assert result.loc["OC_1", "n_training_OC"] == 2
    assert result.loc["OT_1", "n_training_Y"] == 3
    assert result.loc["OT_1", "n_training_OC"] == 3


def test_log_cpm_finite() -> None:
    counts = pd.DataFrame([[0, 10], [5, 5]], index=["a", "b"])
    result = log_cpm(counts)
    assert np.isfinite(result.to_numpy()).all()
