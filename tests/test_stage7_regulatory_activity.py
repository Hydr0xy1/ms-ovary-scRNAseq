from __future__ import annotations

import pandas as pd

from ms_ovary_scrna.de_stage1_5 import ALL_LIBRARIES
from ms_ovary_scrna.stage7_regulatory_activity import (
    activity_contrasts,
    exact_activity_permutation,
)


def test_activity_contrasts_reversal() -> None:
    values = []
    for library in ALL_LIBRARIES:
        group = library.split("_", 1)[0]
        values.append({"library_id": library, "Tfa": {"Y": 0.0, "OC": 2.0, "OT": 0.5}[group]})
    activity = pd.DataFrame(values).set_index("library_id")
    result = activity_contrasts(activity).set_index("program")
    assert result.loc["Tfa", "directionally_reversed"]
    assert result.loc["Tfa", "closer_to_young_after_treatment"]
    assert result.loc["Tfa", "aging_effect_OC_minus_Y"] == 2.0
    assert result.loc["Tfa", "treatment_effect_OT_minus_OC"] == -1.5


def test_exact_permutation_has_twenty_assignments() -> None:
    activity = pd.DataFrame(
        {"Tfa": [0.0, 0.1, -0.1, 2.0, 2.1, 1.9, 0.2, 0.3, 0.1]},
        index=ALL_LIBRARIES,
    )
    result = exact_activity_permutation(activity, ["Tfa"])
    assert len(result) == 20
    assert result["is_observed"].sum() == 1
    assert result["empirical_p"].between(0.05, 1.0).all()
