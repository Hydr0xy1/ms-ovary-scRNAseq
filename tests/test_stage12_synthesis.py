from __future__ import annotations

import pandas as pd

from ms_ovary_scrna.stage12_synthesis import _geometry_summary


def test_geometry_summary_is_library_group_mean() -> None:
    projection = pd.DataFrame(
        {
            "population": ["Granulosa"] * 3,
            "group": ["Y", "OC", "OT"],
            "aging_axis_projection": [0.0, 1.0, 0.3],
        }
    )
    result = _geometry_summary(projection, pd.DataFrame()).iloc[0]
    assert result["Y"] == 0.0
    assert result["OC"] == 1.0
    assert result["OT"] == 0.3
