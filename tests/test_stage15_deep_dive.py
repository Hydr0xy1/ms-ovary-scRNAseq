from __future__ import annotations

import pandas as pd

from ms_ovary_scrna.deep_dive_audit import _missingness
from ms_ovary_scrna.stage15_projection_state import _log_cpm


def test_stage15_missingness_marks_unknown_and_todo() -> None:
    table = pd.DataFrame({"batch": ["unknown", "B1"], "estrous_stage": ["TODO", "unknown"]})
    result = _missingness(table, ["batch", "estrous_stage", "pool_mouse_ids"])
    assert result.set_index("field").loc["batch", "n_missing"] == 1
    assert result.set_index("field").loc["estrous_stage", "n_missing"] == 2
    assert result.set_index("field").loc["pool_mouse_ids", "present"] is False


def test_stage15_log_cpm_is_row_normalized_and_finite() -> None:
    counts = pd.DataFrame([[10, 0, 5], [0, 4, 6]], index=["Y_1", "OC_1"], columns=["A", "B", "C"])
    transformed = _log_cpm(counts)
    assert transformed.shape == counts.shape
    assert transformed.notna().all().all()
    assert transformed.loc["Y_1", "A"] > transformed.loc["Y_1", "B"]
