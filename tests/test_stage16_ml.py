from __future__ import annotations

import pandas as pd

from ms_ovary_scrna.stage16_ml import _cosine, _safe_float


def test_stage16_safe_float_and_cosine() -> None:
    assert _safe_float("1.5") == 1.5
    assert _safe_float("not-a-number") != _safe_float("not-a-number")
    assert _cosine(pd.Series([1.0, 0.0]).to_numpy(), pd.Series([1.0, 0.0]).to_numpy()) == 1.0
