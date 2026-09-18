from __future__ import annotations

from ms_ovary_scrna.stage15_synthesis import _fmt


def test_stage15_synthesis_formats_missing_values_conservatively() -> None:
    assert _fmt(None) == "NA"
    assert _fmt(float("nan")) == "NA"
    assert _fmt(1.23456) == "1.235"

