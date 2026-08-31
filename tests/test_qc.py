import pandas as pd

from ms_ovary_scrna.qc import mad_outlier


def test_mad_outlier_flags_extreme_high_value() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    result = mad_outlier(values, n_mads=3.0, side="high")
    assert result.tolist() == [False, False, False, False, True]
