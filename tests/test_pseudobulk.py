import pandas as pd

from ms_ovary_scrna.pseudobulk import rescue_table


def _contrast(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene": ["rescued", "same_direction", "small_aging"],
            "log2FoldChange": values,
            "padj": [0.01, 0.02, 0.5],
        }
    )


def test_rescue_table_requires_opposite_direction_and_closer_to_y() -> None:
    result = rescue_table(
        aging=_contrast([1.0, 1.0, 0.1]),
        treatment=_contrast([-0.6, 0.5, -0.1]),
        residual=_contrast([0.4, 1.5, 0.0]),
        min_abs_aging_lfc=0.25,
    ).set_index("gene")
    assert result.loc["rescued", "rescue_class"] == "partially_or_fully_rescued"
    assert result.loc["same_direction", "rescue_class"] == "not_rescued"
    assert result.loc["small_aging", "rescue_class"] == "not_rescued"
