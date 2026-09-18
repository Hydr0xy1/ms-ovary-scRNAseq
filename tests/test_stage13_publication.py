from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ms_ovary_scrna.stage13_publication import (
    GROUP_COLORS,
    TIER_A_ANCHORS,
    _normalized_figure_stem,
    assign_publication_candidate_tiers,
    configure_publication_style,
)


def _candidate_table() -> pd.DataFrame:
    rows = []
    for cell_type, gene in TIER_A_ANCHORS:
        rows.append((cell_type, gene, True, True, True, True, True, False, True, 0.001))
    for index in range(21):
        rows.append(
            (
                "Stromal_fibroblast",
                f"B{index:02d}",
                True,
                True,
                True,
                True,
                index % 3 == 0,
                index % 4 == 0,
                index % 2 == 0,
                0.01 + index / 1000,
            )
        )
    for index in range(30):
        rows.append(
            (
                "Immune",
                f"C{index:02d}",
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                0.1 + index / 1000,
            )
        )
    return pd.DataFrame(
        rows,
        columns=[
            "cell_type",
            "gene",
            "stage1_6_level2_de_supported",
            "stage1_6_population_signature_supported",
            "stage2_leading_edge",
            "stage3_subtype_localized",
            "external_aging_supported",
            "stage9_communication_member",
            "stage7_supported_tf_target",
            "treatment_FDR",
        ],
    )


def test_candidate_tiers_have_requested_sizes_without_score() -> None:
    result = assign_publication_candidate_tiers(_candidate_table())
    counts = result["publication_tier"].value_counts()
    assert 5 <= counts["Tier A - immediate validation"] <= 10
    assert 10 <= counts["Tier B - second priority"] <= 20
    assert counts.sum() == len(result)
    assert not any("score" in column.lower() for column in result.columns)


def test_color_registry_is_fixed_and_colorblind_aware() -> None:
    assert GROUP_COLORS == {"Y": "#0072B2", "OC": "#D55E00", "OT": "#009E73"}


def test_publication_style_keeps_svg_text_editable() -> None:
    configure_publication_style()
    assert plt.rcParams["svg.fonttype"] == "none"
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.right"] is False


def test_similar_figure_names_share_normalized_stem() -> None:
    assert _normalized_figure_stem(Path("plot_v2.png")) == "plot"
