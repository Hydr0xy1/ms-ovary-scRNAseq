#!/usr/bin/env python
"""Create a single Nature-style, library-resolved cell-cycle figure.

This script is deliberately descriptive: the biological unit is the library/pool
(n=3 per condition), so it does not attach per-cell significance tests to the plot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Mandatory editable-text settings for publication SVG output.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

GROUP_ORDER = ["Y", "OC", "OT"]
GROUP_LABELS = {
    "Y": "Young vehicle (4 months)",
    "OC": "Old vehicle (10 months)",
    "OT": "Old + MRJP1 (10 months)",
}
GROUP_COLORS = {"Y": "#3E6FAE", "OC": "#D28B3C", "OT": "#2A9188"}
GROUP_OFFSETS = {"Y": -0.22, "OC": 0.0, "OT": 0.22}
REPLICATE_JITTER = {"1": -0.045, "2": 0.0, "3": 0.045}

INCLUDED_CELL_TYPES = [
    "Granulosa",
    "Immune",
    "Ciliated_epithelial",
    "Vascular_endothelial",
    "Stromal_fibroblast",
    "Ovarian_epithelial",
    "Lymphatic_endothelial",
    "Smooth_muscle_pericyte",
    "Theca_steroidogenic",
    "Luteal",
]

DISPLAY_LABELS = {
    "Granulosa": "Granulosa",
    "Immune": "Immune",
    "Ciliated_epithelial": "Ciliated epithelial",
    "Vascular_endothelial": "Vascular endothelial",
    "Stromal_fibroblast": "Stromal fibroblast",
    "Ovarian_epithelial": "Ovarian epithelial",
    "Lymphatic_endothelial": "Lymphatic endothelial",
    "Smooth_muscle_pericyte": "Smooth muscle / pericyte",
    "Theca_steroidogenic": "Theca / steroidogenic",
    "Luteal": "Luteal",
}


def prepare_source(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    required = {
        "cell_type_broad_v2",
        "library_id",
        "group",
        "phase",
        "n_cells",
        "total_cells",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    observed = table.loc[table["cell_type_broad_v2"].isin(INCLUDED_CELL_TYPES)].copy()
    wide = (
        observed.pivot_table(
            index=["cell_type_broad_v2", "library_id", "group"],
            columns="phase",
            values="n_cells",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    for phase in ["G1", "S", "G2M"]:
        if phase not in wide:
            wide[phase] = 0
    wide["total_cells"] = wide[["G1", "S", "G2M"]].sum(axis=1)
    wide["cycling_cells"] = wide["S"] + wide["G2M"]
    wide["cycling_fraction"] = wide["cycling_cells"] / wide["total_cells"]
    wide["cycling_percent"] = 100 * wide["cycling_fraction"]
    wide["replicate"] = wide["library_id"].str.rsplit("_", n=1).str[-1]

    order = (
        wide.groupby("cell_type_broad_v2", observed=True)["cycling_fraction"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    wide["cell_type_order"] = pd.Categorical(
        wide["cell_type_broad_v2"], categories=order, ordered=True
    )
    wide = wide.sort_values(
        ["cell_type_order", "group", "library_id"], kind="stable"
    ).reset_index(drop=True)

    summary = (
        wide.groupby(["cell_type_broad_v2", "group"], observed=True)
        .agg(
            n_libraries=("library_id", "nunique"),
            mean_cycling_fraction=("cycling_fraction", "mean"),
            min_cycling_fraction=("cycling_fraction", "min"),
            max_cycling_fraction=("cycling_fraction", "max"),
            pooled_cells=("total_cells", "sum"),
        )
        .reset_index()
    )
    summary["cell_type_order"] = pd.Categorical(
        summary["cell_type_broad_v2"], categories=order, ordered=True
    )
    summary["group"] = pd.Categorical(
        summary["group"], categories=GROUP_ORDER, ordered=True
    )
    summary = summary.sort_values(
        ["cell_type_order", "group"], kind="stable"
    ).reset_index(drop=True)
    return wide, summary, order


def make_figure(
    points: pd.DataFrame, summary: pd.DataFrame, order: list[str]
) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.65,
            "legend.frameon": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )

    fig, ax = plt.subplots(figsize=(183 / 25.4, 120 / 25.4))
    y_positions = {cell_type: i for i, cell_type in enumerate(order)}

    # The dominant biological signal is localized to granulosa cells.
    if "Granulosa" in y_positions:
        y = y_positions["Granulosa"]
        ax.axhspan(y - 0.43, y + 0.43, color="#EEF3F8", zorder=0)

    for y in range(len(order)):
        if y % 2 == 1:
            ax.axhspan(y - 0.5, y + 0.5, color="#FAFAFA", zorder=-1)

    for cell_type in order:
        base_y = y_positions[cell_type]
        cell_points = points.loc[points["cell_type_broad_v2"].eq(cell_type)]
        for group in GROUP_ORDER:
            block = cell_points.loc[cell_points["group"].eq(group)].copy()
            if block.empty:
                continue
            y_center = base_y + GROUP_OFFSETS[group]
            color = GROUP_COLORS[group]
            group_summary = summary.loc[
                summary["cell_type_broad_v2"].eq(cell_type)
                & summary["group"].astype(str).eq(group)
            ].iloc[0]
            low = 100 * group_summary["min_cycling_fraction"]
            high = 100 * group_summary["max_cycling_fraction"]
            mean = 100 * group_summary["mean_cycling_fraction"]
            ax.plot(
                [low, high],
                [y_center, y_center],
                color=color,
                lw=1.0,
                alpha=0.72,
                solid_capstyle="round",
                zorder=2,
            )
            jitter = block["replicate"].map(REPLICATE_JITTER).fillna(0).to_numpy()
            ax.scatter(
                block["cycling_percent"],
                y_center + jitter,
                s=15,
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.78,
                zorder=3,
            )
            ax.scatter(
                [mean],
                [y_center],
                s=28,
                marker="D",
                facecolor=color,
                edgecolor="#303030",
                linewidth=0.55,
                zorder=4,
            )

    totals = (
        points.groupby("cell_type_broad_v2", observed=True)["total_cells"]
        .sum()
        .to_dict()
    )
    for cell_type, y in y_positions.items():
        n_libraries = points.loc[
            points["cell_type_broad_v2"].eq(cell_type), "library_id"
        ].nunique()
        ax.text(
            61.5,
            y,
            f"{int(totals[cell_type]):,} cells; {n_libraries}/9 libs",
            ha="right",
            va="center",
            fontsize=5.8,
            color="#6B6B6B",
        )

    ax.set_yticks(
        range(len(order)), [DISPLAY_LABELS.get(cell_type, cell_type) for cell_type in order]
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 62)
    ax.set_xticks(np.arange(0, 61, 10))
    ax.set_xlabel("Transcriptomic cycling fraction (S + G2M), % of captured cells")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", width=0.6, length=3)

    if "Granulosa" in order:
        ax.get_yticklabels()[order.index("Granulosa")].set_fontweight("bold")

    ax.set_title(
        "Cycling transcriptional states concentrate in granulosa cells\n"
        "Library-resolved S/G2M scoring across the mouse ovary",
        loc="left",
        fontweight="bold",
        pad=10,
    )

    handles = [
        plt.Line2D(
            [],
            [],
            marker="D",
            linestyle="-",
            markersize=4.5,
            linewidth=1.2,
            markeredgecolor="#303030",
            markeredgewidth=0.45,
            color=GROUP_COLORS[group],
            label=GROUP_LABELS[group],
        )
        for group in GROUP_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.62, 1.005),
        ncol=3,
        columnspacing=1.25,
        handletextpad=0.45,
        borderaxespad=0,
    )
    fig.text(
        0.30,
        0.012,
        "Circles: independent library/pool; diamonds: unweighted group mean; lines: observed library range. "
        "No per-cell significance tests.",
        ha="left",
        va="bottom",
        fontsize=6.1,
        color="#4F4F4F",
    )
    fig.subplots_adjust(left=0.30, right=0.985, top=0.80, bottom=0.14)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/deep_dive_stage26_all_cell_cycle/"
            "PHASE_BY_BROAD_CELL_TYPE_AND_LIBRARY.tsv"
        ),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures/deep_dive_stage26_all_cell_cycle"),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(
            "results/deep_dive_stage26_all_cell_cycle/figure_source_data"
        ),
    )
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.source_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.input, sep="\t")
    points, summary, order = prepare_source(table)
    points.drop(columns="cell_type_order").to_csv(
        args.source_dir / "cell_cycle_activity_by_cell_type_library_points.tsv",
        sep="\t",
        index=False,
    )
    summary.drop(columns="cell_type_order").to_csv(
        args.source_dir / "cell_cycle_activity_by_cell_type_group_summary.tsv",
        sep="\t",
        index=False,
    )

    figure = make_figure(points, summary, order)
    base = args.figure_dir / "cell_cycle_activity_by_cell_type_and_library"
    # Preserve the declared 183 x 120 mm canvas in every export.
    figure.savefig(base.with_suffix(".svg"))
    figure.savefig(base.with_suffix(".pdf"))
    figure.savefig(base.with_suffix(".png"), dpi=600)
    plt.close(figure)
    print(f"FIGURE_COMPLETE={base}")


if __name__ == "__main__":
    main()
