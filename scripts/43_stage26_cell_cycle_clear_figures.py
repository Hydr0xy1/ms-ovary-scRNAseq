#!/usr/bin/env python
"""Create two clear, standalone cell-cycle figures using Python only."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

# Nature-figure export requirements: editable SVG text and embedded TrueType PDF text.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

LIBRARIES = ["Y_1", "Y_2", "Y_3", "OC_1", "OC_2", "OC_3", "OT_1", "OT_2", "OT_3"]
GROUPS = ["Y", "OC", "OT"]
GROUP_LABELS = {
    "Y": "Young vehicle\n4 months",
    "OC": "Old vehicle\n10 months",
    "OT": "Old + MRJP1\n10 months",
}
PHASES = ["G1", "S", "G2M"]
PHASE_COLORS = {"G1": "#D7D9DC", "S": "#4C78A8", "G2M": "#D77A61"}
GROUP_BANDS = {"Y": "#EFF4FA", "OC": "#FBF3E8", "OT": "#EAF6F3"}

CURATED_CELL_TYPES = [
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


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.labelsize": 7.8,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.65,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )


def prepare_phase_table(table: pd.DataFrame) -> pd.DataFrame:
    required = {
        "cell_type_broad_v2",
        "library_id",
        "group",
        "phase",
        "n_cells",
        "total_cells",
        "phase_fraction",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = table.copy()
    result["library_id"] = pd.Categorical(result["library_id"], LIBRARIES, ordered=True)
    result["phase"] = pd.Categorical(result["phase"], PHASES, ordered=True)
    return result.sort_values(
        ["cell_type_broad_v2", "library_id", "phase"], kind="stable"
    ).reset_index(drop=True)


def granulosa_source(table: pd.DataFrame) -> pd.DataFrame:
    source = table.loc[table["cell_type_broad_v2"].eq("Granulosa")].copy()
    grid = pd.MultiIndex.from_product(
        [LIBRARIES, PHASES], names=["library_id", "phase"]
    ).to_frame(index=False)
    source["library_id"] = source["library_id"].astype(str)
    source["phase"] = source["phase"].astype(str)
    source = grid.merge(source, on=["library_id", "phase"], how="left")
    if source["n_cells"].isna().any():
        raise ValueError("Granulosa phase grid is incomplete")
    source["group"] = source["library_id"].str.split("_", n=1).str[0]
    source["phase_percent"] = 100 * source["phase_fraction"]
    cycling = (
        source.loc[source["phase"].isin(["S", "G2M"])]
        .groupby("library_id", observed=True)["phase_fraction"]
        .sum()
    )
    source["cycling_percent"] = 100 * source["library_id"].map(cycling)
    return source


def cycling_heatmap_source(table: pd.DataFrame) -> pd.DataFrame:
    source = table.loc[
        table["cell_type_broad_v2"].isin(CURATED_CELL_TYPES)
        & table["phase"].astype(str).isin(["S", "G2M"])
    ].copy()
    source["library_id"] = source["library_id"].astype(str)
    source = (
        source.groupby(["cell_type_broad_v2", "library_id", "group"], observed=True)
        .agg(
            cycling_cells=("n_cells", "sum"),
            total_cells=("total_cells", "first"),
        )
        .reset_index()
    )
    source["cycling_percent"] = 100 * source["cycling_cells"] / source["total_cells"]
    order = (
        source.groupby("cell_type_broad_v2", observed=True)["cycling_percent"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    source["cell_type_order"] = pd.Categorical(
        source["cell_type_broad_v2"], order, ordered=True
    )
    source["library_id"] = pd.Categorical(source["library_id"], LIBRARIES, ordered=True)
    return source.sort_values(["cell_type_order", "library_id"], kind="stable").reset_index(
        drop=True
    )


def plot_granulosa_composition(source: pd.DataFrame) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(183 / 25.4, 105 / 25.4))
    x = np.arange(len(LIBRARIES))

    for start, stop, group in [(0, 2, "Y"), (3, 5, "OC"), (6, 8, "OT")]:
        ax.axvspan(start - 0.55, stop + 0.55, color=GROUP_BANDS[group], zorder=-2)
        ax.text(
            (start + stop) / 2,
            -15.0,
            GROUP_LABELS[group],
            ha="center",
            va="top",
            fontsize=7.2,
            fontweight="bold",
            linespacing=1.2,
            clip_on=False,
        )
    ax.axvline(2.5, color="white", lw=5.5, zorder=-1)
    ax.axvline(5.5, color="white", lw=5.5, zorder=-1)

    bottom = np.zeros(len(LIBRARIES), dtype=float)
    wide = source.pivot(index="library_id", columns="phase", values="phase_percent").reindex(
        LIBRARIES
    )
    totals = source.groupby("library_id", observed=True)["total_cells"].first().reindex(LIBRARIES)
    cycling = source.groupby("library_id", observed=True)["cycling_percent"].first().reindex(
        LIBRARIES
    )
    for phase in PHASES:
        values = wide[phase].to_numpy(float)
        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.68,
            color=PHASE_COLORS[phase],
            edgecolor="white",
            linewidth=0.8,
            label=phase,
            zorder=2,
        )
        for bar, value, base in zip(bars, values, bottom):
            if value >= 5.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    fontweight="bold" if phase != "G1" else "normal",
                    color="white" if phase != "G1" else "#3F3F3F",
                )
        bottom += values

    for i, (cycle_value, total) in enumerate(zip(cycling, totals)):
        ax.text(
            i,
            103.3,
            f"S+G2M {cycle_value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=6.2,
            fontweight="bold",
            color="#444444",
        )
        ax.text(
            i,
            100.7,
            f"n={int(total):,}",
            ha="center",
            va="bottom",
            fontsize=5.6,
            color="#6D6D6D",
        )

    ax.set_xticks(x, LIBRARIES)
    ax.set_xlim(-0.65, len(LIBRARIES) - 0.35)
    ax.set_ylim(0, 111)
    ax.set_yticks([0, 20, 40, 60, 80, 100], ["0", "20", "40", "60", "80", "100"])
    ax.set_ylabel("Cells assigned to phase (%)")
    ax.set_xlabel("")
    ax.grid(axis="y", color="#FFFFFF", lw=0.9)
    ax.set_axisbelow(False)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", width=0.6, length=3)
    ax.set_title(
        "Granulosa cell-cycle composition across biological libraries\n"
        "Transcriptomic phase assignment reveals substantial within-group variability",
        loc="left",
        fontweight="bold",
        pad=10,
    )
    ax.legend(
        title="Assigned phase",
        loc="upper center",
        bbox_to_anchor=(0.50, -0.27),
        ncol=3,
        handlelength=1.4,
        columnspacing=1.2,
        title_fontsize=6.7,
    )
    fig.text(
        0.12,
        0.012,
        "Each bar is one independent library/pool. G1 indicates no dominant S/G2M transcriptional signal; "
        "phase assignment is not a direct proliferation assay.",
        ha="left",
        va="bottom",
        fontsize=6.0,
        color="#555555",
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.78, bottom=0.25)
    return fig


def plot_cell_type_heatmap(source: pd.DataFrame) -> plt.Figure:
    apply_style()
    matrix = source.pivot(
        index="cell_type_broad_v2", columns="library_id", values="cycling_percent"
    )
    order = source["cell_type_order"].cat.categories.tolist()
    matrix = matrix.reindex(index=order, columns=LIBRARIES)
    cmap = LinearSegmentedColormap.from_list(
        "cycling",
        ["#F5F7F8", "#DCE8F2", "#8FB5D1", "#39749D", "#173F5F"],
    )

    fig, ax = plt.subplots(figsize=(183 / 25.4, 115 / 25.4))
    values = matrix.to_numpy(float)
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, cmap=cmap, vmin=0, vmax=55, aspect="auto")
    image.cmap.set_bad("#E3E3E3")

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            label = "NA" if not np.isfinite(value) else f"{value:.0f}"
            color = "#6A6A6A" if not np.isfinite(value) else ("white" if value >= 30 else "#333333")
            ax.text(col, row, label, ha="center", va="center", fontsize=6.4, color=color)

    ax.set_xticks(np.arange(len(LIBRARIES)), LIBRARIES)
    ax.set_yticks(
        np.arange(len(order)), [DISPLAY_LABELS.get(cell_type, cell_type) for cell_type in order]
    )
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-0.5, len(LIBRARIES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axvline(2.5, color="white", lw=4)
    ax.axvline(5.5, color="white", lw=4)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for start, stop, group in [(0, 2, "Y"), (3, 5, "OC"), (6, 8, "OT")]:
        ax.text(
            (start + stop) / 2,
            -1.28,
            GROUP_LABELS[group].replace("\n", " "),
            ha="center",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
            clip_on=False,
        )

    cbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.018)
    cbar.set_label("S + G2M cells (%)", fontsize=7.2)
    cbar.ax.tick_params(labelsize=6.5, width=0.5, length=2.5)
    cbar.outline.set_linewidth(0.5)
    ax.set_title(
        "Cycling transcriptional states are concentrated in granulosa cells\n"
        "S + G2M fraction for each curated cell type and biological library",
        loc="left",
        fontweight="bold",
        pad=26,
    )
    fig.text(
        0.28,
        0.025,
        "Numbers are percentages of captured cells. Grey indicates that the cell type was not detected in that library.",
        ha="left",
        va="bottom",
        fontsize=6.0,
        color="#555555",
    )
    fig.subplots_adjust(left=0.28, right=0.93, top=0.76, bottom=0.13)
    return fig


def save_figure(figure: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".svg"))
    figure.savefig(base.with_suffix(".pdf"))
    figure.savefig(base.with_suffix(".png"), dpi=600)
    plt.close(figure)


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

    table = prepare_phase_table(pd.read_csv(args.input, sep="\t"))
    granulosa = granulosa_source(table)
    heatmap = cycling_heatmap_source(table)
    granulosa.to_csv(
        args.source_dir / "granulosa_phase_composition_by_library.tsv", sep="\t", index=False
    )
    heatmap.drop(columns="cell_type_order").to_csv(
        args.source_dir / "cycling_fraction_heatmap_source.tsv", sep="\t", index=False
    )

    save_figure(
        plot_granulosa_composition(granulosa),
        args.figure_dir / "granulosa_cell_cycle_composition_by_library",
    )
    save_figure(
        plot_cell_type_heatmap(heatmap),
        args.figure_dir / "cell_cycle_activity_heatmap_by_library",
    )
    print("CLEAR_CELL_CYCLE_FIGURES_COMPLETE")


if __name__ == "__main__":
    main()
