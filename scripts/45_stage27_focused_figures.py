#!/usr/bin/env python
"""Python-only standalone figures for granulosa/stromal focused analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ms_ovary_scrna.stage27_granulosa_stromal import LIBRARIES, OUT, PATHWAYS, POPS  # noqa: E402

DISPLAY = {
    "Granulosa": "Granulosa",
    "Stromal_fibroblast": "Stromal fibroblast",
    "Granulosa_antral_like": "Antral-like",
    "Granulosa_preantral_like": "Preantral-like",
    "Granulosa_atretic_like": "Atretic-like",
    "Granulosa_cycling": "Cycling",
    "Granulosa_low_complexity_candidate": "Low-complexity candidate",
    "Stromal_fibroblast_candidate": "Fibroblast candidate",
    "ECM_high_candidate": "ECM-high candidate",
    "Smooth_muscle_candidate": "Smooth-muscle candidate",
    "Unresolved": "Unresolved",
    "all_annotated": "All annotated",
    "Tier1_primary": "Tier1 primary",
    "Tier1_plus_Tier2": "Tier1 + Tier2",
    "without_low_complexity": "Without low-complexity",
    "without_cycling_subtype": "Without cycling subtype",
    "without_low_complexity_and_cycling": "Without low-complexity + cycling",
    "fibroblast_and_ECM_only": "Fibroblast + ECM only",
    "fixed_subtype_weights": "Fixed subtype weights",
}
PALETTE = {
    "Granulosa_antral_like": "#83B4CC",
    "Granulosa_preantral_like": "#3C779D",
    "Granulosa_atretic_like": "#D69A80",
    "Granulosa_cycling": "#8B7AA8",
    "Granulosa_low_complexity_candidate": "#CBCDCF",
    "Stromal_fibroblast_candidate": "#6D9BB7",
    "ECM_high_candidate": "#C18E75",
    "Smooth_muscle_candidate": "#ABA0BB",
    "Unresolved": "#D6D6D6",
}
GROUP_BANDS = ["#F0F4F9", "#FBF3E9", "#EEF6F3"]
GROUP_LABELS = ["Y: young vehicle", "OC: old vehicle", "OT: old + MRJP1"]
PATHWAY_LABELS = [
    "G2M checkpoint",
    "E2F targets",
    "Mitotic spindle",
    "MYC targets V1",
    "p53 pathway",
    "Apoptosis",
    "DNA repair",
    "Reactive oxygen species pathway",
    "Oxidative phosphorylation",
    "Unfolded protein response",
    "TGFβ signaling",
    "ECM-related / EMT gene set",
    "TNFα signaling via NFκB",
    "Estrogen response late",
]
SEQ = LinearSegmentedColormap.from_list(
    "cycle", ["#F5F7F8", "#DCE8F2", "#8FB5D1", "#39749D", "#173F5F"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "direction", ["#416F9C", "#D7E3ED", "#F8F8F8", "#EDD3CA", "#AB5648"]
)


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )


def save(fig: plt.Figure, base: Path) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.findobj(match=Text):
        if not item.get_visible() or not item.get_text():
            continue
        box = item.get_window_extent(renderer)
        if (
            box.x0 < -1
            or box.y0 < -1
            or box.x1 > fig.bbox.width + 1
            or box.y1 > fig.bbox.height + 1
        ):
            raise ValueError(f"Text outside figure canvas: {base.name}: {item.get_text()}")
    for suffix in ["svg", "pdf", "png"]:
        fig.savefig(base.with_suffix("." + suffix), dpi=600)
    plt.close(fig)


def composition_figure(source: pd.DataFrame, pop: str) -> plt.Figure:
    block = source[source.population.eq(pop)]
    subtypes = (
        [
            "Granulosa_preantral_like",
            "Granulosa_antral_like",
            "Granulosa_atretic_like",
            "Granulosa_cycling",
            "Granulosa_low_complexity_candidate",
        ]
        if pop == "Granulosa"
        else [
            "Stromal_fibroblast_candidate",
            "ECM_high_candidate",
            "Smooth_muscle_candidate",
            "Unresolved",
        ]
    )
    matrix = (
        block.pivot(index="library_id", columns="subtype", values="fraction_of_parent")
        .reindex(index=LIBRARIES, columns=subtypes)
        .fillna(0)
        * 100
    )
    if not np.allclose(matrix.sum(axis=1), 100):
        raise ValueError("Subtype composition does not sum to 100%")
    fig, ax = plt.subplots(figsize=(183 / 25.4, 120 / 25.4))
    for start, band, label in zip([0, 3, 6], GROUP_BANDS, GROUP_LABELS):
        ax.axvspan(start - 0.5, start + 2.5, color=band, zorder=-2)
        ax.text(start + 1, -15, label, ha="center", va="top", fontsize=7, clip_on=False)
    bottom = np.zeros(9)
    for subtype in subtypes:
        values = matrix[subtype].to_numpy()
        ax.bar(
            np.arange(9),
            values,
            bottom=bottom,
            width=0.70,
            color=PALETTE[subtype],
            edgecolor="white",
            lw=0.7,
            label=DISPLAY[subtype],
        )
        for i, value in enumerate(values):
            if value >= 8:
                ax.text(
                    i,
                    bottom[i] + value / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white"
                    if subtype in ["Granulosa_preantral_like", "Stromal_fibroblast_candidate"]
                    else "#333333",
                )
        bottom += values
    totals = block.groupby("library_id").parent_cells.first().reindex(LIBRARIES)
    for i, value in enumerate(totals):
        ax.text(i, 102, f"n={int(value):,}", ha="center", va="bottom", fontsize=6.4)
    ax.set_xticks(np.arange(9), LIBRARIES)
    ax.set_xlim(-0.65, 8.65)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Subtype fraction within captured population (%)")
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_title(
        f"{DISPLAY[pop]} subtype composition by library", loc="left", weight="bold", pad=10
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=3,
        fontsize=7,
        columnspacing=1.2,
    )
    fig.text(
        0.12,
        0.015,
        "One bar = one independent library/pool; n=3/group. Captured fractions are not absolute tissue abundance.",
        fontsize=6.5,
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.85, bottom=0.29)
    return fig


def heatmap(
    matrix: pd.DataFrame,
    title: str,
    cbar_label: str,
    footer: str,
    *,
    diverging: bool = False,
    limit: float = 60,
    support: pd.DataFrame | None = None,
    library_columns: bool = True,
) -> plt.Figure:
    height = max(90, 43 + len(matrix) * 6.3)
    fig, ax = plt.subplots(figsize=(183 / 25.4, height / 25.4))
    cmap = (DIVERGING if diverging else SEQ).copy()
    cmap.set_bad("#D8D8D8")
    values = matrix.to_numpy(float)
    im = ax.imshow(
        np.ma.masked_invalid(values),
        cmap=cmap,
        vmin=-limit if diverging else 0,
        vmax=limit,
        aspect="auto",
    )
    for (i, j), value in np.ndenumerate(values):
        text = "NA" if not np.isfinite(value) else f"{value:.1f}"
        if support is not None and np.isfinite(value) and support.iloc[i, j]:
            text += "•"
        rgba = cmap(im.norm(value)) if np.isfinite(value) else (0.8, 0.8, 0.8, 1)
        color = "white" if np.dot(rgba[:3], [0.299, 0.587, 0.114]) < 0.55 else "#333333"
        ax.text(j, i, text, ha="center", va="center", fontsize=6.8, color=color)
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if library_columns:
        ax.axvline(2.5, color="white", lw=3)
        ax.axvline(5.5, color="white", lw=3)
        for position, label in zip([1, 4, 7], ["Y", "OC", "OT"]):
            ax.text(
                (position + 0.5) / len(matrix.columns),
                1.025,
                label,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                weight="bold",
                fontsize=7.5,
            )
    ax.set_title(title, loc="left", weight="bold", pad=28 if library_columns else 14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.020)
    cbar.set_label(cbar_label, fontsize=7)
    cbar.outline.set_linewidth(0.5)
    fig.text(0.02, 0.025, footer, va="bottom", fontsize=6.3, linespacing=1.4)
    fig.subplots_adjust(left=0.38, right=0.94, bottom=0.16, top=0.80)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    out = args.root / "results" / OUT
    figure_dir = args.root / "figures" / OUT
    sources = out / "figure_source_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    sources.mkdir(parents=True, exist_ok=True)
    style()
    summary = pd.read_csv(out / "LIBRARY_CYCLE_SUMMARY.tsv", sep="\t")
    composition = pd.read_csv(out / "SUBTYPE_COMPOSITION_BY_LIBRARY.tsv", sep="\t")
    pathways = pd.read_csv(out / "EXISTING_HALLMARK_EVIDENCE.tsv", sep="\t")
    modules = pd.read_csv(out / "EXISTING_STATE_MODULE_LIBRARY_SCORES.tsv", sep="\t")
    manifest = []
    for pop in POPS:
        slug = pop.lower()
        for name, source, fig in [
            (
                f"{slug}_subtype_composition",
                composition[composition.population.eq(pop)],
                composition_figure(composition, pop),
            ),
        ]:
            source.to_csv(sources / f"{name}.tsv", sep="\t", index=False)
            save(fig, figure_dir / name)
            manifest.append(
                dict(
                    figure=name,
                    source_data=f"figure_source_data/{name}.tsv",
                    evidence="descriptive_composition",
                    n_libraries=9,
                )
            )
        block = summary[summary.population.eq(pop)]
        view_order = ["all_annotated", "Tier1_primary", "Tier1_plus_Tier2"]
        view_order += (
            [
                "without_low_complexity",
                "without_cycling_subtype",
                "without_low_complexity_and_cycling",
            ]
            if pop == "Granulosa"
            else ["fibroblast_and_ECM_only"]
        )
        view_order += ["fixed_subtype_weights"]
        # 极少数 unresolved 不拉伸主图色阶；数值与所有诊断视图仍完整保留在源表。
        view_order += sorted(
            block[
                block.view.str.startswith("subtype::") & block.view.ne("subtype::Unresolved")
            ].view.unique()
        )
        matrix = block.pivot(index="view", columns="library_id", values="cycling_percent").reindex(
            index=view_order, columns=LIBRARIES
        )
        matrix.index = [DISPLAY.get(x.removeprefix("subtype::"), x) for x in matrix.index]
        name = f"{slug}_cycle_sensitivity"
        block.to_csv(sources / f"{name}.tsv", sep="\t", index=False)
        save(
            heatmap(
                matrix,
                f"{DISPLAY[pop]}: cell-cycle sensitivity",
                "S + G2M cells (%)",
                "Values are percentages in each library/view; n=3 independent pools/group. No new filtering or phase labels.\n"
                + (
                    "Fixed subtype weights are descriptive; cycling annotation shares markers with phase scoring."
                    if pop == "Granulosa"
                    else "Nested views are not independent evidence; rare unresolved cells remain in the source tables."
                ),
                limit=max(10, np.ceil(matrix.max().max() / 10) * 10),
            ),
            figure_dir / name,
        )
        manifest.append(
            dict(
                figure=name,
                source_data=f"figure_source_data/{name}.tsv",
                evidence="descriptive_cycle_sensitivity",
                n_libraries=9,
            )
        )
        weak = block[block.view.eq("all_annotated")][
            [
                "library_id",
                "group",
                "n_cells",
                "cycling_percent",
                "cycling_score_ge_0.05_percent",
                "cycling_score_ge_0.10_percent",
                "cycling_score_ge_0.20_percent",
            ]
        ]
        matrix = (
            weak.set_index("library_id")
            .reindex(LIBRARIES)[
                [
                    "cycling_percent",
                    "cycling_score_ge_0.05_percent",
                    "cycling_score_ge_0.10_percent",
                    "cycling_score_ge_0.20_percent",
                ]
            ]
            .T
        )
        matrix.index = [
            "Original S + G2M assignment",
            "+ max cycle score ≥0.05",
            "+ max cycle score ≥0.10",
            "+ max cycle score ≥0.20",
        ]
        name = f"{slug}_weak_score_sensitivity"
        weak.to_csv(sources / f"{name}.tsv", sep="\t", index=False)
        save(
            heatmap(
                matrix,
                f"{DISPLAY[pop]}: weak-score sensitivity",
                "Cells meeting criteria (%)",
                "Denominator: all annotated cells of this population in each library; n=3 independent pools/group.\n"
                "Additional score cutoffs are exploratory diagnostics, not biological phase boundaries or filtering rules.",
                limit=max(10, np.ceil(matrix.max().max() / 10) * 10),
            ),
            figure_dir / name,
        )
        manifest.append(
            dict(
                figure=name,
                source_data=f"figure_source_data/{name}.tsv",
                evidence="descriptive_weak_signal_sensitivity",
                n_libraries=9,
            )
        )
        p = (
            pathways[pathways.population.eq(pop)]
            .set_index("pathway")
            .reindex([f"HALLMARK_{x}" for x in PATHWAYS])
        )
        matrix = p[["aging_NES", "treatment_NES", "residual_NES"]].copy()
        support = p[["aging_FDR", "treatment_FDR", "residual_FDR"]].lt(0.05)
        matrix.columns = ["OC − Y", "OT − OC", "OT − Y"]
        matrix.index = PATHWAY_LABELS
        support.index, support.columns = matrix.index, matrix.columns
        name = f"{slug}_existing_hallmark"
        p.reset_index().to_csv(sources / f"{name}.tsv", sep="\t", index=False)
        save(
            heatmap(
                matrix,
                f"{DISPLAY[pop]}: existing functional evidence",
                "Existing GSEA NES",
                "Numbers are frozen unified-model NES; • existing GSEA FDR <0.05, not a new library-level permutation test.\n"
                "Positive NES indicates enrichment toward the first condition. Transcriptional enrichment is not pathway activity.",
                diverging=True,
                limit=3.5,
                support=support,
                library_columns=False,
            ),
            figure_dir / name,
        )
        manifest.append(
            dict(
                figure=name,
                source_data=f"figure_source_data/{name}.tsv",
                evidence="existing_unified_model_GSEA",
                n_libraries=9,
            )
        )
        m = modules[modules.population.eq(pop) & modules.metric.eq("mean")].copy()
        matrix = m.pivot(index="module", columns="library_id", values="score").reindex(
            columns=LIBRARIES
        )
        z = matrix.sub(matrix.mean(axis=1), axis=0).div(
            matrix.std(axis=1, ddof=0).replace(0, np.nan), axis=0
        )
        m["display_row_z"] = [z.loc[row.module, row.library_id] for row in m.itertuples()]
        z.index = [
            x.replace("_", " ")
            .replace("p53 p21 cell cycle arrest", "p53 / p21 checkpoint")
            .replace("mitochondria OXPHOS", "Mitochondria / OXPHOS")
            .capitalize()
            for x in z.index
        ]
        name = f"{slug}_existing_state_modules"
        m.to_csv(sources / f"{name}.tsv", sep="\t", index=False)
        save(
            heatmap(
                z,
                f"{DISPLAY[pop]}: library-level state modules",
                "Within-module z score",
                "Existing mean module scores; row z scores use all nine libraries (population SD, ddof=0).\n"
                "Scores are descriptive, modules overlap, and absolute magnitudes cannot be compared between rows.",
                diverging=True,
                limit=3,
            ),
            figure_dir / name,
        )
        manifest.append(
            dict(
                figure=name,
                source_data=f"figure_source_data/{name}.tsv",
                evidence="existing_module_library_scores",
                n_libraries=9,
            )
        )
    pd.DataFrame(manifest).to_csv(out / "FIGURE_MANIFEST.tsv", sep="\t", index=False)
    print("FOCUSED_STANDALONE_FIGURES_COMPLETE")


if __name__ == "__main__":
    main()
