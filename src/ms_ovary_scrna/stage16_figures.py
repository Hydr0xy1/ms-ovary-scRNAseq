"""Independent publication-style figures for Stage 16-22.

The module deliberately avoids composite/multi-panel layouts. Every figure has
one evidentiary role and writes its own source-data table.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


GROUP_ORDER = ["Y", "OC", "OT"]
GROUP_COLORS = {"Y": "#4C78A8", "OC": "#8E8E8E", "OT": "#D47A63"}
POP_COLORS = {"Granulosa": "#4C78A8", "Stromal_fibroblast": "#B07AA1"}


def _configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t", compression="infer")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _display_label(value: object) -> str:
    """Convert machine-readable labels into concise figure text."""
    return str(value).replace("_", " ")


def _save(fig: mpl.figure.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _write_source(frame: pd.DataFrame, source_dir: Path, name: str) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(source_dir / f"{name}.tsv", sep="\t", index=False)


def _style_axis(ax: mpl.axes.Axes) -> None:
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.7, length=2.5)


def _library_points(
    frame: pd.DataFrame,
    value: str,
    title: str,
    ylabel: str,
    group_order: Iterable[str] = GROUP_ORDER,
) -> mpl.figure.Figure:
    order = list(group_order)
    fig, ax = plt.subplots(figsize=(3.50, 2.75))
    rng = np.random.default_rng(20260920)
    group_sizes: list[int] = []
    for x, group in enumerate(order):
        block = frame.loc[frame["group"].astype(str).eq(group)]
        values = pd.to_numeric(block[value], errors="coerce").dropna()
        group_sizes.append(len(values))
        jitter = rng.normal(0, 0.035, len(values))
        ax.scatter(
            np.repeat(x, len(values)) + jitter,
            values,
            s=24,
            color=GROUP_COLORS.get(group, "#555555"),
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        if len(values):
            mean = float(values.mean())
            ax.plot([x - 0.18, x + 0.18], [mean, mean], color="#222222", lw=1.1)
    tick_labels = [
        f"{_display_label(group)}\n(n={size})"
        for group, size in zip(order, group_sizes, strict=True)
    ]
    ax.set_xticks(range(len(order)), tick_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    _style_axis(ax)
    fig.tight_layout()
    return fig


def generate_stage16_figures(project_root: Path) -> list[Path]:
    """Generate independent SVG/PDF/PNG/TIFF figures and source data."""
    _configure()
    stage = project_root / "results/deep_dive_stage16_ml"
    synthesis = stage / "10_synthesis"
    main_dir = synthesis / "main_figure_drafts"
    supp_dir = synthesis / "supplementary_figure_drafts"
    source_dir = synthesis / "figure_source_data"
    outputs: list[Path] = []

    library = _read(stage / "03_scvi_reference/SCVI_LATENT_LIBRARY_SUMMARY.tsv")
    if not library.empty:
        for population, block in library.groupby("population", observed=True):
            name = f"scvi_library_outlier_{_safe_name(population)}"
            summary = (
                block.groupby(["sample_id", "group"], observed=True)[
                    "library_outlier_robust_z"
                ]
                .median()
                .reset_index()
            )
            summary["group"] = pd.Categorical(
                summary["group"], GROUP_ORDER, ordered=True
            )
            summary = summary.sort_values(["group", "sample_id"])
            fig = _library_points(
                summary,
                "library_outlier_robust_z",
                f"{population}: scVI library outlier score",
                "Robust z distance to internal centroid",
            )
            base = supp_dir / name
            _save(fig, base)
            _write_source(summary, source_dir, name)
            outputs.append(base.with_suffix(".png"))

    stability = _read(stage / "03_scvi_reference/SCVI_STABILITY.tsv")
    if not stability.empty:
        for population, block in stability.groupby("population", observed=True):
            name = f"scvi_seed_geometry_stability_{_safe_name(population)}"
            block = block.copy()
            block["seed_pair"] = (
                block["seed_a"].astype(str) + "-" + block["seed_b"].astype(str)
            )
            fig, ax = plt.subplots(figsize=(3.50, 2.75))
            x = np.arange(len(block))
            ax.scatter(
                x,
                block["library_distance_spearman"],
                s=30,
                color=POP_COLORS.get(str(population), "#4C78A8"),
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax.axhline(0.8, color="#B0B0B0", ls="--", lw=0.8)
            ax.set_xticks(x, block["seed_pair"], rotation=30, ha="right")
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel("Spearman r of library distances")
            ax.set_xlabel("Seed pair")
            ax.set_title(
                f"{population}: cross-seed geometry stability",
                loc="left",
                fontweight="bold",
            )
            _style_axis(ax)
            fig.tight_layout()
            base = supp_dir / name
            _save(fig, base)
            _write_source(block, source_dir, name)
            outputs.append(base.with_suffix(".png"))

    geometry = _read(stage / "04_latent_geometry_ot/LATENT_GEOMETRY.tsv")
    if not geometry.empty:
        gran = geometry.loc[
            geometry["population"].astype(str).eq("Granulosa")
            & geometry["analysis_level"].astype(str).eq("broad")
        ].copy()
        for value, label, reference, name in [
            (
                "distance_ratio_OT_over_OC",
                "distance(OT,Y) / distance(OC,Y)",
                1.0,
                "granulosa_latent_distance_ratio",
            ),
            (
                "aging_treatment_cosine",
                "Cosine: aging vs treatment vector",
                0.0,
                "granulosa_aging_treatment_cosine",
            ),
            (
                "treatment_residual_ratio",
                "Orthogonal residual / treatment norm",
                None,
                "granulosa_treatment_orthogonal_fraction",
            ),
        ]:
            if gran.empty:
                continue
            fig, ax = plt.subplots(figsize=(3.50, 2.75))
            x = np.arange(len(gran))
            ax.scatter(
                x,
                gran[value],
                s=32,
                color=POP_COLORS["Granulosa"],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            if reference is not None:
                ax.axhline(reference, color="#7A7A7A", ls="--", lw=0.8)
            ax.set_xticks(x, gran["seed"].astype(str))
            ax.set_xlabel("scVI seed")
            ax.set_ylabel(label)
            ax.set_title(
                "Granulosa: latent treatment geometry",
                loc="left",
                fontweight="bold",
            )
            _style_axis(ax)
            fig.tight_layout()
            base = main_dir / name
            _save(fig, base)
            _write_source(gran, source_dir, name)
            outputs.append(base.with_suffix(".png"))

    ot = _read(stage / "04_latent_geometry_ot/OT_SENSITIVITY.tsv")
    if not ot.empty:
        block = ot.loc[
            ot["population"].astype(str).eq("Granulosa")
            & ot["analysis_level"].astype(str).eq("broad")
            & ot["metric"].astype(str).eq("Sinkhorn_root_cost")
        ].copy()
        if not block.empty:
            summary = (
                block.groupby(["comparison", "regularization"], observed=True)[
                    "median"
                ]
                .median()
                .reset_index()
            )
            fig, ax = plt.subplots(figsize=(3.50, 2.75))
            colors = {
                "OC_vs_Y": "#8E8E8E",
                "OT_vs_Y": "#D47A63",
                "OT_vs_OC": "#4C78A8",
            }
            for comparison, sub in summary.groupby("comparison", observed=True):
                sub = sub.sort_values("regularization")
                ax.plot(
                    sub["regularization"],
                    sub["median"],
                    marker="o",
                    ms=4,
                    lw=1.2,
                    color=colors.get(str(comparison), "#555555"),
                    label=_display_label(comparison),
                )
            ax.set_xlabel("Sinkhorn regularization multiplier")
            ax.set_ylabel("Median library-pair root cost")
            ax.set_title(
                "Granulosa: Sinkhorn sensitivity",
                loc="left",
                fontweight="bold",
            )
            ax.legend(title=None)
            _style_axis(ax)
            fig.tight_layout()
            name = "granulosa_sinkhorn_regularization_sensitivity"
            base = main_dir / name
            _save(fig, base)
            _write_source(summary, source_dir, name)
            outputs.append(base.with_suffix(".png"))

    age = _read(stage / "05_external_age_models/EXTERNAL_AGE_SCORE_LIBRARY.tsv")
    if not age.empty:
        for (population, model), block in age.groupby(
            ["population", "model"], observed=True
        ):
            name = (
                f"external_age_score_{_safe_name(population)}_"
                f"{_safe_name(model)}"
            )
            block = block.copy()
            block["group"] = pd.Categorical(
                block["group"], GROUP_ORDER, ordered=True
            )
            block = block.sort_values(["group", "library_id"])
            fig = _library_points(
                block,
                "external_age_score",
                f"{population}: {model} external age score",
                "External age-related score",
            )
            base = main_dir if population == "Granulosa" else supp_dir
            out_base = base / name
            _save(fig, out_base)
            _write_source(block, source_dir, name)
            outputs.append(out_base.with_suffix(".png"))

    cvi = _read(
        stage / "06_contrastivevi/CONTRASTIVEVI_LIBRARY_SCORES.tsv"
    )
    if not cvi.empty:
        full = cvi.loc[cvi["run_type"].astype(str).eq("full")].copy()
        for context, block in full.groupby("context", observed=True):
            summary = (
                block.groupby(["library_id", "group"], observed=True)[
                    "salient_norm"
                ]
                .median()
                .reset_index()
            )
            if summary.empty:
                continue
            name = f"contrastivevi_salient_score_{_safe_name(context)}"
            fig = _library_points(
                summary,
                "salient_norm",
                f"{_display_label(context)}: contrastiveVI salient score",
                "Median salient latent norm",
                group_order=["OC", "OT"],
            )
            base = supp_dir / name
            _save(fig, base)
            _write_source(summary, source_dir, name)
            outputs.append(base.with_suffix(".png"))

    manifest = pd.DataFrame(
        {
            "preview_png": [str(p.relative_to(stage)) for p in outputs],
            "svg": [str(p.with_suffix(".svg").relative_to(stage)) for p in outputs],
            "pdf": [str(p.with_suffix(".pdf").relative_to(stage)) for p in outputs],
            "tiff_600dpi": [
                str(p.with_suffix(".tiff").relative_to(stage)) for p in outputs
            ],
        }
    )
    manifest.to_csv(
        synthesis / "FIGURE_MANIFEST.tsv", sep="\t", index=False
    )
    return outputs
