"""Unified nine-library broad-population DE and rescue-ready effect geometry.

The three pairwise contrasts are extracted from one ``~ group`` PyDESeq2 fit per
population.  This module intentionally works on the audited Tier1 pseudobulk raw
count tables only; it never reads normalized single-cell expression or modifies an
AnnData object.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .de_stage1 import (
    CONTRASTS,
    LIBRARIES_BY_GROUP,
    load_broad_inputs,
    prefilter_genes,
)
from .project import project_paths, require_compute_resources, setup_logging
from .pseudobulk import safe_name

ALL_LIBRARIES = sum(LIBRARIES_BY_GROUP.values(), [])
CONTRAST_NAMES = ("OC_vs_Y", "OT_vs_OC", "OT_vs_Y")
ALPHA = 0.05
AGING_LFC_MIN = 0.5
TREATMENT_LFC_MIN = 0.25
PERSISTENT_FRACTION = 0.75
NEAR_YOUNG_LFC = 0.25
EXTREME_ABS_LFC = 20.0


@dataclass(frozen=True)
class UnifiedContrast:
    name: str
    numerator: str
    denominator: str


UNIFIED_CONTRASTS = (
    UnifiedContrast("OC_vs_Y", "OC", "Y"),
    UnifiedContrast("OT_vs_OC", "OT", "OC"),
    UnifiedContrast("OT_vs_Y", "OT", "Y"),
)


@dataclass
class Capture:
    """Result and diagnostics from one PyDESeq2 operation."""

    value: Any
    python_warnings: list[str]
    stderr_text: str
    stdout_text: str

    @property
    def warning_lines(self) -> list[str]:
        lines: list[str] = []
        for source in self.python_warnings + self.stderr_text.splitlines():
            text = str(source).strip()
            if not text:
                continue
            if "warning" in text.lower() or "overflow" in text.lower() or "underflow" in text.lower():
                if text not in lines:
                    lines.append(text)
        return lines


def _capture_operation(func: Callable[[], Any]) -> Capture:
    """Capture Python and OS-level warnings, including worker-process stderr."""

    out = io.StringIO()
    err = io.StringIO()
    with tempfile.TemporaryFile(mode="w+b") as fd_stderr:
        os.set_inheritable(fd_stderr.fileno(), True)
        old_fd = os.dup(2)
        try:
            os.dup2(fd_stderr.fileno(), 2)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    value = func()
                py_warnings = [f"{w.category.__name__}: {w.message}" for w in caught]
        finally:
            os.dup2(old_fd, 2)
            os.close(old_fd)
        fd_stderr.seek(0)
        os_stderr = fd_stderr.read().decode(errors="replace")
    return Capture(value, py_warnings, err.getvalue() + os_stderr, out.getvalue())


def _finite_audit(
    result: pd.DataFrame | None,
    *,
    operation: str,
    population: str,
    warning_lines: list[str],
    warning_source: str,
    expected_cooks_genes: set[str] | None = None,
) -> dict[str, Any]:
    """Summarize warnings and non-finite formal result values."""

    row: dict[str, Any] = {
        "population": population,
        "operation": operation,
        "warning_source": warning_source,
        "warning_count": len(warning_lines),
        "warning_messages": " | ".join(warning_lines) if warning_lines else "None",
        "formal_result_available": result is not None,
        "nonfinite_baseMean": 0,
        "nonfinite_log2FoldChange": 0,
        "nonfinite_lfcSE": 0,
        "nonfinite_stat": 0,
        "nonfinite_pvalue": 0,
        "nonfinite_shrunk_lfc": 0,
        "padj_na_count": 0,
        "pvalue_na_count": 0,
        "cooks_filtered_pvalue_count": 0,
        "unexpected_nonfinite_pvalue": 0,
        "extreme_abs_lfc_gt_20": 0,
        "affected_genes": "None",
        "numerical_status": "not_applicable" if result is None else "finite",
    }
    if result is None:
        return row
    expected_cooks_genes = expected_cooks_genes or set()
    checks = {
        "baseMean": "nonfinite_baseMean",
        "log2FoldChange": "nonfinite_log2FoldChange",
        "lfcSE": "nonfinite_lfcSE",
        "stat": "nonfinite_stat",
        "pvalue": "nonfinite_pvalue",
    }
    # ``log2FC_shrunk`` is intentionally NaN in the OT_vs_OC Wald table because
    # that contrast is algebraically derived from the two apeGLM-shrunk base
    # coefficients.  Check it only for the dedicated apeGLM audit operation.
    if warning_source == "apeGLM_shrinkage":
        checks["log2FC_shrunk"] = "nonfinite_shrunk_lfc"
    affected: set[str] = set()
    for column, out_column in checks.items():
        if column not in result:
            continue
        bad = ~np.isfinite(pd.to_numeric(result[column], errors="coerce"))
        row[out_column] = int(bad.sum())
        if column == "pvalue":
            row["pvalue_na_count"] = int(result[column].isna().sum())
            # A NaN p-value is expected for a gene filtered by DESeq2's Cook's
            # rule.  Only non-Cook's non-finite p-values are numerical anomalies.
            if "gene" in result:
                gene_values = result["gene"].astype(str)
                unexpected = bad & ~gene_values.isin(expected_cooks_genes)
                if unexpected.any():
                    affected.update(gene_values.loc[unexpected].tolist())
        elif bad.any() and "gene" in result:
            affected.update(result.loc[bad, "gene"].astype(str).tolist())
    if "padj" in result:
        row["padj_na_count"] = int(result["padj"].isna().sum())
    if "pvalue" in result and "gene" in result:
        pvalue_na = result["pvalue"].isna()
        row["cooks_filtered_pvalue_count"] = int(
            sum(pvalue_na & result["gene"].astype(str).isin(expected_cooks_genes))
        )
        row["unexpected_nonfinite_pvalue"] = int(
            row["nonfinite_pvalue"] - row["cooks_filtered_pvalue_count"]
        )
    if "log2FoldChange" in result:
        row["extreme_abs_lfc_gt_20"] = int((result["log2FoldChange"].abs() > EXTREME_ABS_LFC).sum())
    nonfinite_columns = [name for name in checks.values() if row[name] > 0]
    if row["unexpected_nonfinite_pvalue"] == 0:
        nonfinite_columns = [name for name in nonfinite_columns if name != "nonfinite_pvalue"]
    row["affected_genes"] = ",".join(sorted(affected)[:100]) if affected else "None"
    if nonfinite_columns or row["extreme_abs_lfc_gt_20"] > 0:
        row["numerical_status"] = "formal_result_anomaly"
    elif warning_lines:
        row["numerical_status"] = "warning_but_formal_values_finite"
    elif row["cooks_filtered_pvalue_count"] > 0:
        row["numerical_status"] = "expected_cooks_filtering"
    elif row["padj_na_count"] > 0:
        row["numerical_status"] = "expected_independent_filtering"
    return row


def _make_metadata(metadata: pd.DataFrame, population: str) -> pd.DataFrame:
    pop_meta = metadata[metadata["population"].astype(str) == population].copy()
    pop_meta["library"] = pop_meta["library"].astype(str)
    if pop_meta["library"].duplicated().any():
        raise ValueError(f"Duplicate library metadata for {population}")
    pop_meta = pop_meta.set_index("library").reindex(ALL_LIBRARIES)
    if pop_meta["group"].isna().any():
        missing = pop_meta.index[pop_meta["group"].isna()].tolist()
        raise ValueError(f"{population} is missing library metadata: {missing}")
    groups = pop_meta["group"].astype(str)
    expected = [lib.split("_")[0] for lib in ALL_LIBRARIES]
    if groups.tolist() != expected:
        raise ValueError(f"Unexpected group assignment for {population}: {groups.to_dict()}")
    pop_meta["group"] = pd.Categorical(groups, categories=["Y", "OC", "OT"], ordered=True)
    return pop_meta


def _contrast_result(
    dds: Any,
    spec: UnifiedContrast,
    *,
    alpha: float,
    inference: Any,
) -> tuple[pd.DataFrame, Any, Capture, Capture | None, str | None]:
    """Extract a Wald contrast from a shared fitted dds and optionally shrink a base coefficient."""

    from pydeseq2.ds import DeseqStats

    def make_stats() -> Any:
        return DeseqStats(
            dds,
            contrast=["group", spec.numerator, spec.denominator],
            alpha=alpha,
            cooks_filter=True,
            independent_filter=True,
            inference=inference,
            quiet=True,
        )

    stats = make_stats()
    summary_capture = _capture_operation(stats.summary)
    raw = stats.results_df.copy()
    raw.index.name = "gene"
    raw = raw.reset_index()
    raw["log2FC_raw"] = raw["log2FoldChange"]
    raw["log2FC_shrunk"] = np.nan
    raw["shrunk_lfc_source"] = "not_run"
    shrink_capture: Capture | None = None
    shrink_coeff: str | None = None

    # apeGLM supports one fitted coefficient at a time.  The two base coefficients
    # are shrunk here; OT_vs_OC is derived from their shrunk values later, while
    # the raw Wald/MLE contrast remains the primary rescue geometry.
    if spec.name in {"OC_vs_Y", "OT_vs_Y"}:
        candidate = spec.numerator
        available = [str(c) for c in stats.LFC.columns]
        matching = [c for c in available if c != "Intercept" and re.search(rf"(?:T\.|\[){candidate}(?:\]|$)", c)]
        if matching:
            shrink_coeff = matching[0]

            def shrink() -> None:
                stats.lfc_shrink(coeff=shrink_coeff)  # type: ignore[arg-type]

            shrink_capture = _capture_operation(shrink)
            if hasattr(stats, "results_df") and "log2FoldChange" in stats.results_df:
                raw["log2FC_shrunk"] = stats.results_df["log2FoldChange"].reindex(raw["gene"]).to_numpy()
                raw["shrunk_lfc_source"] = f"apeGLM:{shrink_coeff}"
    raw["numerator"] = spec.numerator
    raw["denominator"] = spec.denominator
    raw["contrast"] = spec.name
    raw["lfcSE_raw"] = raw["lfcSE"]
    return raw, stats, summary_capture, shrink_capture, shrink_coeff


def _flatten_wide(long_results: pd.DataFrame) -> pd.DataFrame:
    columns = ["baseMean", "log2FoldChange", "log2FC_raw", "log2FC_shrunk", "lfcSE", "stat", "pvalue", "padj"]
    pivot = long_results.pivot(index="gene", columns="contrast", values=columns)
    pivot.columns = [f"{contrast}_{metric}" for metric, contrast in pivot.columns]
    pivot = pivot.reset_index()
    return pivot


def _add_derived_shrunk(wide: pd.DataFrame) -> pd.DataFrame:
    if "OT_vs_OC_log2FC_shrunk" not in wide:
        wide["OT_vs_OC_log2FC_shrunk"] = np.nan
    if {"OC_vs_Y_log2FC_shrunk", "OT_vs_Y_log2FC_shrunk"}.issubset(wide.columns):
        wide["OT_vs_OC_log2FC_shrunk"] = wide["OT_vs_Y_log2FC_shrunk"] - wide["OC_vs_Y_log2FC_shrunk"]
    return wide


def build_rescue_ready_effects(wide: pd.DataFrame) -> pd.DataFrame:
    """Build effect geometry and non-exclusive directional flags from raw Wald LFCs."""

    required = {
        "OC_vs_Y_log2FoldChange",
        "OT_vs_OC_log2FoldChange",
        "OT_vs_Y_log2FoldChange",
        "OC_vs_Y_padj",
        "OT_vs_OC_padj",
    }
    missing = required - set(wide.columns)
    if missing:
        raise ValueError(f"Missing unified contrast columns: {sorted(missing)}")
    out = wide.copy()
    out["aging_effect"] = out["OC_vs_Y_log2FoldChange"]
    out["treatment_effect"] = out["OT_vs_OC_log2FoldChange"]
    out["residual_effect"] = out["OT_vs_Y_log2FoldChange"]
    out["aging_padj"] = out["OC_vs_Y_padj"]
    out["treatment_padj"] = out["OT_vs_OC_padj"]
    out["residual_padj"] = out["OT_vs_Y_padj"]
    out["aging_distance"] = out["aging_effect"].abs()
    out["residual_distance"] = out["residual_effect"].abs()
    out["distance_reduction"] = out["aging_distance"] - out["residual_distance"]
    out["recovery_fraction"] = np.where(
        out["aging_distance"] >= AGING_LFC_MIN,
        1.0 - out["residual_distance"] / out["aging_distance"],
        np.nan,
    )
    out["aging_primary"] = (out["aging_padj"] < ALPHA) & (out["aging_distance"] >= AGING_LFC_MIN)
    out["opposite_direction"] = np.sign(out["treatment_effect"]) == -np.sign(out["aging_effect"])
    out["same_direction"] = (
        np.sign(out["treatment_effect"]) == np.sign(out["aging_effect"])
    ) & (out["treatment_effect"] != 0)
    out["negligible_treatment"] = out["treatment_effect"].abs() < TREATMENT_LFC_MIN
    out["treatment_FDR_supported"] = out["treatment_padj"] < ALPHA
    out["directional_response"] = np.select(
        [out["opposite_direction"], out["same_direction"], out["negligible_treatment"]],
        ["opposite_direction", "same_direction", "negligible_treatment"],
        default="indeterminate",
    )
    out["residual_near_young_effect_size"] = out["residual_distance"] <= NEAR_YOUNG_LFC
    out["near_young_effect_size"] = out["residual_near_young_effect_size"]
    # Literal sign rule requested by the analysis specification.  The absolute
    # residual is retained so exact/near-zero values are not over-interpreted.
    out["overshoot_candidate"] = out["aging_primary"] & (
        np.sign(out["residual_effect"]) != np.sign(out["aging_effect"])
    )
    out["directional_rescue_candidate"] = (
        out["aging_primary"]
        & out["opposite_direction"]
        & (out["treatment_effect"].abs() >= TREATMENT_LFC_MIN)
        & (out["residual_distance"] < out["aging_distance"])
    )
    out["FDR_supported_rescue_candidate"] = (
        out["aging_primary"]
        & out["opposite_direction"]
        & out["treatment_FDR_supported"]
        & (out["residual_distance"] < out["aging_distance"])
    )
    clear_opposite = out["opposite_direction"] & (out["treatment_effect"].abs() >= TREATMENT_LFC_MIN)
    out["persistent_aging"] = (
        out["aging_primary"]
        & (out["residual_distance"] >= PERSISTENT_FRACTION * out["aging_distance"])
        & ~clear_opposite
    )
    out["same_direction_treatment"] = (
        out["aging_primary"]
        & out["same_direction"]
        & (out["treatment_effect"].abs() >= TREATMENT_LFC_MIN)
    )
    # Flags remain non-exclusive.  The primary label makes downstream summaries
    # mutually interpretable; overshoot is prioritized because it describes a
    # distinct crossing geometry, then statistical/directional rescue strength.
    out["primary_classification"] = "aging_primary_other"
    out.loc[~out["aging_primary"], "primary_classification"] = "not_aging_primary"
    priority = [
        ("persistent_aging", "persistent_aging"),
        ("same_direction_treatment", "same_direction_treatment"),
        ("directional_rescue_candidate", "directional_rescue_candidate"),
        ("FDR_supported_rescue_candidate", "FDR_supported_rescue_candidate"),
        ("overshoot_candidate", "overshoot_candidate"),
    ]
    for flag, label in priority:
        out.loc[out["aging_primary"] & out[flag], "primary_classification"] = label
    out["primary_classification"] = pd.Categorical(
        out["primary_classification"],
        categories=[
            "not_aging_primary",
            "aging_primary_other",
            "persistent_aging",
            "same_direction_treatment",
            "directional_rescue_candidate",
            "FDR_supported_rescue_candidate",
            "overshoot_candidate",
        ],
        ordered=True,
    )
    return out


def compare_stage1_unified(
    wide: pd.DataFrame,
    stage1_root: Path,
    population: str,
) -> pd.DataFrame:
    """Compare raw unified LFCs and FDR sets against the prior pairwise fits."""

    rows: list[dict[str, Any]] = []
    unified_columns = {
        "OC_vs_Y": "OC_vs_Y_log2FoldChange",
        "OT_vs_OC": "OT_vs_OC_log2FoldChange",
        "OT_vs_Y": "OT_vs_Y_log2FoldChange",
    }
    for contrast, unified_lfc_col in unified_columns.items():
        path = stage1_root / safe_name(population) / f"{contrast}.tsv.gz"
        if not path.exists():
            rows.append({"population": population, "contrast": contrast, "status": "stage1_file_missing"})
            continue
        old = pd.read_csv(path, sep="\t")
        old_lfc = "log2FC_raw" if "log2FC_raw" in old else "log2FoldChange"
        old_padj = "padj" if "padj" in old else None
        merged = wide[["gene", unified_lfc_col, f"{contrast}_padj"]].merge(
            old[["gene", old_lfc] + ([old_padj] if old_padj else [])], on="gene", how="inner"
        )
        finite = np.isfinite(merged[unified_lfc_col]) & np.isfinite(merged[old_lfc])
        finite_df = merged.loc[finite].copy()
        if len(finite_df) >= 2:
            pearson = float(finite_df[unified_lfc_col].corr(finite_df[old_lfc], method="pearson"))
            spearman = float(finite_df[unified_lfc_col].corr(finite_df[old_lfc], method="spearman"))
            sign = float((np.sign(finite_df[unified_lfc_col]) == np.sign(finite_df[old_lfc])).mean())
            delta = (finite_df[unified_lfc_col] - finite_df[old_lfc]).abs()
            median_delta = float(delta.median())
            p95_delta = float(delta.quantile(0.95))
        else:
            pearson = spearman = sign = median_delta = p95_delta = np.nan
        unified_sig = set(merged.loc[merged[f"{contrast}_padj"] < ALPHA, "gene"])
        old_sig = set(merged.loc[merged[old_padj] < ALPHA, "gene"]) if old_padj else set()
        overlap = len(unified_sig & old_sig)
        union = len(unified_sig | old_sig)
        rows.append(
            {
                "population": population,
                "contrast": contrast,
                "status": "ok",
                "n_genes_overlap": int(len(merged)),
                "n_genes_finite_lfc": int(finite.sum()),
                "pearson_lfc": pearson,
                "spearman_lfc": spearman,
                "sign_concordance": sign,
                "median_abs_lfc_difference": median_delta,
                "p95_abs_lfc_difference": p95_delta,
                "stage1_fdr_genes": int(len(old_sig)),
                "unified_fdr_genes": int(len(unified_sig)),
                "fdr_overlap_genes": int(overlap),
                "fdr_jaccard": float(overlap / union) if union else np.nan,
                "重点报告": bool((np.isfinite(spearman) and spearman < 0.9) or (np.isfinite(sign) and sign < 0.95)),
            }
        )
    return pd.DataFrame(rows)


def _population_summary(
    population: str,
    rescue: pd.DataFrame,
    *,
    model_success: bool,
    max_algebra_error: float,
) -> dict[str, Any]:
    primary = rescue[rescue["aging_primary"]].copy()
    n_primary = len(primary)
    return {
        "population": population,
        "unified_model_success": bool(model_success),
        "n_genes_tested": int(len(rescue)),
        "n_aging_primary": int(n_primary),
        "n_opposite_direction": int(primary["opposite_direction"].sum()),
        "opposite_direction_fraction": float(primary["opposite_direction"].mean()) if n_primary else np.nan,
        "n_directional_rescue": int(primary["directional_rescue_candidate"].sum()),
        "n_FDR_supported_rescue": int(primary["FDR_supported_rescue_candidate"].sum()),
        "n_near_young": int(primary["residual_near_young_effect_size"].sum()),
        "n_persistent_aging": int(primary["persistent_aging"].sum()),
        "n_same_direction_treatment": int(primary["same_direction_treatment"].sum()),
        "n_overshoot": int(primary["overshoot_candidate"].sum()),
        "median_recovery_fraction": float(primary["recovery_fraction"].median()) if n_primary else np.nan,
        "max_algebra_abs_error": float(max_algebra_error),
    }


def _save_geometry_figures(rescue: pd.DataFrame, population: str, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    prefix = figure_dir / safe_name(population)
    primary = rescue[rescue["aging_primary"]]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(rescue["aging_effect"], rescue["treatment_effect"], s=5, alpha=0.15, color="#999999")
    ax.scatter(primary["aging_effect"], primary["treatment_effect"], s=9, alpha=0.55, color="#D62728", label="aging_primary")
    ax.axvline(0, color="black", lw=0.7)
    ax.axhline(0, color="black", lw=0.7)
    ax.set(xlabel="OC vs Y log2FC", ylabel="OT vs OC log2FC", title=f"{population}: aging vs treatment")
    labels = primary.assign(abs_effect=primary["treatment_effect"].abs()).nlargest(5, "abs_effect")
    for _, row in labels.iterrows():
        ax.text(row["aging_effect"], row["treatment_effect"], row["gene"], fontsize=7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_aging_vs_treatment.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(rescue["aging_effect"], rescue["residual_effect"], s=5, alpha=0.15, color="#999999")
    ax.scatter(primary["aging_effect"], primary["residual_effect"], s=9, alpha=0.55, color="#D62728")
    lo = float(np.nanmin([rescue["aging_effect"].min(), rescue["residual_effect"].min()]))
    hi = float(np.nanmax([rescue["aging_effect"].max(), rescue["residual_effect"].max()]))
    ax.plot([lo, hi], [lo, hi], ls="--", color="#333333", lw=0.8, label="y=x")
    ax.axhline(0, color="#1f77b4", lw=0.8, label="y=0")
    ax.set(xlabel="OC vs Y log2FC", ylabel="OT vs Y log2FC", title=f"{population}: aging vs residual")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_aging_vs_residual.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    values = primary["recovery_fraction"].replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(values, bins=30, color="#4C78A8", alpha=0.85)
    ax.axvline(0, color="black", lw=0.7)
    ax.set(xlabel="Recovery fraction (effect-size description)", ylabel="Genes", title=f"{population}: recovery distribution")
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_recovery_fraction.png"), dpi=180)
    plt.close(fig)

    counts = primary["primary_classification"].value_counts().reindex(
        ["FDR_supported_rescue_candidate", "directional_rescue_candidate", "overshoot_candidate", "same_direction_treatment", "persistent_aging", "aging_primary_other"],
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(8, 4.2))
    counts.plot.bar(ax=ax, color="#59A14F")
    ax.set(xlabel="Primary classification", ylabel="Genes", title=f"{population}: rescue-ready categories")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_rescue_classification_counts.png"), dpi=180)
    plt.close(fig)


def _plain_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "(none)"
    return df.head(max_rows).to_string(index=False)


def run_de_stage1_5(config: dict[str, Any], *, allow_low_memory: bool = False) -> Path:
    """Run the unified model and produce rescue-ready descriptive outputs."""

    require_compute_resources(config, allow_low_memory=allow_low_memory)
    paths = project_paths(config)
    logger = setup_logging("09_de_stage1_5_unified", config)
    output_root = paths["results"] / "de_stage1_5"
    figure_root = paths["figures"] / "de_stage1_5"
    output_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    inputs = load_broad_inputs(config)
    counts = inputs["counts"]  # type: ignore[assignment]
    metadata = inputs["metadata"]  # type: ignore[assignment]
    eligibility_path = paths["results"] / "de_stage1" / "broad_population_eligibility.tsv"
    if not eligibility_path.exists():
        raise FileNotFoundError(eligibility_path)
    eligibility = pd.read_csv(eligibility_path, sep="\t")
    primary = eligibility[eligibility["final_de_status"] == "Primary_DE_ready"]
    primary_pops = sorted(primary.groupby("population")["contrast"].nunique().loc[lambda s: s == 3].index.astype(str))
    if not primary_pops:
        raise ValueError("No population has all three Primary_DE_ready contrasts")

    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference

    all_summary: list[dict[str, Any]] = []
    all_concordance: list[pd.DataFrame] = []
    all_warning_audit: list[dict[str, Any]] = []
    all_rescue: list[pd.DataFrame] = []
    n_cpus = int(config.get("pseudobulk", {}).get("n_cpus", 4))
    alpha = float(config.get("pseudobulk", {}).get("alpha", ALPHA))

    for population in primary_pops:
        logger.info("Unified model starting: %s", population)
        if population not in counts.index.get_level_values("population"):
            raise ValueError(f"Population missing from broad counts: {population}")
        pop_counts = counts.loc[population].reindex(ALL_LIBRARIES)
        if pop_counts.isna().any().any():
            raise ValueError(f"Missing unified counts for {population}")
        pop_meta = _make_metadata(metadata, population)
        filtered = prefilter_genes(pop_counts, min_count=10, min_samples=3)
        if filtered.shape[1] == 0:
            raise ValueError(f"No genes pass the unified prefilter for {population}")
        inference = DefaultInference(n_cpus=n_cpus)
        dds = DeseqDataSet(
            counts=filtered.astype(np.int64),
            metadata=pop_meta,
            design="~group",
            refit_cooks=True,
            inference=inference,
            low_memory=False,
        )
        core_capture = _capture_operation(dds.deseq2)
        all_warning_audit.append(
            _finite_audit(
                None,
                population=population,
                operation="DESeq_core_fitting",
                warning_lines=core_capture.warning_lines,
                warning_source="DESeq_core_fitting",
            )
        )

        long_results: list[pd.DataFrame] = []
        stat_objects: dict[str, Any] = {}
        captures: dict[str, tuple[Capture, Capture | None]] = {}
        for spec in UNIFIED_CONTRASTS:
            result, stats, summary_capture, shrink_capture, shrink_coeff = _contrast_result(
                dds, spec, alpha=alpha, inference=inference
            )
            result["genes_before"] = int(pop_counts.shape[1])
            result["genes_after"] = int(filtered.shape[1])
            result["model_gene_universe"] = "9-library unified prefilter: count>=10 in >=3 samples"
            long_results.append(result)
            stat_objects[spec.name] = stats
            captures[spec.name] = (summary_capture, shrink_capture)
            all_warning_audit.append(
                _finite_audit(
                    result,
                    population=population,
                    operation=f"Wald_summary_{spec.name}",
                    warning_lines=summary_capture.warning_lines,
                    warning_source="Wald_test_and_FDR",
                    expected_cooks_genes=set(
                        pd.Index(dds.var_names)[
                            np.asarray(dds.cooks_outlier(), dtype=bool)
                        ].astype(str)
                    ),
                )
            )
            if shrink_capture is not None:
                shrink_result = result[["gene", "log2FC_shrunk"]].copy()
                all_warning_audit.append(
                    _finite_audit(
                        shrink_result.rename(columns={"log2FC_shrunk": "log2FC_shrunk"}),
                        population=population,
                        operation=f"apeGLM_shrinkage_{spec.name}",
                        warning_lines=shrink_capture.warning_lines,
                        warning_source="apeGLM_shrinkage",
                    )
                )

        long_df = pd.concat(long_results, ignore_index=True)
        wide = _add_derived_shrunk(_flatten_wide(long_df))
        derived_ot_oc_shrunk = wide.set_index("gene")["OT_vs_OC_log2FC_shrunk"]
        derived_mask = long_df["contrast"] == "OT_vs_OC"
        long_df.loc[derived_mask, "log2FC_shrunk"] = long_df.loc[derived_mask, "gene"].map(derived_ot_oc_shrunk)
        long_df.loc[derived_mask, "shrunk_lfc_source"] = "derived:apeGLM(OT_vs_Y)-apeGLM(OC_vs_Y)"
        rescue = build_rescue_ready_effects(wide)
        rescue.insert(0, "population", population)
        long_df.insert(0, "population", population)
        pop_dir = output_root / safe_name(population)
        pop_dir.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(pop_dir / "unified_all_genes.tsv.gz", sep="\t", index=False, compression="gzip")
        rescue.to_csv(pop_dir / "rescue_ready_effects.tsv.gz", sep="\t", index=False, compression="gzip")
        _save_geometry_figures(rescue, population, figure_root)

        algebra_error = np.abs(
            rescue["residual_effect"] - rescue["aging_effect"] - rescue["treatment_effect"]
        )
        max_error = float(algebra_error.max())
        all_summary.append(
            _population_summary(population, rescue, model_success=True, max_algebra_error=max_error)
        )
        concordance = compare_stage1_unified(
            wide,
            paths["results"] / "de_stage1",
            population,
        )
        all_concordance.append(concordance)
        all_rescue.append(rescue)
        logger.info("Unified model complete: %s (%d genes)", population, len(filtered.columns))

    summary_df = pd.DataFrame(all_summary)
    concordance_df = pd.concat(all_concordance, ignore_index=True)
    warning_df = pd.DataFrame(all_warning_audit)
    rescue_df = pd.concat(all_rescue, ignore_index=True)
    summary_df.to_csv(output_root / "unified_model_summary.tsv", sep="\t", index=False)
    concordance_df.to_csv(output_root / "stage1_vs_unified_model_concordance.tsv", sep="\t", index=False)
    warning_df.to_csv(output_root / "numerical_warning_audit.tsv", sep="\t", index=False)
    rescue_df.to_csv(output_root / "rescue_candidate_summary.tsv", sep="\t", index=False)

    historical_log = paths["logs"] / "08_de_stage1.log"
    historical_warning_note = ""
    if historical_log.exists():
        text = historical_log.read_text(encoding="utf-8", errors="replace")
        if "RuntimeWarning" in text or "overflow" in text.lower():
            historical_warning_note = (
                "历史 stage1 日志包含 NumPy overflow；本轮通过分离 DESeq core、Wald 和 apeGLM 操作重新定位，"
                "详见 numerical_warning_audit.tsv。"
            )
    core_warning = warning_df[warning_df["warning_source"] == "DESeq_core_fitting"]
    wald_warning = warning_df[warning_df["warning_source"] == "Wald_test_and_FDR"]
    ape_warning = warning_df[warning_df["warning_source"] == "apeGLM_shrinkage"]
    formal_anomalies = warning_df[warning_df["numerical_status"] == "formal_result_anomaly"]
    relation_max = float(summary_df["max_algebra_abs_error"].max()) if not summary_df.empty else np.nan
    concordance_focus = concordance_df[concordance_df.get("重点报告", False) == True] if not concordance_df.empty else pd.DataFrame()
    primary_class_counts = rescue_df[rescue_df["aging_primary"]]["primary_classification"].value_counts().to_dict()
    report = [
        "# DE stage 1.5 unified model report",
        "",
        "本阶段对每个 broad population 使用全部 9 个 library、同一 `~ group` 模型（Y 为 reference），从同一套拟合系数提取 OC vs Y、OT vs OC 和 OT vs Y。输入为 Tier1 broad pseudobulk `layers['counts']` 汇总的 raw integer UMI counts；没有使用 log-normalized X、Harmony、UMAP、scaled expression，也没有运行 subtype DE、GSEA、GO/KEGG、CellChat、trajectory 或 SCENIC。",
        "",
        f"- 统一模型 population 数：{len(summary_df)}；成功：{int(summary_df['unified_model_success'].sum())}。",
        f"- 统一 prefilter：每个 population 在 9 个 sample 中至少 3 个 raw count >=10；每个 population 的 gene universe 对三个 contrast 相同。",
        f"- 三个 contrast 的最大代数误差 `OT-Y - (OC-Y) - (OT-OC)`：{relation_max:.3g}。",
        f"- formal result anomaly：{len(formal_anomalies)} 个 operation；padj NA 单独记录为 independent filtering，不等同于数值异常。",
        "",
        "## Numerical warning localization",
        "",
        historical_warning_note or "未在历史日志中发现可复核的 overflow 文本。",
        f"- DESeq core warning operations：{int((core_warning['warning_count'] > 0).sum())}；Wald/FDR：{int((wald_warning['warning_count'] > 0).sum())}；apeGLM：{int((ape_warning['warning_count'] > 0).sum())}。",
        "- warning 是否产生非有限正式结果，必须以 `numerical_warning_audit.tsv` 中的 `numerical_status`、各 nonfinite 列和 affected_genes 为准；本报告不静默忽略任何正式结果异常。",
        "",
        "## Unified model summary",
        "",
        _plain_table(summary_df),
        "",
        "## Stage1 pairwise versus unified concordance",
        "",
        _plain_table(concordance_df),
        "",
        f"重点报告阈值为 Spearman < 0.9 或 sign concordance < 0.95；命中 {len(concordance_focus)} 个 population × contrast。",
        "",
        "## Rescue-ready descriptive geometry",
        "",
        "`aging_primary` 定义为统一模型 OC vs Y 的 Wald/MLE `padj < 0.05` 且 `abs(LFC) >= 0.5`。`0.25` 是本分析的工作效应阈值，不是生物学真理。`treatment_FDR_supported` 只表示 OT vs OC 的 FDR 支持，不能把 FDR 缺失当作无效治疗。`near_young_effect_size` 使用 `abs(OT-Y LFC) <= 0.25`，不是统计等价性检验。所有 rescue flags 可重叠；`primary_classification` 只提供一个优先级标签，优先级为 overshoot、FDR-supported rescue、directional rescue、same-direction treatment、persistent aging、aging_primary_other。",
        "",
        _plain_table(summary_df[["population", "n_aging_primary", "n_opposite_direction", "opposite_direction_fraction", "n_directional_rescue", "n_FDR_supported_rescue", "n_near_young", "n_persistent_aging", "n_same_direction_treatment", "n_overshoot", "median_recovery_fraction"]]),
        "",
        f"全局 primary classification counts：{primary_class_counts}",
        "",
        "本阶段只提供统一模型、数值审计、代数一致性和方向性分类；没有将任何结果命名为 anti-aging、rescue mechanism、restored 或 reversed，也没有根据 DE 结果修改 annotation。",
    ]
    (output_root / "DE_STAGE1_5_UNIFIED_MODEL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    logger.info("UNIFIED_RESCUE_READY_DE_COMPLETE: %s", output_root)
    print("UNIFIED_RESCUE_READY_DE_COMPLETE")
    return output_root
