"""Exact library-label permutation validation for Stage 1.5 effect geometry.

The six aged libraries are exhaustively split into three pseudo-OC and three
pseudo-OT libraries (20 labelled allocations).  Y libraries never move.  Each
allocation is fitted from raw pseudobulk counts with its own ``~ group`` model,
and the complete aging-primary selection and rescue-ready classification are
repeated.  No pathway or mechanism analysis is performed here.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .de_stage1 import LIBRARIES_BY_GROUP, load_broad_inputs, prefilter_genes
from .de_stage1_5 import (
    ALPHA,
    ALL_LIBRARIES,
    UNIFIED_CONTRASTS,
    _capture_operation,
    build_rescue_ready_effects,
)
from .project import project_paths, require_compute_resources, setup_logging
from .pseudobulk import safe_name

AGED_LIBRARIES = LIBRARIES_BY_GROUP["OC"] + LIBRARIES_BY_GROUP["OT"]
EXPECTED_PERMUTATIONS = 20
PRIMARY_ENDPOINT = "FDR_supported_rescue_fraction"
SECONDARY_ENDPOINTS = ("directional_rescue_fraction", "median_recovery_fraction")
HIGHER_IS_MORE_EXTREME = {
    PRIMARY_ENDPOINT: True,
    "directional_rescue_fraction": True,
    "median_recovery_fraction": True,
    "whole_all_spearman": False,
    "whole_all_cosine_similarity": False,
}


def enumerate_permutation_assignments() -> pd.DataFrame:
    """Return all 20 exact 3-vs-3 aged-library allocations plus fixed Y rows."""

    observed_oc = frozenset(LIBRARIES_BY_GROUP["OC"])
    rows: list[dict[str, Any]] = []
    combinations = list(itertools.combinations(AGED_LIBRARIES, 3))
    if len(combinations) != EXPECTED_PERMUTATIONS:
        raise AssertionError("Expected C(6,3)=20 label assignments")
    # Lexicographic input order makes the real OC set P001, but downstream code
    # relies only on ``is_observed`` rather than this convenient ordering.
    for index, pseudo_oc_tuple in enumerate(combinations, start=1):
        pseudo_oc = frozenset(pseudo_oc_tuple)
        permutation_id = f"P{index:03d}"
        is_observed = pseudo_oc == observed_oc
        for library in ALL_LIBRARIES:
            original_group = library.split("_", maxsplit=1)[0]
            if library in LIBRARIES_BY_GROUP["Y"]:
                model_group = "Y"
                assignment_label = "Y_fixed"
            elif library in pseudo_oc:
                model_group = "OC"
                assignment_label = "pseudo_OC"
            else:
                model_group = "OT"
                assignment_label = "pseudo_OT"
            rows.append(
                {
                    "permutation_id": permutation_id,
                    "is_observed": is_observed,
                    "library_id": library,
                    "original_group": original_group,
                    "model_group": model_group,
                    "assignment_label": assignment_label,
                }
            )
    result = pd.DataFrame(rows)
    validate_permutation_assignments(result)
    return result


def validate_permutation_assignments(assignments: pd.DataFrame) -> None:
    """Fail unless assignments are the complete exact design requested."""

    required = {"permutation_id", "is_observed", "library_id", "model_group"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Permutation assignments missing columns: {sorted(missing)}")
    grouped = assignments.groupby("permutation_id", observed=True)
    if grouped.ngroups != EXPECTED_PERMUTATIONS:
        raise ValueError(f"Expected 20 permutations, found {grouped.ngroups}")
    for permutation_id, sub in grouped:
        if set(sub["library_id"]) != set(ALL_LIBRARIES):
            raise ValueError(f"{permutation_id} does not contain all nine libraries")
        sizes = sub["model_group"].value_counts().to_dict()
        if sizes != {"Y": 3, "OC": 3, "OT": 3}:
            raise ValueError(f"{permutation_id} has invalid group sizes: {sizes}")
        fixed_y = sub[sub["library_id"].isin(LIBRARIES_BY_GROUP["Y"])]
        if not (fixed_y["model_group"] == "Y").all():
            raise ValueError(f"{permutation_id} moves a Y library")
    observed_ids = assignments.loc[assignments["is_observed"], "permutation_id"].unique()
    if len(observed_ids) != 1:
        raise ValueError(f"Expected exactly one observed assignment, found {observed_ids.tolist()}")


def _metadata_for_assignment(assignments: pd.DataFrame) -> pd.DataFrame:
    sub = assignments.set_index("library_id").reindex(ALL_LIBRARIES)
    if sub["model_group"].isna().any():
        raise ValueError("Assignment lacks one or more required libraries")
    metadata = pd.DataFrame(index=ALL_LIBRARIES)
    metadata.index.name = "library"
    metadata["group"] = pd.Categorical(
        sub["model_group"].astype(str), categories=["Y", "OC", "OT"], ordered=True
    )
    return metadata


def _fit_permutation(
    filtered_counts: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    alpha: float,
    n_cpus: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Fit one unified model and return the three raw Wald contrasts in wide form."""

    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    metadata = _metadata_for_assignment(assignments)
    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        counts=filtered_counts.reindex(ALL_LIBRARIES).astype(np.int64),
        metadata=metadata,
        design="~group",
        refit_cooks=True,
        inference=inference,
        low_memory=False,
    )
    warnings_seen: list[str] = []
    core_capture = _capture_operation(dds.deseq2)
    warnings_seen.extend(core_capture.warning_lines)
    wide: pd.DataFrame | None = None
    for spec in UNIFIED_CONTRASTS:
        stats = DeseqStats(
            dds,
            contrast=["group", spec.numerator, spec.denominator],
            alpha=alpha,
            cooks_filter=True,
            independent_filter=True,
            inference=inference,
            quiet=True,
        )
        summary_capture = _capture_operation(stats.summary)
        warnings_seen.extend(summary_capture.warning_lines)
        result = stats.results_df[["log2FoldChange", "pvalue", "padj"]].copy()
        result.columns = [f"{spec.name}_{column}" for column in result.columns]
        wide = result if wide is None else wide.join(result, how="inner")
    if wide is None:
        raise RuntimeError("No Wald contrasts were produced")
    wide.index.name = "gene"
    return wide.reset_index(), warnings_seen


def cosine_similarity(x: Iterable[float], y: Iterable[float]) -> float:
    """Finite-pair cosine similarity; return NaN for an empty or zero vector."""

    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    if len(a) == 0:
        return float("nan")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else float("nan")


def whole_signature_metrics(
    rescue: pd.DataFrame,
    *,
    population: str,
    permutation_id: str,
    is_observed: bool,
) -> pd.DataFrame:
    """Calculate all-tested-gene primary and aging-primary-only secondary metrics."""

    rows: list[dict[str, Any]] = []
    for scope, mask in (
        ("all_tested_genes", pd.Series(True, index=rescue.index)),
        ("aging_primary_only", rescue["aging_primary"].astype(bool)),
    ):
        frame = rescue.loc[mask, ["aging_effect", "treatment_effect"]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        pearson = (
            float(frame["aging_effect"].corr(frame["treatment_effect"], method="pearson"))
            if len(frame) >= 2
            else np.nan
        )
        spearman = (
            float(frame["aging_effect"].corr(frame["treatment_effect"], method="spearman"))
            if len(frame) >= 2
            else np.nan
        )
        cosine = cosine_similarity(frame["aging_effect"], frame["treatment_effect"])
        rows.append(
            {
                "population": population,
                "permutation_id": permutation_id,
                "is_observed": bool(is_observed),
                "scope": scope,
                "n_genes": int(len(frame)),
                "pearson": pearson,
                "spearman": spearman,
                "cosine_similarity": cosine,
            }
        )
    return pd.DataFrame(rows)


def summarize_permutation(
    rescue: pd.DataFrame,
    *,
    population: str,
    permutation_id: str,
    is_observed: bool,
    max_algebra_abs_error: float,
    whole: pd.DataFrame,
    warning_lines: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize the fully reselected classification for one population/allocation."""

    primary = rescue[rescue["aging_primary"]].copy()
    n_primary = len(primary)
    all_whole = whole[whole["scope"] == "all_tested_genes"].iloc[0]
    primary_whole = whole[whole["scope"] == "aging_primary_only"].iloc[0]
    warning_lines = list(dict.fromkeys(warning_lines or []))

    def fraction(column: str) -> float:
        return float(primary[column].mean()) if n_primary else np.nan

    return {
        "population": population,
        "permutation_id": permutation_id,
        "is_observed": bool(is_observed),
        "n_genes_tested": int(len(rescue)),
        "n_aging_primary": int(n_primary),
        "n_directional_rescue": int(primary["directional_rescue_candidate"].sum()),
        "directional_rescue_fraction": fraction("directional_rescue_candidate"),
        "n_FDR_supported_rescue": int(primary["FDR_supported_rescue_candidate"].sum()),
        "FDR_supported_rescue_fraction": fraction("FDR_supported_rescue_candidate"),
        "median_recovery_fraction": (
            float(primary["recovery_fraction"].median()) if n_primary else np.nan
        ),
        "fraction_residual_distance_reduced": (
            float((primary["residual_distance"] < primary["aging_distance"]).mean())
            if n_primary
            else np.nan
        ),
        "fraction_near_young": fraction("residual_near_young_effect_size"),
        "fraction_overshoot": fraction("overshoot_candidate"),
        "aging_primary_pearson": float(primary_whole["pearson"]),
        "aging_primary_spearman": float(primary_whole["spearman"]),
        "aging_primary_cosine_similarity": float(primary_whole["cosine_similarity"]),
        "whole_all_pearson": float(all_whole["pearson"]),
        "whole_all_spearman": float(all_whole["spearman"]),
        "whole_all_cosine_similarity": float(all_whole["cosine_similarity"]),
        "max_algebra_abs_error": float(max_algebra_abs_error),
        "warning_count": int(len(warning_lines)),
        "warning_messages": " | ".join(warning_lines) if warning_lines else "None",
    }


def effect_magnitude_summary(
    rescue: pd.DataFrame,
    *,
    population: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize |treatment|/|aging| among observed aging-primary genes."""

    primary = rescue[rescue["aging_primary"]].copy()
    ratio = (primary["treatment_effect"].abs() / primary["aging_effect"].abs()).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    detail = primary.loc[ratio.index, ["gene", "aging_effect", "treatment_effect"]].copy()
    detail.insert(0, "population", population)
    detail["absolute_treatment_to_aging_ratio"] = ratio
    row = {
        "population": population,
        "n_aging_primary_with_finite_ratio": int(len(ratio)),
        "ratio_p25": float(ratio.quantile(0.25)) if len(ratio) else np.nan,
        "ratio_median": float(ratio.median()) if len(ratio) else np.nan,
        "ratio_p75": float(ratio.quantile(0.75)) if len(ratio) else np.nan,
        "fraction_ratio_ge_0.25": float((ratio >= 0.25).mean()) if len(ratio) else np.nan,
        "fraction_ratio_ge_0.5": float((ratio >= 0.5).mean()) if len(ratio) else np.nan,
        "fraction_ratio_ge_0.75": float((ratio >= 0.75).mean()) if len(ratio) else np.nan,
        "fraction_ratio_ge_1.0": float((ratio >= 1.0).mean()) if len(ratio) else np.nan,
    }
    return pd.DataFrame([row]), detail


def empirical_null_comparison(
    permutation_summary: pd.DataFrame,
    metric: str,
    *,
    higher_is_more_extreme: bool,
) -> dict[str, Any]:
    """Compare observed with the 19-label null using the requested +1 formula."""

    observed_rows = permutation_summary[permutation_summary["is_observed"]]
    null = pd.to_numeric(
        permutation_summary.loc[~permutation_summary["is_observed"], metric], errors="coerce"
    ).dropna()
    if len(observed_rows) != 1:
        raise ValueError(f"Expected one observed row for {metric}")
    observed = float(observed_rows.iloc[0][metric])
    if len(null) != EXPECTED_PERMUTATIONS - 1 or not np.isfinite(observed):
        return {
            "observed": observed,
            "null_n": int(len(null)),
            "null_min": float(null.min()) if len(null) else np.nan,
            "null_median": float(null.median()) if len(null) else np.nan,
            "null_max": float(null.max()) if len(null) else np.nan,
            "observed_rank": np.nan,
            "p_empirical": np.nan,
        }
    if higher_is_more_extreme:
        null_as_or_more_extreme = int((null >= observed).sum())
    else:
        null_as_or_more_extreme = int((null <= observed).sum())
    return {
        "observed": observed,
        "null_n": int(len(null)),
        "null_min": float(null.min()),
        "null_median": float(null.median()),
        "null_max": float(null.max()),
        # Rank 1 is most extreme; ties are handled conservatively, matching p.
        "observed_rank": int(1 + null_as_or_more_extreme),
        "p_empirical": float((1 + null_as_or_more_extreme) / (1 + len(null))),
    }


def build_population_evidence(permutation_summary: pd.DataFrame) -> pd.DataFrame:
    """Create one population-level row with calibrated endpoint evidence."""

    rows: list[dict[str, Any]] = []
    for population, sub in permutation_summary.groupby("population", sort=True, observed=True):
        observed = sub[sub["is_observed"]].iloc[0]
        row: dict[str, Any] = {
            "population": population,
            "n_aging_primary_observed": int(observed["n_aging_primary"]),
        }
        for metric, higher in HIGHER_IS_MORE_EXTREME.items():
            comparison = empirical_null_comparison(sub, metric, higher_is_more_extreme=higher)
            for key, value in comparison.items():
                row[f"{metric}_{key}"] = value
        row["primary_FDR_endpoint_at_minimum_p"] = bool(
            np.isclose(row[f"{PRIMARY_ENDPOINT}_p_empirical"], 0.05)
        )
        row["whole_spearman_at_minimum_p"] = bool(
            row["whole_all_spearman_observed"] < 0
            and np.isclose(row["whole_all_spearman_p_empirical"], 0.05)
        )
        row["whole_cosine_at_minimum_p"] = bool(
            row["whole_all_cosine_similarity_observed"] < 0
            and np.isclose(row["whole_all_cosine_similarity_p_empirical"], 0.05)
        )
        row["permutation_supported_signature"] = bool(
            row["primary_FDR_endpoint_at_minimum_p"]
            or row["whole_spearman_at_minimum_p"]
            or row["whole_cosine_at_minimum_p"]
        )
        row["population_evidence_level"] = (
            "Level_3_permutation_supported_signature"
            if row["permutation_supported_signature"]
            else "Level_2_or_lower_no_exact_permutation_support"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def validate_observed_against_stage1_5(
    permutation_summary: pd.DataFrame,
    stage1_5_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Require the re-fitted observed allocation to reproduce Stage 1.5 geometry."""

    observed = permutation_summary[permutation_summary["is_observed"]].copy()
    merged = observed.merge(
        stage1_5_summary,
        on="population",
        how="left",
        suffixes=("_stage1_6", "_stage1_5"),
        validate="one_to_one",
    )
    checks = {
        "n_aging_primary": 0.0,
        "n_directional_rescue": 0.0,
        "n_FDR_supported_rescue": 0.0,
        "median_recovery_fraction": 1e-10,
    }
    rows: list[dict[str, Any]] = []
    for _, record in merged.iterrows():
        for metric, tolerance in checks.items():
            current = float(record[f"{metric}_stage1_6"])
            previous = float(record[f"{metric}_stage1_5"])
            absolute_difference = abs(current - previous)
            matches = bool(np.isfinite(absolute_difference) and absolute_difference <= tolerance)
            rows.append(
                {
                    "population": record["population"],
                    "metric": metric,
                    "stage1_6_observed": current,
                    "stage1_5_reference": previous,
                    "absolute_difference": absolute_difference,
                    "tolerance": tolerance,
                    "matches": matches,
                }
            )
    audit = pd.DataFrame(rows)
    if len(audit) != 7 * len(checks) or not audit["matches"].all():
        failures = audit.loc[~audit["matches"]].to_dict("records")
        raise RuntimeError(f"Observed permutation does not reproduce Stage 1.5: {failures}")
    return audit


def _add_observed_gene_levels(
    rescue: pd.DataFrame,
    *,
    population: str,
    population_level3: bool,
) -> pd.DataFrame:
    out = rescue.copy()
    out.insert(0, "population", population)
    out["Level_1_directional_candidate"] = out["directional_rescue_candidate"].astype(bool)
    out["Level_2_DE_supported_candidate"] = out["FDR_supported_rescue_candidate"].astype(bool)
    out["Level_3_population_signature_supported"] = bool(population_level3)
    out["evidence_note"] = np.select(
        [
            out["Level_2_DE_supported_candidate"] & population_level3,
            out["Level_2_DE_supported_candidate"],
            out["Level_1_directional_candidate"],
        ],
        [
            (
                "gene-level DE support within a permutation-supported population; "
                "not a confirmed rescue gene"
            ),
            "gene-level DE support; population did not reach minimum exact-permutation p",
            "directional candidate only",
        ],
        default="not a directional candidate",
    )
    return out


def _plot_metric(
    summary: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    populations = sorted(summary["population"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharey=False)
    axes_flat = axes.ravel()
    for ax, population in zip(axes_flat, populations, strict=False):
        sub = summary[summary["population"] == population]
        null = sub.loc[~sub["is_observed"], metric].dropna().to_numpy(dtype=float)
        observed = float(sub.loc[sub["is_observed"], metric].iloc[0])
        ax.scatter(np.ones(len(null)), null, s=28, color="#8C8C8C", alpha=0.75, label="19 null")
        ax.scatter([1], [observed], s=70, color="#D62728", marker="D", label="observed")
        ax.set_xlim(0.75, 1.25)
        ax.set_xticks([])
        ax.set_title(population.replace("_", "\n"), fontsize=10)
        ax.set_ylabel(ylabel)
    for ax in axes_flat[len(populations):]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", frameon=False)
    fig.suptitle("Observed assignment versus exact 19-allocation null")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_effect_magnitude(details: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    populations = sorted(details["population"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharex=True)
    axes_flat = axes.ravel()
    for ax, population in zip(axes_flat, populations, strict=False):
        values = details.loc[
            details["population"] == population, "absolute_treatment_to_aging_ratio"
        ].clip(upper=3)
        ax.hist(values, bins=35, color="#4C78A8", alpha=0.85)
        for threshold in (0.25, 0.5, 0.75, 1.0):
            ax.axvline(threshold, color="#333333", lw=0.6, ls="--")
        ax.set_title(population.replace("_", "\n"), fontsize=10)
        ax.set_xlabel("|treatment effect| / |aging effect| (clipped at 3)")
        ax.set_ylabel("Aging-primary genes")
    for ax in axes_flat[len(populations):]:
        ax.axis("off")
    fig.suptitle("Observed treatment-to-aging effect magnitude")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plain_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if df.empty:
        return "(none)"
    return df.loc[:, columns].to_string(index=False) if columns else df.to_string(index=False)


def _report(
    output_root: Path,
    evidence: pd.DataFrame,
    magnitude: pd.DataFrame,
    permutation_summary: pd.DataFrame,
) -> None:
    columns = [
        "population",
        "n_aging_primary_observed",
        "directional_rescue_fraction_observed",
        "directional_rescue_fraction_null_median",
        "directional_rescue_fraction_observed_rank",
        "directional_rescue_fraction_p_empirical",
        "FDR_supported_rescue_fraction_observed",
        "FDR_supported_rescue_fraction_null_median",
        "FDR_supported_rescue_fraction_observed_rank",
        "FDR_supported_rescue_fraction_p_empirical",
        "median_recovery_fraction_observed",
        "median_recovery_fraction_null_median",
        "median_recovery_fraction_observed_rank",
        "median_recovery_fraction_p_empirical",
        "whole_all_spearman_observed",
        "whole_all_spearman_null_median",
        "whole_all_spearman_observed_rank",
        "whole_all_spearman_p_empirical",
        "whole_all_cosine_similarity_observed",
        "whole_all_cosine_similarity_p_empirical",
        "permutation_supported_signature",
    ]
    report = [
        "# DE Stage 1.6 exact permutation validation",
        "",
        (
            "本阶段枚举 6 个十月龄 library 的全部 C(6,3)=20 个三对三标签排列；三个 Y "
            "library 始终固定。每个 population、每个排列均从相同 raw integer pseudobulk "
            "counts 重新拟合 `~ group`，并重新完成 pseudo-OC vs Y 的 aging-primary 筛选。"
            "因此 null 同时包含共享 pseudo-OC contrast、aging-gene selection 和 "
            "regression-to-the-mean 影响。"
        ),
        "",
        "- observed：真实 OC/OT 标签排列；null reference：其余 19 个排列。",
        (
            "- primary endpoint：FDR-supported rescue fraction；secondary endpoints："
            "directional-rescue fraction 和 median recovery fraction。"
        ),
        (
            "- 经验 p：`(1 + # null at least as extreme as observed) / 20`；"
            "20 个精确排列决定最小可能 p=0.05。"
        ),
        (
            "- whole-signature primary scope：全部共同 tested genes；aging-primary-only "
            "仅为 secondary。负 Spearman/cosine 表示 effect vector 方向相反。"
        ),
        "",
        "## Population-level permutation evidence",
        "",
        _plain_table(evidence, columns),
        "",
        "## Observed effect magnitude",
        "",
        (
            "比值为 `abs(treatment effect) / abs(aging effect)`，仅描述 effect size，"
            "不是生物学恢复百分比。"
        ),
        "",
        _plain_table(magnitude),
        "",
        "## Completeness and safeguards",
        "",
        f"- population 数：{permutation_summary['population'].nunique()}。",
        (
            f"- 成功的 population × permutation：{len(permutation_summary)} / "
            f"{7 * EXPECTED_PERMUTATIONS}。"
        ),
        (
            "- 每个 population 的 permutation 数："
            f"{permutation_summary.groupby('population').size().to_dict()}。"
        ),
        (
            "- 最大 contrast 代数误差："
            f"{permutation_summary['max_algebra_abs_error'].max():.3g}。"
        ),
        (
            "- 捕获到 warning 的 population × permutation："
            f"{int((permutation_summary['warning_count'] > 0).sum())}。"
        ),
        (
            "- Level 3 是 population-level exact-permutation evidence；不会把该 population "
            "中所有 gene 自动升级为 confirmed rescue genes。"
        ),
        (
            "- 本阶段未运行 GSEA、GO/KEGG、subtype DE、CellChat、trajectory、SCENIC "
            "或 MRJP1 机制解释。"
        ),
    ]
    (output_root / "DE_STAGE1_6_PERMUTATION_VALIDATION_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def run_de_stage1_6(
    config: dict[str, Any],
    *,
    allow_low_memory: bool = False,
    resume: bool = True,
) -> Path:
    """Run exact label-permutation validation and stop after calibrated outputs."""

    require_compute_resources(config, allow_low_memory=allow_low_memory)
    paths = project_paths(config)
    logger = setup_logging("10_de_stage1_6_permutation", config)
    output_root = paths["results"] / "de_stage1_6"
    figure_root = paths["figures"] / "de_stage1_6"
    output_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    assignments = enumerate_permutation_assignments()
    assignments.to_csv(output_root / "permutation_assignments.tsv", sep="\t", index=False)

    inputs = load_broad_inputs(config)
    counts = inputs["counts"]  # type: ignore[assignment]
    eligibility_path = paths["results"] / "de_stage1" / "broad_population_eligibility.tsv"
    if not eligibility_path.exists():
        raise FileNotFoundError(eligibility_path)
    eligibility = pd.read_csv(eligibility_path, sep="\t")
    primary = eligibility[eligibility["final_de_status"] == "Primary_DE_ready"]
    populations = sorted(
        primary.groupby("population")["contrast"]
        .nunique()
        .loc[lambda value: value == 3]
        .index.astype(str)
    )
    if len(populations) != 7:
        raise ValueError(f"Expected seven Primary_DE_ready populations, found {populations}")

    n_cpus = int(config.get("pseudobulk", {}).get("n_cpus", 4))
    alpha = float(config.get("pseudobulk", {}).get("alpha", ALPHA))
    all_permutation_summaries: list[pd.DataFrame] = []
    all_whole: list[pd.DataFrame] = []
    all_magnitude: list[pd.DataFrame] = []
    all_magnitude_details: list[pd.DataFrame] = []
    observed_rescue: dict[str, pd.DataFrame] = {}

    for population in populations:
        pop_dir = output_root / safe_name(population)
        pop_dir.mkdir(parents=True, exist_ok=True)
        summary_path = pop_dir / "permutation_summary.tsv"
        whole_path = pop_dir / "whole_signature_reversal.tsv"
        magnitude_path = pop_dir / "effect_magnitude_summary.tsv"
        detail_path = pop_dir / "effect_magnitude_genes.tsv.gz"
        observed_path = pop_dir / "observed_rescue_ready_effects.tsv.gz"
        reusable = resume and all(
            path.exists()
            for path in (summary_path, whole_path, magnitude_path, detail_path, observed_path)
        )
        if reusable:
            pop_summary = pd.read_csv(summary_path, sep="\t")
            pop_whole = pd.read_csv(whole_path, sep="\t")
            if (
                pop_summary["permutation_id"].nunique() == EXPECTED_PERMUTATIONS
                and len(pop_summary) == EXPECTED_PERMUTATIONS
                and pop_whole["permutation_id"].nunique() == EXPECTED_PERMUTATIONS
            ):
                logger.info("Permutation checkpoint reused: %s", population)
                all_permutation_summaries.append(pop_summary)
                all_whole.append(pop_whole)
                all_magnitude.append(pd.read_csv(magnitude_path, sep="\t"))
                all_magnitude_details.append(pd.read_csv(detail_path, sep="\t"))
                observed_rescue[population] = pd.read_csv(observed_path, sep="\t")
                continue

        logger.info("Permutation population starting: %s", population)
        pop_counts = counts.loc[population].reindex(ALL_LIBRARIES)  # type: ignore[index]
        if pop_counts.isna().any().any():
            raise ValueError(f"Missing nine-library counts for {population}")
        filtered = prefilter_genes(pop_counts, min_count=10, min_samples=3)
        if filtered.shape[1] == 0:
            raise ValueError(f"No genes pass the unified prefilter for {population}")
        pop_rows: list[dict[str, Any]] = []
        pop_whole_rows: list[pd.DataFrame] = []
        observed: pd.DataFrame | None = None
        for permutation_id, assignment in assignments.groupby(
            "permutation_id", sort=True, observed=True
        ):
            is_observed = bool(assignment["is_observed"].iloc[0])
            logger.info(
                "Permutation fit: %s %s observed=%s", population, permutation_id, is_observed
            )
            wide, warning_lines = _fit_permutation(
                filtered, assignment, alpha=alpha, n_cpus=n_cpus
            )
            rescue = build_rescue_ready_effects(wide)
            algebra_error = np.abs(
                rescue["residual_effect"] - rescue["aging_effect"] - rescue["treatment_effect"]
            )
            whole = whole_signature_metrics(
                rescue,
                population=population,
                permutation_id=str(permutation_id),
                is_observed=is_observed,
            )
            pop_whole_rows.append(whole)
            pop_rows.append(
                summarize_permutation(
                    rescue,
                    population=population,
                    permutation_id=str(permutation_id),
                    is_observed=is_observed,
                    max_algebra_abs_error=float(algebra_error.max()),
                    whole=whole,
                    warning_lines=warning_lines,
                )
            )
            if is_observed:
                observed = rescue.copy()
        if observed is None:
            raise RuntimeError(f"Observed allocation was not run for {population}")
        pop_summary = pd.DataFrame(pop_rows)
        pop_whole = pd.concat(pop_whole_rows, ignore_index=True)
        magnitude, magnitude_detail = effect_magnitude_summary(observed, population=population)
        pop_summary.to_csv(summary_path, sep="\t", index=False)
        pop_whole.to_csv(whole_path, sep="\t", index=False)
        magnitude.to_csv(magnitude_path, sep="\t", index=False)
        magnitude_detail.to_csv(detail_path, sep="\t", index=False, compression="gzip")
        observed.to_csv(observed_path, sep="\t", index=False, compression="gzip")
        all_permutation_summaries.append(pop_summary)
        all_whole.append(pop_whole)
        all_magnitude.append(magnitude)
        all_magnitude_details.append(magnitude_detail)
        observed_rescue[population] = observed
        logger.info("Permutation population complete: %s", population)

    permutation_summary = pd.concat(all_permutation_summaries, ignore_index=True)
    whole_summary = pd.concat(all_whole, ignore_index=True)
    magnitude_summary = pd.concat(all_magnitude, ignore_index=True)
    magnitude_details = pd.concat(all_magnitude_details, ignore_index=True)
    if len(permutation_summary) != len(populations) * EXPECTED_PERMUTATIONS:
        raise RuntimeError("Exact permutation grid is incomplete")
    stage1_5_summary_path = paths["results"] / "de_stage1_5" / "unified_model_summary.tsv"
    if not stage1_5_summary_path.exists():
        raise FileNotFoundError(stage1_5_summary_path)
    stage1_5_summary = pd.read_csv(stage1_5_summary_path, sep="\t")
    observed_audit = validate_observed_against_stage1_5(
        permutation_summary, stage1_5_summary
    )
    observed_audit.to_csv(
        output_root / "observed_vs_stage1_5_concordance.tsv", sep="\t", index=False
    )
    evidence = build_population_evidence(permutation_summary)
    permutation_summary.to_csv(
        output_root / "permutation_rescue_summary.tsv", sep="\t", index=False
    )
    whole_summary.to_csv(
        output_root / "whole_signature_reversal_all_populations.tsv", sep="\t", index=False
    )
    magnitude_summary.to_csv(
        output_root / "effect_magnitude_summary_all_populations.tsv", sep="\t", index=False
    )
    evidence.to_csv(output_root / "population_reversal_evidence.tsv", sep="\t", index=False)

    level3_lookup = evidence.set_index("population")["permutation_supported_signature"].to_dict()
    for population, rescue in observed_rescue.items():
        levels = _add_observed_gene_levels(
            rescue,
            population=population,
            population_level3=bool(level3_lookup[population]),
        )
        levels.to_csv(
            output_root / safe_name(population) / "observed_evidence_levels.tsv.gz",
            sep="\t",
            index=False,
            compression="gzip",
        )

    _plot_metric(
        permutation_summary,
        metric="directional_rescue_fraction",
        ylabel="Directional-rescue fraction",
        output=figure_root / "observed_vs_permutation_directional_rescue_fraction.png",
    )
    _plot_metric(
        permutation_summary,
        metric="FDR_supported_rescue_fraction",
        ylabel="FDR-supported rescue fraction",
        output=figure_root / "observed_vs_permutation_FDR_supported_rescue_fraction.png",
    )
    _plot_metric(
        permutation_summary,
        metric="whole_all_spearman",
        ylabel="Aging–treatment Spearman (all tested genes)",
        output=figure_root / "observed_vs_permutation_aging_treatment_spearman.png",
    )
    _plot_metric(
        permutation_summary,
        metric="whole_all_cosine_similarity",
        ylabel="Aging–treatment cosine similarity (all tested genes)",
        output=figure_root / "observed_vs_permutation_cosine_similarity.png",
    )
    _plot_effect_magnitude(
        magnitude_details, figure_root / "treatment_to_aging_effect_magnitude_distributions.png"
    )
    _report(output_root, evidence, magnitude_summary, permutation_summary)
    logger.info("RESCUE_PERMUTATION_VALIDATION_COMPLETE: %s", output_root)
    print("RESCUE_PERMUTATION_VALIDATION_COMPLETE")
    return output_root
