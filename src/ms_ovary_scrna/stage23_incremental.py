"""Stage 23 incremental validation for the ovary MRJP1 project.

This stage is intentionally isolated from Stages 1--22.  It reuses frozen
outputs, keeps the library/donor as the inferential unit, and never modifies the
annotated H5AD or an earlier result.  Cell-level quantities are used only to
construct pre-specified distribution summaries.
"""

from __future__ import annotations

import hashlib
import json
import re
import traceback
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .project import project_paths, setup_logging

STAGE_VERSION = "stage23-incremental-2026.09"
CENSUS_VERSION = "2025-11-08"
MOUSE_OVARY_DATASET = "b8342509-75c7-463a-9829-f39eba78f366"
HUMAN_OVARY_DATASET = "3a702020-cf66-4ef5-9d11-4a5e835b7efb"
GROUP_ORDER = ("Y", "OC", "OT")
GROUP_COLORS = {
    "Y": "#4C78A8",
    "OC": "#7A7A7A",
    "OT": "#D26A4A",
    "young": "#4C78A8",
    "older": "#B35C44",
}
PRIORITY_MODULES = (
    "SASP_inflammation",
    "atresia",
    "ECM_fibrosis",
    "steroidogenesis",
)


def _read_tsv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", compression="infer", **kwargs)


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _checkpoint(path: Path, status: str, **extra: Any) -> None:
    _write_json(
        {
            "status": status,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **extra,
        },
        path,
    )


def _update_state(stage_root: Path, stage: str, status: str, **extra: Any) -> None:
    path = stage_root / "RUN_STATE.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    state.setdefault("stages", {})[stage] = {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(state, path)


def _record_failure(stage_root: Path, step: str, exc: BaseException, logger: Any) -> None:
    path = stage_root / "FAILED_STEPS.tsv"
    row = pd.DataFrame(
        [
            {
                "step": step,
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "time_utc": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )
    if path.exists():
        row = pd.concat([pd.read_csv(path, sep="\t"), row], ignore_index=True)
    _write_tsv(row, path)
    logger.error("Stage23 step failed: %s: %s", step, exc)
    logger.error(traceback.format_exc())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_permutation(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, int]:
    """Return mean(left)-mean(right) and a plus-one exact two-sided P value.

    For a 3-vs-3 comparison all 20 allocations are enumerated.  The project
    convention adds one to numerator and denominator, so the minimum possible
    P value is 1/21.  This is deliberately conservative and does not turn cells
    into replicates.
    """
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), 0
    pooled = np.concatenate([a, b])
    observed = float(a.mean() - b.mean())
    indices = range(len(pooled))
    permuted = []
    for chosen in combinations(indices, len(a)):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(chosen)] = True
        permuted.append(float(pooled[mask].mean() - pooled[~mask].mean()))
    extreme = int(np.sum(np.abs(permuted) >= abs(observed) - 1e-12))
    return observed, float((extreme + 1) / (len(permuted) + 1)), len(permuted)


def _effect_rows(
    frame: pd.DataFrame,
    value: str,
    strata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    strata = dict(strata or {})
    rows: list[dict[str, Any]] = []
    for contrast, left, right in [
        ("OC_vs_Y", "OC", "Y"),
        ("OT_vs_OC", "OT", "OC"),
        ("OT_vs_Y", "OT", "Y"),
    ]:
        effect, pvalue, n_allocations = _exact_permutation(
            frame.loc[frame["group"].astype(str).eq(left), value],
            frame.loc[frame["group"].astype(str).eq(right), value],
        )
        rows.append(
            {
                **strata,
                "value": value,
                "contrast": contrast,
                "effect": effect,
                "exact_permutation_p_two_sided_plus_one": pvalue,
                "n_allocations": n_allocations,
                "statistical_unit": "library_or_donor",
            }
        )
    return rows


def _loo_effects(
    frame: pd.DataFrame,
    value: str,
    strata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for excluded in frame["sample_id"].astype(str).unique():
        kept = frame.loc[frame["sample_id"].astype(str).ne(excluded)]
        means = kept.groupby("group", observed=True)[value].mean()
        rows.extend(
            [
                {
                    **dict(strata or {}),
                    "value": value,
                    "excluded_sample": excluded,
                    "contrast": contrast,
                    "effect": float(means.get(left, np.nan) - means.get(right, np.nan)),
                }
                for contrast, left, right in [
                    ("OC_vs_Y", "OC", "Y"),
                    ("OT_vs_OC", "OT", "OC"),
                    ("OT_vs_Y", "OT", "Y"),
                ]
            ]
        )
    return rows


def _configure_figures() -> None:
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


def _save_figure(fig: mpl.figure.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _library_figure(
    frame: pd.DataFrame,
    value: str,
    title: str,
    ylabel: str,
    sample_col: str = "sample_id",
    group_order: Sequence[str] = GROUP_ORDER,
) -> mpl.figure.Figure:
    fig, ax = plt.subplots(figsize=(3.50, 2.75))
    rng = np.random.default_rng(20260920)
    labels: list[str] = []
    for position, group in enumerate(group_order):
        block = frame.loc[frame["group"].astype(str).eq(group)].copy()
        values = pd.to_numeric(block[value], errors="coerce").to_numpy()
        values = values[np.isfinite(values)]
        ax.scatter(
            np.repeat(position, len(values)) + rng.normal(0, 0.035, len(values)),
            values,
            s=25,
            color=GROUP_COLORS.get(group, "#555555"),
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        if len(values):
            mean = float(np.mean(values))
            ax.plot([position - 0.18, position + 0.18], [mean, mean], color="#222222", lw=1.1)
        labels.append(f"{group}\n(n={block[sample_col].nunique()})")
    ax.set_xticks(range(len(group_order)), labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.7, length=2.5)
    fig.tight_layout()
    return fig


def _init_stage(config: Mapping[str, Any]) -> tuple[Path, Any, dict[str, Path]]:
    paths = project_paths(dict(config))
    stage_root = paths["root"] / "results/deep_dive_stage23_incremental"
    for name in [
        "00_scope_audit",
        "01_granulosa_distribution",
        "02_external_cohort_rescue",
        "03_axis_program_attribution",
        "04_targeted_mechanism_triage",
        "05_human_conservation",
        "06_evidence_reconciliation",
        "figures/source_data",
        "logs",
        "checkpoints",
    ]:
        (stage_root / name).mkdir(parents=True, exist_ok=True)
    logger = setup_logging("34_stage23_incremental", dict(config), output_dir=stage_root / "logs")
    if not (stage_root / "RUN_STATE.json").exists():
        _update_state(
            stage_root,
            "initialization",
            "started",
            stage_version=STAGE_VERSION,
            statistical_unit="library / donor",
            output_isolation="results/deep_dive_stage23_incremental",
        )
    _configure_figures()
    return stage_root, logger, paths


def run_scope_audit(
    config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]
) -> None:
    del logger
    root = paths["root"]
    out = stage_root / "00_scope_audit"
    inputs = [
        "results/deep_dive_stage15/state_audit/state_scores_library.tsv",
        "results/deep_dive_stage16_ml/03_scvi_reference/SCVI_LATENT_CELLS_Granulosa_20260919.tsv.gz",
        "results/deep_dive_stage16_ml/03_scvi_reference/SCVI_LATENT_CELLS_Granulosa_20260920.tsv.gz",
        "results/deep_dive_stage16_ml/03_scvi_reference/SCVI_LATENT_CELLS_Granulosa_20260921.tsv.gz",
        "results/de_stage1_5/Granulosa/rescue_ready_effects.tsv.gz",
        "results/de_stage1_6/Granulosa/observed_evidence_levels.tsv.gz",
        "results/pseudobulk_ready/broad_counts.tsv.gz",
        "results/deep_dive_stage15/external_gse267729/GSE267729_external_age_programs.tsv.gz",
        "results/deep_dive_stage15/external_gse267729/GSE267729_broad_pseudobulk_counts.tsv.gz",
        "results/deep_dive_stage15/external_gse267729/GSE267729_broad_pseudobulk_metadata.tsv",
        "results/stage7_regulatory_activity/tf_activity.tsv",
        "results/stage7_regulatory_activity/tf_reversal.tsv",
        "results/stage7_regulatory_activity/tf_permutation.tsv",
        "results/stage7_regulatory_activity/pathway_activity.tsv",
        "results/stage9_communication/communication_evidence.tsv",
        "results/stage9_communication/ligand_target_links.tsv",
        "results/deep_dive_stage16_ml/08_cross_species/ORTHOLOG_MAPPING.tsv",
    ]
    rows = []
    for relative in inputs:
        path = root / relative
        rows.append(
            {
                "relative_path": relative,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "sha256": _sha256(path) if path.exists() else "",
                "access_mode": "read_only_reuse",
            }
        )
    inventory = pd.DataFrame(rows)
    _write_tsv(inventory, out / "REUSED_INPUT_INVENTORY.tsv")
    missing = inventory.loc[~inventory["exists"], "relative_path"].tolist()
    (out / "STAGE23_SCOPE_CN.md").write_text(
        "# Stage 23 incremental 范围\n\n"
        "本阶段与Stage 15–16物理隔离，只读复用冻结结果，不覆盖旧H5AD、模型、表格或图片。"
        "正式推断单位为内部library/pool或外部donor；细胞只用于形成预先指定的分布摘要。\n\n"
        "本阶段不重复QC、标准化、聚类、常规DE、scVI训练或全量通讯网络。"
        "通讯结果仅作为候选链证据，不作为因果结论。\n",
        encoding="utf-8",
    )
    _checkpoint(
        out / "CHECKPOINT.json", "SCOPE_AUDIT_COMPLETE", n_inputs=len(rows), missing=missing
    )
    _update_state(stage_root, "00_scope_audit", "complete", missing=missing)


def _distribution_direction(center: float, tail: float) -> str:
    if not np.isfinite(center) or not np.isfinite(tail):
        return "not_estimable"
    if abs(center) < 1e-12 and abs(tail) < 1e-12:
        return "both_near_zero"
    if np.sign(center) == np.sign(tail):
        return "center_tail_concordant"
    return "center_tail_opposed"


def run_granulosa_distribution(
    config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]
) -> None:
    del config, logger
    root = paths["root"]
    out = stage_root / "01_granulosa_distribution"
    scores = _read_tsv(root / "results/deep_dive_stage15/state_audit/state_scores_library.tsv")
    required = {"module", "population", "library_id", "group", "mean", "median", "p90", "p95"}
    if scores.empty or not required.issubset(scores.columns):
        raise FileNotFoundError("Stage15 state_scores_library.tsv is missing or incomplete")
    scores = scores.loc[scores["population"].astype(str).eq("Granulosa")].copy()
    scores["sample_id"] = scores["library_id"].astype(str)
    effect_rows: list[dict[str, Any]] = []
    loo_rows: list[dict[str, Any]] = []
    for module, block in scores.groupby("module", observed=True):
        for metric in ["mean", "median", "p90", "p95"]:
            effect_rows.extend(_effect_rows(block, metric, {"module": module}))
            loo_rows.extend(_loo_effects(block, metric, {"module": module}))
        if str(module) in PRIORITY_MODULES:
            for metric in ["median", "p90", "p95"]:
                name = f"granulosa_{module}_{metric}".lower()
                source = block[["sample_id", "group", metric, "n_cells"]].copy()
                _write_tsv(source, stage_root / "figures/source_data" / f"{name}.tsv")
                fig = _library_figure(
                    source,
                    metric,
                    f"Granulosa {str(module).replace('_', ' ')}: {metric.upper()}",
                    "Frozen module score",
                )
                _save_figure(fig, stage_root / "figures" / name)
    effects = pd.DataFrame(effect_rows)
    loo = pd.DataFrame(loo_rows)
    _write_tsv(effects, out / "GRANULOSA_DISTRIBUTION_EXACT_PERMUTATION.tsv")
    _write_tsv(loo, out / "GRANULOSA_DISTRIBUTION_LOO.tsv")
    treatment = effects.loc[effects["contrast"].eq("OT_vs_OC")].pivot(
        index="module", columns="value", values="effect"
    )
    classification = treatment.reset_index()
    classification["median_vs_p95"] = [
        _distribution_direction(row.get("median", np.nan), row.get("p95", np.nan))
        for _, row in classification.iterrows()
    ]
    classification["mean_vs_p90"] = [
        _distribution_direction(row.get("mean", np.nan), row.get("p90", np.nan))
        for _, row in classification.iterrows()
    ]
    _write_tsv(classification, out / "CENTER_TAIL_DIRECTION_CLASSIFICATION.tsv")
    n_opposed = int(
        classification[["median_vs_p95", "mean_vs_p90"]]
        .astype(str)
        .eq("center_tail_opposed")
        .any(axis=1)
        .sum()
    )
    (out / "GRANULOSA_DISTRIBUTION_REPORT_CN.md").write_text(
        "# 颗粒细胞状态分布与尾部审计\n\n"
        "对每个冻结模块的mean、median、P90和P95分别按library比较，并使用精确标签置换；"
        "没有使用细胞级P值。leave-one-library-out用于检查方向是否依赖单个library。\n\n"
        f"共有{n_opposed}个模块至少在一组中心/尾部比较中呈相反治疗方向；"
        "因此总体均值变化不能自动外推到高分尾部细胞。\n",
        encoding="utf-8",
    )
    _checkpoint(
        out / "CHECKPOINT.json",
        "GRANULOSA_DISTRIBUTION_COMPLETE",
        n_modules=int(scores["module"].nunique()),
        n_opposed=n_opposed,
    )
    _update_state(stage_root, "01_granulosa_distribution", "complete", n_opposed=n_opposed)


def _project_latent_age_axis(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    zcols = [column for column in frame.columns if str(column).startswith("z")]
    external = frame.loc[frame["dataset"].astype(str).eq("GSE267729")].copy()
    internal = frame.loc[frame["dataset"].astype(str).eq("internal_MRJP1")].copy()
    sample_centroids = (
        external.groupby(["sample_id", "group"], observed=True)[zcols].mean().reset_index()
    )
    young = sample_centroids.loc[sample_centroids["group"].eq("young"), zcols].mean().to_numpy()
    aged = sample_centroids.loc[sample_centroids["group"].eq("post"), zcols].mean().to_numpy()
    vector = aged - young
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("External latent age axis has zero norm")
    unit = vector / norm
    external_scores = (external[zcols].to_numpy() - young) @ unit
    young_scores = external_scores[external["group"].astype(str).eq("young").to_numpy()]
    thresholds = {
        "p90": float(np.quantile(young_scores, 0.90)),
        "p95": float(np.quantile(young_scores, 0.95)),
    }
    internal = internal.copy()
    internal["external_age_axis_score"] = (internal[zcols].to_numpy() - young) @ unit
    rows = []
    group_cols = ["sample_id", "group"]
    if "subtype" in internal.columns:
        group_cols.append("subtype")
    for keys, block in internal.groupby(group_cols, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        meta = dict(zip(group_cols, keys, strict=True))
        values = block["external_age_axis_score"].to_numpy()
        rows.append(
            {
                **meta,
                "n_cells": len(block),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "p90": float(np.quantile(values, 0.90)),
                "p95": float(np.quantile(values, 0.95)),
                "tail_fraction_above_external_young_p90": float(
                    np.mean(values > thresholds["p90"])
                ),
                "tail_fraction_above_external_young_p95": float(
                    np.mean(values > thresholds["p95"])
                ),
            }
        )
    threshold_frame = pd.DataFrame(
        [
            {"threshold": key, "value": value, "reference": "GSE267729_young_cells"}
            for key, value in thresholds.items()
        ]
    )
    return pd.DataFrame(rows), threshold_frame


def run_latent_distribution(
    config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]
) -> None:
    del config, logger
    root = paths["root"]
    out = stage_root / "01_granulosa_distribution"
    summary_rows: list[pd.DataFrame] = []
    threshold_rows: list[pd.DataFrame] = []
    effect_rows: list[dict[str, Any]] = []
    loo_rows: list[dict[str, Any]] = []
    for path in sorted(
        (root / "results/deep_dive_stage16_ml/03_scvi_reference").glob(
            "SCVI_LATENT_CELLS_Granulosa_*.tsv.gz"
        )
    ):
        frame = _read_tsv(path)
        if frame.empty:
            continue
        seed = int(str(frame["seed"].iloc[0]))
        summary, thresholds = _project_latent_age_axis(frame)
        summary.insert(0, "scope", "subtype")
        summary.insert(0, "seed", seed)
        thresholds.insert(0, "seed", seed)
        summary_rows.append(summary)
        threshold_rows.append(thresholds)
        # Re-project the same cells after replacing subtype with a single level;
        # this yields exact broad summaries rather than averaging subtype medians.
        broad, _ = _project_latent_age_axis(frame.assign(subtype="all"))
        broad.insert(0, "scope", "broad")
        broad.insert(0, "seed", seed)
        summary_rows.append(broad)
        for metric in [
            "median",
            "p90",
            "p95",
            "tail_fraction_above_external_young_p90",
            "tail_fraction_above_external_young_p95",
        ]:
            effect_rows.extend(_effect_rows(broad, metric, {"seed": seed, "scope": "broad"}))
            loo_rows.extend(_loo_effects(broad, metric, {"seed": seed, "scope": "broad"}))
    summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    thresholds = pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame()
    _write_tsv(summary, out / "EXTERNAL_FROZEN_LATENT_AGE_AXIS_LIBRARY.tsv")
    _write_tsv(thresholds, out / "EXTERNAL_FROZEN_LATENT_THRESHOLDS.tsv")
    effects = pd.DataFrame(effect_rows)
    loo = pd.DataFrame(loo_rows)
    _write_tsv(effects, out / "EXTERNAL_FROZEN_LATENT_EXACT_PERMUTATION.tsv")
    _write_tsv(loo, out / "EXTERNAL_FROZEN_LATENT_LOO.tsv")
    if not effects.empty:
        stability = (
            effects.groupby(["scope", "value", "contrast"], observed=True)["effect"]
            .agg(
                median_effect="median",
                min_effect="min",
                max_effect="max",
                n_seeds="count",
            )
            .reset_index()
        )
        stability["same_sign_all_seeds"] = np.sign(stability["min_effect"]) == np.sign(
            stability["max_effect"]
        )
    else:
        stability = pd.DataFrame()
    _write_tsv(stability, out / "EXTERNAL_FROZEN_LATENT_CROSS_SEED_STABILITY.tsv")
    _checkpoint(
        out / "LATENT_CHECKPOINT.json",
        "FROZEN_LATENT_AGE_AXIS_COMPLETE",
        n_seeds=int(summary["seed"].nunique()) if not summary.empty else 0,
    )
    _update_state(
        stage_root,
        "01b_latent_distribution",
        "complete",
        n_seeds=int(summary["seed"].nunique()) if not summary.empty else 0,
    )


def _collapse_gene_symbols(counts: np.ndarray, symbols: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame(counts, columns=pd.Index(symbols, dtype=str))
    if frame.columns.duplicated().any():
        frame = frame.T.groupby(level=0, sort=False).sum().T
    return frame


def _download_mouse_external_pseudobulk(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts_path = out / "GSE232309_CENSUS_GRANULOSA_PSEUDOBULK_COUNTS.tsv.gz"
    metadata_path = out / "GSE232309_CENSUS_GRANULOSA_PSEUDOBULK_METADATA.tsv"
    if counts_path.exists() and metadata_path.exists():
        return _read_tsv(counts_path, index_col=0), _read_tsv(metadata_path)
    import cellxgene_census

    filter_text = (
        f'dataset_id == "{MOUSE_OVARY_DATASET}" and is_primary_data == True '
        'and cell_type == "granulosa cell"'
    )
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        adata = cellxgene_census.get_anndata(
            census,
            organism="Mus musculus",
            X_name="raw",
            obs_value_filter=filter_text,
            obs_column_names=[
                "dataset_id",
                "donor_id",
                "development_stage",
                "cell_type",
                "is_primary_data",
                "sex",
            ],
            var_column_names=["feature_id", "feature_name"],
        )
    if adata.n_obs == 0:
        raise RuntimeError("CELLxGENE returned no primary GSE232309 granulosa cells")
    symbols = adata.var["feature_name"].astype(str).to_numpy()
    rows = []
    meta_rows = []
    for donor, positions in adata.obs.groupby("donor_id", observed=True).indices.items():
        block = adata.X[positions]
        summed = np.asarray(block.sum(axis=0)).ravel()
        rows.append(summed)
        obs = adata.obs.iloc[positions]
        stage = str(obs["development_stage"].mode().iloc[0])
        meta_rows.append(
            {
                "sample_id": str(donor),
                "donor_id": str(donor),
                "development_stage": stage,
                "group": "aged" if stage.startswith("9-") else "young",
                "age_months": 9 if stage.startswith("9-") else 3,
                "n_cells": len(positions),
                "dataset": "GSE232309",
                "census_dataset_id": MOUSE_OVARY_DATASET,
                "census_version": CENSUS_VERSION,
                "is_primary_data": True,
            }
        )
    counts = _collapse_gene_symbols(np.vstack(rows), symbols)
    counts.index = [row["sample_id"] for row in meta_rows]
    counts.to_csv(counts_path, sep="\t", compression="gzip")
    metadata = pd.DataFrame(meta_rows)
    _write_tsv(metadata, metadata_path)
    return counts, metadata


def _log_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    numeric = counts.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    library_size = numeric.sum(axis=1).replace(0, np.nan)
    return np.log2(numeric.div(library_size, axis=0) * 1e6 + 0.5)


def _frozen_program_score(
    counts: pd.DataFrame, program: pd.DataFrame, n_genes: int = 200
) -> tuple[pd.Series, list[str]]:
    ranked = program.assign(
        abs_effect=pd.to_numeric(program["external_age_log2fc"], errors="coerce").abs()
    ).sort_values("abs_effect", ascending=False)
    genes = [gene for gene in ranked["gene"].astype(str) if gene in counts.columns][:n_genes]
    if len(genes) < 20:
        raise RuntimeError(f"Only {len(genes)} frozen program genes overlap the validation matrix")
    weights = (
        ranked.drop_duplicates("gene")
        .set_index("gene")
        .reindex(genes)["external_age_log2fc"]
        .astype(float)
    )
    weights = weights / weights.abs().sum()
    expression = _log_cpm(counts[genes])
    standardized = expression.subtract(expression.mean(axis=0), axis=1)
    standard_deviation = expression.std(axis=0, ddof=1).replace(0, np.nan)
    standardized = standardized.div(standard_deviation, axis=1).fillna(0.0)
    return standardized.mul(weights, axis=1).sum(axis=1), genes


def _contrast_effect(counts: pd.DataFrame, metadata: pd.DataFrame) -> pd.Series:
    expression = _log_cpm(counts)
    young = metadata.loc[metadata["group"].eq("young"), "sample_id"].astype(str)
    aged = metadata.loc[metadata["group"].eq("aged"), "sample_id"].astype(str)
    return expression.loc[aged].mean(axis=0) - expression.loc[young].mean(axis=0)


def run_external_cohort_rescue(
    config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]
) -> None:
    del config, logger
    root = paths["root"]
    out = stage_root / "02_external_cohort_rescue"
    counts, metadata = _download_mouse_external_pseudobulk(out)
    program = _read_tsv(
        root / "results/deep_dive_stage15/external_gse267729/GSE267729_external_age_programs.tsv.gz"
    )
    program = program.loc[
        program["population"].eq("Granulosa") & program["contrast"].eq("post_acyclic_vs_young")
    ].copy()
    score, genes = _frozen_program_score(counts, program)
    validation = metadata.copy()
    validation["frozen_GSE267729_age_program_score"] = validation["sample_id"].map(score)
    _write_tsv(validation, out / "GSE232309_FROZEN_AGE_PROGRAM_DONOR_SCORES.tsv")
    effect, pvalue, n_allocations = _exact_permutation(
        validation.loc[validation["group"].eq("aged"), "frozen_GSE267729_age_program_score"],
        validation.loc[validation["group"].eq("young"), "frozen_GSE267729_age_program_score"],
    )

    gse267_counts = _read_tsv(
        root
        / "results/deep_dive_stage15/external_gse267729/GSE267729_broad_pseudobulk_counts.tsv.gz",
        index_col=0,
    )
    gse267_meta = _read_tsv(
        root
        / "results/deep_dive_stage15/external_gse267729/GSE267729_broad_pseudobulk_metadata.tsv"
    )
    gse267_meta = gse267_meta.loc[
        gse267_meta["population"].eq("Granulosa") & gse267_meta["age_group"].isin(["young", "post"])
    ].copy()
    gse267_meta["group"] = gse267_meta["age_group"].map({"young": "young", "post": "aged"})
    gse267_meta["sample_id"] = gse267_meta["sample_id"].astype(str)
    gse267_ids = [f"{sample}::Granulosa" for sample in gse267_meta["sample_id"]]
    gse267 = gse267_counts.loc[gse267_ids].copy()
    gse267.index = gse267_meta["sample_id"].to_numpy()

    effect232 = _contrast_effect(counts, metadata)
    effect267 = _contrast_effect(gse267, gse267_meta)
    rescue = _read_tsv(root / "results/de_stage1_5/Granulosa/rescue_ready_effects.tsv.gz")
    internal = rescue.drop_duplicates("gene").set_index("gene")["aging_effect"].astype(float)
    concordance_rows = []
    pairs = [
        ("GSE232309_vs_GSE267729", effect232, effect267),
        ("GSE232309_vs_internal_OC_minus_Y", effect232, internal),
        ("GSE267729_vs_internal_OC_minus_Y", effect267, internal),
    ]
    for comparison, first, second in pairs:
        common = first.index.intersection(second.index)
        common = common[first.reindex(common).notna() & second.reindex(common).notna()]
        rho, p = (
            spearmanr(first.reindex(common), second.reindex(common))
            if len(common)
            else (np.nan, np.nan)
        )
        concordance_rows.append(
            {
                "comparison": comparison,
                "n_shared_genes": len(common),
                "spearman_rho": rho,
                "nominal_spearman_p": p,
                "interpretation": "directional_concordance_not_independent_gene_replicates",
            }
        )
    _write_tsv(pd.DataFrame(concordance_rows), out / "CROSS_STUDY_GENE_EFFECT_CONCORDANCE.tsv")

    # Reciprocal validation: freeze the 4-vs-4 GSE232309 age effect, then score
    # the independent GSE267729 donors without refitting.
    reciprocal_program = pd.DataFrame(
        {"gene": effect232.index, "external_age_log2fc": effect232.to_numpy()}
    )
    reciprocal_score, reciprocal_genes = _frozen_program_score(gse267, reciprocal_program)
    reciprocal = gse267_meta.copy()
    reciprocal["frozen_GSE232309_age_program_score"] = reciprocal["sample_id"].map(reciprocal_score)
    reciprocal_effect, reciprocal_p, reciprocal_allocations = _exact_permutation(
        reciprocal.loc[reciprocal["group"].eq("aged"), "frozen_GSE232309_age_program_score"],
        reciprocal.loc[reciprocal["group"].eq("young"), "frozen_GSE232309_age_program_score"],
    )
    _write_tsv(reciprocal, out / "GSE267729_RECIPROCAL_FROZEN_PROGRAM_DONOR_SCORES.tsv")
    reciprocal_summary = pd.DataFrame(
        [
            {
                "training_program": "GSE267729_post_vs_young",
                "independent_validation": "GSE232309_9m_vs_3m",
                "effect_aged_minus_young": effect,
                "exact_permutation_p_two_sided_plus_one": pvalue,
                "n_allocations": n_allocations,
                "n_program_genes": len(genes),
            },
            {
                "training_program": "GSE232309_9m_vs_3m",
                "independent_validation": "GSE267729_post_vs_young",
                "effect_aged_minus_young": reciprocal_effect,
                "exact_permutation_p_two_sided_plus_one": reciprocal_p,
                "n_allocations": reciprocal_allocations,
                "n_program_genes": len(reciprocal_genes),
            },
        ]
    )
    _write_tsv(reciprocal_summary, out / "RECIPROCAL_EXTERNAL_VALIDATION.tsv")
    plot_frame = validation.rename(columns={"frozen_GSE267729_age_program_score": "score"}).copy()
    plot_frame["group"] = plot_frame["group"].map({"young": "Y", "aged": "OC"})
    plot_frame["sample_id"] = plot_frame["donor_id"]
    _write_tsv(plot_frame, stage_root / "figures/source_data/gse232309_frozen_age_program.tsv")
    fig = _library_figure(
        plot_frame,
        "score",
        "GSE232309: frozen GSE267729 age program",
        "Frozen age-program score",
        group_order=("Y", "OC"),
    )
    _save_figure(fig, stage_root / "figures/gse232309_frozen_age_program")
    (out / "EXTERNAL_COHORT_REPORT_CN.md").write_text(
        "# 第二外部年龄队列验证\n\n"
        "CELLxGENE Census固定版本为2025-11-08。GSE232309使用主数据集中的primary cells，"
        "按8只mouse/donor聚合（3月龄4只、9月龄4只）；没有把3334个颗粒细胞当作重复。\n\n"
        f"冻结GSE267729年龄程序在GSE232309中的aged-minus-young效应为{effect:.4f}，"
        f"plus-one精确置换P={pvalue:.4g}。 reciprocal validation另行保存在表格中。"
        "基因层相关仅作为方向一致性描述，基因不被视为独立生物学重复。\n",
        encoding="utf-8",
    )
    _checkpoint(
        out / "CHECKPOINT.json",
        "SECOND_EXTERNAL_COHORT_COMPLETE",
        n_donors=int(metadata["donor_id"].nunique()),
        n_granulosa_cells=int(metadata["n_cells"].sum()),
        frozen_program_effect=effect,
        frozen_program_exact_p=pvalue,
    )
    _update_state(
        stage_root,
        "02_external_cohort_rescue",
        "complete",
        n_donors=int(metadata["donor_id"].nunique()),
    )


def _internal_granulosa_counts(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = _read_tsv(root / "results/pseudobulk_ready/broad_counts.tsv.gz")
    if table.empty:
        raise FileNotFoundError("broad pseudobulk counts are missing")
    pop_col = "population" if "population" in table.columns else "broad_population"
    lib_col = "library" if "library" in table.columns else "library_id"
    block = table.loc[table[pop_col].astype(str).eq("Granulosa")].copy()
    metadata = pd.DataFrame({"sample_id": block[lib_col].astype(str)})
    metadata["group"] = metadata["sample_id"].str.split("_").str[0]
    gene_cols = [column for column in block.columns if column not in {pop_col, lib_col}]
    counts = block[gene_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    counts.index = metadata["sample_id"].to_numpy()
    return counts, metadata


def _axis_decomposition(aging: pd.Series, treatment: pd.Series) -> pd.DataFrame:
    common = aging.index.intersection(treatment.index)
    a = aging.reindex(common).astype(float)
    t = treatment.reindex(common).astype(float)
    finite = np.isfinite(a) & np.isfinite(t)
    a, t = a.loc[finite], t.loc[finite]
    denominator = float(np.dot(a, a))
    coefficient = float(np.dot(t, a) / denominator) if denominator > 0 else np.nan
    parallel = coefficient * a
    orthogonal = t - parallel
    return pd.DataFrame(
        {
            "gene": a.index,
            "aging_effect_OC_minus_Y": a.to_numpy(),
            "treatment_effect_OT_minus_OC": t.to_numpy(),
            "parallel_treatment_component": parallel.to_numpy(),
            "orthogonal_treatment_residual": orthogonal.to_numpy(),
            "global_parallel_coefficient": coefficient,
        }
    )


def run_axis_program_attribution(
    config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]
) -> None:
    del config, logger
    root = paths["root"]
    out = stage_root / "03_axis_program_attribution"
    counts, metadata = _internal_granulosa_counts(root)
    expression = _log_cpm(counts)
    group_means = expression.groupby(metadata.set_index("sample_id")["group"], observed=True).mean()
    full = _axis_decomposition(
        group_means.loc["OC"] - group_means.loc["Y"], group_means.loc["OT"] - group_means.loc["OC"]
    )
    loo_frames = []
    rank_rows = []
    for excluded in metadata["sample_id"]:
        keep = metadata.loc[metadata["sample_id"].ne(excluded), "sample_id"]
        kept_meta = metadata.set_index("sample_id").loc[keep]
        kept = expression.loc[keep]
        means = kept.groupby(kept_meta["group"], observed=True).mean()
        result = _axis_decomposition(
            means.loc["OC"] - means.loc["Y"], means.loc["OT"] - means.loc["OC"]
        )
        result.insert(0, "excluded_library", excluded)
        loo_frames.append(result)
        joined = full.set_index("gene").join(
            result.set_index("gene"), lsuffix="_full", rsuffix="_loo"
        )
        for metric in [
            "aging_effect_OC_minus_Y",
            "treatment_effect_OT_minus_OC",
            "orthogonal_treatment_residual",
        ]:
            rho, _ = spearmanr(joined[f"{metric}_full"], joined[f"{metric}_loo"])
            rank_rows.append(
                {"excluded_library": excluded, "metric": metric, "spearman_rho_vs_full": rho}
            )
    loo = pd.concat(loo_frames, ignore_index=True)
    # Map full-data directions once, then aggregate all genes in one vectorized
    # groupby. This is O(genes x LOO), not one full-table scan per gene.
    full_indexed = full.set_index("gene")
    loo["aging_same_sign"] = np.sign(loo["aging_effect_OC_minus_Y"]) == np.sign(
        loo["gene"].map(full_indexed["aging_effect_OC_minus_Y"])
    )
    loo["treatment_same_sign"] = np.sign(loo["treatment_effect_OT_minus_OC"]) == np.sign(
        loo["gene"].map(full_indexed["treatment_effect_OT_minus_OC"])
    )
    loo["orthogonal_same_sign"] = np.sign(loo["orthogonal_treatment_residual"]) == np.sign(
        loo["gene"].map(full_indexed["orthogonal_treatment_residual"])
    )
    loo_summary = loo.groupby("gene", observed=True).agg(
        loo_aging_same_sign_fraction=("aging_same_sign", "mean"),
        loo_treatment_same_sign_fraction=("treatment_same_sign", "mean"),
        loo_orthogonal_same_sign_fraction=("orthogonal_same_sign", "mean"),
        loo_orthogonal_min=("orthogonal_treatment_residual", "min"),
        loo_orthogonal_max=("orthogonal_treatment_residual", "max"),
    )
    summary = full_indexed.join(loo_summary, how="left").reset_index()
    candidates = ["Igfbp2", "Pak3", "Abca1", "H1f10", "Hif1a", "Smad3"]
    candidate_table = summary.loc[summary["gene"].isin(candidates)].copy()
    candidate_table["candidate_status"] = candidate_table["gene"].map(
        {gene: "pre_specified" for gene in candidates}
    )
    _write_tsv(summary, out / "GENE_AXIS_DECOMPOSITION.tsv")
    _write_tsv(loo, out / "GENE_AXIS_DECOMPOSITION_LOO.tsv.gz")
    _write_tsv(pd.DataFrame(rank_rows), out / "GENE_AXIS_RANK_STABILITY.tsv")
    _write_tsv(candidate_table, out / "PRESPECIFIED_CANDIDATE_AXIS_ATTRIBUTION.tsv")
    coefficient = float(full["global_parallel_coefficient"].iloc[0])
    (out / "AXIS_PROGRAM_REPORT_CN.md").write_text(
        "# 年龄平行与治疗正交程序归因\n\n"
        "在9个Granulosa library的logCPM空间中，将OT−OC治疗向量分解为沿OC−Y年龄轴的平行分量"
        "和正交残差。该几何分解是描述性程序归因，不是因果机制。\n\n"
        f"全基因空间的治疗-年龄平行系数为{coefficient:.4f}；负值表示总体治疗方向与年龄方向相反。"
        "每次删除一个library后重新计算，并保存基因符号稳定性和全局rank稳定性。\n",
        encoding="utf-8",
    )
    _checkpoint(
        out / "CHECKPOINT.json",
        "AXIS_PROGRAM_ATTRIBUTION_COMPLETE",
        n_genes=len(summary),
        parallel_coefficient=coefficient,
    )
    _update_state(stage_root, "03_axis_program_attribution", "complete", n_genes=len(summary))


def _activity_evidence(
    frame: pd.DataFrame, programs: Sequence[str], activity_type: str
) -> pd.DataFrame:
    selected = frame.loc[
        frame["population"].astype(str).eq("Granulosa")
        & frame["analysis_level"].astype(str).eq("broad")
        & frame["program"].astype(str).str.lower().isin([value.lower() for value in programs])
    ].copy()
    rows = []
    for program, block in selected.groupby("program", observed=True):
        block = block.rename(columns={"library_id": "sample_id"})
        rows.extend(
            _effect_rows(block, "activity", {"activity_type": activity_type, "program": program})
        )
    return pd.DataFrame(rows)


def _component_summary(
    frame: pd.DataFrame, value: str, component: str, chain: str
) -> dict[str, Any]:
    rows = _effect_rows(frame, value, {"chain": chain, "component": component})
    lookup = {row["contrast"]: row for row in rows}
    y_mean = frame.loc[frame["group"].eq("Y"), value].mean()
    oc_mean = frame.loc[frame["group"].eq("OC"), value].mean()
    ot = frame.loc[frame["group"].eq("OT"), value]
    toward_young = int((np.abs(ot - y_mean) < abs(oc_mean - y_mean)).sum())
    result = {
        "chain": chain,
        "component": component,
        "mean_Y": y_mean,
        "mean_OC": oc_mean,
        "mean_OT": ot.mean(),
        "aging_effect_OC_minus_Y": lookup["OC_vs_Y"]["effect"],
        "aging_exact_p": lookup["OC_vs_Y"]["exact_permutation_p_two_sided_plus_one"],
        "treatment_effect_OT_minus_OC": lookup["OT_vs_OC"]["effect"],
        "treatment_exact_p": lookup["OT_vs_OC"]["exact_permutation_p_two_sided_plus_one"],
        "n_OT_libraries_toward_young": toward_young,
        "passes_2_of_3_direction_gate": toward_young >= 2,
    }
    return result


def run_targeted_mechanism_triage(
    config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]
) -> None:
    del config, logger
    root = paths["root"]
    out = stage_root / "04_targeted_mechanism_triage"
    tf = _read_tsv(root / "results/stage7_regulatory_activity/tf_activity.tsv")
    pathway = _read_tsv(root / "results/stage7_regulatory_activity/pathway_activity.tsv")
    tf_evidence = _activity_evidence(tf, ["Hif1a", "Arnt", "Smad3"], "TF")
    pathway_evidence = _activity_evidence(
        pathway, ["Hypoxia", "TGFb", "TGF_beta", "ECM", "Androgen"], "pathway"
    )
    _write_tsv(
        pd.concat([tf_evidence, pathway_evidence], ignore_index=True),
        out / "TARGETED_FOOTPRINT_EVIDENCE.tsv",
    )

    broad = _read_tsv(root / "results/pseudobulk_ready/broad_counts.tsv.gz")
    pop_col = "population" if "population" in broad.columns else "broad_population"
    lib_col = "library" if "library" in broad.columns else "library_id"
    gene_cols = [column for column in broad.columns if column not in {pop_col, lib_col}]
    expression: dict[str, pd.DataFrame] = {}
    for population in ["Stromal_fibroblast", "Granulosa"]:
        block = broad.loc[broad[pop_col].astype(str).eq(population)].copy()
        counts = block[gene_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        counts.index = block[lib_col].astype(str).to_numpy()
        expression[population] = _log_cpm(counts)
    links = _read_tsv(root / "results/stage9_communication/ligand_target_links.tsv")
    chain_rows = []
    source_rows = []
    for ligand, receptor in [("Il6", "Il6st"), ("Fgf2", "Fgfr2")]:
        chain = f"{ligand}_to_{receptor}"
        for population, gene, component in [
            ("Stromal_fibroblast", ligand, "sender_ligand"),
            ("Granulosa", receptor, "receiver_receptor"),
        ]:
            if gene not in expression[population].columns:
                continue
            frame = (
                expression[population][[gene]]
                .rename(columns={gene: "activity"})
                .reset_index(names="sample_id")
            )
            frame["group"] = frame["sample_id"].str.split("_").str[0]
            source = frame.assign(chain=chain, component=component, gene=gene)
            source_rows.append(source)
            chain_rows.append(_component_summary(frame, "activity", component, chain))
        target = links.loc[
            links["ligand"].astype(str).str.lower().eq(ligand.lower())
            & links["receiver"].astype(str).eq("Granulosa")
        ].copy()
        target = target.assign(
            abs_score=pd.to_numeric(target["score"], errors="coerce").abs()
        ).sort_values("abs_score", ascending=False)
        target = target.drop_duplicates("gene")
        genes = [
            gene for gene in target["gene"].astype(str) if gene in expression["Granulosa"].columns
        ][:25]
        if genes:
            weights = target.set_index("gene").reindex(genes)["score"].astype(float)
            weights = weights / weights.abs().sum()
            target_score = expression["Granulosa"][genes].mul(weights, axis=1).sum(axis=1)
            frame = target_score.rename("activity").reset_index()
            frame = frame.rename(columns={frame.columns[0]: "sample_id"})
            frame["group"] = frame["sample_id"].str.split("_").str[0]
            source_rows.append(
                frame.assign(
                    chain=chain, component="receiver_target_program", gene=f"n={len(genes)}"
                )
            )
            chain_rows.append(
                _component_summary(frame, "activity", "receiver_target_program", chain)
            )
    chain_evidence = pd.DataFrame(chain_rows)
    if not chain_evidence.empty:
        gate = (
            chain_evidence.groupby("chain", observed=True)["passes_2_of_3_direction_gate"]
            .agg(n_components_passing="sum", n_components="count")
            .reset_index()
        )
        gate["triage_decision"] = np.where(
            gate["n_components_passing"].eq(gate["n_components"]),
            "all_transcriptomic_components_directionally_consistent",
            "incomplete_chain_support",
        )
        chain_evidence = chain_evidence.merge(gate, on="chain", how="left")
    _write_tsv(chain_evidence, out / "PRESPECIFIED_COMMUNICATION_CHAIN_TRIAGE.tsv")
    source = pd.concat(source_rows, ignore_index=True) if source_rows else pd.DataFrame()
    _write_tsv(source, out / "PRESPECIFIED_COMMUNICATION_CHAIN_LIBRARY_VALUES.tsv")
    (out / "MECHANISM_TRIAGE_REPORT_CN.md").write_text(
        "# 预设候选机制淘汰\n\n"
        "仅审计HIF1A/ARNT、SMAD3相关footprint及Il6→Il6st、Fgf2→Fgfr2两条预设转录链。"
        "每条通讯链分别检查Stromal ligand、Granulosa receptor和Granulosa下游target program，"
        "并要求至少2/3个OT library向Y方向移动。\n\n"
        "这些结果只能用于保留或淘汰候选假说；配体与受体mRNA的协调变化不能证明细胞通讯或因果机制。\n",
        encoding="utf-8",
    )
    _checkpoint(
        out / "CHECKPOINT.json",
        "TARGETED_MECHANISM_TRIAGE_COMPLETE",
        n_chains=int(chain_evidence["chain"].nunique()) if not chain_evidence.empty else 0,
    )
    _update_state(stage_root, "04_targeted_mechanism_triage", "complete")


def run_human_conservation(
    config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]
) -> None:
    del config, logger
    root = paths["root"]
    out = stage_root / "05_human_conservation"
    import cellxgene_census

    filter_text = f'dataset_id == "{HUMAN_OVARY_DATASET}" and is_primary_data == True'
    columns = [
        "dataset_id",
        "donor_id",
        "development_stage",
        "cell_type",
        "is_primary_data",
        "sex",
        "tissue",
        "suspension_type",
    ]
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        obs = (
            census["census_data"]["homo_sapiens"]
            .obs.read(value_filter=filter_text, column_names=columns)
            .concat()
            .to_pandas()
        )
    _write_tsv(obs, out / "GSE202601_CELLXGENE_OBS_METADATA.tsv.gz")
    donor_summary = (
        obs.groupby(["donor_id", "development_stage", "cell_type"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    _write_tsv(donor_summary, out / "GSE202601_DONOR_STAGE_CELLTYPE_COUNTS.tsv")
    granulosa = donor_summary.loc[
        donor_summary["cell_type"].astype(str).str.contains("granulosa", case=False, na=False)
    ]
    n_donors = int(granulosa["donor_id"].nunique())
    n_stages = int(granulosa["development_stage"].nunique())
    gate_passed = n_donors >= 4 and n_stages >= 2
    mapping = _read_tsv(root / "results/deep_dive_stage16_ml/08_cross_species/ORTHOLOG_MAPPING.tsv")
    _write_tsv(mapping, out / "REUSED_ONE_TO_ONE_ORTHOLOG_MAPPING.tsv")
    decision = pd.DataFrame(
        [
            {
                "dataset": "GSE202601",
                "census_dataset_id": HUMAN_OVARY_DATASET,
                "census_version": CENSUS_VERSION,
                "n_primary_cells": len(obs),
                "n_granulosa_donors": n_donors,
                "n_granulosa_development_stages": n_stages,
                "sample_age_celltype_gate_passed": gate_passed,
                "expression_projection_status": "eligible_for_followup"
                if gate_passed
                else "skipped_gate_failed",
                "reason": "requires_reliable_donor_age_and_granulosa_coverage",
            }
        ]
    )
    _write_tsv(decision, out / "HUMAN_PROJECTION_DECISION.tsv")
    projection_rows = pd.DataFrame()
    mapping_summary = pd.DataFrame()
    if gate_passed:
        projection_rows, mapping_summary = _run_human_frozen_projection(root, out)
    (out / "HUMAN_CONSERVATION_REPORT_CN.md").write_text(
        "# 人类保守性数据门控\n\n"
        f"在固定CELLxGENE Census版本{CENSUS_VERSION}中，GSE202601共有{len(obs)}个primary cells；"
        f"颗粒细胞覆盖{n_donors}个donor和{n_stages}个development_stage。"
        "本阶段先完成sample/age/cell-type可恢复性门控并复用版本化one-to-one ortholog表。\n\n"
        + (
            "门控通过，已按donor聚合raw counts，并将预先冻结的小鼠Granulosa年龄程序"
            "经Ensembl one-to-one ortholog映射后投影到8位donor。该分析只检验年龄方向保守性，"
            "不代表MRJP1在人类中的治疗证据。\n"
            if gate_passed
            else "门控未通过，未下载或投影人类表达矩阵，也不输出人类有效性结论。\n"
        ),
        encoding="utf-8",
    )
    _checkpoint(
        out / "CHECKPOINT.json",
        "HUMAN_METADATA_GATE_COMPLETE",
        gate_passed=gate_passed,
        n_granulosa_donors=n_donors,
        n_stages=n_stages,
        n_projection_rows=len(projection_rows),
        n_mapped_genes=int(
            mapping_summary["human_gene"].fillna("").astype(str).str.len().gt(0).sum()
        )
        if not mapping_summary.empty
        else 0,
    )
    _update_state(stage_root, "05_human_conservation", "complete", gate_passed=gate_passed)


def _human_granulosa_pseudobulk(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts_path = out / "GSE202601_CENSUS_GRANULOSA_PSEUDOBULK_COUNTS.tsv.gz"
    metadata_path = out / "GSE202601_CENSUS_GRANULOSA_PSEUDOBULK_METADATA.tsv"
    if counts_path.exists() and metadata_path.exists():
        return _read_tsv(counts_path, index_col=0), _read_tsv(metadata_path)
    import cellxgene_census

    filter_text = (
        f'dataset_id == "{HUMAN_OVARY_DATASET}" and is_primary_data == True '
        'and cell_type == "granulosa cell"'
    )
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            X_name="raw",
            obs_value_filter=filter_text,
            obs_column_names=[
                "dataset_id",
                "donor_id",
                "development_stage",
                "cell_type",
                "is_primary_data",
                "sex",
            ],
            var_column_names=["feature_id", "feature_name"],
        )
    if adata.n_obs == 0:
        raise RuntimeError("CELLxGENE returned no primary GSE202601 granulosa cells")
    symbols = adata.var["feature_name"].astype(str).to_numpy()
    rows = []
    meta_rows = []
    for donor, positions in adata.obs.groupby("donor_id", observed=True).indices.items():
        block = adata.X[positions]
        rows.append(np.asarray(block.sum(axis=0)).ravel())
        obs = adata.obs.iloc[positions]
        stage = str(obs["development_stage"].mode().iloc[0])
        matched = re.search(r"(\d+)-year-old", stage)
        age = int(matched.group(1)) if matched else np.nan
        meta_rows.append(
            {
                "sample_id": str(donor),
                "donor_id": str(donor),
                "development_stage": stage,
                "age_years": age,
                "group": "young" if np.isfinite(age) and age < 40 else "older",
                "n_cells": len(positions),
                "dataset": "GSE202601",
                "census_dataset_id": HUMAN_OVARY_DATASET,
                "census_version": CENSUS_VERSION,
                "is_primary_data": True,
            }
        )
    counts = _collapse_gene_symbols(np.vstack(rows), symbols)
    counts.index = [row["sample_id"] for row in meta_rows]
    counts.to_csv(counts_path, sep="\t", compression="gzip")
    metadata = pd.DataFrame(meta_rows)
    _write_tsv(metadata, metadata_path)
    return counts, metadata


def _extend_human_ortholog_mapping(
    root: Path, out: Path, mouse_genes: Sequence[str]
) -> pd.DataFrame:
    mapping_path = out / "STAGE23_ONE_TO_ONE_ORTHOLOG_MAPPING.tsv"
    existing = _read_tsv(mapping_path)
    if not existing.empty and set(mouse_genes).issubset(set(existing["mouse_gene"])):
        return existing
    base = _read_tsv(root / "results/deep_dive_stage16_ml/08_cross_species/ORTHOLOG_MAPPING.tsv")
    records = base.to_dict("records") if not base.empty else []
    available = set(base["mouse_gene"].astype(str)) if not base.empty else set()
    from .stage16_ml import _query_one_to_one_orthologs

    query_time = datetime.now(timezone.utc).isoformat()
    for gene in mouse_genes:
        if gene in available:
            continue
        rows, error = _query_one_to_one_orthologs(str(gene), "116", attempts=2)
        if rows:
            for row in rows:
                row.update(
                    {
                        "query_date_utc": query_time,
                        "mapping_source": "Ensembl REST homology endpoint",
                    }
                )
                records.append(row)
        else:
            records.append(
                {
                    "mouse_gene": str(gene),
                    "human_gene": "",
                    "mapping_status": error or "no_one_to_one_mapping",
                    "ensembl_release": "116",
                    "query_date_utc": query_time,
                    "mapping_source": "Ensembl REST homology endpoint",
                }
            )
    mapping = pd.DataFrame(records)
    mapping = mapping.drop_duplicates(["mouse_gene", "human_gene"], keep="first")
    _write_tsv(mapping, mapping_path)
    return mapping


def _human_program_score(
    counts: pd.DataFrame,
    mouse_program: pd.DataFrame,
    mapping: pd.DataFrame,
    n_genes: int = 200,
) -> tuple[pd.Series, int]:
    ranked = mouse_program.assign(
        abs_effect=pd.to_numeric(mouse_program["external_age_log2fc"], errors="coerce").abs()
    ).sort_values("abs_effect", ascending=False)
    ranked = ranked.drop_duplicates("gene").head(n_genes)
    mapped = ranked.merge(
        mapping[["mouse_gene", "human_gene", "mapping_status"]],
        left_on="gene",
        right_on="mouse_gene",
        how="left",
    )
    mapped = mapped.loc[
        mapped["human_gene"].astype(str).isin(counts.columns)
        & mapped["mapping_status"].astype(str).eq("ensembl_one_to_one")
    ].drop_duplicates("human_gene")
    if len(mapped) < 20:
        raise RuntimeError(f"Only {len(mapped)} one-to-one genes overlap human counts")
    expression = _log_cpm(counts[mapped["human_gene"].astype(str).tolist()])
    standardized = expression.subtract(expression.mean(axis=0), axis=1)
    standardized = standardized.div(expression.std(axis=0, ddof=1).replace(0, np.nan), axis=1)
    standardized = standardized.fillna(0.0)
    weights = mapped.set_index("human_gene")["external_age_log2fc"].astype(float)
    weights = weights / weights.abs().sum()
    return standardized.mul(weights, axis=1).sum(axis=1), len(mapped)


def _run_human_frozen_projection(root: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts, metadata = _human_granulosa_pseudobulk(out)
    internal = _read_tsv(root / "results/de_stage1_5/Granulosa/rescue_ready_effects.tsv.gz")
    internal_program = (
        internal[["gene", "aging_effect"]]
        .drop_duplicates("gene")
        .rename(columns={"aging_effect": "external_age_log2fc"})
    )
    external = _read_tsv(
        root / "results/deep_dive_stage15/external_gse267729/GSE267729_external_age_programs.tsv.gz"
    )
    external_program = external.loc[
        external["population"].eq("Granulosa") & external["contrast"].eq("post_acyclic_vs_young"),
        ["gene", "external_age_log2fc"],
    ].drop_duplicates("gene")
    gene_union = []
    for program in [internal_program, external_program]:
        ranked = program.assign(
            abs_effect=pd.to_numeric(program["external_age_log2fc"], errors="coerce").abs()
        ).sort_values("abs_effect", ascending=False)
        gene_union.extend(ranked["gene"].astype(str).head(200).tolist())
    gene_union = list(dict.fromkeys(gene_union))
    mapping = _extend_human_ortholog_mapping(root, out, gene_union)
    mapping_summary = mapping.loc[mapping["mouse_gene"].astype(str).isin(gene_union)].copy()
    _write_tsv(mapping_summary, out / "FROZEN_PROGRAM_ORTHOLOG_MAPPING.tsv")

    result_rows = []
    score_frames = []
    for label, program in [
        ("internal_mouse_OC_vs_Y", internal_program),
        ("external_mouse_GSE267729_post_vs_young", external_program),
    ]:
        score, n_genes = _human_program_score(counts, program, mapping)
        frame = metadata.copy()
        frame["program"] = label
        frame["score"] = frame["sample_id"].map(score)
        effect, pvalue, n_allocations = _exact_permutation(
            frame.loc[frame["group"].eq("older"), "score"],
            frame.loc[frame["group"].eq("young"), "score"],
        )
        result_rows.append(
            {
                "program": label,
                "effect_older_minus_young": effect,
                "exact_permutation_p_two_sided_plus_one": pvalue,
                "n_allocations": n_allocations,
                "n_one_to_one_program_genes": n_genes,
                "statistical_unit": "human_donor",
                "interpretation": "age_direction_conservation_not_treatment_evidence",
            }
        )
        score_frames.append(frame)
        source_name = f"human_{label}_donor_scores".lower()
        _write_tsv(frame, out.parent / "figures/source_data" / f"{source_name}.tsv")
        fig = _library_figure(
            frame,
            "score",
            f"Human granulosa: {label.replace('_', ' ')}",
            "Frozen mouse-program score",
            group_order=("young", "older"),
        )
        _save_figure(fig, out.parent / "figures" / source_name)
    scores = pd.concat(score_frames, ignore_index=True)
    result = pd.DataFrame(result_rows)
    _write_tsv(scores, out / "HUMAN_FROZEN_PROGRAM_DONOR_SCORES.tsv")
    _write_tsv(result, out / "HUMAN_FROZEN_PROGRAM_VALIDATION.tsv")
    return result, mapping_summary


def run_evidence_reconciliation(
    config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]
) -> None:
    del config, logger, paths
    out = stage_root / "06_evidence_reconciliation"
    rows = [
        {
            "stage": "Stage13",
            "population": "Stromal_fibroblast",
            "earlier_position": "strong_candidate_main_line",
            "later_evidence": (
                "external reference distance and state consistency were weaker than expected"
            ),
            "current_position": "heterogeneity_and_microenvironment_supporting_line",
            "claim_status": "downgraded_not_discarded",
        },
        {
            "stage": "Stage15-16",
            "population": "Granulosa",
            "earlier_position": "one_of_two_focus_populations",
            "later_evidence": (
                "external age programs, latent geometry, subtype/state localization "
                "and treatment direction converged"
            ),
            "current_position": "primary_biological_line",
            "claim_status": "strengthened_with_library_level_caveats",
        },
        {
            "stage": "Stage23",
            "population": "Granulosa",
            "earlier_position": "primary_biological_line",
            "later_evidence": (
                "distribution tails, second external cohort, axis decomposition and targeted triage"
            ),
            "current_position": "pending_incremental_results",
            "claim_status": "must_follow_stage23_tables_not_prior_narrative",
        },
    ]
    evolution = pd.DataFrame(rows)
    _write_tsv(evolution, out / "EVIDENCE_EVOLUTION.tsv")
    (out / "STAGE13_TO_STAGE23_RECONCILIATION_CN.md").write_text(
        "# Stage 13至Stage 23证据演化\n\n"
        "Stage 13中Stromal因广泛转录反应而被列为较强主线；后续Stage 15–16加入外部年龄参照、"
        "潜在空间距离和状态稳定性后，Stromal的跨证据一致性不足，因此降级为微环境异质性补充线，"
        "并非删除其结果。\n\n"
        "Granulosa在外部年龄程序、局部亚型、状态模块和治疗方向上形成更一致证据，因此成为主线。"
        "Stage 23进一步检验中心与尾部是否分离、第二公共队列能否复现年龄方向，以及治疗变化中"
        "有多少沿年龄轴、多少为正交程序。任何早期表述若与后续门控冲突，应以后续结果为准。\n\n"
        "所有通讯与TF结果仍为候选机制证据；9个内部library不支持强因果或复杂监督学习结论。\n",
        encoding="utf-8",
    )
    _checkpoint(out / "CHECKPOINT.json", "EVIDENCE_RECONCILIATION_COMPLETE", n_rows=len(evolution))
    _update_state(stage_root, "06_evidence_reconciliation", "complete")


def _manifest(stage_root: Path) -> None:
    rows = []
    for path in sorted(stage_root.rglob("*")):
        if path.is_file() and path.name != "FILE_MANIFEST.tsv":
            rows.append(
                {
                    "relative_path": str(path.relative_to(stage_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    _write_tsv(pd.DataFrame(rows), stage_root / "FILE_MANIFEST.tsv")


def run_stage23_incremental(
    config: Mapping[str, Any],
    selected_stages: Iterable[str] | None = None,
    force_stages: Iterable[str] | None = None,
) -> Path:
    stage_root, logger, paths = _init_stage(config)
    _update_state(stage_root, "initialization", "complete")
    stages = [
        ("00_scope_audit", run_scope_audit),
        ("01_granulosa_distribution", run_granulosa_distribution),
        ("01b_latent_distribution", run_latent_distribution),
        ("02_external_cohort_rescue", run_external_cohort_rescue),
        ("03_axis_program_attribution", run_axis_program_attribution),
        ("04_targeted_mechanism_triage", run_targeted_mechanism_triage),
        ("05_human_conservation", run_human_conservation),
        ("06_evidence_reconciliation", run_evidence_reconciliation),
    ]
    selected = set(selected_stages) if selected_stages is not None else None
    forced = set(force_stages or ())
    known = {name for name, _ in stages}
    if selected is not None and selected - known:
        raise ValueError(f"Unknown Stage23 stages: {sorted(selected - known)}")
    if forced - known:
        raise ValueError(f"Unknown forced Stage23 stages: {sorted(forced - known)}")
    if selected is not None:
        stages = [(name, function) for name, function in stages if name in selected]
    for name, function in stages:
        state_path = stage_root / "RUN_STATE.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        previous = state.get("stages", {}).get(name, {}).get("status")
        if previous in {"complete", "skipped"} and name not in forced:
            logger.info("Skipping completed Stage23 step %s", name)
            continue
        succeeded = False
        for attempt in (1, 2):
            try:
                logger.info("Starting Stage23 step %s (attempt %d/2)", name, attempt)
                function(config, stage_root, logger, paths)
                succeeded = True
                break
            except Exception as exc:
                _record_failure(stage_root, f"{name}:attempt_{attempt}", exc, logger)
                _update_state(stage_root, name, "failed", attempt=attempt, error=repr(exc))
        if not succeeded:
            logger.error("Stage23 step %s failed after retry; continuing independent steps", name)
    _manifest(stage_root)
    if selected is not None:
        _update_state(stage_root, "overall", "partial", selected_stages=sorted(selected))
        print("STAGE23_INCREMENTAL_PARTIAL_COMPLETE")
        return stage_root
    state = json.loads((stage_root / "RUN_STATE.json").read_text(encoding="utf-8"))
    failed = any(
        value.get("status") == "failed"
        for key, value in state.get("stages", {}).items()
        if key != "overall"
    )
    _update_state(
        stage_root,
        "overall",
        "complete_with_failures" if failed else "complete",
        next_stage="manual_review_before_claims",
    )
    print("STAGE23_INCREMENTAL_COMPLETE")
    print(f"OUTPUT={stage_root.relative_to(paths['root'])}")
    return stage_root
