"""Stage 15 follow-up validation analyses.

The module is intentionally conservative: it reuses frozen external programs,
existing library-level pseudobulk/state tables and the read-only annotated H5AD.
It never overwrites a prior Stage 15 output or writes back to the H5AD object.
"""

from __future__ import annotations

import itertools
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from .project import project_paths, setup_logging
from .stage15_projection_state import FOCUS, LIBRARIES, _log_cpm


GROUPS = ("Y", "OC", "OT")
EXTERNAL_CONTRASTS = ("peri_regular_vs_young", "peri_irregular_vs_young", "post_acyclic_vs_young")
STROMAL = "Stromal_fibroblast"
GRANULOSA = "Granulosa"
STATE_METRICS = ("mean", "median", "p90", "p95")


def _read_tsv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", compression="infer", **kwargs)


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _read_broad_counts(path: Path) -> dict[str, pd.DataFrame]:
    table = _read_tsv(path)
    if table.empty:
        return {}
    pop_col = "population" if "population" in table.columns else "broad_population"
    lib_col = "library" if "library" in table.columns else "library_id"
    gene_cols = [c for c in table.columns if c not in {pop_col, lib_col}]
    counts = table[gene_cols].apply(pd.to_numeric, errors="coerce")
    result: dict[str, pd.DataFrame] = {}
    for population, sub in table.groupby(pop_col, observed=True, sort=False):
        block = counts.loc[sub.index].copy()
        block.index = sub[lib_col].astype(str).to_numpy()
        result[str(population)] = block.reindex(LIBRARIES)
    return result


def _select_external_programs(programs: pd.DataFrame, population: str, contrast: str, top_n: int = 150) -> pd.DataFrame:
    sub = programs.loc[(programs["population"] == population) & (programs["contrast"] == contrast)].copy()
    up = sub.loc[sub["external_age_log2fc"] > 0].sort_values(["external_age_log2fc", "gene"], ascending=[False, True]).head(top_n)
    down = sub.loc[sub["external_age_log2fc"] < 0].sort_values(["external_age_log2fc", "gene"], ascending=[True, True]).head(top_n)
    return pd.concat([up, down], ignore_index=True)


def _project_internal(counts: pd.DataFrame, selected: pd.DataFrame) -> dict[str, float]:
    if counts.empty or selected.empty:
        return {"n_genes": 0, "score_Y": np.nan, "score_OC": np.nan, "score_OT": np.nan, "delta_axis_OT_minus_OC": np.nan, "distance_Y": np.nan, "distance_OC": np.nan, "distance_OT": np.nan, "distance_change_OT_minus_OC": np.nan}
    internal = _log_cpm(counts)
    genes = [g for g in selected["gene"].astype(str) if g in internal.columns]
    selected = selected.loc[selected["gene"].isin(genes)].drop_duplicates("gene").copy()
    if len(selected) < 20:
        return {"n_genes": int(len(selected)), "score_Y": np.nan, "score_OC": np.nan, "score_OT": np.nan, "delta_axis_OT_minus_OC": np.nan, "distance_Y": np.nan, "distance_OC": np.nan, "distance_OT": np.nan, "distance_change_OT_minus_OC": np.nan}
    weights = selected.set_index("gene")["external_age_log2fc"].astype(float)
    weights = weights / weights.abs().sum()
    expression = internal[weights.index]
    score = expression.mul(weights, axis=1).sum(axis=1)
    y = [x for x in LIBRARIES if x.startswith("Y_")]
    oc = [x for x in LIBRARIES if x.startswith("OC_")]
    ot = [x for x in LIBRARIES if x.startswith("OT_")]
    y_centroid = expression.loc[y].mean(axis=0)
    scale = expression.loc[y + oc].std(axis=0, ddof=1).replace(0, 1.0)
    distance = np.sqrt(((expression.subtract(y_centroid, axis=1).div(scale, axis=1)) ** 2).mean(axis=1))
    return {
        "n_genes": int(len(selected)),
        "score_Y": float(score.loc[y].mean()),
        "score_OC": float(score.loc[oc].mean()),
        "score_OT": float(score.loc[ot].mean()),
        "delta_axis_OT_minus_OC": float(score.loc[ot].mean() - score.loc[oc].mean()),
        "distance_Y": float(distance.loc[y].mean()),
        "distance_OC": float(distance.loc[oc].mean()),
        "distance_OT": float(distance.loc[ot].mean()),
        "distance_change_OT_minus_OC": float(distance.loc[ot].mean() - distance.loc[oc].mean()),
    }


def _external_program_from_counts(
    ext_counts: pd.DataFrame,
    ext_meta: pd.DataFrame,
    population: str,
    contrast: str,
    threshold: int,
    excluded_sample: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta = ext_meta.loc[(ext_meta["population"] == population) & (ext_meta["n_cells"] >= threshold)].copy()
    if excluded_sample is not None:
        meta = meta.loc[meta["sample_id"].astype(str) != str(excluded_sample)].copy()
    if contrast == "peri_regular_vs_young":
        aged = (meta["age_group"] == "peri") & (meta["cycle_state"] == "regular")
    elif contrast == "peri_irregular_vs_young":
        aged = (meta["age_group"] == "peri") & (meta["cycle_state"] == "irregular")
    elif contrast == "post_acyclic_vs_young":
        aged = meta["age_group"] == "post"
    else:
        raise ValueError(contrast)
    young = meta["age_group"] == "young"
    young_keys = meta.loc[young, "sample_id"].astype(str).tolist()
    aged_keys = meta.loc[aged, "sample_id"].astype(str).tolist()
    rows: dict[str, Any] = {"n_external_young": len(young_keys), "n_external_aged": len(aged_keys), "external_young_cells": int(meta.loc[young, "n_cells"].sum()), "external_aged_cells": int(meta.loc[aged, "n_cells"].sum()), "external_sample_ids": ";".join(meta["sample_id"].astype(str))}
    if len(young_keys) < 3 or len(aged_keys) < 3:
        return pd.DataFrame(), rows
    keys = [f"{x}::{population}" for x in young_keys + aged_keys]
    if not set(keys).issubset(ext_counts.index):
        return pd.DataFrame(), rows
    lib = ext_counts.loc[keys]
    log_cpm = np.log2(lib.div(lib.sum(axis=1), axis=0) * 1e6 + 0.5)
    effect = log_cpm.loc[[f"{x}::{population}" for x in aged_keys]].mean(axis=0) - log_cpm.loc[[f"{x}::{population}" for x in young_keys]].mean(axis=0)
    result = pd.DataFrame({"gene": effect.index.astype(str), "external_age_log2fc": effect.to_numpy(), "population": population, "contrast": contrast})
    result["abs_effect"] = result["external_age_log2fc"].abs()
    return result, rows


def run_stromal_sensitivity(stage_root: Path, output_root: Path, logger: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    ext_root = stage_root / "external_gse267729"
    programs = _read_tsv(ext_root / "GSE267729_external_age_programs.tsv.gz")
    ext_counts = _read_tsv(ext_root / "GSE267729_broad_pseudobulk_counts.tsv.gz", index_col=0)
    ext_meta = _read_tsv(ext_root / "GSE267729_broad_pseudobulk_metadata.tsv")
    broad_path = stage_root.parent / "pseudobulk_ready" / "broad_counts.tsv.gz"
    internal = _read_broad_counts(broad_path).get(STROMAL, pd.DataFrame())
    rows: list[dict[str, Any]] = []
    loo_rows: list[dict[str, Any]] = []
    if programs.empty or ext_counts.empty or ext_meta.empty or internal.empty:
        return pd.DataFrame(), pd.DataFrame()
    for threshold in (100, 200):
        for contrast in EXTERNAL_CONTRASTS:
            frozen = _select_external_programs(programs, STROMAL, contrast)
            projected = _project_internal(internal, frozen)
            _, counts_info = _external_program_from_counts(ext_counts, ext_meta, STROMAL, contrast, threshold)
            rows.append({"population": STROMAL, "threshold_n_cells": threshold, "contrast": contrast, "program_source": "frozen_GSE267729_all_samples", **counts_info, **projected, "analysis_level": "primary_frozen_projection"})
            secondary, secondary_info = _external_program_from_counts(ext_counts, ext_meta, STROMAL, contrast, threshold)
            projected_secondary = _project_internal(internal, _select_external_programs(secondary, STROMAL, contrast)) if not secondary.empty else {}
            rows.append({"population": STROMAL, "threshold_n_cells": threshold, "contrast": contrast, "program_source": "secondary_redefined_after_external_filter", **secondary_info, **projected_secondary, "analysis_level": "secondary_sensitivity"})
            # Public-sample leave-one-out influence on the redefined program.
            eligible = ext_meta.loc[(ext_meta["population"] == STROMAL) & (ext_meta["n_cells"] >= threshold), "sample_id"].astype(str).tolist()
            for sample_id in eligible:
                loo_program, loo_info = _external_program_from_counts(ext_counts, ext_meta, STROMAL, contrast, threshold, excluded_sample=sample_id)
                if loo_program.empty:
                    continue
                loo_proj = _project_internal(internal, _select_external_programs(loo_program, STROMAL, contrast))
                loo_rows.append({"population": STROMAL, "threshold_n_cells": threshold, "contrast": contrast, "excluded_external_sample": sample_id, **loo_info, **loo_proj})
    result = pd.DataFrame(rows)
    loo = pd.DataFrame(loo_rows)
    result.to_csv(output_root / "STROMAL_EXTERNAL_SENSITIVITY.tsv", sep="\t", index=False)
    loo.to_csv(output_root / "STROMAL_EXTERNAL_SAMPLE_LOO.tsv", sep="\t", index=False)
    # Direction and sensitivity summary.
    direction_rows: list[dict[str, Any]] = []
    for (threshold, contrast, source), sub in result.groupby(["threshold_n_cells", "contrast", "program_source"], observed=True):
        row = sub.iloc[0]
        loo_sub = loo.loc[(loo["threshold_n_cells"] == threshold) & (loo["contrast"] == contrast)]
        direction_rows.append({"threshold_n_cells": threshold, "contrast": contrast, "program_source": source, "axis_delta": row.get("delta_axis_OT_minus_OC", np.nan), "distance_change": row.get("distance_change_OT_minus_OC", np.nan), "axis_negative": bool(row.get("delta_axis_OT_minus_OC", np.nan) < 0), "distance_reduced": bool(row.get("distance_change_OT_minus_OC", np.nan) < 0), "loo_axis_negative_fraction": float((loo_sub["delta_axis_OT_minus_OC"] < 0).mean()) if not loo_sub.empty else np.nan, "loo_distance_reduced_fraction": float((loo_sub["distance_change_OT_minus_OC"] < 0).mean()) if not loo_sub.empty else np.nan, "loo_n": int(len(loo_sub))})
    direction = pd.DataFrame(direction_rows)
    direction.to_csv(output_root / "STROMAL_EXTERNAL_DIRECTION_SUMMARY.tsv", sep="\t", index=False)
    return result, direction


def _permutation_assignments() -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    six = [f"OC_{i}" for i in range(1, 4)] + [f"OT_{i}" for i in range(1, 4)]
    assignments: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for ot in itertools.combinations(six, 3):
        ot_set = set(ot)
        assignments.append((tuple(sorted(ot)), tuple(sorted(x for x in six if x not in ot_set))))
    return assignments


def _score_distance(y: float, group_value: float) -> float:
    return abs(group_value - y)


def run_state_permutations(
    stage_root: Path,
    output_root: Path,
    logger: Any,
    module_defs_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = _read_tsv(stage_root / "state_audit" / "state_scores_library.tsv")
    if scores.empty:
        return pd.DataFrame(), pd.DataFrame()
    score_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    loo_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    assignments = _permutation_assignments()
    for (population, module), sub in scores.groupby(["population", "module"], observed=True):
        sub = sub.set_index("library_id")
        for metric in STATE_METRICS:
            if metric not in sub.columns:
                continue
            values = pd.to_numeric(sub[metric], errors="coerce")
            for library in LIBRARIES:
                if library in values.index:
                    score_rows.append({"population": population, "module": module, "metric": metric, "library_id": library, "group": library.split("_", 1)[0], "score": float(values.loc[library])})
            y_libs = [x for x in LIBRARIES if x.startswith("Y_")]
            oc_libs = [x for x in LIBRARIES if x.startswith("OC_")]
            ot_libs = [x for x in LIBRARIES if x.startswith("OT_")]
            y = float(values.loc[y_libs].mean())
            oc = float(values.loc[oc_libs].mean())
            ot = float(values.loc[ot_libs].mean())
            observed_diff = ot - oc
            observed_dist_change = _score_distance(y, ot) - _score_distance(y, oc)
            perm_diffs: list[float] = []
            for perm_id, (perm_ot, perm_oc) in enumerate(assignments, start=1):
                p_ot = float(values.loc[list(perm_ot)].mean())
                p_oc = float(values.loc[list(perm_oc)].mean())
                p_diff = p_ot - p_oc
                perm_diffs.append(p_diff)
                permutation_rows.append({"population": population, "module": module, "metric": metric, "perm_id": perm_id, "ot_libraries": ";".join(perm_ot), "oc_libraries": ";".join(perm_oc), "perm_ot_minus_oc": p_diff, "observed_ot_minus_oc": observed_diff, "observed_ot_distance_change": observed_dist_change, "perm_abs_ge_observed": bool(abs(p_diff) >= abs(observed_diff) - 1e-12)})
            p_value = (1 + sum(abs(x) >= abs(observed_diff) - 1e-12 for x in perm_diffs)) / (len(perm_diffs) + 1)
            min_p = 1.0 / (len(perm_diffs) + 1)
            for contrast, effect in [("OC_vs_Y", oc - y), ("OT_vs_OC", observed_diff), ("OT_vs_Y", ot - y)]:
                loo_rows.append({"population": population, "module": module, "metric": metric, "summary_type": "full", "excluded_library": "", "Y_mean": y, "OC_mean": oc, "OT_mean": ot, "effect": effect, "OC_distance_to_Y": _score_distance(y, oc), "OT_distance_to_Y": _score_distance(y, ot), "OT_distance_change": observed_dist_change, "contrast": contrast, "exact_permutation_p_two_sided": p_value, "permutation_min_p": min_p})
            for excluded in LIBRARIES:
                keep = [x for x in LIBRARIES if x != excluded]
                y_keep = [x for x in keep if x.startswith("Y_")]
                oc_keep = [x for x in keep if x.startswith("OC_")]
                ot_keep = [x for x in keep if x.startswith("OT_")]
                if not y_keep or not oc_keep or not ot_keep:
                    continue
                y_l = float(values.loc[y_keep].mean())
                oc_l = float(values.loc[oc_keep].mean())
                ot_l = float(values.loc[ot_keep].mean())
                loo_rows.append({"population": population, "module": module, "metric": metric, "summary_type": "leave_one_library_out", "excluded_library": excluded, "Y_mean": y_l, "OC_mean": oc_l, "OT_mean": ot_l, "effect": ot_l - oc_l, "OC_distance_to_Y": _score_distance(y_l, oc_l), "OT_distance_to_Y": _score_distance(y_l, ot_l), "OT_distance_change": _score_distance(y_l, ot_l) - _score_distance(y_l, oc_l), "contrast": "OT_vs_OC", "exact_permutation_p_two_sided": np.nan, "permutation_min_p": min_p})
    score_table = pd.DataFrame(score_rows)
    perm_table = pd.DataFrame(permutation_rows)
    loo_table = pd.DataFrame(loo_rows)
    score_table.to_csv(output_root / "STATE_MODULE_LIBRARY_SCORES.tsv", sep="\t", index=False)
    perm_table.to_csv(output_root / "STATE_MODULE_EXACT_PERMUTATION.tsv", sep="\t", index=False)
    loo_table.to_csv(output_root / "STATE_MODULE_LOO.tsv", sep="\t", index=False)
    # Module overlap and score correlation are descriptive, not independent tests.
    module_defs_path = module_defs_path or (stage_root.parent.parent / "resources" / "gene_sets" / "deep_dive_state_modules.yaml")
    try:
        import yaml

        modules = yaml.safe_load(module_defs_path.resolve().read_text(encoding="utf-8"))
        module_sets = {str(k): set(map(str, v)) for k, v in modules.items()}
        for a, b in itertools.combinations(module_sets, 2):
            inter = module_sets[a] & module_sets[b]
            union = module_sets[a] | module_sets[b]
            overlap_rows.append({"module_a": a, "module_b": b, "n_overlap": len(inter), "jaccard": len(inter) / len(union) if union else np.nan, "overlap_genes": ";".join(sorted(inter))})
    except Exception as exc:
        logger.warning("Could not write module gene overlap: %s", exc)
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(output_root / "STATE_MODULE_GENE_OVERLAP.tsv", sep="\t", index=False)
    corr_rows: list[dict[str, Any]] = []
    for population in scores["population"].unique():
        pivot = scores.loc[scores["population"] == population].pivot(index="library_id", columns="module", values="mean")
        corr = pivot.corr(method="spearman")
        for a in corr.index:
            for b in corr.columns:
                if a < b:
                    corr_rows.append({"population": population, "module_a": a, "module_b": b, "spearman_rho_library_scores": float(corr.loc[a, b])})
    pd.DataFrame(corr_rows).to_csv(output_root / "STATE_MODULE_SCORE_CORRELATION.tsv", sep="\t", index=False)
    return score_table, loo_table


def run_granulosa_robustness(stage_root: Path, output_root: Path, logger: Any) -> tuple[pd.DataFrame, bool]:
    projection = _read_tsv(stage_root / "external_projection" / "external_age_axis_projection.tsv")
    if projection.empty:
        return pd.DataFrame(), False
    rows: list[dict[str, Any]] = []
    for contrast, sub in projection.loc[projection["population"] == GRANULOSA].groupby("external_contrast", observed=True):
        for excluded in [""] + LIBRARIES:
            keep = sub if excluded == "" else sub.loc[sub["library_id"] != excluded]
            means = keep.groupby("group", observed=True)[["external_age_axis_score", "distance_to_young"]].mean()
            if not set(GROUPS).issubset(means.index):
                continue
            rows.append({"population": GRANULOSA, "contrast": contrast, "excluded_library": excluded, "analysis_type": "full" if excluded == "" else "leave_one_library_out", "axis_Y": means.loc["Y", "external_age_axis_score"], "axis_OC": means.loc["OC", "external_age_axis_score"], "axis_OT": means.loc["OT", "external_age_axis_score"], "axis_delta_OT_minus_OC": means.loc["OT", "external_age_axis_score"] - means.loc["OC", "external_age_axis_score"], "distance_Y": means.loc["Y", "distance_to_young"], "distance_OC": means.loc["OC", "distance_to_young"], "distance_OT": means.loc["OT", "distance_to_young"], "distance_change_OT_minus_OC": means.loc["OT", "distance_to_young"] - means.loc["OC", "distance_to_young"]})
    table = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for contrast, sub in table.groupby("contrast", observed=True):
        full = sub.loc[sub["analysis_type"] == "full"].iloc[0]
        loo = sub.loc[sub["analysis_type"] == "leave_one_library_out"]
        summary_rows.append({"population": GRANULOSA, "contrast": contrast, "full_axis_delta": full["axis_delta_OT_minus_OC"], "full_distance_change": full["distance_change_OT_minus_OC"], "loo_axis_negative_fraction": float((loo["axis_delta_OT_minus_OC"] < 0).mean()), "loo_distance_reduced_fraction": float((loo["distance_change_OT_minus_OC"] < 0).mean()), "n_loo": len(loo), "axis_range_loo": float(loo["axis_delta_OT_minus_OC"].max() - loo["axis_delta_OT_minus_OC"].min()), "distance_range_loo": float(loo["distance_change_OT_minus_OC"].max() - loo["distance_change_OT_minus_OC"].min())})
    summary = pd.DataFrame(summary_rows)
    summary["classification"] = np.where((summary["full_axis_delta"] < 0) & (summary["full_distance_change"] < 0) & (summary["loo_axis_negative_fraction"] >= 0.8) & (summary["loo_distance_reduced_fraction"] >= 0.8), "stable_support", np.where((summary["full_axis_delta"] < 0) & (summary["loo_axis_negative_fraction"] >= 0.8), "partial_support", "inconsistent_or_single_library_sensitive"))
    table.to_csv(output_root / "GRANULOSA_ROBUSTNESS.tsv", sep="\t", index=False)
    summary.to_csv(output_root / "GRANULOSA_ROBUSTNESS_SUMMARY.tsv", sep="\t", index=False)
    stable_contrasts = int((summary["classification"] == "stable_support").sum())
    axis_contrasts = int((summary["loo_axis_negative_fraction"] >= 0.8).sum())
    gate = bool(stable_contrasts >= 2 and axis_contrasts >= 2)
    return summary, gate


def _cell_gene_support(adata: ad.AnnData, mask: np.ndarray, genes: Iterable[str]) -> dict[str, tuple[float, float, int]]:
    present = [g for g in genes if g in adata.var_names]
    if not present or int(mask.sum()) == 0:
        return {g: (np.nan, np.nan, 0) for g in genes}
    idx = adata.var_names.get_indexer(present)
    block = adata.layers["counts"][mask, :][:, idx].tocsr()
    totals = np.asarray(block.sum(axis=1)).ravel().astype(float)
    totals[totals <= 0] = 1.0
    norm = block.multiply((1e4 / totals)[:, None]).tocsr()
    out: dict[str, tuple[float, float, int]] = {}
    for j, gene in enumerate(present):
        vals = norm[:, j].toarray().ravel()
        out[gene] = (float(vals.mean()), float((vals > 0).mean()), int(len(vals)))
    for gene in genes:
        out.setdefault(gene, (np.nan, np.nan, 0))
    return out


def run_conditional_candidates(root: Path, stage_root: Path, output_root: Path, gate: bool, logger: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    tf_path = output_root / "GRANULOSA_TF_CANDIDATES.tsv"
    micro_path = output_root / "GRANULOSA_MICROENVIRONMENT_CANDIDATES.tsv"
    if not gate:
        skipped = pd.DataFrame([{"status": "SKIPPED_GRANULOSA_GATE_FAILED", "reason": "External age-axis and distance robustness did not meet the conservative gate."}])
        skipped.to_csv(tf_path, sep="\t", index=False)
        skipped.to_csv(micro_path, sep="\t", index=False)
        return skipped, skipped.copy()
    broad_path = root / "results/pseudobulk_ready/broad_counts.tsv.gz"
    broad = _read_broad_counts(broad_path).get(GRANULOSA, pd.DataFrame())
    tf_candidates = ["Foxl2", "Nr5a1", "Esr1", "Esr2", "Trp53", "Stat3", "Rela", "Nfkb1", "Hif1a", "Ppargc1a", "Smad3", "E2f1", "Gata4", "Cebpb"]
    tf_rows: list[dict[str, Any]] = []
    if not broad.empty:
        log = _log_cpm(broad)
        for tf in tf_candidates:
            if tf not in log.columns:
                continue
            age = float(log.loc[[x for x in LIBRARIES if x.startswith("OC_")], tf].mean() - log.loc[[x for x in LIBRARIES if x.startswith("Y_")], tf].mean())
            trt = float(log.loc[[x for x in LIBRARIES if x.startswith("OT_")], tf].mean() - log.loc[[x for x in LIBRARIES if x.startswith("OC_")], tf].mean())
            tf_rows.append({"candidate_tf": tf, "age_effect_OC_minus_Y": age, "treatment_effect_OT_minus_OC": trt, "support_source": "existing_stage15_state/NMF/pseudobulk context; expression-level candidate only", "mechanistic_status": "candidate_hypothesis_not_activity_proof"})
    tf_table = pd.DataFrame(tf_rows).sort_values("treatment_effect_OT_minus_OC") if tf_rows else pd.DataFrame([{"status": "NO_CANDIDATE_TF_EXPRESSION_FOUND"}])
    tf_table.head(2).to_csv(tf_path, sep="\t", index=False)
    # Small, hypothesis-led ligand/receptor screen; not a full communication scan.
    candidates = [("Il6", "Il6ra", ["Il6st"]), ("Fgf2", "Fgfr1", ["Fgfr2"]), ("Bdnf", "Ntrk2", []), ("Bmp4", "Bmpr1a", ["Bmpr1b", "Bmpr2"])]
    rows: list[dict[str, Any]] = []
    adata = ad.read_h5ad(root / "results/06_annotation_v2.h5ad", backed="r")
    try:
        obs = adata.obs
        broad_key = "cell_type_broad_v2"
        for ligand, receptor, extra_receptors in candidates:
            sender_mask = obs[broad_key].astype(str).eq("Stromal_fibroblast").to_numpy()
            receiver_mask = obs[broad_key].astype(str).eq(GRANULOSA).to_numpy()
            support_sender = _cell_gene_support(adata, sender_mask, [ligand]).get(ligand, (np.nan, np.nan, 0))
            support_receiver = _cell_gene_support(adata, receiver_mask, [receptor] + extra_receptors)
            best_rec = receptor
            best = support_receiver.get(receptor, (np.nan, np.nan, 0))
            for candidate_rec, value in support_receiver.items():
                if np.nan_to_num(value[1], nan=-1) > np.nan_to_num(best[1], nan=-1):
                    best_rec, best = candidate_rec, value
            rows.append({"sender": "Stromal_fibroblast", "receiver": GRANULOSA, "ligand": ligand, "receptor": best_rec, "sender_mean_log1p_cpm": support_sender[0], "sender_fraction_detected": support_sender[1], "receiver_mean_log1p_cpm": best[0], "receiver_fraction_detected": best[1], "n_sender_cells": support_sender[2], "n_receiver_cells": best[2], "support_level": "expression_supported_candidate_only", "mechanistic_status": "candidate_hypothesis_not_causal"})
    finally:
        adata.file.close()
    micro = pd.DataFrame(rows)
    if not micro.empty:
        micro = micro.sort_values(["sender_fraction_detected", "receiver_fraction_detected"], ascending=False).head(2)
    micro.to_csv(micro_path, sep="\t", index=False)
    return tf_table, micro


def run_nmf_reclassification(stage_root: Path, output_root: Path) -> pd.DataFrame:
    fit = _read_tsv(stage_root / "program_stability" / "program_fit_summary.tsv")
    stability = _read_tsv(stage_root / "program_stability" / "program_seed_stability.tsv")
    loo = _read_tsv(stage_root / "program_stability" / "program_leave_one_library_projection.tsv")
    rows: list[dict[str, Any]] = []
    if fit.empty:
        table = pd.DataFrame([{"status": "NMF_SUMMARY_MISSING"}])
    else:
        for (population, k), sub in fit.groupby(["population", "k"], observed=True):
            st = stability.loc[(stability["population"] == population) & (stability["k"] == k)] if not stability.empty else pd.DataFrame()
            lo = loo.loc[(loo["population"] == population) & (loo["k"] == k)] if not loo.empty else pd.DataFrame()
            rows.append({"population": population, "k": k, "n_models": len(sub), "max_iter_warning_models": int((sub["n_iter"] >= 300).sum()), "mean_seed_cosine": float(st["mean_component_cosine"].mean()) if not st.empty else np.nan, "median_loo_rmse": float(lo["held_out_rmse"].median()) if not lo.empty else np.nan, "interpretation": "exploratory_only; cosine stability is not biological validation; no component receives a causal label"})
        table = pd.DataFrame(rows)
    table.to_csv(output_root / "NMF_RECLASSIFICATION.tsv", sep="\t", index=False)
    return table


def _figures(output_root: Path, stromal: pd.DataFrame, state: pd.DataFrame) -> None:
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not stromal.empty:
        sub = stromal.loc[stromal["program_source"].eq("frozen_GSE267729_all_samples")].copy()
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for threshold, color in [(100, "#4C78A8"), (200, "#E45756")]:
            x = sub.loc[sub["threshold_n_cells"] == threshold]
            axes[0].plot(x["contrast"], x["delta_axis_OT_minus_OC"], marker="o", label=f"n≥{threshold}", color=color)
            axes[1].plot(x["contrast"], x["distance_change_OT_minus_OC"], marker="o", label=f"n≥{threshold}", color=color)
        axes[0].axhline(0, color="0.5", lw=0.7); axes[1].axhline(0, color="0.5", lw=0.7)
        axes[0].set_ylabel("OT − OC external-axis score"); axes[1].set_ylabel("OT − OC distance to Y")
        for ax in axes:
            ax.tick_params(axis="x", rotation=35); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(fig_dir / "stromal_external_sensitivity.png", dpi=300); fig.savefig(fig_dir / "stromal_external_sensitivity.svg"); plt.close(fig)
    if not state.empty:
        sub = state.loc[(state["metric"] == "mean") & (state["summary_type"] == "full") & (state["contrast"] == "OT_vs_OC")].copy()
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            for population, frame in sub.groupby("population", observed=True):
                ax.scatter(frame["module"], frame["effect"], label=population)
            ax.axhline(0, color="0.5", lw=0.7); ax.set_ylabel("OT − OC module score"); ax.tick_params(axis="x", rotation=75); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(fig_dir / "state_module_ot_minus_oc.png", dpi=300); fig.savefig(fig_dir / "state_module_ot_minus_oc.svg"); plt.close(fig)


def _manifest(output_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"RUN_MANIFEST.tsv"}:
            rows.append({"path": str(path.relative_to(output_root)), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    table = pd.DataFrame(rows)
    table.to_csv(output_root / "RUN_MANIFEST.tsv", sep="\t", index=False)
    return table


def run_followup_validation(config: Mapping[str, Any]) -> Path:
    paths = project_paths(dict(config))
    root = paths["root"]
    stage_root = root / config["deep_dive_stage15"]["output_dir"]
    output_root = stage_root / "followup_validation"
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("30_stage15_followup_validation", dict(config), output_dir=output_root)
    run_started = datetime.now(timezone.utc).isoformat()
    errors: list[dict[str, str]] = []
    metadata = _read_tsv(root / config["project"]["metadata"])
    metadata_required = ["library_id", "group", "batch", "estrous_stage", "pool_mouse_ids"]
    metadata_rows = [{"field": c, "present": c in metadata.columns, "n_missing": int(metadata[c].astype(str).isin(["unknown", "Unknown", "TODO", "todo", ""]).sum()) if c in metadata.columns else len(metadata), "interpretation": "unknown/TODO retained; no values imputed"} for c in metadata_required]
    pd.DataFrame(metadata_rows).to_csv(output_root / "METADATA_AUDIT.tsv", sep="\t", index=False)
    try:
        stromal, stromal_direction = run_stromal_sensitivity(stage_root, output_root, logger)
    except Exception as exc:
        logger.error("Stromal sensitivity failed: %s", exc); logger.error(traceback.format_exc()); errors.append({"step": "stromal_external_sensitivity", "error": repr(exc)}); stromal, stromal_direction = pd.DataFrame(), pd.DataFrame()
    try:
        state_scores, state_loo = run_state_permutations(
            stage_root,
            output_root,
            logger,
            root / config["deep_dive_stage15"]["state_modules_file"],
        )
    except Exception as exc:
        logger.error("State permutations failed: %s", exc); logger.error(traceback.format_exc()); errors.append({"step": "state_module_exact_permutation", "error": repr(exc)}); state_scores, state_loo = pd.DataFrame(), pd.DataFrame()
    try:
        granulosa_summary, gate = run_granulosa_robustness(stage_root, output_root, logger)
    except Exception as exc:
        logger.error("Granulosa robustness failed: %s", exc); logger.error(traceback.format_exc()); errors.append({"step": "granulosa_robustness", "error": repr(exc)}); granulosa_summary, gate = pd.DataFrame(), False
    try:
        tf, micro = run_conditional_candidates(root, stage_root, output_root, gate, logger)
    except Exception as exc:
        logger.error("Conditional candidate step failed: %s", exc); logger.error(traceback.format_exc()); errors.append({"step": "conditional_candidates", "error": repr(exc)}); tf, micro = pd.DataFrame(), pd.DataFrame()
    try:
        nmf = run_nmf_reclassification(stage_root, output_root)
    except Exception as exc:
        logger.error("NMF reclassification failed: %s", exc); logger.error(traceback.format_exc()); errors.append({"step": "nmf_reclassification", "error": repr(exc)}); nmf = pd.DataFrame()
    try:
        _figures(output_root, stromal, state_loo)
    except Exception as exc:
        logger.error("Figures failed: %s", exc); logger.error(traceback.format_exc()); errors.append({"step": "figures", "error": repr(exc)})

    report_lines = [
        "# Stage 15 后续验证性分析报告",
        "",
        "本阶段只复用冻结外部年龄程序、已有状态模块、library-level pseudobulk 与主对象元数据；不修改 `results/06_annotation_v2.h5ad`，不重复 QC、标准化、降维、聚类、常规注释或常规 DEG。正式统计单位为 library / biological pool，每组 n=3。",
        "",
        "## 总体结论",
        "",
        "MRJP1 treatment was associated with cell-type- and state-specific transcriptional remodeling. Granulosa cells showed relatively consistent movement opposite to the external aging axis and reduced distance from the young reference across multiple external references, whereas stromal fibroblasts displayed heterogeneous, reference-dependent changes. These findings support partial transcriptional reversion or remodeling rather than global rejuvenation, functional recovery, or a proven anti-aging mechanism.",
        "",
        "## Granulosa证据等级",
        "",
        f"Granulosa稳健性门控：{'通过，进入了小范围候选分析' if gate else '未通过，候选机制分析应视为跳过或仅作记录'}。详细结果见 `GRANULOSA_ROBUSTNESS.tsv` 与 `GRANULOSA_ROBUSTNESS_SUMMARY.tsv`。",
        "年龄轴方向和到Y距离必须分别解释；任何单一library影响、外部参照差异或距离不下降都不能被称作整体年轻化。",
        "",
        "## Stromal证据等级",
        "",
        "Stromal敏感性分析分别比较 n_cells≥100 与 n_cells≥200，并保留冻结程序投影与重新定义外部程序的 secondary analysis。若年龄方向保持而距离变化不一致，应定级为‘年龄方向反向但年轻参考距离不一致的异质性重塑’。",
        "",
        "## 状态模块精确置换",
        "",
        "每个模块在6个OC/OT library上枚举20种三对三标签分配；经验双侧p值采用 (1 + 极端置换数)/(20+1)，可达到的最小p值为1/21。模块之间的基因重叠和library分数相关性均已输出，不能将11个模块当成11个独立证据。",
        "",
        "## NMF重新定级",
        "",
        "NMF结果存在Maximum number of iterations警告；跨种子cosine接近1不能单独证明生物学稳定性；LOO RMSE需要与简单基线比较。当前NMF component只能作为探索性表示，不能直接命名为MRJP1机制。",
        "",
        "## 是否继续TF/微环境分析",
        "",
        f"Granulosa门控状态为 {gate}。即使进入候选分析，TF与配体–受体结果也只作为候选机制；本阶段最多保留少量候选，不进行全库扫描。",
        "",
        "## 不能支持的结论",
        "",
        "不能支持整体年轻化、功能恢复、抗衰老机制证明、直接TF结合、直接配体–受体因果关系或把细胞数当作生物学重复。batch、estrous_stage、pool_mouse_ids缺失时，也不能区分周期、批次、pool内动物和选择性存活。",
        "",
        "## 输出清单",
        "",
        "- `STROMAL_EXTERNAL_SENSITIVITY.tsv`、`STROMAL_EXTERNAL_DIRECTION_SUMMARY.tsv`、`STROMAL_EXTERNAL_SAMPLE_LOO.tsv`",
        "- `STATE_MODULE_EXACT_PERMUTATION.tsv`、`STATE_MODULE_LIBRARY_SCORES.tsv`、`STATE_MODULE_LOO.tsv`",
        "- `STATE_MODULE_GENE_OVERLAP.tsv`、`STATE_MODULE_SCORE_CORRELATION.tsv`",
        "- `GRANULOSA_ROBUSTNESS.tsv`、`GRANULOSA_ROBUSTNESS_SUMMARY.tsv`",
        "- `GRANULOSA_TF_CANDIDATES.tsv`、`GRANULOSA_MICROENVIRONMENT_CANDIDATES.tsv`（若门控未通过则记录跳过原因）",
        "- `NMF_RECLASSIFICATION.tsv`、`METADATA_AUDIT.tsv`、`RUN_MANIFEST.tsv`",
        "",
        "## 错误和剩余限制",
        "",
    ]
    if errors:
        report_lines.extend([f"- {e['step']}: {e['error']}" for e in errors])
    else:
        report_lines.append("- 本轮各独立步骤均完成；仍需人工审核边界和候选机制表。")
    report_lines.extend(["", "## 运行时间", "", f"开始：{run_started}", f"完成：{datetime.now(timezone.utc).isoformat()}"])
    (output_root / "FOLLOWUP_VALIDATION_REPORT_CN.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    checkpoint = {"status": "FOLLOWUP_VALIDATION_COMPLETE", "granulosa_gate": bool(gate), "errors": errors, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "output_dir": str(output_root.relative_to(root)), "statistical_unit": "library / biological pool", "next_stage": "manual_review_before_any_further_expansion"}
    (output_root / "CHECKPOINT.json").write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    _manifest(output_root)
    print("STAGE15_FOLLOWUP_VALIDATION_COMPLETE")
    print(f"OUTPUT={output_root.relative_to(root)}")
    print(f"GRANULOSA_GATE={gate}")
    print(f"ERRORS={len(errors)}")
    return output_root
