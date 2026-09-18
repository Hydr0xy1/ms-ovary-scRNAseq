"""Stage 15 frozen external-axis projection and multidimensional state audit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from .project import project_paths, setup_logging


LIBRARIES = ["Y_1", "Y_2", "Y_3", "OC_1", "OC_2", "OC_3", "OT_1", "OT_2", "OT_3"]
FOCUS = ["Granulosa", "Stromal_fibroblast"]


def _read_broad_counts(path: Path) -> dict[str, pd.DataFrame]:
    table = pd.read_csv(path, sep="\t", compression="infer")
    population_column = "population" if "population" in table.columns else "broad_population"
    library_column = "library" if "library" in table.columns else "library_id"
    metadata = {population_column, library_column}
    counts = table.drop(columns=[c for c in table.columns if c in metadata])
    counts = counts.apply(pd.to_numeric, errors="raise")
    result: dict[str, pd.DataFrame] = {}
    for population, sub in table.groupby(population_column, observed=True, sort=False):
        genes = counts.loc[sub.index].copy()
        genes.index = sub[library_column].astype(str).tolist()
        result[str(population)] = genes.reindex(LIBRARIES)
    return result


def _log_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    return np.log2(counts.div(counts.sum(axis=1), axis=0) * 1e6 + 0.5)


def run_external_projection(config: Mapping[str, Any]) -> Path:
    paths = project_paths(dict(config))
    settings = config["deep_dive_stage15"]
    root = paths["root"]
    external_root = root / settings["output_dir"] / "external_gse267729"
    output_root = root / settings["output_dir"] / "external_projection"
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("26_stage15_external_projection", dict(config))
    programs_path = external_root / "GSE267729_external_age_programs.tsv.gz"
    if not programs_path.exists():
        raise FileNotFoundError(programs_path)
    programs = pd.read_csv(programs_path, sep="\t", compression="gzip")
    counts_by_population = _read_broad_counts(root / "results/pseudobulk_ready/broad_counts.tsv.gz")
    definitions: list[pd.DataFrame] = []
    projections: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    top_n = int(settings.get("external_program_top_n_each_direction", 150))
    for population in FOCUS:
        if population not in counts_by_population:
            continue
        internal = _log_cpm(counts_by_population[population])
        for contrast, ext in programs.loc[programs["population"].eq(population)].groupby("contrast", observed=True):
            up = ext.loc[ext["external_age_log2fc"].gt(0)].sort_values(["external_age_log2fc", "gene"], ascending=[False, True]).head(top_n)
            down = ext.loc[ext["external_age_log2fc"].lt(0)].sort_values(["external_age_log2fc", "gene"], ascending=[True, True]).head(top_n)
            selected = pd.concat([up, down], ignore_index=True)
            selected = selected.loc[selected["gene"].isin(internal.columns)].copy()
            if len(selected) < 20:
                logger.warning("Too few projected genes for %s/%s: %s", population, contrast, len(selected))
                continue
            weights = selected.set_index("gene")["external_age_log2fc"].astype(float)
            weights = weights / weights.abs().sum()
            expression = internal[weights.index]
            score = expression.mul(weights, axis=1).sum(axis=1)
            y_centroid = expression.loc[[x for x in LIBRARIES if x.startswith("Y_")]].mean(axis=0)
            scale = expression.loc[[x for x in LIBRARIES if x.startswith(("Y_", "OC_"))]].std(axis=0, ddof=1).replace(0, 1.0)
            distance = np.sqrt(((expression.subtract(y_centroid, axis=1).div(scale, axis=1)) ** 2).mean(axis=1))
            proj = pd.DataFrame({"population": population, "external_contrast": contrast, "library_id": score.index, "group": [x.split("_", 1)[0] for x in score.index], "external_age_axis_score": score.to_numpy(), "distance_to_young": distance.to_numpy(), "n_projected_genes": len(selected)})
            projections.append(proj)
            selected["population"] = population
            selected["external_contrast"] = contrast
            selected["projection_weight"] = selected["gene"].map(weights)
            selected["direction"] = np.where(selected["external_age_log2fc"].gt(0), "external_age_up", "external_age_down")
            definitions.append(selected)
            means = proj.groupby("group", observed=True)[["external_age_axis_score", "distance_to_young"]].mean()
            if set(means.index) >= {"Y", "OC", "OT"}:
                summaries.append({"population": population, "external_contrast": contrast, "n_genes": len(selected), "mean_score_Y": means.loc["Y", "external_age_axis_score"], "mean_score_OC": means.loc["OC", "external_age_axis_score"], "mean_score_OT": means.loc["OT", "external_age_axis_score"], "treatment_minus_aging_axis": means.loc["OT", "external_age_axis_score"] - means.loc["OC", "external_age_axis_score"], "mean_distance_Y": means.loc["Y", "distance_to_young"], "mean_distance_OC": means.loc["OC", "distance_to_young"], "mean_distance_OT": means.loc["OT", "distance_to_young"], "distance_change_OT_vs_OC": means.loc["OT", "distance_to_young"] - means.loc["OC", "distance_to_young"]})
    projection = pd.concat(projections, ignore_index=True) if projections else pd.DataFrame()
    definition = pd.concat(definitions, ignore_index=True) if definitions else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    projection.to_csv(output_root / "external_age_axis_projection.tsv", sep="\t", index=False)
    definition.to_csv(output_root / "external_age_program_definition.tsv", sep="\t", index=False)
    summary.to_csv(output_root / "external_age_projection_summary.tsv", sep="\t", index=False)
    status = {"status": "EXTERNAL_PROJECTION_COMPLETE", "n_projection_rows": len(projection), "n_program_definitions": len(definition), "top_n_each_direction": top_n, "selection_source": "GSE267729 only; internal OT not used to select genes", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "next_stage": "multidimensional_cell_state_audit"}
    (output_root / "CHECKPOINT.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# Stage 15 外部年龄轴投影", "", "年龄程序完全由 GSE267729 的公开样本定义；本项目 OT 未参与基因筛选。", "", summary.to_markdown(index=False) if not summary.empty else "没有足够的可投影结果。", "", "这里的 axis score 表示与外部年龄方向的转录相似度，不等同于生物学年龄，也不直接证明年轻化。"]
    (output_root / "EXTERNAL_PROJECTION_REPORT_CN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("STAGE15_EXTERNAL_PROJECTION_COMPLETE")
    return output_root


def _load_state_modules(path: Path) -> dict[str, list[str]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {str(k): [str(g) for g in v] for k, v in payload.items()}


def run_state_audit(config: Mapping[str, Any]) -> Path:
    paths = project_paths(dict(config))
    settings = config["deep_dive_stage15"]
    root = paths["root"]
    output_root = root / settings["output_dir"] / "state_audit"
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("27_stage15_state_audit", dict(config))
    modules = _load_state_modules(root / settings["state_modules_file"])
    adata = ad.read_h5ad(root / settings["input_object"], backed="r")
    obs = adata.obs
    broad_key, tier_key = settings["broad_key"], settings["tier_key"]
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for module, genes in modules.items():
        present = [g for g in genes if g in adata.var_names]
        audit_rows.append({"module": module, "n_requested": len(genes), "n_present": len(present), "present_genes": ";".join(present)})
        if len(present) < int(settings["state_min_genes"]):
            continue
        idx = adata.var_names.get_indexer(present)
        idx = idx[idx >= 0]
        x = adata.layers[settings["counts_layer"]][:, idx].tocsr()
        totals = np.asarray(adata.obs["total_counts"], dtype=float)
        totals[totals <= 0] = 1.0
        x = x.multiply((1e4 / totals)[:, None])
        x.data = np.log1p(x.data)
        score = np.asarray(x.mean(axis=1)).ravel()
        score_series = pd.Series(score, index=obs.index)
        for population in FOCUS:
            mask = obs[broad_key].astype(str).eq(population) & obs[tier_key].astype(str).eq("Tier1_primary")
            for library, cell_idx in obs.loc[mask].groupby("library_id", observed=True).groups.items():
                values = score_series.loc[cell_idx].to_numpy()
                if len(values) < int(settings["state_min_cells_per_library"]):
                    continue
                rows.append({"module": module, "population": population, "library_id": str(library), "group": str(library).split("_", 1)[0], "n_cells": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)), "p90": float(np.quantile(values, 0.90)), "p95": float(np.quantile(values, 0.95)), "max": float(np.max(values))})
        logger.info("State module complete: %s (%s genes)", module, len(present))
    adata.file.close()
    table = pd.DataFrame(rows)
    effects: list[dict[str, Any]] = []
    if not table.empty:
        for (module, population), sub in table.groupby(["module", "population"], observed=True):
            for metric in ["mean", "median", "p90", "p95"]:
                means = sub.groupby("group", observed=True)[metric].mean()
                if set(means.index) >= {"Y", "OC", "OT"}:
                    effects.append({"module": module, "population": population, "metric": metric, "Y": means["Y"], "OC": means["OC"], "OT": means["OT"], "aging_OC_minus_Y": means["OC"] - means["Y"], "treatment_OT_minus_OC": means["OT"] - means["OC"], "residual_OT_minus_Y": means["OT"] - means["Y"], "directionally_reversed": bool((means["OC"] - means["Y"]) * (means["OT"] - means["OC"]) < 0), "closer_to_young": bool(abs(means["OT"] - means["Y"]) < abs(means["OC"] - means["Y"]))})
    table.to_csv(output_root / "state_scores_library.tsv", sep="\t", index=False)
    pd.DataFrame(effects).to_csv(output_root / "state_effect_summary.tsv", sep="\t", index=False)
    pd.DataFrame(audit_rows).to_csv(output_root / "state_module_gene_audit.tsv", sep="\t", index=False)
    status = {"status": "STATE_AUDIT_COMPLETE", "n_modules": len(modules), "n_score_rows": len(table), "n_effect_rows": len(effects), "score_definition": "cell-level log1p(CPM) module mean summarized by library; raw counts layer read-only", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "next_stage": "cNMF_stability_or_regulatory_followup"}
    (output_root / "CHECKPOINT.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "STATE_AUDIT_REPORT_CN.md").write_text("# Stage 15 多维细胞状态审计\n\n各模块独立计算均值和高分位数；没有合成总衰老分数。正式组间摘要使用library均值，cell-level数值只用于描述分布。\n\n" + (pd.DataFrame(effects).to_markdown(index=False) if effects else "没有可用效应摘要。") + "\n", encoding="utf-8")
    print("STAGE15_STATE_AUDIT_COMPLETE")
    return output_root
