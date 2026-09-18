"""Stage 15 deep-dive project-state and metadata audit.

This module is intentionally read-only with respect to all existing analysis
objects.  It creates a new audit directory and records missing metadata rather
than guessing batch, estrous stage, or pooled-animal identities.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import anndata as ad
import pandas as pd

from .project import project_paths, setup_logging


EXTERNAL_REFERENCES = [
    {
        "accession": "GSE232309",
        "species": "mouse",
        "modality": "scRNA-seq",
        "age_or_state": "3-month young vs 9-month aged",
        "cycle": "not used as a primary stratifier in this audit",
        "availability": "completed_stage6",
        "usable_for": "independent Granulosa/Stromal aging signature",
        "caveat": "different age window and study design; no direct merge",
    },
    {
        "accession": "GSE267729",
        "species": "mouse",
        "modality": "scRNA-seq",
        "age_or_state": "young, peri-estropausal, post-estropause, senescent-cell enriched",
        "cycle": "regular/irregular/acyclic states reported",
        "availability": "GEO_metadata_verified_raw_matrix_not_yet_staged",
        "usable_for": "cycle-aware external age/state reference if processed matrix is recoverable",
        "caveat": "GEO series currently exposes RAW supplementary tar rather than a ready pseudobulk matrix",
    },
    {
        "accession": "SCP1914",
        "species": "mouse",
        "modality": "single-cell ovary atlas",
        "age_or_state": "normal estrous cycle and follicular states",
        "cycle": "cycle/follicle-state reference",
        "availability": "portal_metadata_not_yet_verified",
        "usable_for": "state-aware reference and cycle confounding audit",
        "caveat": "portal access and processed matrix availability must be verified before use",
    },
    {
        "accession": "GSE202601",
        "species": "human",
        "modality": "snRNA-seq + snATAC-seq",
        "age_or_state": "four young vs four reproductively aged human ovaries",
        "cycle": "not assumed comparable to mouse cycle states",
        "availability": "GEO_metadata_verified_processed_snRNA_resource_listed",
        "usable_for": "cross-species program/regulatory support, not direct mouse age effect",
        "caveat": "human nuclei and species differences; use as supportive evidence only",
    },
]


def _missingness(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in table.columns:
            rows.append({"field": column, "present": False, "n_rows": len(table), "n_missing": len(table), "missing_fraction": 1.0})
            continue
        values = table[column].astype("string")
        missing = values.isna() | values.str.strip().isin(["", "unknown", "Unknown", "TODO", "todo"])
        rows.append({"field": column, "present": True, "n_rows": len(table), "n_missing": int(missing.sum()), "missing_fraction": float(missing.mean())})
    return pd.DataFrame(rows)


def _stage_inventory(root: Path) -> pd.DataFrame:
    entries = [
        ("input_object", "results/06_annotation_v2.h5ad"),
        ("stage3", "results/stage3_subtype_localization/COMPLETE.json"),
        ("stage4", "results/stage4_composition_decomposition/COMPLETE.json"),
        ("stage5", "results/stage5_rejuvenation_geometry/COMPLETE.json"),
        ("stage6", "results/stage6_external_validation/COMPLETE.json"),
        ("stage7", "results/stage7_regulatory_activity/COMPLETE.json"),
        ("stage8", "results/stage8_gene_programs/COMPLETE.json"),
        ("stage9", "results/stage9_communication/COMPLETE.json"),
        ("stage10", "results/stage10_candidates/COMPLETE.json"),
        ("stage11", "results/stage11_phenotype_framework/COMPLETE.json"),
        ("stage12", "results/stage12_final_synthesis/COMPLETE.json"),
        ("stage13", "results/publication_stage13/COMPLETE.json"),
        ("stage14", "results/stage14_conventional/COMPLETE.json"),
    ]
    rows: list[dict[str, Any]] = []
    for stage, relative in entries:
        path = root / relative
        row: dict[str, Any] = {"stage": stage, "path": relative, "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
        if path.name == "COMPLETE.json" and path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                row["status"] = payload.get("status", "unknown")
            except (OSError, json.JSONDecodeError):
                row["status"] = "unreadable"
        else:
            row["status"] = "present" if path.exists() else "missing"
        rows.append(row)
    return pd.DataFrame(rows)


def run_stage15_audit(config: Mapping[str, Any]) -> Path:
    paths = project_paths(dict(config))
    settings = config["deep_dive_stage15"]
    root = paths["root"]
    output_root = root / settings["output_dir"] / "status_audit"
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("24_deep_dive_audit", dict(config))

    metadata_path = root / config["project"]["metadata"]
    metadata = pd.read_csv(metadata_path, sep="\t", dtype="string")
    required_metadata = ["library_id", "group", "age_months", "treatment", "dose_mg_kg", "batch", "estrous_stage", "pool_mouse_ids"]
    metadata_missing = _missingness(metadata, required_metadata)

    h5ad_path = root / settings["input_object"]
    if not h5ad_path.exists():
        raise FileNotFoundError(h5ad_path)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs.copy()
    broad_key = settings["broad_key"]
    subtype_key = settings["subtype_key"]
    tier_key = settings["tier_key"]
    required_obs = ["library_id", "group", broad_key, subtype_key, tier_key]
    missing_obs = [column for column in required_obs if column not in obs.columns]
    if missing_obs:
        adata.file.close()
        raise KeyError(f"Missing required obs columns: {missing_obs}")

    merged = metadata.merge(obs["library_id"].value_counts().rename("n_cells").reset_index().rename(columns={"index": "library_id"}), on="library_id", how="outer")
    merged["n_cells"] = merged["n_cells"].fillna(0).astype(int)
    primary_mask = obs[tier_key].astype(str).eq("Tier1_primary")
    primary_counts = obs.loc[primary_mask, "library_id"].value_counts().rename("n_primary_cells")
    merged = merged.merge(primary_counts, left_on="library_id", right_index=True, how="left")
    merged["n_primary_cells"] = merged["n_primary_cells"].fillna(0).astype(int)
    for population in settings["focus_populations"]:
        counts = obs.loc[obs[broad_key].astype(str).eq(population), "library_id"].value_counts().rename(f"n_{population}_cells")
        merged = merged.merge(counts, left_on="library_id", right_index=True, how="left")
        merged[f"n_{population}_cells"] = merged[f"n_{population}_cells"].fillna(0).astype(int)
    merged["primary_analysis_eligible"] = merged["n_primary_cells"].ge(int(settings["min_primary_cells_per_library"]))
    merged["metadata_status"] = "complete" 
    missing_fields = metadata_missing.loc[metadata_missing["missing_fraction"].gt(0), "field"].tolist()
    if missing_fields:
        merged["metadata_status"] = "incomplete: " + ",".join(missing_fields)

    celltype_summary = obs.groupby(["library_id", "group", broad_key, subtype_key, tier_key], observed=True).size().rename("n_cells").reset_index()
    celltype_summary["fraction_within_library"] = celltype_summary["n_cells"] / celltype_summary.groupby("library_id")["n_cells"].transform("sum")

    metadata_missing.to_csv(output_root / "metadata_missingness.tsv", sep="\t", index=False)
    merged.to_csv(output_root / "library_analysis_table.tsv", sep="\t", index=False)
    celltype_summary.to_csv(output_root / "library_celltype_inventory.tsv", sep="\t", index=False)
    _stage_inventory(root).to_csv(output_root / "existing_stage_inventory.tsv", sep="\t", index=False)
    pd.DataFrame(EXTERNAL_REFERENCES).to_csv(output_root / "external_reference_registry.tsv", sep="\t", index=False)
    adata.file.close()

    status = {
        "stage": "deep_dive_stage15",
        "status": "AUDIT_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_object": str(h5ad_path.relative_to(root)),
        "input_shape": [int(adata.n_obs), int(adata.n_vars)],
        "statistical_unit": "library / biological pool",
        "n_libraries": int(metadata["library_id"].nunique()),
        "missing_metadata_fields": missing_fields,
        "existing_stage6_to_14_reusable": True,
        "next_stage": "external_reference_accessibility_and_frozen_age_programs",
    }
    (output_root / "CHECKPOINT.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# 深入挖掘 Stage 15：项目状态与元数据审查",
        "",
        f"审查对象：`{settings['input_object']}`，shape={status['input_shape'][0]} cells × {status['input_shape'][1]} features。",
        "",
        "## 已有结果",
        "Stage 3–14 的结果文件已存在并可复用；其中 Stage 6–9 已分别完成外部 GSE232309 验证、调控活性、NMF 程序和定向通讯。此次不重复这些分析。",
        "",
        "## 元数据限制",
        "`batch`、`estrous_stage` 和 `pool_mouse_ids` 当前仍为 unknown/TODO；不能据此进行周期、批次或动物配对校正，也不能把library自动解释为独立动物。",
        "",
        "## 当前决策",
        "先核查公共年龄参照的可访问性和样本元数据，再冻结外部年龄程序；之后进入组成/状态分解。若外部矩阵不能可靠获得，则保留GSE232309已完成结果，并将新增队列降级为元数据/程序层支持。",
        "",
        "## 输出",
        "- `metadata_missingness.tsv`：元数据缺失审计",
        "- `library_analysis_table.tsv`：library级分析表",
        "- `library_celltype_inventory.tsv`：library×broad×subtype×tier库存",
        "- `existing_stage_inventory.tsv`：已有阶段及完成状态",
        "- `external_reference_registry.tsv`：公共队列可用性登记",
        "- `CHECKPOINT.json`：断点和下一阶段",
    ]
    (output_root / "STAGE15_STATUS_AUDIT_CN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    logger.info("Stage15 status audit complete: %s", output_root)
    print("STAGE15_STATUS_AUDIT_COMPLETE")
    print(f"OUTPUT={output_root.relative_to(root)}")
    print(f"MISSING_METADATA={','.join(missing_fields) if missing_fields else 'none'}")
    return output_root
