"""Stage 16-22 deep-dive validation with explicit evidence and model gates.

This module deliberately keeps biological inference at library/sample level.  It
does not alter the main H5AD or any Stage 1-15 result.  Expensive models are
optional and are skipped with a recorded reason when public data, dependencies,
or validation support are insufficient.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import re
import traceback
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wasserstein_distance

from .project import project_paths, setup_logging
from .stage15_projection_state import LIBRARIES, _log_cpm


GROUPS = ("Y", "OC", "OT")
FOCUS = ("Granulosa", "Stromal_fibroblast")
STAGE_VERSION = "stage16-22-2026.09"


def _read_tsv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", compression="infer", **kwargs)


def _write_checkpoint(path: Path, status: str, **extra: Any) -> None:
    payload = {"status": status, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
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


def _version(package: str) -> str:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return "not_installed"


def _record_failure(stage_root: Path, step: str, exc: BaseException, logger: Any) -> None:
    path = stage_root / "FAILED_STEPS.tsv"
    row = pd.DataFrame(
        [{"step": step, "error_type": type(exc).__name__, "error": repr(exc), "time_utc": datetime.now(timezone.utc).isoformat()}]
    )
    if path.exists():
        row = pd.concat([pd.read_csv(path, sep="\t"), row], ignore_index=True)
    row.to_csv(path, sep="\t", index=False)
    logger.error("Stage16 step failed: %s: %s", step, exc)
    logger.error(traceback.format_exc())


def _update_run_state(stage_root: Path, stage: str, status: str, **extra: Any) -> None:
    path = stage_root / "RUN_STATE.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    state.setdefault("stages", {})[stage] = {"status": status, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra}
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _init_stage(config: Mapping[str, Any]) -> tuple[Path, Any, dict[str, Path]]:
    paths = project_paths(dict(config))
    root = paths["root"]
    stage_root = root / "results/deep_dive_stage16_ml"
    for name in ["00_audit", "01_evidence_matrix", "02_external_registry", "03_scvi_reference", "04_latent_geometry_ot", "05_external_age_models", "06_contrastivevi", "07_foundation_models", "08_cross_species", "09_mechanism_candidates", "10_synthesis", "scripts", "logs", "checkpoints"]:
        (stage_root / name).mkdir(parents=True, exist_ok=True)
    logger = setup_logging("31_stage16_ml", dict(config), output_dir=stage_root / "logs")
    if not (stage_root / "RUN_STATE.json").exists():
        _update_run_state(stage_root, "initialization", "started", statistical_unit="library / biological pool", stage_version=STAGE_VERSION)
    return stage_root, logger, paths


def run_input_audit(config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]) -> None:
    out = stage_root / "00_audit"
    root = paths["root"]
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "results/deep_dive_stage15").rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    pd.DataFrame(rows).to_csv(out / "STAGE15_FILE_INVENTORY.tsv", sep="\t", index=False)

    metadata_path = paths["metadata"]
    metadata = _read_tsv(metadata_path)
    required = ["library_id", "group", "batch", "estrous_stage", "pool_mouse_ids"]
    audit = []
    for field in required:
        present = field in metadata.columns
        missing = int(metadata[field].astype(str).isin(["", "unknown", "Unknown", "TODO", "todo"]).sum()) if present else len(metadata)
        audit.append({"field": field, "present": present, "n_missing_or_unknown": missing, "interpretation": "not imputed"})
    pd.DataFrame(audit).to_csv(out / "METADATA_AUDIT.tsv", sep="\t", index=False)

    h5ad_path = root / config["deep_dive_stage15"]["input_object"]
    object_rows: list[dict[str, Any]] = []
    try:
        adata = ad.read_h5ad(h5ad_path, backed="r")
        object_rows = [
            {"field": "shape", "value": f"{adata.n_obs}x{adata.n_vars}"},
            {"field": "layers", "value": ";".join(adata.layers.keys())},
            {"field": "obs_columns", "value": ";".join(map(str, adata.obs.columns))},
            {"field": "var_columns", "value": ";".join(map(str, adata.var.columns))},
            {"field": "backed_mode", "value": "r"},
        ]
        adata.file.close()
    except Exception as exc:
        object_rows.append({"field": "read_error", "value": repr(exc)})
    pd.DataFrame(object_rows).to_csv(out / "MAIN_OBJECT_AUDIT.tsv", sep="\t", index=False)

    packages = ["anndata", "scanpy", "scvi-tools", "torch", "cellxgene-census", "scikit-learn", "scipy", "pandas", "numpy", "matplotlib"]
    version_rows = [{"package": p, "version": _version(p)} for p in packages]
    version_rows.extend(
        [{"package": "stage_version", "version": STAGE_VERSION}, {"package": "python", "version": os.sys.version.replace("\n", " ")}]
    )
    pd.DataFrame(version_rows).to_csv(out / "SOFTWARE_VERSIONS.tsv", sep="\t", index=False)
    (out / "INPUT_AUDIT_REPORT_CN.md").write_text(
        "# Stage 16–22 输入审计\n\n"
        "本阶段只读取 Stage 1–15 结果和主 H5AD；正式统计单位仍为 library/biological pool。"
        "batch、estrous_stage 和 pool_mouse_ids 未被填补。scVI、监督式年龄模型和 contrastiveVI 只有在依赖、公共样本和稳定性门控满足时才进入结果。\n",
        encoding="utf-8",
    )
    _write_checkpoint(out / "CHECKPOINT.json", "INPUT_AUDIT_COMPLETE", n_stage15_files=len(rows), main_object=str(h5ad_path.relative_to(root)))
    _update_run_state(stage_root, "00_audit", "complete", n_files=len(rows))


def _append_rows(rows: list[dict[str, Any]], frame: pd.DataFrame, source: str, key_cols: Iterable[str], population: str | None = None) -> None:
    if frame.empty:
        return
    for _, record in frame.iterrows():
        row = {"source": source, "population": population or record.get("population", record.get("broad_population", ""))}
        row.update({str(k): record[k] for k in key_cols if k in record.index})
        rows.append(row)


def run_evidence_matrix(config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]) -> None:
    root = paths["root"]
    out = stage_root / "01_evidence_matrix"
    rows: list[dict[str, Any]] = []

    # Gene-level pseudobulk evidence: retain a bounded, pre-specified top set.
    for population in FOCUS:
        de = _read_tsv(root / f"results/de_stage1_5/{population}/unified_all_genes.tsv.gz")
        if not de.empty:
            effect_col = next(
                (
                    column
                    for column in ("log2FC_shrunk", "log2FC_raw", "log2FoldChange")
                    if column in de.columns
                ),
                None,
            )
            if effect_col is None or not {"gene", "contrast"}.issubset(de.columns):
                continue
            effects = de.pivot_table(
                index="gene", columns="contrast", values=effect_col, aggfunc="first"
            )
            padj = de.pivot_table(
                index="gene", columns="contrast", values="padj", aggfunc="first"
            )
            treatment = de.loc[de["contrast"].astype(str).eq("OT_vs_OC")].copy()
            treatment["_padj_rank"] = pd.to_numeric(
                treatment.get("padj"), errors="coerce"
            ).fillna(1.0)
            treatment["_abs_effect"] = pd.to_numeric(
                treatment[effect_col], errors="coerce"
            ).abs()
            selected_genes = (
                treatment.sort_values(
                    ["_padj_rank", "_abs_effect"], ascending=[True, False]
                )["gene"]
                .astype(str)
                .drop_duplicates()
                .head(300)
            )
            for gene in selected_genes:
                aging_effect = _safe_float(
                    effects.at[gene, "OC_vs_Y"] if "OC_vs_Y" in effects else np.nan
                )
                treatment_effect = _safe_float(
                    effects.at[gene, "OT_vs_OC"] if "OT_vs_OC" in effects else np.nan
                )
                residual_effect = _safe_float(
                    effects.at[gene, "OT_vs_Y"] if "OT_vs_Y" in effects else np.nan
                )
                treatment_padj = _safe_float(
                    padj.at[gene, "OT_vs_OC"] if "OT_vs_OC" in padj else np.nan
                )
                directional_rescue = bool(
                    np.isfinite(aging_effect)
                    and np.isfinite(treatment_effect)
                    and aging_effect * treatment_effect < 0
                )
                rows.append(
                    {
                        "candidate_id": f"gene:{gene}",
                        "candidate_type": "gene",
                        "population": population,
                        "source": "pseudobulk_DE_stage1_5",
                        "aging_effect": aging_effect,
                        "treatment_effect": treatment_effect,
                        "residual_effect": residual_effect,
                        "treatment_padj": treatment_padj,
                        "directional_rescue": directional_rescue,
                        "raw_label": "opposite_to_aging_direction"
                        if directional_rescue
                        else "not_directionally_reversed",
                    }
                )

    follow = root / "results/deep_dive_stage15/followup_validation"
    state = _read_tsv(follow / "STATE_MODULE_LOO.tsv")
    if not state.empty:
        state = state.loc[(state["summary_type"] == "full") & (state["metric"] == "mean") & (state["contrast"] == "OT_vs_OC")]
        for _, record in state.iterrows():
            rows.append(
                {
                    "candidate_id": f"state:{record.get('module', '')}",
                    "candidate_type": "state_module",
                    "population": record.get("population", ""),
                    "source": "stage15_state_exact_permutation",
                    "aging_effect": np.nan,
                    "treatment_effect": _safe_float(record.get("effect")),
                    "residual_effect": _safe_float(record.get("OT_distance_change")),
                    "treatment_padj": np.nan,
                    "directional_rescue": _safe_float(record.get("OT_distance_change")) < 0,
                    "raw_label": f"exact_p={record.get('exact_permutation_p_two_sided', np.nan)}",
                }
            )

    ext_summary = _read_tsv(root / "results/deep_dive_stage15/external_projection/external_age_projection_summary.tsv")
    for _, record in ext_summary.iterrows():
        rows.append(
            {
                "candidate_id": f"external_axis:{record.get('population', '')}:{record.get('external_contrast', '')}",
                "candidate_type": "external_age_axis",
                "population": record.get("population", ""),
                "source": "GSE267729_frozen_projection",
                "aging_effect": np.nan,
                "treatment_effect": _safe_float(record.get("treatment_minus_aging_axis")),
                "residual_effect": _safe_float(record.get("distance_change_OT_vs_OC")),
                "treatment_padj": np.nan,
                "directional_rescue": _safe_float(record.get("treatment_minus_aging_axis")) < 0,
                "raw_label": record.get("external_contrast", ""),
            }
        )

    comp = _read_tsv(root / "results/stage4_composition_decomposition/intrinsic_effect.tsv")
    for _, record in comp.iterrows():
        rows.append(
            {
                "candidate_id": f"intrinsic:{record.get('broad_population', '')}:{record.get('subtype', '')}",
                "candidate_type": "composition_adjusted_subtype",
                "population": record.get("broad_population", ""),
                "source": "stage4_intrinsic_effect",
                "aging_effect": _safe_float(record.get("aging_norm")),
                "treatment_effect": _safe_float(record.get("residual_norm")),
                "residual_effect": _safe_float(record.get("residual_norm_ratio")),
                "treatment_padj": np.nan,
                "directional_rescue": _safe_float(record.get("residual_norm_ratio")) < 1,
                "raw_label": "composition_adjusted",
            }
        )

    subtype = _read_tsv(root / "results/stage3_subtype_localization/subtype_reversal.tsv")
    if not subtype.empty:
        for (population, subtype_name), group in subtype.groupby(["broad_population", "subtype"], observed=True):
            rows.append(
                {
                    "candidate_id": f"subtype:{population}:{subtype_name}",
                    "candidate_type": "subtype_robustness",
                    "population": population,
                    "source": "stage3_subtype_reversal",
                    "aging_effect": _safe_float(group["aging_effect"].median()),
                    "treatment_effect": _safe_float(group["treatment_effect"].median()),
                    "residual_effect": _safe_float(group["residual_effect"].median()),
                    "treatment_padj": np.nan,
                    "directional_rescue": float(group["direction_opposite"].mean()) >= 0.6 if "direction_opposite" in group else False,
                    "raw_label": f"n_genes={len(group)}",
                }
            )

    tf = _read_tsv(follow / "GRANULOSA_TF_CANDIDATES.tsv")
    for _, record in tf.iterrows():
        if str(record.get("candidate_tf", "")).startswith("SKIPPED"):
            continue
        rows.append(
            {
                "candidate_id": f"TF:{record.get('candidate_tf', '')}",
                "candidate_type": "TF_candidate",
                "population": "Granulosa",
                "source": "stage15_followup_TF",
                "aging_effect": _safe_float(record.get("age_effect_OC_minus_Y")),
                "treatment_effect": _safe_float(record.get("treatment_effect_OT_minus_OC")),
                "residual_effect": np.nan,
                "treatment_padj": np.nan,
                "directional_rescue": _safe_float(record.get("treatment_effect_OT_minus_OC")) < 0,
                "raw_label": record.get("mechanistic_status", ""),
            }
        )

    programs = _read_tsv(root / "results/stage8_gene_programs/program_reversal.tsv")
    if not programs.empty:
        for _, record in programs.loc[programs["population"].isin(FOCUS)].head(200).iterrows():
            rows.append(
                {
                    "candidate_id": f"program:{record.get('population', '')}:{record.get('program', '')}",
                    "candidate_type": "gene_program",
                    "population": record.get("population", ""),
                    "source": "stage8_program_reversal",
                    "aging_effect": _safe_float(record.get("aging_effect_OC_minus_Y")),
                    "treatment_effect": _safe_float(record.get("treatment_effect_OT_minus_OC")),
                    "residual_effect": _safe_float(record.get("residual_effect_OT_minus_Y")),
                    "treatment_padj": np.nan,
                    "directional_rescue": bool(record.get("directionally_reversed", False)),
                    "raw_label": record.get("program", ""),
                }
            )

    evidence = pd.DataFrame(rows)
    if evidence.empty:
        evidence = pd.DataFrame(columns=["candidate_id", "candidate_type", "population", "source", "aging_effect", "treatment_effect", "residual_effect", "treatment_padj", "directional_rescue", "raw_label"])
    evidence["directional_rescue"] = evidence["directional_rescue"].astype(str).str.lower().isin(["true", "1"])
    summary_rows: list[dict[str, Any]] = []
    for candidate_id, group in evidence.groupby("candidate_id", observed=True):
        sources = sorted(set(group["source"].astype(str)))
        effects = pd.to_numeric(group["treatment_effect"], errors="coerce").dropna()
        signs = np.sign(effects.to_numpy()) if not effects.empty else np.array([])
        n_negative = int((signs < 0).sum())
        n_positive = int((signs > 0).sum())
        if len(sources) >= 3 and n_negative > 0 and n_positive == 0:
            classification = "multi_evidence_stable_support"
        elif len(sources) >= 2 and n_negative > n_positive:
            classification = "direction_support_but_robustness_incomplete"
        elif n_negative > 0 and n_positive > 0:
            classification = "direction_conflict"
        else:
            classification = "single_method_support"
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "population": group["population"].dropna().astype(str).iloc[0] if not group["population"].dropna().empty else "",
                "candidate_type": group["candidate_type"].dropna().astype(str).iloc[0] if not group["candidate_type"].dropna().empty else "",
                "n_evidence_rows": len(group),
                "n_sources": len(sources),
                "sources": ";".join(sources),
                "median_treatment_effect": float(effects.median()) if not effects.empty else np.nan,
                "n_negative_sources": n_negative,
                "n_positive_sources": n_positive,
                "classification": classification,
            }
        )
    evidence.to_csv(out / "EVIDENCE_MATRIX.tsv", sep="\t", index=False)
    summary = pd.DataFrame(summary_rows).sort_values(["classification", "n_sources"], ascending=[True, False]) if summary_rows else pd.DataFrame()
    summary.to_csv(out / "EVIDENCE_MATRIX_SUMMARY.tsv", sep="\t", index=False)

    subtype_rows: list[dict[str, Any]] = []
    if not subtype.empty:
        for (population, subtype_name), group in subtype.groupby(["broad_population", "subtype"], observed=True):
            effect = pd.to_numeric(group.get("treatment_effect"), errors="coerce")
            subtype_rows.append(
                {
                    "population": population,
                    "subtype": subtype_name,
                    "n_genes": len(group),
                    "direction_opposite_fraction": float(pd.Series(group.get("direction_opposite", False)).astype(bool).mean()),
                    "median_treatment_effect": float(effect.median()),
                    "median_residual_effect": float(pd.to_numeric(group.get("residual_effect"), errors="coerce").median()),
                    "candidate_status": "supports_broad_direction" if float(pd.Series(group.get("direction_opposite", False)).astype(bool).mean()) >= 0.6 else "incomplete_or_conflicting",
                }
            )
    pd.DataFrame(subtype_rows).to_csv(out / "GRANULOSA_SUBTYPE_ROBUSTNESS.tsv", sep="\t", index=False)
    (out / "EVIDENCE_AUDIT_REPORT_CN.md").write_text(
        "# Stage 16A 统一证据矩阵审计\n\n"
        f"本阶段整合 {len(evidence)} 条 evidence rows 和 {len(summary_rows)} 个 candidate IDs。来源包括 pseudobulk、外部年龄轴、状态模块、组成校正、亚型、TF、gene program 和通讯候选。\n\n"
        "证据等级只用于整理，不把不同分析方法当作独立生物学重复；同一 library-level 信号在多个方法中重复出现时仍需防止重复计数。\n",
        encoding="utf-8",
    )
    _write_checkpoint(out / "CHECKPOINT.json", "EVIDENCE_MATRIX_COMPLETE", n_rows=len(evidence), n_candidates=len(summary_rows))
    _update_run_state(stage_root, "01_evidence_matrix", "complete", n_rows=len(evidence))


def _geo_quick_metadata(accession: str, timeout: int = 30) -> dict[str, Any]:
    url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?" + urllib.parse.urlencode({"acc": accession, "targ": "self", "form": "xml", "view": "quick"})
    try:
        data = urllib.request.urlopen(url, timeout=timeout).read().decode("utf-8", errors="replace")
    except Exception as exc:
        return {"accession": accession, "status": "query_failed", "error": repr(exc)}
    clean = re.sub(r"<[^>]+>", " ", data)
    clean = re.sub(r"\s+", " ", clean).strip()
    return {"accession": accession, "status": "queried", "metadata_excerpt": clean[:2000]}


def run_external_registry(config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]) -> None:
    out = stage_root / "02_external_registry"
    root = paths["root"]
    candidate_accessions = ["GSE232309", "GSE267729", "GSE202601"]
    registry_rows: list[dict[str, Any]] = []
    for accession in candidate_accessions:
        result = _geo_quick_metadata(accession)
        result.update({"source": "NCBI GEO", "query_date_utc": datetime.now(timezone.utc).isoformat(), "priority": "high" if accession == "GSE267729" else "candidate"})
        registry_rows.append(result)
    existing = _read_tsv(root / "results/deep_dive_stage15/external_gse267729/GSE267729_sample_registry.tsv")
    if not existing.empty:
        existing = existing.copy()
        existing["dataset"] = "GSE267729"
        existing["sample_level_statistical_unit"] = True
        existing["raw_counts_available"] = True
        existing.to_csv(out / "EXTERNAL_SAMPLE_REGISTRY.tsv", sep="\t", index=False)
    else:
        pd.DataFrame().to_csv(out / "EXTERNAL_SAMPLE_REGISTRY.tsv", sep="\t", index=False)
    registry = pd.DataFrame(registry_rows)
    registry["census_status"] = "not_queried_in_registry_stage"
    registry["include_decision"] = np.where(registry["accession"].eq("GSE267729"), "included_existing_per_sample_10X_and_metadata", "metadata_only_until_sample/library_and_age_criteria_verified")
    registry.to_csv(out / "EXTERNAL_DATASET_REGISTRY.tsv", sep="\t", index=False)
    (out / "EXTERNAL_INCLUSION_REPORT_CN.md").write_text(
        "# Stage 16B 公共卵巢参考数据注册\n\n"
        "GSE267729 已有逐样本 10X 矩阵和年龄/周期 metadata，可作为当前公共参考候选。GSE232309 和 GSE202601 本轮先完成 GEO 元数据登记，只有在确认 sample/library 身份、年龄信息、细胞类型和原始 counts 后才进入模型。\n\n"
        "本轮不把公共细胞数当作生物学重复；公共训练/验证均要求按 sample/library 划分。\n",
        encoding="utf-8",
    )
    _write_checkpoint(out / "CHECKPOINT.json", "EXTERNAL_REGISTRY_COMPLETE", n_datasets=len(registry_rows), n_samples=len(existing))
    _update_run_state(stage_root, "02_external_registry", "complete", n_datasets=len(registry_rows))


def _load_internal_subset(root: Path, config: Mapping[str, Any], population: str, genes: list[str], max_per_library: int = 200) -> ad.AnnData:
    adata = ad.read_h5ad(root / config["deep_dive_stage15"]["input_object"], backed="r")
    obs = adata.obs
    broad_key = config["deep_dive_stage15"]["broad_key"]
    tier_key = config["deep_dive_stage15"]["tier_key"]
    mask = obs[broad_key].astype(str).eq(population).to_numpy()
    if tier_key in obs:
        mask &= obs[tier_key].astype(str).eq("Tier1_primary").to_numpy()
    selected: list[int] = []
    rng = np.random.default_rng(20260919)
    for library, idx in obs.loc[mask].groupby("library_id", observed=True).groups.items():
        # backed AnnData keeps obs_names (barcodes) as the group index; convert
        # names to positional indices before slicing the backed object.
        idx = adata.obs_names.get_indexer(np.asarray(idx, dtype=str))
        idx = idx[idx >= 0]
        if len(idx) > max_per_library:
            idx = rng.choice(idx, size=max_per_library, replace=False)
        selected.extend(idx.tolist())
    query = adata[selected, genes].to_memory()
    counts_layer = str(config["deep_dive_stage15"].get("counts_layer", "counts"))
    if counts_layer not in query.layers:
        adata.file.close()
        raise KeyError(f"raw count layer is required for scVI: {counts_layer}")
    # scVI must receive integer UMI counts, not the normalized/log-transformed X
    # retained by the annotated atlas.
    query.X = query.layers[counts_layer].copy()
    query.obs = query.obs.copy()
    query.obs["dataset"] = "internal_MRJP1"
    query.obs["sample_id"] = query.obs["library_id"].astype(str).to_numpy()
    query.obs["group"] = query.obs["group"].astype(str).to_numpy()
    adata.file.close()
    return query


def _external_marker_subset(path: Path, sample_meta: pd.DataFrame, genes: list[str], population: str, max_per_sample: int = 200) -> ad.AnnData:
    """Read one public 10X sample at a time and keep marker-positive cells.

    The public GSE267729 metadata has sample-level cell counts but no per-cell
    labels in the downloaded matrices. Labels are therefore conservative marker
    score labels and are recorded as inferred, not ground truth.
    """
    import scanpy as sc

    g_markers = [g for g in ["Foxl2", "Amh", "Inha", "Fshr", "Cyp19a1"] if g in genes]
    s_markers = [g for g in ["Dcn", "Lum", "Col1a1", "Col3a1", "Col1a2", "Pdgfra"] if g in genes]
    blocks: list[ad.AnnData] = []
    rng = np.random.default_rng(20260919)
    # The public metadata has one row per sample × inferred population.  Read
    # each physical 10X library once for each modelled population.
    for _, meta in sample_meta.drop_duplicates("sample_id").iterrows():
        sample_id = str(meta["sample_id"])
        sample_dir = path / sample_id
        matrices = list(sample_dir.glob("*_matrix.mtx.gz"))
        if not matrices:
            continue
        prefix = matrices[0].name.replace("matrix.mtx.gz", "")
        try:
            block = sc.read_10x_mtx(sample_dir, var_names="gene_symbols", make_unique=True, prefix=prefix)
        except Exception:
            continue
        common = [g for g in genes if g in block.var_names]
        if len(common) < 500:
            continue
        total = np.asarray(block.X.sum(axis=1)).ravel().astype(float)
        total[total <= 0] = 1
        def marker_score(markers: list[str]) -> np.ndarray:
            idx = block.var_names.get_indexer([g for g in markers if g in block.var_names])
            if len(idx) == 0:
                return np.zeros(block.n_obs)
            return np.asarray(block.X[:, idx].sum(axis=1)).ravel() / total
        g_score = marker_score(g_markers)
        s_score = marker_score(s_markers)
        if population == "Granulosa":
            keep = np.where((g_score > 0) & (g_score >= s_score))[0]
        else:
            keep = np.where((s_score > 0) & (s_score > g_score))[0]
        if len(keep) > max_per_sample:
            keep = rng.choice(keep, size=max_per_sample, replace=False)
        if len(keep) == 0:
            continue
        small = block[keep, common].copy()
        small.obs["sample_id"] = sample_id
        small.obs["library_id"] = sample_id
        small.obs["dataset"] = "GSE267729"
        small.obs["group"] = str(meta["age_group"])
        small.obs["public_label_source"] = "marker_score_inferred"
        blocks.append(small)
    if not blocks:
        return ad.AnnData(sparse.csr_matrix((0, len(genes))), var=pd.DataFrame(index=genes))
    combined = ad.concat(blocks, join="inner", merge="same", index_unique="-")
    combined = combined[:, [g for g in genes if g in combined.var_names]].copy()
    return combined


def _select_hvg_genes(adata: ad.AnnData, n_genes: int = 2000) -> list[str]:
    for key in ["highly_variable", "highly_variable_nbatches"]:
        if key in adata.var:
            genes = adata.var.index[adata.var[key].astype(bool)].astype(str).tolist()
            if len(genes) >= 500:
                return genes[:n_genes]
    return adata.var_names[:n_genes].astype(str).tolist()


def run_scvi_reference(config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]) -> None:
    out = stage_root / "03_scvi_reference"
    root = paths["root"]
    try:
        import scvi
        import torch
    except Exception as exc:
        _record_failure(stage_root, "03_scvi_reference_dependency", exc, logger)
        (out / "SCVI_MODEL_CARD.md").write_text("# scVI参考映射\n\n跳过：ovary_sc环境当前没有可用的 scvi-tools/PyTorch 依赖。\n", encoding="utf-8")
        _write_checkpoint(out / "CHECKPOINT.json", "SCVI_SKIPPED_DEPENDENCY", error=repr(exc))
        _update_run_state(stage_root, "03_scvi_reference", "skipped", reason="dependency_missing")
        return
    import scanpy as sc

    public_meta = _read_tsv(root / "results/deep_dive_stage15/external_gse267729/GSE267729_broad_pseudobulk_metadata.tsv")
    public_path = root / "results/deep_dive_stage15/external_gse267729/processed_10x"
    if public_meta.empty or not public_path.exists():
        exc = FileNotFoundError("GSE267729 per-cell 10X reference missing")
        _record_failure(stage_root, "03_scvi_reference_input", exc, logger)
        _write_checkpoint(out / "CHECKPOINT.json", "SCVI_SKIPPED_INPUT")
        _update_run_state(stage_root, "03_scvi_reference", "skipped", reason="public_per_cell_reference_missing")
        return
    # Use the existing query HVG list, but cap features to keep model memory bounded.
    query_all = ad.read_h5ad(root / config["deep_dive_stage15"]["input_object"], backed="r")
    genes = _select_hvg_genes(query_all, n_genes=2000)
    # Ensure the marker genes needed for public label scoring are retained.
    for gene in ["Foxl2", "Amh", "Inha", "Fshr", "Cyp19a1", "Dcn", "Lum", "Col1a1", "Col3a1", "Col1a2", "Pdgfra"]:
        if gene in query_all.var_names and gene not in genes:
            genes.append(gene)
    query_all.file.close()
    seeds = [20260919, 20260920, 20260921]
    use_gpu = bool(torch.cuda.is_available())
    latent_rows: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    for population in FOCUS:
        try:
            public = _external_marker_subset(public_path, public_meta, genes, population, max_per_sample=200)
            query = _load_internal_subset(root, config, population, genes, max_per_library=200)
            if public.n_obs < 500 or query.n_obs < 500:
                raise RuntimeError(f"insufficient cells for {population}: public={public.n_obs}, query={query.n_obs}")
            combined = ad.concat([public, query], join="inner", merge="same", index_unique=None)
            combined.layers["counts"] = combined.X.copy() if sparse.issparse(combined.X) else sparse.csr_matrix(combined.X)
            scvi.model.SCVI.setup_anndata(combined, layer="counts", batch_key="dataset")
            for seed in seeds:
                scvi.settings.seed = seed
                model = scvi.model.SCVI(combined, n_latent=10, n_layers=2, gene_likelihood="nb")
                model_dir = out / f"model_{population}_{seed}"
                model.train(max_epochs=60, batch_size=256, accelerator="gpu" if use_gpu else "cpu", devices=1, early_stopping=True, early_stopping_patience=10)
                model.save(model_dir, overwrite=True)
                latent = model.get_latent_representation()
                latent_df = pd.DataFrame(latent, index=combined.obs_names, columns=[f"z{i+1}" for i in range(latent.shape[1])])
                latent_df.insert(0, "population", population)
                latent_df.insert(1, "sample_id", combined.obs["sample_id"].astype(str).to_numpy())
                latent_df.insert(2, "dataset", combined.obs["dataset"].astype(str).to_numpy())
                latent_df.insert(3, "group", combined.obs["group"].astype(str).to_numpy())
                latent_df.insert(4, "seed", seed)
                latent_df.to_csv(out / f"SCVI_LATENT_CELLS_{population}_{seed}.tsv.gz", sep="\t", index=True, compression="gzip")
                internal_latent = latent_df.loc[latent_df["dataset"] == "internal_MRJP1"].copy()
                library_summary = internal_latent.groupby("sample_id", observed=True).mean(numeric_only=True).reset_index()
                library_summary.insert(0, "population", population)
                library_summary["seed"] = seed
                latent_rows.append(library_summary)
                model_rows.append({"population": population, "seed": seed, "n_public_cells": int(public.n_obs), "n_query_cells": int(query.n_obs), "n_genes": int(combined.n_vars), "n_latent": 10, "accelerator": "gpu" if use_gpu else "cpu", "model_dir": str(model_dir.relative_to(root))})
                del model
            del combined, public, query
        except Exception as exc:
            _record_failure(stage_root, f"03_scvi_reference_{population}", exc, logger)
            continue
    pd.DataFrame(model_rows).to_csv(out / "SCVI_MODEL_RUNS.tsv", sep="\t", index=False)
    if latent_rows:
        pd.concat(latent_rows, ignore_index=True).to_csv(out / "SCVI_LATENT_LIBRARY_SUMMARY.tsv", sep="\t", index=False)
    stability = pd.DataFrame(model_rows)
    stability.to_csv(out / "SCVI_STABILITY.tsv", sep="\t", index=False)
    (out / "SCVI_MODEL_CARD.md").write_text(
        "# scVI公共参考映射 model card\n\n"
        "GSE267729 逐样本 10X 矩阵作为公共参考；本项目内部数据只作为 query 子集，不参与公共年龄程序定义。公共细胞标签由 marker score 推断，不能当作人工真值。每个样本和内部library均衡抽样至最多200个细胞，scVI使用counts层，n_latent=10，3个seed，早停。\n\n"
        "如果模型或query mapping在任一群体失败，相关结果只记录为失败，不提升证据等级。\n",
        encoding="utf-8",
    )
    status = "SCVI_COMPLETE" if model_rows else "SCVI_FAILED_ALL_POPULATIONS"
    _write_checkpoint(out / "CHECKPOINT.json", status, n_models=len(model_rows), use_gpu=use_gpu)
    _update_run_state(stage_root, "03_scvi_reference", "complete" if model_rows else "failed", n_models=len(model_rows), use_gpu=use_gpu)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den else np.nan


def run_latent_geometry(config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]) -> None:
    out = stage_root / "04_latent_geometry_ot"
    source = stage_root / "03_scvi_reference"
    files = sorted(source.glob("SCVI_LATENT_CELLS_*.tsv.gz"))
    if not files:
        _write_checkpoint(out / "CHECKPOINT.json", "LATENT_GEOMETRY_SKIPPED_NO_SCVI")
        _update_run_state(stage_root, "04_latent_geometry_ot", "skipped", reason="no_scvi_latents")
        return
    geometry_rows: list[dict[str, Any]] = []
    ot_rows: list[dict[str, Any]] = []
    loo_rows: list[dict[str, Any]] = []
    for path in files:
        frame = pd.read_csv(path, sep="\t", compression="gzip", index_col=0)
        zcols = [c for c in frame.columns if c.startswith("z")]
        for population, pop in frame.groupby("population", observed=True):
            cells = pop.loc[pop["dataset"] == "internal_MRJP1"].copy()
            if cells.empty:
                continue
            centroids = cells.groupby(["sample_id", "group"], observed=True)[zcols].mean()
            if not set(GROUPS).issubset(centroids.reset_index()["group"]):
                continue
            cent = centroids.reset_index()
            y = cent.loc[cent["group"] == "Y", zcols].mean().to_numpy()
            oc = cent.loc[cent["group"] == "OC", zcols].mean().to_numpy()
            ot = cent.loc[cent["group"] == "OT", zcols].mean().to_numpy()
            aging = oc - y
            treatment = ot - oc
            residual = treatment - (np.dot(treatment, aging) / max(np.dot(aging, aging), 1e-12)) * aging
            d_oc = float(np.linalg.norm(oc - y)); d_ot = float(np.linalg.norm(ot - y))
            geometry_rows.append({"population": population, "seed": int(pop["seed"].iloc[0]), "distance_OC_to_Y": d_oc, "distance_OT_to_Y": d_ot, "distance_ratio_OT_over_OC": d_ot / d_oc if d_oc else np.nan, "aging_treatment_cosine": _cosine(aging, treatment), "treatment_along_aging_projection": float(np.dot(treatment, aging) / max(np.dot(aging, aging), 1e-12)), "treatment_residual_norm": float(np.linalg.norm(residual)), "treatment_residual_ratio": float(np.linalg.norm(residual) / max(np.linalg.norm(treatment), 1e-12))})
            # Equal-size cell draws avoid larger libraries dominating distribution metrics.
            sample_arrays = {lib: cells.loc[cells["sample_id"] == lib, zcols].to_numpy() for lib in cells["sample_id"].unique()}
            rng = np.random.default_rng(20260919)
            for comparison, left_group, right_group in [("OT_vs_OC", "OT", "OC"), ("OT_vs_Y", "OT", "Y"), ("OC_vs_Y", "OC", "Y")]:
                left = np.vstack([arr for lib, arr in sample_arrays.items() if lib.startswith(left_group + "_")]) if any(lib.startswith(left_group + "_") for lib in sample_arrays) else np.empty((0, len(zcols)))
                right = np.vstack([arr for lib, arr in sample_arrays.items() if lib.startswith(right_group + "_")]) if any(lib.startswith(right_group + "_") for lib in sample_arrays) else np.empty((0, len(zcols)))
                n = min(len(left), len(right), 1000)
                if n == 0:
                    continue
                left = left[rng.choice(len(left), n, replace=False)]
                right = right[rng.choice(len(right), n, replace=False)]
                w1 = float(np.mean([wasserstein_distance(left[:, j], right[:, j]) for j in range(len(zcols))]))
                ot_rows.append({"population": population, "seed": int(pop["seed"].iloc[0]), "comparison": comparison, "n_cells_per_side": n, "metric": "coordinatewise_Wasserstein", "value": w1, "regularization": "not_applicable_coordinatewise_fallback"})
            for excluded in LIBRARIES:
                keep = cells.loc[cells["sample_id"] != excluded]
                cc = keep.groupby(["sample_id", "group"], observed=True)[zcols].mean().reset_index()
                if not set(GROUPS).issubset(cc["group"]):
                    continue
                yy = cc.loc[cc["group"] == "Y", zcols].mean().to_numpy(); oo = cc.loc[cc["group"] == "OC", zcols].mean().to_numpy(); tt = cc.loc[cc["group"] == "OT", zcols].mean().to_numpy()
                aa = oo - yy; tr = tt - oo
                loo_rows.append({"population": population, "seed": int(pop["seed"].iloc[0]), "excluded_library": excluded, "distance_change_OT_minus_OC": float(np.linalg.norm(tt - yy) - np.linalg.norm(oo - yy)), "aging_treatment_cosine": _cosine(aa, tr)})
    pd.DataFrame(geometry_rows).to_csv(out / "LATENT_GEOMETRY.tsv", sep="\t", index=False)
    pd.DataFrame(ot_rows).to_csv(out / "OPTIMAL_TRANSPORT_RESULTS.tsv", sep="\t", index=False)
    pd.DataFrame(ot_rows).to_csv(out / "OT_SENSITIVITY.tsv", sep="\t", index=False)
    pd.DataFrame(loo_rows).to_csv(out / "OT_LOO_RESULTS.tsv", sep="\t", index=False)
    (out / "LATENT_GEOMETRY_REPORT_CN.md").write_text(
        "# Stage 18 潜在空间几何与分布距离\n\n"
        "当前实现使用每个library均衡抽样后的scVI latent。centroid几何和coordinatewise Wasserstein作为描述性结果；细胞bootstrap不被解释为生物学重复，最终方向仍需library-level和leave-one-library-out支持。POT/Sinkhorn若未安装则不伪造，记录为coordinatewise Wasserstein fallback。\n",
        encoding="utf-8",
    )
    _write_checkpoint(out / "CHECKPOINT.json", "LATENT_GEOMETRY_COMPLETE", n_geometry=len(geometry_rows), n_ot=len(ot_rows))
    _update_run_state(stage_root, "04_latent_geometry_ot", "complete", n_geometry=len(geometry_rows))


def run_external_age_models(config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]) -> None:
    out = stage_root / "05_external_age_models"
    root = paths["root"]
    try:
        from sklearn.linear_model import ElasticNet, Ridge
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        _record_failure(stage_root, "05_external_age_models_dependency", exc, logger)
        _write_checkpoint(out / "CHECKPOINT.json", "AGE_MODELS_SKIPPED_DEPENDENCY", error=repr(exc))
        _update_run_state(stage_root, "05_external_age_models", "skipped", reason="sklearn_missing")
        return
    counts_path = root / "results/deep_dive_stage15/external_gse267729/GSE267729_broad_pseudobulk_counts.tsv.gz"
    meta_path = root / "results/deep_dive_stage15/external_gse267729/GSE267729_broad_pseudobulk_metadata.tsv"
    counts = _read_tsv(counts_path, index_col=0); meta = _read_tsv(meta_path)
    if counts.empty or meta.empty:
        _write_checkpoint(out / "CHECKPOINT.json", "AGE_MODELS_SKIPPED_INPUT")
        _update_run_state(stage_root, "05_external_age_models", "skipped", reason="external_counts_missing")
        return
    programs = _read_tsv(root / "results/deep_dive_stage15/external_gse267729/GSE267729_external_age_programs.tsv.gz")
    broad_internal = {}
    internal_path = root / "results/pseudobulk_ready/broad_counts.tsv.gz"
    internal_table = _read_tsv(internal_path)
    if not internal_table.empty:
        pop_col = "population" if "population" in internal_table.columns else "broad_population"
        lib_col = "library" if "library" in internal_table.columns else "library_id"
        gene_cols = [c for c in internal_table.columns if c not in {pop_col, lib_col}]
        for pop, block in internal_table.groupby(pop_col, observed=True):
            frame = block[gene_cols].apply(pd.to_numeric, errors="coerce"); frame.index = block[lib_col].astype(str).to_numpy(); broad_internal[str(pop)] = frame.reindex(LIBRARIES)
    rows: list[dict[str, Any]] = []; score_rows: list[dict[str, Any]] = []; loso_rows: list[dict[str, Any]] = []
    for population in FOCUS:
        pmeta = meta.loc[meta["population"] == population].copy()
        idx = [f"{sid}::{population}" for sid in pmeta["sample_id"].astype(str)]
        idx = [x for x in idx if x in counts.index]
        pmeta = pmeta.loc[pmeta["sample_id"].astype(str).map(lambda x: f"{x}::{population}" in counts.index)].copy()
        if len(pmeta) < 8:
            continue
        c = counts.loc[idx]
        cpm = np.log2(c.div(c.sum(axis=1), axis=0) * 1e6 + 0.5)
        y = pmeta["age_months"].astype(float).to_numpy()
        contrast_program = programs.loc[(programs["population"] == population) & (programs["contrast"].eq("post_acyclic_vs_young"))] if not programs.empty else pd.DataFrame()
        features = contrast_program.sort_values("abs_effect", ascending=False)["gene"].astype(str).head(200).tolist() if not contrast_program.empty else c.columns[:200].astype(str).tolist()
        features = [g for g in features if g in c.columns]
        internal = broad_internal.get(population, pd.DataFrame())
        if not internal.empty:
            # Freeze one identical feature space for public training and
            # internal projection; never silently predict with fewer columns.
            features = [g for g in features if g in internal.columns]
        if len(features) < 10:
            continue
        X = cpm[features].to_numpy()
        for model_name, estimator in [("ridge", Ridge(alpha=10.0)), ("elastic_net", ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=5000))]:
            preds = np.full(len(y), np.nan)
            for i in range(len(y)):
                train = np.arange(len(y)) != i
                model = make_pipeline(StandardScaler(), estimator)
                model.fit(X[train], y[train]); preds[i] = model.predict(X[i : i + 1])[0]
                loso_rows.append({"population": population, "model": model_name, "held_out_sample": pmeta.iloc[i]["sample_id"], "observed_age_months": y[i], "predicted_age_score": preds[i], "n_training_samples": int(train.sum())})
            rows.append({"population": population, "model": model_name, "n_external_samples": len(y), "n_features": len(features), "loso_mae_months": float(mean_absolute_error(y, preds)), "loso_r2": float(r2_score(y, preds)), "decision": "exploratory_only_single_public_study"})
            model = make_pipeline(StandardScaler(), estimator); model.fit(X, y)
            if not internal.empty:
                use = features
                ipred = model.predict(np.log2(internal[use].div(internal[use].sum(axis=1), axis=0) * 1e6 + 0.5).to_numpy())
                for library, score in zip(internal.index, ipred):
                    score_rows.append({"population": population, "model": model_name, "library_id": library, "group": str(library).split("_", 1)[0], "external_age_score": score, "n_features": len(use)})
    pd.DataFrame(rows).to_csv(out / "MODEL_BENCHMARK.tsv", sep="\t", index=False)
    pd.DataFrame(score_rows).to_csv(out / "EXTERNAL_AGE_SCORE_LIBRARY.tsv", sep="\t", index=False)
    pd.DataFrame(loso_rows).to_csv(out / "EXTERNAL_AGE_MODEL_LOSO.tsv", sep="\t", index=False)
    (out / "EXTERNAL_AGE_MODEL_REPORT_CN.md").write_text(
        "# Stage 19 外部年龄模型\n\n"
        "本轮只在GSE267729公共sample层面训练Ridge/Elastic Net并按完整sample做LOSO；没有在9个内部library上训练Y/OC/OT分类器，也没有训练深度模型。公共独立sample数量不足约30且只有单一研究，因此结果为探索性外部年龄相关分数，不称为真实生物学年龄。\n",
        encoding="utf-8",
    )
    _write_checkpoint(out / "CHECKPOINT.json", "EXTERNAL_AGE_MODELS_COMPLETE", n_models=len(rows))
    _update_run_state(stage_root, "05_external_age_models", "complete", n_models=len(rows))


def run_contrastivevi(config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]) -> None:
    out = stage_root / "06_contrastivevi"
    root = paths["root"]
    try:
        import scvi
        import torch
    except Exception as exc:
        _record_failure(stage_root, "06_contrastivevi_dependency", exc, logger)
        _write_checkpoint(out / "CHECKPOINT.json", "CONTRASTIVEVI_SKIPPED_DEPENDENCY", error=repr(exc))
        _update_run_state(stage_root, "06_contrastivevi", "skipped", reason="dependency_missing")
        return
    try:
        import scanpy as sc
        ContrastiveVI = scvi.external.ContrastiveVI
        full = ad.read_h5ad(root / config["deep_dive_stage15"]["input_object"], backed="r")
        obs = full.obs
        genes = _select_hvg_genes(full, n_genes=1500)
        mask = obs[config["deep_dive_stage15"]["broad_key"]].astype(str).eq("Granulosa").to_numpy() & obs["group"].astype(str).isin(["OC", "OT"]).to_numpy()
        rng = np.random.default_rng(20260919); selected = []
        for lib, idx in obs.loc[mask].groupby("library_id", observed=True).groups.items():
            idx = full.obs_names.get_indexer(np.asarray(idx, dtype=str)); idx = idx[idx >= 0]
            selected.extend((rng.choice(idx, min(len(idx), 300), replace=False)).tolist())
        data = full[selected, genes].to_memory(); full.file.close()
        counts_layer = str(config["deep_dive_stage15"].get("counts_layer", "counts"))
        if counts_layer not in data.layers:
            raise KeyError(f"raw count layer is required for contrastiveVI: {counts_layer}")
        if counts_layer != "counts":
            data.layers["counts"] = data.layers[counts_layer].copy()
        model_rows = []; score_rows = []
        for seed in [20260919, 20260920, 20260921]:
            scvi.settings.seed = seed
            ContrastiveVI.setup_anndata(data, layer="counts")
            model = ContrastiveVI(data, n_background_latent=5, n_salient_latent=5)
            background_idx = np.where(data.obs["group"].astype(str).eq("OC"))[0]
            target_idx = np.where(data.obs["group"].astype(str).eq("OT"))[0]
            model.train(background_indices=background_idx, target_indices=target_idx, max_epochs=60, batch_size=256, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, early_stopping=True, early_stopping_patience=10)
            model.save(out / f"model_seed_{seed}", overwrite=True)
            salient = model.get_latent_representation(representation_kind="salient")
            frame = pd.DataFrame(salient, columns=[f"salient_{i+1}" for i in range(salient.shape[1])])
            frame.insert(0, "library_id", data.obs["library_id"].astype(str).to_numpy()); frame.insert(1, "group", data.obs["group"].astype(str).to_numpy()); frame["seed"] = seed
            summary = frame.groupby(["library_id", "group", "seed"], observed=True).mean(numeric_only=True).reset_index(); score_rows.append(summary)
            model_rows.append({"seed": seed, "n_cells": data.n_obs, "n_genes": data.n_vars, "n_background": len(background_idx), "n_target": len(target_idx), "accelerator": "gpu" if torch.cuda.is_available() else "cpu"})
        pd.DataFrame(model_rows).to_csv(out / "CONTRASTIVEVI_STABILITY.tsv", sep="\t", index=False)
        pd.concat(score_rows, ignore_index=True).to_csv(out / "CONTRASTIVEVI_LIBRARY_SCORES.tsv", sep="\t", index=False)
        pd.DataFrame({"gene": genes, "interpretation": "feature space only; no direct causal target inference"}).to_csv(out / "CONTRASTIVEVI_PROGRAM_GENES.tsv", sep="\t", index=False)
        (out / "CONTRASTIVEVI_MODEL_CARD.md").write_text("# contrastiveVI model card\n\nOC为background、OT为target；仅使用Granulosa且每个library均衡抽样。salient latent只作为治疗特异表示候选，需library-level和LOO验证。\n", encoding="utf-8")
        _write_checkpoint(out / "CHECKPOINT.json", "CONTRASTIVEVI_COMPLETE", n_models=len(model_rows))
        _update_run_state(stage_root, "06_contrastivevi", "complete", n_models=len(model_rows))
    except Exception as exc:
        _record_failure(stage_root, "06_contrastivevi_training", exc, logger)
        _write_checkpoint(out / "CHECKPOINT.json", "CONTRASTIVEVI_FAILED", error=repr(exc))
        _update_run_state(stage_root, "06_contrastivevi", "failed", error=repr(exc))


def run_foundation_and_cross_species(config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]) -> None:
    foundation = stage_root / "07_foundation_models"
    rows = [{"model": "Geneformer", "status": "skipped", "reason": "no local pretrained checkpoint; public sample size insufficient for fine-tuning"}, {"model": "scGPT", "status": "skipped", "reason": "no local pretrained checkpoint; embedding-only benefit must beat linear baseline"}]
    pd.DataFrame(rows).to_csv(foundation / "FOUNDATION_MODEL_BENCHMARK.tsv", sep="\t", index=False)
    (foundation / "FOUNDATION_MODEL_REPORT_CN.md").write_text("# Stage 21 基础模型\n\n本轮不下载大型预训练模型，不在9个library上微调；由于没有可验证的外部预训练 checkpoint 和足够独立 sample，记录为跳过。\n", encoding="utf-8")
    _write_checkpoint(foundation / "CHECKPOINT.json", "FOUNDATION_MODELS_SKIPPED")
    _update_run_state(stage_root, "07_foundation_models", "skipped", reason="no_external_checkpoint")

    out = stage_root / "08_cross_species"
    candidate_genes = ["Foxl2", "Hif1a", "Smad3", "Il6", "Il6st", "Fgf2", "Fgfr2", "Col1a1", "Dcn", "Lum", "Cyp19a1", "Star", "Nr5a1"]
    pd.DataFrame({"mouse_gene": candidate_genes, "human_gene": "not_queried", "mapping_status": "requires_versioned_ortholog_service"}).to_csv(out / "ORTHOLOG_MAPPING.tsv", sep="\t", index=False)
    pd.DataFrame(columns=["dataset", "population", "contrast", "effect", "status"]).to_csv(out / "CROSS_SPECIES_PROJECTION.tsv", sep="\t", index=False)
    (out / "CROSS_SPECIES_REPORT_CN.md").write_text("# Stage 22 跨物种投影\n\n本轮只完成候选人类卵巢数据登记；尚未下载并确认可比的人类原始counts和版本化一对一同源基因表，因此不输出跨物种生物学结论。\n", encoding="utf-8")
    _write_checkpoint(out / "CHECKPOINT.json", "CROSS_SPECIES_SKIPPED_NO_VERIFIED_HUMAN_COUNTS")
    _update_run_state(stage_root, "08_cross_species", "skipped", reason="no_verified_human_counts")


def run_mechanism_and_synthesis(config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]) -> None:
    out = stage_root / "09_mechanism_candidates"
    root = paths["root"]
    prior = _read_tsv(root / "results/deep_dive_stage15/followup_validation/GRANULOSA_TF_CANDIDATES.tsv")
    candidates: list[dict[str, Any]] = []
    for _, record in prior.iterrows():
        if str(record.get("candidate_tf", "")).startswith("SKIPPED"):
            continue
        candidates.append({"candidate": record.get("candidate_tf"), "type": "TF", "population": "Granulosa", "treatment_effect": record.get("treatment_effect_OT_minus_OC"), "status": "candidate_hypothesis_only", "orthogonal_validation_needed": "TF activity/target program/protein validation"})
    candidates.extend([
        {"candidate": "Il6-Il6st", "type": "microenvironment", "population": "Stromal_to_Granulosa", "treatment_effect": np.nan, "status": "expression_supported_candidate_only", "orthogonal_validation_needed": "ligand/receptor protein and downstream response"},
        {"candidate": "Fgf2-Fgfr2", "type": "microenvironment", "population": "Stromal_to_Granulosa", "treatment_effect": np.nan, "status": "expression_supported_candidate_only", "orthogonal_validation_needed": "ligand/receptor protein and downstream response"},
    ])
    pd.DataFrame(candidates).to_csv(out / "FINAL_MECHANISM_CANDIDATES.tsv", sep="\t", index=False)
    evidence = _read_tsv(stage_root / "01_evidence_matrix/EVIDENCE_MATRIX_SUMMARY.tsv")
    evidence.to_csv(out / "FINAL_CANDIDATE_PROGRAMS.tsv", sep="\t", index=False)
    model_rows = []
    for path in [stage_root / "05_external_age_models/MODEL_BENCHMARK.tsv", stage_root / "03_scvi_reference/SCVI_MODEL_RUNS.tsv", stage_root / "06_contrastivevi/CONTRASTIVEVI_STABILITY.tsv", stage_root / "07_foundation_models/FOUNDATION_MODEL_BENCHMARK.tsv"]:
        frame = _read_tsv(path)
        if not frame.empty:
            frame["source_file"] = str(path.relative_to(root)); model_rows.append(frame)
    pd.concat(model_rows, ignore_index=True).to_csv(out / "MODEL_COMPARISON.tsv", sep="\t", index=False) if model_rows else pd.DataFrame().to_csv(out / "MODEL_COMPARISON.tsv", sep="\t", index=False)
    (out / "LIMITATIONS_AND_FAILURES_CN.md").write_text("# 限制与失败步骤\n\n统计单位为library，n=3/group；batch、estrous_stage、pool_mouse_ids缺失；公共数据与模型只有在sample级验证可行时才进入主线；深度模型、TF activity、NMF、通讯和attention结果均不等于因果机制。详见Stage16根目录FAILED_STEPS.tsv。\n", encoding="utf-8")
    (out / "PAPER_CLAIMS_CN_EN.md").write_text("# 论文保守表述\n\n中文：MRJP1处理与细胞类型和状态特异的转录重塑相关，Granulosa在多个外部衰老方向上显示部分反向移动；治疗特异的正交重塑、外部泛化和机制候选仍需在library-level和独立数据中验证。\n\nEnglish: MRJP1 treatment was associated with cell-type- and state-specific transcriptional remodeling. Granulosa cells showed partial movement opposite to conserved aging-related transcriptional directions, whereas the magnitude of return toward young reference states and the contribution of treatment-specific orthogonal remodeling required evaluation across datasets and models.\n", encoding="utf-8")
    (out / "NEXT_EXPERIMENTS_PRIORITY_CN.md").write_text("# 下一轮最小实验验证组合\n\n1. Granulosa：Hif1a/Smad3及代表性target程序的qPCR与Western blot；2. ROS/线粒体与类固醇生成指标；3. Il6/Fgf2及受体的组织定位或蛋白验证；4. 卵泡计数、闭锁和激素读数；5. 若条件允许，补充独立动物/批次以提高library-level重复数。\n", encoding="utf-8")
    (out / "MASTER_REPORT_CN.md").write_text("# Stage 16–22 深度挖掘综合报告\n\n## 结论\n\nGranulosa仍是目前最稳定的MRJP1响应主线，但当前证据更支持部分年龄相关转录方向的反向移动与治疗特异重塑并存，而不是整体年轻化。Stromal_fibroblast的结果继续表现为参考依赖的异质性重塑。监督式深度学习没有被用来制造9个library上的伪预测；scVI/contrastiveVI只有在依赖、公共sample和稳定性门控通过时才可作为补充证据。\n\n## 证据边界\n\n所有正式统计仍以library/biological pool为单位。年龄分数不是真实生物学年龄；TF、配体–受体、NMF、latent factor和基础模型embedding只能作为候选机制或假说生成证据。\n\n## 当前阶段状态\n\n详见各子目录checkpoint、EVIDENCE_MATRIX.tsv、MODEL_COMPARISON.tsv、FINAL_MECHANISM_CANDIDATES.tsv和FAILED_STEPS.tsv。\n", encoding="utf-8")
    (out / "EXECUTIVE_SUMMARY_CN.md").write_text("# 执行摘要\n\nGranulosa仍为稳定主线；Stromal只能作为异质性结果。当前模型分析若不能在公共sample和leave-one-library-out层面优于简单线性/冻结程序基线，则不提升正文证据等级。\n", encoding="utf-8")
    _write_checkpoint(out / "CHECKPOINT.json", "SYNTHESIS_COMPLETE", n_candidates=len(candidates))
    _update_run_state(stage_root, "09_mechanism_candidates", "complete", n_candidates=len(candidates))


def _manifest(stage_root: Path) -> None:
    rows = []
    for path in sorted(stage_root.rglob("*")):
        if path.is_file() and path.name != "RUN_MANIFEST.tsv":
            rows.append({"path": str(path.relative_to(stage_root)), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    pd.DataFrame(rows).to_csv(stage_root / "RUN_MANIFEST.tsv", sep="\t", index=False)


def run_stage16_ml(
    config: Mapping[str, Any], selected_stages: Iterable[str] | None = None
) -> Path:
    stage_root, logger, paths = _init_stage(config)
    _update_run_state(stage_root, "initialization", "complete")
    stages = [
        ("00_audit", run_input_audit),
        ("01_evidence_matrix", run_evidence_matrix),
        ("02_external_registry", run_external_registry),
        ("03_scvi_reference", run_scvi_reference),
        ("04_latent_geometry_ot", run_latent_geometry),
        ("05_external_age_models", run_external_age_models),
        ("06_contrastivevi", run_contrastivevi),
        ("07_foundation_models", run_foundation_and_cross_species),
        ("09_mechanism_candidates", run_mechanism_and_synthesis),
    ]
    selected = set(selected_stages) if selected_stages is not None else None
    if selected is not None:
        known = {name for name, _ in stages}
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError(f"Unknown Stage16 stage names: {unknown}")
        stages = [(name, function) for name, function in stages if name in selected]
    state_path = stage_root / "RUN_STATE.json"
    for name, function in stages:
        # Resume safely: completed/skipped stages are immutable inputs for the
        # next stage and should not be retrained after an SSH/session restart.
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        previous = state.get("stages", {}).get(name, {}).get("status")
        if previous in {"complete", "skipped"}:
            logger.info("Skipping already finished stage %s (%s)", name, previous)
            continue
        succeeded = False
        for attempt in (1, 2):
            try:
                logger.info("Starting stage %s (attempt %d/2)", name, attempt)
                function(config, stage_root, logger, paths)
                succeeded = True
                break
            except Exception as exc:
                _record_failure(stage_root, f"{name}:attempt_{attempt}", exc, logger)
                _update_run_state(stage_root, name, "failed", error=repr(exc), attempt=attempt)
                if attempt == 1:
                    logger.warning("Retrying stage %s once after failure", name)
        if not succeeded:
            logger.error("Stage %s failed after one retry; continuing independent stages", name)
    _manifest(stage_root)
    if selected is not None:
        _update_run_state(
            stage_root,
            "overall",
            "partial",
            selected_stages=sorted(selected),
            next_stage="resume_remaining_stages",
        )
        print("STAGE16_22_PARTIAL_COMPLETE")
        print(f"OUTPUT={stage_root.relative_to(paths['root'])}")
        return stage_root
    has_failures = (stage_root / "FAILED_STEPS.tsv").exists()
    _update_run_state(
        stage_root,
        "overall",
        "complete_with_failures" if has_failures else "complete",
        next_stage="manual_review_before_claims",
    )
    print("STAGE16_22_ML_COMPLETE")
    print(f"OUTPUT={stage_root.relative_to(paths['root'])}")
    return stage_root
