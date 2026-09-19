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
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, wasserstein_distance

from .project import project_paths, setup_logging
from .stage15_projection_state import LIBRARIES, _log_cpm
from .stage16_synthesis import run_stage16_synthesis


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
                        "candidate_id": f"gene:{population}:{gene}",
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

    publication_source = root / "results/publication_stage13/source_data"
    # Pathway-level evidence is kept distinct from gene-level evidence, while
    # reusing a population:pathway ID so internal, subtype and external support
    # can be audited together without treating them as biological replicates.
    for population, filename in [
        ("Granulosa", "pathway_reversal_granulosa.tsv"),
        ("Stromal_fibroblast", "pathway_reversal_stromal_fibroblast.tsv"),
    ]:
        pathways = _read_tsv(publication_source / filename)
        for _, record in pathways.iterrows():
            rows.append(
                {
                    "candidate_id": f"pathway:{population}:{record.get('pathway', '')}",
                    "candidate_type": "pathway",
                    "population": population,
                    "source": "stage13_pathway_GSEA_and_permutation",
                    "aging_effect": _safe_float(record.get("aging_NES")),
                    "treatment_effect": _safe_float(record.get("treatment_NES")),
                    "residual_effect": _safe_float(record.get("residual_NES")),
                    "treatment_padj": _safe_float(record.get("treatment_FDR")),
                    "directional_rescue": bool(record.get("direction_opposite", False)),
                    "raw_label": record.get("pathway_evidence_level", ""),
                }
            )

    external_pathways = _read_tsv(publication_source / "external_pathway_validation.tsv")
    for _, record in external_pathways.iterrows():
        population = str(record.get("population", ""))
        rows.append(
            {
                "candidate_id": f"pathway:{population}:{record.get('pathway', '')}",
                "candidate_type": "pathway",
                "population": population,
                "source": "external_pathway_validation",
                "aging_effect": _safe_float(record.get("external_aging_NES")),
                "treatment_effect": _safe_float(record.get("treatment_NES")),
                "residual_effect": np.nan,
                "treatment_padj": _safe_float(record.get("external_aging_FDR_q")),
                "directional_rescue": bool(
                    record.get("direction_consistent_and_treatment_opposed", False)
                ),
                "raw_label": "external_direction_check",
            }
        )

    for filename in [
        "subtype_localization_granulosa.tsv",
        "subtype_localization_stromal_fibroblast.tsv",
    ]:
        subtype_pathways = _read_tsv(publication_source / filename)
        for _, record in subtype_pathways.iterrows():
            population = str(record.get("broad_population", ""))
            rows.append(
                {
                    "candidate_id": f"pathway:{population}:{record.get('pathway', '')}",
                    "candidate_type": "pathway",
                    "population": population,
                    "source": "stage3_subtype_localization",
                    "aging_effect": _safe_float(record.get("aging_NES")),
                    "treatment_effect": _safe_float(record.get("treatment_NES")),
                    "residual_effect": _safe_float(record.get("residual_NES")),
                    "treatment_padj": _safe_float(record.get("treatment_FDR")),
                    "directional_rescue": bool(record.get("direction_opposite", False)),
                    "raw_label": str(record.get("subtype", "")),
                }
            )

    exact = _read_tsv(publication_source / "population_exact_permutation_evidence.tsv")
    for _, record in exact.iterrows():
        population = str(record.get("population", ""))
        rows.append(
            {
                "candidate_id": f"population_signature:{population}",
                "candidate_type": "population_signature",
                "population": population,
                "source": "exact_3v3_permutation",
                "aging_effect": np.nan,
                "treatment_effect": _safe_float(record.get("whole_all_cosine_similarity_observed")),
                "residual_effect": _safe_float(record.get("median_recovery_fraction_observed")),
                "treatment_padj": _safe_float(record.get("whole_all_cosine_similarity_p_empirical")),
                "directional_rescue": bool(record.get("permutation_supported_signature", False)),
                "raw_label": record.get("population_evidence_level", ""),
            }
        )

    communication = _read_tsv(publication_source / "communication_ligand_hypotheses.tsv")
    for _, record in communication.iterrows():
        sender = str(record.get("sender", ""))
        receiver = str(record.get("receiver", ""))
        ligand = str(record.get("ligand", ""))
        rows.append(
            {
                "candidate_id": f"communication:{sender}:{receiver}:{ligand}",
                "candidate_type": "ligand_receptor_hypothesis",
                "population": receiver,
                "source": "stage13_communication_candidate",
                "aging_effect": _safe_float(record.get("sender_aging_effect")),
                "treatment_effect": _safe_float(record.get("sender_treatment_effect")),
                "residual_effect": _safe_float(record.get("sender_residual_effect")),
                "treatment_padj": np.nan,
                "directional_rescue": bool(
                    record.get("sender_directional_rescue_candidate", False)
                ),
                "raw_label": "expression_supported_not_causal",
            }
        )

    for population, filename in [
        ("Granulosa", "nmf_programs_granulosa.tsv"),
        ("Stromal_fibroblast", "nmf_programs_stromal_fibroblast.tsv"),
    ]:
        nmf = _read_tsv(publication_source / filename)
        if nmf.empty:
            continue
        for program, block in nmf.groupby("program", observed=True):
            means = block.groupby("group", observed=True)["program_activity"].mean()
            y_mean = _safe_float(means.get("Y"))
            oc_mean = _safe_float(means.get("OC"))
            ot_mean = _safe_float(means.get("OT"))
            rows.append(
                {
                    "candidate_id": f"nmf:{population}:{program}",
                    "candidate_type": "NMF_program",
                    "population": population,
                    "source": "stage8_NMF_program",
                    "aging_effect": oc_mean - y_mean,
                    "treatment_effect": ot_mean - oc_mean,
                    "residual_effect": ot_mean - y_mean,
                    "treatment_padj": np.nan,
                    "directional_rescue": bool(
                        block["directionally_reversed"].astype(bool).all()
                    ),
                    "raw_label": "exploratory_program_not_causal",
                }
            )

    follow = root / "results/deep_dive_stage15/followup_validation"
    state = _read_tsv(follow / "STATE_MODULE_LOO.tsv")
    if not state.empty:
        state = state.loc[(state["summary_type"] == "full") & (state["metric"] == "mean") & (state["contrast"] == "OT_vs_OC")]
        for _, record in state.iterrows():
            rows.append(
                {
                    "candidate_id": (
                        f"state:{record.get('population', '')}:{record.get('module', '')}"
                    ),
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
        source_support = (
            group.assign(_support=group["directional_rescue"].astype(bool))
            .groupby("source", observed=True)["_support"]
            .mean()
            .ge(0.5)
        )
        n_supporting = int(source_support.sum())
        n_non_supporting = int((~source_support).sum())
        if len(sources) >= 3 and n_supporting == len(sources):
            classification = "multi_evidence_stable_support"
        elif len(sources) >= 2 and n_supporting > n_non_supporting:
            classification = "direction_support_but_robustness_incomplete"
        elif len(sources) >= 2 and n_supporting > 0 and n_non_supporting > 0:
            classification = "direction_conflict"
        elif n_supporting > 0:
            classification = "single_method_support"
        else:
            classification = "no_directional_support"
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "population": group["population"].dropna().astype(str).iloc[0] if not group["population"].dropna().empty else "",
                "candidate_type": group["candidate_type"].dropna().astype(str).iloc[0] if not group["candidate_type"].dropna().empty else "",
                "n_evidence_rows": len(group),
                "n_sources": len(sources),
                "sources": ";".join(sources),
                "median_treatment_effect": float(effects.median()) if not effects.empty else np.nan,
                "n_supporting_sources": n_supporting,
                "n_non_supporting_sources": n_non_supporting,
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


def _ensembl_json(url: str, timeout: int = 45) -> dict[str, Any]:
    """Read a small Ensembl REST response without adding a new dependency."""
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "ms-ovary-scrnaseq-stage16/2026.09",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_one_to_one_orthologs(
    payload: Mapping[str, Any], mouse_symbol: str, release: str
) -> list[dict[str, Any]]:
    """Return auditable mouse-human one-to-one mappings from Ensembl payload."""
    rows: list[dict[str, Any]] = []
    for datum in payload.get("data", []) or []:
        for homology in datum.get("homologies", []) or []:
            target = homology.get("target", {}) or {}
            source = homology.get("source", {}) or {}
            homology_type = str(homology.get("type", ""))
            if str(target.get("species", "")) != "homo_sapiens":
                continue
            if homology_type not in {"ortholog_one2one", "ortholog_one2one_apparent"}:
                continue
            rows.append(
                {
                    "mouse_gene": mouse_symbol,
                    "mouse_ensembl_gene_id": source.get("id", ""),
                    "mouse_protein_id": source.get("protein_id", ""),
                    "human_gene": target.get("display_id", ""),
                    "human_ensembl_gene_id": target.get("id", ""),
                    "human_protein_id": target.get("protein_id", ""),
                    "homology_type": homology_type,
                    "mouse_identity_pct": source.get("perc_id", np.nan),
                    "human_identity_pct": target.get("perc_id", np.nan),
                    "confidence": homology.get("confidence", np.nan),
                    "mapping_status": "ensembl_one_to_one",
                    "ensembl_release": release,
                }
            )
    return rows


def _query_one_to_one_orthologs(
    mouse_symbol: str, release: str, attempts: int = 2
) -> tuple[list[dict[str, Any]], str]:
    """Query and resolve symbols, retrying transient Ensembl failures once."""
    params = urllib.parse.urlencode(
        {"target_species": "homo_sapiens", "type": "orthologues"}
    )
    url = (
        "https://rest.ensembl.org/homology/symbol/mus_musculus/"
        + urllib.parse.quote(mouse_symbol)
        + "?"
        + params
    )
    last_error = ""
    for _ in range(attempts):
        try:
            payload = _ensembl_json(url)
            extracted = _extract_one_to_one_orthologs(
                payload, mouse_symbol, release
            )
            for row in extracted:
                if not row.get("human_gene") and row.get(
                    "human_ensembl_gene_id"
                ):
                    lookup_url = (
                        "https://rest.ensembl.org/lookup/id/"
                        + urllib.parse.quote(
                            str(row["human_ensembl_gene_id"])
                        )
                    )
                    lookup = _ensembl_json(lookup_url)
                    row["human_gene"] = lookup.get("display_name", "")
            return extracted, ""
        except Exception as exc:
            last_error = f"query_failed:{type(exc).__name__}"
    return [], last_error


def run_external_registry(config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]) -> None:
    out = stage_root / "02_external_registry"
    root = paths["root"]
    candidate_accessions = ["GSE232309", "GSE267729", "GSE202601"]
    registry_rows: list[dict[str, Any]] = []
    for accession in candidate_accessions:
        result = _geo_quick_metadata(accession)
        result.update({"source": "NCBI GEO", "query_date_utc": datetime.now(timezone.utc).isoformat(), "priority": "high" if accession == "GSE267729" else "candidate"})
        registry_rows.append(result)
    census_version = "2025-11-08"
    census_candidates = pd.DataFrame()
    census_tissue_counts = pd.DataFrame()
    try:
        import cellxgene_census

        with cellxgene_census.open_soma(census_version=census_version) as census:
            datasets = census["census_info"]["datasets"].read().concat().to_pandas()
            text_columns = [
                column
                for column in datasets.columns
                if any(
                    token in str(column).lower()
                    for token in ["title", "collection", "organism", "dataset"]
                )
            ]
            mask = pd.Series(False, index=datasets.index)
            for column in text_columns:
                mask |= datasets[column].astype(str).str.contains(
                    r"ovar|oocyte|follic", case=False, regex=True, na=False
                )
            census_candidates = datasets.loc[mask].copy()
            summary_counts = (
                census["census_info"]["summary_cell_counts"]
                .read()
                .concat()
                .to_pandas()
            )
            label_column = next(
                (c for c in ["label", "value"] if c in summary_counts.columns), None
            )
            if label_column is not None:
                census_tissue_counts = summary_counts.loc[
                    summary_counts[label_column]
                    .astype(str)
                    .str.contains(r"ovar", case=False, regex=True, na=False)
                ].copy()
        census_candidates.to_csv(
            out / "CELLXGENE_OVARY_DATASETS.tsv", sep="\t", index=False
        )
        census_tissue_counts.to_csv(
            out / "CELLXGENE_OVARY_SUMMARY_COUNTS.tsv", sep="\t", index=False
        )
        for _, record in census_candidates.iterrows():
            dataset_id = str(record.get("dataset_id", record.get("soma_joinid", "")))
            title = str(
                record.get("dataset_title", record.get("collection_name", ""))
            )
            registry_rows.append(
                {
                    "accession": dataset_id,
                    "status": "census_metadata_candidate",
                    "metadata_excerpt": title,
                    "source": "CZ CELLxGENE Census",
                    "query_date_utc": datetime.now(timezone.utc).isoformat(),
                    "priority": "candidate",
                    "census_version": census_version,
                }
            )
        census_status = f"queried_{census_version}"
    except Exception as exc:
        census_status = f"query_failed:{type(exc).__name__}"
        _record_failure(stage_root, "02_external_registry_cellxgene", exc, logger)
        pd.DataFrame().to_csv(
            out / "CELLXGENE_OVARY_DATASETS.tsv", sep="\t", index=False
        )
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
    registry["census_status"] = census_status
    registry["include_decision"] = np.where(
        registry["accession"].eq("GSE267729"),
        "included_existing_per_sample_10X_and_metadata",
        "metadata_only_until_sample/library_and_age_criteria_verified",
    )
    registry.to_csv(out / "EXTERNAL_DATASET_REGISTRY.tsv", sep="\t", index=False)
    (out / "EXTERNAL_INCLUSION_REPORT_CN.md").write_text(
        "# Stage 16B 公共卵巢参考数据注册\n\n"
        f"GSE267729 已有逐样本 10X 矩阵和年龄/周期 metadata，可作为当前公共参考候选。GSE232309 和 GSE202601 本轮先完成 GEO 元数据登记，只有在确认 sample/library 身份、年龄信息、细胞类型和原始 counts 后才进入模型。CELLxGENE Census固定查询版本为{census_version}，共登记{len(census_candidates)}个标题/集合元数据候选。\n\n"
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


def _external_marker_subset(
    path: Path,
    sample_meta: pd.DataFrame,
    genes: list[str],
    population: str,
    max_per_sample: int = 200,
    min_common_genes: int = 500,
) -> ad.AnnData:
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
        if len(common) < min_common_genes:
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


def _scvi_history_summary(model: Any, output: Path) -> dict[str, Any]:
    """Save tidy training history and return final optimization diagnostics."""
    records: list[dict[str, Any]] = []
    history = getattr(model, "history", {}) or {}
    for metric, values in history.items():
        frame = values if isinstance(values, pd.DataFrame) else pd.DataFrame(values)
        if frame.empty:
            continue
        numeric = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
        for epoch, value in enumerate(numeric.to_numpy()):
            records.append({"epoch": epoch, "metric": str(metric), "value": value})
    pd.DataFrame(records).to_csv(output, sep="\t", index=False)

    summary: dict[str, Any] = {"epochs_trained": 0}
    if records:
        tidy = pd.DataFrame(records)
        summary["epochs_trained"] = int(tidy["epoch"].max() + 1)
        for label, candidates in {
            "train_elbo_final": ("elbo_train", "train_loss_epoch"),
            "validation_elbo_final": ("elbo_validation", "validation_loss"),
            "reconstruction_loss_final": (
                "reconstruction_loss_train",
                "reconstruction_loss_validation",
            ),
        }.items():
            summary[label] = np.nan
            for metric in candidates:
                values = tidy.loc[tidy["metric"].eq(metric), "value"].dropna()
                if not values.empty:
                    summary[label] = float(values.iloc[-1])
                    break
    return summary


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
    if use_gpu:
        torch.set_float32_matmul_precision("high")
    latent_rows: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    for population in FOCUS:
        try:
            public = _external_marker_subset(public_path, public_meta, genes, population, max_per_sample=200)
            query = _load_internal_subset(root, config, population, genes, max_per_library=200)
            if public.n_obs < 500 or query.n_obs < 500:
                raise RuntimeError(f"insufficient cells for {population}: public={public.n_obs}, query={query.n_obs}")
            # join='inner' also intersects observation annotations. Add an
            # explicit public placeholder so that the query-only subtype label
            # survives concatenation and can be used for internal geometry.
            subtype_key = str(
                config["deep_dive_stage15"].get("subtype_key", "")
            )
            if subtype_key:
                if subtype_key not in public.obs:
                    public.obs[subtype_key] = "not_available"
                if subtype_key not in query.obs:
                    query.obs[subtype_key] = "not_available"
            combined = ad.concat([public, query], join="inner", merge="same", index_unique=None)
            combined.layers["counts"] = combined.X.copy() if sparse.issparse(combined.X) else sparse.csr_matrix(combined.X)
            scvi.model.SCVI.setup_anndata(combined, layer="counts", batch_key="dataset")
            for seed in seeds:
                scvi.settings.seed = seed
                model = scvi.model.SCVI(combined, n_latent=10, n_layers=2, gene_likelihood="nb")
                model_dir = out / f"model_{population}_{seed}"
                model.train(
                    max_epochs=60,
                    batch_size=256,
                    accelerator="gpu" if use_gpu else "cpu",
                    devices=1,
                    early_stopping=True,
                    early_stopping_patience=10,
                    enable_progress_bar=False,
                )
                model.save(model_dir, overwrite=True)
                history = _scvi_history_summary(
                    model, out / f"SCVI_TRAINING_HISTORY_{population}_{seed}.tsv"
                )
                latent = model.get_latent_representation()
                latent_df = pd.DataFrame(latent, index=combined.obs_names, columns=[f"z{i+1}" for i in range(latent.shape[1])])
                latent_df.insert(0, "population", population)
                latent_df.insert(1, "sample_id", combined.obs["sample_id"].astype(str).to_numpy())
                latent_df.insert(2, "dataset", combined.obs["dataset"].astype(str).to_numpy())
                latent_df.insert(3, "group", combined.obs["group"].astype(str).to_numpy())
                latent_df.insert(4, "seed", seed)
                subtype = (
                    combined.obs[subtype_key].astype(str).to_numpy()
                    if subtype_key in combined.obs
                    else np.repeat("not_available", combined.n_obs)
                )
                latent_df.insert(5, "subtype", subtype)
                latent_df.to_csv(out / f"SCVI_LATENT_CELLS_{population}_{seed}.tsv.gz", sep="\t", index=True, compression="gzip")
                internal_latent = latent_df.loc[latent_df["dataset"] == "internal_MRJP1"].copy()
                library_summary = internal_latent.groupby("sample_id", observed=True).mean(numeric_only=True).reset_index()
                library_summary.insert(0, "population", population)
                library_summary.insert(
                    2,
                    "group",
                    library_summary["sample_id"].astype(str).str.split("_", n=1).str[0],
                )
                library_summary["seed"] = seed
                latent_rows.append(library_summary)
                public_latent = latent_df.loc[latent_df["dataset"] == "GSE267729"]
                public_centroids = public_latent.groupby("group", observed=True)[
                    [f"z{i+1}" for i in range(latent.shape[1])]
                ].mean()
                public_young_old = np.nan
                if {"young", "old"}.issubset(public_centroids.index):
                    public_young_old = float(
                        np.linalg.norm(
                            public_centroids.loc["old"].to_numpy()
                            - public_centroids.loc["young"].to_numpy()
                        )
                    )
                model_rows.append(
                    {
                        "population": population,
                        "seed": seed,
                        "n_public_cells": int(public.n_obs),
                        "n_query_cells": int(query.n_obs),
                        "n_genes": int(combined.n_vars),
                        "n_latent": 10,
                        "accelerator": "gpu" if use_gpu else "cpu",
                        "model_dir": str(model_dir.relative_to(root)),
                        "public_young_old_centroid_distance": public_young_old,
                        "study_mixing_assessment": "not_assessable_single_public_study",
                        "cell_identity_assessment": "population_specific_marker_inferred_reference",
                        **history,
                    }
                )
                del model
            del combined, public, query
        except Exception as exc:
            _record_failure(stage_root, f"03_scvi_reference_{population}", exc, logger)
            continue
    pd.DataFrame(model_rows).to_csv(out / "SCVI_MODEL_RUNS.tsv", sep="\t", index=False)
    stability_rows: list[dict[str, Any]] = []
    if latent_rows:
        library_latent = pd.concat(latent_rows, ignore_index=True)
        zcols = [c for c in library_latent.columns if c.startswith("z")]
        library_latent["distance_to_internal_centroid"] = np.nan
        library_latent["library_outlier_robust_z"] = np.nan
        for (_, _), index in library_latent.groupby(
            ["population", "seed"], observed=True
        ).groups.items():
            coords = library_latent.loc[index, zcols].to_numpy()
            distance = np.linalg.norm(coords - coords.mean(axis=0), axis=1)
            median = float(np.median(distance))
            mad = float(np.median(np.abs(distance - median)))
            robust_z = (distance - median) / max(1.4826 * mad, 1e-12)
            library_latent.loc[index, "distance_to_internal_centroid"] = distance
            library_latent.loc[index, "library_outlier_robust_z"] = robust_z
        library_latent.to_csv(
            out / "SCVI_LATENT_LIBRARY_SUMMARY.tsv", sep="\t", index=False
        )
        for population, block in library_latent.groupby("population", observed=True):
            for seed_a, seed_b in combinations(sorted(block["seed"].unique()), 2):
                left = block.loc[block["seed"].eq(seed_a)].set_index("sample_id")
                right = block.loc[block["seed"].eq(seed_b)].set_index("sample_id")
                common = left.index.intersection(right.index).sort_values()
                a = left.loc[common, zcols].to_numpy()
                b = right.loc[common, zcols].to_numpy()
                distance_a = pdist(a)
                distance_b = pdist(b)
                correlation = spearmanr(distance_a, distance_b).statistic
                _, _, disparity = procrustes(a, b)
                stability_rows.append(
                    {
                        "population": population,
                        "seed_a": int(seed_a),
                        "seed_b": int(seed_b),
                        "n_libraries": len(common),
                        "library_distance_spearman": float(correlation),
                        "procrustes_disparity": float(disparity),
                        "public_cohort_loo": "not_assessable_single_public_study",
                        "interpretation": "geometry_stability_not_biological_replication",
                    }
                )
    pd.DataFrame(stability_rows).to_csv(out / "SCVI_STABILITY.tsv", sep="\t", index=False)
    (out / "SCVI_MODEL_CARD.md").write_text(
        "# scVI公共参考映射 model card\n\n"
        "GSE267729 逐样本 10X 矩阵作为公共参考；本项目内部数据只作为 query 子集，不参与公共年龄程序定义。公共细胞标签由 marker score 推断，不能当作人工真值。每个样本和内部library均衡抽样至最多200个细胞，scVI使用原始counts层，n_latent=10，3个seed，早停。\n\n"
        "模型保存训练历史、公共young-old centroid距离、内部library离群度、跨seed library距离相关和Procrustes差异。由于当前只有一个合格公共study，study mixing和leave-one-public-cohort-out不可评估；这是外部泛化的明确限制。\n\n"
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
    try:
        import ot
    except Exception:
        ot = None
    geometry_rows: list[dict[str, Any]] = []
    ot_rows: list[dict[str, Any]] = []
    loo_rows: list[dict[str, Any]] = []
    excluded_contexts: list[dict[str, Any]] = []
    regularization_grid = [0.01, 0.05, 0.10]
    for path in files:
        frame = pd.read_csv(path, sep="\t", compression="gzip", index_col=0)
        zcols = [c for c in frame.columns if c.startswith("z")]
        for population, pop in frame.groupby("population", observed=True):
            cells = pop.loc[pop["dataset"] == "internal_MRJP1"].copy()
            if cells.empty:
                continue
            seed = int(pop["seed"].iloc[0])
            contexts: list[tuple[str, str, pd.DataFrame]] = [("broad", "all", cells)]
            if "subtype" in cells:
                invalid = {"", "nan", "None", "not_available", "Unresolved"}
                for subtype, subtype_cells in cells.groupby("subtype", observed=True):
                    if str(subtype) in invalid:
                        continue
                    counts = subtype_cells.groupby(
                        ["group", "sample_id"], observed=True
                    ).size()
                    complete = True
                    for group in GROUPS:
                        group_counts = counts.loc[group] if group in counts.index else pd.Series(dtype=float)
                        if len(group_counts) < 3 or int(group_counts.min()) < 10:
                            complete = False
                    if complete:
                        contexts.append(("subtype", str(subtype), subtype_cells))
                    else:
                        excluded_contexts.append(
                            {
                                "population": population,
                                "seed": seed,
                                "subtype": subtype,
                                "reason": "requires_3_libraries_per_group_and_min_10_sampled_cells",
                            }
                        )

            for analysis_level, state, context in contexts:
                cent = context.groupby(["sample_id", "group"], observed=True)[
                    zcols
                ].mean().reset_index()
                if not set(GROUPS).issubset(cent["group"]):
                    continue
                y = cent.loc[cent["group"] == "Y", zcols].mean().to_numpy()
                oc = cent.loc[cent["group"] == "OC", zcols].mean().to_numpy()
                treated = cent.loc[cent["group"] == "OT", zcols].mean().to_numpy()
                aging = oc - y
                treatment = treated - oc
                projection = float(
                    np.dot(treatment, aging) / max(np.dot(aging, aging), 1e-12)
                )
                residual = treatment - projection * aging
                d_oc = float(np.linalg.norm(oc - y))
                d_ot = float(np.linalg.norm(treated - y))
                geometry_rows.append(
                    {
                        "population": population,
                        "analysis_level": analysis_level,
                        "state": state,
                        "seed": seed,
                        "n_cells": len(context),
                        "n_libraries": int(cent["sample_id"].nunique()),
                        "distance_OC_to_Y": d_oc,
                        "distance_OT_to_Y": d_ot,
                        "distance_change_OT_minus_OC": d_ot - d_oc,
                        "distance_ratio_OT_over_OC": d_ot / d_oc if d_oc else np.nan,
                        "aging_treatment_cosine": _cosine(aging, treatment),
                        "treatment_along_aging_projection": projection,
                        "treatment_residual_norm": float(np.linalg.norm(residual)),
                        "treatment_residual_ratio": float(
                            np.linalg.norm(residual)
                            / max(np.linalg.norm(treatment), 1e-12)
                        ),
                    }
                )

                for excluded in sorted(context["sample_id"].astype(str).unique()):
                    keep = context.loc[context["sample_id"].astype(str).ne(excluded)]
                    cc = keep.groupby(["sample_id", "group"], observed=True)[
                        zcols
                    ].mean().reset_index()
                    if any(cc.loc[cc["group"].eq(g), "sample_id"].nunique() < 2 for g in GROUPS):
                        continue
                    yy = cc.loc[cc["group"] == "Y", zcols].mean().to_numpy()
                    oo = cc.loc[cc["group"] == "OC", zcols].mean().to_numpy()
                    tt = cc.loc[cc["group"] == "OT", zcols].mean().to_numpy()
                    aa = oo - yy
                    tr = tt - oo
                    loo_rows.append(
                        {
                            "population": population,
                            "analysis_level": analysis_level,
                            "state": state,
                            "seed": seed,
                            "excluded_library": excluded,
                            "distance_change_OT_minus_OC": float(
                                np.linalg.norm(tt - yy) - np.linalg.norm(oo - yy)
                            ),
                            "distance_ratio_OT_over_OC": float(
                                np.linalg.norm(tt - yy)
                                / max(np.linalg.norm(oo - yy), 1e-12)
                            ),
                            "aging_treatment_cosine": _cosine(aa, tr),
                        }
                    )

                sample_arrays = {
                    str(library): block[zcols].to_numpy()
                    for library, block in context.groupby("sample_id", observed=True)
                }
                rng = np.random.default_rng(20260919 + seed)
                for comparison, left_group, right_group in [
                    ("OT_vs_OC", "OT", "OC"),
                    ("OT_vs_Y", "OT", "Y"),
                    ("OC_vs_Y", "OC", "Y"),
                ]:
                    left_libraries = sorted(
                        lib for lib in sample_arrays if lib.startswith(left_group + "_")
                    )
                    right_libraries = sorted(
                        lib for lib in sample_arrays if lib.startswith(right_group + "_")
                    )
                    for left_library in left_libraries:
                        for right_library in right_libraries:
                            left = sample_arrays[left_library]
                            right = sample_arrays[right_library]
                            n = min(len(left), len(right), 150)
                            if n < 10:
                                continue
                            a = left[rng.choice(len(left), n, replace=False)]
                            b = right[rng.choice(len(right), n, replace=False)]
                            coordinate_w1 = float(
                                np.mean(
                                    [
                                        wasserstein_distance(a[:, j], b[:, j])
                                        for j in range(len(zcols))
                                    ]
                                )
                            )
                            ot_rows.append(
                                {
                                    "population": population,
                                    "analysis_level": analysis_level,
                                    "state": state,
                                    "seed": seed,
                                    "comparison": comparison,
                                    "left_library": left_library,
                                    "right_library": right_library,
                                    "n_cells_per_library": n,
                                    "metric": "coordinatewise_Wasserstein",
                                    "regularization": np.nan,
                                    "value": coordinate_w1,
                                }
                            )
                            if ot is None:
                                continue
                            cost = ot.dist(a, b, metric="sqeuclidean")
                            positive = cost[cost > 0]
                            cost_scale = float(np.median(positive)) if positive.size else 1.0
                            weights = np.repeat(1.0 / n, n)
                            for multiplier in regularization_grid:
                                sinkhorn_cost = ot.sinkhorn2(
                                    weights,
                                    weights,
                                    cost,
                                    reg=max(multiplier * cost_scale, 1e-6),
                                    method="sinkhorn_log",
                                    numItermax=3000,
                                    stopThr=1e-7,
                                    warn=False,
                                )
                                ot_rows.append(
                                    {
                                        "population": population,
                                        "analysis_level": analysis_level,
                                        "state": state,
                                        "seed": seed,
                                        "comparison": comparison,
                                        "left_library": left_library,
                                        "right_library": right_library,
                                        "n_cells_per_library": n,
                                        "metric": "Sinkhorn_root_cost",
                                        "regularization": multiplier,
                                        "value": float(np.sqrt(max(float(sinkhorn_cost), 0.0))),
                                    }
                                )
    pd.DataFrame(geometry_rows).to_csv(out / "LATENT_GEOMETRY.tsv", sep="\t", index=False)
    pd.DataFrame(ot_rows).to_csv(out / "OPTIMAL_TRANSPORT_RESULTS.tsv", sep="\t", index=False)
    ot_frame = pd.DataFrame(ot_rows)
    if not ot_frame.empty:
        sensitivity = (
            ot_frame.groupby(
                [
                    "population",
                    "analysis_level",
                    "state",
                    "seed",
                    "comparison",
                    "metric",
                    "regularization",
                ],
                observed=True,
                dropna=False,
            )["value"]
            .agg(["mean", "median", "std", "count"])
            .reset_index()
        )
    else:
        sensitivity = pd.DataFrame()
    sensitivity.to_csv(out / "OT_SENSITIVITY.tsv", sep="\t", index=False)
    pd.DataFrame(loo_rows).to_csv(out / "OT_LOO_RESULTS.tsv", sep="\t", index=False)
    pd.DataFrame(excluded_contexts).to_csv(
        out / "SUBTYPE_COVERAGE_EXCLUSIONS.tsv", sep="\t", index=False
    )
    geometry = pd.DataFrame(geometry_rows)
    granulosa = geometry.loc[
        geometry["population"].eq("Granulosa")
        & geometry["analysis_level"].eq("broad")
    ] if not geometry.empty else pd.DataFrame()
    if not granulosa.empty:
        ratio_text = f"{granulosa['distance_ratio_OT_over_OC'].median():.3f}"
        cosine_text = f"{granulosa['aging_treatment_cosine'].median():.3f}"
        residual_text = f"{granulosa['treatment_residual_ratio'].median():.3f}"
    else:
        ratio_text = cosine_text = residual_text = "NA"
    (out / "LATENT_GEOMETRY_REPORT_CN.md").write_text(
        "# Stage 18 潜在空间几何与分布距离\n\n"
        "当前实现使用每个library均衡抽样后的scVI latent。所有centroid先在library内聚合，再以library等权形成组中心；细胞层距离只作为技术分布诊断。\n\n"
        f"Granulosa broad层面跨seed的中位 distance(OT,Y)/distance(OC,Y)={ratio_text}，中位 aging-treatment cosine={cosine_text}，治疗向量正交残差占比={residual_text}。这些数值需要与leave-one-library-out和Sinkhorn正则化敏感性共同解读。\n\n"
        + (
            "已使用POT计算三个正则化强度的library-pair Sinkhorn距离。\n"
            if ot is not None
            else "POT不可用，因此只保留coordinatewise Wasserstein并明确降级。\n"
        )
        + "细胞bootstrap或大量library-pair不能被解释为额外生物学重复；正式推断仍以9个library及既有精确置换为边界。\n",
        encoding="utf-8",
    )
    _write_checkpoint(
        out / "CHECKPOINT.json",
        "LATENT_GEOMETRY_COMPLETE",
        n_geometry=len(geometry_rows),
        n_ot=len(ot_rows),
        sinkhorn_available=ot is not None,
        regularization_grid=regularization_grid,
    )
    _update_run_state(
        stage_root,
        "04_latent_geometry_ot",
        "complete",
        n_geometry=len(geometry_rows),
        n_ot=len(ot_rows),
        sinkhorn_available=ot is not None,
    )


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
        for model_name, estimator in [
            ("ridge", Ridge(alpha=10.0)),
            (
                "elastic_net",
                ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=50000, tol=1e-3),
            ),
        ]:
            preds = np.full(len(y), np.nan)
            for i in range(len(y)):
                train = np.arange(len(y)) != i
                model = make_pipeline(StandardScaler(), estimator)
                model.fit(X[train], y[train]); preds[i] = model.predict(X[i : i + 1])[0]
                loso_rows.append(
                    {
                        "validation_scope": "external_sample_LOSO",
                        "population": population,
                        "model": model_name,
                        "held_out_sample": pmeta.iloc[i]["sample_id"],
                        "observed_age_months": y[i],
                        "predicted_age_score": preds[i],
                        "n_training_samples": int(train.sum()),
                    }
                )
            rows.append(
                {
                    "population": population,
                    "model": model_name,
                    "status": "trained",
                    "n_external_samples": len(y),
                    "n_external_studies": 1,
                    "n_features": len(features),
                    "loso_mae_months": float(mean_absolute_error(y, preds)),
                    "loso_r2": float(r2_score(y, preds)),
                    "decision": "exploratory_only_single_public_study",
                }
            )
            model = make_pipeline(StandardScaler(), estimator); model.fit(X, y)
            if not internal.empty:
                use = features
                ipred = model.predict(np.log2(internal[use].div(internal[use].sum(axis=1), axis=0) * 1e6 + 0.5).to_numpy())
                for library, score in zip(internal.index, ipred):
                    score_rows.append({"population": population, "model": model_name, "library_id": library, "group": str(library).split("_", 1)[0], "external_age_score": score, "n_features": len(use)})
        rows.extend(
            [
                {
                    "population": population,
                    "model": "frozen_external_age_program",
                    "status": "available_from_stage15",
                    "n_external_samples": len(y),
                    "n_external_studies": 1,
                    "n_features": len(features),
                    "decision": "retain_as_simple_frozen_baseline",
                },
                {
                    "population": population,
                    "model": "XGBoost",
                    "status": "skipped",
                    "n_external_samples": len(y),
                    "n_external_studies": 1,
                    "decision": "fewer_than_30_independent_samples",
                },
                {
                    "population": population,
                    "model": "scVI_latent_shallow_predictor",
                    "status": "skipped",
                    "n_external_samples": len(y),
                    "n_external_studies": 1,
                    "decision": "single_public_study_no_leave_study_out_test",
                },
                {
                    "population": population,
                    "model": "attention_MIL",
                    "status": "skipped",
                    "n_external_samples": len(y),
                    "n_external_studies": 1,
                    "decision": "deep_model_gate_failed_sample_and_study_count",
                },
            ]
        )
    score_frame = pd.DataFrame(score_rows)
    contrast_rows: list[dict[str, Any]] = []
    if not score_frame.empty:
        for (population, model_name), block in score_frame.groupby(
            ["population", "model"], observed=True
        ):
            means = block.groupby("group", observed=True)["external_age_score"].mean()
            for contrast, numerator, denominator in [
                ("OC_vs_Y", "OC", "Y"),
                ("OT_vs_OC", "OT", "OC"),
                ("OT_vs_Y", "OT", "Y"),
            ]:
                contrast_rows.append(
                    {
                        "population": population,
                        "model": model_name,
                        "contrast": contrast,
                        "effect": _safe_float(means.get(numerator))
                        - _safe_float(means.get(denominator)),
                        "n_libraries_per_group": 3,
                    }
                )
            for excluded in block["library_id"].astype(str):
                kept = block.loc[block["library_id"].astype(str).ne(excluded)]
                kept_means = kept.groupby("group", observed=True)[
                    "external_age_score"
                ].mean()
                loso_rows.append(
                    {
                        "validation_scope": "internal_score_leave_one_library_out",
                        "population": population,
                        "model": model_name,
                        "held_out_sample": excluded,
                        "aging_effect_OC_minus_Y": _safe_float(kept_means.get("OC"))
                        - _safe_float(kept_means.get("Y")),
                        "treatment_effect_OT_minus_OC": _safe_float(
                            kept_means.get("OT")
                        )
                        - _safe_float(kept_means.get("OC")),
                        "residual_effect_OT_minus_Y": _safe_float(kept_means.get("OT"))
                        - _safe_float(kept_means.get("Y")),
                    }
                )
    pd.DataFrame(rows).to_csv(out / "MODEL_BENCHMARK.tsv", sep="\t", index=False)
    score_frame.to_csv(out / "EXTERNAL_AGE_SCORE_LIBRARY.tsv", sep="\t", index=False)
    pd.DataFrame(contrast_rows).to_csv(
        out / "EXTERNAL_AGE_SCORE_CONTRASTS.tsv", sep="\t", index=False
    )
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
        ContrastiveVI = scvi.external.ContrastiveVI
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
        full = ad.read_h5ad(
            root / config["deep_dive_stage15"]["input_object"], backed="r"
        )
        obs = full.obs
        broad_key = str(config["deep_dive_stage15"]["broad_key"])
        subtype_key = str(config["deep_dive_stage15"]["subtype_key"])
        tier_key = str(config["deep_dive_stage15"].get("tier_key", ""))
        genes = _select_hvg_genes(full, n_genes=1500)
        base_mask = (
            obs[broad_key].astype(str).eq("Granulosa")
            & obs["group"].astype(str).isin(["OC", "OT"])
        )
        if tier_key in obs:
            base_mask &= obs[tier_key].astype(str).eq("Tier1_primary")

        contexts: list[tuple[str, np.ndarray]] = [("Granulosa_all", base_mask.to_numpy())]
        if subtype_key in obs:
            candidate_obs = obs.loc[base_mask, [subtype_key, "library_id"]].copy()
            coverage = candidate_obs.groupby(
                [subtype_key, "library_id"], observed=True
            ).size().unstack(fill_value=0)
            eligible_libraries = [
                library for library in LIBRARIES if library.startswith(("OC_", "OT_"))
            ]
            eligible = coverage.reindex(columns=eligible_libraries, fill_value=0)
            eligible = eligible.loc[eligible.min(axis=1).ge(50)]
            for subtype in eligible.min(axis=1).sort_values(ascending=False).head(3).index:
                contexts.append(
                    (
                        str(subtype),
                        (
                            base_mask
                            & obs[subtype_key].astype(str).eq(str(subtype))
                        ).to_numpy(),
                    )
                )

        gate = pd.DataFrame(
            [
                {
                    "population": "Granulosa",
                    "gate_passed": True,
                    "basis": "Stage15 exact permutation, external age axes, state modules and subtype/composition review",
                    "limitation": "n=3 libraries per group; candidate representation only",
                }
            ]
        )
        gate.to_csv(out / "CONTRASTIVEVI_GATE.tsv", sep="\t", index=False)

        counts_layer = str(config["deep_dive_stage15"].get("counts_layer", "counts"))
        rng = np.random.default_rng(20260919)
        model_rows: list[dict[str, Any]] = []
        score_rows: list[pd.DataFrame] = []
        stability_rows: list[dict[str, Any]] = []
        gene_rows: list[dict[str, Any]] = []

        for context_name, context_mask in contexts:
            selected: list[int] = []
            for _, index_names in obs.loc[context_mask].groupby(
                "library_id", observed=True
            ).groups.items():
                positions = full.obs_names.get_indexer(np.asarray(index_names, dtype=str))
                positions = positions[positions >= 0]
                if len(positions) > 300:
                    positions = rng.choice(positions, 300, replace=False)
                selected.extend(positions.tolist())
            data = full[selected, genes].to_memory()
            if counts_layer not in data.layers:
                raise KeyError(
                    f"raw count layer is required for contrastiveVI: {counts_layer}"
                )
            if counts_layer != "counts":
                data.layers["counts"] = data.layers[counts_layer].copy()
            background_idx = np.where(data.obs["group"].astype(str).eq("OC"))[0]
            target_idx = np.where(data.obs["group"].astype(str).eq("OT"))[0]
            if len(background_idx) < 100 or len(target_idx) < 100:
                stability_rows.append(
                    {
                        "context": context_name,
                        "run_type": "skipped",
                        "reason": "insufficient_balanced_cells",
                    }
                )
                continue
            ContrastiveVI.setup_anndata(data, layer="counts")
            context_scores: list[pd.DataFrame] = []
            for seed in [20260919, 20260920, 20260921]:
                scvi.settings.seed = seed
                model = ContrastiveVI(
                    data, n_background_latent=5, n_salient_latent=5
                )
                model.train(
                    background_indices=background_idx.tolist(),
                    target_indices=target_idx.tolist(),
                    max_epochs=60,
                    batch_size=256,
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices=1,
                    early_stopping=True,
                    early_stopping_patience=10,
                    enable_progress_bar=False,
                )
                model_dir = out / f"model_{context_name}_{seed}"
                model.save(model_dir, overwrite=True)
                history = _scvi_history_summary(
                    model,
                    out / f"CONTRASTIVEVI_TRAINING_HISTORY_{context_name}_{seed}.tsv",
                )
                salient = model.get_latent_representation(
                    representation_kind="salient"
                )
                frame = pd.DataFrame(
                    salient,
                    columns=[f"salient_{i+1}" for i in range(salient.shape[1])],
                )
                frame.insert(
                    0, "library_id", data.obs["library_id"].astype(str).to_numpy()
                )
                frame.insert(1, "group", data.obs["group"].astype(str).to_numpy())
                frame["salient_norm"] = np.linalg.norm(salient, axis=1)
                frame["seed"] = seed
                frame["context"] = context_name
                summary = (
                    frame.groupby(
                        ["context", "library_id", "group", "seed"], observed=True
                    )
                    .mean(numeric_only=True)
                    .reset_index()
                )
                summary["run_type"] = "full"
                summary["excluded_library"] = ""
                context_scores.append(summary)
                score_rows.append(summary)
                group_means = summary.groupby("group", observed=True)[
                    "salient_norm"
                ].mean()
                stability_rows.append(
                    {
                        "context": context_name,
                        "run_type": "full_seed",
                        "seed": seed,
                        "excluded_library": "",
                        "n_cells": data.n_obs,
                        "n_genes": data.n_vars,
                        "salient_norm_effect_OT_minus_OC": _safe_float(
                            group_means.get("OT")
                        )
                        - _safe_float(group_means.get("OC")),
                        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                        **history,
                    }
                )
                model_rows.append(
                    {
                        "context": context_name,
                        "seed": seed,
                        "n_cells": data.n_obs,
                        "n_genes": data.n_vars,
                        "n_background": len(background_idx),
                        "n_target": len(target_idx),
                        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                        "model_dir": str(model_dir.relative_to(root)),
                        **history,
                    }
                )

                counts = data.layers["counts"]
                dense = counts.toarray() if sparse.issparse(counts) else np.asarray(counts)
                totals = dense.sum(axis=1, keepdims=True)
                logged = np.log1p(dense / np.maximum(totals, 1.0) * 1e4)
                expression_centered = logged - logged.mean(axis=0, keepdims=True)
                latent_centered = salient - salient.mean(axis=0, keepdims=True)
                numerator = latent_centered.T @ expression_centered
                denominator = np.sqrt(
                    (latent_centered**2).sum(axis=0)[:, None]
                    * (expression_centered**2).sum(axis=0)[None, :]
                )
                correlations = numerator / np.maximum(denominator, 1e-12)
                max_abs = np.max(np.abs(correlations), axis=0)
                best_factor = np.argmax(np.abs(correlations), axis=0)
                ranks = pd.Series(-max_abs).rank(method="min").astype(int).to_numpy()
                for gene_index, gene in enumerate(data.var_names.astype(str)):
                    factor = int(best_factor[gene_index])
                    gene_rows.append(
                        {
                            "context": context_name,
                            "seed": seed,
                            "gene": gene,
                            "best_salient_factor": factor + 1,
                            "correlation": float(correlations[factor, gene_index]),
                            "max_abs_correlation": float(max_abs[gene_index]),
                            "within_seed_rank": int(ranks[gene_index]),
                        }
                    )
                del model

            if context_scores:
                all_context_scores = pd.concat(context_scores, ignore_index=True)
                for seed_a, seed_b in combinations(
                    sorted(all_context_scores["seed"].unique()), 2
                ):
                    left = all_context_scores.loc[
                        all_context_scores["seed"].eq(seed_a)
                    ].set_index("library_id")
                    right = all_context_scores.loc[
                        all_context_scores["seed"].eq(seed_b)
                    ].set_index("library_id")
                    common = left.index.intersection(right.index)
                    correlation = spearmanr(
                        left.loc[common, "salient_norm"],
                        right.loc[common, "salient_norm"],
                    ).statistic
                    stability_rows.append(
                        {
                            "context": context_name,
                            "run_type": "seed_pair_stability",
                            "seed": f"{seed_a}:{seed_b}",
                            "excluded_library": "",
                            "library_score_spearman": float(correlation),
                        }
                    )

            # Leave-one-library-out sensitivity uses one fixed seed and reports
            # invariant salient norm rather than comparing rotated latent axes.
            for excluded in sorted(data.obs["library_id"].astype(str).unique()):
                keep = np.where(data.obs["library_id"].astype(str).ne(excluded))[0]
                loo = data[keep].copy()
                ContrastiveVI.setup_anndata(loo, layer="counts")
                loo_background = np.where(loo.obs["group"].astype(str).eq("OC"))[0]
                loo_target = np.where(loo.obs["group"].astype(str).eq("OT"))[0]
                scvi.settings.seed = 20260919
                loo_model = ContrastiveVI(
                    loo, n_background_latent=5, n_salient_latent=5
                )
                loo_model.train(
                    background_indices=loo_background.tolist(),
                    target_indices=loo_target.tolist(),
                    max_epochs=40,
                    batch_size=256,
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices=1,
                    early_stopping=True,
                    early_stopping_patience=8,
                    enable_progress_bar=False,
                )
                loo_salient = loo_model.get_latent_representation(
                    representation_kind="salient"
                )
                loo_frame = pd.DataFrame(
                    {
                        "context": context_name,
                        "library_id": loo.obs["library_id"].astype(str).to_numpy(),
                        "group": loo.obs["group"].astype(str).to_numpy(),
                        "seed": 20260919,
                        "salient_norm": np.linalg.norm(loo_salient, axis=1),
                        "run_type": "leave_one_library_out",
                        "excluded_library": excluded,
                    }
                )
                loo_summary = (
                    loo_frame.groupby(
                        [
                            "context",
                            "library_id",
                            "group",
                            "seed",
                            "run_type",
                            "excluded_library",
                        ],
                        observed=True,
                    )["salient_norm"]
                    .mean()
                    .reset_index()
                )
                score_rows.append(loo_summary)
                means = loo_summary.groupby("group", observed=True)[
                    "salient_norm"
                ].mean()
                stability_rows.append(
                    {
                        "context": context_name,
                        "run_type": "leave_one_library_out",
                        "seed": 20260919,
                        "excluded_library": excluded,
                        "salient_norm_effect_OT_minus_OC": _safe_float(means.get("OT"))
                        - _safe_float(means.get("OC")),
                    }
                )
                del loo_model, loo
            del data
        full.file.close()

        scores = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
        stability = pd.DataFrame(stability_rows)
        genes_raw = pd.DataFrame(gene_rows)
        evidence = _read_tsv(stage_root / "01_evidence_matrix/EVIDENCE_MATRIX.tsv")
        supported_genes = set()
        if not evidence.empty:
            supported = evidence.loc[
                evidence["population"].astype(str).eq("Granulosa")
                & evidence["candidate_type"].astype(str).eq("gene")
                & evidence["directional_rescue"].astype(bool)
            ]
            supported_genes = {
                str(value).rsplit(":", 1)[-1] for value in supported["candidate_id"]
            }
        program_rows: list[dict[str, Any]] = []
        if not genes_raw.empty:
            for (context_name, gene), block in genes_raw.groupby(
                ["context", "gene"], observed=True
            ):
                n_top100 = int(block["within_seed_rank"].le(100).sum())
                loo_effects = stability.loc[
                    stability["context"].eq(context_name)
                    & stability["run_type"].eq("leave_one_library_out"),
                    "salient_norm_effect_OT_minus_OC",
                ].dropna()
                loo_sign_fraction = float((loo_effects > 0).mean()) if len(loo_effects) else np.nan
                independent = gene in supported_genes
                stable_seed = n_top100 >= 2
                retained = bool(
                    stable_seed
                    and independent
                    and np.isfinite(loo_sign_fraction)
                    and loo_sign_fraction >= 0.8
                )
                program_rows.append(
                    {
                        "context": context_name,
                        "gene": gene,
                        "median_max_abs_correlation": float(
                            block["max_abs_correlation"].median()
                        ),
                        "n_seeds_top100": n_top100,
                        "cross_seed_stable": stable_seed,
                        "loo_positive_effect_fraction": loo_sign_fraction,
                        "independent_gene_level_support": independent,
                        "retained_candidate": retained,
                        "interpretation": "correlation_with_salient_latent_not_causal_loading",
                    }
                )
        programs = pd.DataFrame(program_rows)
        if not programs.empty:
            programs = programs.sort_values(
                ["retained_candidate", "n_seeds_top100", "median_max_abs_correlation"],
                ascending=[False, False, False],
            )
        scores.to_csv(out / "CONTRASTIVEVI_LIBRARY_SCORES.tsv", sep="\t", index=False)
        stability.to_csv(out / "CONTRASTIVEVI_STABILITY.tsv", sep="\t", index=False)
        programs.to_csv(out / "CONTRASTIVEVI_PROGRAM_GENES.tsv", sep="\t", index=False)
        pd.DataFrame(model_rows).to_csv(
            out / "CONTRASTIVEVI_MODEL_RUNS.tsv", sep="\t", index=False
        )
        n_retained = int(programs["retained_candidate"].sum()) if not programs.empty else 0
        (out / "CONTRASTIVEVI_MODEL_CARD.md").write_text(
            "# contrastiveVI model card\n\n"
            "OC为background、OT为target；只使用Granulosa Tier1细胞，并在每个library内等量抽样。模型覆盖Granulosa整体及满足每个OC/OT library至少50个细胞的主要亚型，使用3个完整模型seed和逐library留一敏感性。\n\n"
            "模型不把library作为需消除的batch，因为library与实验组完全嵌套；这也意味着技术library差异仍是明确限制。salient latent以旋转不变的norm聚合到library层面。代表基因来自表达与salient factor的相关性，只是解释性候选，不是因果调控或模型attention。\n",
            encoding="utf-8",
        )
        (out / "CONTRASTIVEVI_REPORT_CN.md").write_text(
            "# Stage 20 contrastiveVI结果\n\n"
            f"共评估{len(contexts)}个Granulosa层级；经过跨seed、leave-one-library-out和独立gene-level证据联合门控后，保留{n_retained}个候选基因。即使通过门控，结果仍只表示与OC/OT contrastive salient表示相关，不能解释为MRJP1直接靶点。\n\n"
            "如果salient norm主要由单个library、测序深度或亚型覆盖驱动，应放弃相应context；详见CONTRASTIVEVI_STABILITY.tsv和library scores。\n",
            encoding="utf-8",
        )
        _write_checkpoint(
            out / "CHECKPOINT.json",
            "CONTRASTIVEVI_COMPLETE",
            n_contexts=len(contexts),
            n_full_models=len(contexts) * 3,
            n_loo_models=len(contexts) * 6,
            n_retained_genes=n_retained,
        )
        _update_run_state(
            stage_root,
            "06_contrastivevi",
            "complete",
            n_contexts=len(contexts),
            n_retained_genes=n_retained,
        )
    except Exception as exc:
        _record_failure(stage_root, "06_contrastivevi_training", exc, logger)
        _write_checkpoint(out / "CHECKPOINT.json", "CONTRASTIVEVI_FAILED", error=repr(exc))
        _update_run_state(stage_root, "06_contrastivevi", "failed", error=repr(exc))


def run_foundation_and_cross_species(config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]) -> None:
    foundation = stage_root / "07_foundation_models"
    sample_registry = _read_tsv(
        stage_root / "02_external_registry/EXTERNAL_SAMPLE_REGISTRY.tsv"
    )
    n_external_samples = int(
        sample_registry.get("sample_id", pd.Series(dtype=str)).astype(str).nunique()
    )
    n_external_studies = int(
        sample_registry.get("dataset", pd.Series(dtype=str)).astype(str).nunique()
    )
    rows = [
        {
            "model": "Geneformer",
            "status": "skipped_by_decision_gate",
            "n_external_samples": n_external_samples,
            "n_external_studies": n_external_studies,
            "pretrained_checkpoint_verified": False,
            "external_leave_study_out_possible": n_external_studies >= 2,
            "reason": "no verified local checkpoint and no multi-study external test; embedding-only benefit cannot be tested against the linear baseline",
        },
        {
            "model": "scGPT",
            "status": "skipped_by_decision_gate",
            "n_external_samples": n_external_samples,
            "n_external_studies": n_external_studies,
            "pretrained_checkpoint_verified": False,
            "external_leave_study_out_possible": n_external_studies >= 2,
            "reason": "no verified local checkpoint and no multi-study external test; do not spend the run budget on an uninterpretable embedding",
        },
    ]
    pd.DataFrame(rows).to_csv(foundation / "FOUNDATION_MODEL_BENCHMARK.tsv", sep="\t", index=False)
    (foundation / "FOUNDATION_MODEL_REPORT_CN.md").write_text(
        "# Stage 21 单细胞基础模型决策门\n\n"
        f"当前可用公共参考只有{n_external_samples}个独立sample、{n_external_studies}个study。"
        "因此无法做留一study外部泛化测试，也没有已验证的本地预训练checkpoint。"
        "本轮不下载Geneformer/scGPT大模型，不在9个library上微调，也不会用无法证明优于Ridge/Elastic Net的embedding增加证据等级。\n",
        encoding="utf-8",
    )
    _write_checkpoint(
        foundation / "CHECKPOINT.json",
        "FOUNDATION_MODELS_SKIPPED_BY_GATE",
        n_external_samples=n_external_samples,
        n_external_studies=n_external_studies,
    )
    _update_run_state(
        stage_root,
        "07_foundation_models",
        "skipped",
        reason="foundation_model_external_generalization_gate_failed",
        n_external_samples=n_external_samples,
        n_external_studies=n_external_studies,
    )

    out = stage_root / "08_cross_species"
    candidate_genes = ["Foxl2", "Hif1a", "Smad3", "Il6", "Il6st", "Fgf2", "Fgfr2", "Col1a1", "Dcn", "Lum", "Cyp19a1", "Star", "Nr5a1"]
    query_time = datetime.now(timezone.utc).isoformat()
    try:
        release_payload = _ensembl_json("https://rest.ensembl.org/info/data")
        releases = release_payload.get("releases", []) or []
        release = str(max(releases)) if releases else "REST_release_unreported"
    except Exception as exc:
        release = f"query_failed:{type(exc).__name__}"
    mapping_rows: list[dict[str, Any]] = []
    for gene in candidate_genes:
        extracted, query_error = _query_one_to_one_orthologs(gene, release)
        if extracted:
            mapping_rows.extend(extracted)
        else:
            mapping_rows.append(
                {
                    "mouse_gene": gene,
                    "human_gene": "",
                    "homology_type": "",
                    "mapping_status": (
                        query_error or "no_one_to_one_ortholog_returned"
                    ),
                    "ensembl_release": release,
                }
            )
    mapping = pd.DataFrame(mapping_rows)
    mapping["query_date_utc"] = query_time
    mapping["mapping_source"] = "Ensembl REST homology endpoint"
    mapping.to_csv(out / "ORTHOLOG_MAPPING.tsv", sep="\t", index=False)
    pd.DataFrame(columns=["dataset", "population", "contrast", "effect", "status"]).to_csv(out / "CROSS_SPECIES_PROJECTION.tsv", sep="\t", index=False)
    n_mapped = int(mapping["mapping_status"].eq("ensembl_one_to_one").sum())
    (out / "CROSS_SPECIES_REPORT_CN.md").write_text(
        "# Stage 22 跨物种投影\n\n"
        f"已使用Ensembl REST（release {release}）对{len(candidate_genes)}个预先限定的小鼠候选基因查询mouse–human一对一同源关系，返回{n_mapped}条一对一映射记录。"
        "GSE202601和其他人卵巢候选数据尚未确认为具备可比的原始counts、独立sample和Granulosa/Stromal标签，因此本轮不做表达投影，也不输出人类有效性结论。\n\n"
        "ORTHOLOG_MAPPING.tsv是版本化映射产物；CROSS_SPECIES_PROJECTION.tsv保留为空表，明确表示投影决策门未通过。\n",
        encoding="utf-8",
    )
    checkpoint_status = (
        "ORTHOLOG_MAPPING_COMPLETE_PROJECTION_SKIPPED"
        if n_mapped
        else "ORTHOLOG_MAPPING_FAILED_PROJECTION_SKIPPED"
    )
    _write_checkpoint(
        out / "CHECKPOINT.json",
        checkpoint_status,
        ensembl_release=release,
        n_candidate_genes=len(candidate_genes),
        n_one_to_one_records=n_mapped,
        projection_reason="no_verified_human_counts",
    )
    _update_run_state(
        stage_root,
        "08_cross_species",
        "complete" if n_mapped else "skipped",
        ortholog_mapping="complete" if n_mapped else "failed",
        projection="skipped",
        reason="no_verified_human_counts",
        n_one_to_one_records=n_mapped,
    )


def run_mechanism_and_synthesis(config: Mapping[str, Any], stage_root: Path, logger: Any, paths: Mapping[str, Path]) -> None:
    result = run_stage16_synthesis(config, stage_root, logger, paths)
    _update_run_state(
        stage_root,
        "09_mechanism_candidates",
        "complete",
        n_candidates=result["n_candidates"],
        n_programs=result["n_programs"],
    )
    _update_run_state(
        stage_root,
        "10_synthesis",
        "complete",
        interpretation=result["geometry_interpretation"],
        n_evidence_candidates=result["evidence_candidates"],
    )


def _manifest(stage_root: Path) -> None:
    rows = []
    for path in sorted(stage_root.rglob("*")):
        if path.is_file() and path.name != "RUN_MANIFEST.tsv":
            rows.append({"path": str(path.relative_to(stage_root)), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    pd.DataFrame(rows).to_csv(stage_root / "RUN_MANIFEST.tsv", sep="\t", index=False)


def run_stage16_ml(
    config: Mapping[str, Any],
    selected_stages: Iterable[str] | None = None,
    force_stages: Iterable[str] | None = None,
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
    forced = set(force_stages or ())
    known = {name for name, _ in stages}
    if selected is not None:
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError(f"Unknown Stage16 stage names: {unknown}")
        stages = [(name, function) for name, function in stages if name in selected]
    unknown_forced = sorted(forced - known)
    if unknown_forced:
        raise ValueError(f"Unknown forced Stage16 stage names: {unknown_forced}")
    state_path = stage_root / "RUN_STATE.json"
    for name, function in stages:
        # Resume safely: completed/skipped stages are immutable inputs for the
        # next stage and should not be retrained after an SSH/session restart.
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        previous = state.get("stages", {}).get(name, {}).get("status")
        if previous in {"complete", "skipped"} and name not in forced:
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
    final_state = json.loads(state_path.read_text(encoding="utf-8"))
    has_failures = any(
        value.get("status") == "failed"
        for key, value in final_state.get("stages", {}).items()
        if key != "overall"
    )
    _update_run_state(
        stage_root,
        "overall",
        "complete_with_failures" if has_failures else "complete",
        next_stage="manual_review_before_claims",
    )
    print("STAGE16_22_ML_COMPLETE")
    print(f"OUTPUT={stage_root.relative_to(paths['root'])}")
    return stage_root
