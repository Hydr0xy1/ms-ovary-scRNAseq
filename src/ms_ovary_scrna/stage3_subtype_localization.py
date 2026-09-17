"""Stage 3: audited subtype-level localization of broad MRJP1 signals.

Formal inference is performed only on raw integer Tier1 pseudobulk counts with
``library_id`` as the replicate.  The annotated single-cell object is opened
read-only and never modified.  Harmony coordinates and per-cell tests are not
used for differential expression.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .de_stage1 import prefilter_genes
from .de_stage1_5 import ALL_LIBRARIES, UNIFIED_CONTRASTS, _capture_operation
from .de_stage1_6 import _fit_permutation, enumerate_permutation_assignments
from .pathway_stage2 import (
    GSEA_MAX_SIZE,
    GSEA_MIN_SIZE,
    GSEA_PERMUTATIONS,
    GSEA_SEED,
    deduplicate_rank_table,
    exact_permutation_calibration,
    parse_gmt,
    pathway_effect_geometry,
    pathway_member_overlap,
)
from .project import project_paths, require_compute_resources, setup_logging
from .pseudobulk import safe_name

CONTRASTS = ("OC_vs_Y", "OT_vs_OC", "OT_vs_Y")
REQUIRED_OBS = (
    "cell_type_broad_v2",
    "cell_type_subtype_v2",
    "cell_state_v2",
    "annotation_confidence_v2",
    "analysis_tier_v2",
    "doublet_concern_v2",
    "library_id",
    "group",
)
UNSTABLE_LABEL_TOKENS = ("unresolved", "mixed", "uncertain", "low_complexity")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_subtype_eligibility(
    coverage: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    min_primary: int = 50,
    min_sensitivity: int = 20,
    min_confident_fraction: float = 0.80,
) -> pd.DataFrame:
    """Assign fixed, coverage-driven subtype analysis tiers.

    Primary requires all nine libraries to contain at least ``min_primary``
    Tier1 cells.  Sensitivity uses Tier1+Tier2 cells but never silently promotes
    them into the formal Tier1 pseudobulk analysis.
    """

    required = {
        "broad_population",
        "subtype",
        "library_id",
        "n_cells_tier1",
        "n_cells_tier1_or_tier2",
    }
    missing = required - set(coverage.columns)
    if missing:
        raise ValueError(f"Coverage lacks columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    inv = inventory.set_index(["broad_population", "subtype"])
    for (broad, subtype), sub in coverage.groupby(
        ["broad_population", "subtype"], observed=True, sort=True
    ):
        sub = sub.set_index("library_id").reindex(ALL_LIBRARIES)
        complete_grid = not sub["n_cells_tier1"].isna().any()
        tier1_min = int(sub["n_cells_tier1"].fillna(0).min())
        tier12_min = int(sub["n_cells_tier1_or_tier2"].fillna(0).min())
        details = inv.loc[(broad, subtype)]
        label_stable = not any(token in str(subtype).lower() for token in UNSTABLE_LABEL_TOKENS)
        confidence_ok = float(details["confident_identity_fraction"]) >= min_confident_fraction
        doublet_ok = float(details["doublet_concern_fraction"]) < 0.05
        identity_stable = bool(label_stable and confidence_ok and doublet_ok)
        if complete_grid and tier1_min >= min_primary and identity_stable:
            status = "Primary_DE_ready"
            reason = "all_9_libraries_have_at_least_50_Tier1_cells_and_identity_is_stable"
        elif complete_grid and tier12_min >= min_sensitivity and identity_stable:
            status = "Sensitivity_only"
            reason = "all_9_libraries_have_at_least_20_Tier1_or_Tier2_cells_but_primary_rule_fails"
        else:
            status = "Descriptive_only"
            reasons: list[str] = []
            if tier12_min < min_sensitivity:
                reasons.append("one_or_more_libraries_below_20_cells")
            if not label_stable:
                reasons.append("unstable_or_low_complexity_identity")
            if not confidence_ok:
                reasons.append("confidence_fraction_below_fixed_threshold")
            if not doublet_ok:
                reasons.append("strong_mixed_or_doublet_concern")
            reason = ";".join(reasons) or "coverage_or_identity_not_primary_ready"
        rows.append(
            {
                "broad_population": broad,
                "subtype": subtype,
                "eligibility": status,
                "reason": reason,
                "n_libraries_present": int((sub["n_cells_all"].fillna(0) > 0).sum()),
                "min_tier1_cells_per_library": tier1_min,
                "min_tier1_or_tier2_cells_per_library": tier12_min,
                "identity_stable": identity_stable,
                "confident_identity_fraction": float(details["confident_identity_fraction"]),
                "doublet_concern_fraction": float(details["doublet_concern_fraction"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["broad_population", "eligibility", "subtype"], kind="stable"
    )


def _inventory_and_counts(
    h5ad_path: Path,
    target_broad: Sequence[str],
    *,
    counts_layer: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build inventory, coverage and Tier1 pseudobulk without densifying cells × genes."""

    import anndata as ad
    from scipy import sparse

    adata = ad.read_h5ad(h5ad_path)
    try:
        missing = set(REQUIRED_OBS) - set(adata.obs.columns)
        if missing:
            raise KeyError(f"Input object lacks columns: {sorted(missing)}")
        if counts_layer not in adata.layers:
            raise KeyError(f"Input object lacks raw count layer: {counts_layer}")
        matrix = adata.layers[counts_layer]
        if not sparse.issparse(matrix):
            raise ValueError("Stage 3 requires a sparse raw-count layer")
        matrix = matrix.tocsr()
        obs = adata.obs
        broad_values = obs["cell_type_broad_v2"].astype(str)
        subtype_values = obs["cell_type_subtype_v2"].astype(str)
        library_values = obs["library_id"].astype(str)
        group_values = obs["group"].astype(str)
        tier_values = obs["analysis_tier_v2"].astype(str)
        confidence_values = obs["annotation_confidence_v2"].astype(str)
        doublet_values = obs["doublet_concern_v2"].astype(str).str.lower().eq("true")
        state_values = obs["cell_state_v2"].astype(str)

        inventory_rows: list[dict[str, Any]] = []
        coverage_rows: list[dict[str, Any]] = []
        count_records: list[np.ndarray] = []
        count_index: list[tuple[str, str, str]] = []
        target_mask = broad_values.isin(list(target_broad)).to_numpy()
        pairs = (
            pd.DataFrame(
                {
                    "broad_population": broad_values[target_mask].to_numpy(),
                    "subtype": subtype_values[target_mask].to_numpy(),
                }
            )
            .drop_duplicates()
            .sort_values(["broad_population", "subtype"], kind="stable")
        )
        for broad, subtype in pairs.itertuples(index=False):
            subtype_mask = (broad_values.eq(broad) & subtype_values.eq(subtype)).to_numpy()
            sub_n = int(subtype_mask.sum())
            states = state_values[subtype_mask].value_counts()
            inventory_rows.append(
                {
                    "broad_population": broad,
                    "subtype": subtype,
                    "n_cells": sub_n,
                    "n_tier1": int(
                        (subtype_mask & tier_values.eq("Tier1_primary").to_numpy()).sum()
                    ),
                    "n_tier2": int(
                        (subtype_mask & tier_values.eq("Tier2_sensitivity").to_numpy()).sum()
                    ),
                    "dominant_cell_state": str(states.index[0]) if len(states) else "None",
                    "confident_identity_fraction": float(
                        confidence_values[subtype_mask].isin(["High", "Medium"]).mean()
                    ),
                    "doublet_concern_fraction": float(doublet_values[subtype_mask].mean()),
                }
            )
            for library in ALL_LIBRARIES:
                lib_mask = subtype_mask & library_values.eq(library).to_numpy()
                tier1_mask = lib_mask & tier_values.eq("Tier1_primary").to_numpy()
                tier12_mask = lib_mask & tier_values.isin(
                    ["Tier1_primary", "Tier2_sensitivity"]
                ).to_numpy()
                if tier1_mask.any():
                    summed = np.asarray(matrix[tier1_mask, :].sum(axis=0)).ravel()
                else:
                    summed = np.zeros(adata.n_vars, dtype=np.int64)
                if not np.isfinite(summed).all() or (summed < 0).any():
                    raise ValueError(f"Invalid raw counts for {subtype}/{library}")
                rounded = np.rint(summed).astype(np.int64)
                if not np.array_equal(summed, rounded):
                    raise ValueError("Raw pseudobulk contains non-integer values")
                coverage_rows.append(
                    {
                        "broad_population": broad,
                        "subtype": subtype,
                        "library_id": library,
                        "group": str(group_values[library_values.eq(library)].iloc[0]),
                        "n_cells_all": int(lib_mask.sum()),
                        "n_cells_tier1": int(tier1_mask.sum()),
                        "n_cells_tier1_or_tier2": int(tier12_mask.sum()),
                        "total_umi_tier1": int(rounded.sum()),
                        "detected_genes_tier1": int((rounded > 0).sum()),
                    }
                )
                count_records.append(rounded)
                count_index.append((broad, subtype, library))
        counts = pd.DataFrame(
            count_records,
            index=pd.MultiIndex.from_tuples(
                count_index, names=["broad_population", "subtype", "library_id"]
            ),
            columns=adata.var_names.astype(str),
        )
        return pd.DataFrame(inventory_rows), pd.DataFrame(coverage_rows), counts
    finally:
        del adata


def _fit_subtype_model(
    subtype: str,
    broad: str,
    counts: pd.DataFrame,
    *,
    min_count: int,
    min_samples: int,
    n_cpus: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    counts = counts.reindex(ALL_LIBRARIES)
    filtered = prefilter_genes(counts, min_count=min_count, min_samples=min_samples)
    metadata = pd.DataFrame(index=ALL_LIBRARIES)
    metadata.index.name = "library_id"
    metadata["group"] = pd.Categorical(
        [library.split("_", 1)[0] for library in ALL_LIBRARIES],
        categories=["Y", "OC", "OT"],
        ordered=True,
    )
    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        counts=filtered.astype(np.int64),
        metadata=metadata,
        design="~group",
        refit_cooks=True,
        inference=inference,
        low_memory=False,
    )
    core = _capture_operation(dds.deseq2)
    records: list[pd.DataFrame] = []
    warning_rows: list[dict[str, Any]] = []
    for spec in UNIFIED_CONTRASTS:
        stats = DeseqStats(
            dds,
            contrast=["group", spec.numerator, spec.denominator],
            alpha=0.05,
            cooks_filter=True,
            independent_filter=True,
            inference=inference,
            quiet=True,
        )
        captured = _capture_operation(stats.summary)
        result = stats.results_df.reset_index().rename(columns={"index": "gene"})
        result.insert(0, "contrast", spec.name)
        result.insert(0, "subtype", subtype)
        result.insert(0, "broad_population", broad)
        records.append(result)
        warning_rows.append(
            {
                "broad_population": broad,
                "subtype": subtype,
                "contrast": spec.name,
                "n_tested_genes": int(len(result)),
                "warning_count": len(core.warning_lines) + len(captured.warning_lines),
                "warnings": " | ".join(core.warning_lines + captured.warning_lines) or "None",
            }
        )
    combined = pd.concat(records, ignore_index=True)
    wide = combined.pivot(index="gene", columns="contrast", values="log2FoldChange")
    error = (
        wide["OT_vs_Y"] - (wide["OC_vs_Y"] + wide["OT_vs_OC"])
    ).abs().max()
    if float(error) > 1e-8:
        raise RuntimeError(f"LFC algebra failed for {subtype}: {error}")
    return combined, pd.DataFrame(warning_rows)


def _gsea_one(
    ranking: pd.DataFrame,
    gene_sets: Mapping[str, Sequence[str]],
    *,
    broad: str,
    subtype: str,
    contrast: str,
    threads: int,
    permutations: int,
    min_size: int,
    max_size: int,
    seed: int,
) -> pd.DataFrame:
    import gseapy as gp

    ordered = ranking[["canonical_mouse_symbol", "stat"]].copy()
    pre = gp.prerank(
        rnk=ordered,
        gene_sets=dict(gene_sets),
        permutation_num=permutations,
        min_size=min_size,
        max_size=max_size,
        weight=1.0,
        ascending=False,
        threads=threads,
        seed=seed,
        outdir=None,
        no_plot=True,
        verbose=False,
    )
    result = pre.res2d.rename(
        columns={
            "Term": "pathway",
            "NOM p-val": "nominal_p",
            "FDR q-val": "FDR_q",
            "FWER p-val": "FWER_p",
            "Lead_genes": "leading_edge",
        }
    )
    if "leading_edge" not in result:
        result["leading_edge"] = ""
    overlap = pathway_member_overlap(gene_sets, ordered["canonical_mouse_symbol"])
    result = result.merge(overlap, on="pathway", how="left", validate="one_to_one")
    result.insert(0, "contrast", contrast)
    result.insert(0, "subtype", subtype)
    result.insert(0, "broad_population", broad)
    for column in ("ES", "NES", "nominal_p", "FDR_q", "FWER_p"):
        result[column] = pd.to_numeric(result[column], errors="raise")
    return result


def _run_gsea_grid(
    de: pd.DataFrame,
    mapping: pd.DataFrame,
    gene_sets: Mapping[str, Sequence[str]],
    *,
    workers: int,
    threads: int,
    permutations: int,
    min_size: int,
    max_size: int,
    seed: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks: list[tuple[str, str, str, pd.DataFrame]] = []
    resolutions: list[pd.DataFrame] = []
    for (broad, subtype, contrast), table in de.groupby(
        ["broad_population", "subtype", "contrast"], observed=True, sort=True
    ):
        selected, resolution = deduplicate_rank_table(table, mapping)
        if not resolution.empty:
            resolution.insert(0, "contrast", contrast)
            resolution.insert(0, "subtype", subtype)
            resolution.insert(0, "broad_population", broad)
            resolutions.append(resolution)
        tasks.append((broad, subtype, contrast, selected))
    results: list[pd.DataFrame] = []
    max_jobs = min(workers, max(1, (os.cpu_count() or 1) // max(threads, 1)))
    for start in range(0, len(tasks), max_jobs):
        batch = tasks[start : start + max_jobs]
        with ProcessPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(
                    _gsea_one,
                    ranking,
                    gene_sets,
                    broad=broad,
                    subtype=subtype,
                    contrast=contrast,
                    threads=threads,
                    permutations=permutations,
                    min_size=min_size,
                    max_size=max_size,
                    seed=seed,
                ): (subtype, contrast)
                for broad, subtype, contrast, ranking in batch
            }
            for future in as_completed(futures):
                subtype, contrast = futures[future]
                frame = future.result()
                results.append(frame)
                logger.info("Subtype GSEA complete: %s/%s sets=%d", subtype, contrast, len(frame))
    return pd.concat(results, ignore_index=True), (
        pd.concat(resolutions, ignore_index=True) if resolutions else pd.DataFrame()
    )


def build_localization_table(
    hallmark: pd.DataFrame,
    broad_evidence: pd.DataFrame,
) -> pd.DataFrame:
    predefined = broad_evidence[
        broad_evidence["pathway_evidence_level"].isin(
            ["Strong_multi_metric_reversal", "Permutation_supported_reversal"]
        )
    ][["population", "pathway", "pathway_evidence_level"]].rename(
        columns={"population": "broad_population", "pathway_evidence_level": "broad_evidence_level"}
    )
    selected = hallmark.merge(predefined, on=["broad_population", "pathway"], how="inner")
    rows: list[dict[str, Any]] = []
    for (broad, subtype, pathway), sub in selected.groupby(
        ["broad_population", "subtype", "pathway"], observed=True, sort=True
    ):
        indexed = sub.set_index("contrast")
        if not set(CONTRASTS).issubset(indexed.index):
            continue
        aging = indexed.loc["OC_vs_Y"]
        treatment = indexed.loc["OT_vs_OC"]
        residual = indexed.loc["OT_vs_Y"]
        opposite = np.sign(aging["NES"]) == -np.sign(treatment["NES"])
        rows.append(
            {
                "broad_population": broad,
                "subtype": subtype,
                "pathway": pathway,
                "broad_evidence_level": str(aging["broad_evidence_level"]),
                "aging_NES": float(aging["NES"]),
                "aging_FDR": float(aging["FDR_q"]),
                "treatment_NES": float(treatment["NES"]),
                "treatment_FDR": float(treatment["FDR_q"]),
                "residual_NES": float(residual["NES"]),
                "residual_FDR": float(residual["FDR_q"]),
                "direction_opposite": bool(opposite),
                "localized_GSEA_support": bool(
                    aging["FDR_q"] < 0.05 and treatment["FDR_q"] < 0.05 and opposite
                ),
            }
        )
    return pd.DataFrame(rows)


def _permutation_for_subtype(
    broad: str,
    subtype: str,
    counts: pd.DataFrame,
    selected_symbols: pd.DataFrame,
    gene_sets: Mapping[str, Sequence[str]],
    predefined_pathways: Sequence[str],
    assignments: pd.DataFrame,
    *,
    n_cpus: int,
    checkpoint_dir: str | Path,
) -> pd.DataFrame:
    checkpoint_dir = Path(checkpoint_dir) / safe_name(subtype)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filtered = prefilter_genes(counts.reindex(ALL_LIBRARIES), min_count=10, min_samples=3)
    rows: list[dict[str, Any]] = []
    for permutation_id, assignment in assignments.groupby("permutation_id", observed=True):
        checkpoint = checkpoint_dir / f"{permutation_id}.tsv.gz"
        if checkpoint.exists():
            wide = pd.read_csv(checkpoint, sep="\t")
        else:
            wide, _ = _fit_permutation(filtered, assignment, alpha=0.05, n_cpus=n_cpus)
            wide.to_csv(checkpoint, sep="\t", index=False, compression="gzip")
        effects = selected_symbols[["feature_id", "canonical_mouse_symbol"]].merge(
            wide,
            left_on="feature_id",
            right_on="gene",
            how="inner",
            validate="one_to_one",
        ).rename(
            columns={
                "OC_vs_Y_log2FoldChange": "aging_effect",
                "OT_vs_OC_log2FoldChange": "treatment_effect",
                "OT_vs_Y_log2FoldChange": "residual_effect",
            }
        )
        is_observed = bool(assignment["is_observed"].iloc[0])
        for pathway in predefined_pathways:
            geometry = pathway_effect_geometry(effects, gene_sets[pathway])
            rows.append(
                {
                    "broad_population": broad,
                    "subtype": subtype,
                    "pathway": pathway,
                    "permutation_id": permutation_id,
                    "is_observed": is_observed,
                    **geometry,
                }
            )
    return pd.DataFrame(rows)


def _calibrate_subtype_permutations(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    calibrated: list[pd.DataFrame] = []
    for _, frame in raw.groupby(
        ["broad_population", "subtype", "pathway"], observed=True, sort=True
    ):
        frame = frame.copy()
        for metric, higher in (
            ("rho_aging_treatment", False),
            ("cosine", False),
            ("residual_norm_ratio", False),
            ("directional_fraction", True),
        ):
            result = exact_permutation_calibration(frame, metric, higher_is_more_extreme=higher)
            frame[f"{metric}_observed_rank"] = result["observed_rank"]
            frame[f"{metric}_empirical_p"] = result["empirical_p"]
        calibrated.append(frame)
    return pd.concat(calibrated, ignore_index=True)


def _write_manifest(output_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "output_manifest.tsv":
            rows.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_root / "output_manifest.tsv", sep="\t", index=False)
    return manifest


def _report(
    output_root: Path,
    inventory: pd.DataFrame,
    eligibility: pd.DataFrame,
    localization: pd.DataFrame,
    permutation: pd.DataFrame,
    gmt_checksum: str,
) -> None:
    primary = eligibility[eligibility["eligibility"].eq("Primary_DE_ready")]
    sensitivity = eligibility[eligibility["eligibility"].eq("Sensitivity_only")]
    localized = (
        localization[localization["localized_GSEA_support"]]
        if not localization.empty
        else pd.DataFrame()
    )
    perm_obs = permutation[permutation["is_observed"]] if not permutation.empty else pd.DataFrame()
    lines = [
        "# Stage 3：亚型定位分析报告",
        "",
        "## 1. 为什么做这一步",
        "",
        (
            "Stage 2只说明Granulosa和Stromal fibroblast这两个大类细胞存在较稳定的"
            "MRJP1相关转录程序反向变化。本阶段用于判断这些信号来自哪些既有亚型，"
            "而不根据结果重新命名细胞。"
        ),
        "",
        "## 2. 输入与统计单位",
        "",
        (
            "输入为只读的`results/06_annotation_v2.h5ad`。正式DE仅使用"
            "`layers['counts']`中的raw integer UMI并按library × subtype求和；独立统计单位"
            "始终是9个library，而不是单细胞。Harmony、UMAP和单细胞p值均未进入DE。"
        ),
        "",
        "## 3. 方法",
        "",
        (
            "预先固定Primary规则为每个组3/3 library且每个library至少50个Tier1细胞；"
            "20–49个Tier1/Tier2细胞仅作为敏感性层。Primary亚型采用统一9-library "
            "`~ group`模型，从同一次拟合提取OC vs Y、OT vs OC和OT vs Y。Hallmark使用"
            "与Stage 2相同的Mouse MSigDB 2026.1.Mm、Wald stat和10,000次preranked "
            "GSEA。精确置换仅检查Stage 2预先定义的强或置换支持通路。"
        ),
        "",
        "## 4. 亚型盘点与准入",
        "",
        f"共盘点{len(inventory)}个Granulosa、Stromal和Immune亚型；Primary_DE_ready={len(primary)}，Sensitivity_only={len(sensitivity)}。",
        "",
        "Primary亚型：" + (", ".join(primary["subtype"].astype(str)) or "无"),
        "",
        "Sensitivity亚型：" + (", ".join(sensitivity["subtype"].astype(str)) or "无"),
        "",
        "## 5. 主要结果",
        "",
        (
            "Stage 2预定义通路中，在具体亚型同时满足aging与treatment GSEA FDR<0.05"
            f"且方向相反的亚型-通路组合为{len(localized)}个。"
        ),
        "",
    ]
    if not localized.empty:
        for (broad, subtype), sub in localized.groupby(
            ["broad_population", "subtype"], observed=True
        ):
            lines.append(f"- {broad} / {subtype}: " + ", ".join(sub["pathway"].astype(str)))
    else:
        lines.append("- 没有亚型达到这一预设的双侧GSEA支持标准；阴性结果被保留。")
    if not perm_obs.empty:
        rank1 = perm_obs[
            perm_obs["rho_aging_treatment_observed_rank"].eq(1)
            & perm_obs["cosine_observed_rank"].eq(1)
        ]
        lines.extend(
            [
                "",
                (
                    "在预定义通路的20种exact label assignments中，Spearman和cosine均"
                    f"rank 1/20的亚型-通路组合为{len(rank1)}个。经验p值最低只能为0.05。"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 6. 可以说明什么",
            "",
            "本阶段可以把broad-population转录程序定位到已有、覆盖充分的亚型，并判断真实OC/OT标签是否优于其余19种标签配置。",
            "",
            "## 7. 不能说明什么",
            "",
            "亚型定位不证明细胞谱系转换、因果机制或卵巢功能恢复；GSEA通路名称也不是直接生化功能测定。Sensitivity/Descriptive亚型不能获得与Primary相同的证据等级。",
            "",
            "## 8. 局限",
            "",
            (
                "每组只有3个library，exact permutation只有20种；最低经验p=0.05，不能"
                "形成多重检验校正后的通路发现。亚型标签来自既有annotation v2，本阶段"
                "没有为强化故事而改名。"
            ),
            "",
            "## 9. 技术审计",
            "",
            (
                f"Mouse Hallmark GMT SHA-256：`{gmt_checksum}`。所有正式模型均检查"
                "OT_vs_Y = OC_vs_Y + OT_vs_OC的LFC代数关系。"
            ),
            "",
            "## 10. 下一步",
            "",
            (
                "下一步将以library为单位比较亚型组成变化与同一亚型内部表达变化，避免把"
                "composition remodeling误解为cell-intrinsic regulation。"
            ),
        ]
    )
    (output_root / "SUBTYPE_STAGE3_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def run_stage3(
    config: dict[str, Any],
    *,
    allow_low_memory: bool = False,
    skip_gsea: bool = False,
    skip_permutation: bool = False,
) -> Path:
    require_compute_resources(config, allow_low_memory=allow_low_memory)
    paths = project_paths(config)
    settings = config["stage3_subtype_localization"]
    logger = setup_logging("12_stage3_subtype_localization", config)
    output_root = paths["root"] / settings["output_dir"]
    output_root.mkdir(parents=True, exist_ok=True)
    h5ad_path = paths["root"] / settings["input_object"]
    if not h5ad_path.exists():
        raise FileNotFoundError(h5ad_path)

    inventory_path = output_root / "subtype_inventory.tsv"
    coverage_path = output_root / "subtype_coverage.tsv"
    eligibility_path = output_root / "subtype_eligibility.tsv"
    counts_path = output_root / "subtype_pseudobulk_counts.tsv.gz"
    inventory_checkpoints = (
        inventory_path,
        coverage_path,
        eligibility_path,
        counts_path,
    )
    if all(path.exists() for path in inventory_checkpoints):
        inventory = pd.read_csv(inventory_path, sep="\t")
        coverage = pd.read_csv(coverage_path, sep="\t")
        eligibility = pd.read_csv(eligibility_path, sep="\t")
        counts = pd.read_csv(counts_path, sep="\t", index_col=[0, 1, 2])
        counts.index.names = ["broad_population", "subtype", "library_id"]
        logger.info("Reused audited Stage 3 inventory and pseudobulk checkpoint")
    else:
        inventory, coverage, counts = _inventory_and_counts(
            h5ad_path,
            settings["target_broad_populations"],
            counts_layer=config["ingest"]["counts_layer"],
        )
        eligibility = classify_subtype_eligibility(
            coverage,
            inventory,
            min_primary=int(settings["min_primary_cells_per_library"]),
            min_sensitivity=int(settings["min_sensitivity_cells_per_library"]),
            min_confident_fraction=float(settings["min_confident_identity_fraction"]),
        )
        inventory.to_csv(inventory_path, sep="\t", index=False)
        coverage.to_csv(coverage_path, sep="\t", index=False)
        eligibility.to_csv(eligibility_path, sep="\t", index=False)
        sensitivity = coverage.merge(
            eligibility[["broad_population", "subtype", "eligibility", "reason"]],
            on=["broad_population", "subtype"],
            how="left",
        )
        sensitivity.to_csv(output_root / "subtype_sensitivity.tsv", sep="\t", index=False)
        counts.to_csv(counts_path, sep="\t", compression="gzip")

    primary = eligibility[eligibility["eligibility"].eq("Primary_DE_ready")]
    de_path = output_root / "subtype_DE.tsv"
    audit_path = output_root / "model_audit.tsv"
    reversal_path = output_root / "subtype_reversal.tsv"
    if all(path.exists() for path in (de_path, audit_path, reversal_path)):
        subtype_de = pd.read_csv(de_path, sep="\t")
        logger.info("Reused completed unified subtype DE checkpoint")
    else:
        de_frames: list[pd.DataFrame] = []
        audits: list[pd.DataFrame] = []
        for row in primary.itertuples(index=False):
            logger.info("Unified subtype DE: %s", row.subtype)
            subtype_counts = counts.loc[(row.broad_population, row.subtype)]
            de, audit = _fit_subtype_model(
                row.subtype,
                row.broad_population,
                subtype_counts,
                min_count=int(settings["min_gene_count"]),
                min_samples=int(settings["min_gene_samples"]),
                n_cpus=int(settings["deseq_cpus"]),
            )
            de_frames.append(de)
            audits.append(audit)
        subtype_de = pd.concat(de_frames, ignore_index=True)
        subtype_de.to_csv(de_path, sep="\t", index=False)
        pd.concat(audits, ignore_index=True).to_csv(audit_path, sep="\t", index=False)
        wide = subtype_de.pivot_table(
            index=["broad_population", "subtype", "gene"],
            columns="contrast",
            values=["log2FoldChange", "padj", "stat"],
            aggfunc="first",
        )
        wide.columns = [f"{contrast}_{metric}" for metric, contrast in wide.columns]
        wide = wide.reset_index()
        wide["aging_effect"] = wide["OC_vs_Y_log2FoldChange"]
        wide["treatment_effect"] = wide["OT_vs_OC_log2FoldChange"]
        wide["residual_effect"] = wide["OT_vs_Y_log2FoldChange"]
        wide["direction_opposite"] = (
            np.sign(wide["aging_effect"]) == -np.sign(wide["treatment_effect"])
        )
        wide["residual_closer_to_y"] = (
            wide["residual_effect"].abs() < wide["aging_effect"].abs()
        )
        wide.to_csv(reversal_path, sep="\t", index=False)

    gmt_path = paths["root"] / "resources" / "gene_sets" / "mh.all.v2026.1.Mm.symbols.gmt"
    gene_sets = parse_gmt(gmt_path)
    gmt_checksum = sha256_file(gmt_path)
    mapping = pd.read_csv(
        paths["results"] / "pathway_stage2" / "gene_identifier_mapping.tsv", sep="\t"
    )
    hallmark_path = output_root / "subtype_hallmark.tsv"
    resolution_path = output_root / "duplicate_symbol_resolution.tsv"
    hallmark = pd.DataFrame()
    resolution = pd.DataFrame()
    if not skip_gsea and hallmark_path.exists() and hallmark_path.stat().st_size > 100:
        hallmark = pd.read_csv(hallmark_path, sep="\t")
        if resolution_path.exists() and resolution_path.stat().st_size > 0:
            resolution = pd.read_csv(resolution_path, sep="\t")
        logger.info("Reused completed subtype GSEA checkpoint")
    elif not skip_gsea:
        hallmark, resolution = _run_gsea_grid(
            subtype_de,
            mapping,
            gene_sets,
            workers=int(settings["gsea_workers"]),
            threads=int(settings["gsea_threads"]),
            permutations=int(settings.get("gsea_permutations", GSEA_PERMUTATIONS)),
            min_size=int(settings.get("gsea_min_size", GSEA_MIN_SIZE)),
            max_size=int(settings.get("gsea_max_size", GSEA_MAX_SIZE)),
            seed=int(settings.get("random_seed", GSEA_SEED)),
            logger=logger,
        )
        hallmark.to_csv(hallmark_path, sep="\t", index=False)
        resolution.to_csv(resolution_path, sep="\t", index=False)
    broad_evidence = pd.read_csv(
        paths["results"] / "pathway_stage2" / "hallmark_pathway_evidence.tsv", sep="\t"
    )
    localization = (
        build_localization_table(hallmark, broad_evidence) if not hallmark.empty else pd.DataFrame()
    )
    localization.to_csv(
        output_root / "broad_to_subtype_localization.tsv", sep="\t", index=False
    )

    permutation = pd.DataFrame()
    if not skip_permutation:
        assignments = enumerate_permutation_assignments()
        predefined = broad_evidence[
            broad_evidence["pathway_evidence_level"].isin(
                ["Strong_multi_metric_reversal", "Permutation_supported_reversal"]
            )
        ].groupby("population", observed=True)["pathway"].apply(list)
        futures = {}
        frames: list[pd.DataFrame] = []
        checkpoint_dir = output_root / "permutation_checkpoints"
        primary_rows = [
            row
            for row in primary.itertuples(index=False)
            if row.broad_population in predefined.index
        ]
        with ProcessPoolExecutor(
            max_workers=min(int(settings["model_workers"]), len(primary_rows)),
            mp_context=mp.get_context("spawn"),
        ) as executor:
            for row in primary_rows:
                subtype_counts = counts.loc[(row.broad_population, row.subtype)]
                observed_de = subtype_de[
                    subtype_de["subtype"].eq(row.subtype)
                    & subtype_de["contrast"].eq("OC_vs_Y")
                ]
                selected, _ = deduplicate_rank_table(observed_de, mapping)
                future = executor.submit(
                    _permutation_for_subtype,
                    row.broad_population,
                    row.subtype,
                    subtype_counts,
                    selected,
                    gene_sets,
                    predefined[row.broad_population],
                    assignments,
                    n_cpus=int(settings["deseq_cpus"]),
                    checkpoint_dir=checkpoint_dir,
                )
                futures[future] = row.subtype
            for future in as_completed(futures):
                subtype = futures[future]
                frame = future.result()
                frames.append(frame)
                logger.info("Subtype exact permutation complete: %s rows=%d", subtype, len(frame))
        if frames:
            permutation = _calibrate_subtype_permutations(pd.concat(frames, ignore_index=True))
    permutation.to_csv(output_root / "subtype_permutation.tsv", sep="\t", index=False)

    _report(output_root, inventory, eligibility, localization, permutation, gmt_checksum)
    provenance = {
        "status": "complete",
        "input": str(h5ad_path),
        "input_sha256": sha256_file(h5ad_path),
        "gmt_sha256": gmt_checksum,
        "random_seed": int(settings["random_seed"]),
        "software": {
            name: version(name)
            for name in ("anndata", "scanpy", "pydeseq2", "gseapy", "pandas", "numpy")
        },
        "primary_subtypes": primary["subtype"].astype(str).tolist(),
        "n_primary_subtypes": int(len(primary)),
        "n_gsea_analyses": int(
            hallmark[["subtype", "contrast"]].drop_duplicates().shape[0]
            if not hallmark.empty
            else 0
        ),
    }
    (output_root / "COMPLETE.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_manifest(output_root)
    print("STAGE3_SUBTYPE_LOCALIZATION_COMPLETE")
    return output_root
