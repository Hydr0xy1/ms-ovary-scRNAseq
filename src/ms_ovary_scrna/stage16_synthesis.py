"""File-grounded Stage 16-22 synthesis and conservative decision gates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t", compression="infer")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _number(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _fmt(value: Any, digits: int = 3) -> str:
    value = _number(value)
    return f"{value:.{digits}f}" if np.isfinite(value) else "NA"


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _checkpoint(path: Path, status: str, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _geometry_summary(geometry: pd.DataFrame, loo: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ratio": np.nan,
        "cosine": np.nan,
        "residual_ratio": np.nan,
        "seed_distance_support": np.nan,
        "loo_distance_support": np.nan,
        "loo_cosine_support": np.nan,
    }
    if geometry.empty:
        return result
    broad = geometry.loc[
        geometry["population"].astype(str).eq("Granulosa")
        & geometry["analysis_level"].astype(str).eq("broad")
    ].copy()
    if broad.empty:
        return result
    result.update(
        {
            "ratio": pd.to_numeric(
                broad["distance_ratio_OT_over_OC"], errors="coerce"
            ).median(),
            "cosine": pd.to_numeric(
                broad["aging_treatment_cosine"], errors="coerce"
            ).median(),
            "residual_ratio": pd.to_numeric(
                broad["treatment_residual_ratio"], errors="coerce"
            ).median(),
            "seed_distance_support": float(
                pd.to_numeric(
                    broad["distance_change_OT_minus_OC"], errors="coerce"
                ).lt(0).mean()
            ),
        }
    )
    if not loo.empty:
        block = loo.loc[
            loo["population"].astype(str).eq("Granulosa")
            & loo["analysis_level"].astype(str).eq("broad")
        ]
        if not block.empty:
            result["loo_distance_support"] = float(
                pd.to_numeric(
                    block["distance_change_OT_minus_OC"], errors="coerce"
                ).lt(0).mean()
            )
            result["loo_cosine_support"] = float(
                pd.to_numeric(
                    block["aging_treatment_cosine"], errors="coerce"
                ).lt(0).mean()
            )
    return result


def _ot_summary(frame: pd.DataFrame) -> dict[str, Any]:
    result = {"oc_y": np.nan, "ot_y": np.nan, "ratio": np.nan}
    if frame.empty:
        return result
    block = frame.loc[
        frame["population"].astype(str).eq("Granulosa")
        & frame["analysis_level"].astype(str).eq("broad")
        & frame["metric"].astype(str).eq("Sinkhorn_root_cost")
    ]
    if block.empty:
        return result
    medians = block.groupby("comparison", observed=True)["median"].median()
    oc_y = _number(medians.get("OC_vs_Y"))
    ot_y = _number(medians.get("OT_vs_Y"))
    result.update(
        {
            "oc_y": oc_y,
            "ot_y": ot_y,
            "ratio": ot_y / oc_y if np.isfinite(oc_y) and oc_y != 0 else np.nan,
        }
    )
    return result


def _age_summary(
    benchmark: pd.DataFrame, contrasts: pd.DataFrame, loso: pd.DataFrame
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ridge_r2": np.nan,
        "elastic_r2": np.nan,
        "reverse_fraction": np.nan,
        "loo_reverse_fraction": np.nan,
        "n_samples": 0,
        "n_studies": 0,
    }
    if not benchmark.empty:
        gran = benchmark.loc[benchmark["population"].astype(str).eq("Granulosa")]
        for model, key in [("ridge", "ridge_r2"), ("elastic_net", "elastic_r2")]:
            row = gran.loc[gran["model"].astype(str).eq(model)]
            if not row.empty:
                result[key] = _number(row.iloc[0].get("loso_r2"))
        if not gran.empty:
            result["n_samples"] = int(
                pd.to_numeric(gran["n_external_samples"], errors="coerce").max()
            )
            result["n_studies"] = int(
                pd.to_numeric(gran["n_external_studies"], errors="coerce").max()
            )
    if not contrasts.empty:
        gran = contrasts.loc[contrasts["population"].astype(str).eq("Granulosa")]
        signs: list[bool] = []
        for _, block in gran.groupby("model", observed=True):
            values = block.set_index("contrast")["effect"]
            aging = _number(values.get("OC_vs_Y"))
            treatment = _number(values.get("OT_vs_OC"))
            if np.isfinite(aging) and np.isfinite(treatment) and aging != 0:
                signs.append(aging * treatment < 0)
        if signs:
            result["reverse_fraction"] = float(np.mean(signs))
    if not loso.empty:
        block = loso.loc[
            loso["validation_scope"]
            .astype(str)
            .eq("internal_score_leave_one_library_out")
            & loso["population"].astype(str).eq("Granulosa")
        ]
        if not block.empty:
            aging = pd.to_numeric(
                block["aging_effect_OC_minus_Y"], errors="coerce"
            )
            treatment = pd.to_numeric(
                block["treatment_effect_OT_minus_OC"], errors="coerce"
            )
            valid = aging.notna() & treatment.notna() & aging.ne(0)
            if valid.any():
                result["loo_reverse_fraction"] = float(
                    (aging[valid] * treatment[valid] < 0).mean()
                )
    return result


def _programs(
    robust: pd.DataFrame,
    tf: pd.DataFrame,
    contrastive: pd.DataFrame,
    geometry: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not robust.empty:
        stable = float(
            robust["classification"].astype(str).eq("stable_support").mean()
        )
        rows.append(
            {
                "priority": 1,
                "program": "conserved_external_aging_axis",
                "population": "Granulosa",
                "effect_class": "partial_age_opposite_movement_with_orthogonal_component",
                "library_level_support": f"stable external contrasts fraction={stable:.3f}",
                "model_support": (
                    f"distance ratio={_fmt(geometry['ratio'])}; "
                    f"cosine={_fmt(geometry['cosine'])}; "
                    f"orthogonal residual={_fmt(geometry['residual_ratio'])}"
                ),
                "status": "core_program_if_all_model_gates_hold",
                "validation": "independent cohort, Granulosa-resolved expression and ovarian phenotype",
            }
        )
    for tf_name, label in [
        ("Hif1a", "Hif1a-associated hypoxia/ROS response"),
        ("Smad3", "Smad3-associated TGF-beta/ECM response"),
    ]:
        block = tf.loc[tf["candidate_tf"].astype(str).eq(tf_name)]
        if block.empty:
            continue
        row = block.iloc[0]
        aging = _number(row.get("age_effect_OC_minus_Y"))
        treatment = _number(row.get("treatment_effect_OT_minus_OC"))
        if not (
            np.isfinite(aging)
            and np.isfinite(treatment)
            and aging * treatment < 0
        ):
            continue
        rows.append(
            {
                "priority": len(rows) + 1,
                "program": label,
                "population": "Granulosa",
                "effect_class": "age-opposite expression candidate; TF activity unproven",
                "library_level_support": (
                    f"direction consistency={_fmt(row.get('treatment_direction_consistency'))}; "
                    f"OC-Y={_fmt(aging)}; OT-OC={_fmt(treatment)}"
                ),
                "model_support": "global Granulosa geometry only; not candidate-specific",
                "status": "candidate_program_not_mechanism",
                "validation": "TF protein/activity or localization plus target-gene panel",
            }
        )
        if len(rows) >= 3:
            break
    if len(rows) < 3 and not contrastive.empty:
        retained = contrastive.loc[
            _bool_series(contrastive["retained_candidate"])
        ]
        if not retained.empty:
            genes = ";".join(retained["gene"].astype(str).head(10))
            rows.append(
                {
                    "priority": len(rows) + 1,
                    "program": "contrastiveVI treatment-salient gene set",
                    "population": "Granulosa",
                    "effect_class": "treatment-salient candidate",
                    "library_level_support": (
                        f"retained genes={len(retained)}; representatives={genes}"
                    ),
                    "model_support": "cross-seed, LOO and independent gene evidence gate",
                    "status": "exploratory_treatment_specific_program",
                    "validation": "targeted qPCR in independent biological pools",
                }
            )
    return pd.DataFrame(rows[:3])


def _mechanisms(
    tf: pd.DataFrame, micro: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tf_name in ["Hif1a", "Smad3"]:
        block = tf.loc[tf["candidate_tf"].astype(str).eq(tf_name)]
        if block.empty:
            continue
        row = block.iloc[0]
        aging = _number(row.get("age_effect_OC_minus_Y"))
        treatment = _number(row.get("treatment_effect_OT_minus_OC"))
        consistency = _number(row.get("treatment_direction_consistency"))
        rows.append(
            {
                "candidate": tf_name,
                "candidate_type": "TF_expression_hypothesis",
                "population_or_chain": "Granulosa",
                "library_level_support": f"direction consistency={_fmt(consistency)}",
                "age_axis_consistency": bool(
                    np.isfinite(aging)
                    and np.isfinite(treatment)
                    and aging * treatment < 0
                ),
                "optimal_transport_consistency": "Granulosa-global only; not TF-specific",
                "external_data_support": "candidate-specific external activity not established",
                "leave_one_library_out": (
                    "dedicated TF LOO absent; sensitivity range="
                    + _fmt(row.get("single_library_sensitivity_range"))
                ),
                "composition_risk": row.get("composition_context", "not re-estimated"),
                "effect_class": "age-opposite candidate",
                "evidence_grade": (
                    "prioritize"
                    if tf_name == "Hif1a" and consistency >= 0.8
                    else "secondary"
                ),
                "required_validation": "qPCR and Western blot, activity/localization and target panel",
            }
        )
    for ligand, receptor in [("Il6", "Il6st"), ("Fgf2", "Fgfr2")]:
        block = micro.loc[
            micro["ligand"].astype(str).eq(ligand)
            & micro["receptor"].astype(str).eq(receptor)
        ]
        if block.empty:
            continue
        rows.append(
            {
                "candidate": f"{ligand}->{receptor}",
                "candidate_type": "microenvironment_expression_chain",
                "population_or_chain": "Stromal_fibroblast->Granulosa",
                "library_level_support": "pooled-cell expression only; group direction not established",
                "age_axis_consistency": "not_assessed_for_chain",
                "optimal_transport_consistency": "not_ligand_specific",
                "external_data_support": "not independently validated",
                "leave_one_library_out": "not_available",
                "composition_risk": "high; sender/receiver abundance may contribute",
                "effect_class": "microenvironment_hypothesis_not_age_return",
                "evidence_grade": "exploratory",
                "required_validation": "protein localization, downstream signaling and perturbation",
            }
        )
    return pd.DataFrame(rows[:4])


def run_stage16_synthesis(
    config: Mapping[str, Any],
    stage_root: Path,
    logger: Any,
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    root = paths["root"]
    mechanism_dir = stage_root / "09_mechanism_candidates"
    synthesis = stage_root / "10_synthesis"
    for directory in [
        mechanism_dir,
        synthesis,
        synthesis / "main_figure_drafts",
        synthesis / "supplementary_figure_drafts",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    evidence = _read(stage_root / "01_evidence_matrix/EVIDENCE_MATRIX.tsv")
    evidence_summary = _read(
        stage_root / "01_evidence_matrix/EVIDENCE_MATRIX_SUMMARY.tsv"
    )
    subtype = _read(
        stage_root / "01_evidence_matrix/GRANULOSA_SUBTYPE_ROBUSTNESS.tsv"
    )
    geometry_table = _read(
        stage_root / "04_latent_geometry_ot/LATENT_GEOMETRY.tsv"
    )
    geometry_loo = _read(
        stage_root / "04_latent_geometry_ot/OT_LOO_RESULTS.tsv"
    )
    ot_table = _read(
        stage_root / "04_latent_geometry_ot/OT_SENSITIVITY.tsv"
    )
    age_benchmark = _read(
        stage_root / "05_external_age_models/MODEL_BENCHMARK.tsv"
    )
    age_contrasts = _read(
        stage_root / "05_external_age_models/EXTERNAL_AGE_SCORE_CONTRASTS.tsv"
    )
    age_loso = _read(
        stage_root / "05_external_age_models/EXTERNAL_AGE_MODEL_LOSO.tsv"
    )
    scvi_runs = _read(
        stage_root / "03_scvi_reference/SCVI_MODEL_RUNS.tsv"
    )
    scvi_stability = _read(
        stage_root / "03_scvi_reference/SCVI_STABILITY.tsv"
    )
    cvi_stability = _read(
        stage_root / "06_contrastivevi/CONTRASTIVEVI_STABILITY.tsv"
    )
    cvi_programs = _read(
        stage_root / "06_contrastivevi/CONTRASTIVEVI_PROGRAM_GENES.tsv"
    )
    foundation = _read(
        stage_root / "07_foundation_models/FOUNDATION_MODEL_BENCHMARK.tsv"
    )
    ortholog = _read(
        stage_root / "08_cross_species/ORTHOLOG_MAPPING.tsv"
    )
    follow = root / "results/deep_dive_stage15/followup_validation"
    tf = _read(follow / "GRANULOSA_TF_CANDIDATES.tsv")
    micro = _read(follow / "GRANULOSA_MICROENVIRONMENT_CANDIDATES.tsv")
    robust = _read(follow / "GRANULOSA_ROBUSTNESS_SUMMARY.tsv")

    geometry = _geometry_summary(geometry_table, geometry_loo)
    ot_summary = _ot_summary(ot_table)
    age = _age_summary(age_benchmark, age_contrasts, age_loso)
    programs = _programs(robust, tf, cvi_programs, geometry)
    mechanisms = _mechanisms(tf, micro)

    final_evidence = evidence.copy()
    if not final_evidence.empty:
        final_evidence["synthesis_scope"] = (
            "method-level evidence; repeated methods are not independent replicates"
        )
    final_evidence.to_csv(
        synthesis / "FINAL_EVIDENCE_MATRIX.tsv", sep="\t", index=False
    )
    for directory in [mechanism_dir, synthesis]:
        programs.to_csv(
            directory / "FINAL_CANDIDATE_PROGRAMS.tsv", sep="\t", index=False
        )
        mechanisms.to_csv(
            directory / "FINAL_MECHANISM_CANDIDATES.tsv",
            sep="\t",
            index=False,
        )

    model_frames: list[pd.DataFrame] = []
    for label, frame in [
        ("scVI", scvi_runs),
        ("scVI_stability", scvi_stability),
        ("external_age", age_benchmark),
        ("contrastiveVI", cvi_stability),
        ("foundation_model_gate", foundation),
    ]:
        if frame.empty:
            continue
        item = frame.copy()
        item.insert(0, "analysis", label)
        model_frames.append(item)
    model_comparison = (
        pd.concat(model_frames, ignore_index=True, sort=False)
        if model_frames
        else pd.DataFrame()
    )
    for directory in [mechanism_dir, synthesis]:
        model_comparison.to_csv(
            directory / "MODEL_COMPARISON.tsv", sep="\t", index=False
        )

    subtype_support = (
        float(
            subtype["candidate_status"]
            .astype(str)
            .eq("supports_broad_direction")
            .mean()
        )
        if not subtype.empty
        else np.nan
    )
    granulosa_subtype_geometry = geometry_table.loc[
        geometry_table["population"].astype(str).eq("Granulosa")
        & geometry_table["analysis_level"].astype(str).eq("subtype")
    ]
    n_latent_subtypes = int(
        granulosa_subtype_geometry["state"].astype(str).nunique()
    )
    subtype_ratio_by_state = (
        granulosa_subtype_geometry.groupby("state", observed=True)[
            "distance_ratio_OT_over_OC"
        ].median()
        if not granulosa_subtype_geometry.empty
        else pd.Series(dtype=float)
    )
    granulosa_subtype_loo = geometry_loo.loc[
        geometry_loo["population"].astype(str).eq("Granulosa")
        & geometry_loo["analysis_level"].astype(str).eq("subtype")
    ]
    subtype_loo_distance_support = (
        float(
            pd.to_numeric(
                granulosa_subtype_loo["distance_change_OT_minus_OC"],
                errors="coerce",
            )
            .lt(0)
            .mean()
        )
        if not granulosa_subtype_loo.empty
        else np.nan
    )
    subtype_loo_cosine_support = (
        float(
            pd.to_numeric(
                granulosa_subtype_loo["aging_treatment_cosine"],
                errors="coerce",
            )
            .lt(0)
            .mean()
        )
        if not granulosa_subtype_loo.empty
        else np.nan
    )
    external_stable = (
        float(
            robust["classification"].astype(str).eq("stable_support").mean()
        )
        if not robust.empty
        else np.nan
    )
    n_cvi = (
        int(_bool_series(cvi_programs["retained_candidate"]).sum())
        if not cvi_programs.empty
        else 0
    )
    n_ortholog = (
        int(
            ortholog["mapping_status"]
            .astype(str)
            .eq("ensembl_one_to_one")
            .sum()
        )
        if not ortholog.empty
        else 0
    )
    distance_support = (
        np.isfinite(_number(geometry["ratio"]))
        and _number(geometry["ratio"]) < 1
    )
    opposite_support = (
        np.isfinite(_number(geometry["cosine"]))
        and _number(geometry["cosine"]) < 0
    )
    orthogonal_large = (
        np.isfinite(_number(geometry["residual_ratio"]))
        and _number(geometry["residual_ratio"]) >= 0.5
    )
    if distance_support and opposite_support and orthogonal_large:
        interpretation = (
            "partial age-opposite movement with substantial treatment-specific "
            "orthogonal remodeling"
        )
    elif distance_support and opposite_support:
        interpretation = "partial movement toward the young reference"
    elif opposite_support:
        interpretation = (
            "age-opposite direction without convincing return toward young"
        )
    else:
        interpretation = "no stable latent-space support for age reversal"

    failures = _read(stage_root / "FAILED_STEPS.tsv")
    historical_failures = len(failures)
    unresolved = (
        int(
            (~failures["error"].astype(str).str.contains("_rank", regex=False)).sum()
        )
        if not failures.empty
        else 0
    )

    master = f"""# Stage 16-22 深度挖掘综合报告

## 核心结论

Granulosa仍是最稳定的MRJP1响应主线。scVI潜在空间中，Granulosa broad层面跨seed的中位distance(OT,Y)/distance(OC,Y)为{_fmt(geometry['ratio'])}，aging-treatment cosine为{_fmt(geometry['cosine'])}，治疗向量正交残差占比为{_fmt(geometry['residual_ratio'])}。综合解释为：{interpretation}。这不等于整体年轻化、功能恢复或因果机制。

Stage 15外部年龄轴中，稳定支持的Granulosa contrast占比为{_fmt(external_stable)}。Stage 16外部模型只基于{age['n_samples']}个sample、{age['n_studies']}个study；Ridge/Elastic Net LOSO R2分别为{_fmt(age['ridge_r2'])}/{_fmt(age['elastic_r2'])}，只能称为外部年龄相关分数。

## 对H1-H4的回答

1. H1部分年龄回移：有方向性支持。跨seed距离下降比例={_fmt(geometry['seed_distance_support'])}，逐library LOO距离下降比例={_fmt(geometry['loo_distance_support'])}，LOO cosine为负比例={_fmt(geometry['loo_cosine_support'])}。
2. H2正交治疗重塑：同时存在；正交残差不可忽略。contrastiveVI保留{n_cvi}个通过联合门控的候选基因，但不构成直接靶点证据。
3. H3组成或单library驱动：既有gene-level亚型审计支持broad方向的比例为{_fmt(subtype_support)}；scVI latent中有{n_latent_subtypes}个Granulosa亚型通过覆盖门，其distance ratio范围为{_fmt(subtype_ratio_by_state.min())}-{_fmt(subtype_ratio_by_state.max())}，全部亚型LOO的距离下降/cosine为负比例分别为{_fmt(subtype_loo_distance_support)}/{_fmt(subtype_loo_cosine_support)}。这降低了纯亚型比例解释，但n=3/group仍是主要限制。
4. H4外部泛化：尚未充分排除。公共训练和验证主要来自单一GSE267729，不能做leave-one-study-out。

## 模型增量信息

scVI和Sinkhorn提供了分布几何证据，Granulosa Sinkhorn OT_vs_Y/OC_vs_Y中位距离比为{_fmt(ot_summary['ratio'])}。监督式深度年龄模型、MIL和基础模型均未通过sample/study数决策门，因此没有证据表明它们优于简单线性基线。

## 候选机制收敛

仅保留{len(programs)}个Granulosa程序、最多2个TF表达候选和2条微环境表达链。Hif1a优先级高于Smad3；Il6->Il6st与Fgf2->Fgfr2只有表达层支持，尚无逐library方向、蛋白或下游信号证据。

## Stromal与跨物种

Stromal_fibroblast继续作为异质性、参考依赖的补充结果，不概括为统一回移。Ensembl版本化查询获得{n_ortholog}条mouse-human一对一映射记录；因缺少满足原始counts、独立sample和细胞类型标签要求的人卵巢数据，本轮不做人类表达投影，也不声称MRJP1在人中有效。
"""
    (synthesis / "MASTER_REPORT_CN.md").write_text(master, encoding="utf-8")

    executive = f"""# 执行摘要

- Granulosa仍是稳定主线；Stromal_fibroblast仅作异质性补充。
- MRJP1效应更符合部分沿衰老方向反向移动加治疗特异正交重塑，不是整体年轻化。
- scVI/Sinkhorn增加几何证据；没有监督式深度模型或基础模型通过外部泛化门。
- 优先验证Hif1a相关应激/氧化还原程序，Smad3为次级候选；微环境链仅为探索性假说。
- 最小实验组合应覆盖表达、蛋白/定位、ROS/线粒体和卵巢功能表型，并增加独立生物学重复。
"""
    (synthesis / "EXECUTIVE_SUMMARY_CN.md").write_text(
        executive, encoding="utf-8"
    )

    limitations = f"""# 限制与失败审计

1. 正式统计单位是9个library/biological pool，n=3/group；细胞数不是生物学重复。
2. batch、estrous_stage、pool_mouse_ids缺失且未被模型填补。
3. 公共年龄模型只有{age['n_studies']}个study，不能检验跨study泛化；分数不是真实生物学年龄。
4. scVI/contrastiveVI latent、TF、NMF和配体-受体都是候选生成证据，不是直接机制。
5. 无spliced/unspliced，未做RNA velocity；无空间数据，未做空间GNN。
6. FAILED_STEPS历史记录{historical_failures}条；已知_rank KeyError已修复并成功重跑，估计待解决非历史条目{unresolved}条。
"""
    (synthesis / "LIMITATIONS_AND_FAILURES_CN.md").write_text(
        limitations, encoding="utf-8"
    )
    claims = """# 论文表述边界（中英对照）

## 可支持

中文：MRJP1处理与细胞类型和状态特异的转录重塑相关。Granulosa细胞沿保守衰老相关转录方向显示部分反向移动，同时存在治疗特异的正交重塑。

English: MRJP1 treatment was associated with cell-type- and state-specific transcriptional remodeling. Granulosa cells showed partial movement opposite to conserved aging-related transcriptional directions, together with a substantial treatment-specific orthogonal component.

## 不可支持

不应表述为整体年轻化、恢复卵巢功能、减少了若干月生物学年龄、证明Hif1a/Smad3直接机制，或证明MRJP1在人类中有效。
"""
    (synthesis / "PAPER_CLAIMS_CN_EN.md").write_text(
        claims, encoding="utf-8"
    )
    experiments = """# 下一轮实验优先级

1. 最小核心组合：增加独立动物/批次；Granulosa富集或定位的Hif1a及Hmox1/Nqo1/Vegfa qPCR；HIF1A蛋白/定位；ROS与线粒体功能；E2/AMH；卵泡计数和闭锁组织学。
2. 次级机制组合：Smad3、p-SMAD3及Serpine1/ECM target面板；只在library方向复现后进入阻断或激动实验。
3. 微环境候选：优先选一条链做IL6/IL6ST或FGF2/FGFR2定位、蛋白和下游磷酸化；没有方向复现前不同时铺开两条链。
4. 设计加强：记录动情周期、个体/池化和batch；增加剂量或时间点，以区分短期应激与持续状态转变。
"""
    (synthesis / "NEXT_EXPERIMENTS_PRIORITY_CN.md").write_text(
        experiments, encoding="utf-8"
    )
    figure_contract = """# 独立图绘制合同（Python）

所有图独立输出，不拼成组图。核心结论是Granulosa表现为部分沿衰老方向反向移动，但治疗特异正交重塑仍占重要成分。

- Archetype：独立定量图；每张图只回答一个证据问题。
- Backend：Python、matplotlib和seaborn。
- Export：SVG、PDF、PNG预览；投稿候选另输出600 dpi TIFF。
- Statistics：n定义为library/biological pool；seed、library-pair或cell bootstrap只是技术稳定性，不增加生物学n。
- Source data：只读取Stage 16-22 TSV产物。
- Reviewer risk：单一公共study、n=3/group、latent空间可旋转性、细胞层距离不是生物学重复。
"""
    (synthesis / "FIGURE_CONTRACTS_CN.md").write_text(
        figure_contract, encoding="utf-8"
    )

    _checkpoint(
        mechanism_dir / "CHECKPOINT.json",
        "MECHANISM_CANDIDATES_COMPLETE",
        n_programs=len(programs),
        n_mechanism_candidates=len(mechanisms),
    )
    _checkpoint(
        synthesis / "CHECKPOINT.json",
        "SYNTHESIS_COMPLETE",
        n_programs=len(programs),
        n_mechanism_candidates=len(mechanisms),
        geometry_interpretation=interpretation,
        n_ortholog_records=n_ortholog,
    )
    logger.info(
        "Synthesis complete: programs=%d candidates=%d interpretation=%s",
        len(programs),
        len(mechanisms),
        interpretation,
    )
    return {
        "n_programs": len(programs),
        "n_candidates": len(mechanisms),
        "geometry_interpretation": interpretation,
        "evidence_candidates": len(evidence_summary),
    }
