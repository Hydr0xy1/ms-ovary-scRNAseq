"""Stage 12: final auditable synthesis of all advanced-analysis stages."""
# ruff: noqa: E501

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .project import project_paths, setup_logging
from .stage7_regulatory_activity import sha256_file


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t") if path.exists() else pd.DataFrame()


def _stage_status(results: Path) -> pd.DataFrame:
    specs = {
        3: ("stage3_subtype_localization", "COMPLETE.json"),
        4: ("stage4_composition_decomposition", "COMPLETE.json"),
        5: ("stage5_rejuvenation_geometry", "COMPLETE.json"),
        6: ("stage6_external_validation", "COMPLETE.json"),
        7: ("stage7_regulatory_activity", "COMPLETE.json"),
        8: ("stage8_gene_programs", "COMPLETE.json"),
        9: ("stage9_communication", "COMPLETE.json"),
        10: ("stage10_candidates", "COMPLETE.json"),
        11: ("stage11_phenotype_framework", "COMPLETE.json"),
    }
    rows: list[dict[str, Any]] = []
    for stage, (folder, marker) in specs.items():
        root = results / folder
        marker_path = root / marker
        status = "COMPLETE" if marker_path.exists() else "MISSING"
        reason = ""
        if not marker_path.exists():
            skip = list(root.glob("*SKIPPED*.md")) + list(root.glob("*NOT_AVAILABLE*.md")) + list(root.glob("*NOT_YET_JUSTIFIED*.md"))
            failed = list(root.glob("*FAILED*.md"))
            if skip:
                status, reason = "SKIPPED", skip[0].name
            elif failed:
                status, reason = "FAILED", failed[0].name
        rows.append({"stage": stage, "folder": folder, "status": status, "reason": reason})
    return pd.DataFrame(rows)


def _table_or_text(table: pd.DataFrame, columns: list[str] | None = None, n: int = 20) -> str:
    if table.empty:
        return "本阶段没有可用结果，见对应SKIPPED/FAILED报告。"
    selected = table
    if columns:
        selected = selected[[c for c in columns if c in selected.columns]]
    return selected.head(n).to_markdown(index=False)


def _geometry_summary(projection: pd.DataFrame, permutation: pd.DataFrame) -> pd.DataFrame:
    if projection.empty:
        return pd.DataFrame()
    means = projection.groupby(["population", "group"], observed=True)["aging_axis_projection"].mean().unstack("group").reset_index()
    if not permutation.empty:
        observed = permutation.loc[permutation["is_observed"].fillna(False).astype(bool), [
            "population", "observed_rank_low_is_more_young_directed", "empirical_p"
        ]].drop_duplicates("population")
        means = means.merge(observed, on="population", how="left")
    return means


def run_stage12(config: Mapping[str, Any]) -> None:
    paths = project_paths(config)
    results = paths["results"]
    output_root = results / "stage12_final_synthesis"
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(paths["logs"] / "stage12_final_synthesis.log")
    status = _stage_status(results)
    critical = status.loc[status["stage"].isin([3, 4, 5, 10])]
    if not critical["status"].eq("COMPLETE").all():
        raise RuntimeError("Stage 12 gate failed: Stage 3/4/5/10 must be COMPLETE")

    population = _read(results / "de_stage1_6" / "population_reversal_evidence.tsv")
    pathways = _read(results / "pathway_stage2" / "primary_pathway_reversal_summary.tsv")
    localization = _read(results / "stage3_subtype_localization" / "broad_to_subtype_localization.tsv")
    decomposition = _read(results / "stage4_composition_decomposition" / "decomposition_summary.tsv")
    composition_distance = _read(results / "stage4_composition_decomposition" / "aitchison_distance.tsv")
    projection = _read(results / "stage5_rejuvenation_geometry" / "aging_axis_projection.tsv")
    geometry_perm = _read(results / "stage5_rejuvenation_geometry" / "permutation_geometry.tsv")
    geometry = _geometry_summary(projection, geometry_perm)
    external = _read(results / "stage6_external_validation" / "internal_external_concordance.tsv")
    regulators = _read(results / "stage7_regulatory_activity" / "candidate_regulators.tsv")
    programs = _read(results / "stage8_gene_programs" / "program_reversal.tsv")
    communication = _read(results / "stage9_communication" / "communication_evidence.tsv")
    genes = _read(results / "stage10_candidates" / "candidate_gene_evidence.tsv")
    shortlist = _read(results / "stage10_candidates" / "experimental_validation_shortlist.tsv")
    phenotype = _read(results / "stage11_phenotype_framework" / "recommended_validation_assays.tsv")

    strong_pathways = pathways.loc[
        pathways.get("pathway_evidence_level", pd.Series(index=pathways.index, dtype=str)).astype(str).eq("Strong_multi_metric_reversal")
    ] if not pathways.empty else pathways
    localized = localization.loc[
        localization.get("localized_GSEA_support", pd.Series(index=localization.index, dtype=bool)).fillna(False).astype(bool)
    ] if not localization.empty else localization
    tier_counts = genes.groupby(["cell_type", "evidence_tier"], observed=True).size().rename("n").reset_index() if not genes.empty else pd.DataFrame()
    top_regulators = regulators.loc[
        regulators.get("directionally_reversed", pd.Series(index=regulators.index, dtype=bool)).fillna(False).astype(bool)
        & regulators.get("closer_to_young_after_treatment", pd.Series(index=regulators.index, dtype=bool)).fillna(False).astype(bool)
    ].sort_values("opposition_magnitude", ascending=False) if not regulators.empty else regulators
    top_programs = programs.loc[
        programs.get("directionally_reversed", pd.Series(index=programs.index, dtype=bool)).fillna(False).astype(bool)
        & programs.get("closer_to_young_after_treatment", pd.Series(index=programs.index, dtype=bool)).fillna(False).astype(bool)
    ].sort_values("opposition_magnitude", ascending=False) if not programs.empty else programs

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paths["root"], text=True).strip()
    except Exception:
        git_commit = "unavailable"

    cn = [
        "# 小鼠卵巢衰老 / MRJP1 高级单细胞分析最终交接报告",
        "",
        "## 1 项目设计",
        "Y为4月龄vehicle，OC为10月龄vehicle，OT为10月龄MRJP1 200 mg/kg；每组3个独立library。所有正式统计以library为生物学重复，single cell仅用于定义population/state与聚合。",
        "",
        "## 2 已完成分析流程",
        "已从输入审计、QC、annotation、raw-count pseudobulk、统一9-library DE、20种exact permutation和Mouse Hallmark，推进到如下高级阶段：",
        status.to_markdown(index=False),
        "",
        f"代码版本：`{git_commit}`。",
        "",
        "## 3 数据质量与统计单位",
        "主对象为105,763 cells × 57,132 features，raw UMI保存在layers['counts']。DE仅使用raw integer pseudobulk；Harmony只用于表示空间。每组n=3意味着FDR与置换分辨率有限，exact permutation最低经验p为0.05。",
        "",
        "## 4 Cell atlas",
        "broad annotation与annotation v2已经完成；Granulosa和Stromal_fibroblast是当前最稳定的MRJP1相关反向population。亚型分析只把9个library均覆盖、每library至少50个Tier1细胞且identity稳定者列为Primary_DE_ready。",
        "",
        "## 5 Broad-cell aging response",
        _table_or_text(population, ["population", "n_aging_primary_observed", "whole_all_spearman_observed", "whole_all_cosine_similarity_observed", "population_evidence_level"], 20),
        "",
        "## 6 MRJP1 broad-cell reversal",
        "Granulosa与Stromal_fibroblast的whole-transcriptome aging vs treatment方向最稳定且真实标签rank 1/20；Immune、Ovarian_epithelial和Smooth_muscle_pericyte为次级/探索性；Vascular_endothelial证据较弱；Theca_steroidogenic没有稳定population-level exact-permutation支持。这些结果表示MRJP1-associated transcriptomic reversal，不等同于功能性年轻化。",
        "",
        "## 7 Hallmark pathway reversal",
        f"共有{len(strong_pathways)}条strong multi-metric population-pathway证据。",
        _table_or_text(strong_pathways, ["population", "pathway", "aging_NES", "treatment_NES", "residual_NES", "rho_perm_rank", "cosine_perm_rank", "pathway_evidence_level"], 30),
        "",
        "## 8 Subtype localization",
        f"有{len(localized)}条broad pathway–subtype组合达到既定localization支持。",
        _table_or_text(localized, ["broad_population", "subtype", "pathway", "aging_NES", "treatment_NES", "localized_GSEA_support"], 30),
        "",
        "## 9 Composition vs intrinsic decomposition",
        "下表为linear CPM空间的Shapley双因素分解。projection_fraction_of_total是几何投影，不是可直接解释的生物学百分比；组成与亚型内表达可同向、反向或相互抵消。",
        _table_or_text(decomposition, ["broad_population", "contrast", "effect_source", "vector_norm", "projection_fraction_of_total", "cosine_with_total", "n_subtypes"], 30),
        "",
        "library-level Aitchison距离用于描述OT组成是否更接近Y，n=3/group不夸大显著性：",
        _table_or_text(composition_distance, ["broad_population", "analysis_set", "library_id", "group", "aitchison_distance_to_young"], 30),
        "",
        "## 10 Aging/rejuvenation geometry",
        "aging axis完全由Y/OC训练，OT不参与轴定义；Y/OC使用leave-one-library-out，OT另用leave-pair-out敏感性。数值只称为向young reference的转录投影。",
        _table_or_text(geometry, None, 20),
        "Mahalanobis距离因仅3个young libraries无法稳定估计协方差而预先跳过。",
        "",
        "## 11 External validation",
        _table_or_text(external, None, 20),
        "若Stage 6为SKIPPED，表示无法可靠获得/解析带sample metadata的processed reference，不会用不可信镜像或强制batch correction制造一致性。",
        "",
        "## 12 Regulatory programs",
        "TF活性来自CollecTRI靶基因模式，PROGENy来自响应基因；不以TF mRNA替代TF activity。",
        _table_or_text(top_regulators, ["analysis_level", "population", "activity_type", "program", "aging_effect_OC_minus_Y", "treatment_effect_OT_minus_OC", "residual_effect_OT_minus_Y", "observed_rank_high_is_more_opposed"], 30),
        "",
        "## 13 Data-driven programs",
        "NMF在Granulosa与Stromal分开、按library平衡抽样、仅2,000 HVG、多K多seed；正式比较聚合到library。",
        _table_or_text(top_programs, ["population", "program", "aging_effect_OC_minus_Y", "treatment_effect_OT_minus_OC", "residual_effect_OT_minus_Y", "opposition_magnitude"], 20),
        "",
        "## 14 Cell-cell communication",
        "只在前序gate通过时分析预定义Stromal→Granulosa方向，同时报告sender/receiver丰度。所有结果均为communication hypothesis。",
        _table_or_text(communication, ["sender", "receiver", "ligand", "receptor", "n_supported_receiver_targets", "n_stage2_leading_edge_targets", "interpretation"], 30),
        "",
        "## 15 Leading-edge genes",
        tier_counts.to_markdown(index=False) if not tier_counts.empty else "没有可用的候选tier表。",
        "",
        _table_or_text(shortlist, ["cell_type", "gene", "aging_LFC", "treatment_LFC", "residual_LFC", "evidence_tier", "stage3_localized_subtypes", "stage7_candidate_TFs", "stage9_communication_role", "suggested_validation"], 40),
        "",
        "## 16 Phenotype validation framework",
        _table_or_text(phenotype, None, 20),
        "当前没有确认的matched individual-level phenotype，因此没有计算虚假的转录组–表型相关。",
        "",
        "## 17 当前最强证据",
        "- LEVEL A（当前数据直接支持）：library-level raw-count pseudobulk、统一模型、exact label permutation以及复现的Granulosa/Stromal broad反向方向。",
        "- LEVEL B（统计支持的转录推断）：Mouse Hallmark、subtype localization、composition/intrinsic和cross-validated aging-axis结果。",
        "- LEVEL C（多分析支持的机制假说）：靶基因推断TF、data-driven programs与定向Stromal→Granulosa ligand-target组合。",
        "- LEVEL D（必须实验验证）：具体上游TF直接调控、配体分泌/受体激活、氧化应激或卵巢功能改善，以及任何‘rejuvenation’结论。",
        "",
        "## 18 Negative / unsupported results",
        "Theca_steroidogenic缺少稳定population-level exact-permutation支持；不是所有broad population或pathway都被MRJP1反向。OT vs Y不显著不被机械定义为恢复。Mahalanobis因不稳定跳过；任何optional stage的SKIPPED状态均保留在上表。",
        "",
        "## 19 Limitations",
        "每组n=3；文库可能是pool而非单只动物；横断面设计不能给出真实时间方向；知识库具有组织情境偏差；scRNA捕获与dropout影响稀有表达；组成分解是model-based而非因果；外部reference年龄、平台与注释可能不完全一致。",
        "",
        "## 20 推荐后续实验",
        "优先进行同一动物配对的卵泡计数、AMH/FSH/E2、ROS/MDA/SOD/ATP/MMP、p16/p21/γH2AX、Ki67/BrdU、TUNEL/caspase与IL-1β/IL-6/TNFα；对短名单做细胞类型定位、蛋白/磷酸化、共培养/条件培养基和阻断/功能干预。",
        "",
        "## 21 推荐论文主图结构",
        "详见PAPER_FIGURE_PLAN_CN.md。主线应以Granulosa和Stromal的可复现转录反向为核心，subtype、composition/intrinsic和sample-level geometry为结构性证据；regulatory与communication放在机制假说层，功能实验负责升级结论。",
        "",
        "## 22 下一步计算分析是否还值得做",
        "在缺少新增样本/表型/验证实验时，继续扩大算法数量的边际价值有限。优先级应从‘更多计算’转向：确认候选蛋白与功能、增加biological replicates、获得matched phenotype，并在新数据中预注册复现当前最强结论。",
    ]
    (results / "FINAL_ADVANCED_ANALYSIS_HANDOFF_CN.md").write_text("\n".join(cn), encoding="utf-8")

    en = [
        "# Final advanced analysis handoff: ovarian aging and MRJP1 scRNA-seq",
        "",
        "## Design and statistical boundary",
        "Y: 4-month vehicle; OC: 10-month vehicle; OT: 10-month MRJP1 (200 mg/kg); three independent libraries per group. The biological replicate is the library. Differential expression used raw-integer pseudobulk counts; Harmony was not used for inference.",
        "",
        "## Stage status",
        status.to_markdown(index=False),
        "",
        "## Main conclusion",
        "Granulosa and Stromal_fibroblast show the most reproducible MRJP1-associated transcriptomic reversal. This wording does not imply proven phenotypic rejuvenation. Secondary populations are less stable, and Theca_steroidogenic lacks stable population-level exact-permutation support.",
        "",
        "## Hallmark and subtype localization",
        _table_or_text(strong_pathways, ["population", "pathway", "aging_NES", "treatment_NES", "pathway_evidence_level"], 30),
        "",
        _table_or_text(localized, ["broad_population", "subtype", "pathway", "localized_GSEA_support"], 30),
        "",
        "## Composition, intrinsic expression, and aging geometry",
        _table_or_text(decomposition, ["broad_population", "contrast", "effect_source", "projection_fraction_of_total", "cosine_with_total"], 30),
        "",
        _table_or_text(geometry, None, 20),
        "Projection values describe movement toward a young transcriptomic reference and are not a rejuvenation percentage.",
        "",
        "## Regulatory, de novo program, and communication hypotheses",
        _table_or_text(top_regulators, ["population", "activity_type", "program", "aging_effect_OC_minus_Y", "treatment_effect_OT_minus_OC"], 20),
        "",
        _table_or_text(top_programs, ["population", "program", "aging_effect_OC_minus_Y", "treatment_effect_OT_minus_OC"], 15),
        "",
        _table_or_text(communication, ["sender", "receiver", "ligand", "receptor", "interpretation"], 20),
        "",
        "## Candidate shortlist",
        _table_or_text(shortlist, ["cell_type", "gene", "evidence_tier", "suggested_validation"], 40),
        "",
        "## Evidence levels and limitations",
        "Level A: directly supported sample-level transcriptomic results. Level B: statistically supported transcriptomic inference. Level C: multi-analysis mechanistic hypotheses. Level D: claims requiring experiments. Major limitations are n=3 libraries/group, cross-sectional design, 0.05 exact-permutation resolution, knowledge-base context dependence, and the absence of matched phenotypes.",
        "",
        "## Recommended next work",
        "Prioritize matched ovarian reserve/endocrine, oxidative/mitochondrial, senescence/DNA-damage, proliferation, apoptosis, inflammatory, and fertility assays, followed by targeted perturbation of the highest-tier regulators or ligand-receptor hypotheses.",
    ]
    (results / "FINAL_ADVANCED_ANALYSIS_HANDOFF_EN.md").write_text("\n".join(en), encoding="utf-8")

    learned = [
        "# 我们从这组数据学到了什么？",
        "",
        "## 一开始的问题",
        "我们想知道：正常卵巢衰老时哪些细胞和基因程序发生变化，MRJP1处理后这些变化是否朝年轻组方向移动。",
        "",
        "## 为什么不能把十万多个细胞当成十万个重复？",
        "同一文库里的细胞共享同一次动物处理和制备过程，不是独立动物。因此正式比较先把同一种细胞的UMI按library相加，再用每组3个library比较。这样更保守，但更可信。",
        "",
        "## permutation为什么会让一些漂亮结果降级？",
        "只有6个老龄文库要分成3个OC和3个OT，恰好有20种分法。我们把真实分组和其余19种都算一遍。如果真实标签不是最极端，说明结果也可能由这6个文库的自然波动产生。最低可能p是1/20=0.05，所以‘rank 1/20’比普通小p值更直观。",
        "",
        "## 目前最可信的细胞",
        "Granulosa和Stromal_fibroblast最稳定：它们在整体基因变化、exact permutation和多条通路上都显示衰老方向与MRJP1方向相反。Theca没有同等稳定支持，不能因为它生物学上重要就强行写入主线。",
        "",
        "## subtype又告诉了什么？",
        f"Stage 3把broad信号向下定位，共得到{len(localized)}条达到既定支持的pathway–subtype组合。这样可以区分‘所有亚型共同变化’和‘少数亚型主导’，但稀有或低置信亚型只保留描述。",
        "",
        "## composition和expression有什么区别？",
        "一种总体变化可能来自某亚型变多/变少（composition），也可能是同一亚型内部基因表达改变（cell-intrinsic）。Stage 4把两者在一个明确的数学模型里拆开；它帮助理解信号来源，但不是因果实验，也不能把投影值当成真实贡献百分比。",
        "",
        "## sample-level状态空间说明什么？",
        "Stage 5只用9个library建立衰老方向，并让被评价样本不参与定义自己的轴。如果OT比OC更靠近Y，只能说MRJP1相关转录状态向年轻参考方向移动，不能说卵巢已经年轻化。",
        "",
        "## 上游机制和细胞通讯到什么程度？",
        "TF活性、NMF程序和Stromal→Granulosa配体–受体分析都是由表达模式推断的候选机制。它们适合帮助选实验，不是TF结合、蛋白分泌或受体激活的直接证据。",
        "",
        "## MRJP1现在到底可以说什么？",
        "可以说：在当前9个library中，MRJP1与Granulosa和Stromal若干衰老相关转录程序的部分、细胞类型/亚型特异性反向变化相关。",
        "",
        "## 还不能说什么？",
        "不能说已经证明卵巢功能恢复、逆转生物年龄或确定了唯一作用机制。要升级这些结论，需要同一动物的卵泡、激素、氧化应激、线粒体、衰老、凋亡和生育力数据，以及对候选TF/配体的功能干预。",
    ]
    (results / "WHAT_HAVE_WE_LEARNED_CN.md").write_text("\n".join(learned), encoding="utf-8")

    flow = """# 高级分析流程图

```mermaid
flowchart TD
    A[Filtered Cell Ranger matrices\n9 libraries] --> B[Input audit and QC]
    B --> C[Annotation v2\n105,763 cells]
    C --> D[Raw-count library pseudobulk]
    D --> E[Unified 9-library DE]
    E --> F[20 exact OC/OT permutations]
    F --> G[Mouse Hallmark reversal]
    G --> H[Stage 3 subtype localization]
    H --> I[Stage 4 composition vs intrinsic]
    I --> J[Stage 5 cross-validated aging geometry]
    J --> K[Stage 6 external aging validation]
    K --> L[Stage 7 TF and pathway activity]
    L --> M[Stage 8 lineage-specific NMF programs]
    M --> N{Prior evidence gate}
    N -->|pass| O[Stage 9 targeted Stromal to Granulosa communication]
    N -->|fail| P[Formal communication skip]
    O --> Q[Stage 10 evidence matrix]
    P --> Q
    Q --> R[Stage 11 prospective phenotype validation]
    R --> S[Stage 12 final synthesis]
```

所有正式DE和组间推断的统计单位均为library；Harmony仅用于representation，cell-level结果不作为独立重复。
"""
    (results / "ADVANCED_ANALYSIS_FLOWCHART.md").write_text(flow, encoding="utf-8")

    figure_plan = [
        "# 论文主图建议",
        "",
        "## Figure 1｜实验设计、数据质量与卵巢cell atlas",
        "9个library设计、QC保留、annotation v2 UMAP、每library细胞组成。结论：数据结构与重复层级清楚。",
        "",
        "## Figure 2｜自然衰老的cell-type-resolved landscape",
        "OC vs Y pseudobulk DE、sample PCA、主要Hallmark。结论：衰老变化集中在哪些population/program。",
        "",
        "## Figure 3｜MRJP1 broad transcriptomic reversal",
        "Granulosa/Stromal aging-vs-treatment散点、exact permutation rank、negative populations并列。结论：反向方向可复现但population-specific。",
        "",
        "## Figure 4｜Stromal机制主线",
        "Stromal strong Hallmarks、stable subtypes、composition/intrinsic decomposition、TF/PROGENy与验证候选。结论必须停留在转录机制假说。",
        "",
        "## Figure 5｜Granulosa subtype localization",
        "preantral/antral等Primary subtypes的pathway定位、sample-level projection与候选genes。",
        "",
        "## Figure 6｜Cross-cell niche hypothesis",
        "只有Stage 9 gate通过时展示Stromal→Granulosa候选ligand–receptor–target链，并在同一图报告sender/receiver abundance；若Stage 9跳过，改为composition/intrinsic与NMF programs。",
        "",
        "## Figure 7｜表型验证与整合模型",
        "用证据等级标注的整合模型连接transcriptomic programs与未来AMH/FSH/E2、ROS/MDA/SOD/ATP/MMP、p16/p21/γH2AX、Ki67/TUNEL、炎症和生育力实验。没有表型数据前作为study design figure而非结果图。",
        "",
        "## Supplementary",
        "完整QC、所有library pseudobulk QC、20 permutations、全部pathway/subtype、NMF K/seed稳定性、资源checksum、negative results和optional-stage skip理由。",
    ]
    (results / "PAPER_FIGURE_PLAN_CN.md").write_text("\n".join(figure_plan), encoding="utf-8")

    status.to_csv(output_root / "stage_status.tsv", sep="\t", index=False)
    final_files = [
        results / "FINAL_ADVANCED_ANALYSIS_HANDOFF_CN.md",
        results / "FINAL_ADVANCED_ANALYSIS_HANDOFF_EN.md",
        results / "WHAT_HAVE_WE_LEARNED_CN.md",
        results / "ADVANCED_ANALYSIS_FLOWCHART.md",
        results / "PAPER_FIGURE_PLAN_CN.md",
        output_root / "stage_status.tsv",
    ]
    manifest = pd.DataFrame([{"path": str(p.relative_to(paths["root"])), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in final_files])
    manifest.to_csv(output_root / "manifest.tsv", sep="\t", index=False)
    complete = {
        "stage": 12,
        "status": "COMPLETE",
        "git_commit": git_commit,
        "successful_stages": status.loc[status["status"].eq("COMPLETE"), "stage"].tolist(),
        "skipped_stages": status.loc[status["status"].eq("SKIPPED"), "stage"].tolist(),
        "failed_stages": status.loc[status["status"].eq("FAILED"), "stage"].tolist(),
        "manifest_sha256": sha256_file(output_root / "manifest.tsv"),
    }
    (output_root / "COMPLETE.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    logger.info("Stage 12 complete")
    print("========================================")
    print("OVARY_ADVANCED_ANALYSIS_COMPLETE")
    print("========================================")
    print(f"FINAL_GIT_COMMIT={git_commit}")
    print("SUCCESSFUL_STAGES=" + ",".join(map(str, complete["successful_stages"])))
    print("SKIPPED_STAGES=" + ",".join(map(str, complete["skipped_stages"])))
    print("FAILED_STAGES=" + ",".join(map(str, complete["failed_stages"])))
    print("FINAL_ADVANCED_ANALYSIS_HANDOFF_CN=results/FINAL_ADVANCED_ANALYSIS_HANDOFF_CN.md")
    print("WHAT_HAVE_WE_LEARNED_CN=results/WHAT_HAVE_WE_LEARNED_CN.md")
    print("PAPER_FIGURE_PLAN_CN=results/PAPER_FIGURE_PLAN_CN.md")
