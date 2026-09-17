"""Stage 11: prospective phenotype validation framework (no fabricated correlations)."""

from __future__ import annotations

import json
from typing import Any, Mapping

import pandas as pd

from .project import project_paths, setup_logging
from .stage7_regulatory_activity import sha256_file


PROGRAM_ASSAYS = [
    ("OXIDATIVE_PHOSPHORYLATION|ROS_PATHWAY", "oxidative stress / mitochondria", "ROS;MDA;T-AOC;SOD;GSH-Px;ATP;MMP", "biochemical assay plus tissue/cell localization"),
    ("DNA_REPAIR|P53_PATHWAY|UV_RESPONSE", "DNA damage / senescence", "γH2AX;p16;p21;SA-β-gal", "protein/immunostaining plus functional senescence assay"),
    ("INFLAMMATORY_RESPONSE|TNFA_SIGNALING_VIA_NFKB|COMPLEMENT", "inflammation / SASP", "IL-1β;IL-6;TNFα;NF-κB activation", "serum/tissue cytokine plus phospho-protein"),
    ("G2M_CHECKPOINT|MITOTIC_SPINDLE|MYC_TARGETS", "proliferation", "Ki67;BrdU/EdU;cell-cycle protein", "cell-type-resolved immunostaining"),
    ("APOPTOSIS", "apoptosis", "TUNEL;cleaved caspase-3/7;BAX/BCL2", "histology plus protein/activity assay"),
    ("ESTROGEN_RESPONSE|ANDROGEN_RESPONSE|STEROID", "ovarian endocrine function", "AMH;FSH;E2;steroidogenic proteins", "serum hormone plus ovarian protein"),
    ("PROTEIN_SECRETION|UNFOLDED_PROTEIN_RESPONSE", "secretory / proteostasis", "secretome panel;ER-stress proteins", "conditioned medium/proteomics plus protein validation"),
    ("MYOGENESIS|APICAL_JUNCTION|ECM", "stromal/ECM remodeling", "collagen/fibronectin;fibrosis staining;matrix organization", "Masson/Sirius red plus ECM protein"),
]


def map_pathway_to_assays(pathway: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    upper = str(pathway).upper()
    for pattern, phenotype, assays, modality in PROGRAM_ASSAYS:
        if any(token in upper for token in pattern.split("|")):
            rows.append(
                {
                    "pathway": pathway,
                    "phenotype_domain": phenotype,
                    "recommended_assays": assays,
                    "recommended_modality": modality,
                }
            )
    return rows


def run_stage11(config: Mapping[str, Any]) -> None:
    paths = project_paths(config)
    output_root = paths["results"] / "stage11_phenotype_framework"
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(paths["logs"] / "stage11_phenotype_framework.log")
    pathway_path = paths["results"] / "stage10_candidates" / "candidate_pathway_evidence.tsv"
    pathways = pd.read_csv(pathway_path, sep="\t")
    map_rows: list[dict[str, Any]] = []
    for row in pathways.itertuples(index=False):
        for mapped in map_pathway_to_assays(str(row.pathway)):
            mapped.update(
                {
                    "population": str(row.population),
                    "pathway_evidence_level": str(getattr(row, "pathway_evidence_level", "")),
                    "aging_NES": getattr(row, "aging_NES", None),
                    "treatment_NES": getattr(row, "treatment_NES", None),
                    "localized_subtypes": str(getattr(row, "localized_subtypes", "")),
                    "interpretation": "prospective_validation_mapping_not_observed_correlation",
                }
            )
            map_rows.append(mapped)
    mapping = pd.DataFrame(map_rows).drop_duplicates()
    mapping.to_csv(output_root / "transcriptome_phenotype_map.tsv", sep="\t", index=False)

    assays = pd.DataFrame(
        [
            ("ovarian reserve", "follicle counts;AMH", "histology plus serum", "essential", "tests whether transcriptomic response accompanies reserve/function"),
            ("gonadal axis", "FSH;E2", "serum", "essential", "endocrine consequence"),
            ("oxidative stress", "ROS;MDA;T-AOC;SOD;GSH-Px", "ovary homogenate and cell-resolved follow-up", "essential", "tests OXPHOS/ROS inference"),
            ("mitochondria", "ATP;MMP", "fresh ovarian cells/tissue", "high", "direct functional complement to transcriptomic metabolism"),
            ("senescence/DNA damage", "SA-β-gal;p16;p21;γH2AX", "histology/protein", "essential", "distinguishes senescence from generic stress"),
            ("proliferation", "Ki67;BrdU/EdU", "cell-type-resolved histology", "high", "tests G2M/mitotic programs"),
            ("apoptosis", "TUNEL;cleaved caspase", "histology/protein", "high", "tests apoptosis inference"),
            ("inflammation/SASP", "IL-1β;IL-6;TNFα", "serum plus ovary", "essential", "tests stromal inflammatory hypothesis"),
            ("fertility", "estrous cycle;mating rate;litter size", "longitudinal animal outcome", "highest biological value", "required before phenotypic rejuvenation claim"),
        ],
        columns=["phenotype_domain", "assays", "sample_or_modality", "priority", "rationale"],
    )
    assays.to_csv(output_root / "recommended_validation_assays.tsv", sep="\t", index=False)

    shortlist_path = paths["results"] / "stage10_candidates" / "experimental_validation_shortlist.tsv"
    shortlist = pd.read_csv(shortlist_path, sep="\t")
    if shortlist.empty:
        priority = pd.DataFrame(columns=["cell_type", "candidate", "evidence_tier", "molecular_validation", "paired_phenotype", "required_pairing"])
    else:
        priority = shortlist[["cell_type", "gene", "evidence_tier", "suggested_validation"]].rename(
            columns={"gene": "candidate", "suggested_validation": "molecular_validation"}
        )
        priority["paired_phenotype"] = priority["cell_type"].map(
            {
                "Granulosa": "follicle stage/count;AMH/E2;proliferation/apoptosis",
                "Stromal_fibroblast": "ECM/fibrosis;SASP cytokines;oxidative stress",
            }
        ).fillna("population-relevant phenotype")
        priority["required_pairing"] = "same animal/library or prospectively matched biological replicate"
    priority.to_csv(output_root / "priority_validation_matrix.tsv", sep="\t", index=False)

    report = [
        "# Stage 11 转录组—表型整合框架",
        "",
        "## 1. 为什么做？",
        "把当前转录程序转化成可执行的动物/组织/蛋白验证方案，并明确哪些表型才足以支持功能改善或卵巢年轻化。",
        "",
        "## 2. 当前输入",
        "输入是Stage10 pathway与gene evidence matrix。当前项目没有确认的同一动物/同一library配对表型值，因此本阶段只建立前瞻性验证对应表。",
        "",
        "## 3. 绝对边界",
        "没有计算Pearson或Spearman相关性，没有声称转录组与AMH、ROS、卵泡数等已相关。未来只有在同一动物或可追溯匹配的生物学重复上测得表型后，才允许做sample-level相关/联合模型。",
        "",
        "## 4. 推荐框架",
        mapping.to_markdown(index=False),
        "",
        "## 5. 推荐优先级",
        assays.to_markdown(index=False),
        "",
        "## 6. 设计建议",
        "尽可能让scRNA-seq、血清激素、组织学、氧化应激和功能结局来自同一动物；预先定义primary endpoints；保留每只动物标识；盲法计数卵泡；在Granulosa/Stromal层面做定位验证。",
        "",
        "## 7. 可以与不能说明什么？",
        "该框架说明下一步测什么、如何匹配证据；不能替代真实表型数据，也不能把transcriptomic reversal写成proven rejuvenation。",
    ]
    (output_root / "PHENOTYPE_INTEGRATION_REPORT_CN.md").write_text("\n".join(report), encoding="utf-8")

    outputs = [p for p in output_root.rglob("*") if p.is_file() and p.name not in {"manifest.tsv", "COMPLETE.json"}]
    manifest = pd.DataFrame([{"path": str(p.relative_to(paths["root"])), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in outputs])
    manifest.to_csv(output_root / "manifest.tsv", sep="\t", index=False)
    complete = {
        "stage": 11,
        "status": "COMPLETE",
        "matched_phenotype_available": False,
        "correlation_computed": False,
        "manifest_sha256": sha256_file(output_root / "manifest.tsv"),
    }
    (output_root / "COMPLETE.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    logger.info("Stage 11 complete: prospective framework only")
    print("STAGE11_PHENOTYPE_FRAMEWORK_COMPLETE")
