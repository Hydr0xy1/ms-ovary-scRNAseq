"""Stage 10: transparent evidence-matrix mechanism candidate prioritization."""
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .project import project_paths, setup_logging
from .stage7_regulatory_activity import sha256_file


def assign_evidence_tier(table: pd.DataFrame) -> pd.Series:
    """Assign predefined evidence tiers without a weighted composite score."""

    independent = (
        table[[
            "stage2_leading_edge",
            "stage3_subtype_localized",
            "external_aging_supported",
            "stage7_supported_tf_target",
            "stage9_communication_member",
        ]]
        .fillna(False)
        .astype(bool)
        .sum(axis=1)
    )
    tier_a = (
        table["stage1_6_level2_de_supported"].fillna(False).astype(bool)
        & table["stage1_6_population_signature_supported"].fillna(False).astype(bool)
        & independent.ge(2)
    )
    tier_b = (
        table["stage1_6_level1_directional"].fillna(False).astype(bool)
        & table["stage2_leading_edge"].fillna(False).astype(bool)
        & table["stage3_subtype_localized"].fillna(False).astype(bool)
    )
    result = pd.Series("Tier C exploratory", index=table.index, dtype=object)
    result.loc[tier_b] = "Tier B DE_pathway_subtype"
    result.loc[tier_a] = "Tier A multi_layer"
    return result


def _leading_edge_map(path: Path) -> dict[str, set[str]]:
    table = pd.read_csv(path, sep="\t")
    mapping: dict[str, set[str]] = {}
    for population, sub in table.groupby("population", observed=True):
        genes: set[str] = set()
        for column in ["shared_opposite_direction_genes", "aging_leading_edge", "treatment_leading_edge"]:
            if column in sub:
                for value in sub[column].dropna().astype(str):
                    genes.update(x for x in value.split(";") if x and x.lower() != "nan")
        mapping[str(population)] = genes
    return mapping


def _subtype_map(path: Path) -> dict[tuple[str, str], list[str]]:
    table = pd.read_csv(path, sep="\t")
    supported = table.loc[
        table["direction_opposite"].fillna(False).astype(bool)
        & table["residual_closer_to_y"].fillna(False).astype(bool)
    ]
    return (
        supported.groupby(["broad_population", "gene"], observed=True)["subtype"]
        .agg(lambda x: sorted(set(map(str, x))))
        .to_dict()
    )


def _external_map(stage6_root: Path) -> set[tuple[str, str]]:
    path = stage6_root / "external_signature_reversal.tsv"
    if not path.exists():
        return set()
    table = pd.read_csv(path, sep="\t")
    population_col = "population" if "population" in table else "internal_population"
    support_col = next((c for c in ["external_aging_supported", "direction_concordant_external_aging"] if c in table), None)
    if support_col is None or "gene" not in table:
        return set()
    return set(
        map(
            tuple,
            table.loc[table[support_col].fillna(False).astype(bool), [population_col, "gene"]]
            .astype(str)
            .itertuples(index=False, name=None),
        )
    )


def _tf_target_map(stage7_root: Path) -> dict[tuple[str, str], list[str]]:
    candidates_path = stage7_root / "candidate_regulators.tsv"
    resource_path = stage7_root / "resources" / "collectri_mouse.tsv.gz"
    if not candidates_path.exists() or not resource_path.exists():
        return {}
    candidates = pd.read_csv(candidates_path, sep="\t")
    candidates = candidates.loc[candidates["activity_type"].astype(str).eq("TF")]
    if "observed_rank_high_is_more_opposed" in candidates:
        candidates = candidates.loc[
            candidates["observed_rank_high_is_more_opposed"].isna()
            | candidates["observed_rank_high_is_more_opposed"].le(3)
        ]
    network = pd.read_csv(resource_path, sep="\t")
    rows: dict[tuple[str, str], list[str]] = {}
    for row in candidates.itertuples(index=False):
        targets = network.loc[network["source"].astype(str).eq(str(row.program)), "target"].astype(str)
        for gene in targets:
            rows.setdefault((str(row.population), gene), []).append(str(row.program))
    return {key: sorted(set(value)) for key, value in rows.items()}


def _communication_sets(stage9_root: Path) -> tuple[set[str], set[str], set[str]]:
    ligand_path = stage9_root / "candidate_ligands.tsv"
    receptor_path = stage9_root / "candidate_receptors.tsv"
    target_path = stage9_root / "ligand_target_links.tsv"
    ligands = set(pd.read_csv(ligand_path, sep="\t")["ligand"].astype(str)) if ligand_path.exists() else set()
    receptors = set(pd.read_csv(receptor_path, sep="\t")["receptor"].astype(str)) if receptor_path.exists() else set()
    targets = set(pd.read_csv(target_path, sep="\t")["target_gene"].astype(str)) if target_path.exists() else set()
    ligand_genes = {part for value in ligands for part in value.split("_")}
    receptor_genes = {part for value in receptors for part in value.split("_")}
    return ligand_genes, receptor_genes, targets


def run_stage10(config: Mapping[str, Any]) -> None:
    paths = project_paths(config)
    output_root = paths["results"] / "stage10_candidates"
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("19_stage10_candidates", dict(config))
    stage1_6 = paths["results"] / "de_stage1_6"
    populations = sorted(p.name for p in stage1_6.iterdir() if p.is_dir() and (p / "observed_evidence_levels.tsv.gz").exists())
    leading = _leading_edge_map(paths["results"] / "pathway_stage2" / "hallmark_leading_edge_review.tsv")
    subtype = _subtype_map(paths["results"] / "stage3_subtype_localization" / "subtype_reversal.tsv")
    external = _external_map(paths["results"] / "stage6_external_validation")
    tf_targets = _tf_target_map(paths["results"] / "stage7_regulatory_activity")
    ligands, receptors, communication_targets = _communication_sets(paths["results"] / "stage9_communication")

    gene_rows: list[pd.DataFrame] = []
    for population in populations:
        table = pd.read_csv(stage1_6 / population / "observed_evidence_levels.tsv.gz", sep="\t")
        table = table.loc[
            table["Level_1_directional_candidate"].fillna(False).astype(bool)
            | table["Level_2_DE_supported_candidate"].fillna(False).astype(bool)
        ].copy()
        table = table.rename(
            columns={
                "aging_effect": "aging_LFC",
                "aging_padj": "aging_FDR",
                "treatment_effect": "treatment_LFC",
                "treatment_padj": "treatment_FDR",
                "residual_effect": "residual_LFC",
                "residual_padj": "residual_FDR",
                "opposite_direction": "directionally_reversed",
                "Level_1_directional_candidate": "stage1_6_level1_directional",
                "Level_2_DE_supported_candidate": "stage1_6_level2_de_supported",
                "Level_3_population_signature_supported": "stage1_6_population_signature_supported",
            }
        )
        table["cell_type"] = population
        table["stage2_leading_edge"] = table["gene"].astype(str).isin(leading.get(population, set()))
        table["stage3_localized_subtypes"] = [
            ";".join(subtype.get((population, str(gene)), [])) for gene in table["gene"]
        ]
        table["stage3_subtype_localized"] = table["stage3_localized_subtypes"].str.len().gt(0)
        table["external_aging_supported"] = [(population, str(gene)) in external for gene in table["gene"]]
        table["stage7_candidate_TFs"] = [";".join(tf_targets.get((population, str(gene)), [])) for gene in table["gene"]]
        table["stage7_supported_tf_target"] = table["stage7_candidate_TFs"].str.len().gt(0)
        if population == "Stromal_fibroblast":
            communication_role = ["ligand" if str(gene) in ligands else "" for gene in table["gene"]]
        elif population == "Granulosa":
            communication_role = [
                "receptor" if str(gene) in receptors else ("target" if str(gene) in communication_targets else "")
                for gene in table["gene"]
            ]
        else:
            communication_role = [""] * len(table)
        table["stage9_communication_role"] = communication_role
        table["stage9_communication_member"] = table["stage9_communication_role"].str.len().gt(0)
        gene_rows.append(table)

    genes = pd.concat(gene_rows, ignore_index=True)
    genes["evidence_tier"] = assign_evidence_tier(genes)
    core = [
        "cell_type", "gene", "aging_LFC", "aging_FDR", "treatment_LFC", "treatment_FDR",
        "residual_LFC", "residual_FDR", "directionally_reversed",
        "stage1_6_level1_directional", "stage1_6_level2_de_supported",
        "stage1_6_population_signature_supported", "stage2_leading_edge",
        "stage3_subtype_localized", "stage3_localized_subtypes", "external_aging_supported",
        "stage7_supported_tf_target", "stage7_candidate_TFs", "stage9_communication_member",
        "stage9_communication_role", "evidence_tier", "primary_classification",
    ]
    for column in core:
        if column not in genes:
            genes[column] = np.nan
    genes = genes[core].sort_values(
        ["evidence_tier", "stage1_6_level2_de_supported", "stage2_leading_edge", "stage3_subtype_localized", "aging_FDR", "gene"],
        ascending=[True, False, False, False, True, True],
        kind="stable",
    )
    genes.to_csv(output_root / "candidate_gene_evidence.tsv", sep="\t", index=False)

    pathways = pd.read_csv(paths["results"] / "pathway_stage2" / "primary_pathway_reversal_summary.tsv", sep="\t")
    localization = pd.read_csv(paths["results"] / "stage3_subtype_localization" / "broad_to_subtype_localization.tsv", sep="\t")
    localized = localization.loc[localization["localized_GSEA_support"].fillna(False).astype(bool)].groupby(
        ["broad_population", "pathway"], observed=True
    )["subtype"].agg(lambda x: ";".join(sorted(set(map(str, x))))).rename("localized_subtypes").reset_index()
    pathways = pathways.merge(
        localized,
        left_on=["population", "pathway"],
        right_on=["broad_population", "pathway"],
        how="left",
    ).drop(columns=["broad_population"], errors="ignore")
    pathways["localized_subtypes"] = pathways["localized_subtypes"].fillna("")
    pathways.to_csv(output_root / "candidate_pathway_evidence.tsv", sep="\t", index=False)

    communication_path = paths["results"] / "stage9_communication" / "communication_evidence.tsv"
    crosscell = pd.read_csv(communication_path, sep="\t") if communication_path.exists() else pd.DataFrame(
        columns=["sender", "receiver", "ligand", "receptor", "interpretation"]
    )
    crosscell.to_csv(output_root / "candidate_crosscell_evidence.tsv", sep="\t", index=False)

    shortlist = genes.loc[genes["evidence_tier"].isin(["Tier A multi_layer", "Tier B DE_pathway_subtype"])].copy()
    shortlist = (
        shortlist.groupby("cell_type", observed=True, group_keys=False)
        .head(20)
        .reset_index(drop=True)
    )
    shortlist["suggested_validation"] = np.select(
        [
            shortlist["stage9_communication_role"].eq("ligand"),
            shortlist["stage9_communication_role"].eq("receptor"),
            shortlist["stage7_supported_tf_target"].fillna(False).astype(bool),
        ],
        [
            "secreted protein/conditioned medium/blocking experiment",
            "receptor protein/phosphorylation/blockade",
            "target expression plus candidate TF activity perturbation",
        ],
        default="qPCR/protein plus population-specific functional validation",
    )
    shortlist.to_csv(output_root / "experimental_validation_shortlist.tsv", sep="\t", index=False)

    tier_counts = genes.groupby(["cell_type", "evidence_tier"], observed=True).size().rename("n_genes").reset_index()
    report = [
        "# Stage 10 机制候选优先级报告",
        "",
        "## 1. 为什么做？",
        "把DE、exact permutation、Hallmark leading edge、subtype定位、外部衰老、TF靶基因和定向通讯证据放入同一矩阵，形成可验证短名单。",
        "",
        "## 2. 方法与统计单位",
        "基础DE来自raw integer pseudobulk，以library为重复。没有给不同证据任意加权，也没有把single cell当重复。排序使用预定义的证据层级与布尔证据列。",
        "",
        "## 3. Evidence tiers",
        "- Tier A：Stage1.6 DE-supported且population signature受置换支持，并至少再有两类独立证据。",
        "- Tier B：方向性候选，同时属于Stage2 leading edge并在Stage3亚型中定位。",
        "- Tier C：只有较少或单一证据来源，保留为探索性。",
        "",
        tier_counts.to_markdown(index=False),
        "",
        "## 4. 优先验证短名单",
        shortlist.to_markdown(index=False),
        "",
        "## 5. 可以与不能说明什么？",
        "该矩阵能显示每个候选由哪些独立分析支持，便于选择实验；它不是因果模型，也不把预测TF/通信证据当作直接机制。Tier高表示证据来源更多且一致，不表示效应必然更大。",
        "",
        "## 6. 主要局限",
        "每组n=3，exact permutation最低分辨率0.05；external/Stage9若技术性跳过，相应证据列保持False而不是臆测；不同知识库存在组织情境偏差。",
        "",
        "## 7. 下一步",
        "用Stage11把候选program映射到可测表型，并优先选择Granulosa、Stromal和cross-cell候选做正交验证。",
    ]
    (output_root / "MECHANISM_CANDIDATE_REPORT_CN.md").write_text("\n".join(report), encoding="utf-8")

    outputs = [p for p in output_root.rglob("*") if p.is_file() and p.name not in {"manifest.tsv", "COMPLETE.json"}]
    manifest = pd.DataFrame([{"path": str(p.relative_to(paths["root"])), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in outputs])
    manifest.to_csv(output_root / "manifest.tsv", sep="\t", index=False)
    complete = {
        "stage": 10,
        "status": "COMPLETE",
        "statistical_unit": "library",
        "n_candidates": len(genes),
        "n_shortlist": len(shortlist),
        "manifest_sha256": sha256_file(output_root / "manifest.tsv"),
    }
    (output_root / "COMPLETE.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    logger.info("Stage 10 complete: %d candidates, %d shortlist", len(genes), len(shortlist))
    print("STAGE10_CANDIDATE_PRIORITIZATION_COMPLETE")
