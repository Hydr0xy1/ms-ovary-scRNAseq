"""Stage 13: publication-level evidence consolidation and standalone figures.

This stage is deliberately read-only with respect to Stage 1-12 outputs. It
consolidates existing evidence, records traceable source data, and renders
single-axis publication plots. It never re-runs DE, enrichment, integration,
annotation, or clustering.
"""
# ruff: noqa: E501

from __future__ import annotations

import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .project import project_paths, setup_logging
from .stage7_regulatory_activity import sha256_file

GROUP_COLORS = {"Y": "#0072B2", "OC": "#D55E00", "OT": "#009E73"}
CELL_TYPE_COLORS = {
    "Granulosa": "#E69F00",
    "Stromal_fibroblast": "#56B4E9",
    "Theca_steroidogenic": "#CC79A7",
    "Immune": "#009E73",
    "Ovarian_epithelial": "#F0E442",
    "Smooth_muscle_pericyte": "#0072B2",
    "Vascular_endothelial": "#D55E00",
    "Lymphatic_endothelial": "#8B6F47",
    "Ciliated_epithelial": "#7F7F7F",
    "Luteal_candidate": "#A6761D",
    "Uncertain": "#C7C7C7",
}

SUBTYPE_COLORS = {
    "Granulosa_antral_like": "#E69F00",
    "Granulosa_preantral_like": "#F4A6C1",
    "Granulosa_cycling": "#CC79A7",
    "Granulosa_atretic_like": "#D55E00",
    "Granulosa_low_complexity_candidate": "#BDBDBD",
    "ECM_high_candidate": "#56B4E9",
    "Smooth_muscle_candidate": "#0072B2",
    "Stromal_fibroblast_candidate": "#009E73",
}

TIER_A_ANCHORS = [
    ("Stromal_fibroblast", "Il6st"),
    ("Stromal_fibroblast", "Hif1a"),
    ("Stromal_fibroblast", "Fn1"),
    ("Stromal_fibroblast", "Tnc"),
    ("Stromal_fibroblast", "Zfpm2"),
    ("Granulosa", "Prkdc"),
    ("Granulosa", "Bard1"),
    ("Granulosa", "Abi1"),
    ("Granulosa", "Abca1"),
]


def configure_publication_style() -> None:
    """Apply a restrained, editable Nature-style Matplotlib configuration."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "lines.linewidth": 0.9,
            "patch.linewidth": 0.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _read_tsv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", low_memory=False)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _clean_pathway(value: str) -> str:
    return str(value).replace("HALLMARK_", "").replace("_", " ").title()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def assign_publication_candidate_tiers(
    shortlist: pd.DataFrame,
    *,
    tier_a_anchors: list[tuple[str, str]] | None = None,
    tier_b_limit: int = 16,
) -> pd.DataFrame:
    """Compress 60 candidates with an explicit evidence hierarchy.

    There is no weighted composite score. Tier A is a prespecified set of
    biologically interpretable anchors that must also pass all core evidence
    gates. Tier B uses an ordered set of Boolean gates and deterministic
    tie-breakers. All remaining candidates are retained in Tier C.
    """

    table = shortlist.copy()
    anchors = tier_a_anchors or TIER_A_ANCHORS
    anchor_rank = {pair: rank for rank, pair in enumerate(anchors)}
    pairs = list(zip(table["cell_type"].astype(str), table["gene"].astype(str), strict=False))
    core = (
        _as_bool(table["stage1_6_level2_de_supported"])
        & _as_bool(table["stage1_6_population_signature_supported"])
        & _as_bool(table["stage2_leading_edge"])
        & _as_bool(table["stage3_subtype_localized"])
    )
    table["publication_tier"] = "Tier C - exploratory"
    table["tier_rationale"] = "Retained for transparency; insufficient priority for the first validation set."
    tier_a_mask = pd.Series(
        [pair in anchor_rank for pair in pairs], index=table.index
    ) & core
    table.loc[tier_a_mask, "publication_tier"] = "Tier A - immediate validation"
    table.loc[tier_a_mask, "tier_rationale"] = (
        "Strong-population DE, population permutation, Hallmark leading-edge and subtype localization support; "
        "manually retained as an interpretable anchor for stromal or granulosa validation."
    )

    remaining = table.loc[~tier_a_mask].copy()
    remaining["strong_population"] = remaining["cell_type"].isin(
        ["Stromal_fibroblast", "Granulosa"]
    )
    remaining["external_gate"] = _as_bool(remaining["external_aging_supported"])
    remaining["communication_gate"] = _as_bool(remaining["stage9_communication_member"])
    remaining["regulatory_gate"] = _as_bool(remaining["stage7_supported_tf_target"])
    remaining["core_gate"] = core.loc[remaining.index]
    remaining["treatment_FDR_numeric"] = pd.to_numeric(
        remaining["treatment_FDR"], errors="coerce"
    ).fillna(np.inf)
    remaining["population_order"] = remaining["cell_type"].map(
        {"Stromal_fibroblast": 0, "Granulosa": 1, "Immune": 2}
    ).fillna(9)
    ordered = remaining.sort_values(
        [
            "strong_population",
            "core_gate",
            "external_gate",
            "communication_gate",
            "regulatory_gate",
            "population_order",
            "treatment_FDR_numeric",
            "gene",
        ],
        ascending=[False, False, False, False, False, True, True, True],
        kind="stable",
    )
    tier_b_index = ordered.head(tier_b_limit).index
    table.loc[tier_b_index, "publication_tier"] = "Tier B - second priority"
    table.loc[tier_b_index, "tier_rationale"] = (
        "Passes the core multi-layer hierarchy or has orthogonal external, regulatory, or communication support; "
        "placed after the prespecified Tier A anchors."
    )
    table["tier_order"] = table["publication_tier"].map(
        {
            "Tier A - immediate validation": 1,
            "Tier B - second priority": 2,
            "Tier C - exploratory": 3,
        }
    )
    table["anchor_order"] = [anchor_rank.get(pair, 999) for pair in pairs]
    return table.sort_values(
        ["tier_order", "anchor_order", "cell_type", "gene"], kind="stable"
    ).drop(columns=["anchor_order"])


def _leading_edge_memberships(path: Path) -> dict[tuple[str, str], list[str]]:
    table = _read_tsv(path)
    mapping: dict[tuple[str, str], set[str]] = {}
    gene_columns = [
        "aging_leading_edge",
        "treatment_leading_edge",
        "shared_leading_edge_genes",
        "shared_opposite_direction_genes",
    ]
    for row in table.itertuples(index=False):
        population = str(row.population)
        pathway = str(row.pathway)
        for column in gene_columns:
            value = getattr(row, column, "")
            if pd.isna(value):
                continue
            for gene in str(value).split(";"):
                if gene:
                    mapping.setdefault((population, gene), set()).add(pathway)
    return {key: sorted(values) for key, values in mapping.items()}


def _nmf_memberships(path: Path) -> dict[tuple[str, str], list[str]]:
    table = _read_tsv(path)
    return (
        table.groupby(["population", "gene"], observed=True)["program"]
        .agg(lambda values: sorted(set(map(str, values))))
        .to_dict()
    )


def build_candidate_cards(
    shortlist: pd.DataFrame,
    population_evidence: pd.DataFrame,
    leading_edge_path: Path,
    program_gene_path: Path,
) -> pd.DataFrame:
    tiered = assign_publication_candidate_tiers(shortlist)
    leading = _leading_edge_memberships(leading_edge_path)
    nmf = _nmf_memberships(program_gene_path)
    population_map = population_evidence.set_index("population").to_dict("index")
    rows: list[dict[str, Any]] = []
    for row in tiered.itertuples(index=False):
        population = str(row.cell_type)
        gene = str(row.gene)
        pop = population_map.get(population, {})
        pathways = leading.get((population, gene), [])
        programs = nmf.get((population, gene), [])
        communication_role = str(getattr(row, "stage9_communication_role", "") or "")
        if communication_role and communication_role.lower() != "nan":
            feasibility = "High: targeted protein/receptor assay and perturbation are directly definable."
        elif gene in {"Fn1", "Tnc", "Il6st", "Cd44", "Abca1"}:
            feasibility = "High: established antibody/protein or targeted expression assays are available."
        else:
            feasibility = "Moderate: qPCR/protein validation followed by population-resolved functional assay."
        contradiction: list[str] = []
        if not bool(getattr(row, "external_aging_supported", False)):
            contradiction.append("no gene-level external aging support in the current reference")
        if _safe_float(getattr(row, "treatment_FDR", np.nan)) >= 0.05:
            contradiction.append("treatment contrast is not FDR-supported")
        if population == "Immune":
            contradiction.append("population support is secondary to Granulosa/Stromal")
        rows.append(
            {
                "gene": gene,
                "cell_type": population,
                "subtype": getattr(row, "stage3_localized_subtypes", ""),
                "aging_LFC": getattr(row, "aging_LFC", np.nan),
                "aging_FDR": getattr(row, "aging_FDR", np.nan),
                "treatment_LFC": getattr(row, "treatment_LFC", np.nan),
                "treatment_FDR": getattr(row, "treatment_FDR", np.nan),
                "residual_LFC": getattr(row, "residual_LFC", np.nan),
                "exact_permutation_support": bool(pop.get("permutation_supported_signature", False)),
                "exact_permutation_detail": (
                    f"population signature; Spearman rank {pop.get('whole_all_spearman_observed_rank', '')}/20; "
                    f"cosine rank {pop.get('whole_all_cosine_similarity_observed_rank', '')}/20"
                ),
                "Hallmark_membership": ";".join(pathways),
                "leading_edge": bool(pathways),
                "external_aging_validation": bool(getattr(row, "external_aging_supported", False)),
                "PROGENy_or_TF_association": getattr(row, "stage7_candidate_TFs", ""),
                "NMF_program": ";".join(programs),
                "communication_role": communication_role,
                "known_biological_relevance": "project_internal_evidence_only; needs_manual_literature_review",
                "experimental_feasibility": feasibility,
                "primary_evidence": (
                    "Stage1.6 gene reversal + population exact permutation + Stage2 leading edge + Stage3 subtype localization"
                ),
                "supporting_evidence": "; ".join(
                    item
                    for item in [
                        "external aging" if bool(getattr(row, "external_aging_supported", False)) else "",
                        "regulatory target association" if bool(getattr(row, "stage7_supported_tf_target", False)) else "",
                        f"communication {communication_role}" if communication_role and communication_role.lower() != "nan" else "",
                        f"NMF: {','.join(programs)}" if programs else "",
                    ]
                    if item
                ),
                "contradictory_evidence": "; ".join(contradiction) if contradiction else "none identified in current matrices",
                "publication_tier": row.publication_tier,
                "tier_rationale": row.tier_rationale,
            }
        )
    return pd.DataFrame(rows)


def _claim_row(
    claim_id: str,
    category: str,
    cn: str,
    en: str,
    *,
    cell_type: str = "",
    subtype: str = "",
    pathway_or_gene: str = "",
    stage: str,
    files: str,
    statistical: str,
    permutation: str = "",
    external: str = "",
    negative: str = "",
    level: str,
    main_text: bool,
    main_figure: str = "",
    supplement: str = "",
    limitations: str,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "claim_category": category,
        "claim_text_cn": cn,
        "claim_text_en": en,
        "cell_type": cell_type,
        "subtype": subtype,
        "pathway_or_gene": pathway_or_gene,
        "supporting_stage": stage,
        "supporting_files": files,
        "statistical_support": statistical,
        "permutation_support": permutation,
        "external_support": external,
        "negative_evidence": negative,
        "evidence_level": level,
        "recommend_main_text": main_text,
        "recommend_main_figure": main_figure,
        "recommend_supplement": supplement,
        "limitations": limitations,
    }


def build_evidence_ledger(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    pop = tables["population"].set_index("population")
    pathways = tables["pathways"]
    localized = tables["localized"]
    decomposition = tables["decomposition"]
    geometry = tables["geometry"]
    concordance = tables["concordance"]
    external_paths = tables["external_paths"]
    regulators = tables["regulators"]
    programs = tables["programs"]
    ligands = tables["ligands"]
    rows: list[dict[str, Any]] = []
    for population, figure in [("Granulosa", "Figure 3"), ("Stromal_fibroblast", "Figure 3")]:
        row = pop.loc[population]
        rows.append(
            _claim_row(
                f"C{len(rows)+1:02d}",
                "MRJP1_reversal",
                f"{population}显示稳定的MRJP1相关全转录组反向变化。",
                f"{population} shows a stable MRJP1-associated whole-transcriptome reversal.",
                cell_type=population,
                stage="Stage 1.6",
                files="results/de_stage1_6/population_reversal_evidence.tsv",
                statistical=(
                    f"Spearman={row['whole_all_spearman_observed']:.3f}; cosine={row['whole_all_cosine_similarity_observed']:.3f}"
                ),
                permutation=(
                    f"Spearman rank {int(row['whole_all_spearman_observed_rank'])}/20; "
                    f"cosine rank {int(row['whole_all_cosine_similarity_observed_rank'])}/20"
                ),
                level="LEVEL B",
                main_text=True,
                main_figure=figure,
                supplement="S8",
                limitations="Association in n=3 libraries/group; transcriptomic reversal is not functional rejuvenation.",
            )
        )
    strong = pathways.loc[pathways["pathway_evidence_level"].eq("Strong_multi_metric_reversal")]
    for population in ["Stromal_fibroblast", "Granulosa"]:
        sub = strong.loc[strong["population"].eq(population)]
        rows.append(
            _claim_row(
                f"C{len(rows)+1:02d}",
                "MRJP1_reversal",
                f"{population}具有{sub.shape[0]}条strong multi-metric Hallmark反向证据。",
                f"{population} has {sub.shape[0]} strong multi-metric Hallmark reversal signals.",
                cell_type=population,
                pathway_or_gene=";".join(sub["pathway"].astype(str)),
                stage="Stage 2",
                files="results/pathway_stage2/primary_pathway_reversal_summary.tsv",
                statistical="Aging/treatment GSEA plus gene-level geometry metrics.",
                permutation="Pathway metrics evaluated over all 20 exact assignments.",
                level="LEVEL B",
                main_text=True,
                main_figure="Figure 4" if population.startswith("Stromal") else "Figure 5",
                supplement="S9",
                limitations="GSEA activity does not establish biochemical pathway activation or inhibition.",
            )
        )
    for population in ["Granulosa", "Stromal_fibroblast"]:
        n_links = int(
            localized.loc[
                localized["broad_population"].eq(population)
                & _as_bool(localized["localized_GSEA_support"])
            ].shape[0]
        )
        rows.append(
            _claim_row(
                f"C{len(rows)+1:02d}",
                "subtype_localization",
                f"{population}的broad信号可定位到{n_links}个受支持的pathway-subtype组合。",
                f"The broad {population} signal localizes to {n_links} supported pathway-subtype pairs.",
                cell_type=population,
                stage="Stage 3",
                files="results/stage3_subtype_localization/broad_to_subtype_localization.tsv",
                statistical="Subtype raw-count pseudobulk GSEA with predefined readiness rules.",
                level="LEVEL B",
                main_text=True,
                main_figure="Figure 4" if population.startswith("Stromal") else "Figure 5",
                supplement="S10",
                limitations="Localization is restricted to adequately represented stable subtypes.",
            )
        )
    for population in ["Granulosa", "Stromal_fibroblast", "Immune"]:
        sub = decomposition.loc[
            decomposition["broad_population"].eq(population)
            & decomposition["effect_source"].eq("cell_intrinsic")
        ]
        values = "; ".join(
            f"{r.contrast}={r.projection_fraction_of_total:.3f}" for r in sub.itertuples()
        )
        rows.append(
            _claim_row(
                f"C{len(rows)+1:02d}",
                "composition",
                f"{population}的总体变化主要由亚型内表达改变解释，而非亚型比例恢复。",
                f"Within-subtype expression changes dominate the aggregate {population} shift rather than subtype proportion restoration.",
                cell_type=population,
                stage="Stage 4",
                files="results/stage4_composition_decomposition/decomposition_summary.tsv",
                statistical=f"Linear CPM two-factor Shapley projection: {values}",
                level="LEVEL B",
                main_text=True,
                main_figure="Figure 5",
                supplement="S11",
                limitations="Projection fractions are model-space geometry, not causal biological percentages.",
            )
        )
    for population in ["Granulosa", "Stromal_fibroblast"]:
        sub = geometry.loc[geometry["population"].eq(population)]
        means = sub.groupby("group", observed=True)["aging_axis_projection"].mean()
        rows.append(
            _claim_row(
                f"C{len(rows)+1:02d}",
                "aging_geometry",
                f"{population}中OT沿衰老轴相对OC向年轻参考方向移动。",
                f"In {population}, OT projects toward the young reference relative to OC along the aging axis.",
                cell_type=population,
                stage="Stage 5",
                files="results/stage5_rejuvenation_geometry/aging_axis_projection.tsv;results/stage5_rejuvenation_geometry/permutation_geometry.tsv",
                statistical=f"Group means Y={means.get('Y', np.nan):.3f}, OC={means.get('OC', np.nan):.3f}, OT={means.get('OT', np.nan):.3f}",
                permutation="Observed young-directed rank 1/20; empirical p=0.05.",
                level="LEVEL B",
                main_text=True,
                main_figure="Figure 3",
                supplement="S12",
                limitations="Young-directed transcriptomic projection is not rejuvenation or biological-age reversal.",
            )
        )
    for population in ["Granulosa", "Stromal_fibroblast"]:
        row = concordance.loc[
            concordance["population"].eq(population)
            & concordance["scope"].eq("both_aging_FDR_lt_0.05")
        ].iloc[0]
        rows.append(
            _claim_row(
                f"C{len(rows)+1:02d}",
                "external_validation",
                f"{population}的内部与外部显著衰老效应呈正相关。",
                f"Significant internal and external aging effects are positively concordant in {population}.",
                cell_type=population,
                stage="Stage 6",
                files="results/stage6_external_validation/internal_external_concordance.tsv",
                statistical=f"n={int(row['n_genes'])}; Spearman={row['spearman']:.3f}; cosine={row['cosine']:.3f}",
                external=f"GSE232309; same-direction fraction={row['same_direction_fraction']:.3f}",
                level="LEVEL A",
                main_text=True,
                main_figure="Figure 4" if population.startswith("Stromal") else "Figure 5",
                supplement="S13",
                limitations="The external reference differs in age, platform, and preprocessing.",
            )
        )
    stromal_external = external_paths.loc[external_paths["population"].eq("Stromal_fibroblast")]
    n_external = int(
        (
            _as_bool(stromal_external["external_internal_aging_same_direction"])
            & _as_bool(stromal_external["treatment_opposes_external_aging"])
        ).sum()
    )
    rows.append(
        _claim_row(
            f"C{len(rows)+1:02d}",
            "external_validation",
            f"Stromal的12条强通路中有{n_external}条同时满足外部衰老方向一致和治疗方向相反。",
            f"{n_external} of 12 strong Stromal pathways are externally aging-concordant and treatment-opposed.",
            cell_type="Stromal_fibroblast",
            stage="Stage 6",
            files="results/stage6_external_validation/external_strong_pathway_validation.tsv",
            statistical=f"{n_external}/12 direction-concordant and treatment-opposed.",
            external="GSE232309 pathway-level validation.",
            level="LEVEL B",
            main_text=True,
            main_figure="Figure 4",
            supplement="S13",
            limitations="Direction agreement is not independent treatment replication.",
        )
    )
    granulosa_ext = external_paths.loc[external_paths["population"].eq("Granulosa")].iloc[0]
    rows.append(
        _claim_row(
            f"C{len(rows)+1:02d}",
            "external_validation",
            "Granulosa最强的Protein secretion通路未获得外部通路方向复现。",
            "The strongest Granulosa Protein secretion pathway lacks pathway-level directional replication externally.",
            cell_type="Granulosa",
            pathway_or_gene="HALLMARK_PROTEIN_SECRETION",
            stage="Stage 6",
            files="results/stage6_external_validation/external_strong_pathway_validation.tsv",
            statistical=(
                f"Internal aging NES={granulosa_ext['aging_NES']:.3f}; external aging NES={granulosa_ext['external_aging_NES']:.3f}"
            ),
            external="Direction discordant in GSE232309.",
            negative="External pathway direction is opposite to the internal aging direction.",
            level="LEVEL A",
            main_text=True,
            main_figure="",
            supplement="S13",
            limitations="This limits pathway-level generalization despite gene-level concordance.",
        )
    )
    selected_regulators = regulators.loc[
        regulators["analysis_level"].eq("broad")
        & regulators["program"].isin(["PI3K", "Arnt", "Hoxa5"])
        & regulators["population"].isin(["Granulosa", "Stromal_fibroblast"])
    ]
    rows.append(
        _claim_row(
            f"C{len(rows)+1:02d}",
            "regulatory",
            "PI3K、Arnt和Hoxa5构成优先的候选调控活动，但不是MRJP1直接靶点证据。",
            "PI3K, Arnt, and Hoxa5 are prioritized regulatory-activity hypotheses, not evidence of direct MRJP1 targets.",
            cell_type="Granulosa;Stromal_fibroblast",
            pathway_or_gene="PI3K;Arnt;Hoxa5",
            stage="Stage 7",
            files="results/stage7_regulatory_activity/candidate_regulators.tsv",
            statistical=f"{len(selected_regulators)} broad lineage-program rows; rank 1/20 where exact permutation is available.",
            permutation="Stromal PI3K/Arnt/Hoxa5 and Granulosa PI3K rank 1/20.",
            level="LEVEL C",
            main_text=True,
            main_figure="Figure 4;Figure 5",
            supplement="S14",
            limitations="Inferred activity is not direct binding, phosphorylation, or target engagement.",
        )
    )
    n_reversed = int(
        (_as_bool(programs["directionally_reversed"]) & _as_bool(programs["closer_to_young_after_treatment"])).sum()
    )
    rows.append(
        _claim_row(
            f"C{len(rows)+1:02d}",
            "gene_program",
            f"Granulosa与Stromal的10个NMF程序中有{n_reversed}个方向反向且更接近年轻组。",
            f"{n_reversed} of 10 Granulosa/Stromal NMF programs reverse direction and move closer to young.",
            cell_type="Granulosa;Stromal_fibroblast",
            stage="Stage 8",
            files="results/stage8_gene_programs/program_reversal.tsv",
            statistical=f"{n_reversed}/10 programs meet both direction criteria; K=5 per lineage.",
            level="LEVEL C",
            main_text=True,
            main_figure="Figure 4;Figure 5",
            supplement="S15",
            limitations="NMF programs are data-driven co-expression patterns, not causal modules.",
        )
    )
    rows.append(
        _claim_row(
            f"C{len(rows)+1:02d}",
            "communication",
            "IL6、FGF2、BDNF和BMP4是Stromal到Granulosa的定向通讯假说。",
            "IL6, FGF2, BDNF, and BMP4 are targeted Stromal-to-Granulosa communication hypotheses.",
            cell_type="Stromal_fibroblast->Granulosa",
            pathway_or_gene=";".join(ligands["ligand"].astype(str)),
            stage="Stage 9",
            files="results/stage9_communication/candidate_ligands.tsv;results/stage9_communication/communication_evidence.tsv",
            statistical=f"{ligands['ligand'].nunique()} ligands passed the targeted evidence gate.",
            level="LEVEL D",
            main_text=True,
            main_figure="Figure 6",
            supplement="S16",
            limitations="Expression and ligand-target databases do not prove secretion, receptor activation, or direction of signaling.",
        )
    )
    theca = pop.loc["Theca_steroidogenic"]
    rows.append(
        _claim_row(
            f"C{len(rows)+1:02d}",
            "negative_evidence",
            "Theca虽有较强数值反向，但缺少稳定的population-level exact permutation支持。",
            "Theca shows numerically strong reversal but lacks stable population-level exact-permutation support.",
            cell_type="Theca_steroidogenic",
            stage="Stage 1.6",
            files="results/de_stage1_6/population_reversal_evidence.tsv",
            statistical=f"Spearman={theca['whole_all_spearman_observed']:.3f}; cosine={theca['whole_all_cosine_similarity_observed']:.3f}",
            permutation=(
                f"Spearman rank {int(theca['whole_all_spearman_observed_rank'])}/20; "
                f"cosine rank {int(theca['whole_all_cosine_similarity_observed_rank'])}/20"
            ),
            negative="No exact-permutation-supported population signature.",
            level="LEVEL A",
            main_text=True,
            supplement="S17",
            limitations="Retain as exploratory and do not promote to the central mechanism.",
        )
    )
    return pd.DataFrame(rows)


def _normalized_figure_stem(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"(?:_v\d+|_final|_draft|_old|_copy|_\d{3,4}dpi)$", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem


def build_figure_inventory(root: Path, figure_root: Path, result_root: Path) -> pd.DataFrame:
    """Inventory pre-Stage-13 figures without deleting or rewriting any file."""

    candidates = []
    for base in [figure_root, result_root]:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in {".png", ".pdf", ".svg", ".tif", ".tiff"}
                and "publication" not in path.parts
            ):
                candidates.append(path)
    hashes: dict[str, list[Path]] = {}
    for path in candidates:
        digest = sha256_file(path)
        hashes.setdefault(digest, []).append(path)
    rows: list[dict[str, Any]] = []
    for path in sorted(candidates):
        relative = path.relative_to(root)
        lower = str(relative).lower()
        digest = sha256_file(path)
        exact_duplicate = len(hashes[digest]) > 1
        stage_match = re.search(r"stage\d+(?:_\d+)?", lower)
        source_stage = stage_match.group(0) if stage_match else relative.parts[1] if len(relative.parts) > 1 else "general"
        figure_type = "umap" if "umap" in lower else "heatmap" if "heatmap" in lower else "scatter" if "scatter" in lower else "bar_or_summary" if any(x in lower for x in ["summary", "count", "composition", "fraction"]) else "other"
        cell_type = ";".join(
            label
            for token, label in [
                ("granulosa", "Granulosa"),
                ("stromal", "Stromal_fibroblast"),
                ("immune", "Immune"),
                ("theca", "Theca_steroidogenic"),
            ]
            if token in lower
        )
        if exact_duplicate:
            quality = "exact_duplicate"
        elif any(token in lower for token in ["qc", "exploratory", "diagnostic", "sensitivity"]):
            quality = "diagnostic_or_supplementary"
        elif path.suffix.lower() == ".pdf":
            quality = "vector_existing_candidate"
        else:
            quality = "review_or_regenerate_for_publication"
        main_or_supp = (
            "supplement_candidate"
            if quality in {"diagnostic_or_supplementary", "exact_duplicate"}
            else "main_or_supp_review_candidate"
        )
        rows.append(
            {
                "figure_path": str(relative),
                "source_stage": source_stage,
                "figure_type": figure_type,
                "cell_type": cell_type,
                "contrast": "OC_vs_Y;OT_vs_OC" if any(x in lower for x in ["reversal", "aging", "treatment"]) else "",
                "pathway": "",
                "purpose": "existing analysis output; retained unchanged for audit",
                "quality_status": quality,
                "main_or_supp_candidate": main_or_supp,
                "duplicate_group": f"sha256:{digest[:12]}" if exact_duplicate else f"stem:{_normalized_figure_stem(path)}",
                "notes": "No original file was deleted or modified.",
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    return pd.DataFrame(rows)


def _figure_qa(svg_path: Path, pdf_path: Path, png_path: Path) -> dict[str, Any]:
    from PIL import Image, ImageStat

    ET.parse(svg_path)
    svg_text = svg_path.read_text(encoding="utf-8")
    with Image.open(png_path) as image:
        rgb = image.convert("RGB")
        stats = ImageStat.Stat(rgb.resize((128, 128)))
        width, height = image.size
        channel_std = float(np.mean(stats.stddev))
    pdf_ok = pdf_path.read_bytes()[:4] == b"%PDF"
    return {
        "png_width": width,
        "png_height": height,
        "png_channel_std": channel_std,
        "png_nonblank": channel_std > 1.0,
        "svg_valid_xml": True,
        "svg_text_editable": "<text" in svg_text,
        "pdf_signature_valid": pdf_ok,
        "single_axis_enforced": True,
        "qa_pass": channel_std > 1.0 and pdf_ok and "<text" in svg_text,
    }


def _save_plot(
    fig: mpl.figure.Figure,
    plot_id: str,
    source_data: pd.DataFrame,
    figure_root: Path,
    source_root: Path,
    *,
    title: str,
    source_files: str,
    conclusion: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(fig.axes) != 1:
        raise ValueError(f"{plot_id} has {len(fig.axes)} axes; Stage 13 permits one axis per file")
    source_path = source_root / f"{plot_id}.tsv"
    source_data.to_csv(source_path, sep="\t", index=False)
    paths = {suffix: figure_root / f"{plot_id}.{suffix}" for suffix in ["svg", "pdf", "png"]}
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight")
    plt.close(fig)
    qa = _figure_qa(paths["svg"], paths["pdf"], paths["png"])
    manifest = {
        "plot_id": plot_id,
        "title": title,
        "conclusion": conclusion,
        "source_files": source_files,
        "source_data": str(source_path),
        "svg": str(paths["svg"]),
        "pdf": str(paths["pdf"]),
        "png_600dpi": str(paths["png"]),
        "n_axes": 1,
        "assembled_multipanel": False,
        "svg_sha256": sha256_file(paths["svg"]),
        "pdf_sha256": sha256_file(paths["pdf"]),
        "png_sha256": sha256_file(paths["png"]),
    }
    return manifest, {"plot_id": plot_id, **qa}


def _new_figure(width: float = 3.5, height: float = 2.8) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    return fig, ax


def _plot_umap(
    frame: pd.DataFrame,
    label_column: str,
    colors: Mapping[str, str],
    plot_id: str,
    figure_root: Path,
    source_root: Path,
    *,
    title: str,
    source_file: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fig, ax = _new_figure(4.2, 3.5)
    labels = frame[label_column].fillna("Uncertain").astype(str)
    order = sorted(labels.unique(), key=lambda item: (item != "Uncertain", item))
    for label in order:
        sub = frame.loc[labels.eq(label)]
        ax.scatter(
            sub["UMAP1"],
            sub["UMAP2"],
            s=0.35,
            linewidths=0,
            alpha=0.75 if label != "Uncertain" else 0.25,
            c=colors.get(label, "#999999"),
            label=label,
            rasterized=True,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.legend(markerscale=6, frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
    return _save_plot(
        fig,
        plot_id,
        frame,
        figure_root,
        source_root,
        title=title,
        source_files=source_file,
        conclusion="Cell identities occupy structured regions of the existing Harmony-derived UMAP.",
    )


def _plot_composition(table: pd.DataFrame, figure_root: Path, source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    long = table.melt(id_vars="library_id", var_name="cell_type", value_name="fraction")
    long["group"] = long["library_id"].str.split("_").str[0]
    display = long.loc[long["cell_type"].isin(CELL_TYPE_COLORS)].copy()
    pivot = display.pivot(index="library_id", columns="cell_type", values="fraction").fillna(0)
    order = [f"{group}_{rep}" for group in ["Y", "OC", "OT"] for rep in [1, 2, 3]]
    pivot = pivot.reindex([item for item in order if item in pivot.index])
    fig, ax = _new_figure(5.3, 3.0)
    bottom = np.zeros(len(pivot))
    for cell_type in [key for key in CELL_TYPE_COLORS if key in pivot.columns]:
        values = pivot[cell_type].to_numpy()
        ax.bar(pivot.index, values, bottom=bottom, color=CELL_TYPE_COLORS[cell_type], width=0.78, label=cell_type)
        bottom += values
    ax.set_ylabel("Cell fraction")
    ax.set_xlabel("Library")
    ax.set_title("Broad cell-type composition by library", loc="left")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
    return _save_plot(
        fig,
        "atlas_library_composition",
        long,
        figure_root,
        source_root,
        title="Broad cell-type composition by library",
        source_files="results/composition/library_broad_fractions.tsv",
        conclusion="All libraries contribute multiple ovarian populations, with visible biological composition variation.",
    )


def _plot_aging_de_summary(table: pd.DataFrame, figure_root: Path, source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sub = table.loc[table["contrast"].eq("OC_vs_Y")].copy()
    sub = sub.sort_values("padj_lt_0.05_and_absLFC_gt_0.5")
    fig, ax = _new_figure(4.1, 3.0)
    colors = [CELL_TYPE_COLORS.get(item, "#777777") for item in sub["population"]]
    ax.barh(sub["population"].str.replace("_", " "), sub["padj_lt_0.05_and_absLFC_gt_0.5"], color=colors)
    ax.set_xlabel("Genes with FDR < 0.05 and |LFC| > 0.5")
    ax.set_ylabel("")
    ax.set_title("Cell-type-specific ovarian aging effects", loc="left")
    return _save_plot(
        fig,
        "aging_de_summary",
        sub,
        figure_root,
        source_root,
        title="Cell-type-specific ovarian aging effects",
        source_files="results/de_stage1/broad_de_summary.tsv",
        conclusion="Aging-associated differential expression varies substantially among ovarian populations.",
    )


def _plot_reversal_scatter(table: pd.DataFrame, population: str, figure_root: Path, source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sub = table.copy()
    sub["classification"] = np.where(
        _as_bool(sub["Level_2_DE_supported_candidate"]),
        "FDR-supported reversal",
        np.where(_as_bool(sub["Level_1_directional_candidate"]), "Directional candidate", "Other tested genes"),
    )
    draw_order = ["Other tested genes", "Directional candidate", "FDR-supported reversal"]
    palette = {"Other tested genes": "#D0D0D0", "Directional candidate": "#56B4E9", "FDR-supported reversal": "#D55E00"}
    fig, ax = _new_figure(3.5, 3.25)
    for label in draw_order:
        part = sub.loc[sub["classification"].eq(label)]
        ax.scatter(part["aging_effect"], part["treatment_effect"], s=3 if label != "Other tested genes" else 1.2, alpha=0.65 if label != "Other tested genes" else 0.25, color=palette[label], linewidths=0, label=label, rasterized=True)
    limit = float(np.nanmax(np.abs(sub[["aging_effect", "treatment_effect"]].to_numpy())))
    ax.plot([-limit, limit], [limit, -limit], color="#444444", linestyle="--", linewidth=0.7)
    ax.axhline(0, color="#BBBBBB", linewidth=0.5)
    ax.axvline(0, color="#BBBBBB", linewidth=0.5)
    ax.set_xlabel("Aging effect, OC - Y (log2 FC)")
    ax.set_ylabel("Treatment effect, OT - OC (log2 FC)")
    ax.set_title(population.replace("_", " "), loc="left")
    ax.legend(frameon=False, loc="best")
    plot_id = f"reversal_scatter_{population.lower()}"
    return _save_plot(
        fig,
        plot_id,
        sub,
        figure_root,
        source_root,
        title=f"Aging and treatment effects in {population}",
        source_files=f"results/de_stage1_6/{population}/observed_evidence_levels.tsv.gz",
        conclusion="Aging and MRJP1-associated effects are broadly opposed; highlighted genes meet predefined reversal evidence levels.",
    )


def _plot_population_permutation(table: pd.DataFrame, figure_root: Path, source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sub = table.sort_values("whole_all_spearman_observed").copy()
    colors = [CELL_TYPE_COLORS.get(item, "#777777") for item in sub["population"]]
    fig, ax = _new_figure(4.1, 3.0)
    ax.barh(sub["population"].str.replace("_", " "), sub["whole_all_spearman_observed"], color=colors)
    for y, (_, row) in enumerate(sub.iterrows()):
        ax.text(row["whole_all_spearman_observed"] - 0.015, y, f"rank {int(row['whole_all_spearman_observed_rank'])}/20", va="center", ha="right", color="white", fontsize=5.5)
    ax.axvline(0, color="#333333", linewidth=0.6)
    ax.set_xlabel("Spearman(aging effect, treatment effect)")
    ax.set_ylabel("")
    ax.set_title("Whole-transcriptome reversal across populations", loc="left")
    return _save_plot(
        fig,
        "population_exact_permutation_evidence",
        sub,
        figure_root,
        source_root,
        title="Whole-transcriptome reversal across populations",
        source_files="results/de_stage1_6/population_reversal_evidence.tsv",
        conclusion="Granulosa and Stromal_fibroblast show the strongest stable whole-transcriptome opposition.",
    )


def _plot_aging_axis(table: pd.DataFrame, population: str, figure_root: Path, source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sub = table.loc[table["population"].eq(population)].copy()
    order = ["Y", "OC", "OT"]
    fig, ax = _new_figure(2.5, 3.0)
    rng = np.random.default_rng(20260918)
    for x, group in enumerate(order):
        values = sub.loc[sub["group"].eq(group), "aging_axis_projection"].to_numpy()
        jitter = rng.uniform(-0.06, 0.06, len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, s=22, color=GROUP_COLORS[group], edgecolor="white", linewidth=0.4, zorder=3)
        ax.plot([x - 0.18, x + 0.18], [values.mean(), values.mean()], color="#222222", linewidth=1.2)
    ax.set_xticks(range(3), order)
    ax.set_ylabel("Cross-validated aging-axis projection")
    ax.set_xlabel("")
    ax.set_title(population.replace("_", " "), loc="left")
    return _save_plot(
        fig,
        f"aging_axis_{population.lower()}",
        sub,
        figure_root,
        source_root,
        title=f"Young-directed projection in {population}",
        source_files="results/stage5_rejuvenation_geometry/aging_axis_projection.tsv",
        conclusion="OT libraries project toward the young reference relative to OC in this population.",
    )


def _plot_pathway_reversal(table: pd.DataFrame, population: str, figure_root: Path, source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sub = table.loc[table["population"].eq(population)].copy()
    if population == "Stromal_fibroblast":
        sub = sub.loc[sub["pathway_evidence_level"].eq("Strong_multi_metric_reversal")]
    else:
        priority = ["Strong_multi_metric_reversal", "Permutation_supported_reversal", "GSEA_supported_reversal_candidate"]
        sub["level_order"] = sub["pathway_evidence_level"].map({name: i for i, name in enumerate(priority)}).fillna(9)
        sub = sub.loc[sub["level_order"].lt(9)].sort_values(["level_order", "aging_FDR"]).head(9)
    sub = sub.sort_values("aging_NES")
    y = np.arange(len(sub))
    fig, ax = _new_figure(4.7, max(2.6, 0.28 * len(sub) + 0.8))
    ax.hlines(y, sub["aging_NES"], sub["treatment_NES"], color="#BDBDBD", linewidth=0.8)
    ax.scatter(sub["aging_NES"], y, color=GROUP_COLORS["OC"], s=22, label="Aging NES")
    ax.scatter(sub["treatment_NES"], y, color=GROUP_COLORS["OT"], s=22, label="Treatment NES")
    ax.axvline(0, color="#444444", linewidth=0.6)
    ax.set_yticks(y, [_clean_pathway(value) for value in sub["pathway"]])
    ax.set_xlabel("Normalized enrichment score")
    ax.set_ylabel("")
    ax.set_title(f"{population.replace('_', ' ')} pathway reversal", loc="left")
    ax.legend(frameon=False, loc="best")
    return _save_plot(
        fig,
        f"pathway_reversal_{population.lower()}",
        sub.drop(columns=["level_order"], errors="ignore"),
        figure_root,
        source_root,
        title=f"{population} pathway reversal",
        source_files="results/pathway_stage2/primary_pathway_reversal_summary.tsv",
        conclusion="Selected aging-associated pathways shift in the opposite direction after treatment.",
    )


def _plot_subtype_localization(table: pd.DataFrame, population: str, figure_root: Path, source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sub = table.loc[
        table["broad_population"].eq(population) & _as_bool(table["localized_GSEA_support"])
    ].copy()
    sub["label"] = sub["subtype"].str.replace("_", " ") + " | " + sub["pathway"].map(_clean_pathway)
    sub = sub.sort_values("aging_NES")
    y = np.arange(len(sub))
    fig, ax = _new_figure(5.5, max(2.8, 0.22 * len(sub) + 0.7))
    ax.hlines(y, sub["aging_NES"], sub["treatment_NES"], color="#C0C0C0", linewidth=0.6)
    ax.scatter(sub["aging_NES"], y, color=GROUP_COLORS["OC"], s=15, label="Aging NES")
    ax.scatter(sub["treatment_NES"], y, color=GROUP_COLORS["OT"], s=15, label="Treatment NES")
    ax.axvline(0, color="#444444", linewidth=0.6)
    ax.set_yticks(y, sub["label"])
    ax.set_xlabel("Subtype normalized enrichment score")
    ax.set_ylabel("")
    ax.set_title(f"{population.replace('_', ' ')} subtype localization", loc="left")
    ax.legend(frameon=False, loc="best")
    return _save_plot(
        fig,
        f"subtype_localization_{population.lower()}",
        sub,
        figure_root,
        source_root,
        title=f"{population} subtype localization",
        source_files="results/stage3_subtype_localization/broad_to_subtype_localization.tsv",
        conclusion="Broad pathway reversal localizes to defined, adequately represented subtypes.",
    )


def _plot_external_validation(table: pd.DataFrame, figure_root: Path, source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sub = table.copy().sort_values("external_aging_NES")
    supported = _as_bool(sub["external_internal_aging_same_direction"]) & _as_bool(sub["treatment_opposes_external_aging"])
    colors = np.where(supported, "#009E73", "#BDBDBD")
    fig, ax = _new_figure(4.6, 4.2)
    y = np.arange(len(sub))
    ax.scatter(sub["external_aging_NES"], y, color=colors, s=24)
    ax.axvline(0, color="#444444", linewidth=0.6)
    ax.set_yticks(y, [_clean_pathway(value) for value in sub["pathway"]])
    ax.set_xlabel("External aging NES (GSE232309)")
    ax.set_ylabel("")
    ax.set_title("External validation of strong pathways", loc="left")
    ax.text(0.99, 0.02, f"{int(supported.sum())}/{len(sub)} direction-consistent\nand treatment-opposed", transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
    return _save_plot(
        fig,
        "external_pathway_validation",
        sub.assign(direction_consistent_and_treatment_opposed=supported),
        figure_root,
        source_root,
        title="External validation of strong pathways",
        source_files="results/stage6_external_validation/external_strong_pathway_validation.tsv",
        conclusion="External pathway support is substantially stronger for Stromal than for the single Granulosa strong pathway.",
    )


def _plot_regulators(table: pd.DataFrame, population: str, figure_root: Path, source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    programs = ["PI3K"] if population == "Granulosa" else ["PI3K", "Arnt", "Hoxa5"]
    sub = table.loc[
        table["analysis_level"].eq("broad")
        & table["population"].eq(population)
        & table["program"].isin(programs)
    ].copy()
    long = sub.melt(
        id_vars=["population", "activity_type", "program", "observed_rank_high_is_more_opposed", "empirical_p"],
        value_vars=["mean_Y", "mean_OC", "mean_OT"],
        var_name="group",
        value_name="activity",
    )
    long["group"] = long["group"].str.replace("mean_", "", regex=False)
    fig, ax = _new_figure(3.2, 3.0)
    for program, part in long.groupby("program", observed=True):
        part = part.set_index("group").reindex(["Y", "OC", "OT"]).reset_index()
        ax.plot(range(3), part["activity"], marker="o", markersize=3.5, label=program)
    ax.set_xticks(range(3), ["Y", "OC", "OT"])
    ax.set_ylabel("Inferred activity")
    ax.set_xlabel("")
    ax.set_title(f"{population.replace('_', ' ')} regulatory candidates", loc="left")
    ax.legend(frameon=False)
    return _save_plot(
        fig,
        f"regulatory_candidates_{population.lower()}",
        long,
        figure_root,
        source_root,
        title=f"{population} regulatory candidates",
        source_files="results/stage7_regulatory_activity/candidate_regulators.tsv",
        conclusion="Candidate activity patterns move toward young levels after treatment; these are regulatory hypotheses.",
    )


def _plot_nmf(table: pd.DataFrame, population: str, figure_root: Path, source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sub = table.loc[table["population"].eq(population)].copy()
    long = sub.melt(
        id_vars=["population", "program", "directionally_reversed", "closer_to_young_after_treatment"],
        value_vars=["mean_Y", "mean_OC", "mean_OT"],
        var_name="group",
        value_name="program_activity",
    )
    long["group"] = long["group"].str.replace("mean_", "", regex=False)
    fig, ax = _new_figure(3.5, 3.0)
    for program, part in long.groupby("program", observed=True):
        part = part.set_index("group").reindex(["Y", "OC", "OT"]).reset_index()
        ax.plot(range(3), part["program_activity"], marker="o", markersize=3, label=program.replace(population + "_", ""))
    ax.set_xticks(range(3), ["Y", "OC", "OT"])
    ax.set_ylabel("Mean NMF program activity")
    ax.set_xlabel("")
    ax.set_title(f"{population.replace('_', ' ')} NMF programs", loc="left")
    ax.legend(frameon=False, ncol=2)
    return _save_plot(
        fig,
        f"nmf_programs_{population.lower()}",
        long,
        figure_root,
        source_root,
        title=f"{population} NMF programs",
        source_files="results/stage8_gene_programs/program_reversal.tsv",
        conclusion="Most lineage-specific co-expression programs move in the opposite aging direction after treatment.",
    )


def _plot_decomposition(table: pd.DataFrame, figure_root: Path, source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sub = table.loc[table["effect_source"].isin(["composition", "cell_intrinsic"])].copy()
    sub["label"] = sub["broad_population"].str.replace("_", " ") + " | " + sub["contrast"]
    pivot = sub.pivot(index="label", columns="effect_source", values="projection_fraction_of_total").fillna(0)
    pivot = pivot.sort_index()
    y = np.arange(len(pivot))
    fig, ax = _new_figure(4.5, 3.2)
    ax.barh(y, pivot.get("cell_intrinsic", 0), color="#0072B2", label="Within-subtype expression")
    ax.barh(y, pivot.get("composition", 0), left=pivot.get("cell_intrinsic", 0), color="#E69F00", label="Composition")
    ax.axvline(1, color="#555555", linewidth=0.5, linestyle="--")
    ax.set_yticks(y, pivot.index)
    ax.set_xlabel("Projection fraction of aggregate effect")
    ax.set_ylabel("")
    ax.set_title("Composition versus within-subtype expression", loc="left")
    ax.legend(frameon=False)
    return _save_plot(
        fig,
        "composition_intrinsic_decomposition",
        sub,
        figure_root,
        source_root,
        title="Composition versus within-subtype expression",
        source_files="results/stage4_composition_decomposition/decomposition_summary.tsv",
        conclusion="Aggregate changes are dominated by within-subtype expression rather than subtype composition.",
    )


def _plot_communication(table: pd.DataFrame, figure_root: Path, source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sub = table.copy()
    fig, ax = _new_figure(3.4, 3.1)
    sizes = 18 + 3 * sub["n_supported_receiver_targets"].to_numpy()
    ax.scatter(sub["sender_aging_effect"], sub["sender_treatment_effect"], s=sizes, color="#56B4E9", edgecolor="#333333", linewidth=0.5)
    for row in sub.itertuples(index=False):
        ax.annotate(str(row.ligand).upper(), (row.sender_aging_effect, row.sender_treatment_effect), xytext=(3, 3), textcoords="offset points", fontsize=6)
    limit = float(np.nanmax(np.abs(sub[["sender_aging_effect", "sender_treatment_effect"]].to_numpy()))) * 1.15
    ax.plot([-limit, limit], [limit, -limit], color="#555555", linestyle="--", linewidth=0.6)
    ax.axhline(0, color="#BBBBBB", linewidth=0.5)
    ax.axvline(0, color="#BBBBBB", linewidth=0.5)
    ax.set_xlabel("Sender aging effect")
    ax.set_ylabel("Sender treatment effect")
    ax.set_title("Stromal-to-Granulosa ligand hypotheses", loc="left")
    return _save_plot(
        fig,
        "communication_ligand_hypotheses",
        sub,
        figure_root,
        source_root,
        title="Stromal-to-Granulosa ligand hypotheses",
        source_files="results/stage9_communication/candidate_ligands.tsv",
        conclusion="Four targeted ligands show sender-side expression reversal and receiver-side database support; they remain hypotheses.",
    )


def _plot_candidate_tiers(cards: pd.DataFrame, figure_root: Path, source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    counts = cards.groupby(["publication_tier", "cell_type"], observed=True).size().rename("n_candidates").reset_index()
    order = ["Tier A - immediate validation", "Tier B - second priority", "Tier C - exploratory"]
    pivot = counts.pivot(index="publication_tier", columns="cell_type", values="n_candidates").fillna(0).reindex(order)
    fig, ax = _new_figure(4.3, 2.8)
    bottom = np.zeros(len(pivot))
    for cell_type in pivot.columns:
        values = pivot[cell_type].to_numpy()
        ax.bar(pivot.index, values, bottom=bottom, color=CELL_TYPE_COLORS.get(cell_type, "#888888"), label=cell_type)
        bottom += values
    ax.set_ylabel("Candidates")
    ax.set_xlabel("")
    ax.set_title("Transparent candidate triage", loc="left")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
    return _save_plot(
        fig,
        "candidate_tier_summary",
        counts,
        figure_root,
        source_root,
        title="Transparent candidate triage",
        source_files="results/publication_stage13/CANDIDATE_EVIDENCE_CARDS.tsv",
        conclusion="The 60-candidate shortlist is reduced to explicit immediate, second-priority, and exploratory tiers.",
    )


def _result_fact_rows(tables: Mapping[str, pd.DataFrame], n_cells: int, n_features: int) -> pd.DataFrame:
    pop = tables["population"].set_index("population")
    pathways = tables["pathways"]
    localized = tables["localized"]
    decomp = tables["decomposition"]
    geometry = tables["geometry"]
    concordance = tables["concordance"]
    external = tables["external_paths"]
    programs = tables["programs"]
    ligands = tables["ligands"]
    facts: list[dict[str, Any]] = []

    def add(section: str, numeric: str, cn: str, en: str, forbidden: str, source: str, key: str) -> None:
        facts.append(
            {
                "fact_id": f"F{len(facts)+1:03d}",
                "section": section,
                "exact_numeric_result": numeric,
                "allowed_wording_cn": cn,
                "allowed_wording_en": en,
                "forbidden_overclaim": forbidden,
                "source_file": source,
                "source_row_or_key": key,
            }
        )

    add("Atlas", f"{n_cells} cells x {n_features} features", "最终对象包含105,763个细胞。", "The final object contains 105,763 cells.", "Do not call cells biological replicates.", "results/06_annotation_v2.h5ad", "shape")
    for population in ["Granulosa", "Stromal_fibroblast", "Theca_steroidogenic"]:
        row = pop.loc[population]
        add("Broad reversal", f"Spearman={row['whole_all_spearman_observed']:.6f}; cosine={row['whole_all_cosine_similarity_observed']:.6f}; ranks={int(row['whole_all_spearman_observed_rank'])}/20,{int(row['whole_all_cosine_similarity_observed_rank'])}/20", f"{population}的衰老与治疗效应呈反向关系。", f"Aging and treatment effects are opposed in {population}.", "Do not claim functional rejuvenation; Theca is exploratory.", "results/de_stage1_6/population_reversal_evidence.tsv", f"population={population}")
    strong = pathways.loc[pathways["pathway_evidence_level"].eq("Strong_multi_metric_reversal")]
    add("Pathway", str(len(strong)), "共有13条strong multi-metric通路反向证据。", "Thirteen strong multi-metric pathway reversals were retained.", "Do not call GSEA biochemical activation/inhibition.", "results/pathway_stage2/primary_pathway_reversal_summary.tsv", "pathway_evidence_level=Strong_multi_metric_reversal")
    n_localized = int(_as_bool(localized["localized_GSEA_support"]).sum())
    add("Subtype", str(n_localized), "38个pathway-subtype组合达到预设定位标准。", "Thirty-eight pathway-subtype pairs met the predefined localization rule.", "Do not generalize to rare or ineligible subtypes.", "results/stage3_subtype_localization/broad_to_subtype_localization.tsv", "localized_GSEA_support=True")
    for population in ["Granulosa", "Stromal_fibroblast", "Immune"]:
        for contrast in ["OC_vs_Y", "OT_vs_OC"]:
            row = decomp.loc[(decomp["broad_population"].eq(population)) & (decomp["contrast"].eq(contrast)) & (decomp["effect_source"].eq("cell_intrinsic"))].iloc[0]
            add("Composition", f"projection_fraction={row['projection_fraction_of_total']:.6f}", f"{population} {contrast}主要由亚型内表达改变构成。", f"The {population} {contrast} aggregate shift is dominated by within-subtype expression.", "Never interpret the projection as a causal percentage.", "results/stage4_composition_decomposition/decomposition_summary.tsv", f"{population}|{contrast}|cell_intrinsic")
    for population in ["Granulosa", "Stromal_fibroblast"]:
        means = geometry.loc[geometry["population"].eq(population)].groupby("group", observed=True)["aging_axis_projection"].mean()
        add("Geometry", f"Y={means['Y']:.6f}; OC={means['OC']:.6f}; OT={means['OT']:.6f}; rank=1/20", f"{population}中OT相对OC向年轻参考方向投影。", f"OT projects toward the young reference relative to OC in {population}.", "Do not say the ovary became younger.", "results/stage5_rejuvenation_geometry/aging_axis_projection.tsv", f"population={population}; group means")
    for population in ["Granulosa", "Stromal_fibroblast"]:
        row = concordance.loc[(concordance["population"].eq(population)) & concordance["scope"].eq("both_aging_FDR_lt_0.05")].iloc[0]
        add("External", f"n={int(row['n_genes'])}; Spearman={row['spearman']:.6f}; cosine={row['cosine']:.6f}", f"{population}显著衰老基因在外部数据中方向一致。", f"Significant aging genes show external concordance in {population}.", "Do not call this treatment replication.", "results/stage6_external_validation/internal_external_concordance.tsv", f"{population}|both_aging_FDR_lt_0.05")
    stromal = external.loc[external["population"].eq("Stromal_fibroblast")]
    supported = _as_bool(stromal["external_internal_aging_same_direction"]) & _as_bool(stromal["treatment_opposes_external_aging"])
    add("External", f"{int(supported.sum())}/12", "Stromal有9/12强通路获得外部方向支持。", "Nine of twelve strong Stromal pathways receive external directional support.", "Do not imply all 12 replicate.", "results/stage6_external_validation/external_strong_pathway_validation.tsv", "population=Stromal_fibroblast")
    reversed_programs = int((_as_bool(programs["directionally_reversed"]) & _as_bool(programs["closer_to_young_after_treatment"])).sum())
    add("NMF", f"{reversed_programs}/10; K=5 per lineage", "10个NMF程序中9个方向反向。", "Nine of ten NMF programs reverse direction.", "Do not treat co-expression programs as causal modules.", "results/stage8_gene_programs/program_reversal.tsv", "all programs")
    add("Communication", f"n_ligands={ligands['ligand'].nunique()}; {','.join(ligands['ligand'].astype(str))}", "四个配体仅作为通讯假说。", "Four ligands are retained as communication hypotheses only.", "Do not claim proven signaling.", "results/stage9_communication/candidate_ligands.tsv", "all rows")
    add("Phenotype", "matched_phenotype_available=false", "当前没有匹配表型，不进行相关分析。", "No matched phenotypes are available; no correlations were performed.", "Do not claim functional rescue.", "results/stage11_phenotype_framework/PHENOTYPE_INTEGRATION_REPORT_CN.md", "matched_phenotype_available=false")
    return pd.DataFrame(facts)


def _negative_ledger(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    pop = tables["population"].set_index("population")
    external = tables["external_paths"]
    granulosa = external.loc[external["population"].eq("Granulosa")].iloc[0]
    rows = [
        {
            "negative_id": "N01",
            "finding": "Theca lacks stable population-level exact-permutation support.",
            "evidence": f"Spearman rank {int(pop.loc['Theca_steroidogenic', 'whole_all_spearman_observed_rank'])}/20; cosine rank {int(pop.loc['Theca_steroidogenic', 'whole_all_cosine_similarity_observed_rank'])}/20.",
            "source_file": "results/de_stage1_6/population_reversal_evidence.tsv",
            "publication_handling": "Report as negative/exploratory; exclude from the core mechanism.",
        },
        {
            "negative_id": "N02",
            "finding": "Granulosa HALLMARK_PROTEIN_SECRETION lacks external pathway-level directional replication.",
            "evidence": f"Internal aging NES={granulosa['aging_NES']:.3f}; external aging NES={granulosa['external_aging_NES']:.3f}.",
            "source_file": "results/stage6_external_validation/external_strong_pathway_validation.tsv",
            "publication_handling": "State explicitly in Results and retain the plot in Supplementary S13.",
        },
        {
            "negative_id": "N03",
            "finding": "Secondary populations lack the multi-endpoint support seen in Granulosa and Stromal.",
            "evidence": "Several populations have one favorable exact endpoint, but not the joint stable pattern across broad, pathway, geometry, and external layers.",
            "source_file": "results/de_stage1_6/population_reversal_evidence.tsv",
            "publication_handling": "Keep in the seven-population comparison and avoid mechanism claims.",
        },
        {
            "negative_id": "N04",
            "finding": "Mahalanobis aging distance was not estimable robustly.",
            "evidence": "Only three young libraries were available to estimate covariance.",
            "source_file": "results/stage5_rejuvenation_geometry/AGING_GEOMETRY_REPORT_CN.md",
            "publication_handling": "Report the prespecified skip; use cross-validated one-dimensional aging-axis projections.",
        },
        {
            "negative_id": "N05",
            "finding": "No matched phenotype data are available.",
            "evidence": "matched_phenotype_available=false; no transcriptome-phenotype correlations were run.",
            "source_file": "results/stage11_phenotype_framework/PHENOTYPE_INTEGRATION_REPORT_CN.md",
            "publication_handling": "Frame phenotypes as future validation, not current evidence.",
        },
        {
            "negative_id": "N06",
            "finding": "TF, PROGENy, NMF, and ligand-target analyses do not prove mechanism.",
            "evidence": "All are activity, co-expression, or knowledge-base inferences without perturbation or protein-level target engagement.",
            "source_file": "results/stage7_regulatory_activity/REGULATORY_ACTIVITY_REPORT_CN.md;results/stage8_gene_programs/GENE_PROGRAM_REPORT_CN.md;results/stage9_communication/CELL_COMMUNICATION_REPORT_CN.md",
            "publication_handling": "Label as hypothesis-generating and assign LEVEL C/D.",
        },
        {
            "negative_id": "N07",
            "finding": "Three of eight Primary_DE_ready subtypes did not enter subtype exact permutation.",
            "evidence": "Stage 3 reports eight Primary_DE_ready subtypes but only five with eligible pathway-level exact-permutation models.",
            "source_file": "results/stage3_subtype_localization/SUBTYPE_STAGE3_REPORT_CN.md",
            "publication_handling": "Do not imply equal inferential depth for every eligible subtype.",
        },
    ]
    return pd.DataFrame(rows)


def _main_figure_selection(publication_plots: pd.DataFrame) -> pd.DataFrame:
    rows = [
        (1, "A", "What was the experimental design?", "planned schematic only", "Essential context; no composite was rendered in Stage 13.", True, "Create as a separate schematic after manuscript design is fixed."),
        (1, "B", "What cells make up the atlas?", "figures/publication/atlas_umap_broad_cell_types.svg", "Existing UMAP, fixed colors, broad identities.", False, ""),
        (1, "C", "How do broad compositions vary by library?", "figures/publication/atlas_library_composition.svg", "Shows biological replicate structure without inferential overclaim.", False, ""),
        (1, "D", "Which follicular states were resolved?", "figures/publication/atlas_umap_granulosa_subtypes.svg", "State-level atlas needed for later localization.", False, ""),
        (1, "E", "Which stromal states were resolved?", "figures/publication/atlas_umap_stromal_subtypes.svg", "State-level atlas needed for the stromal mechanism line.", False, ""),
        (2, "A", "Which populations remodel with aging?", "figures/publication/aging_de_summary.svg", "Library-level pseudobulk aging summary.", False, ""),
        (2, "B", "Which aging pathways characterize Stromal?", "figures/publication/pathway_reversal_stromal_fibroblast.svg", "Contains the strongest program-level aging and treatment opposition.", False, "Use aging points in Figure 2 and reuse the same standalone source in Figure 4 planning."),
        (2, "C", "Which aging pathways characterize Granulosa?", "figures/publication/pathway_reversal_granulosa.svg", "Provides the granulosa aging context before response interpretation.", False, ""),
        (3, "A", "Are Granulosa aging and treatment effects opposed?", "figures/publication/reversal_scatter_granulosa.svg", "Direct gene-level effect comparison.", False, ""),
        (3, "B", "Are Stromal aging and treatment effects opposed?", "figures/publication/reversal_scatter_stromal_fibroblast.svg", "Direct gene-level effect comparison.", False, ""),
        (3, "C", "How specific is support across seven populations?", "figures/publication/population_exact_permutation_evidence.svg", "Displays negative populations alongside core populations.", False, ""),
        (3, "D", "Does Granulosa shift toward young reference?", "figures/publication/aging_axis_granulosa.svg", "Cross-validated sample-level projection.", False, ""),
        (3, "E", "Does Stromal shift toward young reference?", "figures/publication/aging_axis_stromal_fibroblast.svg", "Cross-validated sample-level projection.", False, ""),
        (4, "A", "Where does stromal reversal localize?", "figures/publication/subtype_localization_stromal_fibroblast.svg", "Links broad evidence to supported stromal subtypes.", False, ""),
        (4, "B", "Which stromal pathways reverse most strongly?", "figures/publication/pathway_reversal_stromal_fibroblast.svg", "Twelve strong multi-metric pathways.", False, ""),
        (4, "C", "Does an external aging dataset support the pathway direction?", "figures/publication/external_pathway_validation.svg", "Independent aging context and visible discordance.", False, ""),
        (4, "D", "Which stromal regulatory activities converge?", "figures/publication/regulatory_candidates_stromal_fibroblast.svg", "PI3K, Arnt, and Hoxa5 hypotheses.", False, ""),
        (4, "E", "Do data-driven stromal programs reverse?", "figures/publication/nmf_programs_stromal_fibroblast.svg", "Independent co-expression program support.", False, ""),
        (5, "A", "Where does Granulosa reversal localize?", "figures/publication/subtype_localization_granulosa.svg", "Antral/preantral state localization.", False, ""),
        (5, "B", "Which Granulosa pathways are supported?", "figures/publication/pathway_reversal_granulosa.svg", "Separates the one strongest pathway from broader candidates.", False, ""),
        (5, "C", "Is the aggregate shift compositional?", "figures/publication/composition_intrinsic_decomposition.svg", "Shows the within-subtype dominance with its model boundary.", False, ""),
        (5, "D", "Does PI3K activity show a young-directed shift?", "figures/publication/regulatory_candidates_granulosa.svg", "Granulosa regulatory hypothesis.", False, ""),
        (5, "E", "Do data-driven Granulosa programs reverse?", "figures/publication/nmf_programs_granulosa.svg", "Independent co-expression program support.", False, ""),
        (6, "A", "Which stromal ligands are plausible?", "figures/publication/communication_ligand_hypotheses.svg", "Targeted four-ligand hypothesis display.", False, "Title must retain hypothesis language."),
        (6, "B", "Which receptors and targets support each ligand?", "results/stage9_communication/communication_evidence.tsv", "Evidence table retained; a network cartoon could overstate causality.", True, "Only render after manual choice of edges and explicit hypothesis labels."),
        (7, "A", "What evidence is supported versus hypothetical?", "results/publication_stage13/MASTER_EVIDENCE_LEDGER.tsv", "Evidence-level integrated model planned from the ledger.", True, "Render only after experimental plan and manuscript claims are approved."),
        (7, "B", "Which candidates should be validated first?", "figures/publication/candidate_tier_summary.svg", "Transparent 60-candidate compression.", False, ""),
        (7, "C", "Which functional assays close the evidence gap?", "results/publication_stage13/EXPERIMENTAL_VALIDATION_PRIORITY_CN.md", "No phenotype data exist, so this remains a validation design.", True, "Replace with experimental results when available."),
    ]
    table = pd.DataFrame(rows, columns=["figure_number", "panel", "scientific_question", "source_file", "why_selected", "needs_regeneration", "regeneration_reason"])
    table["rendered_as_standalone"] = table["source_file"].str.startswith("figures/publication/")
    table["assembled_figure_created"] = False
    return table


def _supplement_plan() -> pd.DataFrame:
    categories = [
        ("S1", "QC", "Stage 1 QC and threshold sensitivity"),
        ("S2", "Harmony/batch assessment", "Integrated and unintegrated neighborhood diagnostics"),
        ("S3", "Annotation markers", "Broad marker validation and annotation audit"),
        ("S4", "Follicular/steroidogenic annotation", "Local cluster review and marker evidence"),
        ("S5", "Other subclustering", "Immune, endothelial, epithelial, and stromal reviews"),
        ("S6", "Pseudobulk QC", "Coverage and sample-level PCA"),
        ("S7", "DE numerical audit", "Unified model concordance and warnings"),
        ("S8", "Permutation framework", "All 20 assignments and seven populations"),
        ("S9", "Hallmark full results", "All pathway evidence and leading edges"),
        ("S10", "Subtype readiness", "Eligibility, sensitivity, and localization"),
        ("S11", "Composition decomposition", "Linear CPM Shapley diagnostics"),
        ("S12", "Aging geometry", "Leave-one-out and leave-pair-out sensitivity"),
        ("S13", "External validation", "Gene and pathway concordance including discordance"),
        ("S14", "Regulatory activity", "Full PROGENy and TF activity"),
        ("S15", "NMF", "K/seed stability and gene programs"),
        ("S16", "Communication", "Ligand, receptor, and target evidence"),
        ("S17", "Negative and sensitivity results", "Theca and other unsupported hypotheses"),
    ]
    return pd.DataFrame(categories, columns=["supplement_number", "category", "content_scope"])


def _write_reports(
    output_root: Path,
    ledger: pd.DataFrame,
    cards: pd.DataFrame,
    facts: pd.DataFrame,
    negatives: pd.DataFrame,
    figures: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
) -> None:
    tier_counts = cards["publication_tier"].value_counts()
    tier_a = cards.loc[cards["publication_tier"].eq("Tier A - immediate validation"), "gene"].tolist()
    strong = tables["pathways"].loc[tables["pathways"]["pathway_evidence_level"].eq("Strong_multi_metric_reversal")]
    stromal_strong = strong.loc[strong["population"].eq("Stromal_fibroblast")]
    granulosa_strong = strong.loc[strong["population"].eq("Granulosa")]
    report = f"""# Stage 13 论文级结果收敛报告

## 1 当前项目状态

Stage 1-12已完成。本阶段没有重跑Harmony、UMAP、DE、GSEA、注释或改变任何统计阈值，只整理已有证据并生成独立单图。所有正式比较仍以library为生物学重复（每组n=3）。

## 2 最强结论

Granulosa与Stromal_fibroblast是最稳定的MRJP1-associated transcriptomic reversal populations。Stromal证据链更完整：broad exact permutation、{len(stromal_strong)}条strong Hallmark、subtype定位、外部衰老一致性、regulatory activity与NMF相互支持。

## 3 证据等级

- LEVEL A：当前数据直接支持的观察结果。
- LEVEL B：有统计支持的转录组推断。
- LEVEL C：多层结果支持的机制假说。
- LEVEL D：必须经过功能实验验证的具体机制。

详见`MASTER_EVIDENCE_LEDGER.tsv`。

## 4 Figure 1-7设计

Figure 1-7的科学问题与panel选择已经完成，但遵照用户要求，没有输出任何拼接组图。当前生成{len(figures)}个独立、单轴、可追溯图，每个图均有SVG、PDF和600 dpi PNG。

## 5 Supplementary设计

补充材料按QC、批次、注释、pseudobulk、置换、Hallmark、亚型、分解、几何、外部验证、调控、NMF、通讯和负结果组织为S1-S17。原有223张图不删除，统一登记在`FIGURE_INVENTORY.tsv`。

## 6 Stromal主线

Stromal的aging-treatment全转录组Spearman为-0.763，真实标签rank 1/20；12条strong Hallmark覆盖炎症、TNFA/NFkB、OXPHOS、protein secretion、apoptosis和DNA repair等程序。外部参考中9/12强通路方向一致且治疗方向相反。PI3K、Arnt和Hoxa5及NMF程序提供机制假说，但不是直接机制证据。

## 7 Granulosa主线

Granulosa的aging-treatment全转录组Spearman为-0.740，真实标签rank 1/20；{len(granulosa_strong)}条strong Hallmark为Protein secretion，并定位到antral/preantral-like状态。显著衰老基因有外部一致性，但最强Protein secretion通路在外部数据中方向不一致，必须保留这一限制。

## 8 Composition vs intrinsic

Granulosa、Stromal和Immune的总体变化主要由within-subtype expression change构成。这里使用linear CPM Shapley双因素分解；projection fraction是模型空间的几何投影，不是因果贡献百分比。

## 9 External validation

内部与外部显著衰老LFC的Spearman约为Granulosa 0.615、Stromal 0.645。外部数据提升了Stromal主线的可信度，但不是MRJP1处理的外部复现。

## 10 Regulatory/NMF

Stromal PI3K、Arnt、Hoxa5和Granulosa PI3K为优先activity hypotheses。两条lineage均选择K=5，10个程序中9个方向反向并更接近年轻组。activity和co-expression均不能证明直接靶点或因果通路。

## 11 Communication hypotheses

IL6、FGF2、BDNF、BMP4只作为Stromal->Granulosa定向通讯假说；现有结果不能证明分泌、受体激活或跨细胞因果链。

## 12 Candidate shortlist

60条候选被透明压缩为Tier A={int(tier_counts.get('Tier A - immediate validation', 0))}、Tier B={int(tier_counts.get('Tier B - second priority', 0))}、Tier C={int(tier_counts.get('Tier C - exploratory', 0))}。Tier A为：{', '.join(tier_a)}。分层使用明确证据门槛和人工可解释锚点，没有加权黑箱分数。

## 13 Experimental priorities

Tier 1优先验证Stromal炎症/TNFA-NFkB/OXPHOS/protein secretion及Granulosa protein secretion和cell-state response；Tier 2验证PI3K、Arnt、Hoxa5以及四个通讯候选；Tier 3补充AMH/FSH/E2、卵泡计数、ROS/MDA/SOD/GSH-Px/ATP/MMP、p16/p21/gamma-H2AX、Ki67和TUNEL。

## 14 Negative results

Theca缺少稳定population-level exact permutation支持；Granulosa最强Hallmark缺少外部通路方向复现；次级population没有与Granulosa/Stromal相当的多终点证据；当前没有matched phenotype。详见`NEGATIVE_RESULTS_LEDGER.tsv`。

## 15 Statistical limitations

每组只有3个library；20种exact assignments使经验p值最小为0.05；external atlas存在年龄、平台和预处理差异；cell-level数量不等于生物学重复数。

## 16 推荐论文叙事

自然卵巢衰老伴随cell-type-specific transcriptional remodeling。MRJP1处理与部分衰老相关程序的反向移动相关，主要集中在Stromal fibroblast和Granulosa，并主要反映亚型内表达变化。外部衰老数据更强地支持Stromal主线；调控、NMF和通讯结果用于提出后续机制假说。

## 17 仍缺少哪些实验

缺少同一动物匹配的卵巢储备、激素、线粒体/氧化应激、炎症、衰老、凋亡和生育力数据，也缺少候选调控因子和配体-受体的扰动实验。

## 18 下一步建议

先人工审核独立图和Tier A，再完成最小正交验证。未经功能数据支持，不使用“rejuvenates”“restores youth”“reverses ovarian aging”或确定性通路/通讯机制表述。
"""
    (output_root / "PUBLICATION_STAGE13_REPORT_CN.md").write_text(report, encoding="utf-8")

    explained = """# Stage 13结果给用户的简明解释

## 1 为什么现在不应该继续大量增加分析？

项目已有从细胞类型、基因、通路、亚型、组成、样本几何、外部数据、调控、NMF到通讯的完整证据链。继续增加模型会增加多重比较和选择性叙事风险，却不增加生物学重复数。当前真正缺少的是独立实验验证。

## 2 为什么Stromal是目前最强主线？

Stromal不仅在全转录组置换中rank 1/20，还拥有12条strong Hallmark、明确亚型定位、外部衰老方向支持、调控候选和NMF程序。因此它的证据来自多个相互独立的层面。

## 3 Granulosa与Stromal有什么不同？

Granulosa也有稳定的全转录组反向变化和亚型定位，但只有1条strong Hallmark；而且这条Protein secretion通路在外部数据中方向不一致。Granulosa可以进入主线，但证据深度和外部复现弱于Stromal。

## 4 为什么composition decomposition重要？

它回答“总体变化是因为某类亚型变多/变少，还是同一种亚型内部真的改变表达”。结果显示主要是within-subtype expression change，因此不能把MRJP1的信号简单解释为恢复了细胞比例。这里的分解是linear CPM模型中的几何描述，不是因果百分比。

## 5 为什么external validation提升了Stromal可信度？

独立衰老数据中，Stromal显著衰老基因的方向一致性较高，并且12条强通路中9条同时表现为外部衰老方向一致和本项目治疗方向相反。这说明Stromal衰老背景不是只出现在本批数据中，但仍不等于外部重复了MRJP1治疗。

## 6 NMF、TF和communication分别增加了什么？

NMF说明不依赖预设通路的共表达程序也出现反向移动；TF/PROGENy给出可能的调控活动；communication把Stromal表达变化与Granulosa受体/靶基因连接成可测试假说。

## 7 为什么仍不能证明机制？

这些分析都来自RNA表达或知识库推断，没有直接测蛋白活性、配体分泌、受体激活、TF结合，也没有做阻断或敲降。因此只能说“候选”“一致于”或“假说”。

## 8 论文最终怎样讲？

最稳妥的主线是：自然卵巢衰老发生细胞类型特异的转录重塑；MRJP1处理与Stromal和Granulosa中部分衰老程序的反向移动相关；该响应主要来自亚型内表达变化；Stromal拥有最完整的外部和多层支持。

## 9 现在最需要补什么实验？

第一优先是匹配动物的卵泡计数、AMH/FSH/E2、炎症、氧化应激/线粒体、衰老/DNA损伤、增殖和凋亡指标。随后对PI3K、Arnt、Hoxa5及IL6/FGF2/BDNF/BMP4进行细胞类型定位和干预验证。
"""
    (output_root / "STAGE13_EXPLAINED_FOR_USER_CN.md").write_text(explained, encoding="utf-8")

    outline = """# 论文Results结构与证据索引

## Result 1 A single-cell atlas reveals cell-type-specific remodeling during ovarian aging.

主结论：9个library构成可重复的卵巢单细胞图谱，衰老效应具有cell-type specificity。\
图：Figure 1独立atlas图；Figure 2 aging summary。\
数据：`results/06_annotation_v2.h5ad`、`results/composition/library_broad_fractions.tsv`、`results/de_stage1/broad_de_summary.tsv`。\
统计：library-level pseudobulk；n=3/group。\
限制：cell数量不是生物学重复数。

## Result 2 MRJP1 treatment is associated with cell-type-specific reversal of aging-related transcriptional changes.

主结论：Granulosa和Stromal的aging-treatment effect最稳定地反向。\
图：Figure 3 reversal scatters、permutation comparison和aging-axis plots。\
数据：`results/de_stage1_6/*/observed_evidence_levels.tsv.gz`、`population_reversal_evidence.tsv`、`stage5_rejuvenation_geometry/aging_axis_projection.tsv`。\
统计：Spearman、cosine、20 exact assignments。\
限制：只称transcriptomic reversal/young-directed projection。

## Result 3 Stromal fibroblasts show the strongest and most reproducible program-level reversal.

主结论：Stromal有12条strong pathways、广泛亚型定位和9/12外部方向支持。\
图：Figure 4 standalone pathway、subtype、external、regulatory和NMF图。\
数据：`hallmark_pathway_evidence.tsv`、`broad_to_subtype_localization.tsv`、`external_strong_pathway_validation.tsv`、`candidate_regulators.tsv`、`program_reversal.tsv`。\
统计：GSEA FDR、exact permutation ranks、Spearman/cosine。\
限制：pathway activity不等于生化功能。

## Result 4 Granulosa responses are localized to defined cellular states and are primarily cell-intrinsic.

主结论：Granulosa Protein secretion及相关程序定位于antral/preantral-like状态。\
图：Figure 5 standalone subtype、pathway、decomposition、regulatory和NMF图。\
数据：Stage 2-4、7-8 TSV。\
统计：subtype pseudobulk GSEA与linear CPM Shapley decomposition。\
限制：最强Granulosa通路没有外部通路方向复现。

## Result 5 MRJP1-associated responses are driven predominantly by within-subtype transcriptional changes rather than compositional restoration.

主结论：Granulosa、Stromal和Immune均以within-subtype effect为主。\
图：`composition_intrinsic_decomposition.svg`。\
数据：`results/stage4_composition_decomposition/decomposition_summary.tsv`。\
统计：two-factor exact Shapley decomposition in linear CPM space。\
限制：projection fraction不是causal percentage。

## Result 6 Independent aging references support a substantial portion of the stromal aging/reversal signature.

主结论：Stromal显著衰老基因Spearman约0.645，9/12强通路获得方向支持。\
图：`external_pathway_validation.svg`。\
数据：Stage 6 concordance/pathway validation TSV。\
统计：Spearman、cosine、direction concordance。\
限制：外部参考不是MRJP1治疗重复。

## Result 7 Regulatory and ligand-target analyses nominate candidate mechanisms linking stromal and granulosa responses.

主结论：PI3K/Arnt/Hoxa5及IL6/FGF2/BDNF/BMP4形成可测试假说。\
图：Figure 4/5 regulatory standalone图与Figure 6 ligand hypothesis图。\
数据：Stage 7-9 TSV。\
统计：activity permutation、表达门槛、ligand-target支持。\
限制：均为hypothesis-generating，不证明直接机制。
"""
    (output_root / "RESULTS_OUTLINE_CN.md").write_text(outline, encoding="utf-8")

    validation = """# 实验验证优先级

## Tier 1：必须优先，验证文章主线

1. Stromal炎症/TNFA-NFkB：IL-6、TNF-alpha、NF-kB磷酸化，并进行Stromal定位。
2. Stromal OXPHOS/氧化应激：ROS、MDA、SOD、GSH-Px、ATP、MMP。
3. Stromal与Granulosa protein secretion：分泌蛋白/ER stress蛋白及细胞类型定位。
4. Granulosa cell-state response：AMH、FSH、E2、卵泡计数、Ki67及与antral/preantral marker共染。

## Tier 2：机制深化

1. PI3K活性：磷酸化蛋白读出，并在Granulosa与Stromal分别定位。
2. Arnt与Hoxa5：蛋白/核定位、靶基因读出，随后做细胞类型特异扰动。
3. IL6、FGF2、BDNF、BMP4：先确认Stromal分泌和Granulosa受体表达，再进行阻断/补充实验。

## Tier 3：功能闭环

1. 卵巢功能：AMH、FSH、E2、各级卵泡计数、动情周期和生育力。
2. 线粒体与氧化应激：ROS、MDA、SOD、GSH-Px、ATP、MMP。
3. 衰老/DNA损伤：p16、p21、gamma-H2AX、SA-beta-gal。
4. 增殖/凋亡：Ki67、EdU/BrdU、TUNEL、cleaved caspase。

## 设计底线

尽可能使用与转录组同批或可追溯匹配的动物，预先定义primary endpoints，盲法计数卵泡，并保留动物ID。当前`matched_phenotype_available=false`，所以不能进行或声称transcriptome-phenotype correlation。
"""
    (output_root / "EXPERIMENTAL_VALIDATION_PRIORITY_CN.md").write_text(validation, encoding="utf-8")

    narrative = """# 推荐论文主线

## A 最保守版本

在当前9个library中，自然衰老与卵巢多种细胞群的转录变化相关。MRJP1处理组在Granulosa和Stromal fibroblast中显示部分与衰老方向相反的表达和通路变化。结果主要来自亚型内表达变化；机制和功能意义仍需实验验证。

## B 推荐论文版本

Natural ovarian aging is accompanied by cell-type-specific transcriptional remodeling of granulosa states and the stromal niche. MRJP1 treatment is associated with a partial, population-specific reversal of aging-related programs, with the most coherent evidence in stromal fibroblasts and granulosa cells. Linear-CPM decomposition indicates that these responses predominantly reflect within-subtype transcriptional changes rather than restoration of subtype proportions. Independent aging data preferentially support the stromal program, while regulatory, NMF, and targeted communication analyses nominate experimentally testable hypotheses.

## C 明确禁止的过度表述

- MRJP1 rejuvenates the ovary.
- MRJP1 restores ovarian youth.
- MRJP1 reverses ovarian aging.
- MRJP1 activates/inhibits pathway X when evidence is GSEA only.
- MRJP1 signals through IL6/FGF2/BDNF/BMP4.
- PI3K/Arnt/Hoxa5 is a direct target of MRJP1.

优先使用：associated with、consistent with、transcriptional reversal、young-directed shift、candidate regulatory program、communication hypothesis、requires validation。
"""
    (output_root / "PUBLICATION_NARRATIVE_CN.md").write_text(narrative, encoding="utf-8")


def run_stage13(config: Mapping[str, Any]) -> None:
    paths = project_paths(dict(config))
    logger = setup_logging("22_stage13_publication", dict(config))
    result_root = paths["results"]
    output_root = result_root / "publication_stage13"
    figure_root = paths["figures"] / "publication"
    source_root = output_root / "source_data"
    for directory in [output_root, figure_root, source_root]:
        directory.mkdir(parents=True, exist_ok=True)
    configure_publication_style()

    tables = {
        "population": _read_tsv(result_root / "de_stage1_6" / "population_reversal_evidence.tsv"),
        "pathways": _read_tsv(result_root / "pathway_stage2" / "primary_pathway_reversal_summary.tsv"),
        "localized": _read_tsv(result_root / "stage3_subtype_localization" / "broad_to_subtype_localization.tsv"),
        "decomposition": _read_tsv(result_root / "stage4_composition_decomposition" / "decomposition_summary.tsv"),
        "geometry": _read_tsv(result_root / "stage5_rejuvenation_geometry" / "aging_axis_projection.tsv"),
        "concordance": _read_tsv(result_root / "stage6_external_validation" / "internal_external_concordance.tsv"),
        "external_paths": _read_tsv(result_root / "stage6_external_validation" / "external_strong_pathway_validation.tsv"),
        "regulators": _read_tsv(result_root / "stage7_regulatory_activity" / "candidate_regulators.tsv"),
        "programs": _read_tsv(result_root / "stage8_gene_programs" / "program_reversal.tsv"),
        "ligands": _read_tsv(result_root / "stage9_communication" / "candidate_ligands.tsv"),
    }
    required_reports = [
        result_root / "FINAL_ADVANCED_ANALYSIS_HANDOFF_CN.md",
        result_root / "WHAT_HAVE_WE_LEARNED_CN.md",
        result_root / "PAPER_FIGURE_PLAN_CN.md",
        result_root / "pathway_stage2" / "PATHWAY_STAGE2_HALLMARK_REPORT.md",
    ] + [
        result_root / folder / report
        for folder, report in [
            ("stage3_subtype_localization", "SUBTYPE_STAGE3_REPORT_CN.md"),
            ("stage4_composition_decomposition", "COMPOSITION_INTRINSIC_REPORT_CN.md"),
            ("stage5_rejuvenation_geometry", "AGING_GEOMETRY_REPORT_CN.md"),
            ("stage6_external_validation", "EXTERNAL_AGING_VALIDATION_REPORT_CN.md"),
            ("stage7_regulatory_activity", "REGULATORY_ACTIVITY_REPORT_CN.md"),
            ("stage8_gene_programs", "GENE_PROGRAM_REPORT_CN.md"),
            ("stage9_communication", "CELL_COMMUNICATION_REPORT_CN.md"),
            ("stage10_candidates", "MECHANISM_CANDIDATE_REPORT_CN.md"),
            ("stage11_phenotype_framework", "PHENOTYPE_INTEGRATION_REPORT_CN.md"),
        ]
    ]
    missing_reports = [str(path) for path in required_reports if not path.exists()]
    if missing_reports:
        raise FileNotFoundError("Missing required Stage 1-12 reports: " + "; ".join(missing_reports))

    source_schema_rows = []
    for key, table in tables.items():
        source_schema_rows.append({"source_key": key, "n_rows": len(table), "columns": ";".join(table.columns)})
    pd.DataFrame(source_schema_rows).to_csv(output_root / "SOURCE_SCHEMA_INVENTORY.tsv", sep="\t", index=False)

    inventory = build_figure_inventory(paths["root"], paths["figures"], result_root)
    inventory.to_csv(output_root / "FIGURE_INVENTORY.tsv", sep="\t", index=False)
    ledger = build_evidence_ledger(tables)
    ledger.to_csv(output_root / "MASTER_EVIDENCE_LEDGER.tsv", sep="\t", index=False)

    shortlist = _read_tsv(result_root / "stage10_candidates" / "experimental_validation_shortlist.tsv")
    cards = build_candidate_cards(
        shortlist,
        tables["population"],
        result_root / "pathway_stage2" / "hallmark_leading_edge_review.tsv",
        result_root / "stage8_gene_programs" / "program_genes.tsv",
    )
    cards.to_csv(output_root / "CANDIDATE_EVIDENCE_CARDS.tsv", sep="\t", index=False)
    cards.loc[cards["publication_tier"].eq("Tier A - immediate validation")].to_csv(output_root / "CANDIDATE_TIER_A.tsv", sep="\t", index=False)
    cards.loc[cards["publication_tier"].eq("Tier B - second priority")].to_csv(output_root / "CANDIDATE_TIER_B.tsv", sep="\t", index=False)
    cards.loc[cards["publication_tier"].eq("Tier C - exploratory")].to_csv(output_root / "CANDIDATE_TIER_C.tsv", sep="\t", index=False)

    colors = [
        {"registry": "group", "label": key, "hex_color": value, "meaning": {"Y": "4-month vehicle", "OC": "10-month vehicle", "OT": "10-month MRJP1"}[key]}
        for key, value in GROUP_COLORS.items()
    ] + [
        {"registry": "broad_cell_type", "label": key, "hex_color": value, "meaning": key}
        for key, value in CELL_TYPE_COLORS.items()
    ] + [
        {"registry": "subtype", "label": key, "hex_color": value, "meaning": key}
        for key, value in SUBTYPE_COLORS.items()
    ]
    pd.DataFrame(colors).to_csv(output_root / "publication_color_registry.tsv", sep="\t", index=False)

    figure_manifests: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []

    def add_plot(result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        manifest, qa = result
        figure_manifests.append(manifest)
        qa_rows.append(qa)

    import anndata as ad

    h5ad_path = result_root / "06_annotation_v2.h5ad"
    adata = ad.read_h5ad(h5ad_path, backed="r")
    n_cells, n_features = adata.shape
    coords = np.asarray(adata.obsm["X_umap"])
    atlas = pd.DataFrame(
        {
            "cell_barcode": adata.obs_names.astype(str),
            "UMAP1": coords[:, 0],
            "UMAP2": coords[:, 1],
            "group": adata.obs["group"].astype(str).to_numpy(),
            "library_id": adata.obs["library_id"].astype(str).to_numpy(),
            "cell_type_broad": adata.obs["cell_type_broad"].astype(str).to_numpy(),
            "cell_type_subtype_v2": adata.obs["cell_type_subtype_v2"].astype(str).to_numpy(),
        }
    )
    adata.file.close()
    add_plot(_plot_umap(atlas, "cell_type_broad", CELL_TYPE_COLORS, "atlas_umap_broad_cell_types", figure_root, source_root, title="Mouse ovary single-cell atlas", source_file="results/06_annotation_v2.h5ad::obsm/X_umap + obs/cell_type_broad"))
    granulosa_atlas = atlas.loc[atlas["cell_type_broad"].eq("Granulosa")].copy()
    add_plot(_plot_umap(granulosa_atlas, "cell_type_subtype_v2", SUBTYPE_COLORS, "atlas_umap_granulosa_subtypes", figure_root, source_root, title="Granulosa states", source_file="results/06_annotation_v2.h5ad::obsm/X_umap + obs/cell_type_subtype_v2"))
    stromal_atlas = atlas.loc[atlas["cell_type_broad"].eq("Stromal_fibroblast")].copy()
    add_plot(_plot_umap(stromal_atlas, "cell_type_subtype_v2", SUBTYPE_COLORS, "atlas_umap_stromal_subtypes", figure_root, source_root, title="Stromal states", source_file="results/06_annotation_v2.h5ad::obsm/X_umap + obs/cell_type_subtype_v2"))

    add_plot(_plot_composition(_read_tsv(result_root / "composition" / "library_broad_fractions.tsv"), figure_root, source_root))
    add_plot(_plot_aging_de_summary(_read_tsv(result_root / "de_stage1" / "broad_de_summary.tsv"), figure_root, source_root))
    for population in ["Granulosa", "Stromal_fibroblast"]:
        observed = _read_tsv(result_root / "de_stage1_6" / population / "observed_evidence_levels.tsv.gz")
        add_plot(_plot_reversal_scatter(observed, population, figure_root, source_root))
    add_plot(_plot_population_permutation(tables["population"], figure_root, source_root))
    for population in ["Granulosa", "Stromal_fibroblast"]:
        add_plot(_plot_aging_axis(tables["geometry"], population, figure_root, source_root))
        add_plot(_plot_pathway_reversal(tables["pathways"], population, figure_root, source_root))
        add_plot(_plot_subtype_localization(tables["localized"], population, figure_root, source_root))
        add_plot(_plot_regulators(tables["regulators"], population, figure_root, source_root))
        add_plot(_plot_nmf(tables["programs"], population, figure_root, source_root))
    add_plot(_plot_external_validation(tables["external_paths"], figure_root, source_root))
    add_plot(_plot_decomposition(tables["decomposition"], figure_root, source_root))
    add_plot(_plot_communication(tables["ligands"], figure_root, source_root))
    add_plot(_plot_candidate_tiers(cards, figure_root, source_root))

    figure_manifest = pd.DataFrame(figure_manifests)
    for column in ["source_data", "svg", "pdf", "png_600dpi"]:
        figure_manifest[column] = figure_manifest[column].map(lambda value: str(Path(value).relative_to(paths["root"])))
    figure_manifest.to_csv(output_root / "PUBLICATION_FIGURE_MANIFEST.tsv", sep="\t", index=False)
    qa = pd.DataFrame(qa_rows)
    qa.to_csv(output_root / "PUBLICATION_FIGURE_QA.tsv", sep="\t", index=False)
    if not qa["qa_pass"].all():
        failed = qa.loc[~qa["qa_pass"], "plot_id"].tolist()
        raise RuntimeError(f"Publication figure QA failed: {failed}")

    main_selection = _main_figure_selection(figure_manifest)
    main_selection.to_csv(output_root / "MAIN_FIGURE_SELECTION.tsv", sep="\t", index=False)
    _supplement_plan().to_csv(output_root / "SUPPLEMENTARY_FIGURE_PLAN.tsv", sep="\t", index=False)
    facts = _result_fact_rows(tables, n_cells, n_features)
    facts.to_csv(output_root / "RESULTS_FACT_SHEET_CN.tsv", sep="\t", index=False)
    negatives = _negative_ledger(tables)
    negatives.to_csv(output_root / "NEGATIVE_RESULTS_LEDGER.tsv", sep="\t", index=False)
    _write_reports(output_root, ledger, cards, facts, negatives, figure_manifest, tables)

    outputs = [path for path in output_root.rglob("*") if path.is_file() and path.name not in {"manifest.tsv", "COMPLETE.json"}]
    outputs += [path for path in figure_root.rglob("*") if path.is_file()]
    manifest = pd.DataFrame(
        [
            {
                "path": str(path.relative_to(paths["root"])),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(outputs)
        ]
    )
    manifest.to_csv(output_root / "manifest.tsv", sep="\t", index=False)
    tier_a_names = cards.loc[cards["publication_tier"].eq("Tier A - immediate validation"), "gene"].tolist()
    complete = {
        "stage": 13,
        "status": "COMPLETE",
        "input_git_commit": _git_commit(paths["root"]),
        "n_existing_figures_inventoried": len(inventory),
        "n_standalone_publication_plots": len(figure_manifest),
        "assembled_multipanel_figures_created": 0,
        "figure_1_to_7_panel_planning_complete": True,
        "n_candidate_cards": len(cards),
        "tier_counts": cards["publication_tier"].value_counts().to_dict(),
        "tier_a_candidates": tier_a_names,
        "matched_phenotype_available": False,
        "minimal_recomputation": "publication rendering only; no scientific result recomputed",
        "manifest_sha256": sha256_file(output_root / "manifest.tsv"),
    }
    (output_root / "COMPLETE.json").write_text(json.dumps(complete, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Stage 13 complete: %d standalone plots; %d candidate cards", len(figure_manifest), len(cards))

    print("========================================")
    print("PUBLICATION_STAGE13_COMPLETE")
    print("========================================")
    print("FIGURE_1_TO_7_PANEL_PLANNING=complete")
    print(f"MAIN_TEXT_STANDALONE_PLOTS={int(main_selection['rendered_as_standalone'].sum())}")
    print(f"SUPPLEMENTARY_CANDIDATE_CATEGORIES={len(_supplement_plan())}")
    print(f"TIER_A_COUNT={len(tier_a_names)}")
    print("TIER_A_CANDIDATES=" + ",".join(tier_a_names))
    print("STROMAL_STRONGEST_EVIDENCE=rank_1_of_20;12_strong_pathways;9_of_12_external_direction_support")
    print("GRANULOSA_STRONGEST_EVIDENCE=rank_1_of_20;Protein_secretion;antral_and_preantral_localization")
    print("KEY_NEGATIVE_RESULTS=Theca_no_stable_population_permutation;Granulosa_pathway_external_discordance;no_matched_phenotype")
    print("TOP_VALIDATION=Stromal_inflammation_OXPHOS_secretion;Granulosa_secretion_state;matched_functional_endpoints")
    print("MINIMAL_COMPUTE_NEEDED=publication_rendering_only")
    print("PUBLICATION_STAGE13_REPORT_CN=results/publication_stage13/PUBLICATION_STAGE13_REPORT_CN.md")
    print("STAGE13_EXPLAINED_FOR_USER_CN=results/publication_stage13/STAGE13_EXPLAINED_FOR_USER_CN.md")
    print("MAIN_FIGURE_SELECTION=results/publication_stage13/MAIN_FIGURE_SELECTION.tsv")
    print("RESULTS_FACT_SHEET_CN=results/publication_stage13/RESULTS_FACT_SHEET_CN.tsv")
    print(f"FINAL_GIT_COMMIT={_git_commit(paths['root'])}")
