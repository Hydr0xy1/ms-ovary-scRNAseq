"""Stage 9: targeted Stromal -> Granulosa communication hypotheses.

This module deliberately avoids a global all-cell-type network.  It is gated by prior
evidence, uses the LIANA mouse consensus ligand-receptor resource, and asks whether
CytoSig ligand-response targets explain the pre-existing Granulosa reversal genes.
"""
# ruff: noqa: E501

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .project import project_paths, setup_logging
from .stage7_regulatory_activity import _read_counts, sha256_file

SENDER = "Stromal_fibroblast"
RECEIVER = "Granulosa"
CPM_EXPRESSION_THRESHOLD = 1.0
CYTOSIG_TARGET_QUANTILE = 0.95


def communication_gate(pathway_evidence: pd.DataFrame) -> tuple[bool, str]:
    stromal = pathway_evidence.loc[pathway_evidence["population"].astype(str).eq(SENDER)]
    granulosa = pathway_evidence.loc[pathway_evidence["population"].astype(str).eq(RECEIVER)]
    stromal_supported = stromal.get("pathway_evidence_level", pd.Series(dtype=str)).astype(str).eq(
        "Strong_multi_metric_reversal"
    ).any()
    granulosa_supported = (
        granulosa.get("direction_opposite", pd.Series(dtype=bool)).fillna(False).astype(bool)
        & granulosa.get("aging_GSEA_supported", pd.Series(dtype=bool)).fillna(False).astype(bool)
    ).any()
    passed = bool(stromal_supported and granulosa_supported)
    reason = (
        "Stromal strong multi-metric reversal and Granulosa aging-supported opposite response"
        if passed
        else "Prior evidence gate not met for targeted Stromal-to-Granulosa analysis"
    )
    return passed, reason


def _cpm(counts: pd.DataFrame) -> pd.DataFrame:
    values = counts.to_numpy(dtype=float)
    values = values / np.maximum(values.sum(axis=1, keepdims=True), 1.0) * 1e6
    return pd.DataFrame(values, index=counts.index, columns=counts.columns)


def _split_complex(value: str) -> list[str]:
    return [part for part in str(value).split("_") if part]


def filter_cytosig_targets(
    cytosig: pd.DataFrame,
    quantile: float = CYTOSIG_TARGET_QUANTILE,
) -> pd.DataFrame:
    """Keep each ligand's strongest CytoSig response weights before overlap tests."""
    required = {"ligand_key", "score"}
    missing = required.difference(cytosig.columns)
    if missing:
        raise KeyError(f"CytoSig table missing columns: {sorted(missing)}")
    result = cytosig.copy()
    result["abs_score"] = result["score"].abs()
    thresholds = result.groupby("ligand_key", observed=True)["abs_score"].transform(
        lambda values: values.quantile(quantile)
    )
    return result.loc[result["abs_score"].ge(thresholds)].copy()


def select_candidate_ligands(candidates: pd.DataFrame, min_targets: int = 5) -> pd.DataFrame:
    """Apply the predefined sender-pattern and receiver-target evidence gate."""
    return candidates.loc[
        candidates["sender_directional_rescue_candidate"].fillna(False).astype(bool)
        & candidates["n_supported_receiver_targets"].ge(min_targets)
    ].copy()


def complex_expression_support(cpm: pd.DataFrame, complex_name: str) -> tuple[bool, float, str]:
    subunits = _split_complex(complex_name)
    missing = [gene for gene in subunits if gene not in cpm.columns]
    if missing:
        return False, 0.0, ";".join(missing)
    old = cpm.loc[[idx for idx in cpm.index if str(idx).startswith(("OC_", "OT_"))], subunits]
    per_subunit = old.median(axis=0)
    minimum = float(per_subunit.min())
    return bool((per_subunit >= CPM_EXPRESSION_THRESHOLD).all()), minimum, ""


def _load_resources(resource_root: Path, refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resource_root.mkdir(parents=True, exist_ok=True)
    lr_path = resource_root / "liana_mouseconsensus.tsv.gz"
    cytosig_path = resource_root / "cytosig_mouse.tsv.gz"
    if refresh or not lr_path.exists():
        import liana as li

        li.rs.select_resource("mouseconsensus").to_csv(lr_path, sep="\t", index=False, compression="gzip")
    if refresh or not cytosig_path.exists():
        import decoupler as dc

        dc.op.resource(name="CytoSig", organism="mouse", verbose=True).to_csv(
            cytosig_path, sep="\t", index=False, compression="gzip"
        )
    lr = pd.read_csv(lr_path, sep="\t")
    cytosig = pd.read_csv(cytosig_path, sep="\t")
    provenance = pd.DataFrame(
        [
            {"resource": "LIANA mouseconsensus", "rows": len(lr), "file": str(lr_path), "sha256": sha256_file(lr_path)},
            {"resource": "OmniPath CytoSig mouse translation", "rows": len(cytosig), "file": str(cytosig_path), "sha256": sha256_file(cytosig_path)},
        ]
    )
    return lr, cytosig, provenance


def _receiver_targets(paths: Mapping[str, Path]) -> pd.DataFrame:
    effects = pd.read_csv(
        paths["results"] / "de_stage1_5" / RECEIVER / "rescue_ready_effects.tsv.gz", sep="\t"
    )
    directional = effects.loc[
        effects["directional_rescue_candidate"].fillna(False).astype(bool),
        ["gene", "aging_effect", "treatment_effect", "residual_effect", "aging_padj", "treatment_padj", "primary_classification"],
    ].copy()
    leading = pd.read_csv(
        paths["results"] / "pathway_stage2" / "hallmark_leading_edge_review.tsv", sep="\t"
    )
    leading = leading.loc[leading["population"].astype(str).eq(RECEIVER)]
    leading_genes: set[str] = set()
    for column in ["shared_opposite_direction_genes", "aging_leading_edge", "treatment_leading_edge"]:
        if column in leading:
            for value in leading[column].dropna().astype(str):
                leading_genes.update(x for x in value.split(";") if x and x.lower() != "nan")
    directional["stage2_leading_edge"] = directional["gene"].astype(str).isin(leading_genes)
    return directional


def _abundance_inventory(paths: Mapping[str, Path]) -> pd.DataFrame:
    qc = pd.read_csv(paths["results"] / "pseudobulk_ready" / "pseudobulk_qc.tsv", sep="\t")
    population_col = "population" if "population" in qc.columns else "cell_type_broad_v2"
    selected = qc.loc[qc[population_col].astype(str).isin([SENDER, RECEIVER])].copy()
    rename_columns = {population_col: "population"}
    if "library_id" not in selected.columns and "library" in selected.columns:
        rename_columns["library"] = "library_id"
    if "detected_genes" not in selected.columns and "n_expressed_genes" in selected.columns:
        rename_columns["n_expressed_genes"] = "detected_genes"
    selected = selected.rename(columns=rename_columns)
    keep = [c for c in ["population", "library_id", "group", "n_cells", "total_umi", "detected_genes"] if c in selected.columns]
    selected = selected[keep]
    if "group" not in selected and "library_id" in selected:
        selected["group"] = selected["library_id"].astype(str).str.split("_").str[0]
    if "n_cells" in selected:
        totals = selected.groupby("library_id", observed=True)["n_cells"].transform("sum")
        selected["fraction_within_sender_receiver"] = selected["n_cells"] / totals
    return selected


def run_stage9(config: Mapping[str, Any], *, refresh_resources: bool = False) -> None:
    paths = project_paths(config)
    output_root = paths["results"] / "stage9_communication"
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("18_stage9_communication", dict(config))
    evidence = pd.read_csv(
        paths["results"] / "pathway_stage2" / "primary_pathway_reversal_summary.tsv", sep="\t"
    )
    passed, gate_reason = communication_gate(evidence)
    if not passed:
        (output_root / "COMMUNICATION_NOT_YET_JUSTIFIED.md").write_text(
            f"# Stage 9 skipped\n\n{gate_reason}。因此没有运行全局通信网络，也没有从大量interaction中事后挑选故事。\n",
            encoding="utf-8",
        )
        print("STAGE9_COMMUNICATION_SKIPPED")
        return

    lr, cytosig, provenance = _load_resources(output_root / "resources", refresh_resources)
    provenance.to_csv(output_root / "resource_provenance.tsv", sep="\t", index=False)
    counts = _read_counts(paths["results"] / "pseudobulk_ready" / "broad_counts.tsv.gz")
    sender = _cpm(counts.xs(SENDER, level="population"))
    receiver = _cpm(counts.xs(RECEIVER, level="population"))
    sender_effects = pd.read_csv(
        paths["results"] / "de_stage1_5" / SENDER / "rescue_ready_effects.tsv.gz", sep="\t"
    ).set_index("gene")
    receiver_targets = _receiver_targets(paths)
    target_set = set(receiver_targets["gene"].astype(str))

    lr_rows: list[dict[str, Any]] = []
    for row in lr.drop_duplicates(["ligand", "receptor"]).itertuples(index=False):
        ligand_ok, ligand_cpm, ligand_missing = complex_expression_support(sender, str(row.ligand))
        receptor_ok, receptor_cpm, receptor_missing = complex_expression_support(receiver, str(row.receptor))
        if ligand_ok and receptor_ok:
            lr_rows.append(
                {
                    "sender": SENDER,
                    "receiver": RECEIVER,
                    "ligand": str(row.ligand),
                    "receptor": str(row.receptor),
                    "sender_min_subunit_median_cpm": ligand_cpm,
                    "receiver_min_subunit_median_cpm": receptor_cpm,
                    "ligand_missing_subunits": ligand_missing,
                    "receptor_missing_subunits": receptor_missing,
                }
            )
    supported_lr = pd.DataFrame(lr_rows)
    if supported_lr.empty:
        raise RuntimeError("No expressed Stromal-to-Granulosa ligand-receptor pairs")

    cyto = cytosig.copy()
    cyto["ligand_key"] = cyto["cytokine_genesymbol"].astype(str).str.upper()
    cyto["target_gene"] = cyto["genesymbol"].astype(str)
    cyto = filter_cytosig_targets(cyto)
    cyto = cyto.loc[cyto["target_gene"].isin(target_set)]
    target_rows: list[pd.DataFrame] = []
    ligand_rows: list[dict[str, Any]] = []
    for ligand_complex, lr_sub in supported_lr.groupby("ligand", observed=True):
        ligand_subunits = _split_complex(ligand_complex)
        simple_ligands = [gene for gene in ligand_subunits if gene in sender.columns]
        links = cyto.loc[cyto["ligand_key"].isin([gene.upper() for gene in simple_ligands])].copy()
        if links.empty:
            continue
        links["ligand"] = ligand_complex
        links["sender"] = SENDER
        links["receiver"] = RECEIVER
        links = links.merge(receiver_targets, left_on="target_gene", right_on="gene", how="left")
        target_rows.append(links)
        simple = simple_ligands[0]
        effect = sender_effects.loc[simple] if simple in sender_effects.index else pd.Series(dtype=float)
        ligand_rows.append(
            {
                "sender": SENDER,
                "receiver": RECEIVER,
                "ligand": ligand_complex,
                "n_expressed_receptors": lr_sub["receptor"].nunique(),
                "n_supported_receiver_targets": links["target_gene"].nunique(),
                "n_stage2_leading_edge_targets": links.loc[links["stage2_leading_edge"].fillna(False), "target_gene"].nunique(),
                "sender_aging_effect": effect.get("aging_effect", np.nan),
                "sender_treatment_effect": effect.get("treatment_effect", np.nan),
                "sender_residual_effect": effect.get("residual_effect", np.nan),
                "sender_directional_rescue_candidate": bool(effect.get("directional_rescue_candidate", False)),
                "sender_FDR_supported_rescue_candidate": bool(effect.get("FDR_supported_rescue_candidate", False)),
                "median_sender_cpm": float(sender[simple].median()),
                "median_abs_ligand_target_weight": float(links["score"].abs().median()),
            }
        )
    ligand_targets = pd.concat(target_rows, ignore_index=True) if target_rows else pd.DataFrame()
    candidates = pd.DataFrame(ligand_rows)
    if candidates.empty:
        raise RuntimeError("No CytoSig-supported ligands overlap receiver reversal targets")
    candidates = select_candidate_ligands(candidates, min_targets=5).sort_values(
        ["sender_directional_rescue_candidate", "n_stage2_leading_edge_targets", "n_supported_receiver_targets", "ligand"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    candidate_names = set(candidates["ligand"])
    receptors = supported_lr.loc[supported_lr["ligand"].isin(candidate_names)].copy()
    ligand_targets = ligand_targets.loc[ligand_targets["ligand"].isin(candidate_names)].copy()
    abundance = _abundance_inventory(paths)
    communication = receptors.merge(candidates, on=["sender", "receiver", "ligand"], how="left")
    communication["interpretation"] = "communication_hypothesis_consistent_with_transcriptomic_changes"

    abundance.to_csv(output_root / "sender_receiver_inventory.tsv", sep="\t", index=False)
    candidates.to_csv(output_root / "candidate_ligands.tsv", sep="\t", index=False)
    receptors.to_csv(output_root / "candidate_receptors.tsv", sep="\t", index=False)
    ligand_targets.to_csv(output_root / "ligand_target_links.tsv", sep="\t", index=False)
    communication.to_csv(output_root / "communication_evidence.tsv", sep="\t", index=False)
    top = candidates.head(20)
    report = [
        "# Stage 9 定向细胞通讯分析报告",
        "",
        "## 1. 为什么做及gate",
        f"本阶段只因“{gate_reason}”而进入，预先限定 Stromal_fibroblast → Granulosa；没有运行全局网络后挑选故事。",
        "",
        "## 2. 输入与方法",
        f"LIANA mouseconsensus用于配体-受体配对；sender/receiver表达支持来自各library raw-count pseudobulk CPM，复合物要求所有亚基中位数≥1 CPM。CytoSig作为ligand-response靶基因资源，预先保留每个配体绝对响应权重top {(1 - CYTOSIG_TARGET_QUANTILE) * 100:.0f}%，再检验是否覆盖Stage1.5已有Granulosa reversal genes及Stage2 leading-edge genes。候选配体还必须在sender自身表现为aging–treatment方向反转。",
        "",
        "## 3. 丰度混杂检查",
        "sender_receiver_inventory.tsv并列报告每个library的sender/receiver细胞数与相对丰度。候选interaction不能脱离细胞丰度变化单独解释。",
        "",
        "## 4. 结果",
        f"表达门控后有{len(supported_lr)}条Stromal→Granulosa LR pair；其中{len(candidates)}个ligand至少连接5个预先定义的receiver reversal targets。候选按证据列字典序排列，没有构造任意加权黑箱score。",
        top.to_markdown(index=False),
        "",
        "## 5. 可以说明什么？",
        "这些结果仅提出：某些sender ligand、receiver receptor与receiver转录靶基因的组合与已观察到的衰老/MRJP1反向变化相容。",
        "",
        "## 6. 不能说明什么？",
        "不能声称MRJP1通过某配体发挥作用，也不能证明配体被分泌、受体被激活或存在因果传递。CytoSig跨情境资源和转录表达都不是直接蛋白信号证据。",
        "",
        "## 7. 验证建议",
        "优先对多层证据一致候选做蛋白/上清测定、受体磷酸化、条件培养基/共培养、阻断抗体或受体抑制实验。",
    ]
    (output_root / "CELL_COMMUNICATION_REPORT_CN.md").write_text("\n".join(report), encoding="utf-8")

    outputs = [p for p in output_root.rglob("*") if p.is_file() and p.name not in {"manifest.tsv", "COMPLETE.json"}]
    manifest = pd.DataFrame([{"path": str(p.relative_to(paths["root"])), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in outputs])
    manifest.to_csv(output_root / "manifest.tsv", sep="\t", index=False)
    complete = {
        "stage": 9,
        "status": "COMPLETE",
        "gate": gate_reason,
        "statistical_unit": "library",
        "sender": SENDER,
        "receiver": RECEIVER,
        "software": {"liana": version("liana"), "decoupler": version("decoupler")},
        "manifest_sha256": sha256_file(output_root / "manifest.tsv"),
    }
    (output_root / "COMPLETE.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    logger.info("Stage 9 complete: %d candidate ligands", len(candidates))
    print("STAGE9_TARGETED_COMMUNICATION_COMPLETE")
