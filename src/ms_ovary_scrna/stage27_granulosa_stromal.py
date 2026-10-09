"""Read-only focused cycle/composition/function analysis of two frozen populations."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .stage25_bee_mouse_bridge import bh, exact_test, now, sha
from .stage26_all_cell_cycle import _decode_strings, _read_obs_column

OUT = "deep_dive_stage27_granulosa_stromal"
POPS = ["Granulosa", "Stromal_fibroblast"]
LIBRARIES = [f"{g}_{i}" for g in ["Y", "OC", "OT"] for i in [1, 2, 3]]
CONTRASTS = [("OC", "Y"), ("OT", "OC"), ("OT", "Y")]
LOW = "Granulosa_low_complexity_candidate"
CYCLING = "Granulosa_cycling"
QC_COLUMNS = [
    "library_id",
    "group",
    "cell_type_broad_v2",
    "cell_type_subtype_v2",
    "analysis_tier_v2",
    "annotation_confidence_v2",
    "qc_concern_v2",
    "doublet_concern_v2",
    "total_counts",
    "n_genes_by_counts",
    "pct_counts_mt",
    "doublet_score",
    "predicted_doublet",
]
PATHWAYS = [
    "G2M_CHECKPOINT",
    "E2F_TARGETS",
    "MITOTIC_SPINDLE",
    "MYC_TARGETS_V1",
    "P53_PATHWAY",
    "APOPTOSIS",
    "DNA_REPAIR",
    "REACTIVE_OXYGEN_SPECIES_PATHWAY",
    "OXIDATIVE_PHOSPHORYLATION",
    "UNFOLDED_PROTEIN_RESPONSE",
    "TGF_BETA_SIGNALING",
    "EPITHELIAL_MESENCHYMAL_TRANSITION",
    "TNFA_SIGNALING_VIA_NFKB",
    "ESTROGEN_RESPONSE_LATE",
]
GENE_PANELS = {
    "cell_cycle_checkpoint": [
        "Mki67",
        "Pcna",
        "Top2a",
        "Cdk1",
        "Ccnb1",
        "Cdkn1a",
        "Cdkn2a",
        "Trp53",
    ],
    "granulosa_identity_function": [
        "Foxl2",
        "Fshr",
        "Amh",
        "Cyp19a1",
        "Inha",
        "Inhba",
        "Inhbb",
        "Kitl",
        "Gja1",
        "Star",
    ],
    "ECM_TGFbeta": [
        "Col1a1",
        "Col1a2",
        "Col3a1",
        "Dcn",
        "Lum",
        "Fn1",
        "Acta2",
        "Tagln",
        "Tgfb1",
        "Tgfbr1",
        "Tgfbr2",
        "Smad3",
        "Ctgf",
        "Ccn2",
    ],
    "stress_survival": [
        "Hif1a",
        "Atf4",
        "Ddit3",
        "Hspa5",
        "Xbp1",
        "Bax",
        "Bcl2",
        "Il6",
        "Fgf2",
        "Fgfr2",
    ],
}


def write_table(frame: pd.DataFrame, out: Path, name: str) -> None:
    frame.to_csv(out / name, sep="\t", index=False)


def export_metadata(source: Path, root: Path) -> None:
    """只读取 obs 中少量列；绝不构建整个 AnnData 或读取表达层。"""
    out = root / "results" / OUT
    out.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    with h5py.File(source, "r") as handle:
        obs = handle["obs"]
        index_key = obs.attrs.get("_index", "_index")
        if isinstance(index_key, bytes):
            index_key = index_key.decode()
        values = {"cell_barcode": _decode_strings(obs[index_key])}
        for column in QC_COLUMNS:
            if column not in obs:
                raise ValueError(f"Required metadata missing: {column}")
            node = obs[column]
            if isinstance(node, h5py.Dataset) and node.dtype.kind in "bifu":
                values[column] = node[:]
            else:
                values[column] = _read_obs_column(obs, column)
    frame = pd.DataFrame(values)
    if len(frame) != 105763 or not frame.cell_barcode.is_unique:
        raise ValueError("Frozen object metadata audit failed")
    focused = frame[frame.cell_type_broad_v2.isin(POPS)].copy()
    write_table(focused, out, "QC_METADATA.tsv.gz")
    after = source.stat()
    unchanged = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    if not unchanged:
        raise RuntimeError("Source changed during read-only metadata export")
    audit = dict(
        source=str(source),
        n_atlas_cells=len(frame),
        n_focused_cells=len(focused),
        source_bytes=before.st_size,
        source_mtime_ns=before.st_mtime_ns,
        source_unchanged=True,
        export_sha256=sha(out / "QC_METADATA.tsv.gz"),
    )
    (out / "METADATA_EXPORT_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"FOCUSED_METADATA_EXPORTED={len(focused)}", flush=True)


def views(frame: pd.DataFrame, pop: str) -> dict[str, pd.DataFrame]:
    """只是诊断子集视图：没有从主对象删除任何细胞。"""
    block = frame[frame.cell_type_broad_v2.eq(pop)]
    result = {
        "all_annotated": block,
        "Tier1_primary": block[block.analysis_tier_v2.eq("Tier1_primary")],
        "Tier1_plus_Tier2": block[
            block.analysis_tier_v2.isin(["Tier1_primary", "Tier2_sensitivity"])
        ],
    }
    if pop == "Granulosa":
        result.update(
            {
                "without_low_complexity": block[block.cell_type_subtype_v2.ne(LOW)],
                "without_cycling_subtype": block[block.cell_type_subtype_v2.ne(CYCLING)],
                "without_low_complexity_and_cycling": block[
                    ~block.cell_type_subtype_v2.isin([LOW, CYCLING])
                ],
            }
        )
    else:
        result["fibroblast_and_ECM_only"] = block[
            block.cell_type_subtype_v2.isin(["Stromal_fibroblast_candidate", "ECM_high_candidate"])
        ]
    for subtype, subset in block.groupby("cell_type_subtype_v2", observed=True):
        result[f"subtype::{subtype}"] = subset
    return result


def library_summary(frame: pd.DataFrame, pop: str, view: str) -> pd.DataFrame:
    rows = []
    for library, block in frame.groupby("library_id", observed=True):
        phases = block.phase.value_counts()
        row = dict(
            population=pop,
            view=view,
            library_id=library,
            group=block.group.iloc[0],
            n_cells=len(block),
        )
        for phase in ["G1", "S", "G2M"]:
            row[f"{phase}_cells"] = int(phases.get(phase, 0))
            row[f"{phase}_percent"] = 100 * row[f"{phase}_cells"] / len(block)
        row["cycling_percent"] = row["S_percent"] + row["G2M_percent"]
        for column in [
            "S_score",
            "G2M_score",
            "cycle_score_max",
            "phase_margin",
            "total_counts",
            "n_genes_by_counts",
            "pct_counts_mt",
            "doublet_score",
        ]:
            row[f"{column}_median"] = block[column].median()
            if "score" in column:
                row[f"{column}_p90"] = block[column].quantile(0.9)
        # 分数阈值仅诊断靠近零的弱信号；不更改冻结 phase。
        for cutoff in [0.05, 0.10, 0.20]:
            row[f"cycling_score_ge_{cutoff:.2f}_percent"] = (
                100 * (block.phase.isin(["S", "G2M"]) & block.cycle_score_max.ge(cutoff)).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compare_library_scores(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """复用 Stage25 的穷举置换和 BH；每个值必须来自一个独立 library。"""
    rows, loos = [], []
    keys = ["family", "population", "view", "metric"]
    if scores.duplicated(keys + ["library_id"]).any():
        raise ValueError("Multiple observations per library in a statistical test")
    for key, block in scores.groupby(keys, observed=True):
        fields = dict(zip(keys, key))
        for test, reference in CONTRASTS:
            selected = block[block.group.isin([test, reference]) & np.isfinite(block.value)]
            if min(selected.group.eq(test).sum(), selected.group.eq(reference).sum()) < 3:
                continue
            stat = exact_test(selected.value, selected.group, test, reference)
            loo = []
            for removed in selected.library_id:
                subset = selected[selected.library_id.ne(removed)]
                effect = (
                    subset.loc[subset.group.eq(test), "value"].mean()
                    - subset.loc[subset.group.eq(reference), "value"].mean()
                )
                stable = bool(np.sign(effect) == np.sign(stat["effect"]))
                loo.append(effect)
                loos.append(
                    {
                        **fields,
                        "contrast": f"{test}_vs_{reference}",
                        "omitted_library": removed,
                        "effect": effect,
                        "direction_preserved": stable,
                    }
                )
            rows.append(
                {
                    **fields,
                    "contrast": f"{test}_vs_{reference}",
                    **stat,
                    "test_mean": selected.loc[selected.group.eq(test), "value"].mean(),
                    "reference_mean": selected.loc[selected.group.eq(reference), "value"].mean(),
                    "n_test": 3,
                    "n_reference": 3,
                    "loo_min": min(loo),
                    "loo_max": max(loo),
                    "loo_direction_fraction": np.mean(np.sign(loo) == np.sign(stat["effect"])),
                }
            )
    result = pd.DataFrame(rows)
    result["q_bh_family_metric_contrast"] = result.groupby(
        ["family", "metric", "contrast"], observed=True
    ).p_exact_two_sided.transform(bh)
    return result, pd.DataFrame(loos)


def standardize_subtypes(
    summary: pd.DataFrame, min_cells: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = summary[summary.view.str.startswith("subtype::")].copy()
    sub["subtype"] = sub.view.str.removeprefix("subtype::")
    rows, audits = [], []
    for pop, block in sub.groupby("population", observed=True):
        counts = (
            block.pivot(index="subtype", columns="library_id", values="n_cells")
            .reindex(columns=LIBRARIES)
            .fillna(0)
        )
        rates = block.pivot(
            index="subtype", columns="library_id", values="cycling_percent"
        ).reindex(index=counts.index, columns=LIBRARIES)
        common = counts.index[counts.ge(min_cells).all(axis=1)]
        if not len(common):
            raise ValueError(f"No common subtypes in {pop}")
        fractions = counts.loc[common].div(counts.loc[common].sum(axis=0), axis=1)
        weights = fractions.mean(axis=1)  # 每库等权；仅描述性数据驱动固定权重。
        for subtype in counts.index:
            audits.append(
                dict(
                    population=pop,
                    subtype=subtype,
                    min_library_cells=int(counts.loc[subtype].min()),
                    common_subtype=subtype in common,
                    fixed_weight=weights.get(subtype, np.nan),
                    min_cells_required=min_cells,
                )
            )
        for library in LIBRARIES:
            rows.append(
                dict(
                    population=pop,
                    view="fixed_subtype_weights",
                    library_id=library,
                    group=library.split("_")[0],
                    n_cells=int(counts.loc[common, library].sum()),
                    cycling_percent=float((weights * rates.loc[common, library]).sum()),
                    common_subtype_fraction=counts.loc[common, library].sum()
                    / counts[library].sum(),
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(audits)


def symmetric_decomposition(subtype_summary: pd.DataFrame) -> pd.DataFrame:
    """等权文库均值的精确对称混合分解；不将组成/状态项解释为因果。"""
    sub = subtype_summary[subtype_summary.view.str.startswith("subtype::")].copy()
    sub["subtype"] = sub.view.str.removeprefix("subtype::")
    rows = []
    for pop, block in sub.groupby("population", observed=True):
        counts = (
            block.pivot(index="subtype", columns="library_id", values="n_cells")
            .reindex(columns=LIBRARIES)
            .fillna(0)
        )
        rates = (
            block.pivot(index="subtype", columns="library_id", values="cycling_percent")
            .reindex(index=counts.index, columns=LIBRARIES)
            .fillna(0)
        )
        p = counts.div(counts.sum(axis=0), axis=1)
        # R_g=mean_library(p*r)/mean_library(p)，所以 sum(P_g*R_g) 恰等于组文库均值。
        group_p, group_r = {}, {}
        for group in ["Y", "OC", "OT"]:
            libs = [library for library in LIBRARIES if library.startswith(group + "_")]
            group_p[group] = p[libs].mean(axis=1)
            group_r[group] = (p[libs] * rates[libs]).mean(axis=1).div(group_p[group]).fillna(0)
        for test, reference in CONTRASTS:
            p0, p1 = group_p[reference], group_p[test]
            r0, r1 = group_r[reference], group_r[test]
            for subtype in counts.index:
                composition = (p1[subtype] - p0[subtype]) * (r1[subtype] + r0[subtype]) / 2
                state = (r1[subtype] - r0[subtype]) * (p1[subtype] + p0[subtype]) / 2
                rows.append(
                    dict(
                        population=pop,
                        contrast=f"{test}_vs_{reference}",
                        subtype=subtype,
                        composition_component_pp=composition,
                        state_component_pp=state,
                        total_component_pp=composition + state,
                        reference_fraction=p0[subtype],
                        test_fraction=p1[subtype],
                        reference_cycle_percent=r0[subtype],
                        test_cycle_percent=r1[subtype],
                    )
                )
    return pd.DataFrame(rows)


def qc_by_state(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["cycle_status"] = np.where(frame.phase.eq("G1"), "G1", "S_or_G2M")
    rows = []
    for key, block in frame.groupby(
        ["cell_type_broad_v2", "cell_type_subtype_v2", "library_id", "group", "cycle_status"],
        observed=True,
    ):
        row = dict(zip(["population", "subtype", "library_id", "group", "cycle_status"], key))
        row.update(
            n_cells=len(block),
            median_UMI=block.total_counts.median(),
            median_genes=block.n_genes_by_counts.median(),
            median_pct_mt=block.pct_counts_mt.median(),
            median_cycle_score=block.cycle_score_max.median(),
            fraction_score_ge_0_10=block.cycle_score_max.ge(0.1).mean(),
            Tier1_fraction=block.analysis_tier_v2.eq("Tier1_primary").mean(),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def reuse_functional_results(root: Path, out: Path) -> pd.DataFrame:
    r = root / "results"
    pathways, genes, de_summary, subtype_pathways = [], [], [], []
    # publication 图源是“被选出的反向通路”，不是全部结果；不能把未选中项误标 NA。
    all_hallmark = pd.read_csv(out / "input_cache/hallmark_all_results.tsv", sep="\t")
    all_subtype = pd.read_csv(out / "input_cache/subtype_hallmark.tsv", sep="\t")
    for pop in POPS:
        slug = pop.lower()
        selected = all_hallmark[
            all_hallmark.population.eq(pop)
            & all_hallmark.pathway.isin([f"HALLMARK_{x}" for x in PATHWAYS])
        ]
        if selected.duplicated(["pathway", "contrast"]).any():
            raise ValueError("Duplicated frozen Hallmark contrast/pathway")
        frozen = selected.pivot(index="pathway", columns="contrast", values=["NES", "FDR_q"])
        flat = pd.DataFrame({"pathway": frozen.index, "population": pop})
        for prefix, contrast in [
            ("aging", "OC_vs_Y"),
            ("treatment", "OT_vs_OC"),
            ("residual", "OT_vs_Y"),
        ]:
            flat[f"{prefix}_NES"] = frozen[("NES", contrast)].to_numpy()
            flat[f"{prefix}_FDR"] = frozen[("FDR_q", contrast)].to_numpy()
        if len(flat) != len(PATHWAYS) or flat.isna().any().any():
            raise ValueError("Predefined Hallmark panel is incomplete in frozen full results")
        pathways.append(flat)
        subtype_pathways.append(
            all_subtype[
                all_subtype.broad_population.eq(pop)
                & all_subtype.pathway.isin([f"HALLMARK_{x}" for x in PATHWAYS])
            ]
        )
        for contrast in ["oc_vs_y", "ot_vs_oc"]:
            de = pd.read_csv(
                r / "stage14_conventional/figure_source_data" / f"volcano_{slug}_{contrast}.tsv",
                sep="\t",
            )
            de_summary.append(
                dict(
                    population=pop,
                    contrast=contrast.upper().replace("_VS_", "_vs_"),
                    n_tested=len(de),
                    n_FDR_lt_0_05=int(de.padj.lt(0.05).sum()),
                    n_FDR_and_abs_LFC_ge_0_5=int(
                        (de.padj.lt(0.05) & de.log2FoldChange.abs().ge(0.5)).sum()
                    ),
                )
            )
            for panel, selected in GENE_PANELS.items():
                block = de[de.canonical_mouse_symbol.isin(selected)].copy()
                block["panel"] = panel
                genes.append(block)
    write_table(pd.concat(pathways, ignore_index=True), out, "EXISTING_HALLMARK_EVIDENCE.tsv")
    write_table(
        pd.concat(subtype_pathways, ignore_index=True), out, "EXISTING_SUBTYPE_HALLMARK.tsv"
    )
    write_table(pd.concat(genes, ignore_index=True), out, "EXISTING_CANDIDATE_GENE_DE.tsv")
    write_table(pd.DataFrame(de_summary), out, "EXISTING_DE_OVERVIEW.tsv")
    marker_tables = []
    for slug in ["granulosa", "stromal"]:
        marker = pd.read_csv(
            r / f"stage14_conventional/markers/{slug}_marker_catalogue.tsv", sep="\t"
        )
        marker_tables.append(marker[marker.canonical_marker_flag.eq(True) | marker["rank"].le(5)])
    write_table(
        pd.concat(marker_tables, ignore_index=True), out, "EXISTING_IDENTITY_MARKER_REVIEW.tsv"
    )
    modules = pd.read_csv(
        r / "deep_dive_stage15/followup_validation/STATE_MODULE_LIBRARY_SCORES.tsv", sep="\t"
    )
    write_table(modules, out, "EXISTING_STATE_MODULE_LIBRARY_SCORES.tsv")
    # 旧 Stage15/23 使用 plus-one 版本；本轮不覆盖旧表，而以 Stage25 exact 定义另记新结果。
    modules = modules.rename(columns={"module": "view", "score": "value"})
    modules["family"] = "existing_state_module"
    return modules[["family", "population", "view", "metric", "library_id", "group", "value"]]


def markdown_table(frame: pd.DataFrame, decimals: int = 3) -> str:
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "|" + "|".join(["---"] * len(frame.columns)) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [
            f"{x:.{decimals}f}" if isinstance(x, (float, np.floating)) else str(x) for x in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    out: Path,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    decomposition: pd.DataFrame,
    composition: pd.DataFrame,
    n_cells: int,
) -> None:
    chosen = [
        "all_annotated",
        "Tier1_primary",
        "Tier1_plus_Tier2",
        "without_low_complexity",
        "without_low_complexity_and_cycling",
        "fibroblast_and_ECM_only",
        "fixed_subtype_weights",
    ]
    cycle = (
        summary[summary.view.isin(chosen)]
        .groupby(["population", "view", "group"], observed=True)
        .cycling_percent.mean()
        .unstack()
        .reindex(columns=["Y", "OC", "OT"])
        .reset_index()
    )
    cycle["OC-Y_pp"] = cycle.OC - cycle.Y
    cycle["OT-OC_pp"] = cycle.OT - cycle.OC
    cycle_tests = tests[
        (tests.family.eq("cell_cycle"))
        & tests.metric.eq("cycling_percent")
        & tests.view.isin(chosen)
        & tests.contrast.eq("OT_vs_OC")
    ][
        [
            "population",
            "view",
            "effect",
            "p_exact_two_sided",
            "q_bh_family_metric_contrast",
            "loo_min",
            "loo_max",
            "loo_direction_fraction",
        ]
    ]
    subcomp = (
        composition.groupby(["population", "subtype", "group"], observed=True)
        .fraction_of_parent.mean()
        .unstack()
        .reindex(columns=["Y", "OC", "OT"])
    )
    subcomp *= 100
    subcomp["OC-Y_pp"] = subcomp.OC - subcomp.Y
    subcomp["OT-OC_pp"] = subcomp.OT - subcomp.OC
    decomp = (
        decomposition.groupby(["population", "contrast"], observed=True)[
            ["composition_component_pp", "state_component_pp", "total_component_pp"]
        ]
        .sum()
        .reset_index()
    )
    programs = pd.read_csv(out / "EXISTING_HALLMARK_EVIDENCE.tsv", sep="\t")
    programs = programs[
        [
            "population",
            "pathway",
            "aging_NES",
            "aging_FDR",
            "treatment_NES",
            "treatment_FDR",
            "residual_NES",
            "residual_FDR",
        ]
    ]
    module_tests = tests[
        tests.family.eq("existing_state_module")
        & tests.metric.eq("mean")
        & tests.contrast.eq("OT_vs_OC")
    ][["population", "view", "effect", "p_exact_two_sided", "loo_direction_fraction"]]
    text = f"""# 颗粒细胞与成纤维细胞专门解析（Stage27）

## 核心结论（先读这里）

**这两个群体值得重点研究，但现有证据不支持统一解释为“MRJP1促进增殖”。**

- 颗粒细胞：S/G2M比例Y=34.99%、OC=38.51%、OT=41.68%；治疗全量差+3.16百分点，但Tier1为−3.10百分点；剔除low-complexity后仅+0.63，同时剔除cycling亚型后仅+0.16。方向和幅度对注释/质量视图敏感。所有5,139个low-complexity细胞属于既有Tier2，不应把全量上升当成稳健促增殖结论。
- 亚型组成方面，颗粒low-complexity占比由OC 33.78%降至OT 21.12%，atretic-like由14.91%升至26.07%。这是捕获状态混合改变；“atretic-like”只是转录标签，不能直接认定治疗加重闭锁或降低闭锁。
- 基质：S/G2M比例Y=8.53%、OC=5.14%、OT=7.56%。治疗差+2.43百分点在Tier1为+2.85、限定fibroblast+ECM为+2.49，留一文库方向均保留；精确P=0.20，尚非统计确定。主要fibroblast候选亚型2.87%→5.65%，ECM-high仅14.59%→15.47%；不是ECM-high比例扩张（21.62%→21.66%）。
- 基质弱信号仍是重要限制：要求最大周期分数≥0.10时，OC/OT为2.98%/3.33%；≥0.20时为2.53%/2.65%。全量phase回升较多来自靠近零的评分，不能等同于新增强增殖细胞。
- 既有转录证据更适合“功能重塑”的解释：颗粒mitotic-spindle/G2M GSEA治疗方向下降，FOXL2转录部分回升、HIF1A下降；基质FN1上升的年龄变化被治疗方向逆转，DNA-repair/OXPHOS富集上升、TNFα/NFκB与apoptosis富集下降。这些结果不等于已证明蛋白变化、胶原减少或功能恢复。
- 基质ECM/TGFβ变化可定位到相同注释亚型：治疗TGFβ broad NES=−1.646（FDR=0.0072）、ECM-high=−1.719（FDR=0.0060）、fibroblast候选=−1.551（FDR=0.0200）；EMT基因集在这两个亚型内也下降。这里EMT代表ECM相关转录程序，不能在已是间充质的成纤维细胞中直接推断发生/逆转了“上皮间充质转化”。
- 颗粒细胞同时有治疗DNA-repair（NES=+2.093）、OXPHOS（+2.794）与p53（+1.639）富集，不能只挑“年轻化”方向。Cdkn1a治疗padj=0.973，不能声称p21下降或周期阻滞已解除。代谢/检查点与卵泡支持状态重塑比统一“促增殖”更符合当前数据。
- 具体候选：颗粒Foxl2治疗LFC=+0.478、既有padj=0.0163；Hif1a=−0.369、padj=0.0357。基质Fn1年龄LFC=+1.125、治疗LFC=−1.051（治疗padj≈2×10⁻⁵）；Col1a1治疗padj=0.559，没有相同强度支持。保留“ECM重塑”而不直接写“抗纤维化已证实”。

## 本轮完成了什么

复用全图谱105,763个细胞的冻结周期分数，对两个群体共{n_cells:,}个细胞做亚型、质量Tier、弱分数、文库留一敏感性；整合既有pseudobulk/Hallmark和状态模块。本轮未改主对象、未删除细胞、未重做QC/归一化/聚类/DE。所有组均值是3个独立library/pool等权均值。

## 1. 周期变化及敏感性

下表单位为 S+G2M 百分比/百分点。全量与质量分层都是诊断视图，不是新的过滤决策。

{markdown_table(cycle)}

治疗比较的精确置换及单文库留一：

{markdown_table(cycle_tests)}

颗粒细胞的全量治疗变化对low-complexity亚型敏感。该亚型周期分数偏弱，需结合UMI/genes、注释Tier判断，不能直接把所有细胞视为伪影。cycling亚型与phase评分共享周期标记，100% S/G2M不是独立验证。基质要区分ECM-high、fibroblast和smooth-muscle候选，不能仅用broad标签把它们解释成同一种细胞。

0.05/0.10/0.20最大周期分数诊断见WEAK_SCORE_SENSITIVITY.tsv；阈值是探索性显示条件，不是生物学边界或新phase标签。QC_BY_SUBTYPE_LIBRARY_CYCLE.tsv逐库给出周期/G1细胞的UMI、genes、mt及Tier1比例。

## 2. 亚型组成

分母是每库各自broad群体。捕获比例不是组织中绝对细胞数，也不能直接判断闭锁、卵泡成熟或纤维化程度。

{markdown_table(subcomp.reset_index())}

固定权重标准化仅采用九库均≥10细胞的共同亚型，权重为九库组成的等权均值。覆盖度、权重和排除亚型见SUBTYPE_STANDARDIZATION_AUDIT.tsv / LIBRARY_CYCLE_SUMMARY.tsv。标准化不控制未知状态、发情周期和选择性存活。

对称描述性分解（百分点）：

{markdown_table(decomp)}

该分解用P=组内文库亚型比例均值、R=mean_library(p×r)/P；组成项=ΣΔP×平均R，状态项=ΣΔR×平均P，二者严格相加为组文库周期均值之差。不是因果比例，也不等于已证明细胞自主改变。方向相反的分量不计算百分比归因。

## 3. 与既有转录功能证据对照

以下是从全部冻结结果提取的预定义Hallmark面板，包含未显著条目，避免只展示被选中的反向通路。它来自统一9-library模型，不是新测试。FDR显示为0仅是有限置换输出/精度限制，不是真实概率为0。GSEA显著不能换算为增殖细胞比例或生化活性。

{markdown_table(programs, 4)}

已有11个状态模块的治疗均值差如下（不同模块分数不能直接比较绝对效应大小）：

{markdown_table(module_tests, 4)}

候选基因的既有LFC/FDR保留于EXISTING_CANDIDATE_GENE_DE.tsv，含周期检查点、颗粒身份/激素支持、ECM/TGFβ、应激/存活。没有对挑出的基因重新做细胞级检验。亚型GSEA完整方向见EXISTING_SUBTYPE_HALLMARK.tsv。旧Stage15/23用过plus-one置换；本轮复用Stage25完整穷举定义，另写表，不覆盖旧结果。

既有DE/GSEA沿用当时Primary模型定义，不与本轮all-annotated视图当作同一细胞集合。EXISTING_IDENTITY_MARKER_REVIEW.tsv保留已有canonical和top-5亚型标记：antral-like有Inhbb/Hsd17b1/Gja1，preantral-like有Kitl/Igfbp5，cycling有Mki67/Top2a；smooth-muscle候选以Tagln等另列。标记排名仍是描述性的，不能作为新的组间独立显著性检验，也不能把一个混合/低复杂度标签直接解释为一个纯净谱系。

## 4. 生物学解释与验证优先级

- 颗粒细胞：细胞捕获组成、周期转录状态和卵泡功能程序需分别解释。不能因S/G2M增加就称“卵巢功能恢复”；preantral/antral既有mitotic-spindle与E2F等程序也不必同向。
- 成纤维/基质：优先看ECM/TGFβ与代谢/蛋白稳态是否与周期变化一致。ECM-high是转录候选标签，不是直接组织纤维化证据；ECM-high比例变化和细胞内ECM转录变化是两件事。
- 两群体共同变化可作为卵泡—基质微环境联动假设；九库相关仅作探索性描述，组别去均值后相关也不能证明通讯、先后顺序或因果。
- 周期功能：分型共染FOXL2/Ki67或EdU以及DCN/COL1A1/Ki67；结合DNA含量、p-H3和p21以区分活跃周期与阻滞。G1标签不能区分G0/G1，也不是衰老标记。
- 微环境功能：ECM组织学/胶原染色与p-SMAD3；颗粒侧同时测死亡、E2分泌/卵泡存活。可采用颗粒—基质共培养或卵泡培养验证MRJP1依赖性，不将scRNA/VKO候选直接写成已证明机制。
- 复用前序文献上下文：Isola等Nature Aging 2024（10.1038/s43587-023-00552-5）；Wang等Cell 2020（10.1016/j.cell.2020.01.009）；Wu等Nature Aging 2024（10.1038/s43587-024-00607-1）。这些提供背景，不证明本实验MRJP1机制。

## 5. 统计边界、图件和可复现性

每组n=3个library/pool。精确双侧置换穷举20种分配，最小P=0.10；BH在family×metric×contrast内跨视图/群体校正。未检出显著差异不等于没有效应。LOO保留/翻转只作敏感性，不能将LOO当新增重复。发情阶段、批次及pool内动物组成仍缺失。

所有图为Python生成的独立单张图，SVG/PDF/600dpi PNG，源数据另存。单库缺失亚型显示NA；观察到细胞但0个周期细胞显示0。已修正前序全图谱热图将零误标NA的显示问题；细胞分数与注释未变。

INPUT_MANIFEST.tsv记录冻结来源的SHA256；METADATA_EXPORT_AUDIT.json记录服务器主对象只读核验。分析到此结束，不自动重聚类、回归周期或筛细胞。
"""
    (out / "FOCUSED_ANALYSIS_REPORT_CN.md").write_text(text, encoding="utf-8")


def run(root: Path) -> None:
    out = root / "results" / OUT
    out.mkdir(parents=True, exist_ok=True)
    started = now()
    input_cycle = (
        root / "results/deep_dive_stage26_all_cell_cycle/CELL_CYCLE_SCORES_PER_CELL.tsv.gz"
    )
    qc_path = out / "QC_METADATA.tsv.gz"
    atlas = pd.read_csv(input_cycle, sep="\t")
    qc = pd.read_csv(qc_path, sep="\t")
    export_audit = json.loads((out / "METADATA_EXPORT_AUDIT.json").read_text(encoding="utf-8"))
    if sha(qc_path) != export_audit["export_sha256"]:
        raise ValueError("Downloaded metadata checksum does not match server export")
    if len(atlas) != 105763 or not atlas.cell_barcode.is_unique or not qc.cell_barcode.is_unique:
        raise ValueError("Unexpected cell count/duplicated barcodes")
    frame = atlas[atlas.cell_type_broad_v2.isin(POPS)].merge(
        qc,
        on=["cell_barcode", "library_id", "group", "cell_type_broad_v2", "cell_type_subtype_v2"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not frame._merge.eq("both").all() or len(frame) != len(qc):
        raise ValueError("Cycle export and frozen metadata do not match exactly")
    frame = frame.drop(columns="_merge")
    if set(frame.library_id) != set(LIBRARIES):
        raise ValueError("Unexpected library set")
    summaries = []
    for pop in POPS:
        for view, subset in views(frame, pop).items():
            summaries.append(library_summary(subset, pop, view))
    summary = pd.concat(summaries, ignore_index=True)
    standardized, audit = standardize_subtypes(summary)
    summary = pd.concat([summary, standardized], ignore_index=True)
    write_table(summary, out, "LIBRARY_CYCLE_SUMMARY.tsv")
    write_table(audit, out, "SUBTYPE_STANDARDIZATION_AUDIT.tsv")
    score_columns = [
        "cycling_percent",
        "S_percent",
        "G2M_percent",
        "S_score_median",
        "G2M_score_median",
        "cycle_score_max_median",
        "cycling_score_ge_0.05_percent",
        "cycling_score_ge_0.10_percent",
        "cycling_score_ge_0.20_percent",
    ]
    long = summary.melt(
        id_vars=["population", "view", "library_id", "group"],
        value_vars=score_columns,
        var_name="metric",
        value_name="value",
    )
    long["family"] = "cell_cycle"
    composition = pd.concat(
        [
            pd.read_csv(
                root / f"results/stage14_conventional/abundance/{slug}_subtype_abundance.tsv",
                sep="\t",
            )
            for slug in ["granulosa", "stromal"]
        ],
        ignore_index=True,
    ).rename(columns={"parent_broad": "population"})
    # 确认复用的亚型计数与Stage26逐细胞导出完全一致。
    actual = summary[summary.view.str.startswith("subtype::")].copy()
    actual["subtype"] = actual.view.str.removeprefix("subtype::")
    check = composition.merge(
        actual[["population", "subtype", "library_id", "n_cells"]],
        on=["population", "subtype", "library_id"],
        how="outer",
        validate="one_to_one",
        suffixes=("_old", "_new"),
    )
    if not check.n_cells_old.eq(check.n_cells_new).all():
        raise ValueError("Frozen subtype-count parity failed")
    write_table(composition, out, "SUBTYPE_COMPOSITION_BY_LIBRARY.tsv")
    comp_scores = composition.rename(columns={"subtype": "view", "fraction_of_parent": "value"})
    comp_scores["value"] *= 100
    comp_scores["family"], comp_scores["metric"] = "subtype_composition", "percent_of_parent"
    functions = reuse_functional_results(root, out)
    tests, loo = compare_library_scores(
        pd.concat([long, comp_scores[long.columns], functions], ignore_index=True)
    )
    write_table(tests, out, "LIBRARY_EXACT_PERMUTATION.tsv")
    write_table(loo, out, "LIBRARY_LOO_SENSITIVITY.tsv")
    write_table(
        summary[
            [
                "population",
                "view",
                "library_id",
                "group",
                "n_cells",
                "cycling_percent",
                "cycling_score_ge_0.05_percent",
                "cycling_score_ge_0.10_percent",
                "cycling_score_ge_0.20_percent",
            ]
        ],
        out,
        "WEAK_SCORE_SENSITIVITY.tsv",
    )
    decomposition = symmetric_decomposition(summary)
    totals = decomposition.groupby(["population", "contrast"]).total_component_pp.sum()
    for key, value in totals.items():
        expected = tests[
            (tests.family.eq("cell_cycle"))
            & tests.population.eq(key[0])
            & tests.view.eq("all_annotated")
            & tests.metric.eq("cycling_percent")
            & tests.contrast.eq(key[1])
        ].effect.iloc[0]
        if not np.isclose(value, expected, atol=1e-9):
            raise ValueError("Composition/state decomposition additivity failed")
    write_table(decomposition, out, "CYCLE_COMPOSITION_STATE_DECOMPOSITION.tsv")
    write_table(qc_by_state(frame), out, "QC_BY_SUBTYPE_LIBRARY_CYCLE.tsv")
    coupling = (
        summary[summary.view.eq("all_annotated")]
        .pivot(index="library_id", columns="population", values="cycling_percent")
        .reindex(LIBRARIES)
    )
    group = pd.Series([x.split("_")[0] for x in coupling.index], index=coupling.index)
    residual = coupling - coupling.groupby(group).transform("mean")
    coupling_rows = [
        dict(
            scope=scope,
            n_libraries=9,
            spearman_rho=float(spearmanr(d.Granulosa, d.Stromal_fibroblast).statistic),
            interpretation="descriptive_not_communication_or_causality",
        )
        for scope, d in [("raw", coupling), ("group_centered", residual)]
    ]
    write_table(pd.DataFrame(coupling_rows), out, "CROSS_POPULATION_COUPLING_DESCRIPTIVE.tsv")
    write_report(out, summary, tests, decomposition, composition, len(frame))
    inputs = [
        input_cycle,
        qc_path,
        root / "results/deep_dive_stage15/followup_validation/STATE_MODULE_LIBRARY_SCORES.tsv",
    ]
    inputs += [
        out / "input_cache" / name for name in ["hallmark_all_results.tsv", "subtype_hallmark.tsv"]
    ]
    inputs += [
        root
        / "results/stage14_conventional/figure_source_data"
        / f"volcano_{pop.lower()}_{contrast}.tsv"
        for pop in POPS
        for contrast in ["oc_vs_y", "ot_vs_oc"]
    ]
    inputs += [
        root / f"results/stage14_conventional/abundance/{pop}_subtype_abundance.tsv"
        for pop in ["granulosa", "stromal"]
    ]
    inputs += [
        root / f"results/stage14_conventional/markers/{pop}_marker_catalogue.tsv"
        for pop in ["granulosa", "stromal"]
    ]
    write_table(
        pd.DataFrame(
            [
                dict(input=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=sha(p))
                for p in inputs
            ]
        ),
        out,
        "INPUT_MANIFEST.tsv",
    )
    state = dict(
        status="complete",
        started_at=started,
        finished_at=now(),
        python=platform.python_version(),
        n_focused_cells=len(frame),
        n_atlas_cells=len(atlas),
        n_Granulosa=int(frame.cell_type_broad_v2.eq("Granulosa").sum()),
        n_Stromal=int(frame.cell_type_broad_v2.eq("Stromal_fibroblast").sum()),
        subtype_parity=True,
        decomposition_additivity=True,
        main_h5ad_not_written=True,
        statistical_unit="independent_library_pool",
        n_per_group=3,
    )
    (out / "RUN_STATE.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print("GRANULOSA_STROMAL_FOCUSED_ANALYSIS_COMPLETE", flush=True)
