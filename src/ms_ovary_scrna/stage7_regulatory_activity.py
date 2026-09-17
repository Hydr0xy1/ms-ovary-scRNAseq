"""Stage 7: sample-level TF and pathway activity inference.

The activity estimates use target-gene patterns (CollecTRI/PROGENy) in library-level
pseudobulk log-CPM matrices.  TF messenger-RNA abundance is never used as a proxy for
TF activity.  Formal OC/OT label sensitivity uses the same 20 exact assignments as
Stage 1.6.
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .de_stage1_6 import enumerate_permutation_assignments
from .project import project_paths, setup_logging
from .stage5_rejuvenation_geometry import log_cpm


PRIMARY_BROAD = ("Granulosa", "Stromal_fibroblast")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_counts(path: Path) -> pd.DataFrame:
    counts = pd.read_csv(path, sep="\t", index_col=[0, 1], compression="infer")
    counts.index = pd.MultiIndex.from_tuples(
        [(str(a), str(b)) for a, b in counts.index], names=["population", "library_id"]
    )
    if (counts.to_numpy() < 0).any():
        raise ValueError(f"Negative pseudobulk counts in {path}")
    rounded = np.rint(counts.to_numpy(dtype=float))
    if not np.allclose(counts.to_numpy(dtype=float), rounded):
        raise ValueError(f"Non-integer pseudobulk counts in {path}")
    return counts


def activity_contrasts(activity: pd.DataFrame) -> pd.DataFrame:
    """Summarize library-level activity without treating cells as replicates."""

    required = {"Y", "OC", "OT"}
    groups = pd.Series({idx: str(idx).split("_", 1)[0] for idx in activity.index})
    if set(groups) != required:
        raise ValueError(f"Expected Y/OC/OT libraries, found {sorted(set(groups))}")
    means = {label: activity.loc[groups.eq(label)].mean(axis=0) for label in required}
    rows: list[dict[str, Any]] = []
    for program in activity.columns:
        aging = float(means["OC"][program] - means["Y"][program])
        treatment = float(means["OT"][program] - means["OC"][program])
        residual = float(means["OT"][program] - means["Y"][program])
        rows.append(
            {
                "program": str(program),
                "mean_Y": float(means["Y"][program]),
                "mean_OC": float(means["OC"][program]),
                "mean_OT": float(means["OT"][program]),
                "aging_effect_OC_minus_Y": aging,
                "treatment_effect_OT_minus_OC": treatment,
                "residual_effect_OT_minus_Y": residual,
                "directionally_reversed": bool(aging * treatment < 0),
                "closer_to_young_after_treatment": bool(abs(residual) < abs(aging)),
                "opposition_magnitude": float(-np.sign(aging) * treatment)
                if aging != 0
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def exact_activity_permutation(
    activity: pd.DataFrame,
    selected_programs: list[str],
) -> pd.DataFrame:
    """Evaluate preselected programs across the exact 20 old-library assignments."""

    assignments = enumerate_permutation_assignments()
    y_ids = [idx for idx in activity.index if str(idx).startswith("Y_")]
    observed = activity_contrasts(activity).set_index("program")
    rows: list[dict[str, Any]] = []
    for program in selected_programs:
        observed_aging = float(observed.loc[program, "aging_effect_OC_minus_Y"])
        direction = float(np.sign(observed_aging))
        for permutation_id, assignment in assignments.groupby("permutation_id", observed=True):
            labels = assignment.set_index("library_id")["model_group"]
            oc_ids = labels[labels.eq("OC")].index.tolist()
            ot_ids = labels[labels.eq("OT")].index.tolist()
            treatment = float(activity.loc[ot_ids, program].mean() - activity.loc[oc_ids, program].mean())
            rows.append(
                {
                    "program": program,
                    "permutation_id": permutation_id,
                    "is_observed": bool(assignment["is_observed"].iloc[0]),
                    "observed_aging_direction": direction,
                    "model_aging_effect": float(
                        activity.loc[oc_ids, program].mean() - activity.loc[y_ids, program].mean()
                    ),
                    "model_treatment_effect": treatment,
                    "opposition_statistic": float(-direction * treatment),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    summaries: list[pd.DataFrame] = []
    for program, sub in result.groupby("program", observed=True):
        observed_stat = float(sub.loc[sub["is_observed"], "opposition_statistic"].iloc[0])
        rank = int(1 + (sub.loc[~sub["is_observed"], "opposition_statistic"] >= observed_stat).sum())
        tmp = sub.copy()
        tmp["observed_rank_high_is_more_opposed"] = rank
        tmp["empirical_p"] = rank / len(sub)
        summaries.append(tmp)
    return pd.concat(summaries, ignore_index=True)


def _load_or_fetch_networks(
    resource_root: Path,
    *,
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resource_root.mkdir(parents=True, exist_ok=True)
    collectri_path = resource_root / "collectri_mouse.tsv.gz"
    progeny_path = resource_root / "progeny_mouse_top500.tsv.gz"
    if refresh or not collectri_path.exists() or not progeny_path.exists():
        import decoupler as dc

        collectri = dc.op.collectri(organism="mouse", verbose=True)
        progeny = dc.op.progeny(organism="mouse", top=500, verbose=True)
        collectri.to_csv(collectri_path, sep="\t", index=False, compression="gzip")
        progeny.to_csv(progeny_path, sep="\t", index=False, compression="gzip")
    collectri = pd.read_csv(collectri_path, sep="\t")
    progeny = pd.read_csv(progeny_path, sep="\t")
    provenance = pd.DataFrame(
        [
            {
                "resource": "CollecTRI",
                "organism": "mouse",
                "rows": len(collectri),
                "file": str(collectri_path),
                "sha256": sha256_file(collectri_path),
            },
            {
                "resource": "PROGENy",
                "organism": "mouse",
                "parameter": "top=500",
                "rows": len(progeny),
                "file": str(progeny_path),
                "sha256": sha256_file(progeny_path),
            },
        ]
    )
    return collectri, progeny, provenance


def _estimate_activity(expression: pd.DataFrame, network: pd.DataFrame) -> pd.DataFrame:
    import decoupler as dc

    estimate, _pvalues = dc.mt.ulm(expression, network, tmin=5, verbose=False)
    estimate.index = expression.index
    return estimate


def _eligible_matrices(paths: Mapping[str, Path]) -> list[tuple[str, str, pd.DataFrame]]:
    matrices: list[tuple[str, str, pd.DataFrame]] = []
    broad = _read_counts(paths["results"] / "pseudobulk_ready" / "broad_counts.tsv.gz")
    for population in PRIMARY_BROAD:
        if population in broad.index.get_level_values("population"):
            matrix = broad.xs(population, level="population").reindex(
                ["Y_1", "Y_2", "Y_3", "OC_1", "OC_2", "OC_3", "OT_1", "OT_2", "OT_3"]
            )
            matrices.append(("broad", population, matrix))
    stage3 = paths["results"] / "stage3_subtype_localization"
    eligibility = pd.read_csv(stage3 / "subtype_eligibility.tsv", sep="\t")
    subtype = _read_counts(stage3 / "subtype_pseudobulk_counts.tsv.gz")
    primary = eligibility.loc[
        eligibility["eligibility"].astype(str).eq("Primary_DE_ready"), "subtype"
    ].astype(str)
    for population in primary:
        if population not in subtype.index.get_level_values("population"):
            continue
        matrix = subtype.xs(population, level="population").reindex(
            ["Y_1", "Y_2", "Y_3", "OC_1", "OC_2", "OC_3", "OT_1", "OT_2", "OT_3"]
        )
        if not matrix.isna().any().any():
            matrices.append(("subtype", population, matrix))
    return matrices


def _write_report(
    output_root: Path,
    tf_reversal: pd.DataFrame,
    pathway_reversal: pd.DataFrame,
    permutation: pd.DataFrame,
    provenance: pd.DataFrame,
) -> None:
    tf_supported = tf_reversal.loc[
        tf_reversal["directionally_reversed"] & tf_reversal["closer_to_young_after_treatment"]
    ]
    path_supported = pathway_reversal.loc[
        pathway_reversal["directionally_reversed"]
        & pathway_reversal["closer_to_young_after_treatment"]
    ]
    rank1 = (
        permutation.loc[permutation["is_observed"] & permutation["observed_rank_high_is_more_opposed"].eq(1)]
        if not permutation.empty
        else permutation
    )
    lines = [
        "# Stage 7 调控活性分析报告",
        "",
        "## 1. 为什么做这一步？",
        "基因和Hallmark变化能够描述现象，但不能直接指出上游调控程序。本阶段用已知靶基因的整体表达模式估计TF与信号通路活性，寻找衰老改变且在MRJP1组反向的候选调控程序。",
        "",
        "## 2. 输入与统计单位",
        "输入为Tier1细胞构建的raw integer pseudobulk counts；先在每个population内转换为library-level log-CPM。正式重复为9个library（Y/OC/OT各3个），不是single cell。",
        "",
        "## 3. 方法",
        "使用decoupler ULM。TF活性来自CollecTRI靶基因及其符号权重，通路活性来自PROGENy响应基因；TF自身mRNA不等同于TF活性。每个population分别估计。最强候选在选择后复用20种exact OC/OT labels进行置换。",
        "",
        "## 4. 资源与可审计性",
        provenance.to_markdown(index=False),
        "",
        "## 5. 实际结果",
        f"共得到 {len(tf_reversal):,} 条TF-population对，其中 {len(tf_supported):,} 条同时满足方向反转和OT更接近Y。",
        f"共得到 {len(pathway_reversal):,} 条PROGENy-population对，其中 {len(path_supported):,} 条满足同一描述性条件。",
        f"预先选入exact permutation的候选中，真实标签rank 1/20的有 {len(rank1):,} 条。",
        "",
        "### 最强TF候选（按反向幅度）",
        tf_supported.sort_values("opposition_magnitude", ascending=False).head(20).to_markdown(index=False),
        "",
        "### 最强通路候选（按反向幅度）",
        path_supported.sort_values("opposition_magnitude", ascending=False).head(20).to_markdown(index=False),
        "",
        "## 6. 可以说明什么？",
        "可以提出与衰老/MRJP1转录响应一致的候选TF和信号响应程序，并判断其是否局限于某一亚型。exact permutation说明真实OC/OT标签下的反向幅度在20种可能分配中的相对位置。",
        "",
        "## 7. 不能说明什么？",
        "ULM活性是由靶基因模式推断的转录活性，不是TF蛋白、磷酸化、核转位或DNA结合的直接证据，也不能证明MRJP1直接作用于这些TF/通路。",
        "",
        "## 8. 局限",
        "每组只有3个library；资源网络来自跨组织知识库；pseudobulk会隐藏细胞内异质性；exact permutation分辨率最低为0.05。",
        "",
        "## 9. 下一步",
        "仅把多层证据一致的候选纳入Stage 10 evidence matrix，并用蛋白/磷酸化、CUT&RUN或功能干预进行验证。",
    ]
    (output_root / "REGULATORY_ACTIVITY_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def run_stage7(config: Mapping[str, Any], *, refresh_resources: bool = False) -> None:
    paths = project_paths(config)
    output_root = paths["results"] / "stage7_regulatory_activity"
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(paths["logs"] / "stage7_regulatory_activity.log")
    collectri, progeny, provenance = _load_or_fetch_networks(
        output_root / "resources", refresh=refresh_resources
    )
    provenance.to_csv(output_root / "resource_provenance.tsv", sep="\t", index=False)

    tf_activity_rows: list[pd.DataFrame] = []
    pathway_activity_rows: list[pd.DataFrame] = []
    tf_reversal_rows: list[pd.DataFrame] = []
    pathway_reversal_rows: list[pd.DataFrame] = []
    permutation_rows: list[pd.DataFrame] = []
    for level, population, counts in _eligible_matrices(paths):
        expression = log_cpm(counts)
        for activity_type, network in (("TF", collectri), ("PROGENy", progeny)):
            estimate = _estimate_activity(expression, network)
            long = estimate.rename_axis("library_id").reset_index().melt(
                id_vars="library_id", var_name="program", value_name="activity"
            )
            long.insert(0, "analysis_level", level)
            long.insert(1, "population", population)
            long["group"] = long["library_id"].str.split("_").str[0]
            contrasts = activity_contrasts(estimate)
            contrasts.insert(0, "analysis_level", level)
            contrasts.insert(1, "population", population)
            contrasts.insert(2, "activity_type", activity_type)
            eligible = contrasts.loc[
                contrasts["directionally_reversed"]
                & contrasts["closer_to_young_after_treatment"]
                & contrasts["opposition_magnitude"].gt(0)
            ].sort_values(
                ["opposition_magnitude", "program"], ascending=[False, True], kind="stable"
            )
            selected = eligible.head(10)["program"].tolist()
            perm = exact_activity_permutation(estimate, selected)
            if not perm.empty:
                perm.insert(0, "analysis_level", level)
                perm.insert(1, "population", population)
                perm.insert(2, "activity_type", activity_type)
                permutation_rows.append(perm)
            if activity_type == "TF":
                tf_activity_rows.append(long)
                tf_reversal_rows.append(contrasts)
            else:
                pathway_activity_rows.append(long)
                pathway_reversal_rows.append(contrasts)
        logger.info("Regulatory activity complete: %s/%s", level, population)

    tf_activity = pd.concat(tf_activity_rows, ignore_index=True)
    pathway_activity = pd.concat(pathway_activity_rows, ignore_index=True)
    tf_reversal = pd.concat(tf_reversal_rows, ignore_index=True)
    pathway_reversal = pd.concat(pathway_reversal_rows, ignore_index=True)
    permutation = pd.concat(permutation_rows, ignore_index=True) if permutation_rows else pd.DataFrame()
    tf_activity.to_csv(output_root / "tf_activity.tsv", sep="\t", index=False)
    pathway_activity.to_csv(output_root / "pathway_activity.tsv", sep="\t", index=False)
    tf_reversal.to_csv(output_root / "tf_reversal.tsv", sep="\t", index=False)
    pathway_reversal.to_csv(output_root / "pathway_reversal.tsv", sep="\t", index=False)
    permutation.to_csv(output_root / "tf_permutation.tsv", sep="\t", index=False)

    candidates = pd.concat([tf_reversal, pathway_reversal], ignore_index=True)
    rank = (
        permutation.loc[permutation["is_observed"], [
            "analysis_level", "population", "activity_type", "program",
            "observed_rank_high_is_more_opposed", "empirical_p",
        ]]
        if not permutation.empty
        else pd.DataFrame()
    )
    if not rank.empty:
        candidates = candidates.merge(
            rank,
            on=["analysis_level", "population", "activity_type", "program"],
            how="left",
        )
    candidates = candidates.loc[
        candidates["directionally_reversed"] & candidates["closer_to_young_after_treatment"]
    ].sort_values(
        ["analysis_level", "population", "activity_type", "opposition_magnitude"],
        ascending=[True, True, True, False],
    )
    candidates.to_csv(output_root / "candidate_regulators.tsv", sep="\t", index=False)
    _write_report(output_root, tf_reversal, pathway_reversal, permutation, provenance)

    outputs = [p for p in output_root.rglob("*") if p.is_file() and p.name not in {"manifest.tsv", "COMPLETE.json"}]
    manifest = pd.DataFrame(
        [{"path": str(p.relative_to(paths["root"])), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in outputs]
    )
    manifest.to_csv(output_root / "manifest.tsv", sep="\t", index=False)
    complete = {
        "stage": 7,
        "status": "COMPLETE",
        "seed": int(config.get("project", {}).get("random_seed", 20260810)),
        "statistical_unit": "library",
        "n_libraries_per_group": 3,
        "software": {
            "decoupler": version("decoupler"),
            "pandas": version("pandas"),
            "numpy": version("numpy"),
        },
        "manifest_sha256": sha256_file(output_root / "manifest.tsv"),
    }
    (output_root / "COMPLETE.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print("STAGE7_REGULATORY_ACTIVITY_COMPLETE")
