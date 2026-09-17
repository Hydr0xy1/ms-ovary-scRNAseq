"""Stage 4: library-level composition and cell-intrinsic decomposition."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .de_stage1_5 import ALL_LIBRARIES
from .project import project_paths, require_compute_resources, setup_logging

GROUPS = ("Y", "OC", "OT")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clr_transform_counts(
    table: pd.DataFrame,
    *,
    pseudocount: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Close non-negative component counts and return proportions plus CLR."""

    if pseudocount <= 0:
        raise ValueError("CLR pseudocount must be positive")
    values = table.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Composition counts must be finite and non-negative")
    adjusted = values + pseudocount
    proportions = adjusted / adjusted.sum(axis=1, keepdims=True)
    log_values = np.log(proportions)
    clr = log_values - log_values.mean(axis=1, keepdims=True)
    return (
        pd.DataFrame(proportions, index=table.index, columns=table.columns),
        pd.DataFrame(clr, index=table.index, columns=table.columns),
    )


def shapley_two_factor_decomposition(
    p_a: np.ndarray,
    p_b: np.ndarray,
    mu_a: np.ndarray,
    mu_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact symmetric decomposition of f(p, mu)=p@mu into two factor effects."""

    f_aa = p_a @ mu_a
    f_ba = p_b @ mu_a
    f_ab = p_a @ mu_b
    f_bb = p_b @ mu_b
    composition = 0.5 * ((f_ba - f_aa) + (f_bb - f_ab))
    intrinsic = 0.5 * ((f_ab - f_aa) + (f_bb - f_ba))
    total = f_bb - f_aa
    if not np.allclose(composition + intrinsic, total, rtol=1e-10, atol=1e-10):
        raise AssertionError("Shapley decomposition does not reconstruct total change")
    return composition, intrinsic, total


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else float("nan")


def _projection_fraction(component: np.ndarray, total: np.ndarray) -> float:
    denominator = float(np.dot(total, total))
    return float(np.dot(component, total) / denominator) if denominator > 0 else float("nan")


def _composition_tables(
    h5ad_path: Path,
    populations: list[str],
    analysis_sets: list[str],
    pseudocount: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs = adata.obs
        required = {
            "cell_type_broad_v2",
            "cell_type_subtype_v2",
            "analysis_tier_v2",
            "library_id",
            "group",
        }
        missing = required - set(obs.columns)
        if missing:
            raise KeyError(f"Composition input lacks columns: {sorted(missing)}")
        rows: list[dict[str, Any]] = []
        clr_rows: list[dict[str, Any]] = []
        tier = obs["analysis_tier_v2"].astype(str)
        for broad in populations:
            broad_mask = obs["cell_type_broad_v2"].astype(str).eq(broad)
            for analysis_set in analysis_sets:
                if analysis_set == "tier1_only":
                    eligible = tier.eq("Tier1_primary")
                elif analysis_set == "tier1_plus_tier2":
                    eligible = tier.isin(["Tier1_primary", "Tier2_sensitivity"])
                else:
                    raise ValueError(f"Unknown composition set: {analysis_set}")
                sub = obs.loc[broad_mask & eligible]
                matrix = pd.crosstab(
                    sub["library_id"].astype(str),
                    sub["cell_type_subtype_v2"].astype(str),
                ).reindex(ALL_LIBRARIES, fill_value=0)
                proportions, clr = clr_transform_counts(matrix, pseudocount=pseudocount)
                for library in ALL_LIBRARIES:
                    group = library.split("_", 1)[0]
                    total = int(matrix.loc[library].sum())
                    for subtype in matrix.columns:
                        rows.append(
                            {
                                "broad_population": broad,
                                "analysis_set": analysis_set,
                                "library_id": library,
                                "group": group,
                                "subtype": subtype,
                                "n_cells": int(matrix.loc[library, subtype]),
                                "total_cells_in_broad": total,
                                "raw_proportion": (
                                    float(matrix.loc[library, subtype] / total)
                                    if total > 0
                                    else np.nan
                                ),
                                "pseudocount_closed_proportion": float(
                                    proportions.loc[library, subtype]
                                ),
                            }
                        )
                        clr_rows.append(
                            {
                                "broad_population": broad,
                                "analysis_set": analysis_set,
                                "library_id": library,
                                "group": group,
                                "subtype": subtype,
                                "clr": float(clr.loc[library, subtype]),
                            }
                        )
        return pd.DataFrame(rows), pd.DataFrame(clr_rows)
    finally:
        adata.file.close()


def _aitchison_distances(clr_long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (broad, analysis_set), sub in clr_long.groupby(
        ["broad_population", "analysis_set"], observed=True, sort=True
    ):
        matrix = sub.pivot(index="library_id", columns="subtype", values="clr").reindex(
            ALL_LIBRARIES
        )
        for library in ALL_LIBRARIES:
            if library.startswith("Y_"):
                ref = matrix.loc[[x for x in ALL_LIBRARIES if x.startswith("Y_") and x != library]]
                reference = "leave_one_young_out_centroid"
            else:
                ref = matrix.loc[[x for x in ALL_LIBRARIES if x.startswith("Y_")]]
                reference = "all_young_centroid"
            centroid = ref.mean(axis=0).to_numpy(dtype=float)
            distance = float(np.linalg.norm(matrix.loc[library].to_numpy(dtype=float) - centroid))
            rows.append(
                {
                    "broad_population": broad,
                    "analysis_set": analysis_set,
                    "library_id": library,
                    "group": library.split("_", 1)[0],
                    "reference": reference,
                    "aitchison_distance_to_young": distance,
                }
            )
    result = pd.DataFrame(rows)
    group_summary = (
        result.groupby(["broad_population", "analysis_set", "group"], observed=True)[
            "aitchison_distance_to_young"
        ]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )
    group_summary.columns = [
        "broad_population",
        "analysis_set",
        "group",
        "group_mean",
        "group_median",
        "group_min",
        "group_max",
    ]
    return result.merge(
        group_summary,
        on=["broad_population", "analysis_set", "group"],
        how="left",
    )


def _composition_effects(clr_long: pd.DataFrame, proportions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (broad, analysis_set), sub in clr_long.groupby(
        ["broad_population", "analysis_set"], observed=True, sort=True
    ):
        matrix = sub.pivot(index="library_id", columns="subtype", values="clr").reindex(
            ALL_LIBRARIES
        )
        centroids = {
            group: matrix.loc[[x for x in ALL_LIBRARIES if x.startswith(f"{group}_")]].mean()
            for group in GROUPS
        }
        aging = (centroids["OC"] - centroids["Y"]).to_numpy(dtype=float)
        treatment = (centroids["OT"] - centroids["OC"]).to_numpy(dtype=float)
        residual = (centroids["OT"] - centroids["Y"]).to_numpy(dtype=float)
        rows.append(
            {
                "broad_population": broad,
                "analysis_set": analysis_set,
                "component": "whole_composition_vector",
                "Y_mean_proportion": np.nan,
                "OC_mean_proportion": np.nan,
                "OT_mean_proportion": np.nan,
                "aging_delta": np.nan,
                "treatment_delta": np.nan,
                "rho_aging_treatment": float(
                    pd.Series(aging).corr(pd.Series(treatment), method="spearman")
                ),
                "cosine_aging_treatment": _cosine(aging, treatment),
                "aging_norm": float(np.linalg.norm(aging)),
                "residual_norm": float(np.linalg.norm(residual)),
                "residual_norm_ratio": (
                    float(np.linalg.norm(residual) / np.linalg.norm(aging))
                    if np.linalg.norm(aging) > 0
                    else np.nan
                ),
            }
        )
        psub = proportions[
            proportions["broad_population"].eq(broad)
            & proportions["analysis_set"].eq(analysis_set)
        ]
        means = psub.groupby(["subtype", "group"], observed=True)["raw_proportion"].mean()
        for subtype in sorted(psub["subtype"].unique()):
            y = float(means.get((subtype, "Y"), np.nan))
            oc = float(means.get((subtype, "OC"), np.nan))
            ot = float(means.get((subtype, "OT"), np.nan))
            rows.append(
                {
                    "broad_population": broad,
                    "analysis_set": analysis_set,
                    "component": subtype,
                    "Y_mean_proportion": y,
                    "OC_mean_proportion": oc,
                    "OT_mean_proportion": ot,
                    "aging_delta": oc - y,
                    "treatment_delta": ot - oc,
                    "rho_aging_treatment": np.nan,
                    "cosine_aging_treatment": np.nan,
                    "aging_norm": np.nan,
                    "residual_norm": np.nan,
                    "residual_norm_ratio": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _intrinsic_summary(reversal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (broad, subtype), sub in reversal.groupby(
        ["broad_population", "subtype"], observed=True, sort=True
    ):
        finite = sub[["aging_effect", "treatment_effect", "residual_effect"]].apply(
            pd.to_numeric, errors="coerce"
        )
        finite = finite[np.isfinite(finite).all(axis=1)]
        aging = finite["aging_effect"].to_numpy()
        treatment = finite["treatment_effect"].to_numpy()
        residual = finite["residual_effect"].to_numpy()
        rows.append(
            {
                "broad_population": broad,
                "subtype": subtype,
                "n_genes": len(finite),
                "spearman_aging_treatment": float(
                    pd.Series(aging).corr(pd.Series(treatment), method="spearman")
                ),
                "cosine_aging_treatment": _cosine(aging, treatment),
                "aging_norm": float(np.linalg.norm(aging)),
                "residual_norm": float(np.linalg.norm(residual)),
                "residual_norm_ratio": (
                    float(np.linalg.norm(residual) / np.linalg.norm(aging))
                    if np.linalg.norm(aging) > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _counterfactual_decomposition(
    counts: pd.DataFrame,
    coverage: pd.DataFrame,
    eligibility: pd.DataFrame,
    *,
    target_sum: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    recon_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    eligible = eligibility[
        eligibility["eligibility"].isin(["Primary_DE_ready", "Sensitivity_only"])
    ]
    for broad, broad_elig in eligible.groupby("broad_population", observed=True, sort=True):
        subtypes = sorted(broad_elig["subtype"].astype(str).tolist())
        if len(subtypes) < 2:
            continue
        block = counts.loc[broad]
        genes = block.columns.astype(str)
        mu_by_group: dict[str, np.ndarray] = {}
        p_by_group: dict[str, np.ndarray] = {}
        for group in GROUPS:
            libraries = [x for x in ALL_LIBRARIES if x.startswith(f"{group}_")]
            mu_parts = []
            for subtype in subtypes:
                lib_counts = block.loc[subtype].reindex(libraries).to_numpy(dtype=float)
                lib_totals = lib_counts.sum(axis=1, keepdims=True)
                normalized = lib_counts / np.maximum(lib_totals, 1.0) * target_sum
                mu_parts.append(normalized.mean(axis=0))
            mu_by_group[group] = np.vstack(mu_parts)
            cov = coverage[
                coverage["broad_population"].eq(broad)
                & coverage["subtype"].isin(subtypes)
                & coverage["library_id"].isin(libraries)
            ]
            composition = cov.pivot(
                index="library_id", columns="subtype", values="n_cells_tier1"
            ).reindex(index=libraries, columns=subtypes, fill_value=0)
            lib_prop = composition.div(composition.sum(axis=1).replace(0, np.nan), axis=0)
            centroid = lib_prop.mean(axis=0).fillna(0).to_numpy(dtype=float)
            p_by_group[group] = centroid / centroid.sum()

        scenarios: dict[str, np.ndarray] = {}
        for p_group in GROUPS:
            for mu_group in GROUPS:
                scenarios[f"p_{p_group}__mu_{mu_group}"] = (
                    p_by_group[p_group] @ mu_by_group[mu_group]
                )
        for scenario, vector in scenarios.items():
            recon_rows.append(
                pd.DataFrame(
                    {
                        "broad_population": broad,
                        "scenario": scenario,
                        "gene": genes,
                        "mixture_expression_cpm": vector,
                    }
                )
            )
        for name, group_a, group_b in (("OC_vs_Y", "Y", "OC"), ("OT_vs_OC", "OC", "OT")):
            comp, intrinsic, total = shapley_two_factor_decomposition(
                p_by_group[group_a],
                p_by_group[group_b],
                mu_by_group[group_a],
                mu_by_group[group_b],
            )
            for source, vector in (
                ("composition", comp),
                ("cell_intrinsic", intrinsic),
                ("total", total),
            ):
                summary_rows.append(
                    {
                        "broad_population": broad,
                        "contrast": name,
                        "effect_source": source,
                        "vector_norm": float(np.linalg.norm(vector)),
                        "projection_fraction_of_total": (
                            1.0 if source == "total" else _projection_fraction(vector, total)
                        ),
                        "cosine_with_total": 1.0 if source == "total" else _cosine(vector, total),
                        "n_subtypes": len(subtypes),
                        "subtypes": ";".join(subtypes),
                        "decomposition_method": "two_factor_shapley_exact_in_linear_CPM_space",
                    }
                )
    return pd.concat(recon_rows, ignore_index=True), pd.DataFrame(summary_rows)


def _figures(
    composition: pd.DataFrame,
    distances: pd.DataFrame,
    figure_root: Path,
    primary_set: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root.mkdir(parents=True, exist_ok=True)
    primary = composition[composition["analysis_set"].eq(primary_set)]
    for broad, sub in primary.groupby("broad_population", observed=True):
        matrix = sub.pivot(index="library_id", columns="subtype", values="raw_proportion").reindex(
            ALL_LIBRARIES
        )
        ax = matrix.plot.bar(stacked=True, figsize=(10, 5), width=0.85, colormap="tab20")
        ax.set_ylabel("Subtype proportion")
        ax.set_title(f"{broad}: library-level composition")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
        ax.figure.tight_layout()
        ax.figure.savefig(figure_root / f"composition__{broad}.png", dpi=180)
        plt.close(ax.figure)
    d = distances[distances["analysis_set"].eq(primary_set)]
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(d))
    colors = d["group"].map({"Y": "#4C78A8", "OC": "#E45756", "OT": "#59A14F"})
    ax.bar(positions, d["aitchison_distance_to_young"], color=colors)
    ax.set_xticks(positions, d["library_id"], rotation=45, ha="right")
    ax.set_ylabel("Aitchison distance to young reference")
    ax.set_title("Composition distance (descriptive, library-level)")
    fig.tight_layout()
    fig.savefig(figure_root / "aitchison_distance_to_young.png", dpi=180)
    plt.close(fig)


def _report(
    output_root: Path,
    composition_effect: pd.DataFrame,
    distances: pd.DataFrame,
    decomposition: pd.DataFrame,
    primary_set: str,
) -> None:
    vector = composition_effect[
        composition_effect["analysis_set"].eq(primary_set)
        & composition_effect["component"].eq("whole_composition_vector")
    ]
    dgroup = (
        distances[distances["analysis_set"].eq(primary_set)]
        .groupby(["broad_population", "group"], observed=True)[
            "aitchison_distance_to_young"
        ]
        .mean()
    )
    lines = [
        "# Stage 4：细胞组成与亚型内表达分解",
        "",
        "## 1. 为什么做这一步",
        "",
        (
            "Broad population信号可能来自亚型比例变化，也可能来自相同亚型内部的表达"
            "变化。本阶段把两者并列量化，避免把细胞组成变化误称为细胞内调控。"
        ),
        "",
        "## 2. 输入与统计单位",
        "",
        (
            "细胞组成由annotation v2按library计数；亚型内表达来自Stage 3的raw-count "
            "pseudobulk DE。所有正式比较以library为单位，n=3/group。"
        ),
        "",
        "## 3. 方法",
        "",
        (
            "组成采用原始比例及加0.5细胞伪计数后的CLR变换。Aitchison距离在CLR空间"
            "计算；Y样本使用leave-one-young-out参考，OC/OT使用全部Y的centroid。反事实"
            "表达用E=sum(p_k × mu_k)，并在linear CPM空间采用双因素Shapley分解，保证"
            "composition与intrinsic分量逐基因精确重构总变化。"
        ),
        "",
        "## 4. 描述性组成结果",
        "",
    ]
    for row in vector.itertuples(index=False):
        lines.append(
            f"- {row.broad_population}: aging-treatment CLR向量cosine="
            f"{row.cosine_aging_treatment:.3f}，residual/aging norm="
            f"{row.residual_norm_ratio:.3f}。"
        )
        for group in GROUPS:
            if (row.broad_population, group) in dgroup.index:
                lines.append(
                    f"  - {group}到young reference的平均Aitchison距离="
                    f"{dgroup.loc[(row.broad_population, group)]:.3f}。"
                )
    lines.extend(["", "## 5. Model-based decomposition", ""])
    for (broad, contrast), sub in decomposition.groupby(
        ["broad_population", "contrast"], observed=True
    ):
        comp = sub[sub["effect_source"].eq("composition")].iloc[0]
        intrinsic = sub[sub["effect_source"].eq("cell_intrinsic")].iloc[0]
        lines.append(
            f"- {broad} / {contrast}: composition沿总变化投影="
            f"{comp['projection_fraction_of_total']:.3f}，"
            f"intrinsic沿总变化投影={intrinsic['projection_fraction_of_total']:.3f}。"
        )
    lines.extend(
        [
            "",
            "## 6. 可以说明什么",
            "",
            (
                "结果可描述broad signal在数学上更接近composition remodeling、"
                "within-subtype expression change或两者共同作用。"
            ),
            "",
            "## 7. 不能说明什么",
            "",
            "该分解是基于捕获细胞和归一化表达的模型，不证明组成变化或表达变化具有因果优先级，也不能排除解离/捕获偏倚。投影分量可能为负或大于1，不能机械解释为生物学百分比。",
            "",
            "## 8. 局限",
            "",
            "每组仅3个library；组成分析以效应量和描述为主，不进行cell-level显著性检验。低复杂度和未解析成分保留在组成敏感性表中，但不进入正式反事实表达重构。",
            "",
            "## 9. 下一步",
            "",
            (
                "Stage 5将在library-level多变量表达空间中使用交叉验证的aging axis，"
                "检验OT是否描述性地向young reference移动。"
            ),
        ]
    )
    (output_root / "COMPOSITION_INTRINSIC_REPORT_CN.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _manifest(output_root: Path) -> pd.DataFrame:
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
    result = pd.DataFrame(rows)
    result.to_csv(output_root / "output_manifest.tsv", sep="\t", index=False)
    return result


def run_stage4(config: Mapping[str, Any], *, allow_low_memory: bool = False) -> Path:
    require_compute_resources(dict(config), allow_low_memory=allow_low_memory)
    paths = project_paths(dict(config))
    settings = config["stage4_composition_decomposition"]
    logger = setup_logging("13_stage4_composition_decomposition", dict(config))
    stage3_root = paths["root"] / settings["stage3_dir"]
    if not (stage3_root / "COMPLETE.json").exists():
        raise RuntimeError("Stage 3 completion marker is missing")
    output_root = paths["root"] / settings["output_dir"]
    figure_root = paths["root"] / settings["figure_dir"]
    output_root.mkdir(parents=True, exist_ok=True)

    composition, clr = _composition_tables(
        paths["root"] / settings["input_object"],
        list(settings["broad_populations"]),
        list(settings["composition_analysis_sets"]),
        float(settings["clr_pseudocount_cells"]),
    )
    distances = _aitchison_distances(clr)
    composition_effect = _composition_effects(clr, composition)
    reversal = pd.read_csv(stage3_root / "subtype_reversal.tsv", sep="\t")
    intrinsic = _intrinsic_summary(reversal)
    counts = pd.read_csv(
        stage3_root / "subtype_pseudobulk_counts.tsv.gz",
        sep="\t",
        index_col=[0, 1, 2],
    )
    counts.index.names = ["broad_population", "subtype", "library_id"]
    coverage = pd.read_csv(stage3_root / "subtype_coverage.tsv", sep="\t")
    eligibility = pd.read_csv(stage3_root / "subtype_eligibility.tsv", sep="\t")
    reconstruction, decomposition = _counterfactual_decomposition(
        counts,
        coverage,
        eligibility,
        target_sum=float(settings["expression_target_sum"]),
    )

    composition.to_csv(output_root / "composition_by_library.tsv", sep="\t", index=False)
    clr.to_csv(output_root / "clr_composition.tsv", sep="\t", index=False)
    distances.to_csv(output_root / "aitchison_distance.tsv", sep="\t", index=False)
    composition_effect.to_csv(output_root / "composition_effect.tsv", sep="\t", index=False)
    intrinsic.to_csv(output_root / "intrinsic_effect.tsv", sep="\t", index=False)
    reconstruction.to_csv(
        output_root / "counterfactual_reconstruction.tsv", sep="\t", index=False
    )
    decomposition.to_csv(output_root / "decomposition_summary.tsv", sep="\t", index=False)
    _figures(
        composition,
        distances,
        figure_root,
        str(settings["primary_composition_set"]),
    )
    _report(
        output_root,
        composition_effect,
        distances,
        decomposition,
        str(settings["primary_composition_set"]),
    )
    complete = {
        "status": "complete",
        "stage3_manifest_sha256": sha256_file(stage3_root / "output_manifest.tsv"),
        "random_seed": int(settings["random_seed"]),
        "software": {name: version(name) for name in ("anndata", "pandas", "numpy")},
        "analysis_sets": list(settings["composition_analysis_sets"]),
        "primary_analysis_set": str(settings["primary_composition_set"]),
    }
    (output_root / "COMPLETE.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _manifest(output_root)
    logger.info("STAGE4_COMPOSITION_INTRINSIC_COMPLETE: %s", output_root)
    print("STAGE4_COMPOSITION_INTRINSIC_COMPLETE")
    return output_root
