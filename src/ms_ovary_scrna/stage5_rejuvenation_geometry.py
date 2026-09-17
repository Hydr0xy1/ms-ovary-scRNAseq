"""Stage 5: cross-validated library-level aging-axis geometry."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .de_stage1 import prefilter_genes
from .de_stage1_5 import ALL_LIBRARIES
from .de_stage1_6 import enumerate_permutation_assignments
from .project import project_paths, require_compute_resources, setup_logging


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log_cpm(counts: pd.DataFrame, target: float = 1e6) -> pd.DataFrame:
    values = counts.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Counts must be finite and non-negative")
    library_sizes = values.sum(axis=1, keepdims=True)
    transformed = np.log1p(values / np.maximum(library_sizes, 1.0) * target)
    return pd.DataFrame(transformed, index=counts.index, columns=counts.columns)


def choose_variable_genes(expression: pd.DataFrame, n_genes: int) -> list[str]:
    variances = expression.var(axis=0, ddof=1).replace([np.inf, -np.inf], np.nan).fillna(0)
    order = pd.DataFrame(
        {"gene": variances.index.astype(str), "variance": variances.to_numpy()}
    ).sort_values(["variance", "gene"], ascending=[False, True], kind="stable")
    return order.head(min(n_genes, len(order)))["gene"].tolist()


def projection_onto_axis(
    sample: np.ndarray,
    young_centroid: np.ndarray,
    aged_centroid: np.ndarray,
) -> float:
    axis = aged_centroid - young_centroid
    denominator = float(np.dot(axis, axis))
    return (
        float(np.dot(sample - young_centroid, axis) / denominator)
        if denominator > 0
        else float("nan")
    )


def cross_validated_aging_projection(
    expression: pd.DataFrame,
    *,
    n_variable_genes: int,
) -> pd.DataFrame:
    """Project libraries on axes that never use the held-out Y/OC library."""

    rows: list[dict[str, Any]] = []
    group = pd.Series({library: library.split("_", 1)[0] for library in expression.index})
    for library in expression.index:
        label = group.loc[library]
        if label in {"Y", "OC"}:
            train = expression.drop(index=library)
            scheme = "leave_one_library_out"
        else:
            train = expression.loc[group[group.isin(["Y", "OC"])].index]
            scheme = "OT_not_used_to_define_axis"
        genes = choose_variable_genes(train, n_variable_genes)
        y_train = train.loc[[x for x in train.index if x.startswith("Y_")], genes]
        oc_train = train.loc[[x for x in train.index if x.startswith("OC_")], genes]
        young = y_train.mean(axis=0).to_numpy(dtype=float)
        aged = oc_train.mean(axis=0).to_numpy(dtype=float)
        value = expression.loc[library, genes].to_numpy(dtype=float)
        rows.append(
            {
                "library_id": library,
                "group": label,
                "validation_scheme": scheme,
                "n_training_Y": len(y_train),
                "n_training_OC": len(oc_train),
                "n_genes": len(genes),
                "aging_axis_projection": projection_onto_axis(value, young, aged),
            }
        )
    return pd.DataFrame(rows)


def leave_pair_out_ot_projection(
    expression: pd.DataFrame,
    *,
    n_variable_genes: int,
) -> pd.DataFrame:
    """Sensitivity: define nine Y/OC axes, each leaving one Y and one OC out."""

    rows: list[dict[str, Any]] = []
    young = [x for x in expression.index if x.startswith("Y_")]
    aged = [x for x in expression.index if x.startswith("OC_")]
    treated = [x for x in expression.index if x.startswith("OT_")]
    fold = 0
    for held_y in young:
        for held_oc in aged:
            fold += 1
            training = [x for x in young if x != held_y] + [x for x in aged if x != held_oc]
            train = expression.loc[training]
            genes = choose_variable_genes(train, n_variable_genes)
            y_centroid = expression.loc[[x for x in young if x != held_y], genes].mean().to_numpy()
            oc_centroid = expression.loc[[x for x in aged if x != held_oc], genes].mean().to_numpy()
            for library in treated:
                rows.append(
                    {
                        "fold": f"F{fold:02d}",
                        "held_out_Y": held_y,
                        "held_out_OC": held_oc,
                        "library_id": library,
                        "group": "OT",
                        "n_genes": len(genes),
                        "aging_axis_projection": projection_onto_axis(
                            expression.loc[library, genes].to_numpy(),
                            y_centroid,
                            oc_centroid,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _pca_and_distance(
    expression: pd.DataFrame,
    *,
    n_variable_genes: int,
    n_pcs: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    genes = choose_variable_genes(expression, n_variable_genes)
    n_components = min(n_pcs, len(expression) - 1, len(genes))
    model = PCA(n_components=n_components, random_state=random_seed)
    scores = model.fit_transform(expression[genes])
    columns = [f"PC{i + 1}" for i in range(n_components)]
    score = pd.DataFrame(scores, index=expression.index, columns=columns)
    score["library_id"] = score.index
    score["group"] = [x.split("_", 1)[0] for x in score.index]
    score["explained_variance_ratio"] = ";".join(
        f"{x:.8g}" for x in model.explained_variance_ratio_
    )
    distance_rows: list[dict[str, Any]] = []
    for library in expression.index:
        if library.startswith("Y_"):
            reference_ids = [x for x in expression.index if x.startswith("Y_") and x != library]
            reference = "leave_one_young_out_centroid"
        else:
            reference_ids = [x for x in expression.index if x.startswith("Y_")]
            reference = "all_young_centroid"
        centroid = score.loc[reference_ids, columns].mean(axis=0).to_numpy()
        euclidean = float(np.linalg.norm(score.loc[library, columns].to_numpy() - centroid))
        distance_rows.append(
            {
                "library_id": library,
                "group": library.split("_", 1)[0],
                "reference": reference,
                "euclidean_distance_to_young_in_PC_space": euclidean,
                "mahalanobis_distance": np.nan,
                "mahalanobis_status": (
                    "skipped_unstable_covariance_only_3_young_libraries"
                ),
                "n_pcs": n_components,
            }
        )
    return score.reset_index(drop=True), pd.DataFrame(distance_rows)


def exact_permutation_geometry(
    expression: pd.DataFrame,
    *,
    n_variable_genes: int,
) -> pd.DataFrame:
    assignments = enumerate_permutation_assignments()
    rows: list[dict[str, Any]] = []
    for permutation_id, assignment in assignments.groupby("permutation_id", observed=True):
        labels = assignment.set_index("library_id")["model_group"]
        training_ids = labels[labels.isin(["Y", "OC"])].index.tolist()
        genes = choose_variable_genes(expression.loc[training_ids], n_variable_genes)
        y_ids = labels[labels.eq("Y")].index.tolist()
        oc_ids = labels[labels.eq("OC")].index.tolist()
        ot_ids = labels[labels.eq("OT")].index.tolist()
        y_centroid = expression.loc[y_ids, genes].mean().to_numpy()
        oc_centroid = expression.loc[oc_ids, genes].mean().to_numpy()
        oc_projection = [
            projection_onto_axis(expression.loc[x, genes].to_numpy(), y_centroid, oc_centroid)
            for x in oc_ids
        ]
        ot_projection = [
            projection_onto_axis(expression.loc[x, genes].to_numpy(), y_centroid, oc_centroid)
            for x in ot_ids
        ]
        rows.append(
            {
                "permutation_id": permutation_id,
                "is_observed": bool(assignment["is_observed"].iloc[0]),
                "mean_OC_projection": float(np.mean(oc_projection)),
                "mean_OT_projection": float(np.mean(ot_projection)),
                "OT_minus_OC_projection": float(np.mean(ot_projection) - np.mean(oc_projection)),
                "n_genes": len(genes),
            }
        )
    result = pd.DataFrame(rows)
    observed = float(result.loc[result["is_observed"], "OT_minus_OC_projection"].iloc[0])
    result["observed_rank_low_is_more_young_directed"] = int(
        1 + (result.loc[~result["is_observed"], "OT_minus_OC_projection"] <= observed).sum()
    )
    result["empirical_p"] = result["observed_rank_low_is_more_young_directed"] / len(result)
    return result


def _figures(
    projections: pd.DataFrame,
    pca_scores: pd.DataFrame,
    distances: pd.DataFrame,
    figure_root: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root.mkdir(parents=True, exist_ok=True)
    colors = {"Y": "#4C78A8", "OC": "#E45756", "OT": "#59A14F"}
    for population, sub in projections.groupby("population", observed=True):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(sub))
        ax.scatter(
            x,
            sub["aging_axis_projection"],
            c=sub["group"].map(colors),
            s=70,
        )
        ax.axhline(0, color="#4C78A8", linestyle="--", linewidth=1)
        ax.axhline(1, color="#E45756", linestyle="--", linewidth=1)
        ax.set_xticks(x, sub["library_id"], rotation=45, ha="right")
        ax.set_ylabel("Projection on aging axis (Y=0, OC=1 reference)")
        ax.set_title(f"{population}: cross-validated aging-axis projection")
        fig.tight_layout()
        fig.savefig(figure_root / f"aging_axis_projection__{population}.png", dpi=180)
        plt.close(fig)
    for population, sub in pca_scores.groupby("population", observed=True):
        fig, ax = plt.subplots(figsize=(6, 5))
        for group, group_sub in sub.groupby("group", observed=True):
            ax.scatter(group_sub["PC1"], group_sub["PC2"], label=group, color=colors[group], s=70)
            for row in group_sub.itertuples(index=False):
                ax.text(row.PC1, row.PC2, row.library_id, fontsize=8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{population}: sample-level pseudobulk PCA")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_root / f"sample_pca__{population}.png", dpi=180)
        plt.close(fig)
    primary = distances[distances["population"].isin(["Granulosa", "Stromal_fibroblast"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(primary))
    ax.bar(
        x,
        primary["euclidean_distance_to_young_in_PC_space"],
        color=primary["group"].map(colors),
    )
    ax.set_xticks(
        x,
        primary["population"] + ":" + primary["library_id"],
        rotation=55,
        ha="right",
    )
    ax.set_ylabel("Euclidean distance to young in PC space")
    fig.tight_layout()
    fig.savefig(figure_root / "distance_to_young_primary.png", dpi=180)
    plt.close(fig)


def _report(
    output_root: Path,
    projection: pd.DataFrame,
    distances: pd.DataFrame,
    permutation: pd.DataFrame,
    primary: list[str],
) -> None:
    lines = [
        "# Stage 5：衰老轴与young-directed transcriptomic projection",
        "",
        "## 1. 为什么做这一步",
        "",
        (
            "Gene-by-gene反向变化不一定意味着整个样本状态向年轻参考移动。"
            "本阶段只在library-level pseudobulk空间评估多变量几何。"
        ),
        "",
        "## 2. 输入与统计单位",
        "",
        (
            "输入为既有broad Tier1 raw-count pseudobulk。经固定的log-CPM变换后分析；"
            "统计/展示单位是9个library，而不是105,763个细胞。"
        ),
        "",
        "## 3. 防止循环论证",
        "",
        (
            "每个Y或OC library的投影使用不包含该library的aging axis。OT从不参与aging "
            "axis定义；另以9个leave-one-Y-and-one-OC-out轴检查OT稳定性。变量基因在"
            "每个训练折内重新选择。"
        ),
        "",
        "## 4. 主要结果",
        "",
    ]
    for population in primary:
        sub = projection[projection["population"].eq(population)]
        means = sub.groupby("group", observed=True)["aging_axis_projection"].mean()
        perm = permutation[
            permutation["population"].eq(population) & permutation["is_observed"]
        ].iloc[0]
        distance_means = (
            distances[distances["population"].eq(population)]
            .groupby("group", observed=True)["euclidean_distance_to_young_in_PC_space"]
            .mean()
        )
        lines.append(
            f"- {population}: mean projection Y={means.get('Y', np.nan):.3f}, "
            f"OC={means.get('OC', np.nan):.3f}, OT={means.get('OT', np.nan):.3f}; "
            "observed exact-permutation rank="
            f"{int(perm['observed_rank_low_is_more_young_directed'])}/20。"
        )
        lines.append(
            f"  - mean PC-distance to young: OC={distance_means.get('OC', np.nan):.3f}, "
            f"OT={distance_means.get('OT', np.nan):.3f}。"
        )
    lines.extend(
        [
            "",
            "## 5. 可以说明什么",
            "",
            (
                "若OT投影或距离比OC更靠近Y，这支持‘MRJP1相关转录状态向年轻参考方向"
                "移动’这一sample-level描述。exact permutation用于判断真实标签是否比其余"
                "19种3-vs-3标签分配更极端。"
            ),
            "",
            "## 6. 不能说明什么",
            "",
            "投影值不是‘年轻化百分比’，不能等同于卵巢功能恢复或生物年龄逆转。轴和距离依赖当前基因空间、归一化和样本集合。",
            "",
            "## 7. 局限",
            "",
            (
                "每组仅3个library。只有3个young library，young协方差不可稳定估计，"
                "因此预先放弃Mahalanobis距离，而没有通过不可靠的正则化制造额外结果。"
                "经验p最低为0.05。"
            ),
            "",
            "## 8. 下一步",
            "",
            (
                "下一步尝试用独立公开卵巢衰老数据验证internal aging signature，"
                "若无法获得可靠processed matrix与sample metadata则明确跳过。"
            ),
        ]
    )
    (output_root / "AGING_GEOMETRY_REPORT_CN.md").write_text(
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


def run_stage5(config: Mapping[str, Any], *, allow_low_memory: bool = False) -> Path:
    require_compute_resources(dict(config), allow_low_memory=allow_low_memory)
    paths = project_paths(dict(config))
    settings = config["stage5_rejuvenation_geometry"]
    logger = setup_logging("14_stage5_rejuvenation_geometry", dict(config))
    stage4 = paths["results"] / "stage4_composition_decomposition" / "COMPLETE.json"
    if not stage4.exists():
        raise RuntimeError("Stage 4 completion marker is missing")
    counts_path = paths["root"] / settings["broad_counts"]
    counts = pd.read_csv(counts_path, sep="\t", index_col=[0, 1])
    counts.index.names = ["population", "library_id"]
    populations = list(settings["primary_populations"]) + list(
        settings["supplementary_populations"]
    )
    output_root = paths["root"] / settings["output_dir"]
    figure_root = paths["root"] / settings["figure_dir"]
    output_root.mkdir(parents=True, exist_ok=True)
    projection_frames: list[pd.DataFrame] = []
    leave_pair_frames: list[pd.DataFrame] = []
    pca_frames: list[pd.DataFrame] = []
    distance_frames: list[pd.DataFrame] = []
    permutation_frames: list[pd.DataFrame] = []
    for population in populations:
        logger.info("Aging geometry: %s", population)
        pop_counts = counts.loc[population].reindex(ALL_LIBRARIES)
        filtered = prefilter_genes(
            pop_counts,
            min_count=int(settings["min_gene_count"]),
            min_samples=int(settings["min_gene_samples"]),
        )
        expression = log_cpm(filtered, float(settings["normalization_target"]))
        projection = cross_validated_aging_projection(
            expression, n_variable_genes=int(settings["n_variable_genes"])
        )
        projection.insert(0, "population", population)
        projection_frames.append(projection)
        leave_pair = leave_pair_out_ot_projection(
            expression, n_variable_genes=int(settings["n_variable_genes"])
        )
        leave_pair.insert(0, "population", population)
        leave_pair_frames.append(leave_pair)
        pca, distance = _pca_and_distance(
            expression,
            n_variable_genes=int(settings["n_variable_genes"]),
            n_pcs=int(settings["n_pcs"]),
            random_seed=int(settings["random_seed"]),
        )
        pca.insert(0, "population", population)
        distance.insert(0, "population", population)
        pca_frames.append(pca)
        distance_frames.append(distance)
        permutation = exact_permutation_geometry(
            expression, n_variable_genes=int(settings["n_variable_genes"])
        )
        permutation.insert(0, "population", population)
        permutation_frames.append(permutation)
    projection = pd.concat(projection_frames, ignore_index=True)
    leave_pair = pd.concat(leave_pair_frames, ignore_index=True)
    pca = pd.concat(pca_frames, ignore_index=True)
    distance = pd.concat(distance_frames, ignore_index=True)
    permutation = pd.concat(permutation_frames, ignore_index=True)
    projection.to_csv(output_root / "aging_axis_projection.tsv", sep="\t", index=False)
    leave_pair.to_csv(output_root / "leave_one_out_projection.tsv", sep="\t", index=False)
    pca.to_csv(output_root / "sample_level_pca.tsv", sep="\t", index=False)
    distance.to_csv(output_root / "distance_to_young.tsv", sep="\t", index=False)
    permutation.to_csv(output_root / "permutation_geometry.tsv", sep="\t", index=False)
    _figures(projection, pca, distance, figure_root)
    _report(
        output_root,
        projection,
        distance,
        permutation,
        list(settings["primary_populations"]),
    )
    complete = {
        "status": "complete",
        "counts_sha256": sha256_file(counts_path),
        "random_seed": int(settings["random_seed"]),
        "software": {
            name: version(name) for name in ("pandas", "numpy", "scikit-learn")
        },
        "populations": populations,
        "mahalanobis": "skipped_unstable_covariance_only_3_young_libraries",
    }
    (output_root / "COMPLETE.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _manifest(output_root)
    logger.info("STAGE5_REJUVENATION_GEOMETRY_COMPLETE: %s", output_root)
    print("STAGE5_REJUVENATION_GEOMETRY_COMPLETE")
    return output_root
