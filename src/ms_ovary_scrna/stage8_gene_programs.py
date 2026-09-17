"""Stage 8: balanced, lineage-specific non-negative matrix factorization."""
# ruff: noqa: E501

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.stats import hypergeom
from sklearn.decomposition import NMF

from .pathway_stage2 import parse_gmt
from .project import project_paths, setup_logging
from .stage7_regulatory_activity import activity_contrasts, sha256_file

POPULATIONS = ("Granulosa", "Stromal_fibroblast")
K_GRID = (5, 8, 10, 12, 15)


def estimate_dense_gib(n_cells: int, n_genes: int, n_copies: int = 4) -> float:
    return n_cells * n_genes * 8 * n_copies / 1024**3


def select_hvgs(
    matrix: sparse.csr_matrix,
    genes: pd.Index,
    n_hvg: int,
) -> tuple[sparse.csr_matrix, list[str], pd.DataFrame]:
    """Select variable genes from sparse log-normalized counts without full densification."""

    counts = matrix.astype(np.float64, copy=True)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    scale = 1e4 / np.maximum(totals, 1.0)
    lognorm = sparse.diags(scale).dot(counts).tocsr()
    lognorm.data = np.log1p(lognorm.data)
    mean = np.asarray(lognorm.mean(axis=0)).ravel()
    second = np.asarray(lognorm.power(2).mean(axis=0)).ravel()
    variance = np.maximum(second - mean**2, 0.0)
    dispersion = variance / np.maximum(mean, 1e-8)
    detected = np.asarray((counts > 0).sum(axis=0)).ravel()
    symbol = genes.astype(str)
    excluded = (
        symbol.str.startswith("mt-")
        | symbol.str.startswith("Mt-")
        | symbol.str.match(r"^Rp[sl][0-9]")
    )
    audit = pd.DataFrame(
        {
            "gene": symbol,
            "mean_log_expression": mean,
            "variance": variance,
            "dispersion": dispersion,
            "n_cells_detected": detected,
            "excluded_mt_or_ribosomal": excluded,
        }
    )
    eligible = audit.loc[
        audit["n_cells_detected"].ge(max(10, int(np.ceil(matrix.shape[0] * 0.01))))
        & ~audit["excluded_mt_or_ribosomal"]
    ].sort_values(["dispersion", "gene"], ascending=[False, True], kind="stable")
    chosen = eligible.head(min(n_hvg, len(eligible)))
    positions = genes.get_indexer(chosen["gene"])
    if (positions < 0).any():
        raise ValueError("Selected HVG not found in expression matrix")
    audit["selected_hvg"] = audit["gene"].isin(chosen["gene"])
    return lognorm[:, positions].tocsr(), chosen["gene"].tolist(), audit


def component_stability(signatures: list[np.ndarray]) -> float:
    """Mean optimally matched cosine similarity across all seed pairs."""

    if len(signatures) < 2:
        return float("nan")
    values: list[float] = []
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            a = signatures[i]
            b = signatures[j]
            a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
            b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
            similarity = a @ b.T
            rows, cols = linear_sum_assignment(-similarity)
            values.extend(similarity[rows, cols].tolist())
    return float(np.mean(values))


def choose_rank(stability: pd.DataFrame, minimum_stability: float = 0.80) -> int:
    """Predefined rank rule: stable elbow, otherwise best stable parsimonious K."""

    table = stability.sort_values("k").copy()
    stable = table.loc[table["mean_component_stability"].ge(minimum_stability)]
    if stable.empty:
        return int(table.sort_values(["mean_component_stability", "k"], ascending=[False, True]).iloc[0]["k"])
    stable_k = set(stable["k"].astype(int))
    for row in table.itertuples(index=False):
        if int(row.k) in stable_k and np.isfinite(row.next_k_relative_error_improvement):
            if float(row.next_k_relative_error_improvement) < 0.05:
                return int(row.k)
    return int(stable.sort_values("k").iloc[-1]["k"])


def _balanced_counts(
    h5ad_path: Path,
    population: str,
    *,
    max_cells_per_library: int,
    seed: int,
) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.Index, pd.DataFrame]:
    import anndata as ad

    source = ad.read_h5ad(h5ad_path, backed="r")
    required = {"library_id", "cell_type_broad_v2", "analysis_tier_v2"}
    missing = required.difference(source.obs.columns)
    if missing:
        raise KeyError(f"Missing required obs columns: {sorted(missing)}")
    obs = source.obs[["library_id", "cell_type_broad_v2", "analysis_tier_v2"]].copy()
    eligible = obs["cell_type_broad_v2"].astype(str).eq(population) & obs["analysis_tier_v2"].astype(str).eq("Tier1")
    selected: list[int] = []
    sampling_rows: list[dict[str, Any]] = []
    libraries = ["Y_1", "Y_2", "Y_3", "OC_1", "OC_2", "OC_3", "OT_1", "OT_2", "OT_3"]
    available = {library: np.flatnonzero(eligible.to_numpy() & obs["library_id"].astype(str).eq(library).to_numpy()) for library in libraries}
    target = min(max_cells_per_library, min(len(x) for x in available.values()))
    if target < 200:
        source.file.close()
        raise ValueError(f"{population} has fewer than 200 Tier1 cells in at least one library")
    for offset, library in enumerate(libraries):
        rng = np.random.default_rng(seed + offset * 1009)
        chosen = np.sort(rng.choice(available[library], size=target, replace=False))
        selected.extend(chosen.tolist())
        sampling_rows.append(
            {"population": population, "library_id": library, "available_cells": len(available[library]), "sampled_cells": target}
        )
    order = np.argsort(selected)
    selected_array = np.asarray(selected, dtype=int)[order]
    cell_metadata = obs.iloc[selected_array].copy()
    cell_metadata["cell_barcode"] = source.obs_names[selected_array].astype(str)
    layer = source.layers["counts"]
    matrix = layer[selected_array, :]
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    matrix = sparse.csr_matrix(matrix)
    genes = source.var_names.astype(str).copy()
    source.file.close()
    return matrix, cell_metadata.reset_index(drop=True), genes, pd.DataFrame(sampling_rows)


def _fit_grid(
    x: sparse.csr_matrix,
    genes: list[str],
    cell_metadata: pd.DataFrame,
    *,
    population: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    grid_rows: list[dict[str, Any]] = []
    reference: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    signatures_by_k: dict[int, list[np.ndarray]] = {}
    for k in K_GRID:
        signatures_by_k[k] = []
        errors: list[float] = []
        for seed_offset in range(3):
            model = NMF(
                n_components=k,
                init="nndsvdar",
                solver="cd",
                beta_loss="frobenius",
                max_iter=400,
                random_state=seed + seed_offset,
                tol=1e-4,
            )
            w = model.fit_transform(x)
            h = model.components_
            signatures_by_k[k].append(h)
            errors.append(float(model.reconstruction_err_))
            if seed_offset == 0:
                reference[k] = (w, h)
        grid_rows.append(
            {
                "population": population,
                "k": k,
                "n_seeds": 3,
                "mean_reconstruction_error": float(np.mean(errors)),
                "sd_reconstruction_error": float(np.std(errors, ddof=1)),
                "mean_component_stability": component_stability(signatures_by_k[k]),
            }
        )
    stability = pd.DataFrame(grid_rows).sort_values("k")
    error = stability["mean_reconstruction_error"].to_numpy()
    improvement = (error[:-1] - error[1:]) / np.maximum(error[:-1], 1e-12)
    stability["next_k_relative_error_improvement"] = np.r_[improvement, np.nan]
    selected_k = choose_rank(stability)
    stability["selected_k"] = stability["k"].eq(selected_k)
    w, h = reference[selected_k]

    program_rows: list[dict[str, Any]] = []
    for component in range(selected_k):
        order = np.argsort(-h[component])
        total = max(float(h[component].sum()), 1e-12)
        for rank, position in enumerate(order[:100], start=1):
            program_rows.append(
                {
                    "population": population,
                    "program": f"{population}_P{component + 1:02d}",
                    "rank": rank,
                    "gene": genes[position],
                    "weight": float(h[component, position]),
                    "weight_fraction": float(h[component, position] / total),
                }
            )
    program_genes = pd.DataFrame(program_rows)
    activity = pd.DataFrame(
        w,
        columns=[f"{population}_P{i + 1:02d}" for i in range(selected_k)],
    )
    activity["library_id"] = cell_metadata["library_id"].astype(str).to_numpy()
    activity["group"] = activity["library_id"].str.split("_").str[0]
    library_activity = activity.groupby(["library_id", "group"], observed=True).mean().reset_index()
    return stability, program_genes, library_activity, selected_k


def _bh(pvalues: pd.Series) -> pd.Series:
    values = pvalues.to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return pd.Series(adjusted, index=pvalues.index)


def _hallmark_overlap(program_genes: pd.DataFrame, gene_sets: dict[str, list[str]], universe: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (population, program), sub in program_genes.loc[program_genes["rank"].le(50)].groupby(["population", "program"], observed=True):
        genes = set(sub["gene"]) & universe
        for pathway, members in gene_sets.items():
            reference = set(members) & universe
            overlap = genes & reference
            p = float(hypergeom.sf(len(overlap) - 1, len(universe), len(reference), len(genes))) if genes and reference else 1.0
            rows.append(
                {
                    "population": population,
                    "program": program,
                    "hallmark": pathway,
                    "n_program_genes": len(genes),
                    "n_hallmark_genes": len(reference),
                    "n_overlap": len(overlap),
                    "jaccard": len(overlap) / max(len(genes | reference), 1),
                    "fisher_hypergeom_p": p,
                    "overlap_genes": ";".join(sorted(overlap)),
                }
            )
    result = pd.DataFrame(rows)
    result["fdr"] = _bh(result["fisher_hypergeom_p"])
    return result


def run_stage8(
    config: Mapping[str, Any],
    *,
    max_cells_per_library: int = 1000,
    n_hvg: int = 2000,
) -> None:
    paths = project_paths(config)
    output_root = paths["results"] / "stage8_gene_programs"
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(paths["logs"] / "stage8_gene_programs.log")
    seed = int(config.get("project", {}).get("random_seed", 20260810))
    h5ad_path = paths["results"] / "06_annotation_v2.h5ad"
    predicted_gib = estimate_dense_gib(9 * max_cells_per_library, n_hvg)
    if predicted_gib > 50:
        (output_root / "CNMF_SKIPPED_RESOURCE_LIMIT.md").write_text(
            f"# Stage 8 skipped\n\n预估工作矩阵峰值为 {predicted_gib:.1f} GiB，超过预定义50 GiB上限。\n",
            encoding="utf-8",
        )
        print("STAGE8_GENE_PROGRAMS_SKIPPED")
        return

    all_stability: list[pd.DataFrame] = []
    all_genes: list[pd.DataFrame] = []
    all_activity: list[pd.DataFrame] = []
    all_reversal: list[pd.DataFrame] = []
    all_sampling: list[pd.DataFrame] = []
    universes: dict[str, set[str]] = {}
    selected_summary: list[dict[str, Any]] = []
    for population in POPULATIONS:
        counts, metadata, genes, sampling = _balanced_counts(
            h5ad_path,
            population,
            max_cells_per_library=max_cells_per_library,
            seed=seed,
        )
        x, selected_genes, hvg_audit = select_hvgs(counts, genes, n_hvg)
        hvg_audit.insert(0, "population", population)
        hvg_audit.to_csv(output_root / f"hvg_audit__{population}.tsv.gz", sep="\t", index=False, compression="gzip")
        stability, program_genes, library_activity, selected_k = _fit_grid(
            x,
            selected_genes,
            metadata,
            population=population,
            seed=seed,
        )
        program_columns = [c for c in library_activity.columns if c.startswith(f"{population}_P")]
        contrast = activity_contrasts(library_activity.set_index("library_id")[program_columns])
        contrast.insert(0, "population", population)
        contrast = contrast.rename(columns={"program": "program"})
        all_stability.append(stability)
        all_genes.append(program_genes)
        all_activity.append(library_activity.assign(population=population))
        all_reversal.append(contrast)
        all_sampling.append(sampling)
        universes[population] = set(selected_genes)
        selected_summary.append({"population": population, "selected_k": selected_k, "n_hvg": len(selected_genes), "sampled_cells": len(metadata)})
        logger.info("NMF complete: %s K=%d", population, selected_k)

    stability = pd.concat(all_stability, ignore_index=True)
    program_genes = pd.concat(all_genes, ignore_index=True)
    activity = pd.concat(all_activity, ignore_index=True)
    reversal = pd.concat(all_reversal, ignore_index=True)
    sampling = pd.concat(all_sampling, ignore_index=True)
    gmt_path = paths["root"] / "resources" / "gene_sets" / "mh.all.v2026.1.Mm.symbols.gmt"
    gene_sets = parse_gmt(gmt_path)
    overlap = pd.concat(
        [_hallmark_overlap(program_genes.loc[program_genes["population"].eq(pop)], gene_sets, universe) for pop, universe in universes.items()],
        ignore_index=True,
    )
    stability.to_csv(output_root / "program_stability.tsv", sep="\t", index=False)
    program_genes.to_csv(output_root / "program_genes.tsv", sep="\t", index=False)
    activity.to_csv(output_root / "program_activity.tsv", sep="\t", index=False)
    reversal.to_csv(output_root / "program_reversal.tsv", sep="\t", index=False)
    overlap.to_csv(output_root / "program_hallmark_overlap.tsv", sep="\t", index=False)
    sampling.to_csv(output_root / "sampling_audit.tsv", sep="\t", index=False)
    summary = pd.DataFrame(selected_summary)
    strongest = reversal.loc[reversal["directionally_reversed"] & reversal["closer_to_young_after_treatment"]].sort_values("opposition_magnitude", ascending=False)
    report = [
        "# Stage 8 数据驱动基因程序报告",
        "",
        "## 1. 为什么做？",
        "Hallmark是预定义知识库；NMF用于寻找不依赖Hallmark标签、在本数据中共同变化的非负基因程序。",
        "",
        "## 2. 输入与资源控制",
        f"Granulosa和Stromal_fibroblast分开分析；每个library最多平衡抽样{max_cells_per_library}个Tier1细胞，选择{n_hvg}个非线粒体/非核糖体HVG。未对105k×57k全矩阵densify。预估主要dense工作内存约{predicted_gib:.2f} GiB。",
        "",
        "## 3. 方法",
        "固定测试K=5/8/10/12/15，每个K三个seed。组件用最优匹配后的余弦相似度评价稳定性；按预定义规则选择稳定且达到重构误差肘部的最小K。正式条件比较先把cell program usage聚合到library均值，n=3/group。",
        "",
        "## 4. 结果",
        summary.to_markdown(index=False),
        "",
        f"共有{len(strongest)}个program同时表现为aging与treatment方向相反且OT描述性地更接近Y。",
        strongest.head(20).to_markdown(index=False),
        "",
        "## 5. 与Hallmark的关系",
        "program_hallmark_overlap.tsv记录top50 program genes与Mouse Hallmark的交集、Jaccard和超几何FDR。低重叠program是潜在新信号，但也可能代表技术/细胞状态混合，不能仅凭低重叠宣称新机制。",
        "",
        "## 6. 可以与不能说明什么？",
        "NMF可说明哪些基因在当前数据中协同变化及其population/library分布；不能证明这些基因由同一上游机制直接调控，也不能把cell-level usage差异当独立重复。",
        "",
        "## 7. 局限与下一步",
        "K选择仍依赖稳定性阈值，NMF具有初始化依赖；抽样提高平衡性但损失部分细胞。仅将跨seed稳定、library-level反向且能与其他Stage相互支持的program纳入Stage10。",
    ]
    (output_root / "GENE_PROGRAM_REPORT_CN.md").write_text("\n".join(report), encoding="utf-8")

    outputs = [p for p in output_root.rglob("*") if p.is_file() and p.name not in {"manifest.tsv", "COMPLETE.json"}]
    manifest = pd.DataFrame([{"path": str(p.relative_to(paths["root"])), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in outputs])
    manifest.to_csv(output_root / "manifest.tsv", sep="\t", index=False)
    complete = {
        "stage": 8,
        "status": "COMPLETE",
        "seed": seed,
        "statistical_unit": "library",
        "k_grid": list(K_GRID),
        "n_seeds": 3,
        "software": {"scikit-learn": version("scikit-learn"), "anndata": version("anndata")},
        "gmt_sha256": sha256_file(gmt_path),
        "manifest_sha256": sha256_file(output_root / "manifest.tsv"),
    }
    (output_root / "COMPLETE.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print("STAGE8_GENE_PROGRAMS_COMPLETE")
