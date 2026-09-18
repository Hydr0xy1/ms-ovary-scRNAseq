"""Stage 15 latent gene-program stability audit.

This is a deliberately small, library-balanced NMF audit for the two primary
ovarian populations.  It never writes to the source AnnData object and treats
cells as observations for program learning only; condition-level summaries
remain library-level.  The output is exploratory unless programs are stable
across seeds and leave-one-library projections.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, issparse
from sklearn.decomposition import NMF

from .project import project_paths, setup_logging


FOCUS = ["Granulosa", "Stromal_fibroblast"]
LIBRARIES = [
    "Y_1",
    "Y_2",
    "Y_3",
    "OC_1",
    "OC_2",
    "OC_3",
    "OT_1",
    "OT_2",
    "OT_3",
]


def _balanced_positions(
    obs: pd.DataFrame,
    population: str,
    *,
    cells_per_library: int,
    seed: int,
    broad_key: str,
    tier_key: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return equally sized cell positions for all nine libraries."""

    mask = obs[broad_key].astype(str).eq(population) & obs[tier_key].astype(str).eq(
        "Tier1_primary"
    )
    positions: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for library in LIBRARIES:
        available = np.flatnonzero(mask.to_numpy() & obs["library_id"].astype(str).eq(library).to_numpy())
        if len(available) == 0:
            raise ValueError(f"No eligible cells for {population}/{library}")
        n = min(int(cells_per_library), len(available))
        rng = np.random.default_rng(int(seed) + sum(ord(c) for c in library))
        chosen = np.sort(rng.choice(available, size=n, replace=False))
        positions.append(chosen)
        rows.append(
            {
                "population": population,
                "library_id": library,
                "group": library.split("_", 1)[0],
                "n_available": int(len(available)),
                "n_sampled": int(n),
            }
        )
    n_balanced = min(len(x) for x in positions)
    positions = [x[:n_balanced] for x in positions]
    meta = pd.DataFrame(rows)
    meta["n_sampled_balanced"] = int(n_balanced)
    return np.concatenate(positions), meta


def _log_cpm_dense(counts: Any) -> np.ndarray:
    """Convert a sparse count block to non-negative log1p(CPM) values."""

    matrix = counts.tocsr() if issparse(counts) else csr_matrix(counts)
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
    totals[totals <= 0] = 1.0
    matrix = matrix.multiply((1e4 / totals)[:, None]).tocsr()
    matrix.data = np.log1p(matrix.data)
    return matrix.toarray().astype(np.float32, copy=False)


def _select_hvg(adata: ad.AnnData, n_genes: int) -> np.ndarray:
    if "highly_variable" not in adata.var:
        raise KeyError("The input object has no highly_variable flag")
    flags = adata.var["highly_variable"].to_numpy(dtype=bool)
    candidates = np.flatnonzero(flags)
    if "mt" in adata.var:
        candidates = candidates[~adata.var.iloc[candidates]["mt"].to_numpy(dtype=bool)]
    if "highly_variable_rank" in adata.var:
        ranks = pd.to_numeric(adata.var.iloc[candidates]["highly_variable_rank"], errors="coerce").to_numpy()
        order = np.argsort(np.nan_to_num(ranks, nan=np.inf), kind="stable")
        candidates = candidates[order]
    if len(candidates) < 100:
        raise ValueError(f"Too few HVGs available for NMF: {len(candidates)}")
    return candidates[: min(int(n_genes), len(candidates))]


def _fit_nmf(x: np.ndarray, k: int, seed: int, max_iter: int) -> tuple[NMF, np.ndarray, float]:
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("NMF input must be finite and non-negative")
    model = NMF(
        n_components=int(k),
        # Randomized NNDSVD retains a sensible non-negative start while making
        # the seed-stability check meaningful.
        init="nndsvdar",
        solver="cd",
        beta_loss="frobenius",
        max_iter=int(max_iter),
        tol=1e-4,
        random_state=int(seed),
    )
    w = model.fit_transform(x)
    return model, w, float(model.reconstruction_err_)


def _match_components(reference: np.ndarray, other: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Match two H matrices and return mean/min cosine and assignment."""

    ref_norm = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    oth_norm = other / np.maximum(np.linalg.norm(other, axis=1, keepdims=True), 1e-12)
    similarity = ref_norm @ oth_norm.T
    row, col = linear_sum_assignment(-similarity)
    matched = similarity[row, col]
    return float(np.mean(matched)), float(np.min(matched)), np.c_[row, col]


def _top_genes(
    h: np.ndarray,
    genes: pd.Index,
    *,
    population: str,
    k: int,
    seed: int,
    top_n: int = 50,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for component, weights in enumerate(h):
        order = np.argsort(weights)[::-1][: int(top_n)]
        for rank, idx in enumerate(order, start=1):
            rows.append(
                {
                    "population": population,
                    "k": int(k),
                    "seed": int(seed),
                    "program": int(component),
                    "rank": int(rank),
                    "gene": str(genes[int(idx)]),
                    "weight": float(weights[int(idx)]),
                }
            )
    return pd.DataFrame(rows)


def _load_module_sets(path: Path) -> dict[str, set[str]]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {str(k): set(map(str, v)) for k, v in payload.items()}


def _module_overlap(top: pd.DataFrame, modules: Mapping[str, set[str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (population, k, seed, program), sub in top.groupby(
        ["population", "k", "seed", "program"], observed=True
    ):
        genes = set(sub["gene"].astype(str))
        for module, module_genes in modules.items():
            overlap = genes & module_genes
            rows.append(
                {
                    "population": population,
                    "k": int(k),
                    "seed": int(seed),
                    "program": int(program),
                    "module": module,
                    "n_overlap": int(len(overlap)),
                    "overlap_genes": ";".join(sorted(overlap)),
                }
            )
    return pd.DataFrame(rows)


def run_program_stability(config: Mapping[str, Any]) -> Path:
    paths = project_paths(dict(config))
    settings = config["deep_dive_stage15"]
    root = paths["root"]
    defaults = {
        "output_dir": "results/deep_dive_stage15/program_stability",
        "per_library_cells": 300,
        "n_genes": 2000,
        "k_values": [6, 8, 10],
        "seeds": [20260917, 20260918, 20260919],
        "max_iter": 300,
        "top_genes": 50,
    }
    cfg = {**defaults, **dict(settings.get("program_stability", {}))}
    output_root = root / str(cfg["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("28_stage15_program_stability", dict(config))

    input_path = root / settings["input_object"]
    adata = ad.read_h5ad(input_path, backed="r")
    try:
        obs = adata.obs.copy()
        gene_positions = _select_hvg(adata, int(cfg["n_genes"]))
        genes = pd.Index(adata.var_names[gene_positions].astype(str))
        all_top: list[pd.DataFrame] = []
        fit_rows: list[dict[str, Any]] = []
        stability_rows: list[dict[str, Any]] = []
        loading_rows: list[dict[str, Any]] = []
        loo_rows: list[dict[str, Any]] = []
        for population in FOCUS:
            positions, sample_meta = _balanced_positions(
                obs,
                population,
                cells_per_library=int(cfg["per_library_cells"]),
                seed=int(config["project"]["random_seed"]),
                broad_key=str(settings["broad_key"]),
                tier_key=str(settings["tier_key"]),
            )
            x = _log_cpm_dense(adata.layers[settings["counts_layer"]][positions, :][:, gene_positions])
            informative = np.var(x, axis=0) > 0
            x = x[:, informative]
            gene_index = genes[informative]
            sampled_library = obs.iloc[positions]["library_id"].astype(str).to_numpy()
            models: dict[tuple[int, int], tuple[NMF, np.ndarray]] = {}
            for k in map(int, cfg["k_values"]):
                for seed in map(int, cfg["seeds"]):
                    model, w, error = _fit_nmf(x, k, seed, int(cfg["max_iter"]))
                    models[(k, seed)] = (model, w)
                    all_top.append(
                        _top_genes(
                            model.components_,
                            gene_index,
                            population=population,
                            k=k,
                            seed=seed,
                            top_n=int(cfg["top_genes"]),
                        )
                    )
                    fit_rows.append(
                        {
                            "population": population,
                            "k": k,
                            "seed": seed,
                            "n_cells": int(x.shape[0]),
                            "n_genes": int(x.shape[1]),
                            "n_iter": int(model.n_iter_),
                            "reconstruction_error": error,
                        }
                    )
                    for library, idx in pd.Series(np.arange(len(sampled_library))).groupby(sampled_library).groups.items():
                        loading_rows.append(
                            {
                                "population": population,
                                "k": k,
                                "seed": seed,
                                "library_id": library,
                                "group": str(library).split("_", 1)[0],
                                "n_cells": int(len(idx)),
                                **{f"program_{j}_mean": float(w[idx, j].mean()) for j in range(k)},
                            }
                        )
                for seed_a, seed_b in combinations(map(int, cfg["seeds"]), 2):
                    mean_sim, min_sim, assignment = _match_components(
                        models[(k, seed_a)][0].components_, models[(k, seed_b)][0].components_
                    )
                    stability_rows.append(
                        {
                            "population": population,
                            "k": k,
                            "seed_a": seed_a,
                            "seed_b": seed_b,
                            "mean_component_cosine": mean_sim,
                            "min_component_cosine": min_sim,
                            "n_components": int(len(assignment)),
                        }
                    )
                # Leave-one-library-out projection using one frozen seed per K.
                seed = int(list(map(int, cfg["seeds"]))[0])
                for held_out in LIBRARIES:
                    train = sampled_library != held_out
                    test = ~train
                    if not test.any() or not train.any():
                        continue
                    model, _, _ = _fit_nmf(x[train], k, seed, int(cfg["max_iter"]))
                    w_test = model.transform(x[test])
                    reconstructed = w_test @ model.components_
                    error = float(np.sqrt(np.mean((x[test] - reconstructed) ** 2)))
                    loo_rows.append(
                        {
                            "population": population,
                            "k": k,
                            "seed": seed,
                            "held_out_library": held_out,
                            "n_cells": int(test.sum()),
                            "held_out_rmse": error,
                        }
                    )
            sample_meta.to_csv(output_root / f"balanced_sampling_{population}.tsv", sep="\t", index=False)
            logger.info("Program stability complete for %s (%s cells, %s genes)", population, x.shape[0], x.shape[1])
    finally:
        adata.file.close()

    top = pd.concat(all_top, ignore_index=True)
    fits = pd.DataFrame(fit_rows)
    stability = pd.DataFrame(stability_rows)
    loadings = pd.DataFrame(loading_rows)
    loo = pd.DataFrame(loo_rows)
    modules = _load_module_sets(root / settings["state_modules_file"])
    overlap = _module_overlap(top, modules)
    top.to_csv(output_root / "program_top_genes.tsv", sep="\t", index=False)
    fits.to_csv(output_root / "program_fit_summary.tsv", sep="\t", index=False)
    stability.to_csv(output_root / "program_seed_stability.tsv", sep="\t", index=False)
    loadings.to_csv(output_root / "program_library_loadings.tsv", sep="\t", index=False)
    loo.to_csv(output_root / "program_leave_one_library_projection.tsv", sep="\t", index=False)
    overlap.to_csv(output_root / "program_state_module_overlap.tsv", sep="\t", index=False)
    status = {
        "status": "PROGRAM_STABILITY_COMPLETE",
        "input_object": str(input_path.relative_to(root)),
        "populations": FOCUS,
        "k_values": list(map(int, cfg["k_values"])),
        "seeds": list(map(int, cfg["seeds"])),
        "per_library_cells": int(cfg["per_library_cells"]),
        "n_genes_requested": int(cfg["n_genes"]),
        "n_fit_rows": int(len(fits)),
        "n_loo_rows": int(len(loo)),
        "interpretation": "Exploratory unless component stability and held-out-library error are acceptable; no component is assigned a causal pathway label.",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "next_stage": "regulatory_and_microenvironment_followup_only_for_stable_programs",
    }
    (output_root / "CHECKPOINT.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# Stage 15 潜在基因程序稳定性审计",
        "",
        "仅对 Granulosa 和 Stromal_fibroblast 做 library-balanced NMF；原始 H5AD 只读。",
        "正式条件比较仍以 library 为统计单位；程序学习和细胞 loading 只用于结构探索。",
        "",
        "## 运行设置",
        f"- 每个 library 抽样 {cfg['per_library_cells']} 个 Tier1_primary 细胞（按可用细胞数平衡）。",
        f"- HVG 数量上限：{cfg['n_genes']}；K：{list(map(int, cfg['k_values']))}；随机种子：{list(map(int, cfg['seeds']))}。",
        "- 留一 library 投影只用于泛化诊断，不是独立生物学验证。",
        "",
        "## 判读边界",
        "程序只有在跨随机种子相似度和留一 library 误差均可接受时才保留为候选；与状态模块的重叠仅用于描述，不能把 NMF 成分直接命名为通路或 MRJP1 机制。",
        "",
        "## 输出",
        "- `program_fit_summary.tsv`：拟合误差和迭代状态",
        "- `program_seed_stability.tsv`：跨随机种子 component cosine",
        "- `program_leave_one_library_projection.tsv`：留一 library RMSE",
        "- `program_top_genes.tsv`：各程序 top genes",
        "- `program_state_module_overlap.tsv`：与预定义状态模块的重叠",
        "- `program_library_loadings.tsv`：library-level loading 描述",
    ]
    (output_root / "PROGRAM_STABILITY_REPORT_CN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("STAGE15_PROGRAM_STABILITY_COMPLETE")
    print(f"OUTPUT={output_root.relative_to(root)}")
    return output_root
