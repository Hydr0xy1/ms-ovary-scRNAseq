"""Stage 24: library-aware scTenifoldKnk virtual-knockout pilot.

The stage is isolated from all earlier analyses. It reads raw UMI counts from
the frozen annotated object, constructs OC Granulosa networks after balanced
library sampling, and treats virtual-knockout output as exploratory model
evidence rather than causal or biological-replicate inference.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import re
import subprocess
import traceback
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.stats import fisher_exact, spearmanr

from .project import project_paths, setup_logging

STAGE_VERSION = "stage24-virtual-knockout-2026.09"
PRIMARY_SCENARIO = "tier1_primary"
REFERENCE_GENE_ROLE = "expression_matched_reference"
TARGET_GENE_ROLE = "pre_specified_candidate"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression="infer")


def _update_state(stage_root: Path, step: str, status: str, **extra: Any) -> None:
    path = stage_root / "RUN_STATE.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    state.setdefault("steps", {})[step] = {
        "status": status,
        "updated_at_utc": _utc_now(),
        **extra,
    }
    state["stage_version"] = STAGE_VERSION
    state["updated_at_utc"] = _utc_now()
    statuses = [entry.get("status") for entry in state["steps"].values()]
    state["overall"] = (
        "failed"
        if "failed" in statuses
        else "running"
        if "running" in statuses
        else "complete"
        if statuses and all(item == "complete" for item in statuses)
        else "partial"
    )
    _write_json(state, path)


def _record_failure(stage_root: Path, step: str, exc: BaseException, logger: Any) -> None:
    path = stage_root / "FAILED_STEPS.tsv"
    row = pd.DataFrame(
        [
            {
                "step": step,
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "time_utc": _utc_now(),
            }
        ]
    )
    if path.exists():
        row = pd.concat([pd.read_csv(path, sep="\t"), row], ignore_index=True)
    _write_tsv(row, path)
    logger.error("Stage 24 step failed: %s: %s", step, exc)
    logger.error(traceback.format_exc())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bh_adjust(values: Sequence[float]) -> np.ndarray:
    pvalues = np.asarray(values, dtype=float)
    adjusted = np.full(pvalues.shape, np.nan, dtype=float)
    finite = np.isfinite(pvalues)
    if not finite.any():
        return adjusted
    observed = pvalues[finite]
    order = np.argsort(observed)
    ranked = observed[order]
    corrected = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    restored = np.empty_like(corrected)
    restored[order] = np.clip(corrected, 0, 1)
    adjusted[finite] = restored
    return adjusted


def balanced_sample(
    obs: pd.DataFrame,
    *,
    library_key: str,
    cells_per_library: int,
    seed: int,
    libraries: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Select exactly the same number of cells per library."""

    requested = sorted(map(str, libraries or obs[library_key].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    for library in requested:
        block = obs.loc[obs[library_key].astype(str).eq(library)].copy()
        if len(block) < cells_per_library:
            raise ValueError(
                f"{library} has {len(block)} eligible cells; "
                f"{cells_per_library} were requested"
            )
        chosen = np.sort(rng.choice(len(block), size=cells_per_library, replace=False))
        sampled = block.iloc[chosen].copy()
        sampled["balanced_sampling_seed"] = int(seed)
        sampled["cells_per_library_requested"] = int(cells_per_library)
        rows.append(sampled)
    result = pd.concat(rows, axis=0)
    result["cell_barcode"] = result.index.astype(str)
    return result


def select_gene_universe(
    gene_stats: pd.DataFrame,
    *,
    hvg_rank_column: str,
    forced_genes: Iterable[str],
    n_genes: int,
    forced_min_detection_fraction: float,
) -> pd.DataFrame:
    """Freeze a bounded network universe while retaining pre-specified genes."""

    table = gene_stats.copy()
    table["gene"] = table["gene"].astype(str)
    table = table.drop_duplicates("gene").set_index("gene", drop=False)
    table.index.name = None
    forced_requested = list(dict.fromkeys(map(str, forced_genes)))
    forced = [
        gene
        for gene in forced_requested
        if gene in table.index
        and float(table.loc[gene, "detection_fraction"]) >= forced_min_detection_fraction
    ]
    if len(forced) >= n_genes:
        raise ValueError("Forced gene set is larger than the requested network universe")
    ranked = table.sort_values(
        [hvg_rank_column, "detection_fraction", "gene"],
        ascending=[True, False, True],
        na_position="last",
    )
    fillers = [gene for gene in ranked.index if gene not in forced]
    selected = forced + fillers[: n_genes - len(forced)]
    result = table.loc[selected].copy()
    result["selection_order"] = np.arange(1, len(result) + 1)
    result["forced_gene"] = result.index.isin(forced)
    result["selection_source"] = np.where(
        result["forced_gene"], "pre_specified_force_include", "batch_aware_hvg"
    )
    return result.reset_index(drop=True)


def choose_expression_matched_references(
    gene_stats: pd.DataFrame,
    *,
    targets: Sequence[str],
    excluded_genes: Iterable[str] = (),
) -> pd.DataFrame:
    """Choose unique expression-matched reference genes for target benchmarking."""

    table = gene_stats.drop_duplicates("gene").set_index("gene", drop=False).copy()
    table.index.name = None
    excluded = set(map(str, excluded_genes)) | set(map(str, targets))
    invalid_name = re.compile(r"^(mt-|Rpl|Rps|Gm\d|Hist|H[234][a-z])", re.IGNORECASE)
    pool = table.loc[
        ~table.index.astype(str).isin(excluded)
        & ~table.index.astype(str).str.match(invalid_name)
    ].copy()
    if len(pool) < len(targets):
        raise ValueError("Too few eligible genes for expression-matched references")
    features = ["mean_umi", "detection_fraction"]
    transformed = pd.DataFrame(index=table.index)
    transformed["mean_umi"] = np.log1p(pd.to_numeric(table["mean_umi"], errors="coerce"))
    transformed["detection_fraction"] = pd.to_numeric(
        table["detection_fraction"], errors="coerce"
    )
    scale = transformed.loc[pool.index, features].std(ddof=0).replace(0, 1)
    center = transformed.loc[pool.index, features].mean()
    zscores = (transformed[features] - center) / scale
    used: set[str] = set()
    rows: list[dict[str, Any]] = []
    for target in targets:
        if target not in zscores.index:
            raise ValueError(f"Target is missing from gene universe: {target}")
        candidates = pool.loc[~pool.index.isin(used)].copy()
        distances = np.sqrt(
            ((zscores.loc[candidates.index, features] - zscores.loc[target, features]) ** 2).sum(
                axis=1
            )
        )
        reference = str(distances.sort_values(kind="stable").index[0])
        used.add(reference)
        rows.append(
            {
                "target_gene": target,
                "reference_gene": reference,
                "matching_distance": float(distances.loc[reference]),
                "target_mean_umi": float(table.loc[target, "mean_umi"]),
                "reference_mean_umi": float(table.loc[reference, "mean_umi"]),
                "target_detection_fraction": float(table.loc[target, "detection_fraction"]),
                "reference_detection_fraction": float(
                    table.loc[reference, "detection_fraction"]
                ),
                "interpretation": "expression-matched reference; not a proven biological null",
            }
        )
    return pd.DataFrame(rows)


def pairwise_jaccard(sets: Sequence[set[str]]) -> float:
    values: list[float] = []
    for left, right in combinations(sets, 2):
        union = left | right
        values.append(float(len(left & right) / len(union)) if union else np.nan)
    return float(np.nanmedian(values)) if values else np.nan


def cpm_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """CPM-normalize a genes-by-cells matrix as in the official R workflow."""

    library_sizes = frame.sum(axis=0)
    safe = library_sizes.replace(0, np.nan)
    normalized = frame.multiply(1_000_000 / safe, axis=1)
    return normalized.fillna(0)


def _module_genes(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    return {str(name): list(map(str, genes)) for name, genes in values.items()}


def _stage_paths(config: Mapping[str, Any]) -> tuple[Path, Path, dict[str, Path]]:
    paths = project_paths(dict(config))
    cfg = config["stage24_virtual_knockout"]
    stage_root = paths["root"] / str(cfg["output_dir"])
    figure_root = paths["root"] / str(cfg["figure_dir"])
    for directory in [
        stage_root,
        stage_root / "00_preparation",
        stage_root / "01_runs",
        stage_root / "02_summary",
        stage_root / "03_r_crosscheck",
        stage_root / "logs",
        figure_root,
        figure_root / "source_data",
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    return stage_root, figure_root, paths


def _extract_sparse(layer: Any, rows: np.ndarray, columns: np.ndarray) -> sparse.csr_matrix:
    matrix = layer[rows[:, None], columns]
    return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)


def _evidence_exclusions(root: Path) -> set[str]:
    excluded: set[str] = set()
    candidate_path = root / "results/stage10_candidates/candidate_gene_evidence.tsv"
    if candidate_path.exists():
        frame = pd.read_csv(candidate_path, sep="\t")
        frame = frame.loc[frame.get("cell_type", "").astype(str).eq("Granulosa")]
        evidence_columns = [
            column
            for column in frame.columns
            if column.startswith("stage") or column == "external_aging_supported"
        ]
        supported = np.zeros(len(frame), dtype=bool)
        for column in evidence_columns:
            if pd.api.types.is_bool_dtype(frame[column]):
                supported |= frame[column].fillna(False).to_numpy(dtype=bool)
            else:
                supported |= frame[column].fillna("").astype(str).str.len().gt(0).to_numpy()
        excluded.update(frame.loc[supported, "gene"].astype(str))
    collectri = root / "results/stage7_regulatory_activity/resources/collectri_mouse.tsv.gz"
    if collectri.exists():
        excluded.update(pd.read_csv(collectri, sep="\t", usecols=["source"])["source"].astype(str))
    return excluded


def prepare_stage24(
    config: Mapping[str, Any], stage_root: Path, paths: Mapping[str, Path], logger: Any
) -> None:
    import anndata as ad
    import scanpy as sc

    cfg = config["stage24_virtual_knockout"]
    out = stage_root / "00_preparation"
    done = out / "PREPARED.json"
    if done.exists():
        logger.info("Stage 24 preparation already complete")
        return
    input_path = paths["root"] / str(cfg["input_object"])
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    logger.info("Opening frozen input in backed read-only mode: %s", input_path)
    adata = ad.read_h5ad(input_path, backed="r")
    required_obs = [
        cfg["broad_key"],
        cfg["group_key"],
        cfg["library_key"],
        cfg["tier_key"],
    ]
    missing = [column for column in required_obs if column not in adata.obs]
    if missing:
        raise KeyError(f"Missing obs columns: {missing}")
    if cfg["counts_layer"] not in adata.layers:
        raise KeyError(f"Missing raw-count layer: {cfg['counts_layer']}")
    obs = adata.obs.copy()
    granulosa = obs[cfg["broad_key"]].astype(str).eq(str(cfg["target_population"]))
    reference = obs[cfg["group_key"]].astype(str).eq(str(cfg["reference_group"]))
    sensitivity = obs[cfg["tier_key"]].astype(str).isin(map(str, cfg["sensitivity_tiers"]))
    ref_indices = np.flatnonzero((granulosa & reference & sensitivity).to_numpy())
    if len(ref_indices) == 0:
        raise ValueError("No eligible OC Granulosa cells found")
    all_columns = np.arange(adata.n_vars, dtype=int)
    counts = _extract_sparse(adata.layers[cfg["counts_layer"]], ref_indices, all_columns)
    if counts.data.size and float(counts.data.min()) < 0:
        raise ValueError("Raw-count layer contains negative values")
    rounded = np.rint(counts.data)
    if counts.data.size and not np.allclose(counts.data, rounded):
        raise ValueError("Raw-count layer contains non-integer values")
    n_cells = counts.shape[0]
    detected = np.asarray((counts > 0).sum(axis=0)).ravel()
    mean_umi = np.asarray(counts.mean(axis=0)).ravel()
    detection_fraction = detected / n_cells
    var = adata.var.copy()
    gene_stats = pd.DataFrame(
        {
            "gene": adata.var_names.astype(str),
            "gene_id": var["gene_ids"].astype(str).to_numpy()
            if "gene_ids" in var
            else adata.var_names.astype(str),
            "n_detected_cells": detected,
            "detection_fraction": detection_fraction,
            "mean_umi": mean_umi,
            "mt": var.get("mt", pd.Series(False, index=var.index)).astype(bool).to_numpy(),
            "ribo": var.get("ribo", pd.Series(False, index=var.index)).astype(bool).to_numpy(),
            "hb": var.get("hb", pd.Series(False, index=var.index)).astype(bool).to_numpy(),
        }
    )
    eligible = (
        gene_stats["detection_fraction"].ge(float(cfg["min_detection_fraction"]))
        & ~gene_stats[["mt", "ribo", "hb"]].any(axis=1)
    )
    eligible_indices = np.flatnonzero(eligible.to_numpy())
    logger.info(
        "Selecting batch-aware HVGs from %d OC Granulosa cells and %d eligible genes",
        n_cells,
        len(eligible_indices),
    )
    hvg = ad.AnnData(
        X=counts[:, eligible_indices].copy(),
        obs=obs.iloc[ref_indices][[cfg["library_key"]]].copy(),
        var=pd.DataFrame(index=gene_stats.iloc[eligible_indices]["gene"].astype(str)),
    )
    try:
        sc.pp.highly_variable_genes(
            hvg,
            flavor=str(cfg["hvg_flavor"]),
            n_top_genes=min(int(cfg["n_genes"]), hvg.n_vars),
            batch_key=str(cfg["hvg_batch_key"]),
            inplace=True,
        )
        rank_map = hvg.var["highly_variable_rank"].to_dict()
    except Exception as exc:
        logger.warning("Batch-aware HVG selection failed; using variance fallback: %s", exc)
        means = np.asarray(hvg.X.mean(axis=0)).ravel()
        squares = np.asarray(hvg.X.power(2).mean(axis=0)).ravel()
        score = (squares - means**2) / np.maximum(means, 1e-8)
        rank_map = pd.Series(-score, index=hvg.var_names).rank(method="first").to_dict()
    gene_stats["hvg_rank"] = gene_stats["gene"].map(rank_map)
    modules_path = paths["root"] / str(cfg["state_modules_file"])
    modules = _module_genes(modules_path)
    module_union = set().union(*map(set, modules.values()))
    forced = (
        list(cfg["primary_candidates"])
        + list(cfg["secondary_candidates"])
        + sorted(module_union)
    )
    universe = select_gene_universe(
        gene_stats,
        hvg_rank_column="hvg_rank",
        forced_genes=forced,
        n_genes=int(cfg["n_genes"]),
        forced_min_detection_fraction=float(cfg["forced_min_detection_fraction"]),
    )
    primary_missing = sorted(set(cfg["primary_candidates"]) - set(universe["gene"]))
    if primary_missing:
        raise ValueError(f"Primary candidate(s) failed expression filter: {primary_missing}")
    exclusions = _evidence_exclusions(paths["root"]) | module_union | set(
        cfg["secondary_candidates"]
    )
    references = choose_expression_matched_references(
        universe,
        targets=list(cfg["primary_candidates"]),
        excluded_genes=exclusions,
    )
    universe["gene_role"] = "network_background"
    universe.loc[
        universe["gene"].isin(cfg["primary_candidates"]), "gene_role"
    ] = TARGET_GENE_ROLE
    universe.loc[
        universe["gene"].isin(references["reference_gene"]), "gene_role"
    ] = REFERENCE_GENE_ROLE
    _write_tsv(gene_stats, out / "oc_granulosa_gene_statistics.tsv.gz")
    _write_tsv(universe, out / "gene_universe.tsv")
    _write_tsv(references, out / "expression_matched_reference_genes.tsv")
    counts_by_library = (
        obs.loc[granulosa]
        .groupby(
            [cfg["group_key"], cfg["library_key"], cfg["tier_key"]],
            observed=True,
        )
        .size()
        .rename("n_cells")
        .reset_index()
    )
    _write_tsv(counts_by_library, out / "granulosa_cell_counts.tsv")
    all_granulosa_indices = np.flatnonzero(granulosa.to_numpy())
    target_genes = list(cfg["primary_candidates"]) + list(cfg["secondary_candidates"])
    target_columns = adata.var_names.get_indexer(target_genes)
    if (target_columns < 0).any():
        raise ValueError("One or more candidate genes are absent from var_names")
    target_counts = _extract_sparse(
        adata.layers[cfg["counts_layer"]], all_granulosa_indices, target_columns
    ).toarray()
    coverage_obs = obs.iloc[all_granulosa_indices][
        [cfg["group_key"], cfg["library_key"], cfg["tier_key"]]
    ].copy()
    coverage_rows: list[dict[str, Any]] = []
    for (group, library), positions in coverage_obs.groupby(
        [cfg["group_key"], cfg["library_key"]], observed=True
    ).indices.items():
        local = target_counts[np.asarray(positions, dtype=int)]
        for column, gene in enumerate(target_genes):
            values = local[:, column]
            coverage_rows.append(
                {
                    "group": str(group),
                    "library_id": str(library),
                    "gene": gene,
                    "n_cells": len(values),
                    "n_detected": int((values > 0).sum()),
                    "detection_fraction": float((values > 0).mean()),
                    "mean_umi": float(values.mean()),
                }
            )
    _write_tsv(pd.DataFrame(coverage_rows), out / "candidate_coverage_by_library.tsv")
    audit = {
        "stage_version": STAGE_VERSION,
        "input_object": str(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "input_sha256": _sha256(input_path),
        "input_shape": [int(adata.n_obs), int(adata.n_vars)],
        "counts_layer": str(cfg["counts_layer"]),
        "reference_population": str(cfg["target_population"]),
        "reference_group": str(cfg["reference_group"]),
        "reference_cells": int(n_cells),
        "network_gene_count": int(len(universe)),
        "primary_candidates": list(map(str, cfg["primary_candidates"])),
        "secondary_candidates_not_in_initial_queue": list(
            map(str, cfg["secondary_candidates"])
        ),
        "expression_matched_references": references.to_dict(orient="records"),
        "statistical_unit": "library/pool",
        "interpretation_boundary": (
            "virtual knockout is an exploratory network perturbation prediction, "
            "not a causal experiment or biological-replicate test"
        ),
        "prepared_at_utc": _utc_now(),
    }
    _write_json(audit, done)
    adata.file.close()
    logger.info("Stage 24 preparation complete: %d frozen genes", len(universe))


def _scenario_definitions(config: Mapping[str, Any], preparation: Path) -> list[dict[str, Any]]:
    cfg = config["stage24_virtual_knockout"]
    definitions: list[dict[str, Any]] = []
    for name, values in cfg["scenarios"].items():
        for seed in values["seeds"]:
            definitions.append(
                {
                    "scenario": str(name),
                    "tiers": list(map(str, values["tiers"])),
                    "cells_per_library": int(values["cells_per_library"]),
                    "seed": int(seed),
                    "excluded_library": None,
                }
            )
    loo = cfg.get("leave_one_library_out", {})
    if loo.get("enabled", False):
        counts = pd.read_csv(preparation / "granulosa_cell_counts.tsv", sep="\t")
        libraries = sorted(
            counts.loc[
                counts[cfg["group_key"]].astype(str).eq(cfg["reference_group"]),
                cfg["library_key"],
            ]
            .astype(str)
            .unique()
        )
        for excluded in libraries:
            definitions.append(
                {
                    "scenario": f"loo_exclude_{excluded}",
                    "tiers": list(map(str, loo["tiers"])),
                    "cells_per_library": int(loo["cells_per_library"]),
                    "seed": int(loo["seed"]),
                    "excluded_library": excluded,
                }
            )
    return definitions


def _environment_manifest() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for package in [
        "anndata",
        "scanpy",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "scTenifoldpy",
        "tensorly",
    ]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cpu_count": os.cpu_count(),
    }


def _save_wt_tensor(frame: pd.DataFrame, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "WT_TENSOR.npz",
        values=frame.to_numpy(dtype=np.float64),
        genes=frame.index.astype(str).to_numpy(dtype="U"),
    )


def _load_wt_tensor(directory: Path) -> pd.DataFrame:
    with np.load(directory / "WT_TENSOR.npz") as values:
        genes = values["genes"].astype(str)
        matrix = values["values"]
    return pd.DataFrame(matrix, index=genes, columns=genes)


def _standardize_knockout_result(
    result: pd.DataFrame, *, target: str, role: str, scenario: str, seed: int
) -> pd.DataFrame:
    rename = {
        "Gene": "gene",
        "Distance": "distance",
        "boxcox-transformed distance": "boxcox_distance",
        "Z": "z_score",
        "FC": "fold_change",
        "p-value": "model_pvalue",
        "adjusted p-value": "model_padj",
    }
    table = result.rename(columns=rename).copy()
    required = set(rename.values())
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Unexpected scTenifoldpy output; missing columns: {sorted(missing)}")
    table = table.sort_values(["distance", "gene"], ascending=[False, True]).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    table["rank_fraction"] = table["rank"] / len(table)
    table["ko_target"] = target
    table["ko_role"] = role
    table["scenario"] = scenario
    table["seed"] = int(seed)
    table["is_knockout_target"] = table["gene"].astype(str).eq(target)
    table["pvalue_scope"] = "model_internal_gene_ranking_only"
    return table


def _instantiate_from_tensor(tensor: pd.DataFrame, cfg: Mapping[str, Any]) -> Any:
    from scTenifold import scTenifoldKnk

    model = scTenifoldKnk(
        pd.DataFrame(index=tensor.index),
        ko_method="default",
        strict_lambda=0,
        ma_kws=dict(cfg["manifold_alignment"]),
        dr_kws={"n_ko_genes": 1},
    )
    model.shared_gene_names = tensor.index.astype(str).tolist()
    model.tensor_dict["WT"] = tensor
    return model


def _run_definition(
    config: Mapping[str, Any],
    definition: Mapping[str, Any],
    stage_root: Path,
    paths: Mapping[str, Path],
    logger: Any,
) -> None:
    import anndata as ad
    from threadpoolctl import threadpool_limits

    cfg = config["stage24_virtual_knockout"]
    preparation = stage_root / "00_preparation"
    universe = pd.read_csv(preparation / "gene_universe.tsv", sep="\t")
    references = pd.read_csv(
        preparation / "expression_matched_reference_genes.tsv", sep="\t"
    )
    targets = list(map(str, cfg["primary_candidates"]))
    reference_genes = (
        references.set_index("target_gene")
        .loc[targets, "reference_gene"]
        .astype(str)
        .tolist()
    )
    ko_roles = {gene: TARGET_GENE_ROLE for gene in targets}
    ko_roles.update({gene: REFERENCE_GENE_ROLE for gene in reference_genes})
    scenario = str(definition["scenario"])
    seed = int(definition["seed"])
    run_dir = stage_root / "01_runs" / scenario / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    completion_path = run_dir / "COMPLETE.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") == "complete":
            logger.info("Skipping completed run %s seed %d", scenario, seed)
            return
    input_path = paths["root"] / str(cfg["input_object"])
    adata = ad.read_h5ad(input_path, backed="r")
    obs = adata.obs.copy()
    eligible = (
        obs[cfg["broad_key"]].astype(str).eq(str(cfg["target_population"]))
        & obs[cfg["group_key"]].astype(str).eq(str(cfg["reference_group"]))
        & obs[cfg["tier_key"]].astype(str).isin(definition["tiers"])
    )
    eligible_obs = obs.loc[eligible].copy()
    libraries = sorted(eligible_obs[cfg["library_key"]].astype(str).unique())
    excluded = definition.get("excluded_library")
    if excluded:
        libraries = [library for library in libraries if library != str(excluded)]
    sampled = balanced_sample(
        eligible_obs,
        library_key=str(cfg["library_key"]),
        cells_per_library=int(definition["cells_per_library"]),
        seed=seed,
        libraries=libraries,
    )
    _write_tsv(
        sampled[
            [
                "cell_barcode",
                cfg["library_key"],
                cfg["group_key"],
                cfg["tier_key"],
                "balanced_sampling_seed",
                "cells_per_library_requested",
            ]
        ],
        run_dir / "sampled_cells.tsv.gz",
    )
    cell_indices = adata.obs_names.get_indexer(sampled.index)
    gene_indices = adata.var_names.get_indexer(universe["gene"].astype(str))
    if (cell_indices < 0).any() or (gene_indices < 0).any():
        raise ValueError("Sampled cells or frozen genes are missing from the input object")
    counts = _extract_sparse(adata.layers[cfg["counts_layer"]], cell_indices, gene_indices)
    sums = np.asarray(counts.sum(axis=0)).ravel()
    squares = np.asarray(counts.power(2).sum(axis=0)).ravel()
    variance = squares / counts.shape[0] - (sums / counts.shape[0]) ** 2
    informative = (sums > 0) & (variance > 0)
    dropped = universe.loc[~informative, ["gene", "gene_role"]].copy()
    if not dropped.empty:
        _write_tsv(dropped, run_dir / "constant_or_zero_genes.tsv")
    active_genes = universe.loc[informative, "gene"].astype(str).tolist()
    missing_targets = sorted(set(ko_roles) - set(active_genes))
    if missing_targets:
        raise ValueError(f"KO target/reference genes constant in sampled data: {missing_targets}")
    expression = pd.DataFrame(
        counts[:, informative].T.toarray(),
        index=active_genes,
        columns=sampled.index.astype(str),
        dtype=np.float32,
    )
    expression = cpm_normalize(expression)
    tensor_path = run_dir / "WT_TENSOR.npz"
    if tensor_path.exists():
        logger.info("Loading checkpointed WT tensor for %s seed %d", scenario, seed)
        tensor = _load_wt_tensor(run_dir)
        model = _instantiate_from_tensor(tensor, cfg)
    else:
        from scTenifold import scTenifoldKnk

        logger.info(
            "Building WT network for %s seed %d: %d cells x %d genes",
            scenario,
            seed,
            expression.shape[1],
            expression.shape[0],
        )
        model = scTenifoldKnk(
            expression,
            ko_genes=targets[0],
            ko_method="default",
            strict_lambda=0,
            qc_kws={
                "min_lib_size": 0,
                "remove_outlier_cells": False,
                "min_percent": 0,
                "max_mt_ratio": 1,
                "min_exp_avg": 0,
                "min_exp_sum": 0,
                "plot": False,
            },
            nc_kws={
                **dict(cfg["network"]),
                "random_state": seed,
            },
            td_kws={
                **dict(cfg["tensor_decomposition"]),
                "random_state": seed,
            },
            ma_kws=dict(cfg["manifold_alignment"]),
            dr_kws={"n_ko_genes": 1},
        )
        with threadpool_limits(limits=1):
            model.run_step("qc")
            model.run_step("nc")
        with threadpool_limits(limits=4):
            model.run_step("td")
        tensor = model.tensor_dict["WT"]
        _save_wt_tensor(tensor, run_dir)
        model.network_dict.clear()
        gc.collect()
    metadata = {
        **dict(definition),
        "n_cells": int(expression.shape[1]),
        "n_genes_requested": int(len(universe)),
        "n_genes_modeled": int(len(model.tensor_dict["WT"])),
        "libraries": libraries,
        "ko_targets": ko_roles,
        "network_input_normalization": "CPM to 1e6 within frozen network universe",
        "environment": _environment_manifest(),
        "started_or_resumed_at_utc": _utc_now(),
    }
    _write_json(metadata, run_dir / "RUN_METADATA.json")
    knockout_dir = run_dir / "knockouts"
    knockout_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for target, role in ko_roles.items():
        output = knockout_dir / f"{target}.tsv.gz"
        if output.exists():
            logger.info("Skipping checkpointed KO %s / %s / %d", target, scenario, seed)
            continue
        try:
            logger.info("Virtual KO %s / %s / seed %d", target, scenario, seed)
            with threadpool_limits(limits=4):
                model.run_step("ko", ko_genes=target)
                model.run_step("ma")
                model.run_step("dr")
            standardized = _standardize_knockout_result(
                model.d_regulation,
                target=target,
                role=role,
                scenario=scenario,
                seed=seed,
            )
            _write_tsv(standardized, output)
        except Exception as exc:  # continue remaining targets but retain a visible failure
            failures.append(target)
            _record_failure(stage_root, f"{scenario}/seed_{seed}/{target}", exc, logger)
        finally:
            model.tensor_dict.pop("KO", None)
            model.manifold = None
            model.d_regulation = None
            model.step_comps["ma"] = None
            model.step_comps["dr"] = None
            gc.collect()
    adata.file.close()
    del expression, counts, model
    gc.collect()
    completion = {
        "scenario": scenario,
        "seed": seed,
        "completed_at_utc": _utc_now(),
        "failed_targets": failures,
        "status": "complete" if not failures else "partial",
    }
    _write_json(completion, run_dir / "COMPLETE.json")
    if failures:
        raise RuntimeError(f"One or more virtual knockouts failed: {failures}")


def _crosscheck_definition(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = config["stage24_virtual_knockout"]
    cross = cfg["r_crosscheck"]
    scenario = str(cross["scenario"])
    values = cfg["scenarios"][scenario]
    return {
        "scenario": scenario,
        "tiers": list(map(str, values["tiers"])),
        "cells_per_library": int(values["cells_per_library"]),
        "seed": int(cross["seed"]),
        "excluded_library": None,
    }


def _run_python_crosscheck(
    expression: pd.DataFrame,
    *,
    target: str,
    cfg: Mapping[str, Any],
    random_state: int,
    output: Path,
) -> None:
    from scTenifold import scTenifoldKnk
    from threadpoolctl import threadpool_limits

    if output.exists():
        return
    model = scTenifoldKnk(
        expression,
        ko_genes=target,
        ko_method="default",
        strict_lambda=0,
        qc_kws={
            "min_lib_size": 0,
            "remove_outlier_cells": False,
            "min_percent": 0,
            "max_mt_ratio": 1,
            "min_exp_avg": 0,
            "min_exp_sum": 0,
            "plot": False,
        },
        nc_kws={**dict(cfg["network"]), "random_state": random_state},
        td_kws={**dict(cfg["tensor_decomposition"]), "random_state": random_state},
        ma_kws=dict(cfg["manifold_alignment"]),
        dr_kws={"n_ko_genes": 1},
    )
    with threadpool_limits(limits=1):
        model.run_step("qc")
        model.run_step("nc")
    with threadpool_limits(limits=4):
        model.run_step("td")
        model.run_step("ko")
        model.run_step("ma")
        model.run_step("dr")
    result = _standardize_knockout_result(
        model.d_regulation,
        target=target,
        role=TARGET_GENE_ROLE,
        scenario="python_r_crosscheck",
        seed=random_state,
    )
    _write_tsv(result, output)


def run_r_crosscheck(
    config: Mapping[str, Any], stage_root: Path, paths: Mapping[str, Path], logger: Any
) -> None:
    import anndata as ad

    cfg = config["stage24_virtual_knockout"]
    cross = cfg.get("r_crosscheck", {})
    if not cross.get("enabled", False):
        logger.info("R cross-check disabled")
        return
    out = stage_root / "03_r_crosscheck"
    comparison_path = out / "R_PYTHON_CONCORDANCE.json"
    if comparison_path.exists():
        logger.info("R/Python cross-check already complete")
        return
    target = str(cross["target_gene"])
    definition = _crosscheck_definition(config)
    universe = pd.read_csv(stage_root / "00_preparation/gene_universe.tsv", sep="\t")
    n_genes = int(cross["n_genes"])
    cross_genes = (
        universe.sort_values("selection_order").head(n_genes)["gene"].astype(str).tolist()
    )
    if target not in cross_genes:
        cross_genes[-1] = target
    input_path = paths["root"] / str(cfg["input_object"])
    adata = ad.read_h5ad(input_path, backed="r")
    obs = adata.obs.copy()
    eligible = (
        obs[cfg["broad_key"]].astype(str).eq(str(cfg["target_population"]))
        & obs[cfg["group_key"]].astype(str).eq(str(cfg["reference_group"]))
        & obs[cfg["tier_key"]].astype(str).isin(definition["tiers"])
    )
    eligible_obs = obs.loc[eligible].copy()
    sampled = balanced_sample(
        eligible_obs,
        library_key=str(cfg["library_key"]),
        cells_per_library=int(definition["cells_per_library"]),
        seed=int(definition["seed"]),
    )
    cell_indices = adata.obs_names.get_indexer(sampled.index)
    gene_indices = adata.var_names.get_indexer(cross_genes)
    counts = _extract_sparse(adata.layers[cfg["counts_layer"]], cell_indices, gene_indices)
    sums = np.asarray(counts.sum(axis=0)).ravel()
    squares = np.asarray(counts.power(2).sum(axis=0)).ravel()
    variance = squares / counts.shape[0] - (sums / counts.shape[0]) ** 2
    informative = (sums > 0) & (variance > 0)
    cross_genes = list(np.asarray(cross_genes)[informative])
    if target not in cross_genes:
        raise ValueError(f"Cross-check target {target} is constant in sampled data")
    raw = pd.DataFrame(
        counts[:, informative].T.toarray(),
        index=cross_genes,
        columns=sampled.index.astype(str),
        dtype=np.float32,
    )
    raw.index.name = "gene"
    raw.to_csv(out / "crosscheck_raw_counts.tsv.gz", sep="\t", compression="gzip")
    _write_tsv(
        sampled[["cell_barcode", cfg["library_key"], cfg["tier_key"]]],
        out / "crosscheck_sampled_cells.tsv",
    )
    _write_tsv(
        universe.loc[universe["gene"].isin(cross_genes)].sort_values("selection_order"),
        out / "crosscheck_gene_universe.tsv",
    )
    expression = cpm_normalize(raw)
    python_output = out / f"python_{target}.tsv.gz"
    _run_python_crosscheck(
        expression,
        target=target,
        cfg=cfg,
        random_state=1,
        output=python_output,
    )
    rscript = Path(str(cross["rscript"]))
    if not rscript.exists():
        raise FileNotFoundError(
            f"Official R environment is missing: {rscript}. "
            "Run environment/bootstrap_stage24_r.sh first."
        )
    r_output = out / f"r_{target}.tsv.gz"
    if not r_output.exists():
        log_path = out / "r_crosscheck.log"
        command = [
            str(rscript),
            str(paths["root"] / "scripts/36_stage24_r_crosscheck.R"),
            str(out / "crosscheck_raw_counts.tsv.gz"),
            str(r_output),
            target,
            str(int(cross["n_cores"])),
        ]
        logger.info("Running official R scTenifoldKnk cross-check")
        with log_path.open("w", encoding="utf-8") as handle:
            subprocess.run(
                command,
                cwd=paths["root"],
                check=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
    py = pd.read_csv(python_output, sep="\t")
    r_result = pd.read_csv(r_output, sep="\t")
    merged = py[["gene", "distance", "rank", "rank_fraction"]].merge(
        r_result[["gene", "distance", "rank", "rank_fraction"]],
        on="gene",
        suffixes=("_python", "_r"),
        validate="one_to_one",
    )
    merged = merged.loc[~merged["gene"].astype(str).eq(target)].copy()
    rho = spearmanr(merged["distance_python"], merged["distance_r"])
    n_top = max(1, int(np.ceil(len(merged) * 0.05)))
    top_python = set(merged.nsmallest(n_top, "rank_python")["gene"])
    top_r = set(merged.nsmallest(n_top, "rank_r")["gene"])
    union = top_python | top_r
    comparison = {
        "target_gene": target,
        "n_shared_genes_excluding_target": int(len(merged)),
        "spearman_distance": float(rho.statistic),
        "spearman_pvalue_model_descriptive": float(rho.pvalue),
        "top_5pct_jaccard": float(len(top_python & top_r) / len(union))
        if union
        else np.nan,
        "python_implementation": "scTenifoldpy 0.4.0; explicit CPM before model",
        "r_implementation": "official scTenifoldKnk 1.1; internal CPM",
        "interpretation": (
            "implementation sensitivity only; disagreement does not constitute a "
            "biological-replicate test"
        ),
        "completed_at_utc": _utc_now(),
    }
    _write_tsv(merged, out / "r_python_gene_rank_comparison.tsv.gz")
    _write_json(comparison, comparison_path)
    adata.file.close()
    logger.info(
        "R/Python cross-check complete: rho=%.3f, top5%% Jaccard=%.3f",
        comparison["spearman_distance"],
        comparison["top_5pct_jaccard"],
    )


def run_network_queue(
    config: Mapping[str, Any], stage_root: Path, paths: Mapping[str, Path], logger: Any
) -> None:
    definitions = _scenario_definitions(config, stage_root / "00_preparation")
    manifest = pd.DataFrame(definitions)
    _write_tsv(manifest, stage_root / "00_preparation" / "run_queue.tsv")
    for definition in definitions:
        _run_definition(config, definition, stage_root, paths, logger)


def _read_knockout_results(stage_root: Path) -> pd.DataFrame:
    files = sorted((stage_root / "01_runs").glob("*/seed_*/knockouts/*.tsv.gz"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(path, sep="\t") for path in files]
    return pd.concat(frames, ignore_index=True)


def _summarize_gene_stability(frame: pd.DataFrame, top_fraction: float) -> pd.DataFrame:
    evaluated = frame.loc[~frame["is_knockout_target"].astype(bool)].copy()
    evaluated["top_1pct"] = evaluated["rank_fraction"].le(0.01)
    evaluated["top_fraction"] = evaluated["rank_fraction"].le(top_fraction)
    grouped = evaluated.groupby(
        ["scenario", "ko_target", "ko_role", "gene"], observed=True
    )
    summary = grouped.agg(
        n_seeds=("seed", "nunique"),
        median_distance=("distance", "median"),
        median_rank=("rank", "median"),
        median_rank_fraction=("rank_fraction", "median"),
        top_1pct_frequency=("top_1pct", "mean"),
        top_fraction_frequency=("top_fraction", "mean"),
        model_fdr05_frequency=("model_padj", lambda x: float(pd.to_numeric(x).lt(0.05).mean())),
    ).reset_index()
    summary["stable_top_fraction"] = summary["top_fraction_frequency"].ge(2 / 3)
    return summary


def _target_stability(
    frame: pd.DataFrame, gene_summary: pd.DataFrame, top_fraction: float
) -> pd.DataFrame:
    evaluated = frame.loc[~frame["is_knockout_target"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for (scenario, target, role), block in evaluated.groupby(
        ["scenario", "ko_target", "ko_role"], observed=True
    ):
        top_sets = [
            set(seed_block.loc[seed_block["rank_fraction"].le(top_fraction), "gene"].astype(str))
            for _, seed_block in block.groupby("seed", observed=True)
        ]
        stable = gene_summary.loc[
            gene_summary["scenario"].astype(str).eq(str(scenario))
            & gene_summary["ko_target"].astype(str).eq(str(target))
            & gene_summary["stable_top_fraction"].astype(bool)
        ]
        rows.append(
            {
                "scenario": scenario,
                "ko_target": target,
                "ko_role": role,
                "n_seeds": int(block["seed"].nunique()),
                "median_pairwise_top_fraction_jaccard": pairwise_jaccard(top_sets),
                "n_stable_top_fraction_genes": int(len(stable)),
                "median_stable_gene_distance": float(stable["median_distance"].median())
                if len(stable)
                else np.nan,
                "top_fraction": float(top_fraction),
            }
        )
    return pd.DataFrame(rows)


def _module_enrichment(
    gene_summary: pd.DataFrame,
    modules: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scenario, target, role), block in gene_summary.groupby(
        ["scenario", "ko_target", "ko_role"], observed=True
    ):
        universe = set(block["gene"].astype(str))
        hits = set(block.loc[block["stable_top_fraction"].astype(bool), "gene"].astype(str))
        for module, genes in modules.items():
            module_set = set(map(str, genes)) & universe
            a = len(hits & module_set)
            b = len(hits - module_set)
            c = len(module_set - hits)
            d = len(universe - hits - module_set)
            odds, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append(
                {
                    "scenario": scenario,
                    "ko_target": target,
                    "ko_role": role,
                    "module": module,
                    "network_genes_in_module": len(module_set),
                    "stable_hits_in_module": a,
                    "stable_hit_count": len(hits),
                    "odds_ratio": float(odds),
                    "fisher_pvalue": float(pvalue),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fisher_padj_within_scenario_target"] = result.groupby(
            ["scenario", "ko_target"], observed=True
        )["fisher_pvalue"].transform(lambda x: _bh_adjust(x.to_numpy()))
    return result


def _transcriptomic_alignment(
    gene_summary: pd.DataFrame, root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = root / "results/de_stage1_5/Granulosa/rescue_ready_effects.tsv.gz"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    evidence = pd.read_csv(path, sep="\t")
    columns = [
        "gene",
        "aging_effect",
        "treatment_effect",
        "residual_effect",
        "recovery_fraction",
        "directional_rescue_candidate",
        "FDR_supported_rescue_candidate",
    ]
    evidence = evidence[[column for column in columns if column in evidence]].copy()
    merged = gene_summary.merge(evidence, on="gene", how="left", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    for (scenario, target, role), block in merged.groupby(
        ["scenario", "ko_target", "ko_role"], observed=True
    ):
        stable = block["stable_top_fraction"].fillna(False).astype(bool)
        row: dict[str, Any] = {
            "scenario": scenario,
            "ko_target": target,
            "ko_role": role,
            "n_genes_with_transcriptomic_evidence": int(block["treatment_effect"].notna().sum())
            if "treatment_effect" in block
            else 0,
            "stable_hit_count": int(stable.sum()),
        }
        for column in ["aging_effect", "treatment_effect", "recovery_fraction"]:
            if column not in block:
                continue
            x = pd.to_numeric(block["median_distance"], errors="coerce")
            y = pd.to_numeric(block[column], errors="coerce").abs()
            valid = x.notna() & y.notna()
            correlation = spearmanr(x[valid], y[valid]) if valid.sum() >= 3 else None
            row[f"spearman_distance_vs_abs_{column}"] = (
                float(correlation.statistic) if correlation is not None else np.nan
            )
            row[f"spearman_pvalue_distance_vs_abs_{column}"] = (
                float(correlation.pvalue) if correlation is not None else np.nan
            )
        for column in ["directional_rescue_candidate", "FDR_supported_rescue_candidate"]:
            if column in block:
                values = block[column].fillna(False).astype(bool)
                row[f"stable_hits_overlapping_{column}"] = int((stable & values).sum())
        rows.append(row)
    return pd.DataFrame(rows), merged


def _loo_stability(frame: pd.DataFrame, top_fraction: float) -> pd.DataFrame:
    primary = frame.loc[frame["scenario"].astype(str).eq(PRIMARY_SCENARIO)].copy()
    loo = frame.loc[frame["scenario"].astype(str).str.startswith("loo_exclude_")].copy()
    if primary.empty or loo.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for target in sorted(set(primary["ko_target"]) & set(loo["ko_target"])):
        full_sets = [
            set(block.loc[block["rank_fraction"].le(top_fraction), "gene"].astype(str))
            for _, block in primary.loc[primary["ko_target"].eq(target)].groupby(
                "seed", observed=True
            )
        ]
        full_consensus = set.intersection(*full_sets) if full_sets else set()
        for scenario, block in loo.loc[loo["ko_target"].eq(target)].groupby(
            "scenario", observed=True
        ):
            loo_set = set(block.loc[block["rank_fraction"].le(top_fraction), "gene"].astype(str))
            union = full_consensus | loo_set
            rows.append(
                {
                    "ko_target": target,
                    "ko_role": str(block["ko_role"].iloc[0]),
                    "excluded_library": str(scenario).replace("loo_exclude_", ""),
                    "full_consensus_hit_count": len(full_consensus),
                    "loo_hit_count": len(loo_set),
                    "jaccard_vs_full_seed_intersection": len(full_consensus & loo_set) / len(union)
                    if union
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_stage24(
    config: Mapping[str, Any], stage_root: Path, paths: Mapping[str, Path], logger: Any
) -> None:
    cfg = config["stage24_virtual_knockout"]
    out = stage_root / "02_summary"
    raw = _read_knockout_results(stage_root)
    if raw.empty:
        raise RuntimeError("No virtual-knockout results are available to summarize")
    top_fraction = float(cfg["stability_top_fraction"])
    gene_summary = _summarize_gene_stability(raw, top_fraction)
    target_summary = _target_stability(raw, gene_summary, top_fraction)
    modules = _module_genes(paths["root"] / str(cfg["state_modules_file"]))
    module_summary = _module_enrichment(gene_summary, modules)
    transcript_summary, merged = _transcriptomic_alignment(gene_summary, paths["root"])
    loo = _loo_stability(raw, top_fraction)
    references = pd.read_csv(
        stage_root / "00_preparation/expression_matched_reference_genes.tsv", sep="\t"
    )
    comparisons: list[dict[str, Any]] = []
    indexed = target_summary.set_index(["scenario", "ko_target"])
    for row in references.itertuples(index=False):
        for scenario in target_summary["scenario"].astype(str).unique():
            left = (scenario, str(row.target_gene))
            right = (scenario, str(row.reference_gene))
            if left not in indexed.index or right not in indexed.index:
                continue
            candidate = indexed.loc[left]
            reference = indexed.loc[right]
            comparisons.append(
                {
                    "scenario": scenario,
                    "target_gene": row.target_gene,
                    "reference_gene": row.reference_gene,
                    "candidate_stable_hit_count": candidate["n_stable_top_fraction_genes"],
                    "reference_stable_hit_count": reference["n_stable_top_fraction_genes"],
                    "stable_hit_count_difference": candidate["n_stable_top_fraction_genes"]
                    - reference["n_stable_top_fraction_genes"],
                    "candidate_seed_jaccard": candidate[
                        "median_pairwise_top_fraction_jaccard"
                    ],
                    "reference_seed_jaccard": reference[
                        "median_pairwise_top_fraction_jaccard"
                    ],
                    "seed_jaccard_difference": candidate[
                        "median_pairwise_top_fraction_jaccard"
                    ]
                    - reference["median_pairwise_top_fraction_jaccard"],
                }
            )
    _write_tsv(gene_summary, out / "gene_level_stability.tsv.gz")
    _write_tsv(target_summary, out / "knockout_target_stability.tsv")
    _write_tsv(module_summary, out / "module_enrichment.tsv")
    _write_tsv(transcript_summary, out / "transcriptomic_alignment_summary.tsv")
    _write_tsv(merged, out / "gene_stability_with_real_transcriptomic_evidence.tsv.gz")
    _write_tsv(loo, out / "leave_one_library_out_stability.tsv")
    _write_tsv(pd.DataFrame(comparisons), out / "candidate_vs_matched_reference.tsv")
    primary = target_summary.loc[target_summary["scenario"].astype(str).eq(PRIMARY_SCENARIO)]
    report = [
        "# Stage 24 virtual-knockout pilot",
        "",
        "## Scope and interpretation",
        "",
        (
            "This is a library-balanced scTenifoldKnk network-perturbation analysis "
            "of OC Granulosa cells."
        ),
        (
            "The gene-level probabilities are internal model-ranking quantities, not "
            "tests across the three biological libraries. Virtual knockout is used to "
            "prioritize hypotheses and does not establish causality."
        ),
        "",
        "## Primary multi-seed stability",
        "",
        (
            primary.to_markdown(index=False)
            if not primary.empty
            else "Primary scenario is incomplete."
        ),
        "",
        "## Output guide",
        "",
        (
            "- `gene_level_stability.tsv.gz`: median perturbation and top-rank "
            "recurrence across seeds."
        ),
        "- `knockout_target_stability.tsv`: per-target seed concordance and stable-hit counts.",
        "- `candidate_vs_matched_reference.tsv`: expression-matched benchmark comparison.",
        "- `module_enrichment.tsv`: descriptive over-representation of stable perturbation hits.",
        (
            "- `transcriptomic_alignment_summary.tsv`: association with absolute real "
            "OC/Y and OT/OC effects; scTenifold distance itself is unsigned."
        ),
        "- `leave_one_library_out_stability.tsv`: sensitivity to removing one OC library.",
        "",
        f"Generated: {_utc_now()}",
    ]
    (out / "STAGE24_VIRTUAL_KNOCKOUT_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    logger.info("Stage 24 summary written: %d raw rows", len(raw))


def _configure_figures() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save_figure(fig: mpl.figure.Figure, base: Path, cfg: Mapping[str, Any]) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    formats = set(map(str, cfg.get("figure_formats", ["svg", "pdf", "png"])))
    if "svg" in formats:
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    if "pdf" in formats:
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    if "png" in formats:
        fig.savefig(
            base.with_suffix(".png"),
            dpi=int(cfg.get("png_dpi", 600)),
            bbox_inches="tight",
        )
    if "tiff" in formats:
        fig.savefig(
            base.with_suffix(".tiff"),
            dpi=600,
            bbox_inches="tight",
        )
    plt.close(fig)


def make_stage24_figures(
    config: Mapping[str, Any], stage_root: Path, figure_root: Path, logger: Any
) -> None:
    cfg = config["stage24_virtual_knockout"]
    summary_root = stage_root / "02_summary"
    targets = pd.read_csv(summary_root / "knockout_target_stability.tsv", sep="\t")
    comparisons = pd.read_csv(summary_root / "candidate_vs_matched_reference.tsv", sep="\t")
    loo_path = summary_root / "leave_one_library_out_stability.tsv"
    loo = (
        pd.read_csv(loo_path, sep="\t")
        if loo_path.exists() and loo_path.stat().st_size
        else pd.DataFrame()
    )
    _configure_figures()
    palette = {TARGET_GENE_ROLE: "#C75B39", REFERENCE_GENE_ROLE: "#8A8A8A"}
    source_rows: list[pd.DataFrame] = []
    for scenario, block in targets.groupby("scenario", observed=True):
        block = block.sort_values(["ko_role", "ko_target"]).copy()
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        x = np.arange(len(block))
        colors = [palette.get(str(role), "#777777") for role in block["ko_role"]]
        ax.scatter(
            x,
            block["median_pairwise_top_fraction_jaccard"],
            s=34,
            c=colors,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.set_xticks(x, block["ko_target"], rotation=45, ha="right")
        ax.set_ylabel("Median pairwise top-5% Jaccard")
        ax.set_title(f"Virtual-KO seed stability — {scenario}", loc="left", fontweight="bold")
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        _save_figure(fig, figure_root / f"seed_stability_{scenario}", cfg)
        source = block[
            ["scenario", "ko_target", "ko_role", "median_pairwise_top_fraction_jaccard"]
        ].copy()
        source["figure"] = f"seed_stability_{scenario}"
        source_rows.append(source)
    for scenario, block in comparisons.groupby("scenario", observed=True):
        block = block.sort_values("target_gene").copy()
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        x = np.arange(len(block))
        ax.scatter(
            x - 0.08,
            block["candidate_stable_hit_count"],
            s=30,
            color="#C75B39",
            label="Candidate",
        )
        ax.scatter(
            x + 0.08,
            block["reference_stable_hit_count"],
            s=30,
            color="#8A8A8A",
            label="Expression-matched reference",
        )
        for position, row in enumerate(block.itertuples(index=False)):
            ax.plot(
                [position - 0.08, position + 0.08],
                [row.candidate_stable_hit_count, row.reference_stable_hit_count],
                color="#BDBDBD",
                linewidth=0.7,
                zorder=0,
            )
        ax.set_xticks(x, block["target_gene"], rotation=45, ha="right")
        ax.set_ylabel("Stable top-5% perturbed genes")
        ax.set_title(
            f"Candidate versus matched reference — {scenario}",
            loc="left",
            fontweight="bold",
        )
        ax.legend(loc="best")
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
        fig.tight_layout()
        _save_figure(fig, figure_root / f"candidate_reference_{scenario}", cfg)
        source = block.copy()
        source["figure"] = f"candidate_reference_{scenario}"
        source_rows.append(source)
    if not loo.empty:
        plot = loo.loc[loo["ko_role"].astype(str).eq(TARGET_GENE_ROLE)].copy()
        plot = plot.sort_values(["ko_target", "excluded_library"])
        fig, ax = plt.subplots(figsize=(4.5, 2.9))
        genes = sorted(plot["ko_target"].astype(str).unique())
        libraries = sorted(plot["excluded_library"].astype(str).unique())
        offsets = np.linspace(-0.18, 0.18, max(len(libraries), 1))
        for offset, library in zip(offsets, libraries, strict=True):
            local = plot.loc[plot["excluded_library"].astype(str).eq(library)].set_index(
                "ko_target"
            )
            y = [local.loc[gene, "jaccard_vs_full_seed_intersection"] for gene in genes]
            ax.scatter(
                np.arange(len(genes)) + offset,
                y,
                s=28,
                label=f"Exclude {library}",
            )
        ax.set_xticks(np.arange(len(genes)), genes, rotation=45, ha="right")
        ax.set_ylabel("Jaccard versus full OC consensus")
        ax.set_title("Leave-one-library-out robustness", loc="left", fontweight="bold")
        ax.legend(loc="best", ncol=1)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        _save_figure(fig, figure_root / "leave_one_library_out_robustness", cfg)
        plot["figure"] = "leave_one_library_out_robustness"
        source_rows.append(plot)
    if source_rows:
        _write_tsv(
            pd.concat(source_rows, ignore_index=True),
            figure_root / "source_data/figure_source_data.tsv",
        )
    contract = {
        "core_conclusion": (
            "Candidate virtual knockouts are compared by multi-seed and leave-one-library-out "
            "stability, with expression-matched genes as descriptive references."
        ),
        "evidence_logic": [
            "seed concordance",
            "candidate-versus-expression-matched reference",
            "leave-one-library-out robustness",
        ],
        "archetype": "independent quantitative plots; no assembled multi-panel figure",
        "backend": "Python matplotlib only",
        "final_size": (
            "candidate/reference and LOO plots: 89 mm single-column; "
            "ten-target seed-stability plots: 183 mm double-column"
        ),
        "exports": list(map(str, cfg.get("figure_formats", []))),
        "review_risks": [
            "scTenifold perturbation distance is unsigned",
            "model-internal p-values are not library-level inference",
            "virtual knockout does not establish causality",
        ],
    }
    _write_json(contract, figure_root / "FIGURE_CONTRACT.json")
    logger.info("Stage 24 independent figures complete")


def run_stage24_virtual_knockout(
    config: Mapping[str, Any], selected_steps: Sequence[str] | None = None
) -> None:
    stage_root, figure_root, paths = _stage_paths(config)
    logger = setup_logging(
        "35_stage24_virtual_knockout", dict(config), output_dir=stage_root / "logs"
    )
    steps = list(
        selected_steps
        or ["prepare", "r_crosscheck", "network_queue", "summarize", "figures"]
    )
    functions = {
        "prepare": lambda: prepare_stage24(config, stage_root, paths, logger),
        "r_crosscheck": lambda: run_r_crosscheck(config, stage_root, paths, logger),
        "network_queue": lambda: run_network_queue(config, stage_root, paths, logger),
        "summarize": lambda: summarize_stage24(config, stage_root, paths, logger),
        "figures": lambda: make_stage24_figures(config, stage_root, figure_root, logger),
    }
    unknown = sorted(set(steps) - set(functions))
    if unknown:
        raise ValueError(f"Unknown Stage 24 steps: {unknown}")
    _write_json(
        {
            "stage_version": STAGE_VERSION,
            "output_isolation": str(stage_root),
            "input_is_read_only": True,
            "statistical_unit": "library/pool",
            "selected_steps": steps,
            "environment": _environment_manifest(),
        },
        stage_root / "RUN_MANIFEST.json",
    )
    for step in steps:
        _update_state(stage_root, step, "running")
        try:
            functions[step]()
        except Exception as exc:
            _record_failure(stage_root, step, exc, logger)
            _update_state(stage_root, step, "failed", error=repr(exc))
            raise
        _update_state(stage_root, step, "complete")
    logger.info("Requested Stage 24 steps complete: %s", ", ".join(steps))
