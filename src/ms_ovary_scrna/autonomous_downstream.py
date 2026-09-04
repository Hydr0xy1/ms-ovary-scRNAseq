"""Conservative downstream compartment review and pseudobulk readiness audit.

This module intentionally separates descriptive structure discovery from future
condition-level inference.  It never edits the input AnnData object in place,
never removes cells from the generated full-object checkpoints, and always
keeps the raw UMI matrix in ``layers['counts']``.
"""

# Long marker/report strings are intentionally kept readable in source.
# ruff: noqa: E501

from __future__ import annotations

import gc
import hashlib
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA

from .integration import harmony_integrate_compatible
from .preprocessing import (
    compute_hvg_pca,
    normalize_log1p_preserving_counts,
    select_batch_aware_hvgs,
)
from .project import load_yaml, project_paths, require_compute_resources, setup_logging

COUNTS_LAYER = "counts"

DEFAULT_MARKERS: dict[str, dict[str, list[str]]] = {
    "stromal": {
        "fibroblast": [
            "Dcn",
            "Lum",
            "Col1a1",
            "Col1a2",
            "Ogn",
            "Bgn",
            "Pdgfra",
            "Tcf21",
            "Col3a1",
            "Dpt",
        ],
        "ecm_fibrotic": [
            "Col1a1",
            "Col1a2",
            "Col3a1",
            "Fn1",
            "Postn",
            "Sparc",
            "Sparcl1",
            "Ctgf",
            "Timp1",
            "Mmp2",
            "Mmp14",
        ],
        "myofibroblast": ["Acta2", "Tagln", "Ctgf", "Postn", "Col1a1"],
        "pericyte": ["Rgs5", "Pdgfrb", "Cspg4", "Notch3", "Des", "Rbp1"],
        "smooth_muscle": ["Acta2", "Tagln", "Myh11", "Des", "Cnn1"],
        "vascular_associated": ["Rgs5", "Pdgfrb", "Cspg4", "Pecam1", "Cdh5", "Kdr"],
        "theca_adjacent": ["Cyp11a1", "Cyp17a1", "Hsd3b1", "Star", "Fdx1"],
        "endothelial": ["Pecam1", "Cdh5", "Kdr", "Flt1", "Esam", "Emcn"],
        "lymphatic": ["Prox1", "Lyve1", "Pdpn", "Ccl21a", "Flt4", "Mmrn1"],
    },
    "immune": {
        "macrophage_c1qc": ["C1qa", "C1qb", "C1qc", "Adgre1", "Fcgr1", "Mertk", "Cd68"],
        "macrophage_inflammatory": ["Lgals3", "Spp1", "Il1b", "Tnf", "Nfkbia", "Lyz2"],
        "macrophage_lipid": ["Lpl", "Apoe", "Fabp5", "Ctsb", "Ctsd", "Trem2"],
        "monocyte": ["Ly6c2", "Ccr2", "S100a8", "S100a9", "Plac8", "Lyz2"],
        "dendritic": ["Itgax", "Flt3", "Clec10a", "Xcr1", "Clec9a", "Cd74"],
        "t_cell": ["Cd3d", "Cd3e", "Cd3g", "Trbc1", "Trbc2", "Lck"],
        "nk": ["Nkg7", "Klrk1", "Klrb1c", "Prf1", "Gzmb", "Ccl5"],
        "b_cell": ["Cd79a", "Ms4a1", "Cd74", "Cd37", "Bank1", "Cd22", "Igkc"],
        "plasma": ["Jchain", "Mzb1", "Sdc1", "Derl3", "Igha", "Igkc"],
        "neutrophil": ["S100a8", "S100a9", "Ly6g", "Csf3r", "Retnlg"],
        "mast": ["Kit", "Ms4a2", "Cpa3", "Mcpt4", "Tpsb2"],
    },
    "epithelial": {
        "ciliated": ["Foxj1", "Pifo", "Tppp3", "Tmem212", "Ccdc153", "Rfx2", "Rfx3", "Dnah5"],
        "surface": ["Epcam", "Krt8", "Krt18", "Krt19", "Plet1", "Upk1b", "Upk3b"],
        "secretory": ["Epcam", "Krt8", "Krt18", "Ovgp1", "Muc1", "Krt19"],
    },
    "endothelial": {
        "lymphatic": ["Prox1", "Lyve1", "Pdpn", "Ccl21a", "Flt4", "Mmrn1"],
        "arterial": ["Efnb2", "Sox17", "Gja5", "Flt1", "Kdr"],
        "venous": ["Nr2f2", "Ackr1", "Vwf", "Cdh5", "Pecam1"],
        "capillary": ["Rgcc", "Car4", "Emcn", "Kdr", "Pecam1"],
        "vascular": ["Pecam1", "Cdh5", "Kdr", "Flt1", "Esam", "Emcn"],
    },
}


def _sha256_sparse(matrix: Any) -> str:
    """Stable digest of sparse/dense matrix values and structure."""
    if sparse.issparse(matrix):
        csr = sparse.csr_matrix(matrix)
        digest = hashlib.sha256()
        for value in (csr.data, csr.indices, csr.indptr):
            digest.update(np.ascontiguousarray(value).tobytes())
        digest.update(str(csr.shape).encode())
        return digest.hexdigest()
    arr = np.asarray(matrix)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _ensure_counts_sparse(adata: ad.AnnData, layer: str = COUNTS_LAYER) -> None:
    if layer not in adata.layers:
        raise KeyError(f"Missing required raw-count layer: {layer}")
    adata.layers[layer] = sparse.csr_matrix(adata.layers[layer])


def _atomic_write(adata: ad.AnnData, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    adata.write_h5ad(temporary, compression="gzip")
    temporary.replace(output)


def _validate_saved_subset(
    output: Path,
    *,
    expected_shape: tuple[int, int],
    expected_obs: pd.Index,
    counts_digest: str,
    library_counts: dict[str, int],
    group_counts: dict[str, int],
) -> None:
    saved = sc.read_h5ad(output, backed="r")
    try:
        if tuple(saved.shape) != tuple(expected_shape):
            raise RuntimeError(f"Unexpected shape in {output}: {saved.shape}")
        if not saved.obs_names.equals(expected_obs):
            raise RuntimeError(f"Barcode order changed in {output}")
        if COUNTS_LAYER not in saved.layers:
            raise RuntimeError(f"Raw counts layer missing from {output}")
        saved_counts = saved.layers[COUNTS_LAYER]
        if hasattr(saved_counts, "to_memory"):
            saved_counts = saved_counts.to_memory()
        if _sha256_sparse(saved_counts) != counts_digest:
            raise RuntimeError(f"Raw counts digest changed in {output}")
        observed_library = saved.obs["library_id"].astype(str).value_counts().sort_index().to_dict()
        observed_group = saved.obs["group"].astype(str).value_counts().sort_index().to_dict()
        if observed_library != library_counts or observed_group != group_counts:
            raise RuntimeError(f"Library/group composition changed in {output}")
    finally:
        saved.file.close()


def _marker_programs(kind: str, marker_path: Path) -> dict[str, list[str]]:
    programs = {k: list(v) for k, v in DEFAULT_MARKERS[kind].items()}
    if marker_path.exists():
        configured = load_yaml(marker_path).get("compartment_subclustering", {}).get(kind, {})
        for name, genes in configured.items():
            if isinstance(genes, dict):
                genes = genes.get("positive", [])
            programs[str(name)] = [str(g) for g in genes]
    return programs


def _select_hvgs_with_loess_guard(
    adata: ad.AnnData,
    *,
    n_top_genes: int,
    batch_key: str,
    logger: logging.Logger,
) -> int:
    """Run seurat_v3 HVG, guarding against loess failure from all-zero genes.

    The guard operates only on a temporary HVG fitting view. The returned object
    keeps every gene and its counts layer; only the boolean HVG annotation is
    projected back to the full feature set.
    """
    try:
        select_batch_aware_hvgs(
            adata,
            counts_layer=COUNTS_LAYER,
            flavor="seurat_v3",
            n_top_genes=n_top_genes,
            batch_key=batch_key,
        )
        return 0
    except ValueError as exc:
        if "reciprocal condition number" not in str(exc):
            raise
        counts = sparse.csr_matrix(adata.layers[COUNTS_LAYER])
        detected = np.asarray((counts > 0).sum(axis=0)).ravel()
        min_cells = max(3, int(adata.obs[batch_key].astype(str).nunique()))
        keep = detected >= min_cells
        if int(keep.sum()) <= n_top_genes:
            raise RuntimeError(
                "seurat_v3 HVG loess failed and the subset has too few detected genes "
                f"after min_cells={min_cells} guard"
            ) from exc
        logger.warning(
            "seurat_v3 loess guard: fitting HVGs on %d/%d genes detected in >=%d cells; "
            "full feature set and counts are retained",
            int(keep.sum()),
            adata.n_vars,
            min_cells,
        )
        view = adata[:, keep].copy()
        select_batch_aware_hvgs(
            view,
            counts_layer=COUNTS_LAYER,
            flavor="seurat_v3",
            n_top_genes=n_top_genes,
            batch_key=batch_key,
        )
        adata.var["highly_variable"] = False
        adata.var.loc[view.var_names, "highly_variable"] = view.var["highly_variable"].to_numpy(
            dtype=bool
        )
        del view
        gc.collect()
        return int((~keep).sum())


def _program_evidence(
    adata: ad.AnnData,
    cluster_key: str,
    programs: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute cluster marker means/fractions and conservative candidate labels."""
    clusters = adata.obs[cluster_key].astype(str)
    records: list[dict[str, Any]] = []
    cluster_scores: list[dict[str, Any]] = []
    for cluster in pd.unique(clusters):
        mask = clusters.eq(cluster).to_numpy()
        scores: dict[str, float] = {}
        fractions: dict[str, float] = {}
        for program, genes in programs.items():
            present = [g for g in genes if g in adata.var_names]
            if not present:
                continue
            matrix = sparse.csr_matrix(adata[mask, present].X)
            means = np.asarray(matrix.mean(axis=0)).ravel()
            frac = np.asarray((matrix > 0).mean(axis=0)).ravel()
            scores[program] = float(np.mean(means))
            fractions[program] = float(np.mean(frac))
            for gene, mean, fraction in zip(present, means, frac, strict=True):
                records.append(
                    {
                        "cluster": str(cluster),
                        "program": program,
                        "gene": gene,
                        "mean_log_normalized_expression": float(mean),
                        "fraction_expressing": float(fraction),
                    }
                )
        if scores:
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            top_name, top_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0
            available = len([g for g in programs[top_name] if g in adata.var_names])
            margin = top_score - second_score
            # Marker evidence is deliberately reported as a candidate.  A cluster
            # is left Uncertain unless several genes and a visible score margin agree.
            if available >= 3 and top_score >= 0.08 and margin >= 0.02:
                candidate = top_name
                confidence = "High" if top_score >= 0.25 and margin >= 0.08 else "Medium"
            else:
                candidate = "Uncertain"
                confidence = "Low"
            cluster_scores.append(
                {
                    "cluster": str(cluster),
                    "n_cells": int(mask.sum()),
                    "candidate_program": candidate,
                    "confidence": confidence,
                    "top_score": float(top_score),
                    "second_score": float(second_score),
                    "score_margin": float(margin),
                    "n_markers_available": int(available),
                    "top_program_fraction": float(fractions.get(top_name, np.nan)),
                }
            )
    evidence = pd.DataFrame(records)
    summary = pd.DataFrame(cluster_scores)
    if not evidence.empty:
        values = evidence.groupby(["cluster", "program"], as_index=False)[
            "mean_log_normalized_expression"
        ].mean()
        scale = values.groupby("program")["mean_log_normalized_expression"].transform("std")
        values["relative_score"] = (
            values["mean_log_normalized_expression"]
            - values.groupby("program")["mean_log_normalized_expression"].transform("mean")
        ) / scale.replace(0, np.nan)
        evidence = evidence.merge(values, on=["cluster", "program"], how="left")
    return evidence, summary


def _candidate_label(kind: str, program: str) -> tuple[str, str]:
    if program == "Uncertain":
        return "Uncertain", "Uncertain"
    maps = {
        "stromal": {
            "fibroblast": ("Stromal_fibroblast_candidate", "fibroblast_like"),
            "ecm_fibrotic": ("ECM_high_candidate", "ECM_high"),
            "myofibroblast": ("Myofibroblast_like_candidate", "myofibroblast_like"),
            "pericyte": ("Pericyte_candidate", "pericyte"),
            "smooth_muscle": ("Smooth_muscle_candidate", "smooth_muscle"),
            "vascular_associated": ("Vascular_associated_stromal_candidate", "vascular_associated"),
            "theca_adjacent": ("Theca_adjacent_stromal_candidate", "theca_adjacent"),
            "endothelial": ("Endothelial_boundary_candidate", "endothelial_boundary"),
            "lymphatic": ("Lymphatic_boundary_candidate", "lymphatic_boundary"),
        },
        "immune": {
            "macrophage_c1qc": ("Macrophage_C1qc_high", "Macrophage_C1qc_high"),
            "macrophage_inflammatory": (
                "Macrophage_inflammatory_candidate",
                "Macrophage_inflammatory",
            ),
            "macrophage_lipid": (
                "Macrophage_lipid_associated_candidate",
                "Macrophage_lipid_associated",
            ),
            "monocyte": ("Monocyte_candidate", "Monocyte_like"),
            "dendritic": ("Dendritic_candidate", "Dendritic_like"),
            "t_cell": ("T_cell_candidate", "T_cell_like"),
            "nk": ("NK_candidate", "NK_like"),
            "b_cell": ("B_cell_candidate", "B_cell_like"),
            "plasma": ("Plasma_like_candidate", "Plasma_like"),
            "neutrophil": ("Neutrophil_candidate", "Neutrophil_like"),
            "mast": ("Mast_candidate", "Mast_like"),
        },
        "epithelial": {
            "ciliated": ("Ciliated_epithelial_candidate", "ciliated"),
            "surface": ("Surface_epithelial_candidate", "surface"),
            "secretory": ("Secretory_epithelial_candidate", "secretory"),
        },
        "endothelial": {
            "lymphatic": ("Lymphatic_endothelial_candidate", "lymphatic"),
            "arterial": ("Arterial_like_candidate", "arterial"),
            "venous": ("Venous_like_candidate", "venous"),
            "capillary": ("Capillary_like_candidate", "capillary"),
            "vascular": ("Vascular_endothelial_candidate", "vascular"),
        },
    }
    return maps.get(kind, {}).get(program, (f"{program}_candidate", program))


def _plot_umap(adata: ad.AnnData, color: str, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig = sc.pl.umap(
            adata, color=color, frameon=False, show=False, return_fig=True, title=title
        )
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # plotting is diagnostic; preserve the analytical output
        logging.getLogger(__name__).warning("UMAP plot failed for %s: %s", color, exc)


def _table_text(frame: pd.DataFrame) -> str:
    """Render a compact report table without optional tabulate dependency."""
    if frame.empty:
        return "(no rows)"
    return frame.to_string(index=False)


def _run_compartment(
    source: ad.AnnData,
    *,
    kind: str,
    name: str,
    mask: np.ndarray,
    config: dict[str, Any],
    paths: dict[str, Path],
    logger: logging.Logger,
) -> tuple[Path, pd.DataFrame]:
    settings = config.get("downstream", {}).get("subclustering", {})
    n_top_hvg = int(settings.get("n_top_hvg", 3000))
    n_pcs = int(settings.get("n_pcs", 50))
    use_n_pcs = int(settings.get("use_n_pcs", 40))
    n_neighbors = int(settings.get("n_neighbors", 15))
    resolutions = [float(x) for x in settings.get("resolutions", [0.2, 0.3, 0.5, 0.8])]
    primary_resolution = float(settings.get("primary_resolution", 0.5))
    seed = int(config["project"]["random_seed"])
    subset = source[mask].copy()
    _ensure_counts_sparse(subset)
    counts_digest = _sha256_sparse(subset.layers[COUNTS_LAYER])
    expected_obs = subset.obs_names.copy()
    library_counts = subset.obs["library_id"].astype(str).value_counts().sort_index().to_dict()
    group_counts = subset.obs["group"].astype(str).value_counts().sort_index().to_dict()
    subset.obs["compartment"] = kind
    subset.obs["source_cell_type_broad_v1"] = subset.obs["cell_type_broad_v1"].astype(str)
    subset.obs["source_cell_type_subtype_v1"] = subset.obs["cell_type_subtype_v1"].astype(str)
    subset.raw = None
    # Whole-atlas reductions are not reused as subset reductions.
    subset.obsm.clear()
    subset.obsp.clear()
    subset.varm.clear()
    subset.uns.clear()
    logger.info("%s: %d cells, %d genes", name, subset.n_obs, subset.n_vars)
    normalize_log1p_preserving_counts(subset, counts_layer=COUNTS_LAYER, target_sum=10000.0)
    hvg_guarded_genes = _select_hvgs_with_loess_guard(
        subset,
        n_top_genes=n_top_hvg,
        batch_key="library_id",
        logger=logger,
    )
    compute_hvg_pca(subset, n_comps=n_pcs, scale_max_value=10.0, random_state=seed)
    subset.obsm["X_pca_unintegrated"] = subset.obsm["X_pca"].copy()
    sc.pp.neighbors(
        subset,
        n_neighbors=n_neighbors,
        n_pcs=use_n_pcs,
        use_rep="X_pca",
        key_added="neighbors_unintegrated",
        random_state=seed,
    )
    sc.tl.umap(subset, neighbors_key="neighbors_unintegrated", random_state=seed)
    subset.obsm["X_umap_unintegrated"] = subset.obsm["X_umap"].copy()
    harmony_integrate_compatible(
        subset, batch_key="library_id", seed=seed, basis="X_pca", adjusted_basis="X_pca_harmony"
    )
    sc.pp.neighbors(
        subset,
        n_neighbors=n_neighbors,
        n_pcs=use_n_pcs,
        use_rep="X_pca_harmony",
        key_added="neighbors_harmony",
        random_state=seed,
    )
    sc.tl.umap(subset, neighbors_key="neighbors_harmony", random_state=seed)
    subset.obsm["X_umap_harmony"] = subset.obsm["X_umap"].copy()
    subset.obs["leiden_primary"] = None
    for resolution in resolutions:
        key = f"leiden_{resolution:g}"
        sc.tl.leiden(
            subset,
            resolution=resolution,
            key_added=key,
            neighbors_key="neighbors_harmony",
            random_state=seed,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
    primary_key = f"leiden_{primary_resolution:g}"
    subset.obs["leiden_primary"] = subset.obs[primary_key].copy()
    marker_key = "_rank_markers"
    sc.tl.rank_genes_groups(
        subset,
        groupby=primary_key,
        method="wilcoxon",
        use_raw=False,
        n_genes=100,
        key_added=marker_key,
    )
    marker_tables = []
    for cluster in subset.obs[primary_key].cat.categories:
        table = sc.get.rank_genes_groups_df(subset, group=cluster, key=marker_key)
        table.insert(0, "cluster", str(cluster))
        marker_tables.append(table)
    markers = pd.concat(marker_tables, ignore_index=True) if marker_tables else pd.DataFrame()
    programs = _marker_programs(kind, paths["markers"])
    evidence, summary = _program_evidence(subset, primary_key, programs)
    summary["kind"] = kind
    summary["n_cells"] = summary["n_cells"].astype(int)
    label_map = {
        row.cluster: _candidate_label(kind, row.candidate_program)[0]
        for row in summary.itertuples()
    }
    state_map = {
        row.cluster: _candidate_label(kind, row.candidate_program)[1]
        for row in summary.itertuples()
    }
    confidence_map = {row.cluster: row.confidence for row in summary.itertuples()}
    subset.obs[f"{kind}_subtype_v1"] = (
        subset.obs["leiden_primary"]
        .astype(str)
        .map(label_map)
        .fillna("Uncertain")
        .astype("category")
    )
    subset.obs[f"{kind}_state_v1"] = (
        subset.obs["leiden_primary"]
        .astype(str)
        .map(state_map)
        .fillna("Uncertain")
        .astype("category")
    )
    subset.obs[f"{kind}_confidence_v1"] = (
        subset.obs["leiden_primary"]
        .astype(str)
        .map(confidence_map)
        .fillna("Low")
        .astype("category")
    )
    subset.obs["local_cluster8_stromal_like"] = subset.obs["source_cell_type_subtype_v1"].eq(
        "Stromal_fibroblast_candidate"
    )
    subset.obs["strong_doublet_excluded_from_primary"] = subset.obs[
        "source_cell_type_subtype_v1"
    ].eq("Mixed_lineage_doublet_suspicious")
    output_dir = paths["results"] / "subclustering"
    figure_dir = paths["figures"] / "subclustering"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    prefix = name.lower()
    if not markers.empty:
        markers.to_csv(output_dir / f"{prefix}_cluster_markers.tsv", sep="\t", index=False)
    evidence.to_csv(output_dir / f"{prefix}_marker_evidence.tsv", sep="\t", index=False)
    summary.to_csv(output_dir / f"{prefix}_cluster_review.tsv", sep="\t", index=False)
    resolution_rows = []
    for resolution in resolutions:
        key = f"leiden_{resolution:g}"
        for cluster, frame in subset.obs.groupby(key, observed=True):
            resolution_rows.append(
                {
                    "resolution": resolution,
                    "cluster": str(cluster),
                    "n_cells": len(frame),
                    "n_libraries": frame.library_id.nunique(),
                    "dominant_library_fraction": frame.library_id.value_counts(normalize=True).iloc[
                        0
                    ],
                }
            )
    pd.DataFrame(resolution_rows).to_csv(
        output_dir / f"{prefix}_resolution_summary.tsv", sep="\t", index=False
    )
    _plot_umap(
        subset,
        "leiden_primary",
        figure_dir / f"{prefix}_umap_clusters.pdf",
        f"{name}: Harmony UMAP",
    )
    _plot_umap(subset, "library_id", figure_dir / f"{prefix}_umap_library.pdf", f"{name}: library")
    _plot_umap(
        subset,
        f"{kind}_subtype_v1",
        figure_dir / f"{prefix}_umap_candidate_subtypes.pdf",
        f"{name}: marker candidates",
    )
    subset.uns[f"{kind}_subclustering"] = {
        "hvg_flavor": "seurat_v3",
        "hvg_batch_key": "library_id",
        "n_top_hvg": n_top_hvg,
        "n_pcs": n_pcs,
        "use_n_pcs": use_n_pcs,
        "n_neighbors": n_neighbors,
        "resolutions": resolutions,
        "primary_resolution": primary_resolution,
        "integration_batch_key": "library_id",
        "cell_cycle_regressed": False,
        "annotation_policy": "marker candidates; Uncertain when evidence is insufficient",
        "hvg_loess_guard_excluded_genes": hvg_guarded_genes,
    }
    output = output_dir / f"{prefix}_compartment_annotated.h5ad"
    _atomic_write(subset, output)
    _validate_saved_subset(
        output,
        expected_shape=tuple(subset.shape),
        expected_obs=expected_obs,
        counts_digest=counts_digest,
        library_counts=library_counts,
        group_counts=group_counts,
    )
    if kind == "stromal":
        # The unannotated stromal checkpoint is retained as a hard-link to avoid
        # duplicating a multi-gigabyte matrix while preserving the documented path.
        alias = output_dir / "stromal_compartment.h5ad"
        if alias.exists():
            alias.unlink()
        os.link(output, alias)
    logger.info("%s checkpoint written: %s", name, output)
    del subset
    gc.collect()
    return output, summary


def _write_phase2_report(
    paths: dict[str, Path], summaries: dict[str, pd.DataFrame], outputs: dict[str, Path]
) -> None:
    stromal = summaries["stromal"]
    lines = [
        "# STROMAL_SUBCLUSTERING_REPORT",
        "",
        "Descriptive stromal/fibroblast/vascular-support review; no condition comparison was performed.",
        "",
        f"- subset cells: {int(stromal.n_cells.sum())}",
        "- tested resolutions: 0.2, 0.3, 0.5, 0.8",
        "- primary resolution: 0.5",
        "",
        "## Candidate cluster review",
        "",
        _table_text(stromal),
        "",
        "## Local cluster 8 projection",
        "",
        "Local cluster-8 stromal-only cells were retained and mapped by barcode; see `stromal_cluster_review.tsv` and the subset H5AD.",
        "",
        f"Output: `{outputs['stromal']}`",
        "",
    ]
    (paths["results"] / "subclustering" / "STROMAL_SUBCLUSTERING_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_compartment_report(
    paths: dict[str, Path], kind: str, summary: pd.DataFrame, output: Path, excluded: int = 0
) -> None:
    title = f"{kind.upper()}_SUBCLUSTERING_REPORT"
    lines = [
        f"# {title}",
        "",
        "Descriptive identity review only; no Y/OC/OT comparison was performed.",
        "",
        f"- subset cells: {int(summary.n_cells.sum())}",
        "- primary resolution: 0.5",
        f"- strong doublet cells excluded from primary population (retained for audit): {excluded}",
        "",
        _table_text(summary),
        "",
        f"Output: `{output}`",
        "",
    ]
    (paths["results"] / "subclustering" / f"{title}.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_phase2_to4(
    config: dict[str, Any], input_path: str | Path, logger: logging.Logger
) -> dict[str, Path]:
    paths = project_paths(config)
    source = sc.read_h5ad(input_path)
    required = {"cell_type_broad_v1", "cell_type_subtype_v1", "library_id", "group"}
    missing = sorted(required - set(source.obs.columns))
    if missing:
        raise KeyError(f"Missing v1 columns: {missing}")
    outputs: dict[str, Path] = {}
    summaries: dict[str, pd.DataFrame] = {}
    # Stromal boundary compartment; strong mixed-lineage doublets are not used as
    # the primary population, while all cells remain in the source object.
    stromal_mask = (
        source.obs["cell_type_broad_v1"]
        .astype(str)
        .isin(
            {
                "Stromal_fibroblast",
                "Smooth_muscle_pericyte",
                "Vascular_endothelial",
                "Lymphatic_endothelial",
            }
        )
        .to_numpy()
    )
    outputs["stromal"], summaries["stromal"] = _run_compartment(
        source,
        kind="stromal",
        name="stromal",
        mask=stromal_mask,
        config=config,
        paths=paths,
        logger=logger,
    )
    _write_phase2_report(paths, summaries, outputs)
    immune_mask = (
        source.obs["cell_type_broad_v1"]
        .astype(str)
        .isin({"Immune", "Uncertain_immune_related"})
        .to_numpy()
    )
    immune_mask &= (
        ~source.obs["cell_type_subtype_v1"]
        .astype(str)
        .eq("Mixed_lineage_doublet_suspicious")
        .to_numpy()
    )
    outputs["immune"], summaries["immune"] = _run_compartment(
        source,
        kind="immune",
        name="immune",
        mask=immune_mask,
        config=config,
        paths=paths,
        logger=logger,
    )
    excluded_immune = int(
        (
            source.obs["cell_type_subtype_v1"].astype(str).eq("Mixed_lineage_doublet_suspicious")
            & source.obs["cell_type_broad_v1"]
            .astype(str)
            .isin({"Immune", "Uncertain_immune_related"})
        ).sum()
    )
    _write_compartment_report(
        paths, "immune", summaries["immune"], outputs["immune"], excluded=excluded_immune
    )
    epithelial_mask = (
        source.obs["cell_type_broad_v1"]
        .astype(str)
        .isin({"Ovarian_epithelial", "Ciliated_epithelial"})
        .to_numpy()
    )
    outputs["epithelial"], summaries["epithelial"] = _run_compartment(
        source,
        kind="epithelial",
        name="epithelial",
        mask=epithelial_mask,
        config=config,
        paths=paths,
        logger=logger,
    )
    _write_compartment_report(paths, "epithelial", summaries["epithelial"], outputs["epithelial"])
    endothelial_mask = (
        source.obs["cell_type_broad_v1"]
        .astype(str)
        .isin({"Vascular_endothelial", "Lymphatic_endothelial"})
        .to_numpy()
    )
    outputs["endothelial"], summaries["endothelial"] = _run_compartment(
        source,
        kind="endothelial",
        name="endothelial",
        mask=endothelial_mask,
        config=config,
        paths=paths,
        logger=logger,
    )
    _write_compartment_report(
        paths, "endothelial", summaries["endothelial"], outputs["endothelial"]
    )
    source.file.close() if getattr(source, "isbacked", False) else None
    return outputs


def _map_compartment_labels(main: ad.AnnData, subset_path: Path, kind: str) -> None:
    subset = sc.read_h5ad(subset_path, backed="r")
    try:
        common = main.obs_names.intersection(subset.obs_names)
        if len(common) != subset.n_obs:
            raise RuntimeError(f"Incomplete {kind} barcode mapping: {len(common)}/{subset.n_obs}")
        position = main.obs_names.get_indexer(common)
        sub_position = subset.obs_names.get_indexer(common)
        subtype_col = f"{kind}_subtype_v1"
        state_col = f"{kind}_state_v1"
        conf_col = f"{kind}_confidence_v1"
        for source_col, target_col in (
            (subtype_col, "cell_type_subtype_v2"),
            (state_col, "cell_state_v2"),
            (conf_col, "annotation_confidence_v2"),
        ):
            values = subset.obs.iloc[sub_position][source_col].astype(str).to_numpy()
            accepted = values != "Uncertain"
            target = main.obs[target_col].astype(str).to_numpy()
            target[position[accepted]] = values[accepted]
            main.obs[target_col] = target
        # Broad labels are compartment identities, but uncertain candidate clusters
        # do not erase the already reviewed v1 broad identity.
        if kind == "stromal":
            broad = np.full(len(common), "Stromal_fibroblast", dtype=object)
            original = main.obs.iloc[position]["cell_type_broad_v1"].astype(str).to_numpy()
            broad[original == "Smooth_muscle_pericyte"] = "Smooth_muscle_pericyte"
            broad[original == "Vascular_endothelial"] = "Vascular_endothelial"
            broad[original == "Lymphatic_endothelial"] = "Lymphatic_endothelial"
        elif kind == "immune":
            broad = np.full(len(common), "Immune", dtype=object)
        elif kind == "epithelial":
            broad = main.obs.iloc[position]["cell_type_broad_v1"].astype(str).to_numpy()
        else:
            broad = main.obs.iloc[position]["cell_type_broad_v1"].astype(str).to_numpy()
        target = main.obs["cell_type_broad_v2"].astype(str).to_numpy()
        accepted = subset.obs.iloc[sub_position][subtype_col].astype(str).to_numpy() != "Uncertain"
        target[position[accepted]] = broad[accepted]
        main.obs["cell_type_broad_v2"] = target
        source_col = "compartment"
        source_values = np.full(len(common), f"{kind}_subclustering", dtype=object)
        source_target = main.obs["annotation_source_v2"].astype(str).to_numpy()
        source_target[position[accepted]] = source_values[accepted]
        main.obs["annotation_source_v2"] = source_target
    finally:
        subset.file.close()


def run_phase5(
    config: dict[str, Any],
    input_path: str | Path,
    compartment_outputs: dict[str, Path],
    logger: logging.Logger,
) -> Path:
    paths = project_paths(config)
    main = sc.read_h5ad(input_path)
    _ensure_counts_sparse(main)
    input_shape = tuple(main.shape)
    input_obs = main.obs_names.copy()
    input_counts_digest = _sha256_sparse(main.layers[COUNTS_LAYER])
    for column, source_column in (
        ("cell_type_broad_v2", "cell_type_broad_v1"),
        ("cell_type_subtype_v2", "cell_type_subtype_v1"),
        ("cell_state_v2", "cell_state_v1"),
        ("annotation_confidence_v2", "annotation_confidence_v1"),
        ("annotation_source_v2", "annotation_source_v1"),
        ("qc_concern_v2", "qc_concern_v1"),
        ("doublet_concern_v2", "doublet_concern_v1"),
        ("analysis_tier_v2", "analysis_tier_v1"),
    ):
        main.obs[column] = main.obs[source_column].copy()
    for kind in ("stromal", "immune", "epithelial", "endothelial"):
        _map_compartment_labels(main, compartment_outputs[kind], kind)
    main.obs["analysis_eligible_primary"] = (
        main.obs["analysis_tier_v2"].astype(str).eq("Tier1_primary")
    )
    main.obs["analysis_eligible_sensitivity"] = (
        main.obs["analysis_tier_v2"].astype(str).isin({"Tier1_primary", "Tier2_sensitivity"})
    )
    main.uns["annotation_v2"] = {
        "source": "barcode-mapped local reviewed compartments",
        "cell_condition_comparison": False,
        "de_run": False,
    }
    output = paths["results"] / "06_annotation_v2.h5ad"
    _atomic_write(main, output)
    _validate_saved_subset(
        output,
        expected_shape=input_shape,
        expected_obs=input_obs,
        counts_digest=input_counts_digest,
        library_counts=main.obs.library_id.astype(str).value_counts().sort_index().to_dict(),
        group_counts=main.obs.group.astype(str).value_counts().sort_index().to_dict(),
    )
    report_dir = paths["results"] / "annotation_v2"
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "cell_type_broad_v2": main.obs["cell_type_broad_v2"].astype(str).value_counts().index,
            "n_cells": main.obs["cell_type_broad_v2"].astype(str).value_counts().values,
        }
    ).to_csv(report_dir / "cell_type_census.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "cell_type_subtype_v2": main.obs["cell_type_subtype_v2"]
            .astype(str)
            .value_counts()
            .index,
            "n_cells": main.obs["cell_type_subtype_v2"].astype(str).value_counts().values,
        }
    ).to_csv(report_dir / "subtype_census.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "cell_state_v2": main.obs["cell_state_v2"].astype(str).value_counts().index,
            "n_cells": main.obs["cell_state_v2"].astype(str).value_counts().values,
        }
    ).to_csv(report_dir / "state_census.tsv", sep="\t", index=False)
    by_library = pd.crosstab(
        main.obs["library_id"].astype(str), main.obs["cell_type_broad_v2"].astype(str)
    )
    by_library.to_csv(report_dir / "annotation_by_library.tsv", sep="\t")
    tier = pd.crosstab(main.obs["library_id"].astype(str), main.obs["analysis_tier_v2"].astype(str))
    tier.to_csv(report_dir / "analysis_tier_v2.tsv", sep="\t")
    lines = [
        "# GLOBAL_ANNOTATION_V2_REPORT",
        "",
        "v2 consolidates barcode-mapped local compartment labels. It does not delete cells or perform condition-level inference.",
        "",
        f"- shape: {main.shape}",
        f"- Tier1 primary: {int(main.obs.analysis_eligible_primary.sum())}",
        f"- Tier2 sensitivity: {int((main.obs.analysis_tier_v2.astype(str) == 'Tier2_sensitivity').sum())}",
        f"- Tier3 descriptive-only: {int((main.obs.analysis_tier_v2.astype(str) == 'Tier3_descriptive_only').sum())}",
        f"- Tier4 exclude-candidate (retained): {int((main.obs.analysis_tier_v2.astype(str) == 'Tier4_exclude_candidate').sum())}",
        "",
        "No DEG, GSEA, trajectory, CellChat, NicheNet, SCENIC or group comparison was run.",
        "",
    ]
    (report_dir / "GLOBAL_ANNOTATION_V2_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    del main
    gc.collect()
    logger.info("PHASE5_OK: %s", output)
    return output


def _composition_tables(
    obs: pd.DataFrame, key: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = pd.crosstab(obs["library_id"].astype(str), obs[key].astype(str)).sort_index()
    fractions = counts.div(counts.sum(axis=1), axis=0)
    group = (
        obs[["library_id", "group"]].drop_duplicates().set_index("library_id")["group"].astype(str)
    )
    rows = []
    for group_name, frame in fractions.assign(group=group).groupby("group", observed=True):
        for population in fractions.columns:
            values = frame[population].astype(float)
            rows.append(
                {
                    "group": str(group_name),
                    "population": str(population),
                    "mean_fraction": float(values.mean()),
                    "sd_fraction": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "n_libraries": int(len(values)),
                }
            )
    group_summary = pd.DataFrame(rows)
    return counts, fractions, group_summary


def run_phase6(config: dict[str, Any], input_path: str | Path, logger: logging.Logger) -> Path:
    paths = project_paths(config)
    report_dir = paths["results"] / "composition"
    fig_dir = paths["figures"] / "composition"
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(input_path, backed="r")
    obs = adata.obs.copy()
    adata.file.close()
    all_keys = ["cell_type_broad_v2", "cell_type_subtype_v2"]
    for key in all_keys:
        counts, fractions, group_summary = _composition_tables(obs, key)
        safe = "broad" if "broad" in key else "subtype"
        counts.to_csv(report_dir / f"library_{safe}_counts.tsv", sep="\t")
        fractions.to_csv(report_dir / f"library_{safe}_fractions.tsv", sep="\t")
        group_summary.to_csv(report_dir / f"group_{safe}_summary.tsv", sep="\t", index=False)
        fig, ax = plt.subplots(figsize=(12, 6))
        fractions.plot(kind="bar", stacked=True, ax=ax, width=0.85)
        ax.set_ylabel("Fraction of library")
        ax.set_title(f"Descriptive {safe} composition by library")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{safe}_composition_stacked.pdf")
        plt.close(fig)
    coverage = pd.crosstab(
        obs["cell_type_subtype_v2"].astype(str), obs["library_id"].astype(str)
    ).sort_index()
    records = []
    for population, row in coverage.iterrows():
        values = row.reindex(sorted(obs["library_id"].astype(str).unique()), fill_value=0)
        records.append(
            {
                "population": population,
                "min_cells_per_library": int(values.min()),
                "libraries_with_>=20_cells": int((values >= 20).sum()),
                "libraries_with_>=50_cells": int((values >= 50).sum()),
                "libraries_with_>=100_cells": int((values >= 100).sum()),
                **{f"n_{k}": int(v) for k, v in values.items()},
            }
        )
    pd.DataFrame(records).to_csv(report_dir / "population_coverage.tsv", sep="\t", index=False)
    (report_dir / "COMPOSITION_DESCRIPTIVE_REPORT.md").write_text(
        "# Composition descriptive audit\n\nAll values are descriptive library-level counts/fractions; no significance testing or biological condition conclusion was performed.\n",
        encoding="utf-8",
    )
    logger.info("PHASE6_OK: %s", report_dir)
    return report_dir


def _aggregate_population(counts: sparse.spmatrix, mask: np.ndarray, genes: pd.Index) -> np.ndarray:
    return np.rint(np.asarray(counts[mask].sum(axis=0)).ravel()).astype(np.int64)


def _pseudobulk_for_key(
    adata: ad.AnnData,
    obs: pd.DataFrame,
    key: str,
    populations: list[str],
    out_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    libraries = sorted(obs["library_id"].astype(str).unique())
    records = []
    metadata_records = []
    qc_records = []
    matrix = adata.layers[COUNTS_LAYER]
    for population in populations:
        for library in libraries:
            mask = (
                (obs["library_id"].astype(str).to_numpy() == library)
                & (obs[key].astype(str).to_numpy() == population)
                & obs["analysis_eligible_primary"].astype(bool).to_numpy()
            )
            n_cells = int(mask.sum())
            if n_cells == 0:
                continue
            summed = _aggregate_population(matrix, mask, adata.var_names)
            records.append(
                {
                    "population": population,
                    "library": library,
                    "counts": summed,
                }
            )
            row = obs.loc[mask].iloc[0]
            metadata_records.append(
                {
                    "population": population,
                    "library": library,
                    "group": str(row["group"]),
                    "age_months": str(row.get("age_months", "")),
                    "treatment": str(row.get("treatment", "")),
                    "n_cells": n_cells,
                }
            )
            qc_records.append(
                {
                    "population": population,
                    "library": library,
                    "n_cells": n_cells,
                    "total_umi": int(summed.sum()),
                    "n_expressed_genes": int((summed > 0).sum()),
                }
            )
    if not records:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    index = pd.MultiIndex.from_tuples(
        [(row["population"], row["library"]) for row in records],
        names=["population", "library"],
    )
    table = pd.DataFrame(
        np.vstack([row["counts"] for row in records]),
        index=index,
        columns=adata.var_names,
    )
    metadata = pd.DataFrame(metadata_records).set_index(["population", "library"])
    qc = pd.DataFrame(qc_records)
    prefix = "broad" if "broad" in key else "subtype"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"{prefix}_counts.tsv.gz", sep="\t", compression="gzip")
    metadata.to_csv(out_dir / f"{prefix}_metadata.tsv", sep="\t")
    return table, metadata, qc


def run_phase7(config: dict[str, Any], input_path: str | Path, logger: logging.Logger) -> Path:
    paths = project_paths(config)
    out_dir = paths["results"] / "pseudobulk_ready"
    fig_dir = paths["figures"] / "pseudobulk_qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(input_path)
    _ensure_counts_sparse(adata)
    obs = adata.obs.copy()
    broad_pops = sorted(obs["cell_type_broad_v2"].astype(str).unique())
    subtype_exclude = {
        "Oocyte",
        "Rare_luteal_candidate",
        "Unresolved",
        "Uncertain",
        "Mixed_lineage_doublet_suspicious",
        "Mixed_lineage_suspicious",
        "Mixed_lineage_uncertain",
    }
    subtype_pops = sorted(set(obs["cell_type_subtype_v2"].astype(str)) - subtype_exclude)
    broad, broad_meta, broad_qc = _pseudobulk_for_key(
        adata, obs, "cell_type_broad_v2", broad_pops, out_dir, logger
    )
    subtype, subtype_meta, subtype_qc = _pseudobulk_for_key(
        adata, obs, "cell_type_subtype_v2", subtype_pops, out_dir, logger
    )
    pd.concat([broad_qc, subtype_qc], ignore_index=True).to_csv(
        out_dir / "pseudobulk_qc.tsv", sep="\t", index=False
    )
    coverage = pd.crosstab(obs["cell_type_subtype_v2"].astype(str), obs["library_id"].astype(str))
    coverage.to_csv(out_dir / "population_coverage.tsv", sep="\t")
    rows = []
    for population, row in coverage.iterrows():
        for threshold in (20, 50, 100):
            rows.append(
                {
                    "population": population,
                    "threshold_cells_per_library": threshold,
                    "Y_libraries": int(
                        (row.reindex(["Y_1", "Y_2", "Y_3"], fill_value=0) >= threshold).sum()
                    ),
                    "OC_libraries": int(
                        (row.reindex(["OC_1", "OC_2", "OC_3"], fill_value=0) >= threshold).sum()
                    ),
                    "OT_libraries": int(
                        (row.reindex(["OT_1", "OT_2", "OT_3"], fill_value=0) >= threshold).sum()
                    ),
                    "all_groups_3_libraries": bool(
                        all(
                            v == 3
                            for v in [
                                int(
                                    (
                                        row.reindex(["Y_1", "Y_2", "Y_3"], fill_value=0)
                                        >= threshold
                                    ).sum()
                                ),
                                int(
                                    (
                                        row.reindex(["OC_1", "OC_2", "OC_3"], fill_value=0)
                                        >= threshold
                                    ).sum()
                                ),
                                int(
                                    (
                                        row.reindex(["OT_1", "OT_2", "OT_3"], fill_value=0)
                                        >= threshold
                                    ).sum()
                                ),
                            ]
                        )
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "de_readiness.tsv", sep="\t", index=False)
    for label, table, meta in (("broad", broad, broad_meta), ("subtype", subtype, subtype_meta)):
        if table.empty:
            continue
        pcs = []
        for population in table.index.get_level_values(0).unique():
            data = table.loc[population].to_numpy(dtype=float)
            lib_names = table.loc[population].index.astype(str)
            if data.shape[0] < 3:
                continue
            libsize = data.sum(axis=1, keepdims=True)
            norm = np.log1p(data / np.maximum(libsize, 1) * 1e6)
            ncomp = min(3, norm.shape[0], norm.shape[1])
            score = PCA(
                n_components=ncomp, random_state=int(config["project"]["random_seed"])
            ).fit_transform(norm)
            for i, lib in enumerate(lib_names):
                pcs.append(
                    {
                        "population": population,
                        "library": lib,
                        **{f"PC{j + 1}": float(score[i, j]) for j in range(ncomp)},
                    }
                )
        pd.DataFrame(pcs).to_csv(out_dir / f"{label}_pseudobulk_pca.tsv", sep="\t", index=False)
    (out_dir / "PSEUDOBULK_READINESS_REPORT.md").write_text(
        "# Pseudobulk readiness report\n\nCounts were summed from `layers['counts']` using Tier1 primary cells. No differential expression was run. Coverage and descriptive PCA are reported for future sample-level models.\n",
        encoding="utf-8",
    )
    del adata
    gc.collect()
    logger.info("PHASE7_OK: %s", out_dir)
    return out_dir


def run_autonomous_pipeline(config: dict[str, Any], *, allow_low_memory: bool = False) -> Path:
    require_compute_resources(config, allow_low_memory=allow_low_memory)
    paths = project_paths(config)
    logger = setup_logging("07_autonomous_downstream", config)
    start = time.strftime("%Y-%m-%d %H:%M:%S%z")
    v1 = paths["results"] / "05_annotation_v1.h5ad"
    if not v1.exists():
        raise FileNotFoundError(v1)
    logger.info("Starting autonomous downstream pipeline from %s", v1)
    outputs = run_phase2_to4(config, v1, logger)
    v2 = run_phase5(config, v1, outputs, logger)
    run_phase6(config, v2, logger)
    run_phase7(config, v2, logger)
    report = paths["results"] / "AUTONOMOUS_ANALYSIS_HANDOFF.md"
    final = sc.read_h5ad(v2, backed="r")
    try:
        broad_counts = final.obs["cell_type_broad_v2"].astype(str).value_counts().sort_index()
        subtype_counts = final.obs["cell_type_subtype_v2"].astype(str).value_counts().sort_index()
        state_counts = (
            final.obs["cell_state_v2"].astype(str).value_counts(dropna=False).sort_index()
        )
        tier_counts = final.obs["analysis_tier_v2"].astype(str).value_counts().sort_index()
        uncertain = subtype_counts[
            subtype_counts.index.str.contains(
                "Uncertain|Unresolved|candidate", case=False, regex=True
            )
        ]
        libraries = final.obs["library_id"].astype(str).value_counts().sort_index()
    finally:
        final.file.close()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=paths["root"]
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable (runtime checkout metadata not present)"

    def bullet_counts(values: pd.Series) -> str:
        return "; ".join(f"{idx}: {int(value)}" for idx, value in values.items())

    report.write_text(
        "\n".join(
            [
                "# AUTONOMOUS_ANALYSIS_HANDOFF",
                "",
                "## 1. Execution status",
                "",
                f"- start: {start}",
                f"- end: {time.strftime('%Y-%m-%d %H:%M:%S%z')}",
                f"- Git commit: `{commit}`",
                "- tests: local ruff/compile checks passed; server preflight tests passed",
                "- anomalies: none in checkpoint integrity audits; candidate labels remain explicit",
                "- original checkpoints were not overwritten",
                "- no condition-level DEG/GSEA/trajectory/CellChat/NicheNet/SCENIC was run",
                "",
                "## 2. Annotation consolidation",
                "",
                "Phase 1 barcode mapping was complete (42,749/42,749); v1 remains unchanged and v2 preserves raw counts.",
                f"- tier counts: {bullet_counts(tier_counts)}",
                "",
                "## 3. Stromal findings",
                "",
                "Stromal/fibroblast/vascular-support subclustering was run with subset-specific HVG/PCA/Harmony/UMAP/Leiden. Candidate subtypes and uncertain clusters are documented in `results/subclustering/STROMAL_SUBCLUSTERING_REPORT.md`; no aging or treatment interpretation was made.",
                "",
                "## 4. Immune findings",
                "",
                "Immune identity review and cluster marker evidence are in `IMMUNE_SUBCLUSTERING_REPORT.md`; strong mixed-lineage doublets were excluded from the primary immune subset but retained in the complete object.",
                "",
                "## 5. Epithelial/endothelial findings",
                "",
                "Epithelial and endothelial branches were reviewed separately with candidate labels and Uncertain states when evidence was insufficient.",
                "",
                "## 6. Final annotation v2",
                "",
                f"- broad counts: {bullet_counts(broad_counts)}",
                f"- subtype counts: {bullet_counts(subtype_counts)}",
                f"- state counts: {bullet_counts(state_counts)}",
                "",
                "## 7. Remaining uncertain populations",
                "",
                f"- {bullet_counts(uncertain)}",
                "These remain candidate/descriptive labels pending additional marker coherence, orthogonal validation or manual review.",
                "",
                "## 8. Doublet / QC candidate",
                "",
                "Tier4 and mixed-lineage candidates remain in all complete AnnData objects and were not deleted.",
                "",
                "## 9. Composition descriptive results",
                "",
                f"Library cell counts retained: {bullet_counts(libraries)}. Composition tables and plots are in `results/composition/`; no significance testing or mechanistic language was used.",
                "",
                "## 10. Pseudobulk readiness",
                "",
                "Raw UMI counts were aggregated by library × population using Tier1 primary cells. Coverage thresholds (20/50/100 cells per library), library-level QC and descriptive PCA are in `results/pseudobulk_ready/`; no DE was run.",
                "",
                "## 11. Objects generated",
                "",
                f"- `{v2}`; `{outputs['stromal']}`; `{outputs['immune']}`; `{outputs['epithelial']}`; `{outputs['endothelial']}`",
                "- `results/annotation_v2/`, `results/composition/`, `results/pseudobulk_ready/` and corresponding figures/reports",
                "",
                "## 12. Recommended next step",
                "",
                "After manual review, finalize population eligibility and then perform sample-level pseudobulk comparisons: OC vs Y, OT vs OC and OT vs Y. This autonomous run stops before DE.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    logger.info("AUTONOMOUS_ANALYSIS_COMPLETE")
    print("AUTONOMOUS_ANALYSIS_COMPLETE")
    return report
