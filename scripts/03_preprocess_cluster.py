from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
import scanpy.external as sce

from _project import (
    DEFAULT_CONFIG,
    load_config,
    project_paths,
    require_compute_resources,
    setup_logging,
)


def harmony_integrate_compatible(
    adata: sc.AnnData,
    batch_key: str,
    seed: int,
    basis: str = "X_pca",
    adjusted_basis: str = "X_pca_harmony",
) -> None:
    """Run harmonypy while accepting both historic and current Z_corr layouts.

    Older harmonypy releases expose PCs x cells and current releases may expose
    cells x PCs. Some Scanpy wrappers unconditionally transpose the result.
    Inspecting the dimensions here makes the saved representation independent
    of that package-version mismatch.
    """
    import harmonypy

    harmony_out = harmonypy.run_harmony(
        adata.obsm[basis],
        adata.obs,
        batch_key,
        random_state=seed,
    )
    corrected = np.asarray(harmony_out.Z_corr)
    if corrected.ndim != 2:
        raise ValueError(f"Harmony returned a non-matrix result: {corrected.shape}")
    if corrected.shape[0] == adata.n_obs:
        pass
    elif corrected.shape[1] == adata.n_obs:
        corrected = corrected.T
    else:
        raise ValueError(
            "Harmony result does not contain the expected number of cells: "
            f"result={corrected.shape}, n_obs={adata.n_obs}"
        )
    adata.obsm[adjusted_basis] = corrected.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize, select HVGs, run PCA, compare unintegrated/Harmony, and cluster."
    )
    parser.add_argument("input")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--integration", choices=["none", "harmony", "bbknn"], default=None)
    parser.add_argument("--allow-low-memory", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    require_compute_resources(config, allow_low_memory=args.allow_low_memory)
    paths = project_paths(config)
    logger = setup_logging("03_preprocess_cluster", config)
    preprocess = config["preprocess"]
    integration = args.integration or config["integration"]["primary"]
    counts_layer = config["ingest"]["counts_layer"]
    seed = int(config["project"]["random_seed"])

    adata = sc.read_h5ad(args.input)
    if counts_layer not in adata.layers:
        raise KeyError(f"Required raw-count layer missing: {counts_layer}")
    adata.X = adata.layers[counts_layer].copy()
    sc.pp.normalize_total(adata, target_sum=float(preprocess["target_sum"]))
    sc.pp.log1p(adata)
    adata.raw = adata

    sc.pp.highly_variable_genes(
        adata,
        layer=counts_layer,
        flavor=preprocess["hvg_flavor"],
        n_top_genes=int(preprocess["n_top_hvg"]),
        batch_key=preprocess["hvg_batch_key"],
        subset=False,
    )
    hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(hvg, max_value=float(preprocess["scale_max_value"]))
    sc.tl.pca(
        hvg,
        n_comps=int(preprocess["n_pcs"]),
        svd_solver="arpack",
        random_state=seed,
    )
    adata.obsm["X_pca"] = hvg.obsm["X_pca"].astype(np.float32, copy=False)
    adata.uns["pca"] = hvg.uns["pca"]
    adata.varm["PCs"] = np.zeros((adata.n_vars, hvg.varm["PCs"].shape[1]), dtype=np.float32)
    adata.varm["PCs"][adata.var["highly_variable"].to_numpy(), :] = hvg.varm["PCs"]
    del hvg

    sc.pp.neighbors(
        adata,
        n_neighbors=int(preprocess["n_neighbors"]),
        n_pcs=int(preprocess["use_n_pcs"]),
        use_rep="X_pca",
        key_added="neighbors_unintegrated",
        random_state=seed,
    )
    sc.tl.umap(adata, neighbors_key="neighbors_unintegrated", random_state=seed)
    adata.obsm["X_umap_unintegrated"] = adata.obsm["X_umap"].copy()

    batch_key = config["integration"]["batch_key"]
    if integration == "harmony":
        logger.info("Harmony integration using %s", batch_key)
        harmony_integrate_compatible(adata, batch_key=batch_key, seed=seed)
        sc.pp.neighbors(
            adata,
            n_neighbors=int(preprocess["n_neighbors"]),
            n_pcs=int(preprocess["use_n_pcs"]),
            use_rep="X_pca_harmony",
            random_state=seed,
        )
    elif integration == "bbknn":
        logger.info("BBKNN integration using %s", batch_key)
        sce.pp.bbknn(
            adata,
            batch_key=batch_key,
            use_rep="X_pca",
            n_pcs=int(preprocess["use_n_pcs"]),
            neighbors_within_batch=3,
            copy=False,
        )
    else:
        logger.info("Using unintegrated PCA graph")
        adata.obsp["connectivities"] = adata.obsp["neighbors_unintegrated_connectivities"].copy()
        adata.obsp["distances"] = adata.obsp["neighbors_unintegrated_distances"].copy()
        adata.uns["neighbors"] = adata.uns["neighbors_unintegrated"].copy()

    sc.tl.umap(adata, random_state=seed)
    for resolution in config["clustering"]["resolutions"]:
        key = f"leiden_{resolution}"
        sc.tl.leiden(
            adata,
            resolution=float(resolution),
            key_added=key,
            random_state=seed,
        )
    primary = str(config["clustering"]["primary_resolution"])
    adata.obs["leiden"] = adata.obs[f"leiden_{primary}"].copy()
    adata.uns["integration_method"] = integration

    output = paths["results"] / "03_clustered.h5ad"
    adata.write_h5ad(output, compression="gzip")
    sc.settings.figdir = paths["figures"]
    sc.settings.file_format_figs = "pdf"
    sc.pl.umap(
        adata,
        color=["group", "library_id", "leiden"],
        frameon=False,
        show=False,
        save="_03_atlas_overview.pdf",
    )
    logger.info("PREPROCESS_CLUSTER_OK: %s shape=%s", output, adata.shape)


if __name__ == "__main__":
    main()
