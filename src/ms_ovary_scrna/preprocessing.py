from __future__ import annotations

from pathlib import Path

import numpy as np
import scanpy as sc
import scanpy.external as sce

from .integration import harmony_integrate_compatible
from .project import project_paths, require_compute_resources, setup_logging


def run_preprocess_cluster(
    config: dict,
    input_path: str | Path,
    *,
    integration: str | None = None,
    allow_low_memory: bool = False,
) -> Path:
    require_compute_resources(config, allow_low_memory=allow_low_memory)
    paths = project_paths(config)
    logger = setup_logging("03_preprocess_cluster", config)
    preprocess = config["preprocess"]
    integration_method = integration or config["integration"]["primary"]
    counts_layer = config["ingest"]["counts_layer"]
    seed = int(config["project"]["random_seed"])

    adata = sc.read_h5ad(input_path)
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
    adata.varm["PCs"] = np.zeros(
        (adata.n_vars, hvg.varm["PCs"].shape[1]), dtype=np.float32
    )
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
    if integration_method == "harmony":
        logger.info("Harmony integration using %s", batch_key)
        harmony_integrate_compatible(adata, batch_key=batch_key, seed=seed)
        sc.pp.neighbors(
            adata,
            n_neighbors=int(preprocess["n_neighbors"]),
            n_pcs=int(preprocess["use_n_pcs"]),
            use_rep="X_pca_harmony",
            random_state=seed,
        )
    elif integration_method == "bbknn":
        logger.info("BBKNN integration using %s", batch_key)
        sce.pp.bbknn(
            adata,
            batch_key=batch_key,
            use_rep="X_pca",
            n_pcs=int(preprocess["use_n_pcs"]),
            neighbors_within_batch=3,
            copy=False,
        )
    elif integration_method == "none":
        logger.info("Using unintegrated PCA graph")
        adata.obsp["connectivities"] = adata.obsp[
            "neighbors_unintegrated_connectivities"
        ].copy()
        adata.obsp["distances"] = adata.obsp["neighbors_unintegrated_distances"].copy()
        adata.uns["neighbors"] = adata.uns["neighbors_unintegrated"].copy()
    else:
        raise ValueError(f"Unsupported integration method: {integration_method}")

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
    adata.uns["integration_method"] = integration_method

    output = paths["results"] / "03_clustered.h5ad"
    adata.write_h5ad(output, compression="gzip")
    figure = sc.pl.umap(
        adata,
        color=["group", "library_id", "leiden"],
        frameon=False,
        show=False,
        return_fig=True,
    )
    figure.savefig(paths["figures"] / "03_atlas_overview.pdf", bbox_inches="tight")
    logger.info("PREPROCESS_CLUSTER_OK: %s shape=%s", output, adata.shape)
    return output
