from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scanpy.external as sce
from scipy import sparse

from .integration import harmony_integrate_compatible
from .project import project_paths, require_compute_resources, setup_logging


def normalize_log1p_preserving_counts(
    adata: ad.AnnData,
    *,
    counts_layer: str,
    target_sum: float,
) -> None:
    """Log-normalize X while leaving the raw UMI counts layer untouched."""
    if counts_layer not in adata.layers:
        raise KeyError(f"Required raw-count layer missing: {counts_layer}")
    adata.layers[counts_layer] = sparse.csr_matrix(adata.layers[counts_layer])
    adata.X = adata.layers[counts_layer].copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.uns["expression_semantics"] = {
        "X": f"log1p library-size normalized expression (target_sum={target_sum:g})",
        "counts_layer": counts_layer,
        "counts_layer_values": "raw UMI counts",
    }


def select_batch_aware_hvgs(
    adata: ad.AnnData,
    *,
    counts_layer: str,
    flavor: str,
    n_top_genes: int,
    batch_key: str,
) -> None:
    """Select batch-aware HVGs from the raw counts layer without subsetting genes."""
    sc.pp.highly_variable_genes(
        adata,
        layer=counts_layer,
        flavor=flavor,
        n_top_genes=n_top_genes,
        batch_key=batch_key,
        subset=False,
    )


def compute_hvg_pca(
    adata: ad.AnnData,
    *,
    n_comps: int,
    scale_max_value: float,
    random_state: int,
) -> ad.AnnData:
    """Scale only HVGs, run PCA, and map PCA results back to the full-gene object."""
    if "highly_variable" not in adata.var:
        raise KeyError("HVG annotation is missing")
    hvg_mask = adata.var["highly_variable"].to_numpy(dtype=bool)
    if int(hvg_mask.sum()) <= n_comps:
        raise ValueError("The number of HVGs must exceed the requested PCA components")
    hvg = adata[:, hvg_mask].copy()
    sc.pp.scale(hvg, max_value=scale_max_value)
    sc.tl.pca(
        hvg,
        n_comps=n_comps,
        svd_solver="arpack",
        random_state=random_state,
    )
    adata.obsm["X_pca"] = hvg.obsm["X_pca"].astype(np.float32, copy=False)
    adata.uns["pca"] = hvg.uns["pca"]
    adata.varm["PCs"] = np.zeros((adata.n_vars, hvg.varm["PCs"].shape[1]), dtype=np.float32)
    adata.varm["PCs"][hvg_mask, :] = hvg.varm["PCs"]
    return hvg


def compute_unintegrated_neighbors_umap(
    adata: ad.AnnData,
    *,
    n_neighbors: int,
    n_pcs: int,
    random_state: int,
    neighbors_key: str = "neighbors_unintegrated",
) -> None:
    """Build a PCA neighbor graph and UMAP without batch integration."""
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep="X_pca",
        key_added=neighbors_key,
        random_state=random_state,
    )
    sc.tl.umap(adata, neighbors_key=neighbors_key, random_state=random_state)
    adata.obsm["X_umap_unintegrated"] = adata.obsm["X_umap"].copy()


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
    normalize_log1p_preserving_counts(
        adata,
        counts_layer=counts_layer,
        target_sum=float(preprocess["target_sum"]),
    )
    adata.raw = adata

    select_batch_aware_hvgs(
        adata,
        counts_layer=counts_layer,
        flavor=preprocess["hvg_flavor"],
        n_top_genes=int(preprocess["n_top_hvg"]),
        batch_key=preprocess["hvg_batch_key"],
    )
    hvg = compute_hvg_pca(
        adata,
        n_comps=int(preprocess["n_pcs"]),
        scale_max_value=float(preprocess["scale_max_value"]),
        random_state=seed,
    )
    del hvg

    compute_unintegrated_neighbors_umap(
        adata,
        n_neighbors=int(preprocess["n_neighbors"]),
        n_pcs=int(preprocess["use_n_pcs"]),
        random_state=seed,
    )

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
        adata.obsp["connectivities"] = adata.obsp["neighbors_unintegrated_connectivities"].copy()
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
