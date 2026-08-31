from __future__ import annotations

import numpy as np
import scanpy as sc


def harmony_integrate_compatible(
    adata: sc.AnnData,
    batch_key: str,
    seed: int,
    basis: str = "X_pca",
    adjusted_basis: str = "X_pca_harmony",
) -> None:
    """Run harmonypy while accepting historic and current ``Z_corr`` layouts."""
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
