from importlib.metadata import version
from pathlib import Path

import anndata
import bbknn
import celltypist
import gseapy
import harmonypy
import igraph
import ipykernel
import jupyterlab
import leidenalg
import ms_ovary_scrna
import numpy as np
import pandas as pd
import pydeseq2
import scanpy as sc
import scipy.sparse as sp
import scrublet
import sklearn


packages = [
    "igraph",
    "leidenalg",
    "scanpy",
    "anndata",
    "scrublet",
    "harmonypy",
    "bbknn",
    "celltypist",
    "pydeseq2",
    "gseapy",
    "jupyterlab",
    "ipykernel",
]

print("PACKAGE_VERSIONS")
for package in packages:
    print(f"{package}={version(package)}")
print(f"ms-ovary-scrna={ms_ovary_scrna.__version__}")

# Small end-to-end smoke test: sparse counts -> QC -> PCA -> graph -> UMAP -> Leiden.
rng = np.random.default_rng(20260810)
x = sp.csr_matrix(rng.poisson(1.0, size=(120, 80)).astype(np.float32))
adata = anndata.AnnData(
    x,
    obs=pd.DataFrame(index=[f"cell_{i}" for i in range(x.shape[0])]),
    var=pd.DataFrame(index=[f"gene_{i}" for i in range(x.shape[1])]),
)
sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.pca(adata, n_comps=20)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=15)
sc.tl.umap(adata, random_state=0)
sc.tl.leiden(adata, resolution=0.5, random_state=0)

assert adata.obsm["X_umap"].shape == (120, 2)
assert "leiden" in adata.obs

project_root = Path(__file__).resolve().parents[1]
output = project_root / "logs" / "environment_smoke_test.h5ad"
adata.write_h5ad(output, compression="gzip")
print(f"SMOKE_TEST_OK={output}")
