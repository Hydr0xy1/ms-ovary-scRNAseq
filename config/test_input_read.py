from importlib.metadata import version
from pathlib import Path

import numpy as np
import scanpy as sc
from scipy import sparse


sample = "OT_3"
sample_dir = Path("/root/autodl-tmp/ovary_scRNAseq/data/Summary") / sample
adata = sc.read_10x_mtx(sample_dir, var_names="gene_symbols", make_unique=True)

assert adata.shape == (9844, 57132), adata.shape
assert sparse.issparse(adata.X)
assert adata.X.nnz == 32524350, adata.X.nnz
assert np.all(adata.X.data >= 0)

print(f"scanpy={version('scanpy')}")
print(f"sample={sample}")
print(f"cells={adata.n_obs}")
print(f"features={adata.n_vars}")
print(f"nnz={adata.X.nnz}")
print(f"matrix_type={type(adata.X).__name__}")
print(f"min_count={adata.X.data.min()}")
print(f"max_count={adata.X.data.max()}")
print("INPUT_READ_TEST_OK")
