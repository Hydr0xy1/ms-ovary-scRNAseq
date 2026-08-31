from importlib.metadata import version

import numpy as np
import pytest
import scanpy as sc
from scipy import sparse

from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config, project_paths


SAMPLE = "OT_3"
EXPECTED_SHAPE = (9844, 57132)
EXPECTED_NNZ = 32524350


def read_sample():
    paths = project_paths(load_config(DEFAULT_CONFIG))
    sample_dir = paths["input"] / SAMPLE
    if not sample_dir.exists():
        pytest.skip("Server input matrices are not available in this checkout")
    return sc.read_10x_mtx(sample_dir, var_names="gene_symbols", make_unique=True)


@pytest.mark.integration
def test_filtered_10x_input_is_readable() -> None:
    adata = read_sample()
    assert adata.shape == EXPECTED_SHAPE
    assert sparse.issparse(adata.X)
    assert adata.X.nnz == EXPECTED_NNZ
    assert np.all(adata.X.data >= 0)


if __name__ == "__main__":
    adata = read_sample()
    print(f"scanpy={version('scanpy')}")
    print(f"sample={SAMPLE}")
    print(f"cells={adata.n_obs}")
    print(f"features={adata.n_vars}")
    print(f"nnz={adata.X.nnz}")
    print(f"matrix_type={type(adata.X).__name__}")
    print(f"min_count={adata.X.data.min()}")
    print(f"max_count={adata.X.data.max()}")
    print("INPUT_READ_TEST_OK")
