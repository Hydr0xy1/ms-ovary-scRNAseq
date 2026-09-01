import logging

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from ms_ovary_scrna.qc import mad_outlier, run_scrublet_per_library


def test_mad_outlier_flags_extreme_high_value() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    result = mad_outlier(values, n_mads=3.0, side="high")
    assert result.tolist() == [False, False, False, False, True]


def test_scrublet_accepts_library_mask_for_sparse_counts(monkeypatch) -> None:
    counts = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 2],
                [0, 3, 0],
                [1, 1, 0],
                [0, 0, 4],
            ],
            dtype=np.float32,
        )
    )
    adata = ad.AnnData(
        X=counts.copy(),
        obs=pd.DataFrame(
            {"library_id": ["Y_1", "Y_1", "OC_1", "OC_1"]},
            index=[f"cell_{index}" for index in range(4)],
        ),
        var=pd.DataFrame(index=["gene_1", "gene_2", "gene_3"]),
    )
    adata.layers["counts"] = counts.copy()
    observed_sizes: list[int] = []

    def fake_scrublet(subset, **kwargs) -> None:
        observed_sizes.append(subset.n_obs)
        subset.obs["doublet_score"] = np.arange(subset.n_obs, dtype=float)
        subset.obs["predicted_doublet"] = [False, True]

    monkeypatch.setattr("ms_ovary_scrna.qc.sc.pp.scrublet", fake_scrublet)
    config = {
        "project": {"random_seed": 20260811},
        "ingest": {"counts_layer": "counts"},
        "qc": {"expected_doublet_rate": 0.06},
    }

    run_scrublet_per_library(adata, config, logging.getLogger("test_scrublet"))

    assert observed_sizes == [2, 2]
    assert adata.obs["doublet_score"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert adata.obs["predicted_doublet"].tolist() == [False, True, False, True]
