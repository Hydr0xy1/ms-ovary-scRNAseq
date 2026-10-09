from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from ms_ovary_scrna.stage26_all_cell_cycle import (
    assign_phase,
    scanpy_control_genes,
    stream_cell_scores,
    stream_gene_means,
)


def test_assign_phase_matches_scanpy_rule() -> None:
    observed = assign_phase([-1.0, 2.0, 0.1, -0.2], [-2.0, 1.0, 3.0, 0.4])
    assert observed.tolist() == ["G1", "S", "G2M", "G2M"]


def test_control_gene_selection_matches_scanpy_internal_function() -> None:
    from scanpy.tools._score_genes import _score_genes_bins

    rng = np.random.default_rng(20261009)
    matrix = sparse.random(
        80,
        120,
        density=0.2,
        random_state=rng,
        data_rvs=lambda n: rng.lognormal(size=n),
        format="csr",
    )
    names = pd.Index([f"Gene{i}" for i in range(matrix.shape[1])])
    genes = names[[2, 7, 11, 19, 31, 44, 58, 77, 91, 105]]
    means = np.asarray(matrix.mean(axis=0)).ravel()

    np.random.seed(0)
    expected = pd.Index([], dtype="string")
    get_subset = lambda selected: matrix[:, names.get_indexer(selected)]
    for selected in _score_genes_bins(
        genes,
        names,
        ctrl_as_ref=True,
        ctrl_size=len(genes),
        n_bins=25,
        get_subset=get_subset,
    ):
        expected = expected.union(selected)

    observed = scanpy_control_genes(
        names,
        means,
        genes,
        ctrl_size=len(genes),
        n_bins=25,
        random_state=0,
        ctrl_as_ref=True,
    )
    assert observed == expected.astype(str).tolist()


def test_streaming_csr_means_and_scores_match_in_memory(tmp_path) -> None:
    matrix = sparse.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 2.0, 0.0],
                [3.0, 0.0, 0.0, 1.0],
                [1.0, 4.0, 0.0, 2.0],
            ],
            dtype=np.float32,
        )
    )
    path = tmp_path / "matrix.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("X")
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr)
        means = stream_gene_means(
            group,
            n_obs=matrix.shape[0],
            n_vars=matrix.shape[1],
            chunk_nnz=2,
        )
        s_weight = np.array([0.5, 0.5, -0.5, -0.5])
        g2m_weight = np.array([-0.5, -0.5, 0.5, 0.5])
        s_score, g2m_score = stream_cell_scores(
            group,
            n_obs=matrix.shape[0],
            n_vars=matrix.shape[1],
            s_weight=s_weight,
            g2m_weight=g2m_weight,
            cells_per_chunk=2,
        )
    np.testing.assert_allclose(means, np.asarray(matrix.mean(axis=0)).ravel())
    np.testing.assert_allclose(s_score, np.asarray(matrix @ s_weight).ravel())
    np.testing.assert_allclose(g2m_score, np.asarray(matrix @ g2m_weight).ravel())
