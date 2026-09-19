#!/usr/bin/env python
"""Small real-data smoke test for the Stage 16 scVI dependency and input path."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from ms_ovary_scrna.project import load_config, project_paths
from ms_ovary_scrna.stage16_ml import (
    _external_marker_subset,
    _load_internal_subset,
    _select_hvg_genes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-data scVI smoke test")
    parser.add_argument("--config", default="config/analysis_config.yaml")
    args = parser.parse_args()

    import scvi
    import torch

    config = load_config(args.config)
    paths = project_paths(config)
    root = paths["root"]
    output = root / "results/deep_dive_stage16_ml/03_scvi_reference"
    output.mkdir(parents=True, exist_ok=True)

    source = root / config["deep_dive_stage15"]["input_object"]
    full = ad.read_h5ad(source, backed="r")
    genes = _select_hvg_genes(full, n_genes=256)
    for gene in ["Foxl2", "Amh", "Inha", "Fshr", "Cyp19a1", "Dcn", "Lum", "Col1a1"]:
        if gene in full.var_names and gene not in genes:
            genes.append(gene)
    full.file.close()

    public_meta = pd.read_csv(
        root
        / "results/deep_dive_stage15/external_gse267729/GSE267729_broad_pseudobulk_metadata.tsv",
        sep="\t",
    )
    public = _external_marker_subset(
        root / "results/deep_dive_stage15/external_gse267729/processed_10x",
        public_meta,
        genes,
        "Granulosa",
        max_per_sample=10,
    )
    query = _load_internal_subset(
        root, config, "Granulosa", genes, max_per_library=10
    )
    combined = ad.concat([public, query], join="inner", merge="same", index_unique=None)
    combined.layers["counts"] = (
        combined.X.copy() if sparse.issparse(combined.X) else sparse.csr_matrix(combined.X)
    )
    values = combined.layers["counts"].data
    if values.size and (
        float(values.min()) < 0 or not np.allclose(values, np.rint(values), atol=1e-6)
    ):
        raise ValueError("scVI smoke input is not non-negative integer UMI counts")
    if public.n_obs == 0 or query.n_obs == 0 or combined.n_vars < 100:
        raise ValueError(
            f"insufficient smoke-test data: public={public.n_obs}, "
            f"query={query.n_obs}, genes={combined.n_vars}"
        )

    scvi.settings.seed = 20260919
    scvi.model.SCVI.setup_anndata(combined, layer="counts", batch_key="dataset")
    model = scvi.model.SCVI(combined, n_latent=3, n_layers=1, gene_likelihood="nb")
    model.train(
        max_epochs=2,
        batch_size=128,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=False,
    )
    latent = model.get_latent_representation()
    payload = {
        "status": "passed",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "scvi": scvi.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_public_cells": int(public.n_obs),
        "n_query_cells": int(query.n_obs),
        "n_genes": int(combined.n_vars),
        "latent_shape": list(latent.shape),
        "counts_min": float(values.min()) if values.size else 0.0,
        "counts_max": float(values.max()) if values.size else 0.0,
    }
    (output / "SCVI_SMOKE_TEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
