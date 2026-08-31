from __future__ import annotations

import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from .project import project_paths, require_compute_resources, setup_logging


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def aggregate_cell_type(
    adata: ad.AnnData,
    cell_type: str,
    *,
    sample_key: str,
    group_key: str,
    cell_type_key: str,
    counts_layer: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_series = adata.obs[sample_key].astype(str)
    cell_series = adata.obs[cell_type_key].astype(str)
    sample_to_group = adata.obs.groupby(sample_key, observed=True)[group_key].first().astype(str)
    records: list[np.ndarray] = []
    sample_names: list[str] = []
    metadata_records: list[dict[str, str]] = []
    for sample in sorted(sample_series.unique()):
        mask = (sample_series == sample) & (cell_series == cell_type)
        if not mask.any():
            continue
        matrix = adata.layers[counts_layer][mask.to_numpy(), :]
        summed = np.asarray(matrix.sum(axis=0)).ravel()
        records.append(np.rint(summed).astype(np.int64))
        sample_names.append(sample)
        metadata_records.append({"sample": sample, "group": sample_to_group.loc[sample]})
    counts = pd.DataFrame(records, index=sample_names, columns=adata.var_names)
    metadata = pd.DataFrame(metadata_records).set_index("sample")
    return counts, metadata


def run_deseq2(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    design: str,
    contrast: list[str],
    alpha: float,
    min_total_counts: int,
    n_cpus: int,
) -> pd.DataFrame:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    keep = counts.sum(axis=0) >= min_total_counts
    counts = counts.loc[:, keep]
    metadata = metadata.loc[counts.index].copy()
    metadata["group"] = pd.Categorical(metadata["group"], categories=["Y", "OC", "OT"])
    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design=design,
        refit_cooks=True,
        inference=inference,
        low_memory=True,
    )
    dds.deseq2()
    stats = DeseqStats(
        dds,
        contrast=contrast,
        alpha=alpha,
        cooks_filter=True,
        independent_filter=True,
        inference=inference,
    )
    stats.summary()
    return stats.results_df.reset_index().rename(columns={"index": "gene"})


def rescue_table(
    aging: pd.DataFrame,
    treatment: pd.DataFrame,
    residual: pd.DataFrame,
    min_abs_aging_lfc: float,
) -> pd.DataFrame:
    merged = aging[["gene", "log2FoldChange", "padj"]].rename(
        columns={"log2FoldChange": "aging_lfc", "padj": "aging_padj"}
    )
    merged = merged.merge(
        treatment[["gene", "log2FoldChange", "padj"]].rename(
            columns={"log2FoldChange": "treatment_lfc", "padj": "treatment_padj"}
        ),
        on="gene",
        how="inner",
    )
    merged = merged.merge(
        residual[["gene", "log2FoldChange", "padj"]].rename(
            columns={"log2FoldChange": "residual_lfc", "padj": "residual_padj"}
        ),
        on="gene",
        how="inner",
    )
    eligible = merged["aging_lfc"].abs() >= min_abs_aging_lfc
    merged["rescue_fraction"] = np.nan
    merged.loc[eligible, "rescue_fraction"] = (
        -merged.loc[eligible, "treatment_lfc"] / merged.loc[eligible, "aging_lfc"]
    )
    merged["opposite_direction"] = (
        np.sign(merged["aging_lfc"]) * np.sign(merged["treatment_lfc"])
    ) < 0
    merged["closer_to_y"] = merged["residual_lfc"].abs() < merged["aging_lfc"].abs()
    merged["rescue_class"] = "not_rescued"
    merged.loc[
        eligible & merged["opposite_direction"] & merged["closer_to_y"],
        "rescue_class",
    ] = "partially_or_fully_rescued"
    merged.loc[
        eligible & merged["opposite_direction"] & (merged["rescue_fraction"] > 1.5),
        "rescue_class",
    ] = "overcorrected"
    return merged.sort_values(["rescue_class", "aging_padj", "treatment_padj"])


def run_pseudobulk(
    config: dict,
    input_path: str | Path,
    *,
    skip_deseq2: bool = False,
    allow_low_memory: bool = False,
) -> Path:
    require_compute_resources(config, allow_low_memory=allow_low_memory)
    paths = project_paths(config)
    logger = setup_logging("05_pseudobulk_de", config)
    settings = config["pseudobulk"]
    counts_layer = config["ingest"]["counts_layer"]
    adata = sc.read_h5ad(input_path)

    required_keys = (
        settings["sample_key"],
        settings["group_key"],
        settings["cell_type_key"],
    )
    for key in required_keys:
        if key not in adata.obs:
            raise KeyError(f"Required obs column missing: {key}")
    if counts_layer not in adata.layers:
        raise KeyError(f"Required layer missing: {counts_layer}")

    cell_counts = pd.crosstab(
        adata.obs[settings["sample_key"]], adata.obs[settings["cell_type_key"]]
    )
    output_root = paths["results"] / "05_pseudobulk"
    output_root.mkdir(parents=True, exist_ok=True)
    cell_counts.to_csv(output_root / "cells_per_sample_cell_type.tsv", sep="\t")

    group_by_sample = adata.obs.groupby(settings["sample_key"], observed=True)[
        settings["group_key"]
    ].first()
    required_groups = {str(level) for contrast in settings["contrasts"] for level in contrast[1:]}
    eligible_cell_types: list[str] = []
    for cell_type in cell_counts.columns.astype(str):
        eligible_samples = cell_counts.index[
            cell_counts[cell_type] >= int(settings["min_cells_per_sample_cell_type"])
        ]
        group_counts = group_by_sample.loc[eligible_samples].value_counts()
        if all(
            group_counts.get(group, 0) >= int(settings["min_samples_per_group"])
            for group in required_groups
        ):
            eligible_cell_types.append(cell_type)
        else:
            logger.warning(
                "Skipping %s: insufficient eligible sample replicates %s",
                cell_type,
                group_counts.to_dict(),
            )

    for cell_type in eligible_cell_types:
        logger.info("Pseudobulk: %s", cell_type)
        counts, metadata = aggregate_cell_type(
            adata,
            cell_type,
            sample_key=settings["sample_key"],
            group_key=settings["group_key"],
            cell_type_key=settings["cell_type_key"],
            counts_layer=counts_layer,
        )
        keep_samples = cell_counts.index[
            cell_counts[cell_type] >= int(settings["min_cells_per_sample_cell_type"])
        ]
        counts = counts.loc[counts.index.intersection(keep_samples)]
        metadata = metadata.loc[counts.index]
        cell_dir = output_root / safe_name(cell_type)
        cell_dir.mkdir(parents=True, exist_ok=True)
        counts.T.to_csv(cell_dir / "counts_genes_by_samples.tsv", sep="\t")
        metadata.to_csv(cell_dir / "sample_metadata.tsv", sep="\t")
        if skip_deseq2:
            continue

        contrast_results: dict[str, pd.DataFrame] = {}
        for contrast in settings["contrasts"]:
            name = f"{contrast[1]}_vs_{contrast[2]}"
            result = run_deseq2(
                counts,
                metadata,
                design=settings["design"],
                contrast=list(contrast),
                alpha=float(settings["alpha"]),
                min_total_counts=int(settings["min_total_counts_gene"]),
                n_cpus=int(settings["n_cpus"]),
            )
            result.to_csv(cell_dir / f"{name}.tsv", sep="\t", index=False)
            contrast_results[name] = result

        rescue = rescue_table(
            contrast_results["OC_vs_Y"],
            contrast_results["OT_vs_OC"],
            contrast_results["OT_vs_Y"],
            float(config["pathway"]["rescue_min_abs_aging_lfc"]),
        )
        rescue.to_csv(cell_dir / "gene_rescue_summary.tsv", sep="\t", index=False)
    logger.info("PSEUDOBULK_OK: %d eligible cell types", len(eligible_cell_types))
    return output_root
