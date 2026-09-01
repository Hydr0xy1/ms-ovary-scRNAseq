from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from .project import project_paths, require_compute_resources, setup_logging
from .qc import mad_outlier, run_scrublet_subset

MT_THRESHOLDS = (5.0, 8.0, 10.0, 15.0, 20.0, 25.0)
MAD_LEVELS = (3.0, 4.0, 5.0)
GROUP_ORDER = ("Y", "OC", "OT")
QUANTILES = (0.90, 0.95, 0.975, 0.99, 0.995)


def _percentage(count: int | float, total: int) -> float:
    return 100.0 * float(count) / total if total else float("nan")


def _library_order(obs: pd.DataFrame) -> list[str]:
    libraries = obs[["library_id", "group"]].drop_duplicates().astype(str)
    library_groups = dict(zip(libraries["library_id"], libraries["group"], strict=True))
    group_rank = {group: index for index, group in enumerate(GROUP_ORDER)}
    return sorted(
        library_groups,
        key=lambda library: (
            group_rank.get(library_groups[library], 99),
            library,
        ),
    )


def metric_distribution(values: pd.Series, prefix: str) -> dict[str, float]:
    numeric = values.astype(float)
    result = {
        f"{prefix}_minimum": float(numeric.min()),
        f"{prefix}_median": float(numeric.median()),
        f"{prefix}_maximum": float(numeric.max()),
    }
    for quantile in QUANTILES:
        label = f"{100 * quantile:g}".replace(".", "_")
        result[f"{prefix}_p{label}"] = float(numeric.quantile(quantile))
    return result


def mad_masks_and_thresholds(
    values: pd.Series,
    n_mads: float,
    *,
    log1p: bool,
) -> dict[str, Any]:
    numeric = values.astype(float)
    transformed = np.log1p(numeric) if log1p else numeric
    median = float(np.nanmedian(transformed))
    mad = float(np.nanmedian(np.abs(transformed - median)))
    low_transformed = median - n_mads * mad
    high_transformed = median + n_mads * mad
    low_threshold = float(np.expm1(low_transformed)) if log1p else low_transformed
    high_threshold = float(np.expm1(high_transformed)) if log1p else high_transformed
    return {
        "low_mask": mad_outlier(transformed, n_mads, "low").to_numpy(dtype=bool),
        "high_mask": mad_outlier(transformed, n_mads, "high").to_numpy(dtype=bool),
        "median_transformed": median,
        "mad_transformed": mad,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
    }


def ensure_top20_metric(adata: ad.AnnData, counts_layer: str) -> None:
    column = "pct_counts_in_top_20_genes"
    if column in adata.obs:
        return
    obs_metrics, _ = sc.pp.calculate_qc_metrics(
        adata,
        percent_top=[20],
        layer=counts_layer,
        log1p=False,
        inplace=False,
    )
    adata.obs[column] = obs_metrics[column].to_numpy()


def mt_threshold_sensitivity(
    obs: pd.DataFrame,
    *,
    mt_n_mads: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    entities: list[tuple[str, str, str | None, pd.DataFrame]] = []
    for library_id, frame in obs.groupby("library_id", observed=True, sort=False):
        entities.append(("library", str(library_id), str(frame["group"].iloc[0]), frame))
    for group, frame in obs.groupby("group", observed=True, sort=False):
        entities.append(("group", str(group), str(group), frame))
    entities.append(("overall", "all", None, obs))

    for entity_type, entity_id, group, frame in entities:
        n_cells = len(frame)
        distribution = metric_distribution(frame["pct_counts_mt"], "mt_pct")
        current_mask = frame["qc_mt_outlier"].astype(bool).to_numpy()
        if entity_type == "library":
            mad_result = mad_masks_and_thresholds(frame["pct_counts_mt"], mt_n_mads, log1p=False)
            threshold = mad_result["high_threshold"]
            matches = bool(np.array_equal(current_mask, mad_result["high_mask"]))
        else:
            threshold = float("nan")
            matches = pd.NA

        for threshold_label, threshold_pct, mask in [
            (f"{mt_n_mads:g}_MAD_library_specific", threshold, current_mask),
            *[
                (
                    f"absolute_{absolute:g}_pct",
                    absolute,
                    frame["pct_counts_mt"].to_numpy() > absolute,
                )
                for absolute in MT_THRESHOLDS
            ],
        ]:
            above_count = int(mask.sum())
            above = frame.loc[mask]
            record: dict[str, Any] = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "group": group,
                "threshold_label": threshold_label,
                "threshold_pct": threshold_pct,
                "n_cells": n_cells,
                "cells_above_threshold": above_count,
                "fraction_above_threshold": above_count / n_cells,
                "pct_above_threshold": _percentage(above_count, n_cells),
                "median_total_counts_above": (
                    float(above["total_counts"].median()) if above_count else float("nan")
                ),
                "median_n_genes_above": (
                    float(above["n_genes_by_counts"].median()) if above_count else float("nan")
                ),
                "current_mad_flag_matches_recomputed": matches,
                **distribution,
            }
            if threshold_label.endswith("MAD_library_specific"):
                below_5 = int((mask & (frame["pct_counts_mt"].to_numpy() < 5)).sum())
                below_8 = int((mask & (frame["pct_counts_mt"].to_numpy() < 8)).sum())
                record.update(
                    {
                        "mad_outlier_but_mt_below_5_count": below_5,
                        "mad_outlier_but_mt_below_5_pct_of_mad": _percentage(below_5, above_count),
                        "mad_outlier_but_mt_below_8_count": below_8,
                        "mad_outlier_but_mt_below_8_pct_of_mad": _percentage(below_8, above_count),
                    }
                )
            else:
                record.update(
                    {
                        "mad_outlier_but_mt_below_5_count": np.nan,
                        "mad_outlier_but_mt_below_5_pct_of_mad": np.nan,
                        "mad_outlier_but_mt_below_8_count": np.nan,
                        "mad_outlier_but_mt_below_8_pct_of_mad": np.nan,
                    }
                )
            records.append(record)

    return pd.DataFrame(records)


def retention_sensitivity(mt_table: pd.DataFrame) -> pd.DataFrame:
    absolute = mt_table[mt_table["threshold_label"].str.startswith("absolute_")].copy()
    absolute["would_remove"] = absolute["cells_above_threshold"]
    absolute["would_retain"] = absolute["n_cells"] - absolute["would_remove"]
    absolute["retained_fraction"] = absolute["would_retain"] / absolute["n_cells"]
    absolute["retained_pct"] = 100 * absolute["retained_fraction"]
    return absolute[
        [
            "entity_type",
            "entity_id",
            "group",
            "threshold_label",
            "threshold_pct",
            "n_cells",
            "would_remove",
            "would_retain",
            "retained_fraction",
            "retained_pct",
        ]
    ]


def count_gene_mad_sensitivity(obs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for library_id, frame in obs.groupby("library_id", observed=True, sort=False):
        n_cells = len(frame)
        genes_lt_200 = frame["n_genes_by_counts"].to_numpy() < 200
        top20 = metric_distribution(frame["pct_counts_in_top_20_genes"], "top20_pct")
        for n_mads in MAD_LEVELS:
            counts = mad_masks_and_thresholds(frame["total_counts"], n_mads, log1p=True)
            genes = mad_masks_and_thresholds(frame["n_genes_by_counts"], n_mads, log1p=True)
            low_counts = counts["low_mask"]
            high_counts = counts["high_mask"]
            low_genes = genes["low_mask"]
            high_genes = genes["high_mask"]
            any_counts = low_counts | high_counts
            any_genes = low_genes | high_genes

            count_fields = {
                "low_count": low_counts,
                "high_count": high_counts,
                "low_gene": low_genes,
                "high_gene": high_genes,
                "genes_lt_200": genes_lt_200,
                "low_count_and_genes_lt_200": low_counts & genes_lt_200,
                "high_count_and_genes_lt_200": high_counts & genes_lt_200,
                "low_gene_and_genes_lt_200": low_genes & genes_lt_200,
                "high_gene_and_genes_lt_200": high_genes & genes_lt_200,
                "any_count_and_any_gene": any_counts & any_genes,
                "low_count_and_low_gene": low_counts & low_genes,
                "low_count_and_high_gene": low_counts & high_genes,
                "high_count_and_low_gene": high_counts & low_genes,
                "high_count_and_high_gene": high_counts & high_genes,
            }
            record: dict[str, Any] = {
                "library_id": str(library_id),
                "group": str(frame["group"].iloc[0]),
                "n_cells": n_cells,
                "n_mads": n_mads,
                "low_count_threshold": counts["low_threshold"],
                "high_count_threshold": counts["high_threshold"],
                "low_gene_threshold": genes["low_threshold"],
                "high_gene_threshold": genes["high_threshold"],
                **top20,
            }
            for name, mask in count_fields.items():
                count = int(mask.sum())
                record[f"{name}_count"] = count
                record[f"{name}_pct"] = _percentage(count, n_cells)
            records.append(record)
    return pd.DataFrame(records)


def _overlap_counts(
    frame: pd.DataFrame,
    *,
    n_mads: float,
    high_mt: np.ndarray,
) -> dict[str, int]:
    counts = mad_masks_and_thresholds(frame["total_counts"], n_mads, log1p=True)
    genes = mad_masks_and_thresholds(frame["n_genes_by_counts"], n_mads, log1p=True)
    low_count = counts["low_mask"]
    high_count = counts["high_mask"]
    low_gene = genes["low_mask"]
    high_gene = genes["high_mask"]
    genes_lt_200 = frame["n_genes_by_counts"].to_numpy() < 200
    predicted = frame["predicted_doublet"].astype(bool).to_numpy()
    any_low = low_count | low_gene | genes_lt_200
    any_high = high_count | high_gene

    masks = {
        "low_count": low_count,
        "high_count": high_count,
        "low_gene": low_gene,
        "high_gene": high_gene,
        "genes_lt_200": genes_lt_200,
        "any_low_rna": any_low,
        "any_high_rna": any_high,
        "high_mt": high_mt,
        "predicted_doublet": predicted,
        "high_mt_and_any_low_rna": high_mt & any_low,
        "high_mt_and_low_count": high_mt & low_count,
        "high_mt_and_low_gene": high_mt & low_gene,
        "high_mt_and_genes_lt_200": high_mt & genes_lt_200,
        "high_mt_and_any_high_rna": high_mt & any_high,
        "high_mt_and_normal_rna": high_mt & ~any_low & ~any_high,
        "high_mt_only_all_other_normal": high_mt & ~any_low & ~any_high & ~predicted,
        "predicted_and_any_high_rna": predicted & any_high,
        "predicted_and_high_count": predicted & high_count,
        "predicted_and_high_gene": predicted & high_gene,
        "predicted_and_any_low_rna": predicted & any_low,
        "predicted_and_normal_rna": predicted & ~any_low & ~any_high,
    }
    return {f"{name}_count": int(mask.sum()) for name, mask in masks.items()}


def flag_overlap_summary(obs: pd.DataFrame) -> pd.DataFrame:
    library_records: list[dict[str, Any]] = []
    for library_id, frame in obs.groupby("library_id", observed=True, sort=False):
        current_mt = frame["qc_mt_outlier"].astype(bool).to_numpy()
        mt_rules = {
            "3_MAD_library_specific": current_mt,
            **{
                f"absolute_{threshold:g}_pct": frame["pct_counts_mt"].to_numpy() > threshold
                for threshold in MT_THRESHOLDS
            },
        }
        for n_mads in MAD_LEVELS:
            for mt_rule, high_mt in mt_rules.items():
                library_records.append(
                    {
                        "entity_type": "library",
                        "entity_id": str(library_id),
                        "group": str(frame["group"].iloc[0]),
                        "n_cells": len(frame),
                        "n_mads": n_mads,
                        "mt_rule": mt_rule,
                        **_overlap_counts(frame, n_mads=n_mads, high_mt=high_mt),
                    }
                )

    table = pd.DataFrame(library_records)
    count_columns = [column for column in table if column.endswith("_count")]
    aggregate_records: list[dict[str, Any]] = []
    for entity_type, key in (("group", "group"), ("overall", None)):
        if key is None:
            grouped = [("all", table)]
        else:
            grouped = list(table.groupby(key, observed=True, sort=False))
        for entity_id, entity_frame in grouped:
            for (n_mads, mt_rule), block in entity_frame.groupby(
                ["n_mads", "mt_rule"], observed=True, sort=False
            ):
                record: dict[str, Any] = {
                    "entity_type": entity_type,
                    "entity_id": str(entity_id),
                    "group": str(entity_id) if entity_type == "group" else None,
                    "n_cells": int(block["n_cells"].sum()),
                    "n_mads": n_mads,
                    "mt_rule": mt_rule,
                }
                record.update({column: int(block[column].sum()) for column in count_columns})
                aggregate_records.append(record)

    result = pd.concat([table, pd.DataFrame(aggregate_records)], ignore_index=True)
    for column in count_columns:
        result[column.removesuffix("_count") + "_pct"] = 100 * result[column] / result["n_cells"]
    result["high_mt_any_low_rna_pct_of_high_mt"] = np.where(
        result["high_mt_count"] > 0,
        100 * result["high_mt_and_any_low_rna_count"] / result["high_mt_count"],
        np.nan,
    )
    result["high_mt_normal_rna_pct_of_high_mt"] = np.where(
        result["high_mt_count"] > 0,
        100 * result["high_mt_and_normal_rna_count"] / result["high_mt_count"],
        np.nan,
    )
    result["predicted_any_high_rna_pct_of_predicted"] = np.where(
        result["predicted_doublet_count"] > 0,
        100 * result["predicted_and_any_high_rna_count"] / result["predicted_doublet_count"],
        np.nan,
    )
    return result


def _histogram_overlap(observed: np.ndarray, simulated: np.ndarray) -> float:
    minimum = float(min(observed.min(), simulated.min()))
    maximum = float(max(observed.max(), simulated.max()))
    if maximum == minimum:
        return 1.0
    bins = np.linspace(minimum, maximum, 101)
    observed_hist, _ = np.histogram(observed, bins=bins)
    simulated_hist, _ = np.histogram(simulated, bins=bins)
    observed_prob = observed_hist / observed_hist.sum()
    simulated_prob = simulated_hist / simulated_hist.sum()
    return float(np.minimum(observed_prob, simulated_prob).sum())


def _scrublet_parameter_snapshot(subset: ad.AnnData, config: dict) -> dict[str, Any]:
    explicit = {
        "expected_doublet_rate": float(config["qc"]["expected_doublet_rate"]),
        "random_state": int(config["project"]["random_seed"]),
    }
    names = (
        "sim_doublet_ratio",
        "expected_doublet_rate",
        "stdev_doublet_rate",
        "synthetic_doublet_umi_subsampling",
        "knn_dist_metric",
        "normalize_variance",
        "log_transform",
        "mean_center",
        "n_prin_comps",
        "use_approx_neighbors",
        "get_doublet_neighbor_parents",
        "n_neighbors",
        "threshold",
        "verbose",
        "random_state",
    )
    signature = inspect.signature(sc.pp.scrublet)
    parameters: dict[str, Any] = {}
    for name in names:
        value = explicit.get(name, signature.parameters[name].default)
        if isinstance(value, (str, int, float, bool)) or value is None:
            parameters[name] = value
        else:
            parameters[name] = str(value)
    parameters.update(subset.uns["scrublet"].get("parameters", {}))
    return parameters


def _save_scrublet_distribution(subset: ad.AnnData, library_id: str, output: Path) -> None:
    plt.close("all")
    try:
        axes = sc.pl.scrublet_score_distribution(subset, show=False)
        if axes is None:
            figure = plt.gcf()
        else:
            first_axis = np.asarray(axes, dtype=object).ravel()[0]
            figure = first_axis.figure
        figure.suptitle(f"{library_id}: observed and simulated Scrublet scores")
    except Exception:
        observed = subset.obs["doublet_score"].to_numpy()
        simulated = np.asarray(subset.uns["scrublet"]["doublet_scores_sim"])
        threshold = float(subset.uns["scrublet"]["threshold"])
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.hist(observed, bins=50, density=True, alpha=0.55, label="observed")
        axis.hist(simulated, bins=50, density=True, alpha=0.55, label="simulated")
        axis.axvline(threshold, color="black", linestyle="--", label="automatic threshold")
        axis.set(xlabel="doublet score", ylabel="density", title=library_id)
        axis.legend()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def scrublet_diagnostics(
    adata: ad.AnnData,
    config: dict,
    figure_dir: Path,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_records: list[dict[str, Any]] = []
    top_records: list[pd.DataFrame] = []
    simulated_records: list[pd.DataFrame] = []
    existing_uns_present = "scrublet" in adata.uns

    for library_id in _library_order(adata.obs):
        mask = (adata.obs["library_id"].astype(str) == library_id).to_numpy()
        frame = adata.obs.loc[mask].copy()
        logger.info("Scrublet sensitivity rerun: %s (%d cells)", library_id, mask.sum())
        subset = run_scrublet_subset(adata, mask, config)
        scrublet_uns = subset.uns.get("scrublet", {})
        if "doublet_scores_sim" not in scrublet_uns or "threshold" not in scrublet_uns:
            raise KeyError(f"{library_id}: Scrublet diagnostics missing simulated scores/threshold")

        observed = subset.obs["doublet_score"].to_numpy(dtype=float)
        simulated = np.asarray(scrublet_uns["doublet_scores_sim"], dtype=float)
        threshold = float(scrublet_uns["threshold"])
        predicted = subset.obs["predicted_doublet"].to_numpy(dtype=bool)
        existing_scores = frame["doublet_score"].to_numpy(dtype=float)
        existing_predicted = frame["predicted_doublet"].to_numpy(dtype=bool)
        parameters = _scrublet_parameter_snapshot(subset, config)

        record: dict[str, Any] = {
            "library_id": library_id,
            "group": str(frame["group"].iloc[0]),
            "n_cells": len(frame),
            "existing_object_has_scrublet_uns": existing_uns_present,
            "automatic_threshold": threshold,
            "predicted_doublets": int(predicted.sum()),
            "predicted_doublet_pct": _percentage(int(predicted.sum()), len(frame)),
            "simulated_n": len(simulated),
            "simulated_above_threshold": int((simulated > threshold).sum()),
            "simulated_above_threshold_pct": _percentage(
                int((simulated > threshold).sum()), len(simulated)
            ),
            "observed_simulated_histogram_overlap": _histogram_overlap(observed, simulated),
            "rerun_scores_match_existing": bool(
                np.allclose(observed, existing_scores, rtol=1e-7, atol=1e-9, equal_nan=True)
            ),
            "rerun_predictions_match_existing": bool(np.array_equal(predicted, existing_predicted)),
            "max_abs_score_difference_from_existing": float(
                np.nanmax(np.abs(observed - existing_scores))
            ),
            "score_spearman_total_counts": float(
                pd.Series(observed).corr(
                    frame["total_counts"].reset_index(drop=True), method="spearman"
                )
            ),
            "score_spearman_n_genes": float(
                pd.Series(observed).corr(
                    frame["n_genes_by_counts"].reset_index(drop=True), method="spearman"
                )
            ),
            "parameters_json": json.dumps(parameters, sort_keys=True),
            **metric_distribution(pd.Series(observed), "observed_score"),
            **metric_distribution(pd.Series(simulated), "simulated_score"),
        }
        top_indices = np.argsort(observed)[-min(100, len(observed)) :][::-1]
        top = frame.iloc[top_indices][
            [
                "library_id",
                "group",
                "doublet_score",
                "predicted_doublet",
                "total_counts",
                "n_genes_by_counts",
                "pct_counts_mt",
            ]
        ].copy()
        top.insert(0, "cell_barcode", top.index.astype(str))
        top.insert(3, "score_rank_within_library", np.arange(1, len(top) + 1))
        top_records.append(top.reset_index(drop=True))
        record.update(
            {
                "all_median_total_counts": float(frame["total_counts"].median()),
                "all_median_n_genes": float(frame["n_genes_by_counts"].median()),
                "top100_median_total_counts": float(top["total_counts"].median()),
                "top100_median_n_genes": float(top["n_genes_by_counts"].median()),
                "top100_total_counts_ratio_to_all": float(
                    top["total_counts"].median() / frame["total_counts"].median()
                ),
                "top100_n_genes_ratio_to_all": float(
                    top["n_genes_by_counts"].median() / frame["n_genes_by_counts"].median()
                ),
            }
        )
        summary_records.append(record)
        simulated_records.append(
            pd.DataFrame({"library_id": library_id, "simulated_doublet_score": simulated})
        )
        _save_scrublet_distribution(
            subset,
            library_id,
            figure_dir / f"scrublet_score_distribution_{library_id}.png",
        )
        del subset

    return (
        pd.DataFrame(summary_records),
        pd.concat(top_records, ignore_index=True),
        pd.concat(simulated_records, ignore_index=True),
    )


def _plot_mt_distributions(obs: pd.DataFrame, mt_table: pd.DataFrame, output_dir: Path) -> None:
    libraries = _library_order(obs)
    colors = sns.color_palette("colorblind", len(MT_THRESHOLDS))
    for full_range in (False, True):
        figure, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
        for index, library_id in enumerate(libraries):
            axis = axes.flat[index]
            frame = obs[obs["library_id"].astype(str) == library_id]
            upper = 100 if full_range else 30
            axis.hist(
                frame["pct_counts_mt"],
                bins=np.linspace(0, upper, 101),
                color="#607d8b",
                alpha=0.8,
            )
            axis.set_yscale("log")
            mad_row = mt_table[
                (mt_table["entity_type"] == "library")
                & (mt_table["entity_id"] == library_id)
                & (mt_table["threshold_label"].str.endswith("MAD_library_specific"))
            ].iloc[0]
            axis.axvline(
                mad_row["threshold_pct"],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="3 MAD",
            )
            for threshold, color in zip(MT_THRESHOLDS, colors, strict=True):
                axis.axvline(
                    threshold,
                    color=color,
                    linewidth=1,
                    alpha=0.9,
                    label=f"{threshold:g}%",
                )
            axis.set(title=library_id, xlabel="pct_counts_mt", ylabel="cell count (log)")
            axis.set_xlim(0, upper)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside upper center", ncol=7)
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        suffix = "full_range" if full_range else "0_to_30_pct"
        figure.savefig(
            output_dir / f"mt_distribution_thresholds_{suffix}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)


def _plot_mt_scatter(obs: pd.DataFrame, metric: str, output: Path) -> None:
    libraries = _library_order(obs)
    figure, axes = plt.subplots(3, 3, figsize=(16, 12))
    for index, library_id in enumerate(libraries):
        axis = axes.flat[index]
        frame = obs[obs["library_id"].astype(str) == library_id]
        axis.scatter(
            frame["pct_counts_mt"],
            frame[metric],
            s=2,
            alpha=0.22,
            color="#37474f",
            linewidths=0,
            rasterized=True,
        )
        axis.set_xscale("symlog", linthresh=5)
        axis.set_yscale("log")
        axis.set(title=library_id, xlabel="pct_counts_mt", ylabel=metric)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_count_gene_candidates(obs: pd.DataFrame, n_mads: float, output: Path) -> None:
    libraries = _library_order(obs)
    figure, axes = plt.subplots(3, 3, figsize=(16, 12))
    candidates = (
        ("low count", "#1565c0"),
        ("high count", "#c62828"),
        ("low gene", "#00acc1"),
        ("high gene", "#ef6c00"),
    )
    for index, library_id in enumerate(libraries):
        axis = axes.flat[index]
        frame = obs[obs["library_id"].astype(str) == library_id]
        count_result = mad_masks_and_thresholds(frame["total_counts"], n_mads, log1p=True)
        gene_result = mad_masks_and_thresholds(frame["n_genes_by_counts"], n_mads, log1p=True)
        masks = (
            count_result["low_mask"],
            count_result["high_mask"],
            gene_result["low_mask"],
            gene_result["high_mask"],
        )
        axis.scatter(
            frame["total_counts"],
            frame["n_genes_by_counts"],
            s=2,
            alpha=0.12,
            color="#9e9e9e",
            linewidths=0,
            rasterized=True,
        )
        for mask, (label, color) in zip(masks, candidates, strict=True):
            axis.scatter(
                frame.loc[mask, "total_counts"],
                frame.loc[mask, "n_genes_by_counts"],
                s=8,
                alpha=0.65,
                color=color,
                linewidths=0,
                label=label,
                rasterized=True,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set(title=library_id, xlabel="total_counts", ylabel="n_genes_by_counts")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=4)
    figure.suptitle(f"Counts/genes candidates at {n_mads:g} MAD", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_doublet_scatter(obs: pd.DataFrame, metric: str, output: Path) -> None:
    libraries = _library_order(obs)
    figure, axes = plt.subplots(3, 3, figsize=(16, 12))
    for index, library_id in enumerate(libraries):
        axis = axes.flat[index]
        frame = obs[obs["library_id"].astype(str) == library_id]
        predicted = frame["predicted_doublet"].astype(bool)
        axis.scatter(
            frame.loc[~predicted, metric],
            frame.loc[~predicted, "doublet_score"],
            s=2,
            alpha=0.2,
            color="#607d8b",
            linewidths=0,
            label="not predicted",
            rasterized=True,
        )
        axis.scatter(
            frame.loc[predicted, metric],
            frame.loc[predicted, "doublet_score"],
            s=8,
            alpha=0.7,
            color="#d32f2f",
            linewidths=0,
            label="predicted doublet",
            rasterized=True,
        )
        axis.set_xscale("log")
        axis.set(title=library_id, xlabel=metric, ylabel="doublet_score")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=2)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_markdown_value(value) for value in row) + " |")
    return "\n".join(lines)


def build_markdown_report(
    adata: ad.AnnData,
    mt_table: pd.DataFrame,
    count_gene_table: pd.DataFrame,
    scrublet_table: pd.DataFrame,
    overlap_table: pd.DataFrame,
    retention_table: pd.DataFrame,
) -> str:
    mt_library = mt_table[
        (mt_table["entity_type"] == "library")
        & (mt_table["threshold_label"].str.endswith("MAD_library_specific"))
    ].copy()
    mt_overall = mt_table[
        (mt_table["entity_type"] == "overall")
        & (mt_table["threshold_label"].str.endswith("MAD_library_specific"))
    ].iloc[0]
    retention_overall = retention_table[retention_table["entity_type"] == "overall"]
    retention_groups = retention_table[retention_table["entity_type"] == "group"]

    median_count_min = mt_library.loc[mt_library["median_total_counts_above"].idxmin()]
    mt_p99_max = mt_library.loc[mt_library["mt_pct_p99"].idxmax()]
    mad_totals = (
        count_gene_table.groupby("n_mads", observed=True)
        .agg(
            n_cells=("n_cells", "sum"),
            low_count=("low_count_count", "sum"),
            high_count=("high_count_count", "sum"),
            low_gene=("low_gene_count", "sum"),
            high_gene=("high_gene_count", "sum"),
            genes_lt_200=("genes_lt_200_count", "sum"),
        )
        .reset_index()
    )
    overall_overlap = overlap_table[overlap_table["entity_type"] == "overall"]
    overlap_view = overall_overlap[
        [
            "n_mads",
            "mt_rule",
            "high_mt_count",
            "high_mt_and_any_low_rna_count",
            "high_mt_and_normal_rna_count",
            "predicted_doublet_count",
            "predicted_and_any_high_rna_count",
        ]
    ]
    scrublet_view = scrublet_table[
        [
            "library_id",
            "automatic_threshold",
            "predicted_doublet_pct",
            "observed_simulated_histogram_overlap",
            "score_spearman_total_counts",
            "score_spearman_n_genes",
            "top100_total_counts_ratio_to_all",
            "top100_n_genes_ratio_to_all",
        ]
    ]

    lines = [
        "# QC sensitivity report",
        "",
        "本报告只描述敏感性结果；未选择最终过滤阈值，也未删除任何细胞。",
        "",
        "## Input state",
        "",
        f"- AnnData shape: `{adata.n_obs} × {adata.n_vars}`",
        "- 输入文件仍为 QC annotated 对象；本流程没有写出 filtered H5AD。",
        "- 当前对象没有保存完整 Scrublet `uns` 诊断信息，因此按原参数逐文库重跑。",
        "",
        "## Mitochondrial distributions",
        "",
        f"当前 3-MAD mt 标记总数：{int(mt_overall['cells_above_threshold'])}。",
        f"其中 mt<5%：{int(mt_overall['mad_outlier_but_mt_below_5_count'])}；"
        f"mt<8%：{int(mt_overall['mad_outlier_but_mt_below_8_count'])}。",
        f"3-MAD mt 候选中位 UMI 最低的文库为 {median_count_min['entity_id']}；"
        f"mt P99 最高的文库为 {mt_p99_max['entity_id']}。",
        "",
        _markdown_table(
            mt_library[
                [
                    "entity_id",
                    "threshold_pct",
                    "cells_above_threshold",
                    "mad_outlier_but_mt_below_5_count",
                    "mad_outlier_but_mt_below_8_count",
                    "mt_pct_median",
                    "mt_pct_p95",
                    "mt_pct_p99",
                    "mt_pct_maximum",
                ]
            ].rename(columns={"entity_id": "library_id"})
        ),
        "",
        "## Absolute mt threshold retention sensitivity",
        "",
        _markdown_table(
            retention_overall[
                ["threshold_pct", "n_cells", "would_remove", "would_retain", "retained_pct"]
            ]
        ),
        "",
        "### Group-level retention",
        "",
        _markdown_table(
            retention_groups[
                ["entity_id", "threshold_pct", "would_remove", "would_retain", "retained_pct"]
            ].rename(columns={"entity_id": "group"})
        ),
        "",
        "## Counts and genes MAD sensitivity",
        "",
        _markdown_table(mad_totals),
        "",
        "## QC flag overlap",
        "",
        _markdown_table(overlap_view),
        "",
        "## Scrublet diagnostics",
        "",
        _markdown_table(scrublet_view),
        "",
        "所有 Scrublet 重跑的 observed scores 和 predicted labels 均在"
        " `scrublet_summary.tsv` 中与现有对象逐文库核对。",
        "",
        "## Scope guard",
        "",
        "本流程没有执行细胞过滤、表达矩阵 normalization/log1p、HVG、PCA、Harmony、"
        "UMAP、Leiden、注释、差异分析或 pseudobulk。",
    ]
    return "\n".join(lines) + "\n"


def run_qc_sensitivity(
    config: dict,
    input_path: str | Path,
    *,
    allow_low_memory: bool = False,
) -> Path:
    require_compute_resources(config, allow_low_memory=allow_low_memory)
    paths = project_paths(config)
    logger = setup_logging("02_qc_sensitivity", config)
    result_dir = paths["results"] / "qc_sensitivity"
    figure_dir = paths["figures"] / "qc_sensitivity"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = paths["results"] / "02_qc_filtered.h5ad"
    input_path = Path(input_path)

    def file_signature(path: Path) -> tuple[int, int] | None:
        if not path.exists():
            return None
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns

    input_signature_before = file_signature(input_path)
    filtered_signature_before = file_signature(filtered_path)

    logger.info("Reading annotated QC object: %s", input_path)
    adata = sc.read_h5ad(input_path)
    required_obs = {
        "library_id",
        "group",
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
        "qc_mt_outlier",
        "doublet_score",
        "predicted_doublet",
    }
    missing = sorted(required_obs - set(adata.obs.columns))
    if missing:
        raise KeyError(f"Annotated QC object is missing obs columns: {missing}")
    if config["ingest"]["counts_layer"] not in adata.layers:
        raise KeyError(f"Missing counts layer: {config['ingest']['counts_layer']}")

    logger.info("Calculating pct_counts_in_top_20_genes without modifying X/counts")
    ensure_top20_metric(adata, config["ingest"]["counts_layer"])
    mt_table = mt_threshold_sensitivity(adata.obs, mt_n_mads=float(config["qc"]["mt_n_mads"]))
    count_gene_table = count_gene_mad_sensitivity(adata.obs)
    overlap_table = flag_overlap_summary(adata.obs)
    retention_table = retention_sensitivity(mt_table)

    logger.info("Rendering mitochondrial and counts/genes sensitivity figures")
    _plot_mt_distributions(adata.obs, mt_table, figure_dir)
    _plot_mt_scatter(
        adata.obs,
        "n_genes_by_counts",
        figure_dir / "mt_pct_vs_n_genes_by_library.png",
    )
    _plot_mt_scatter(
        adata.obs,
        "total_counts",
        figure_dir / "mt_pct_vs_total_counts_by_library.png",
    )
    for n_mads in MAD_LEVELS:
        _plot_count_gene_candidates(
            adata.obs,
            n_mads,
            figure_dir / f"counts_vs_genes_candidates_{n_mads:g}_MAD.png",
        )

    scrublet_table, top_doublets, simulated_scores = scrublet_diagnostics(
        adata, config, figure_dir, logger
    )
    _plot_doublet_scatter(
        adata.obs,
        "total_counts",
        figure_dir / "doublet_score_vs_total_counts_by_library.png",
    )
    _plot_doublet_scatter(
        adata.obs,
        "n_genes_by_counts",
        figure_dir / "doublet_score_vs_n_genes_by_library.png",
    )

    mt_table.to_csv(result_dir / "mt_threshold_sensitivity.tsv", sep="\t", index=False)
    count_gene_table.to_csv(result_dir / "count_gene_mad_sensitivity.tsv", sep="\t", index=False)
    scrublet_table.to_csv(result_dir / "scrublet_summary.tsv", sep="\t", index=False)
    top_doublets.to_csv(result_dir / "high_doublet_score_cells.tsv", sep="\t", index=False)
    simulated_scores.to_csv(
        result_dir / "scrublet_simulated_scores.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    overlap_table.to_csv(result_dir / "flag_overlap_summary.tsv", sep="\t", index=False)
    retention_table.to_csv(result_dir / "retention_sensitivity.tsv", sep="\t", index=False)
    report = build_markdown_report(
        adata,
        mt_table,
        count_gene_table,
        scrublet_table,
        overlap_table,
        retention_table,
    )
    report_path = result_dir / "QC_SENSITIVITY_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    if file_signature(input_path) != input_signature_before:
        raise RuntimeError("Forbidden side effect: input annotated H5AD changed")
    if file_signature(filtered_path) != filtered_signature_before:
        raise RuntimeError("Forbidden side effect: filtered H5AD state changed")
    logger.info("QC_SENSITIVITY_OK: %s", report_path)
    return report_path
