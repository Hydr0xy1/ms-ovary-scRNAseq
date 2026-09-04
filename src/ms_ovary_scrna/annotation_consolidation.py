from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from .project import project_paths, require_compute_resources, setup_logging
from .stage1_exploratory import sparse_matrix_audit

NEW_COLUMNS = [
    "cell_type_broad_v1",
    "cell_type_subtype_v1",
    "cell_state_v1",
    "annotation_confidence_v1",
    "annotation_source_v1",
    "qc_concern_v1",
    "doublet_concern_v1",
    "analysis_tier_v1",
]
TIER_ORDER = [
    "Tier1_primary",
    "Tier2_sensitivity",
    "Tier3_descriptive_only",
    "Tier4_exclude_candidate",
]
ANNOTATION_FIELDS = [
    "cell_type_broad_v1",
    "cell_type_subtype_v1",
    "cell_state_v1",
    "annotation_confidence_v1",
    "analysis_tier_v1",
]


def _hash_index(index: pd.Index) -> str:
    digest = hashlib.sha256()
    for value in index.astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _matrix_digest(matrix: object) -> str:
    """Digest sparse arrays after normalizing the representation to CSR."""
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    return str(sparse_matrix_audit(matrix)["sha256_sparse_arrays"])


def _as_bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _first_existing_bool(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(False, index=frame.index, dtype=bool)
    for column in columns:
        result |= _as_bool_series(frame, column)
    return result


def _base_qc_concern(obs: pd.DataFrame) -> pd.Series:
    concern = pd.Series("None", index=obs.index, dtype="string")
    low_quality = _first_existing_bool(
        obs, ["qc_low_quality_strong", "qc_fail", "qc_low_genes_absolute"]
    )
    mt_extreme = _first_existing_bool(obs, ["qc_mt_extreme", "retained_extreme_mt"])
    mt_moderate = _first_existing_bool(obs, ["qc_mt_moderate"])
    low_genes = _first_existing_bool(obs, ["qc_low_genes_5mad"])
    concern.loc[low_genes] = "Low_complexity"
    concern.loc[mt_moderate] = "Mitochondrial_high"
    concern.loc[mt_extreme] = "Mitochondrial_extreme"
    concern.loc[low_quality] = "Low_quality_strong"
    return concern


def _local_series(local_obs: pd.DataFrame, column: str, default: str) -> pd.Series:
    if column not in local_obs:
        return pd.Series(default, index=local_obs.index, dtype="string")
    return local_obs[column].astype("string").fillna(default)


def _load_cluster8_review(path: Path) -> pd.DataFrame:
    review = pd.read_csv(path, sep="\t", dtype={"cell_barcode": str})
    if "cell_barcode" not in review:
        raise KeyError("cluster8 review table must contain cell_barcode")
    review = review.set_index("cell_barcode", drop=False)
    if review.index.has_duplicates:
        raise ValueError("cluster8 review table contains duplicate cell barcodes")
    required = {
        "cluster8_single_cell_class",
        "heterotypic_doublet_supported",
        "predicted_doublet",
    }
    missing = sorted(required - set(review.columns))
    if missing:
        raise KeyError(f"cluster8 review table is missing columns: {missing}")
    return review


def _validate_local_mapping(
    main: ad.AnnData, local: ad.AnnData, local_labels: pd.DataFrame
) -> dict[str, Any]:
    main_names = pd.Index(main.obs_names.astype(str))
    local_names = pd.Index(local.obs_names.astype(str))
    label_names = pd.Index(local_labels.index.astype(str))
    if main_names.has_duplicates:
        raise ValueError("Main AnnData contains duplicate cell barcodes")
    if local_names.has_duplicates:
        raise ValueError("Local AnnData contains duplicate cell barcodes")
    if label_names.has_duplicates:
        raise ValueError("Local cell-label table contains duplicate cell barcodes")
    if not local_names.isin(main_names).all():
        missing = local_names[~local_names.isin(main_names)].tolist()[:10]
        raise ValueError(f"Local barcodes missing from main object, examples={missing}")
    if set(label_names) != set(local_names):
        missing = sorted(set(local_names) - set(label_names))[:10]
        extra = sorted(set(label_names) - set(local_names))[:10]
        raise ValueError(f"Local label barcode mismatch: missing={missing}, extra={extra}")
    return {
        "main_n_cells": int(main.n_obs),
        "local_n_cells": int(local.n_obs),
        "local_labels_n_cells": int(local_labels.shape[0]),
        "main_barcodes_unique": True,
        "local_barcodes_unique": True,
        "local_labels_barcodes_unique": True,
        "local_barcodes_all_in_main": True,
        "local_labels_exactly_match_local_object": True,
    }


def _apply_local_labels(
    main: ad.AnnData,
    local: ad.AnnData,
    local_labels: pd.DataFrame,
    cluster8_review: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create v1 labels on a copy of main and return the local audit table."""
    obs = main.obs
    # Start from the already-reviewed global broad annotation for non-local cells.
    broad = obs.get("cell_type_broad", pd.Series("Uncertain", index=obs.index))
    broad = broad.astype("string").fillna("Uncertain")
    subtype = pd.Series("Unresolved", index=obs.index, dtype="string")
    state = (
        obs.get("cell_state_provisional", pd.Series("None", index=obs.index))
        .astype("string")
        .fillna("None")
    )
    confidence = (
        obs.get("annotation_confidence", pd.Series("Uncertain", index=obs.index))
        .astype("string")
        .fillna("Uncertain")
    )
    source = pd.Series("global_broad_review", index=obs.index, dtype="string")
    qc = _base_qc_concern(obs)
    doublet = _first_existing_bool(
        obs,
        ["predicted_doublet", "qc_doublet_auto", "qc_high_doublet_score_top1pct"],
    )
    local_names = pd.Index(local.obs_names.astype(str))
    local_labels = local_labels.copy()
    local_labels.index = local_labels.index.astype(str)
    local_obs = local.obs.copy()
    local_obs.index = local_obs.index.astype(str)
    cluster8_review = cluster8_review.copy()
    cluster8_review.index = cluster8_review.index.astype(str)
    local_barcodes = local_names.intersection(obs.index.astype(str))
    if len(local_barcodes) != local.n_obs:
        raise ValueError("Local barcode intersection unexpectedly changed")

    # Align every assignment by barcode, never by row position.
    local_cluster = _local_series(local_obs, "follicular_leiden", "Unknown")
    original_cluster = _local_series(local_obs, "original_leiden_cluster", "Unknown")
    local_broad = _local_series(local_labels, "cell_type_broad_local", "Uncertain")
    local_subtype = _local_series(local_labels, "cell_type_subtype_local", "Unresolved")
    local_state = _local_series(local_labels, "cell_state_local", "None")
    local_confidence = _local_series(local_labels, "annotation_confidence_local", "Uncertain")
    local_predicted_doublet = _as_bool_series(local_labels, "predicted_doublet")
    local_cluster8 = cluster8_review.reindex(local_barcodes)
    if local_cluster8["cluster8_single_cell_class"].isna().any():
        # Only cluster-8 cells require this table; missing non-cluster-8 rows
        # are expected in the full review table.
        needed = local_cluster.eq("8")
        if local_cluster8.loc[needed, "cluster8_single_cell_class"].isna().any():
            raise ValueError("Missing cluster-8 single-cell review rows")

    # Assign the ordinary local labels first, then apply the explicit exceptions.
    for barcode in local_barcodes:
        broad.loc[barcode] = local_broad.loc[barcode]
        subtype.loc[barcode] = local_subtype.loc[barcode]
        state.loc[barcode] = local_state.loc[barcode]
        confidence.loc[barcode] = local_confidence.loc[barcode]
        source.loc[barcode] = "follicular_local_annotation_review"
        if local_cluster.loc[barcode] in {"5", "6", "9"}:
            qc.loc[barcode] = "Low_complexity"
        elif local_cluster.loc[barcode] == "13":
            qc.loc[barcode] = "Stress_high"
        elif local_cluster.loc[barcode] == "14":
            qc.loc[barcode] = "Rare_population"
        doublet.loc[barcode] = local_predicted_doublet.loc[barcode]

    cluster8_barcodes = local_barcodes[local_cluster.loc[local_barcodes].eq("8")]
    for barcode in cluster8_barcodes:
        row = local_cluster8.loc[barcode]
        category = str(row["cluster8_single_cell_class"])
        supported = bool(
            _as_bool_series(pd.DataFrame([row]), "heterotypic_doublet_supported").iat[0]
        )
        predicted = bool(local_predicted_doublet.loc[barcode])
        if category == "stromal_only_like":
            broad.loc[barcode] = "Stromal_fibroblast"
            subtype.loc[barcode] = "Stromal_fibroblast_candidate"
            state.loc[barcode] = "None"
            confidence.loc[barcode] = "Medium"
            source.loc[barcode] = "follicular_cluster8_single_cell_review"
            doublet.loc[barcode] = predicted
        elif category in {"stromal_plus_granulosa", "stromal_plus_theca"}:
            source.loc[barcode] = "follicular_cluster8_single_cell_review"
            if supported:
                broad.loc[barcode] = "Mixed"
                subtype.loc[barcode] = "Mixed_lineage_doublet_suspicious"
                state.loc[barcode] = "Doublet_suspicious"
                confidence.loc[barcode] = "Low"
                doublet.loc[barcode] = True
            else:
                broad.loc[barcode] = "Uncertain"
                subtype.loc[barcode] = "Mixed_lineage_uncertain"
                state.loc[barcode] = "None"
                confidence.loc[barcode] = "Low"
                doublet.loc[barcode] = predicted
        elif category == "uncertain":
            broad.loc[barcode] = "Uncertain"
            subtype.loc[barcode] = "Uncertain"
            state.loc[barcode] = "None"
            confidence.loc[barcode] = "Low"
            source.loc[barcode] = "follicular_cluster8_single_cell_review"
            doublet.loc[barcode] = predicted
        else:
            raise ValueError(f"Unexpected cluster-8 class: {category}")

    # The original global cluster 24 takes precedence over local cluster labels,
    # including the three cluster-24 cells that sit inside local cluster 8.
    global24 = original_cluster.eq("24")
    for barcode in local_barcodes[global24.loc[local_barcodes]]:
        broad.loc[barcode] = "Theca_steroidogenic"
        subtype.loc[barcode] = "Theca_cycling"
        state.loc[barcode] = "Cycling"
        confidence.loc[barcode] = "Medium"
        source.loc[barcode] = "global_cluster24_review"
        doublet.loc[barcode] = local_predicted_doublet.loc[barcode]

    global25 = original_cluster.eq("25")
    for barcode in local_barcodes[global25.loc[local_barcodes]]:
        broad.loc[barcode] = "Luteal"
        subtype.loc[barcode] = "Rare_luteal_candidate"
        state.loc[barcode] = "None"
        confidence.loc[barcode] = "Low"
        source.loc[barcode] = "global_cluster25_rare_review"
        qc.loc[barcode] = "Rare_population"
        doublet.loc[barcode] = local_predicted_doublet.loc[barcode]

    # Local cluster 10 is luteal-like, but only global cluster 24 carries the
    # explicitly reviewed cycling-theca state.
    cluster10 = local_cluster.eq("10") & ~global24
    for barcode in local_barcodes[cluster10.loc[local_barcodes]]:
        broad.loc[barcode] = "Luteal"
        subtype.loc[barcode] = "Luteal_like"
        state.loc[barcode] = "None"
        source.loc[barcode] = "follicular_cluster10_state_curation"

    # Local cluster 12 remains deliberately unresolved at the immune subtype level.
    cluster12 = local_cluster.eq("12")
    for barcode in local_barcodes[cluster12.loc[local_barcodes]]:
        broad.loc[barcode] = "Uncertain_immune_related"
        subtype.loc[barcode] = "Mixed_lineage_suspicious"
        state.loc[barcode] = "Doublet_suspicious"
        confidence.loc[barcode] = "Low"
        source.loc[barcode] = "follicular_cluster12_uncertain_review"
        doublet.loc[barcode] = local_predicted_doublet.loc[barcode]

    result = pd.DataFrame(index=obs.index)
    result["cell_type_broad_v1"] = broad.astype("string")
    result["cell_type_subtype_v1"] = subtype.astype("string")
    result["cell_state_v1"] = state.astype("string")
    result["annotation_confidence_v1"] = confidence.astype("string")
    result["annotation_source_v1"] = source.astype("string")
    result["qc_concern_v1"] = qc.astype("string")
    result["doublet_concern_v1"] = doublet.astype(bool)

    # Tiering is an analysis-eligibility flag only; it never subsets the object.
    tier = pd.Series("Tier2_sensitivity", index=obs.index, dtype="string")
    primary = (
        result["annotation_confidence_v1"].isin(["High", "Medium"])
        & ~result["doublet_concern_v1"]
        & result["qc_concern_v1"].eq("None")
        & ~result["cell_type_broad_v1"].str.startswith("Uncertain")
        & ~result["cell_type_broad_v1"].eq("Mixed")
    )
    tier.loc[primary] = "Tier1_primary"
    descriptive = result["cell_type_subtype_v1"].isin(["Oocyte", "Rare_luteal_candidate"]) | result[
        "qc_concern_v1"
    ].eq("Rare_population")
    tier.loc[descriptive] = "Tier3_descriptive_only"
    supported_mixed = (
        result["cell_type_subtype_v1"].eq("Mixed_lineage_doublet_suspicious")
        & result["doublet_concern_v1"]
    )
    tier.loc[supported_mixed] = "Tier4_exclude_candidate"
    # Cluster 12 is explicitly mixed-lineage; only cells with direct doublet
    # evidence are Tier 4, while the remainder stay in sensitivity review.
    cluster12_global = pd.Series(False, index=obs.index, dtype=bool)
    cluster12_global.loc[local_barcodes] = cluster12.loc[local_barcodes]
    tier.loc[cluster12_global & result["doublet_concern_v1"]] = "Tier4_exclude_candidate"
    result["analysis_tier_v1"] = tier
    result.index = main.obs.index

    # The audit table is one row per local cell and records exact final labels.
    local_audit = result.loc[local_barcodes, NEW_COLUMNS].copy()
    local_audit.insert(0, "cell_barcode", local_audit.index.astype(str))
    local_audit.insert(1, "follicular_leiden", local_cluster.loc[local_barcodes].to_numpy())
    local_audit.insert(
        2, "original_leiden_cluster", original_cluster.loc[local_barcodes].to_numpy()
    )
    local_audit["mapping_status"] = "mapped"
    return result, local_audit


def _census_table(obs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [("all", "all", obs)]
    for library_id, frame in obs.groupby("library_id", observed=True, sort=True):
        scopes.append(("library", str(library_id), frame))
    for group, frame in obs.groupby("group", observed=True, sort=True):
        scopes.append(("group", str(group), frame))
    for scope, scope_value, frame in scopes:
        denominator = len(frame)
        for field in ANNOTATION_FIELDS:
            counts = frame[field].astype(str).value_counts(dropna=False).sort_index()
            for label, count in counts.items():
                rows.append(
                    {
                        "scope": scope,
                        "scope_value": scope_value,
                        "field": field,
                        "label": str(label),
                        "n_cells": int(count),
                        "fraction_within_scope": float(count / denominator) if denominator else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _tier_summary(obs: pd.DataFrame) -> pd.DataFrame:
    scopes = [("all", "all", obs)]
    for library_id, frame in obs.groupby("library_id", observed=True, sort=True):
        scopes.append(("library", str(library_id), frame))
    for group, frame in obs.groupby("group", observed=True, sort=True):
        scopes.append(("group", str(group), frame))
    rows: list[dict[str, Any]] = []
    for scope, scope_value, frame in scopes:
        counts = frame["analysis_tier_v1"].astype(str).value_counts()
        for tier in TIER_ORDER:
            count = int(counts.get(tier, 0))
            rows.append(
                {
                    "scope": scope,
                    "scope_value": scope_value,
                    "analysis_tier_v1": tier,
                    "n_cells": count,
                    "fraction_within_scope": float(count / len(frame)) if len(frame) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _audit_rows(checks: dict[str, tuple[Any, Any, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": name,
                "expected": str(expected),
                "observed": str(observed),
                "passed": bool(expected == observed),
                "details": details,
            }
            for name, (expected, observed, details) in checks.items()
        ]
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            text = "" if pd.isna(value) else str(value)
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_report(
    output: Path,
    main: ad.AnnData,
    local_audit: pd.DataFrame,
    audit: pd.DataFrame,
    census: pd.DataFrame,
    tiers: pd.DataFrame,
) -> None:
    local8 = local_audit[local_audit["follicular_leiden"].astype(str).eq("8")].copy()
    global24 = local_audit[local_audit["original_leiden_cluster"].astype(str).eq("24")]
    global25 = local_audit[local_audit["original_leiden_cluster"].astype(str).eq("25")]
    oocyte = local_audit[local_audit["cell_type_subtype_v1"].eq("Oocyte")]
    cluster8_final = pd.Series("uncertain", index=local8.index, dtype="string")
    cluster8_final.loc[local8["cell_type_subtype_v1"].eq("Stromal_fibroblast_candidate")] = (
        "stromal-like"
    )
    cluster8_final.loc[local8["cell_type_subtype_v1"].eq("Mixed_lineage_doublet_suspicious")] = (
        "mixed doublet-supported"
    )
    cluster8_final.loc[local8["cell_type_subtype_v1"].eq("Mixed_lineage_uncertain")] = (
        "mixed uncertain"
    )
    cluster8_final.loc[local8["original_leiden_cluster"].astype(str).eq("24")] = (
        "global24 cycling theca override"
    )
    cluster8_counts = (
        cluster8_final.value_counts()
        .rename_axis("final_cluster8_class")
        .rename("n_cells")
        .reset_index()
    )
    broad_counts = (
        main.obs["cell_type_broad_v1"]
        .astype(str)
        .value_counts()
        .rename_axis("cell_type_broad_v1")
        .rename("n_cells")
        .reset_index()
    )
    subtype_counts = (
        main.obs["cell_type_subtype_v1"]
        .astype(str)
        .value_counts()
        .rename_axis("cell_type_subtype_v1")
        .rename("n_cells")
        .reset_index()
    )
    tier_counts = (
        main.obs["analysis_tier_v1"]
        .astype(str)
        .value_counts()
        .reindex(TIER_ORDER, fill_value=0)
        .rename_axis("analysis_tier_v1")
        .rename("n_cells")
        .reset_index()
    )
    lines = [
        "# Annotation v1 consolidation report",
        "",
        (
            "This is metadata harmonization only. No cells were deleted, no "
            "expression values were transformed, and no Y/OC/OT biological "
            "comparison was performed."
        ),
        "",
        "## Integrity and barcode mapping",
        "",
        _markdown_table(audit),
        "",
        (
            f"All {len(local_audit):,} local follicular/steroidogenic cells "
            "are represented in the barcode-level audit table. The output "
            f"object remains {main.n_obs:,} cells × {main.n_vars:,} features."
        ),
        "",
        "## Final broad labels",
        "",
        _markdown_table(broad_counts),
        "",
        "## Final subtype labels",
        "",
        _markdown_table(subtype_counts),
        "",
        "## Local cluster 8 final classes",
        "",
        _markdown_table(cluster8_counts),
        "",
        (
            "The three global-cluster-24 cells located inside local cluster 8 "
            "are reported as an explicit cycling-theca override rather than "
            f"as cluster-8 mixed cells. Global cluster 24 has {len(global24):,} "
            "cells total."
        ),
        "",
        "## Special populations",
        "",
        f"- Global cluster 24: {len(global24):,} cells, Theca_cycling/Cycling.",
        (
            f"- Global cluster 25: {len(global25):,} cells, "
            "Rare_luteal_candidate, Tier3_descriptive_only."
        ),
        f"- Oocyte: {len(oocyte):,} cells, Tier3_descriptive_only.",
        "",
        "## Analysis tiers",
        "",
        _markdown_table(tier_counts),
        "",
        (
            "Tier labels describe downstream eligibility only. Tier4 cells "
            "remain in the AnnData object and are not deleted."
        ),
        "",
        "## Scope boundary",
        "",
        (
            "Non-follicular cells retain the audited global broad annotation "
            "and receive `Unresolved` subtype until their own subclustering "
            "review. This stage does not run normalization, DEG, pseudobulk, "
            "GSEA, trajectory, CellChat or SCENIC, and does not proceed "
            "automatically to the next analysis step."
        ),
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _atomic_write(adata: ad.AnnData, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.h5ad")
    adata.write_h5ad(temporary, compression="gzip")
    temporary.replace(output)


def run_annotation_consolidation(config: dict[str, Any], *, allow_low_memory: bool = False) -> Path:
    """Consolidate reviewed local labels into a new main AnnData object."""
    require_compute_resources(config, allow_low_memory=allow_low_memory)
    paths = project_paths(config)
    logger = setup_logging("06_annotation_consolidation", config)
    settings = config["annotation_consolidation"]
    root = paths["root"]
    input_path = (root / settings["input_object"]).resolve()
    local_path = (root / settings["local_object"]).resolve()
    labels_path = (root / settings["local_cell_labels"]).resolve()
    cluster8_path = (root / settings["cluster8_review"]).resolve()
    output_path = (root / settings["output_object"]).resolve()
    output_dir = (root / settings["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    counts_layer = str(settings["counts_layer"])

    logger.info("Reading main object: %s", input_path)
    main = sc.read_h5ad(input_path)
    logger.info("Reading reviewed local object: %s", local_path)
    local = sc.read_h5ad(local_path)
    labels = pd.read_csv(labels_path, sep="\t", dtype={"cell_barcode": str})
    if "cell_barcode" not in labels:
        raise KeyError("Local cell-label table must contain cell_barcode")
    labels = labels.set_index("cell_barcode", drop=False)
    cluster8_review = _load_cluster8_review(cluster8_path)
    if counts_layer not in main.layers or counts_layer not in local.layers:
        raise KeyError(f"Missing counts layer: {counts_layer}")
    if main.n_vars != local.n_vars or not np.array_equal(
        main.var_names.astype(str), local.var_names.astype(str)
    ):
        raise ValueError("Main/local feature names or order are not identical")
    _validate_local_mapping(main, local, labels)
    if len(cluster8_review) != 1461:
        raise ValueError(f"Expected 1461 cluster-8 review rows, observed {len(cluster8_review)}")

    before = {
        "shape": tuple(main.shape),
        "x_digest": _matrix_digest(main.X),
        "counts_digest": _matrix_digest(main.layers[counts_layer]),
        "obs_names_digest": _hash_index(main.obs_names),
        "library_counts": main.obs["library_id"].astype(str).value_counts().sort_index().to_dict(),
        "group_counts": main.obs["group"].astype(str).value_counts().sort_index().to_dict(),
    }
    label_frame, local_audit = _apply_local_labels(main, local, labels, cluster8_review)
    for column in NEW_COLUMNS:
        main.obs[column] = label_frame[column].to_numpy()
    main.obs["analysis_tier_v1"] = pd.Categorical(
        main.obs["analysis_tier_v1"], categories=TIER_ORDER, ordered=True
    )
    main.uns["annotation_consolidation_v1"] = {
        "source_main_object": str(input_path),
        "source_local_object": str(local_path),
        "no_cells_deleted": True,
        "no_condition_comparison": True,
        "barcode_mapping": "obs_names_exact_match",
        "new_columns": NEW_COLUMNS,
        "scope": "annotation consolidation only",
    }

    census = _census_table(main.obs)
    by_library = census[census["scope"].eq("library")].copy()
    tiers = _tier_summary(main.obs)
    audit_checks = {
        "local_cells_mapped": (
            int(local.n_obs),
            int(local_audit.shape[0]),
            "barcode-level local audit rows",
        ),
        "local_mapping_percent": (
            100.0,
            round(100.0 * len(local_audit) / local.n_obs, 8),
            "all local cells mapped",
        ),
        "main_shape": (before["shape"], tuple(main.shape), "main shape before/after label columns"),
        "output_expected_shape": (
            before["shape"],
            before["shape"],
            "output is written with unchanged shape",
        ),
        "cluster8_review_rows": (1461, len(cluster8_review), "single-cell review rows"),
        "global24_cells": (
            107,
            int(local_audit["original_leiden_cluster"].astype(str).eq("24").sum()),
            "explicit cycling-theca override",
        ),
        "global25_cells": (
            16,
            int(local_audit["original_leiden_cluster"].astype(str).eq("25").sum()),
            "rare luteal candidate",
        ),
        "oocyte_cells": (
            6,
            int(local_audit["cell_type_subtype_v1"].eq("Oocyte").sum()),
            "descriptive-only population",
        ),
    }
    audit = _audit_rows(audit_checks)
    census.to_csv(output_dir / "annotation_census.tsv", sep="\t", index=False)
    by_library.to_csv(output_dir / "annotation_by_library.tsv", sep="\t", index=False)
    tiers.to_csv(output_dir / "analysis_tier_summary.tsv", sep="\t", index=False)
    local_audit.to_csv(output_dir / "annotation_mapping_cell_audit.tsv", sep="\t", index=False)
    # ``annotation_mapping_audit.tsv`` is the compact pass/fail audit; the
    # barcode-level mapping is retained separately above for direct review.
    audit.to_csv(output_dir / "annotation_mapping_audit.tsv", sep="\t", index=False)
    _write_report(output_dir / "ANNOTATION_V1_REPORT.md", main, local_audit, audit, census, tiers)

    logger.info("Writing new consolidated object: %s", output_path)
    _atomic_write(main, output_path)
    saved = sc.read_h5ad(output_path)
    after = {
        "shape": tuple(saved.shape),
        "x_digest": _matrix_digest(saved.X),
        "counts_digest": _matrix_digest(saved.layers[counts_layer]),
        "obs_names_digest": _hash_index(saved.obs_names),
        "library_counts": saved.obs["library_id"].astype(str).value_counts().sort_index().to_dict(),
        "group_counts": saved.obs["group"].astype(str).value_counts().sort_index().to_dict(),
    }
    final_checks = {
        "output_shape_unchanged": (
            before["shape"],
            after["shape"],
            "shape remains cells x features",
        ),
        "X_digest_unchanged": (before["x_digest"], after["x_digest"], "expression matrix"),
        "counts_digest_unchanged": (
            before["counts_digest"],
            after["counts_digest"],
            "raw counts layer",
        ),
        "obs_names_unchanged": (
            before["obs_names_digest"],
            after["obs_names_digest"],
            "cell barcode order",
        ),
        "library_counts_unchanged": (
            before["library_counts"],
            after["library_counts"],
            "library composition",
        ),
        "group_counts_unchanged": (
            before["group_counts"],
            after["group_counts"],
            "group composition",
        ),
        "new_columns_present": (
            set(NEW_COLUMNS),
            set(saved.obs.columns).intersection(NEW_COLUMNS),
            "v1 metadata columns",
        ),
    }
    final_audit = _audit_rows(final_checks)
    audit = pd.concat([audit, final_audit], ignore_index=True)
    audit.to_csv(output_dir / "annotation_mapping_audit.tsv", sep="\t", index=False)
    if not final_audit["passed"].all():
        raise RuntimeError("Annotation consolidation integrity validation failed")
    if not audit["passed"].all():
        raise RuntimeError("Annotation mapping audit contains failed checks")
    logger.info("ANNOTATION_CONSOLIDATION_OK: %s", output_path)
    return output_path
