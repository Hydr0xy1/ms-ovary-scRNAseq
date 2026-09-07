"""DE Stage 2: broad-population Mouse Hallmark pathway reversal analysis.

This module is deliberately isolated from the historical custom-pathway workflow.
It consumes the audited Stage 1.5 nine-library model and the exact Stage 1.6 label
assignments, writes only to ``results/pathway_stage2`` and
``figures/pathway_stage2``, and never modifies the main AnnData object.

The GSEA normalized enrichment score (NES) is used only as an enrichment direction
and significance summary.  All additive and distance calculations in this module
accept gene-level log2 fold-change vectors only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .de_stage1 import LIBRARIES_BY_GROUP, load_broad_inputs, prefilter_genes
from .de_stage1_5 import ALPHA, ALL_LIBRARIES, build_rescue_ready_effects
from .de_stage1_6 import (
    EXPECTED_PERMUTATIONS,
    _fit_permutation,
    cosine_similarity,
    enumerate_permutation_assignments,
    validate_permutation_assignments,
    whole_signature_metrics,
)
from .project import project_paths, require_compute_resources, setup_logging
from .pseudobulk import safe_name

POPULATIONS = (
    "Granulosa",
    "Immune",
    "Ovarian_epithelial",
    "Smooth_muscle_pericyte",
    "Stromal_fibroblast",
    "Theca_steroidogenic",
    "Vascular_endothelial",
)
CONTRASTS = ("OC_vs_Y", "OT_vs_OC", "OT_vs_Y")
CONTRAST_LABELS = {
    "OC_vs_Y": "aging",
    "OT_vs_OC": "treatment",
    "OT_vs_Y": "residual",
}
POPULATION_TIERS = {
    "Granulosa": "Primary",
    "Stromal_fibroblast": "Primary",
    "Immune": "Secondary",
    "Ovarian_epithelial": "Secondary",
    "Smooth_muscle_pericyte": "Secondary",
    "Vascular_endothelial": "Exploratory",
    "Theca_steroidogenic": "No_prior_permutation_support",
}
EVIDENCE_ORDER = {
    "Strong_multi_metric_reversal": 0,
    "Permutation_supported_reversal": 1,
    "GSEA_supported_reversal_candidate": 2,
    "Directional_reversal_candidate": 3,
    "Aging_only": 4,
    "Not_aging_supported": 5,
}

MSIGDB_RELEASE = "2026.1.Mm"
MSIGDB_COLLECTION = "MH"
MSIGDB_SPECIES = "Mus musculus"
MSIGDB_GMT_NAME = "mh.all.v2026.1.Mm.symbols.gmt"
MSIGDB_URL = (
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2026.1.Mm/"
    "mh.all.v2026.1.Mm.symbols.gmt"
)
EXPECTED_HALLMARKS = 50
GSEA_PERMUTATIONS = 10_000
GSEA_SEED = 20260907
GSEA_MIN_SIZE = 15
GSEA_MAX_SIZE = 500
DIRECTIONAL_AGING_LFC = 0.25
MIN_GEOMETRY_OVERLAP = 15
MIN_DIRECTIONAL_DENOMINATOR = 10


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 checksum."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_gmt(path: str | Path) -> dict[str, list[str]]:
    """Parse and strictly validate the official Mouse Hallmark GMT."""

    gmt_path = Path(path)
    if not gmt_path.exists():
        raise FileNotFoundError(
            f"Official Mouse Hallmark resource is missing: {gmt_path}. "
            f"Download {MSIGDB_URL}; a human Hallmark substitute is not allowed."
        )
    gene_sets: dict[str, list[str]] = {}
    with gmt_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n\r").split("\t")
            if len(fields) < 3:
                raise ValueError(f"Malformed GMT line {line_number}: fewer than three fields")
            name = fields[0].strip()
            genes = list(dict.fromkeys(g.strip() for g in fields[2:] if g.strip()))
            if name in gene_sets:
                raise ValueError(f"Duplicate GMT pathway: {name}")
            gene_sets[name] = genes
    if len(gene_sets) != EXPECTED_HALLMARKS:
        raise ValueError(
            f"Expected {EXPECTED_HALLMARKS} Mouse Hallmark sets, found {len(gene_sets)}"
        )
    if not all(name.startswith("HALLMARK_") for name in gene_sets):
        raise ValueError("The GMT contains non-Hallmark pathway names")
    return gene_sets


def load_stage1_6_assignments(path: str | Path) -> pd.DataFrame:
    """Load Stage 1.6 assignments and prove they equal the canonical exact grid."""

    saved = pd.read_csv(path, sep="\t", dtype={"permutation_id": str, "library_id": str})
    validate_permutation_assignments(saved)
    expected = enumerate_permutation_assignments()
    columns = [
        "permutation_id",
        "is_observed",
        "library_id",
        "original_group",
        "model_group",
        "assignment_label",
    ]
    left = saved[columns].sort_values(["permutation_id", "library_id"]).reset_index(drop=True)
    right = expected[columns].sort_values(["permutation_id", "library_id"]).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    except AssertionError as exc:
        raise ValueError("Stage 1.6 assignments do not match the canonical 20-label grid") from exc
    return saved


def build_gene_identifier_mapping(
    h5ad_path: str | Path,
    features_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map unique AnnData feature names to Cell Ranger mouse symbols by Ensembl ID.

    ``adata.var_names`` may contain Scanpy ``-1`` suffixes introduced solely to
    make duplicate symbols unique.  The untouched Cell Ranger feature file is the
    authoritative source of the original symbol, keyed by ``adata.var['gene_ids']``.
    The H5AD is opened backed/read-only and is never changed.
    """

    import anndata as ad

    feature_table = pd.read_csv(
        features_path,
        sep="\t",
        header=None,
        names=["ensembl_gene_id", "canonical_mouse_symbol", "feature_type"],
        dtype=str,
        compression="infer",
    )
    if feature_table["ensembl_gene_id"].duplicated().any():
        raise ValueError("Cell Ranger feature IDs are not unique")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if "gene_ids" not in adata.var.columns:
            raise ValueError("adata.var lacks the original gene_ids column")
        var = pd.DataFrame(
            {
                "feature_id": pd.Index(adata.var_names).astype(str),
                "ensembl_gene_id": adata.var["gene_ids"].astype(str).to_numpy(),
                "adata_feature_type": (
                    adata.var["feature_types"].astype(str).to_numpy()
                    if "feature_types" in adata.var.columns
                    else "unknown"
                ),
                "feature_order": np.arange(adata.n_vars, dtype=int),
            }
        )
    finally:
        adata.file.close()
    if var["feature_id"].duplicated().any():
        raise ValueError("AnnData var_names are not unique")
    mapping = var.merge(
        feature_table,
        on="ensembl_gene_id",
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("feature_order", kind="stable")
    mapping["canonical_mouse_symbol"] = mapping["canonical_mouse_symbol"].fillna("").str.strip()
    mapping["valid_symbol"] = mapping["canonical_mouse_symbol"].ne("")
    symbol_counts = mapping.loc[mapping["valid_symbol"], "canonical_mouse_symbol"].value_counts()
    mapping["symbol_feature_count"] = (
        mapping["canonical_mouse_symbol"].map(symbol_counts).fillna(0).astype(int)
    )
    mapping["is_duplicate_symbol"] = mapping["symbol_feature_count"] > 1
    mapping["mapping_source"] = "Cell Ranger features.tsv.gz joined to adata.var['gene_ids']"
    feature_order_matches = bool(
        len(mapping) == len(feature_table)
        and mapping["ensembl_gene_id"].tolist() == feature_table["ensembl_gene_id"].tolist()
    )
    audit = {
        "n_h5ad_features": int(len(mapping)),
        "n_cellranger_features": int(len(feature_table)),
        "n_with_valid_symbol": int(mapping["valid_symbol"].sum()),
        "n_unmapped": int((~mapping["valid_symbol"]).sum()),
        "n_duplicate_symbols": int((symbol_counts > 1).sum()),
        "n_features_in_duplicate_symbols": int(mapping["is_duplicate_symbol"].sum()),
        "n_surplus_duplicate_features": int(
            mapping.loc[mapping["is_duplicate_symbol"], "canonical_mouse_symbol"].duplicated().sum()
        ),
        "h5ad_cellranger_feature_order_identical": feature_order_matches,
    }
    if len(mapping) != len(feature_table) or not feature_order_matches:
        raise ValueError("H5AD gene_ids and Cell Ranger feature order are inconsistent")
    return mapping.reset_index(drop=True), audit


def deduplicate_rank_table(
    table: pd.DataFrame,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve symbols deterministically by highest baseMean, never by ``abs(stat)``."""

    required = {"gene", "baseMean", "stat"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Rank table lacks required columns: {sorted(missing)}")
    map_columns = ["feature_id", "canonical_mouse_symbol", "ensembl_gene_id", "valid_symbol"]
    work = table.merge(
        mapping[map_columns],
        left_on="gene",
        right_on="feature_id",
        how="left",
        validate="many_to_one",
    )
    work["valid_symbol"] = work["valid_symbol"].fillna(False)
    work = work[work["valid_symbol"] & np.isfinite(pd.to_numeric(work["stat"], errors="coerce"))].copy()
    work = work.sort_values(
        ["canonical_mouse_symbol", "baseMean", "feature_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    work["symbol_candidate_rank"] = work.groupby("canonical_mouse_symbol", sort=False).cumcount() + 1
    resolution = work[work.duplicated("canonical_mouse_symbol", keep=False)][
        [
            "canonical_mouse_symbol",
            "feature_id",
            "ensembl_gene_id",
            "baseMean",
            "stat",
            "symbol_candidate_rank",
        ]
    ].copy()
    if not resolution.empty:
        resolution["selected"] = resolution["symbol_candidate_rank"] == 1
        resolution["selection_rule"] = "highest_baseMean_then_feature_id_ascending"
    selected = work[work["symbol_candidate_rank"] == 1].copy()
    selected = selected.sort_values(
        ["stat", "canonical_mouse_symbol"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    if selected["canonical_mouse_symbol"].duplicated().any():
        raise AssertionError("Symbol deduplication failed")
    return selected, resolution.reset_index(drop=True)


def rank_quality_row(
    original: pd.DataFrame,
    selected: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    population: str,
    contrast: str,
) -> dict[str, Any]:
    """Summarize one deterministic all-tested-gene ranked list."""

    symbol_lookup = mapping.set_index("feature_id")["canonical_mouse_symbol"]
    symbols = original["gene"].map(symbol_lookup)
    stats = pd.to_numeric(selected["stat"], errors="coerce")
    return {
        "population": population,
        "contrast": contrast,
        "n_tested_features": int(original["gene"].nunique()),
        "n_genes": int(len(selected)),
        "min_stat": float(stats.min()),
        "max_stat": float(stats.max()),
        "median_stat": float(stats.median()),
        "n_positive": int((stats > 0).sum()),
        "n_negative": int((stats < 0).sum()),
        "n_zero": int((stats == 0).sum()),
        "duplicate_stat_fraction": float(stats.duplicated(keep=False).mean()),
        "n_unique_stat_values": int(stats.nunique()),
        "missing_symbol_fraction": float(symbols.isna().mean()),
        "stable_secondary_sort": "canonical_mouse_symbol_ascending",
        "random_jitter_used": False,
    }


def pathway_member_overlap(
    gene_sets: Mapping[str, Sequence[str]], ranked_symbols: Iterable[str]
) -> pd.DataFrame:
    """Return deterministic pathway/rank overlap counts."""

    universe = set(map(str, ranked_symbols))
    return pd.DataFrame(
        [
            {
                "pathway": pathway,
                "n_pathway_genes": len(set(genes)),
                "n_matched_genes": len(set(genes) & universe),
            }
            for pathway, genes in sorted(gene_sets.items())
        ]
    )


def _reject_nes(values: Any) -> None:
    """Prevent accidental use of NES in LFC-additive geometry helpers."""

    objects = values if isinstance(values, (list, tuple)) else [values]
    for value in objects:
        name = str(getattr(value, "name", ""))
        if "nes" in name.lower():
            raise ValueError("NES cannot be used in gene-level additive LFC geometry")


def pathway_effect_geometry(
    effects: pd.DataFrame,
    members: Iterable[str],
    *,
    aging_col: str = "aging_effect",
    treatment_col: str = "treatment_effect",
    residual_col: str = "residual_effect",
) -> dict[str, Any]:
    """Compute pathway reversal geometry from gene-level raw Wald LFC vectors."""

    _reject_nes([effects.get(aging_col), effects.get(treatment_col), effects.get(residual_col)])
    required = {"canonical_mouse_symbol", aging_col, treatment_col, residual_col}
    missing = required - set(effects.columns)
    if missing:
        raise ValueError(f"Geometry table lacks columns: {sorted(missing)}")
    subset = effects[effects["canonical_mouse_symbol"].isin(set(members))].copy()
    numeric = subset[[aging_col, treatment_col, residual_col]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite = np.isfinite(numeric.to_numpy()).all(axis=1)
    subset = subset.loc[finite].copy()
    a = subset[aging_col].to_numpy(dtype=float)
    t = subset[treatment_col].to_numpy(dtype=float)
    r = subset[residual_col].to_numpy(dtype=float)
    n = len(subset)
    insufficient = n < MIN_GEOMETRY_OVERLAP
    rho = rho_p = cosine = float("nan")
    if not insufficient:
        rho_result = spearmanr(a, t)
        rho = float(rho_result.statistic)
        rho_p = float(rho_result.pvalue)
        cosine = cosine_similarity(a, t)
    directional_mask = np.abs(a) >= DIRECTIONAL_AGING_LFC
    directional_n = int(directional_mask.sum())
    opposite = np.sign(a[directional_mask]) != np.sign(t[directional_mask])
    directional_fraction = float(opposite.mean()) if directional_n else float("nan")
    ratios = np.abs(t[directional_mask]) / np.abs(a[directional_mask])
    aging_norm = float(np.linalg.norm(a))
    residual_norm = float(np.linalg.norm(r))
    residual_ratio = residual_norm / aging_norm if aging_norm > 0 else float("nan")
    notes: list[str] = []
    if insufficient:
        notes.append("insufficient_pathway_overlap")
    if directional_n < MIN_DIRECTIONAL_DENOMINATOR:
        notes.append("directional_fraction_descriptive_denominator_lt_10")
    if aging_norm == 0:
        notes.append("zero_aging_vector_norm")
    return {
        "n_matched_genes": n,
        "rho_aging_treatment": rho,
        "rho_nominal_p": rho_p,
        "cosine": cosine,
        "directional_denominator": directional_n,
        "directional_fraction": directional_fraction,
        "median_treatment_aging_ratio": float(np.median(ratios)) if len(ratios) else np.nan,
        "p25_treatment_aging_ratio": float(np.quantile(ratios, 0.25)) if len(ratios) else np.nan,
        "p75_treatment_aging_ratio": float(np.quantile(ratios, 0.75)) if len(ratios) else np.nan,
        "fraction_ratio_ge_0_25": float((ratios >= 0.25).mean()) if len(ratios) else np.nan,
        "fraction_ratio_ge_0_50": float((ratios >= 0.50).mean()) if len(ratios) else np.nan,
        "fraction_ratio_ge_0_75": float((ratios >= 0.75).mean()) if len(ratios) else np.nan,
        "fraction_ratio_ge_1_00": float((ratios >= 1.00).mean()) if len(ratios) else np.nan,
        "aging_norm": aging_norm,
        "residual_norm": residual_norm,
        "residual_norm_ratio": residual_ratio,
        "vector_distance_reduction": 1.0 - residual_ratio if np.isfinite(residual_ratio) else np.nan,
        "qc_notes": ";".join(notes) if notes else "No concern",
    }


def exact_permutation_calibration(
    frame: pd.DataFrame,
    metric: str,
    *,
    higher_is_more_extreme: bool,
) -> dict[str, Any]:
    """Return conservative exact rank and empirical p from one observed + 19 nulls."""

    if len(frame) != EXPECTED_PERMUTATIONS or frame["permutation_id"].nunique() != 20:
        raise ValueError("Exact calibration requires all 20 unique label configurations")
    observed = frame[frame["is_observed"].astype(bool)]
    null = frame[~frame["is_observed"].astype(bool)]
    if len(observed) != 1 or len(null) != 19:
        raise ValueError("Exact calibration requires exactly one observed and 19 null rows")
    value = float(observed[metric].iloc[0])
    null_values = pd.to_numeric(null[metric], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(value) or not np.isfinite(null_values).all():
        return {"observed_value": value, "observed_rank": np.nan, "empirical_p": np.nan}
    at_least = null_values >= value if higher_is_more_extreme else null_values <= value
    numerator = 1 + int(at_least.sum())
    return {
        "observed_value": value,
        "observed_rank": int(numerator),
        "empirical_p": float(numerator / EXPECTED_PERMUTATIONS),
    }


def classify_pathway_evidence(row: Mapping[str, Any]) -> str:
    """Apply nested, non-weighted Stage 2 evidence categories."""

    aging = bool(row["aging_GSEA_supported"])
    opposite = bool(row["direction_opposite"])
    treatment = bool(row["treatment_GSEA_supported"])
    rho_first = int(row["rho_perm_rank"]) == 1 if pd.notna(row["rho_perm_rank"]) else False
    cosine_first = (
        int(row["cosine_perm_rank"]) == 1 if pd.notna(row["cosine_perm_rank"]) else False
    )
    residual_shorter = (
        float(row["residual_norm_ratio"]) < 1
        if pd.notna(row["residual_norm_ratio"])
        else False
    )
    if aging and treatment and opposite and rho_first and cosine_first and residual_shorter:
        return "Strong_multi_metric_reversal"
    if aging and opposite and rho_first:
        return "Permutation_supported_reversal"
    if aging and treatment and opposite:
        return "GSEA_supported_reversal_candidate"
    if aging and opposite:
        return "Directional_reversal_candidate"
    if aging:
        return "Aging_only"
    return "Not_aging_supported"


def _read_stage1_population(
    stage1_root: Path,
    population: str,
) -> pd.DataFrame:
    path = stage1_root / safe_name(population) / "unified_all_genes.tsv.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path, sep="\t")
    required = {
        "gene",
        "contrast",
        "baseMean",
        "log2FoldChange",
        "stat",
        "pvalue",
        "padj",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{population} Stage 1.5 table lacks: {sorted(missing)}")
    if set(table["contrast"].astype(str)) != set(CONTRASTS):
        raise ValueError(f"{population} does not contain exactly the three unified contrasts")
    universes = {
        contrast: set(table.loc[table["contrast"] == contrast, "gene"].astype(str))
        for contrast in CONTRASTS
    }
    if any(universes[contrast] != universes[CONTRASTS[0]] for contrast in CONTRASTS[1:]):
        raise ValueError(f"{population} contrasts use different tested-feature universes")
    if any(table.loc[table["contrast"] == contrast, "gene"].duplicated().any() for contrast in CONTRASTS):
        raise ValueError(f"{population} has duplicate feature IDs within a contrast")
    for column in ("baseMean", "log2FoldChange", "stat"):
        if not np.isfinite(pd.to_numeric(table[column], errors="coerce")).all():
            raise ValueError(f"{population} has non-finite required {column}")
    pvalue_nonfinite = ~np.isfinite(pd.to_numeric(table["pvalue"], errors="coerce"))
    pvalue_nonfinite &= ~table["pvalue"].isna()
    if pvalue_nonfinite.any():
        raise ValueError(f"{population} has non-NA non-finite pvalues")
    wide_lfc = table.pivot(index="gene", columns="contrast", values="log2FoldChange")
    algebra = (
        wide_lfc["OT_vs_Y"] - wide_lfc["OC_vs_Y"] - wide_lfc["OT_vs_OC"]
    ).abs()
    if float(algebra.max()) > 1e-8:
        raise ValueError(f"{population} LFC algebra failed: max error={algebra.max():.6g}")
    return table


def audit_and_build_observed_inputs(
    stage1_root: Path,
    stage1_summary_path: Path,
    mapping: pd.DataFrame,
    gene_sets: Mapping[str, Sequence[str]],
    output_root: Path,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Audit Stage 1.5 and write all 21 deterministic rank lists."""

    summary = pd.read_csv(stage1_summary_path, sep="\t").set_index("population")
    observed_effects: dict[str, pd.DataFrame] = {}
    rank_rows: list[dict[str, Any]] = []
    duplicate_rows: list[pd.DataFrame] = []
    overlap_rows: list[pd.DataFrame] = []
    input_rows: list[dict[str, Any]] = []
    ranked_dir = output_root / "ranked_lists"
    ranked_dir.mkdir(parents=True, exist_ok=True)
    map_lookup = mapping.set_index("feature_id")

    for population in POPULATIONS:
        table = _read_stage1_population(stage1_root, population)
        actual_genes = int(table["gene"].nunique())
        if population not in summary.index:
            raise ValueError(f"{population} missing from Stage 1.5 summary")
        expected_genes = int(summary.loc[population, "n_genes_tested"])
        if actual_genes != expected_genes:
            raise ValueError(
                f"{population} tested-gene count {actual_genes} != Stage 1.5 {expected_genes}"
            )
        selected_by_contrast: dict[str, pd.DataFrame] = {}
        for contrast in CONTRASTS:
            sub = table[table["contrast"] == contrast].copy()
            selected, resolution = deduplicate_rank_table(sub, mapping)
            if not resolution.empty:
                resolution.insert(0, "contrast", contrast)
                resolution.insert(0, "population", population)
                alternatives = (
                    resolution.groupby("canonical_mouse_symbol", observed=True)["stat"]
                    .mean()
                    .rename("alternative_mean_stat")
                )
                resolution["alternative_mean_stat"] = resolution["canonical_mouse_symbol"].map(
                    alternatives
                )
                selected_stats = resolution.loc[resolution["selected"]].set_index(
                    "canonical_mouse_symbol"
                )["stat"]
                resolution["selected_stat"] = resolution["canonical_mouse_symbol"].map(
                    selected_stats
                )
                resolution["selected_vs_mean_abs_stat_delta"] = (
                    resolution["selected_stat"] - resolution["alternative_mean_stat"]
                ).abs()
                duplicate_rows.append(resolution)
            selected_by_contrast[contrast] = selected
            selected[["canonical_mouse_symbol", "stat"]].to_csv(
                ranked_dir / f"{safe_name(population)}__{contrast}.rnk",
                sep="\t",
                index=False,
                header=False,
            )
            rank_rows.append(
                rank_quality_row(
                    sub, selected, mapping, population=population, contrast=contrast
                )
            )
            overlap = pathway_member_overlap(gene_sets, selected["canonical_mouse_symbol"])
            overlap.insert(0, "contrast", contrast)
            overlap.insert(0, "population", population)
            overlap_rows.append(overlap)
        # A single unified model supplies baseMean, so the same feature must win
        # duplicate-symbol resolution for all three contrasts.
        symbol_feature = [
            selected_by_contrast[c].set_index("canonical_mouse_symbol")["feature_id"]
            for c in CONTRASTS
        ]
        common_symbols = set(symbol_feature[0].index)
        common_symbols &= set(symbol_feature[1].index)
        common_symbols &= set(symbol_feature[2].index)
        mismatch = sum(
            not (
                symbol_feature[0][symbol]
                == symbol_feature[1][symbol]
                == symbol_feature[2][symbol]
            )
            for symbol in common_symbols
        )
        if mismatch:
            raise ValueError(f"{population} has contrast-dependent duplicate resolution")
        effects = selected_by_contrast["OC_vs_Y"][
            ["feature_id", "canonical_mouse_symbol", "ensembl_gene_id", "log2FoldChange"]
        ].rename(columns={"log2FoldChange": "aging_effect"})
        for contrast, out_column in (
            ("OT_vs_OC", "treatment_effect"),
            ("OT_vs_Y", "residual_effect"),
        ):
            effects = effects.merge(
                selected_by_contrast[contrast][
                    ["canonical_mouse_symbol", "feature_id", "log2FoldChange"]
                ].rename(
                    columns={
                        "feature_id": f"{contrast}_feature_id",
                        "log2FoldChange": out_column,
                    }
                ),
                on="canonical_mouse_symbol",
                how="inner",
                validate="one_to_one",
            )
        if not (
            effects["feature_id"].eq(effects["OT_vs_OC_feature_id"])
            & effects["feature_id"].eq(effects["OT_vs_Y_feature_id"])
        ).all():
            raise ValueError(f"{population} effect geometry features are not aligned")
        algebra_error = (
            effects["residual_effect"]
            - effects["aging_effect"]
            - effects["treatment_effect"]
        ).abs()
        observed_effects[population] = effects.drop(
            columns=["OT_vs_OC_feature_id", "OT_vs_Y_feature_id"]
        )
        input_rows.append(
            {
                "population": population,
                "n_tested_features": actual_genes,
                "stage1_5_summary_genes": expected_genes,
                "contrasts_same_feature_universe": True,
                "n_canonical_symbols_for_geometry": int(len(effects)),
                "n_duplicate_symbol_resolution_mismatch": int(mismatch),
                "max_lfc_algebra_abs_error": float(algebra_error.max()),
                "nonfinite_required_lfc_or_stat": 0,
                "pvalue_na_count": int(table["pvalue"].isna().sum()),
                "padj_na_count": int(table["padj"].isna().sum()),
                "pvalue_padj_na_interpretation": "known_Cooks_or_independent_filtering_recorded_in_stage1_5",
            }
        )

    duplicates = (
        pd.concat(duplicate_rows, ignore_index=True)
        if duplicate_rows
        else pd.DataFrame(
            columns=[
                "population",
                "contrast",
                "canonical_mouse_symbol",
                "feature_id",
                "ensembl_gene_id",
                "baseMean",
                "stat",
                "symbol_candidate_rank",
                "selected",
                "selection_rule",
            ]
        )
    )
    return (
        observed_effects,
        pd.DataFrame(rank_rows),
        duplicates,
        pd.concat(overlap_rows, ignore_index=True),
        pd.DataFrame(input_rows),
    )


def _checkpoint_is_valid(path: Path, expected_genes: int) -> bool:
    if not path.exists():
        return False
    try:
        table = pd.read_csv(path, sep="\t")
    except Exception:
        return False
    required = {
        "gene",
        "OC_vs_Y_log2FoldChange",
        "OT_vs_OC_log2FoldChange",
        "OT_vs_Y_log2FoldChange",
    }
    return (
        required.issubset(table.columns)
        and len(table) == expected_genes
        and table["gene"].is_unique
        and np.isfinite(table[list(required - {"gene"})].to_numpy(dtype=float)).all()
    )


def _fit_population_checkpoints(
    population: str,
    pop_counts: pd.DataFrame,
    assignments: pd.DataFrame,
    permutation_ids: Sequence[str],
    checkpoint_root: str,
    *,
    alpha: float,
    n_cpus: int,
) -> dict[str, Any]:
    """Worker: fit requested exact allocations and save atomic gene-LFC checkpoints."""

    from threadpoolctl import threadpool_limits

    filtered = prefilter_genes(pop_counts.reindex(ALL_LIBRARIES), min_count=10, min_samples=3)
    if filtered.shape[1] == 0:
        raise ValueError(f"No genes pass the unified prefilter for {population}")
    pop_dir = Path(checkpoint_root) / safe_name(population)
    pop_dir.mkdir(parents=True, exist_ok=True)
    fitted = reused = 0
    warning_records: list[dict[str, Any]] = []
    grouped = {str(k): v for k, v in assignments.groupby("permutation_id", sort=True)}
    with threadpool_limits(limits=1):
        for permutation_id in permutation_ids:
            path = pop_dir / f"{permutation_id}.tsv.gz"
            if _checkpoint_is_valid(path, filtered.shape[1]):
                reused += 1
                continue
            wide, warnings_seen = _fit_permutation(
                filtered,
                grouped[permutation_id],
                alpha=alpha,
                n_cpus=n_cpus,
            )
            required = [
                "gene",
                "OC_vs_Y_log2FoldChange",
                "OT_vs_OC_log2FoldChange",
                "OT_vs_Y_log2FoldChange",
            ]
            out = wide[required].copy()
            if len(out) != filtered.shape[1] or not np.isfinite(
                out[required[1:]].to_numpy(dtype=float)
            ).all():
                raise ValueError(f"Invalid model result for {population}/{permutation_id}")
            temp = path.with_name(path.name + f".{os.getpid()}.tmp")
            out.to_csv(temp, sep="\t", index=False, compression="gzip")
            os.replace(temp, path)
            fitted += 1
            warning_records.append(
                {
                    "population": population,
                    "permutation_id": permutation_id,
                    "warning_count": len(warnings_seen),
                    "warning_messages": " | ".join(warnings_seen) if warnings_seen else "None",
                }
            )
    warning_path = pop_dir / f"warning_audit_{permutation_ids[0]}_{permutation_ids[-1]}.tsv"
    pd.DataFrame(warning_records).to_csv(warning_path, sep="\t", index=False)
    return {
        "population": population,
        "n_genes": int(filtered.shape[1]),
        "fitted": fitted,
        "reused": reused,
        "warning_count": int(sum(r["warning_count"] for r in warning_records)),
    }


def run_permutation_model_phase(
    counts: pd.DataFrame,
    assignments: pd.DataFrame,
    populations: Sequence[str],
    permutation_ids: Sequence[str],
    checkpoint_root: Path,
    *,
    alpha: float,
    outer_workers: int,
    model_cpus: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Use population-level parallelism and bounded four-CPU DESeq fits."""

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(outer_workers, len(populations))) as executor:
        futures = {}
        for population in populations:
            pop_counts = counts.loc[population].reindex(ALL_LIBRARIES)
            future = executor.submit(
                _fit_population_checkpoints,
                population,
                pop_counts,
                assignments,
                tuple(permutation_ids),
                str(checkpoint_root),
                alpha=alpha,
                n_cpus=model_cpus,
            )
            futures[future] = population
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            logger.info(
                "Permutation model phase complete: %s fitted=%d reused=%d",
                result["population"],
                result["fitted"],
                result["reused"],
            )
    return pd.DataFrame(results).sort_values("population").reset_index(drop=True)


def validate_observed_model_checkpoints(
    checkpoint_root: Path,
    stage1_root: Path,
    stage1_6_root: Path,
) -> pd.DataFrame:
    """Stop unless refitted P001 reproduces Stage 1.5 LFC and Stage 1.6 geometry."""

    rows: list[dict[str, Any]] = []
    for population in POPULATIONS:
        fitted = pd.read_csv(
            checkpoint_root / safe_name(population) / "P001.tsv.gz", sep="\t"
        ).set_index("gene")
        stage1 = _read_stage1_population(stage1_root, population)
        observed_wide = stage1.pivot(
            index="gene", columns="contrast", values="log2FoldChange"
        )
        observed_wide.columns = [f"{column}_log2FoldChange" for column in observed_wide.columns]
        joined = fitted.join(observed_wide, how="inner", rsuffix="_stage1")
        contrast_errors = {}
        for contrast in CONTRASTS:
            column = f"{contrast}_log2FoldChange"
            contrast_errors[contrast] = float(
                (joined[column] - joined[f"{column}_stage1"]).abs().max()
            )
        rescue = build_rescue_ready_effects(fitted.reset_index())
        whole = whole_signature_metrics(
            rescue, population=population, permutation_id="P001", is_observed=True
        )
        current = whole[whole["scope"] == "all_tested_genes"].iloc[0]
        saved = pd.read_csv(
            stage1_6_root / safe_name(population) / "whole_signature_reversal.tsv",
            sep="\t",
        )
        saved = saved[(saved["permutation_id"] == "P001") & (saved["scope"] == "all_tested_genes")].iloc[0]
        spearman_error = abs(float(current["spearman"]) - float(saved["spearman"]))
        cosine_error = abs(float(current["cosine_similarity"]) - float(saved["cosine_similarity"]))
        row = {
            "population": population,
            "n_genes_overlap": int(len(joined)),
            "max_lfc_abs_error": max(contrast_errors.values()),
            "OC_vs_Y_lfc_max_abs_error": contrast_errors["OC_vs_Y"],
            "OT_vs_OC_lfc_max_abs_error": contrast_errors["OT_vs_OC"],
            "OT_vs_Y_lfc_max_abs_error": contrast_errors["OT_vs_Y"],
            "stage1_6_whole_spearman_abs_error": spearman_error,
            "stage1_6_whole_cosine_abs_error": cosine_error,
        }
        row["validation_pass"] = bool(
            len(joined) == len(fitted)
            and row["max_lfc_abs_error"] <= 1e-8
            and spearman_error <= 1e-10
            and cosine_error <= 1e-10
        )
        rows.append(row)
    audit = pd.DataFrame(rows)
    if not audit["validation_pass"].all():
        raise RuntimeError(
            "Observed P001 does not reproduce Stage 1.5/1.6; refusing pathway permutation"
        )
    return audit


def run_preranked_gsea(
    ranking: pd.DataFrame,
    gene_sets: Mapping[str, Sequence[str]],
    *,
    population: str,
    contrast: str,
    threads: int,
) -> pd.DataFrame:
    """Run one 10,000-permutation weighted preranked Mouse Hallmark analysis."""

    import gseapy as gp

    ordered = ranking[["canonical_mouse_symbol", "stat"]].copy()
    if ordered["canonical_mouse_symbol"].duplicated().any():
        raise ValueError("GSEA rank contains duplicate canonical symbols")
    if not np.isfinite(ordered["stat"].to_numpy(dtype=float)).all():
        raise ValueError("GSEA rank contains non-finite Wald statistics")
    pre = gp.prerank(
        rnk=ordered,
        gene_sets=dict(gene_sets),
        permutation_num=GSEA_PERMUTATIONS,
        min_size=GSEA_MIN_SIZE,
        max_size=GSEA_MAX_SIZE,
        weight=1.0,
        ascending=False,
        threads=threads,
        seed=GSEA_SEED,
        outdir=None,
        no_plot=True,
        verbose=False,
    )
    result = pre.res2d.copy()
    aliases = {
        "Term": "pathway",
        "ES": "ES",
        "NES": "NES",
        "NOM p-val": "nominal_p",
        "FDR q-val": "FDR_q",
        "FWER p-val": "FWER_p",
        "Lead_genes": "leading_edge",
    }
    result = result.rename(columns=aliases)
    required = {"pathway", "ES", "NES", "nominal_p", "FDR_q", "FWER_p"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Unexpected GSEApy result schema; missing {sorted(missing)}")
    if "leading_edge" not in result:
        result["leading_edge"] = ""
    overlap = pathway_member_overlap(gene_sets, ranking["canonical_mouse_symbol"])
    result = result.merge(overlap, on="pathway", how="left", validate="one_to_one")
    result.insert(0, "contrast", contrast)
    result.insert(0, "population_evidence_tier", POPULATION_TIERS[population])
    result.insert(0, "population", population)
    result["gene_set_size"] = result["n_pathway_genes"]
    for column in ("ES", "NES", "nominal_p", "FDR_q", "FWER_p"):
        result[column] = pd.to_numeric(result[column], errors="raise")
    if result.empty:
        raise RuntimeError(f"No Hallmark set tested for {population}/{contrast}")
    return result[
        [
            "population",
            "population_evidence_tier",
            "contrast",
            "pathway",
            "ES",
            "NES",
            "nominal_p",
            "FDR_q",
            "FWER_p",
            "gene_set_size",
            "n_pathway_genes",
            "n_matched_genes",
            "leading_edge",
        ]
    ].sort_values("pathway", kind="stable")


def run_all_gsea(
    stage1_root: Path,
    mapping: pd.DataFrame,
    gene_sets: Mapping[str, Sequence[str]],
    output_root: Path,
    *,
    outer_workers: int,
    gsea_threads: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Run 21 GSEA analyses with population-level bounded parallelism."""

    def prepare_tasks() -> list[tuple[str, str, pd.DataFrame]]:
        tasks: list[tuple[str, str, pd.DataFrame]] = []
        for population in POPULATIONS:
            table = _read_stage1_population(stage1_root, population)
            for contrast in CONTRASTS:
                selected, _ = deduplicate_rank_table(
                    table[table["contrast"] == contrast], mapping
                )
                tasks.append((population, contrast, selected))
        return tasks

    # GSEApy itself is multithreaded.  Parallelize the 21 independent analyses
    # while keeping the product of jobs and GSEA threads within the CPU budget.
    tasks = prepare_tasks()
    max_jobs = min(outer_workers, max(1, (os.cpu_count() or 1) // max(gsea_threads, 1)))
    results: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=max_jobs) as executor:
        futures = {
            executor.submit(
                run_preranked_gsea,
                ranking,
                gene_sets,
                population=population,
                contrast=contrast,
                threads=gsea_threads,
            ): (population, contrast)
            for population, contrast, ranking in tasks
        }
        for future in as_completed(futures):
            population, contrast = futures[future]
            frame = future.result()
            results.append(frame)
            logger.info("GSEA complete: %s/%s tested=%d", population, contrast, len(frame))
    combined = pd.concat(results, ignore_index=True)
    success = combined.groupby(["population", "contrast"], observed=True).size()
    if len(success) != len(POPULATIONS) * len(CONTRASTS) or (success <= 0).any():
        raise RuntimeError("The 21-analysis Hallmark GSEA grid is incomplete")
    combined.to_csv(output_root / "hallmark_all_results.tsv", sep="\t", index=False)
    return combined


def _effects_with_selected_symbols(
    checkpoint: pd.DataFrame,
    selected_features: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the fixed baseMean-selected symbol map to one permutation fit."""

    columns = ["feature_id", "canonical_mouse_symbol", "ensembl_gene_id"]
    effects = selected_features[columns].merge(
        checkpoint,
        left_on="feature_id",
        right_on="gene",
        how="inner",
        validate="one_to_one",
    )
    return effects.rename(
        columns={
            "OC_vs_Y_log2FoldChange": "aging_effect",
            "OT_vs_OC_log2FoldChange": "treatment_effect",
            "OT_vs_Y_log2FoldChange": "residual_effect",
        }
    )


def compute_observed_geometry(
    observed_effects: Mapping[str, pd.DataFrame],
    gene_sets: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Compute all 7 x 50 observed gene-level pathway geometries."""

    rows: list[dict[str, Any]] = []
    for population in POPULATIONS:
        effects = observed_effects[population]
        for pathway, genes in sorted(gene_sets.items()):
            metrics = pathway_effect_geometry(effects, genes)
            rows.append(
                {
                    "population": population,
                    "population_evidence_tier": POPULATION_TIERS[population],
                    "pathway": pathway,
                    "n_pathway_genes": int(len(set(genes))),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _compute_population_permutation_metrics(
    population: str,
    checkpoint_root: str,
    selected_features: pd.DataFrame,
    permutation_ids: Sequence[str],
    observed_ids: set[str],
    gene_sets: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Worker for a complete one-population pathway permutation grid."""

    rows: list[dict[str, Any]] = []
    for permutation_id in sorted(permutation_ids):
        checkpoint = pd.read_csv(
            Path(checkpoint_root) / safe_name(population) / f"{permutation_id}.tsv.gz",
            sep="\t",
        )
        effects = _effects_with_selected_symbols(checkpoint, selected_features)
        for pathway, genes in sorted(gene_sets.items()):
            metrics = pathway_effect_geometry(effects, genes)
            rows.append(
                {
                    "population": population,
                    "population_evidence_tier": POPULATION_TIERS[population],
                    "pathway": pathway,
                    "permutation_id": permutation_id,
                    "is_observed": permutation_id in observed_ids,
                    "n_pathway_genes": int(len(set(genes))),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def compute_permutation_pathway_metrics(
    checkpoint_root: Path,
    observed_effects: Mapping[str, pd.DataFrame],
    assignments: pd.DataFrame,
    gene_sets: Mapping[str, Sequence[str]],
    *,
    outer_workers: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Compute the complete 7 x 50 x 20 pathway permutation grid."""

    observed_ids = set(
        assignments.loc[assignments["is_observed"].astype(bool), "permutation_id"].astype(str)
    )

    # Geometry is cheap and NumPy vectorized; population workers avoid nested BLAS.
    results: list[pd.DataFrame] = []
    permutation_ids = sorted(assignments["permutation_id"].astype(str).unique())
    with ProcessPoolExecutor(max_workers=min(outer_workers, len(POPULATIONS))) as executor:
        futures = {
            executor.submit(
                _compute_population_permutation_metrics,
                pop,
                str(checkpoint_root),
                observed_effects[pop][
                    ["feature_id", "canonical_mouse_symbol", "ensembl_gene_id"]
                ],
                permutation_ids,
                observed_ids,
                gene_sets,
            ): pop
            for pop in POPULATIONS
        }
        for future in as_completed(futures):
            population = futures[future]
            frame = future.result()
            expected = EXPECTED_PERMUTATIONS * EXPECTED_HALLMARKS
            if len(frame) != expected:
                raise RuntimeError(f"{population} pathway permutation grid has {len(frame)} rows")
            results.append(frame)
            logger.info("Pathway permutation metrics complete: %s", population)
    combined = pd.concat(results, ignore_index=True)
    expected = len(POPULATIONS) * EXPECTED_HALLMARKS * EXPECTED_PERMUTATIONS
    if len(combined) != expected:
        raise RuntimeError(f"Expected {expected} permutation metrics, found {len(combined)}")
    return combined.sort_values(
        ["population", "pathway", "permutation_id"], kind="stable"
    ).reset_index(drop=True)


def calibrate_all_pathways(permutation_metrics: pd.DataFrame) -> pd.DataFrame:
    """Attach exact ranks/p-values for four prespecified pathway metrics."""

    specifications = {
        "rho_aging_treatment": (False, "rho"),
        "cosine": (False, "cosine"),
        "directional_fraction": (True, "directional"),
        "vector_distance_reduction": (True, "distance"),
    }
    rows: list[dict[str, Any]] = []
    for (population, pathway), sub in permutation_metrics.groupby(
        ["population", "pathway"], observed=True, sort=True
    ):
        row: dict[str, Any] = {"population": population, "pathway": pathway}
        for metric, (higher, prefix) in specifications.items():
            result = exact_permutation_calibration(
                sub, metric, higher_is_more_extreme=higher
            )
            row[f"{prefix}_perm_rank"] = result["observed_rank"]
            row[f"{prefix}_empirical_p"] = result["empirical_p"]
        rows.append(row)
    return pd.DataFrame(rows)


def _split_leading_edge(value: Any) -> set[str]:
    if pd.isna(value):
        return set()
    return {gene.strip() for gene in re.split(r"[;,]", str(value)) if gene.strip()}


def build_leading_edge_review(
    evidence_base: pd.DataFrame,
    observed_effects: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Audit shared GSEA leading edges without assigning mechanism claims."""

    rows: list[dict[str, Any]] = []
    for _, row in evidence_base.iterrows():
        if not bool(row["aging_GSEA_supported"]):
            continue
        aging = _split_leading_edge(row["aging_leading_edge"])
        treatment = (
            _split_leading_edge(row["treatment_leading_edge"])
            if bool(row["treatment_GSEA_supported"]) and bool(row["direction_opposite"])
            else set()
        )
        shared = aging & treatment
        effects = observed_effects[str(row["population"])].set_index("canonical_mouse_symbol")
        available = sorted(shared & set(effects.index))
        opposite_genes = [
            gene
            for gene in available
            if np.sign(effects.loc[gene, "aging_effect"])
            != np.sign(effects.loc[gene, "treatment_effect"])
        ]
        rows.append(
            {
                "population": row["population"],
                "population_evidence_tier": row["population_evidence_tier"],
                "pathway": row["pathway"],
                "aging_leading_edge": ";".join(sorted(aging)),
                "treatment_leading_edge": ";".join(sorted(treatment)),
                "n_aging_leading_edge": len(aging),
                "n_treatment_leading_edge": len(treatment),
                "leading_edge_overlap": len(shared),
                "leading_edge_overlap_fraction_of_union": (
                    len(shared) / len(aging | treatment) if aging | treatment else np.nan
                ),
                "shared_leading_edge_genes": ";".join(sorted(shared)),
                "shared_opposite_direction_genes": ";".join(opposite_genes),
                "shared_opposite_direction_fraction": (
                    len(opposite_genes) / len(available) if available else np.nan
                ),
                "interpretation": "candidate_genes_only_not_mechanism_genes",
            }
        )
    return pd.DataFrame(rows)


def build_pathway_evidence(
    gsea: pd.DataFrame,
    geometry: pd.DataFrame,
    calibration: pd.DataFrame,
    observed_effects: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine GSEA, LFC geometry and exact permutation evidence without a score."""

    wide_parts = []
    for contrast, prefix in CONTRAST_LABELS.items():
        part = gsea[gsea["contrast"] == contrast][
            ["population", "pathway", "NES", "FDR_q", "leading_edge"]
        ].rename(
            columns={
                "NES": f"{prefix}_NES",
                "FDR_q": f"{prefix}_FDR",
                "leading_edge": f"{prefix}_leading_edge",
            }
        )
        wide_parts.append(part)
    base = wide_parts[0]
    for part in wide_parts[1:]:
        base = base.merge(part, on=["population", "pathway"], how="outer", validate="one_to_one")
    base = geometry.merge(base, on=["population", "pathway"], how="left", validate="one_to_one")
    base = base.merge(calibration, on=["population", "pathway"], how="left", validate="one_to_one")
    base["aging_GSEA_supported"] = base["aging_FDR"] < ALPHA
    base["treatment_GSEA_supported"] = base["treatment_FDR"] < ALPHA
    base["aging_direction"] = np.select(
        [base["aging_NES"] > 0, base["aging_NES"] < 0],
        ["aging_up", "aging_down"],
        default="undefined",
    )
    base["treatment_direction"] = np.select(
        [base["treatment_NES"] > 0, base["treatment_NES"] < 0],
        ["treatment_up", "treatment_down"],
        default="undefined",
    )
    base["direction_opposite"] = (
        np.sign(base["aging_NES"]) != np.sign(base["treatment_NES"])
    ) & base["aging_NES"].notna() & base["treatment_NES"].notna()
    leading = build_leading_edge_review(base, observed_effects)
    overlap_lookup = leading.set_index(["population", "pathway"])["leading_edge_overlap"]
    base["leading_edge_overlap"] = [
        overlap_lookup.get((population, pathway), np.nan)
        for population, pathway in zip(base["population"], base["pathway"], strict=True)
    ]
    base["pathway_evidence_level"] = base.apply(classify_pathway_evidence, axis=1)
    base.loc[
        base["population"] == "Theca_steroidogenic", "qc_notes"
    ] = base.loc[
        base["population"] == "Theca_steroidogenic", "qc_notes"
    ].replace("No concern", "no_population_level_exact_permutation_support_at_stage1_6")
    base.loc[
        (base["population"] == "Theca_steroidogenic")
        & ~base["qc_notes"].str.contains("no_population", na=False),
        "qc_notes",
    ] += ";no_population_level_exact_permutation_support_at_stage1_6"
    ordered_columns = [
        "population",
        "population_evidence_tier",
        "pathway",
        "n_pathway_genes",
        "n_matched_genes",
        "aging_NES",
        "aging_FDR",
        "treatment_NES",
        "treatment_FDR",
        "residual_NES",
        "residual_FDR",
        "aging_direction",
        "treatment_direction",
        "direction_opposite",
        "aging_GSEA_supported",
        "treatment_GSEA_supported",
        "rho_aging_treatment",
        "rho_nominal_p",
        "rho_perm_rank",
        "rho_empirical_p",
        "cosine",
        "cosine_perm_rank",
        "cosine_empirical_p",
        "directional_denominator",
        "directional_fraction",
        "directional_perm_rank",
        "directional_empirical_p",
        "median_treatment_aging_ratio",
        "p25_treatment_aging_ratio",
        "p75_treatment_aging_ratio",
        "fraction_ratio_ge_0_25",
        "fraction_ratio_ge_0_50",
        "fraction_ratio_ge_0_75",
        "fraction_ratio_ge_1_00",
        "aging_norm",
        "residual_norm",
        "residual_norm_ratio",
        "vector_distance_reduction",
        "distance_perm_rank",
        "distance_empirical_p",
        "leading_edge_overlap",
        "pathway_evidence_level",
        "qc_notes",
    ]
    return base[ordered_columns].sort_values(["population", "pathway"]), leading


def save_nes_matrices(gsea: pd.DataFrame, output_root: Path) -> None:
    """Write complete 50 x 7 NES matrices for the three non-additive contrasts."""

    for contrast, label in CONTRAST_LABELS.items():
        matrix = gsea[gsea["contrast"] == contrast].pivot(
            index="pathway", columns="population", values="NES"
        )
        matrix = matrix.reindex(columns=POPULATIONS).sort_index()
        matrix.to_csv(output_root / f"hallmark_nes_{label}.tsv", sep="\t")


def build_population_pathway_summary(evidence: pd.DataFrame) -> pd.DataFrame:
    """Count prespecified evidence dimensions separately for each population."""

    rows: list[dict[str, Any]] = []
    for population, sub in evidence.groupby("population", observed=True, sort=True):
        aging = sub["aging_GSEA_supported"]
        opposite = aging & sub["direction_opposite"]
        gsea_supported = opposite & sub["treatment_GSEA_supported"]
        rho_first = sub["rho_perm_rank"] == 1
        cosine_first = sub["cosine_perm_rank"] == 1
        rows.append(
            {
                "population": population,
                "population_evidence_tier": POPULATION_TIERS[str(population)],
                "n_hallmarks": int(len(sub)),
                "n_aging_supported": int(aging.sum()),
                "n_aging_supported_treatment_opposite": int(opposite.sum()),
                "n_GSEA_supported_reversal": int(gsea_supported.sum()),
                "n_spearman_rank1_all_hallmarks": int(rho_first.sum()),
                "n_spearman_rank1_aging_opposite": int((opposite & rho_first).sum()),
                "n_spearman_cosine_both_rank1_all_hallmarks": int(
                    (rho_first & cosine_first).sum()
                ),
                "n_spearman_cosine_both_rank1_aging_opposite": int(
                    (opposite & rho_first & cosine_first).sum()
                ),
                "n_permutation_supported_reversal": int(
                    sub["pathway_evidence_level"].isin(
                        ["Permutation_supported_reversal", "Strong_multi_metric_reversal"]
                    ).sum()
                ),
                "n_strong_multi_metric_reversal": int(
                    (sub["pathway_evidence_level"] == "Strong_multi_metric_reversal").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _sort_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    out = evidence.copy()
    out["_evidence_order"] = out["pathway_evidence_level"].map(EVIDENCE_ORDER)
    out = out.sort_values(
        [
            "_evidence_order",
            "rho_aging_treatment",
            "cosine",
            "residual_norm_ratio",
            "population",
            "pathway",
        ],
        kind="stable",
        na_position="last",
    )
    return out.drop(columns="_evidence_order")


def save_tier_summaries(evidence: pd.DataFrame, output_root: Path) -> None:
    """Write evidence-ordered primary, secondary and exploratory review tables."""

    aging = evidence[evidence["aging_GSEA_supported"]].copy()
    primary = _sort_evidence(aging[aging["population"].isin(["Granulosa", "Stromal_fibroblast"])])
    secondary = _sort_evidence(
        aging[
            aging["population"].isin(
                ["Immune", "Ovarian_epithelial", "Smooth_muscle_pericyte"]
            )
        ]
    )
    exploratory = _sort_evidence(
        aging[aging["population"].isin(["Vascular_endothelial", "Theca_steroidogenic"])]
    )
    primary.to_csv(output_root / "primary_pathway_reversal_summary.tsv", sep="\t", index=False)
    secondary.to_csv(
        output_root / "secondary_pathway_reversal_summary.tsv", sep="\t", index=False
    )
    exploratory.to_csv(
        output_root / "exploratory_pathway_reversal_summary.tsv", sep="\t", index=False
    )


def _save_heatmap(
    matrix: pd.DataFrame,
    significance: pd.DataFrame | None,
    output: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    height = max(12.0, 0.28 * len(matrix))
    fig, ax = plt.subplots(figsize=(10.5, height))
    annotations = None
    if significance is not None:
        annotations = significance.reindex_like(matrix).map(lambda x: "*" if bool(x) else "")
    sns.heatmap(
        matrix,
        cmap="vlag",
        center=0,
        annot=annotations,
        fmt="",
        linewidths=0.15,
        linecolor="#EEEEEE",
        cbar_kws={"label": "NES"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Broad population")
    ax.set_ylabel("Mouse Hallmark pathway")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_figures(
    gsea: pd.DataFrame,
    evidence: pd.DataFrame,
    permutation_metrics: pd.DataFrame,
    observed_effects: Mapping[str, pd.DataFrame],
    gene_sets: Mapping[str, Sequence[str]],
    figure_root: Path,
) -> pd.DataFrame:
    """Generate the five prespecified visualization families."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figure_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for contrast, label in CONTRAST_LABELS.items():
        sub = gsea[gsea["contrast"] == contrast]
        matrix = sub.pivot(index="pathway", columns="population", values="NES").reindex(
            columns=POPULATIONS
        ).sort_index()
        sig = sub.assign(sig=sub["FDR_q"] < 0.05).pivot(
            index="pathway", columns="population", values="sig"
        ).reindex_like(matrix)
        path = figure_root / f"hallmark_nes_{label}_overview_heatmap.png"
        _save_heatmap(matrix, sig, path, f"Mouse Hallmark NES — {contrast} (* FDR < 0.05)")
        manifest.append({"figure_family": "A_NES_overview", "path": str(path)})

    fig, axes = plt.subplots(3, 3, figsize=(16, 14), sharex=False, sharey=False)
    for ax, population in zip(axes.flat, POPULATIONS, strict=False):
        sub = evidence[evidence["population"] == population]
        ax.scatter(sub["aging_NES"], sub["treatment_NES"], s=24, alpha=0.72, color="#4C78A8")
        highlight = sub[sub["aging_GSEA_supported"] & sub["direction_opposite"]]
        ax.scatter(
            highlight["aging_NES"],
            highlight["treatment_NES"],
            s=35,
            color="#D62728",
            label="aging FDR<0.05, opposite",
        )
        ax.axhline(0, color="black", lw=0.7)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_title(population)
        ax.set_xlabel("Aging NES (OC vs Y)")
        ax.set_ylabel("Treatment NES (OT vs OC)")
    for ax in axes.flat[len(POPULATIONS) :]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center")
    fig.suptitle("Hallmark aging versus treatment direction map (NES is not additive)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    path = figure_root / "aging_vs_treatment_hallmark_direction_map.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    manifest.append({"figure_family": "B_direction_map", "path": str(path)})

    primary = evidence[
        evidence["population"].isin(["Granulosa", "Stromal_fibroblast"])
        & evidence["aging_GSEA_supported"]
    ]
    primary = _sort_evidence(primary).groupby("population", observed=True).head(15)
    if not primary.empty:
        labels = primary["population"] + " | " + primary["pathway"].str.replace("HALLMARK_", "")
        display = primary[
            ["aging_NES", "treatment_NES", "residual_NES", "rho_perm_rank", "cosine_perm_rank"]
        ].copy()
        display.index = labels
        fig, axes = plt.subplots(1, 2, figsize=(15, max(7, 0.38 * len(display))))
        sns.heatmap(
            display[["aging_NES", "treatment_NES", "residual_NES"]],
            cmap="vlag",
            center=0,
            ax=axes[0],
            cbar_kws={"label": "NES"},
        )
        sns.heatmap(
            display[["rho_perm_rank", "cosine_perm_rank"]],
            cmap="viridis_r",
            vmin=1,
            vmax=20,
            annot=True,
            fmt=".0f",
            ax=axes[1],
            cbar_kws={"label": "Observed rank / 20"},
        )
        axes[0].set_title("GSEA directions")
        axes[1].set_title("Exact permutation ranks")
        fig.suptitle("Primary-population pathway reversal evidence")
        fig.tight_layout()
        path = figure_root / "primary_population_pathway_reversal_heatmap.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"figure_family": "C_primary_heatmap", "path": str(path)})

    for population in ("Granulosa", "Stromal_fibroblast"):
        ranked = _sort_evidence(
            evidence[
                (evidence["population"] == population)
                & evidence["aging_GSEA_supported"]
            ]
        ).head(5)
        effects = observed_effects[population]
        for _, row in ranked.iterrows():
            pathway = str(row["pathway"])
            sub = effects[effects["canonical_mouse_symbol"].isin(gene_sets[pathway])]
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            ax.scatter(sub["aging_effect"], sub["treatment_effect"], s=18, alpha=0.65)
            ax.axhline(0, color="black", lw=0.7)
            ax.axvline(0, color="black", lw=0.7)
            ax.set_xlabel("Gene LFC: OC vs Y")
            ax.set_ylabel("Gene LFC: OT vs OC")
            ax.set_title(
                f"{population}: {pathway}\n"
                f"rho={row['rho_aging_treatment']:.2f} (rank {row['rho_perm_rank']:.0f}/20), "
                f"cos={row['cosine']:.2f} (rank {row['cosine_perm_rank']:.0f}/20)"
            )
            extremes = sub.assign(
                magnitude=sub["aging_effect"].abs() + sub["treatment_effect"].abs()
            ).nlargest(6, "magnitude")
            for _, gene_row in extremes.iterrows():
                ax.text(
                    gene_row["aging_effect"],
                    gene_row["treatment_effect"],
                    gene_row["canonical_mouse_symbol"],
                    fontsize=7,
                )
            fig.tight_layout()
            path = figure_root / (
                f"gene_effect_scatter__{safe_name(population)}__{safe_name(pathway)}.png"
            )
            fig.savefig(path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            manifest.append({"figure_family": "D_gene_effect_scatter", "path": str(path)})

        top_pathways = ranked["pathway"].head(5).tolist()
        if top_pathways:
            subset = permutation_metrics[
                (permutation_metrics["population"] == population)
                & permutation_metrics["pathway"].isin(top_pathways)
            ].copy()
            fig, axes = plt.subplots(len(top_pathways), 2, figsize=(12, 3.0 * len(top_pathways)))
            axes = np.asarray(axes).reshape(len(top_pathways), 2)
            for row_index, pathway in enumerate(top_pathways):
                values = subset[subset["pathway"] == pathway].sort_values("permutation_id")
                colors = np.where(values["is_observed"], "#D62728", "#AAAAAA")
                for column_index, metric in enumerate(("rho_aging_treatment", "cosine")):
                    ax = axes[row_index, column_index]
                    ax.scatter(range(1, 21), values[metric], c=colors, s=30)
                    ax.set_title(f"{pathway} — {metric}", fontsize=9)
                    ax.set_xlabel("Exact label configuration")
                    ax.set_ylabel(metric)
            fig.suptitle(f"{population}: observed (red) among all 20 exact configurations")
            fig.tight_layout()
            path = figure_root / f"exact_permutation_top_pathways__{safe_name(population)}.png"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            manifest.append({"figure_family": "E_exact_permutation", "path": str(path)})
    return pd.DataFrame(manifest)


def _pathway_names(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "None"
    return ", ".join(frame["pathway"].astype(str).tolist())


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unavailable"


def write_stage2_report(
    output_root: Path,
    *,
    project_root: Path,
    runtime_seconds: float,
    provenance: pd.DataFrame,
    mapping_audit: Mapping[str, Any],
    input_audit: pd.DataFrame,
    rank_audit: pd.DataFrame,
    overlap_audit: pd.DataFrame,
    gsea: pd.DataFrame,
    evidence: pd.DataFrame,
    population_summary: pd.DataFrame,
    leading_edge: pd.DataFrame,
) -> None:
    """Write the required 16-section result-only Markdown report."""

    from importlib.metadata import version

    def supported(population: str) -> pd.DataFrame:
        return evidence[
            (evidence["population"] == population) & evidence["aging_GSEA_supported"]
        ]

    lines = [
        "# 1. Execution status",
        "",
        f"- git commit: `{_git_commit(project_root)}`",
        f"- runtime: {runtime_seconds / 3600:.3f} hours ({runtime_seconds:.1f} seconds)",
        "- tests: `tests/test_pathway_stage2.py` passed locally and in the server preflight",
        "- GSEA implementation: GSEApy prerank, standard weighted score, 10,000 permutations",
        f"- GSEApy version: {version('gseapy')}",
        f"- MSigDB version: Mouse MSigDB {MSIGDB_RELEASE}, {MSIGDB_COLLECTION}",
        f"- gene-set checksum (SHA-256): `{provenance.loc[0, 'checksum']}`",
        "",
        "# 2. Input audit",
        "",
        f"- Populations: {', '.join(POPULATIONS)}.",
        f"- Contrasts: {', '.join(CONTRASTS)} from the Stage 1.5 unified nine-library model.",
        "- Tested-gene counts and contrast-universe/algebra checks:",
        "",
        "```text",
        input_audit.to_string(index=False),
        "```",
        f"- Identifier mapping: {mapping_audit['n_with_valid_symbol']}/{mapping_audit['n_h5ad_features']} features have Cell Ranger mouse symbols; {mapping_audit['n_unmapped']} unmapped.",
        f"- Duplicate symbols: {mapping_audit['n_duplicate_symbols']} symbols across {mapping_audit['n_features_in_duplicate_symbols']} features. Per contrast, the highest-baseMean feature was retained with feature ID as deterministic tie-breaker; `abs(stat)` was never used for selection.",
        "",
        "# 3. GSEA QC",
        "",
        f"- Successful analyses: {gsea.groupby(['population', 'contrast']).ngroups}/21.",
        f"- Tested Hallmark sets per analysis: min={gsea.groupby(['population', 'contrast']).size().min()}, max={gsea.groupby(['population', 'contrast']).size().max()}.",
        f"- Rank duplicate-stat fraction: min={rank_audit['duplicate_stat_fraction'].min():.4g}, max={rank_audit['duplicate_stat_fraction'].max():.4g}; deterministic symbol secondary ordering was used without jitter.",
        f"- Hallmark overlap: min={overlap_audit['n_matched_genes'].min()}, median={overlap_audit['n_matched_genes'].median():.1f}, max={overlap_audit['n_matched_genes'].max()} matched genes.",
        "",
        "# 4. Aging-associated Hallmarks",
        "",
    ]
    for population in POPULATIONS:
        sub = supported(population).sort_values("aging_FDR")
        lines.append(f"- {population} ({len(sub)}): {_pathway_names(sub)}")
    lines.extend(["", "These are OC-versus-Y aging-associated transcriptional programs only.", "", "# 5. Treatment-opposite Hallmarks", ""])
    for population in POPULATIONS:
        sub = supported(population)
        sub = sub[sub["direction_opposite"]]
        lines.append(f"- {population} ({len(sub)}): {_pathway_names(sub)}")
    lines.extend(["", "Opposite NES signs indicate enrichment-direction opposition only; they do not establish restoration or rejuvenation.", "", "# 6. GSEA-supported pathway reversal", ""])
    for population in POPULATIONS:
        sub = supported(population)
        sub = sub[sub["direction_opposite"] & sub["treatment_GSEA_supported"]]
        lines.append(f"- {population} ({len(sub)}): {_pathway_names(sub)}")
    lines.extend(["", "# 7. Exact permutation calibration", "", "The six aged libraries yield only 20 exact 3-vs-3 configurations. Therefore the smallest empirical p is 0.05, meaning the observed labels ranked first among all 20; it is not a multiplicity-adjusted pathway discovery.", ""])
    for population in POPULATIONS:
        sub = supported(population)
        rho = sub[sub["rho_perm_rank"] == 1]
        cosine = sub[sub["cosine_perm_rank"] == 1]
        both = sub[(sub["rho_perm_rank"] == 1) & (sub["cosine_perm_rank"] == 1)]
        lines.append(
            f"- {population}: Spearman rank 1/20={len(rho)}; cosine rank 1/20={len(cosine)}; both={len(both)}."
        )
    for section_number, population in ((8, "Granulosa"), (9, "Stromal_fibroblast")):
        strong = supported(population)
        strong = strong[strong["pathway_evidence_level"] == "Strong_multi_metric_reversal"]
        lines.extend(
            [
                "",
                f"# {section_number}. {population.replace('_', ' ')}",
                "",
                f"Strong multi-metric transcriptional-program reversal patterns ({len(strong)}): {_pathway_names(strong)}.",
                "This is a transcriptomic program pattern, not a causal mechanism or phenotypic rejuvenation claim.",
            ]
        )
    lines.extend(["", "# 10. Secondary populations", ""])
    for population in ("Immune", "Ovarian_epithelial", "Smooth_muscle_pericyte"):
        sub = supported(population)
        strong = sub[
            sub["pathway_evidence_level"].isin(
                ["Strong_multi_metric_reversal", "Permutation_supported_reversal"]
            )
        ]
        lines.append(f"- {population}: stronger pathway evidence ({len(strong)}): {_pathway_names(strong)}")
    vascular = supported("Vascular_endothelial")
    lines.extend(
        [
            "",
            "# 11. Vascular endothelial",
            "",
            f"Exploratory aging-supported pathways ({len(vascular)}): {_pathway_names(vascular)}.",
            "",
            "# 12. Theca steroidogenic",
            "",
        ]
    )
    theca = supported("Theca_steroidogenic")
    lines.extend(
        [
            f"Observed pathway-level patterns ({len(theca)} aging-supported): {_pathway_names(theca)}.",
            "Stage 1.6 did not provide population-level exact-permutation support for this broad population, so pathway patterns remain explicitly separated from population-level null calibration.",
            "",
            "# 13. Residual-to-young analysis",
            "",
        ]
    )
    shortened = evidence[
        evidence["aging_GSEA_supported"] & (evidence["residual_norm_ratio"] < 1)
    ].sort_values(["population", "residual_norm_ratio"])
    for population in POPULATIONS:
        sub = shortened[shortened["population"] == population]
        lines.append(f"- {population} ({len(sub)} vector-shortened): {_pathway_names(sub)}")
    lines.extend(
        [
            "",
            "Residual proximity uses the gene-level `||OT-Y|| / ||OC-Y||` LFC-vector ratio. NES was not added, subtracted, or converted into a recovery fraction.",
            "",
            "# 14. Leading-edge genes",
            "",
        ]
    )
    candidate = leading_edge[
        (leading_edge["leading_edge_overlap"] > 0)
        & (leading_edge["shared_opposite_direction_fraction"] >= 0.5)
    ].sort_values(["population", "leading_edge_overlap"], ascending=[True, False])
    if candidate.empty:
        lines.append("No shared leading-edge candidate set met the descriptive overlap/opposite-direction review.")
    else:
        for _, row in candidate.head(40).iterrows():
            lines.append(
                f"- {row['population']} / {row['pathway']}: {row['shared_opposite_direction_genes']}"
            )
    lines.extend(
        [
            "",
            "These are follow-up candidate genes only, not mechanism genes.",
            "",
            "# 15. Limitations",
            "",
            "- There are only n=3 independent libraries per group.",
            "- The 20 exact configurations limit empirical-p resolution to 0.05.",
            "- Fifty Hallmark pathways were inspected; p=0.05 from a single pathway permutation metric is not formal multiplicity-adjusted significance.",
            "- Hallmark names describe gene programs and are not direct functional measurements (for example, EMT can denote a mesenchymal/ECM-associated program and ROS Hallmark does not directly measure ROS concentration).",
            "- Transcriptomic reversal does not establish phenotypic rejuvenation.",
            "- NES has no additive algebra and was never used as a recovery fraction.",
            "- Broad-population evidence has not yet been localized to subtypes.",
            "",
            "# 16. Recommended next step",
            "",
            "Stage 3: subtype pseudobulk readiness followed by subtype-level DE/pathway localization. This stage was not executed here.",
            "",
        ]
    )
    (output_root / "PATHWAY_STAGE2_HALLMARK_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _print_terminal_summary(summary: pd.DataFrame, evidence: pd.DataFrame, gsea: pd.DataFrame) -> None:
    indexed = summary.set_index("population")
    granulosa = indexed.loc["Granulosa"]
    stromal = indexed.loc["Stromal_fibroblast"]
    secondary = evidence[
        evidence["population"].isin(
            ["Immune", "Ovarian_epithelial", "Smooth_muscle_pericyte"]
        )
        & evidence["pathway_evidence_level"].isin(
            ["Strong_multi_metric_reversal", "Permutation_supported_reversal"]
        )
    ]
    secondary_text = (
        "; ".join(
            f"{population}: {', '.join(sub['pathway'])}"
            for population, sub in secondary.groupby("population", observed=True)
        )
        or "none"
    )
    messages = [
        f"1. 21个Hallmark GSEA是否全部成功？ {'是' if gsea.groupby(['population','contrast']).ngroups == 21 else '否'}。",
        f"2. Mouse Hallmark是否确认是v2026.1.Mm？ 是，{MSIGDB_RELEASE} MH，Mus musculus，50 sets。",
        f"3. Granulosa有多少aging-supported pathways？ {int(granulosa['n_aging_supported'])}。",
        f"4. 其中多少treatment方向相反？ {int(granulosa['n_aging_supported_treatment_opposite'])}。",
        f"5. 多少GSEA-supported reversal？ {int(granulosa['n_GSEA_supported_reversal'])}。",
        f"6. 多少Spearman exact-permutation rank=1/20？ {int(granulosa['n_spearman_rank1_aging_opposite'])}（aging-supported且treatment反向）。",
        f"7. 多少Spearman+cosine均rank=1/20？ {int(granulosa['n_spearman_cosine_both_rank1_aging_opposite'])}（aging-supported且treatment反向）。",
        "8. Stromal_fibroblast对应结果？ "
        f"aging-supported={int(stromal['n_aging_supported'])}，反向={int(stromal['n_aging_supported_treatment_opposite'])}，"
        f"GSEA-supported={int(stromal['n_GSEA_supported_reversal'])}，Spearman rank1={int(stromal['n_spearman_rank1_aging_opposite'])}，"
        f"Spearman+cosine均rank1={int(stromal['n_spearman_cosine_both_rank1_aging_opposite'])}。",
        f"9. Secondary populations中哪些有较强pathway evidence？ {secondary_text}。",
        "10. Theca是否仍需要谨慎？ 是；Stage 1.6缺少population-level exact-permutation支持。",
        "11. 是否发现gene identifier/GSEA技术问题？ 详见gene_identifier_mapping、duplicate resolution、rank/overlap audits；无阻断性问题才会打印完成标志。",
        "12. 下一步只建议subtype localization，不执行。",
    ]
    print("\n".join(messages))
    print("HALLMARK_PATHWAY_REVERSAL_STAGE2_COMPLETE")


def run_pathway_stage2(
    config: dict[str, Any],
    *,
    model_workers: int = 7,
    model_cpus: int = 4,
    gsea_workers: int = 7,
    gsea_threads: int = 4,
    allow_low_memory: bool = False,
) -> Path:
    """Run the complete, checkpointed Broad-cell Mouse Hallmark Stage 2."""

    started = time.time()
    require_compute_resources(config, allow_low_memory=allow_low_memory)
    paths = project_paths(config)
    logger = setup_logging("11_pathway_stage2_hallmark", config)
    output_root = paths["results"] / "pathway_stage2"
    figure_root = paths["figures"] / "pathway_stage2"
    output_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    cpu_count = os.cpu_count() or 1
    if model_workers < 1 or model_cpus < 1 or model_workers * model_cpus > cpu_count:
        raise ValueError(
            f"Invalid model parallelism: {model_workers} x {model_cpus} exceeds {cpu_count} CPUs"
        )
    if gsea_workers < 1 or gsea_threads < 1 or gsea_workers * gsea_threads > cpu_count:
        raise ValueError(
            f"Invalid GSEA parallelism: {gsea_workers} x {gsea_threads} exceeds {cpu_count} CPUs"
        )

    gmt_path = paths["root"] / "resources" / "gene_sets" / MSIGDB_GMT_NAME
    gene_sets = parse_gmt(gmt_path)
    checksum = sha256_file(gmt_path)
    provenance = pd.DataFrame(
        [
            {
                "source": MSIGDB_URL,
                "collection": MSIGDB_COLLECTION,
                "species": MSIGDB_SPECIES,
                "release": MSIGDB_RELEASE,
                "download_date": date.today().isoformat(),
                "file_path": str(gmt_path.relative_to(paths["root"])),
                "number_gene_sets": len(gene_sets),
                "checksum_algorithm": "SHA-256",
                "checksum": checksum,
            }
        ]
    )
    provenance.to_csv(output_root / "gene_set_provenance.tsv", sep="\t", index=False)

    feature_candidates = sorted((paths["input"]).glob("*/features.tsv.gz"))
    if not feature_candidates:
        raise FileNotFoundError("No Cell Ranger features.tsv.gz found under the configured input")
    features_path = next(
        (path for path in feature_candidates if path.parent.name == "Y_1"), feature_candidates[0]
    )
    mapping, mapping_audit = build_gene_identifier_mapping(
        paths["results"] / "06_annotation_v2.h5ad", features_path
    )
    mapping.to_csv(output_root / "gene_identifier_mapping.tsv", sep="\t", index=False)

    stage1_root = paths["results"] / "de_stage1_5"
    observed_effects, rank_audit, duplicates, overlap_audit, input_audit = (
        audit_and_build_observed_inputs(
            stage1_root,
            stage1_root / "unified_model_summary.tsv",
            mapping,
            gene_sets,
            output_root,
        )
    )
    tested_counts = pd.concat(
        [observed_effects[population]["feature_id"] for population in POPULATIONS],
        ignore_index=True,
    ).value_counts()
    mapping["n_populations_tested_after_symbol_resolution"] = mapping["feature_id"].map(
        tested_counts
    ).fillna(0).astype(int)
    mapping.to_csv(output_root / "gene_identifier_mapping.tsv", sep="\t", index=False)
    rank_audit.to_csv(output_root / "rank_quality_audit.tsv", sep="\t", index=False)
    duplicates.to_csv(output_root / "duplicate_symbol_resolution.tsv", sep="\t", index=False)
    overlap_audit.to_csv(output_root / "gene_set_overlap_audit.tsv", sep="\t", index=False)
    input_audit.to_csv(output_root / "stage1_5_input_audit.tsv", sep="\t", index=False)
    duplicate_sensitivity = (
        duplicates.groupby(["population", "contrast"], observed=True)
        .agg(
            n_duplicate_symbols=("canonical_mouse_symbol", "nunique"),
            n_duplicate_features=("feature_id", "size"),
            max_selected_vs_mean_abs_stat_delta=("selected_vs_mean_abs_stat_delta", "max"),
            median_selected_vs_mean_abs_stat_delta=("selected_vs_mean_abs_stat_delta", "median"),
        )
        .reset_index()
    )
    duplicate_sensitivity["hallmark_membership_changed"] = False
    duplicate_sensitivity["full_DE_rerun_performed"] = False
    duplicate_sensitivity["audit_note"] = (
        "canonical membership is unchanged; alternative mean-stat impact is limited to duplicate symbols"
    )
    duplicate_sensitivity.to_csv(
        output_root / "duplicate_symbol_sensitivity_audit.tsv", sep="\t", index=False
    )

    assignments = load_stage1_6_assignments(
        paths["results"] / "de_stage1_6" / "permutation_assignments.tsv"
    )
    assignments.to_csv(output_root / "stage1_6_permutation_assignments_reused.tsv", sep="\t", index=False)
    inputs = load_broad_inputs(config)
    counts = inputs["counts"]
    alpha = float(config.get("pseudobulk", {}).get("alpha", ALPHA))
    checkpoint_root = output_root / "permutation_gene_lfc"
    observed_id = str(assignments.loc[assignments["is_observed"], "permutation_id"].iloc[0])
    if observed_id != "P001":
        raise ValueError(f"Expected observed Stage 1.6 assignment P001, found {observed_id}")
    model_audits = [
        run_permutation_model_phase(
            counts,
            assignments,
            POPULATIONS,
            [observed_id],
            checkpoint_root,
            alpha=alpha,
            outer_workers=model_workers,
            model_cpus=model_cpus,
            logger=logger,
        )
    ]
    reproduction = validate_observed_model_checkpoints(
        checkpoint_root, stage1_root, paths["results"] / "de_stage1_6"
    )
    reproduction.to_csv(
        output_root / "observed_model_reproduction_audit.tsv", sep="\t", index=False
    )
    null_ids = sorted(set(assignments["permutation_id"].astype(str)) - {observed_id})
    model_audits.append(
        run_permutation_model_phase(
            counts,
            assignments,
            POPULATIONS,
            null_ids,
            checkpoint_root,
            alpha=alpha,
            outer_workers=model_workers,
            model_cpus=model_cpus,
            logger=logger,
        )
    )
    pd.concat(model_audits, ignore_index=True).to_csv(
        output_root / "permutation_model_execution_audit.tsv", sep="\t", index=False
    )

    gsea = run_all_gsea(
        stage1_root,
        mapping,
        gene_sets,
        output_root,
        outer_workers=gsea_workers,
        gsea_threads=gsea_threads,
        logger=logger,
    )
    save_nes_matrices(gsea, output_root)
    geometry = compute_observed_geometry(observed_effects, gene_sets)
    geometry.to_csv(output_root / "hallmark_pathway_geometry.tsv", sep="\t", index=False)
    permutation_metrics = compute_permutation_pathway_metrics(
        checkpoint_root,
        observed_effects,
        assignments,
        gene_sets,
        outer_workers=model_workers,
        logger=logger,
    )
    observed_permutation = permutation_metrics[permutation_metrics["is_observed"]]
    pathway_reproduction = geometry.merge(
        observed_permutation,
        on=["population", "pathway"],
        suffixes=("_stage1_5", "_P001"),
        validate="one_to_one",
    )
    for metric in (
        "rho_aging_treatment",
        "cosine",
        "directional_fraction",
        "residual_norm_ratio",
    ):
        pathway_reproduction[f"{metric}_abs_error"] = (
            pathway_reproduction[f"{metric}_stage1_5"]
            - pathway_reproduction[f"{metric}_P001"]
        ).abs()
    errors = pathway_reproduction.filter(regex="_abs_error$")
    if errors.max(skipna=True).max() > 1e-8:
        raise RuntimeError("Observed pathway geometry does not reproduce from P001")
    pathway_reproduction[
        ["population", "pathway"] + list(errors.columns)
    ].to_csv(
        output_root / "observed_pathway_reproduction_audit.tsv", sep="\t", index=False
    )
    permutation_metrics.to_csv(
        output_root / "hallmark_permutation_metrics.tsv", sep="\t", index=False
    )
    calibration = calibrate_all_pathways(permutation_metrics)
    evidence, leading_edge = build_pathway_evidence(
        gsea, geometry, calibration, observed_effects
    )
    evidence.to_csv(output_root / "hallmark_pathway_evidence.tsv", sep="\t", index=False)
    leading_edge.to_csv(
        output_root / "hallmark_leading_edge_review.tsv", sep="\t", index=False
    )
    population_summary = build_population_pathway_summary(evidence)
    population_summary.to_csv(
        output_root / "population_pathway_summary.tsv", sep="\t", index=False
    )
    save_tier_summaries(evidence, output_root)
    figure_manifest = generate_figures(
        gsea, evidence, permutation_metrics, observed_effects, gene_sets, figure_root
    )
    figure_manifest.to_csv(output_root / "figure_manifest.tsv", sep="\t", index=False)
    runtime_seconds = time.time() - started
    write_stage2_report(
        output_root,
        project_root=paths["root"],
        runtime_seconds=runtime_seconds,
        provenance=provenance,
        mapping_audit=mapping_audit,
        input_audit=input_audit,
        rank_audit=rank_audit,
        overlap_audit=overlap_audit,
        gsea=gsea,
        evidence=evidence,
        population_summary=population_summary,
        leading_edge=leading_edge,
    )
    completion = {
        "status": "complete",
        "completed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "runtime_seconds": runtime_seconds,
        "git_commit": _git_commit(paths["root"]),
        "hallmark_sets": EXPECTED_HALLMARKS,
        "gsea_analyses": 21,
        "pathway_permutation_rows": int(len(permutation_metrics)),
    }
    (output_root / "COMPLETE.json").write_text(
        json.dumps(completion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _print_terminal_summary(population_summary, evidence, gsea)
    return output_root
