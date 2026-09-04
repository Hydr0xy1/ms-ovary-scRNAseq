"""Conservative sample-level pseudobulk differential expression (DE) stage 1.

This module deliberately starts from the already audited broad pseudobulk files.  It
does not read ``AnnData.X`` and it never changes the main single-cell object.  All
eligibility decisions are made independently for each population and contrast, with
``library_id`` as the biological replicate.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .project import project_paths, require_compute_resources, setup_logging
from .pseudobulk import safe_name

LIBRARIES_BY_GROUP: dict[str, list[str]] = {
    "Y": ["Y_1", "Y_2", "Y_3"],
    "OC": ["OC_1", "OC_2", "OC_3"],
    "OT": ["OT_1", "OT_2", "OT_3"],
}
CONTRASTS: tuple[tuple[str, str], ...] = (("OC", "Y"), ("OT", "OC"), ("OT", "Y"))
DESCRIPTIVE_POPULATIONS = {"Oocyte", "Rare_luteal_candidate"}


@dataclass(frozen=True)
class ContrastSpec:
    """A contrast with a denominator (A) and numerator (B)."""

    numerator: str
    denominator: str

    @property
    def name(self) -> str:
        return f"{self.numerator}_vs_{self.denominator}"

    @property
    def group_a(self) -> str:
        return self.denominator

    @property
    def group_b(self) -> str:
        return self.numerator


def _read_table(path: Path, *, index_col: int | list[int] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="infer", index_col=index_col)


def load_broad_inputs(config: dict[str, Any]) -> dict[str, pd.DataFrame | str]:
    """Read and validate the audited broad pseudobulk input files.

    The function is intentionally strict about integer counts and metadata keys.  A
    failed check stops DE rather than silently changing the input orientation or
    substituting normalized expression values.
    """

    paths = project_paths(config)
    root = paths["results"] / "pseudobulk_ready"
    required = [
        "broad_counts.tsv.gz",
        "broad_metadata.tsv",
        "population_coverage.tsv",
        "pseudobulk_qc.tsv",
        "de_readiness.tsv",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing pseudobulk input files: {missing}")

    counts = _read_table(root / "broad_counts.tsv.gz", index_col=[0, 1])
    counts.index = counts.index.set_names(["population", "library"])
    metadata = _read_table(root / "broad_metadata.tsv")
    coverage = _read_table(root / "population_coverage.tsv")
    pseudobulk_qc = _read_table(root / "pseudobulk_qc.tsv")
    de_readiness = _read_table(root / "de_readiness.tsv")

    if counts.index.has_duplicates:
        raise ValueError("broad_counts contains duplicate population × library rows")
    required_meta = {"population", "library", "group"}
    if not required_meta.issubset(metadata.columns):
        raise ValueError(f"broad_metadata is missing columns: {sorted(required_meta - set(metadata))}")
    if metadata.duplicated(["population", "library"]).any():
        raise ValueError("broad_metadata contains duplicate population × library rows")
    if not {"population", "library", "n_cells", "total_umi", "n_expressed_genes"}.issubset(
        pseudobulk_qc.columns
    ):
        raise ValueError("pseudobulk_qc does not contain the required QC columns")

    values = counts.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("broad_counts contains non-finite values")
    if (values < 0).any():
        raise ValueError("broad_counts contains negative values")
    if not np.equal(values, np.rint(values)).all():
        raise ValueError("broad_counts is not an integer UMI count matrix")
    counts = counts.astype(np.int64)

    expected_pairs = pd.MultiIndex.from_frame(metadata[["population", "library"]])
    if set(expected_pairs) != set(counts.index):
        missing = expected_pairs.difference(counts.index).tolist()
        extra = counts.index.difference(expected_pairs).tolist()
        raise ValueError(f"Counts/metadata keys differ; missing={missing[:5]}, extra={extra[:5]}")

    by_library = metadata.groupby("library", observed=True)["group"].nunique()
    if (by_library > 1).any():
        bad = by_library[by_library > 1].to_dict()
        raise ValueError(f"A library is assigned to more than one group: {bad}")
    observed_libraries = set(metadata["library"].astype(str))
    expected_libraries = set(sum(LIBRARIES_BY_GROUP.values(), []))
    unknown = sorted(observed_libraries - expected_libraries)
    if unknown:
        raise ValueError(f"Unexpected library IDs in metadata: {unknown}")

    readiness_report = root / "PSEUDOBULK_READINESS_REPORT.md"
    source_note = readiness_report.read_text(encoding="utf-8") if readiness_report.exists() else ""
    if "Tier1" not in source_note or "counts" not in source_note:
        raise ValueError(
            "PSEUDOBULK_READINESS_REPORT.md does not verify Tier1 raw-count aggregation; "
            "refusing to run DE without source confirmation"
        )
    return {
        "counts": counts,
        "metadata": metadata,
        "coverage": coverage,
        "pseudobulk_qc": pseudobulk_qc,
        "de_readiness": de_readiness,
        "source_note": source_note,
    }


def prefilter_genes(counts: pd.DataFrame, min_count: int = 10, min_samples: int = 3) -> pd.DataFrame:
    """Keep genes with raw count >= ``min_count`` in at least ``min_samples`` samples."""

    mask = (counts >= min_count).sum(axis=0) >= min_samples
    return counts.loc[:, mask].copy()


def _qc_lookup(qc: pd.DataFrame) -> pd.DataFrame:
    result = qc.copy()
    result["population"] = result["population"].astype(str)
    result["library"] = result["library"].astype(str)
    return result.set_index(["population", "library"])


def _pca_diagnostics(counts: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Return CPM-log PCA scores and a deliberately conservative outlier flag."""

    from sklearn.decomposition import PCA

    libsize = counts.sum(axis=1).to_numpy(dtype=float)
    norm = np.log1p(counts.to_numpy(dtype=float) / np.maximum(libsize[:, None], 1.0) * 1e6)
    n_components = min(3, max(1, norm.shape[0] - 1), norm.shape[1])
    scores = PCA(n_components=n_components, random_state=0).fit_transform(norm)
    score_df = pd.DataFrame(scores, index=counts.index, columns=[f"PC{i + 1}" for i in range(n_components)])
    score_df["library_id"] = score_df.index.astype(str)
    score_df["group"] = metadata.loc[score_df.index, "group"].astype(str).to_numpy()

    # With three replicates per group, use a high (5-MAD) threshold around each
    # group's centroid.  This identifies only a completely detached library; a
    # modest group difference is not an outlier and is never removed automatically.
    outlier = False
    for group, sub in score_df.groupby("group", observed=True):
        if len(sub) < 3:
            continue
        arr = sub.filter(regex=r"^PC").to_numpy(dtype=float)
        centroid = arr.mean(axis=0)
        distances = np.sqrt(((arr - centroid) ** 2).sum(axis=1))
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        threshold = median + 5.0 * max(mad, 1e-9)
        if median > 0 and (distances > max(threshold, 5.0 * median)).any():
            outlier = True
    return score_df, outlier


def build_eligibility(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    qc: pd.DataFrame,
    *,
    min_primary_cells: int = 50,
) -> pd.DataFrame:
    """Build contrast-specific population eligibility and final DE status."""

    qc_idx = _qc_lookup(qc)
    populations = sorted(counts.index.get_level_values("population").unique().astype(str))
    records: list[dict[str, Any]] = []
    for population in populations:
        pop_meta = metadata[metadata["population"].astype(str) == population].copy()
        # PCA is descriptive and uses all available libraries for this population.
        all_rows = counts.loc[population]
        all_meta = pop_meta.set_index("library").loc[all_rows.index.astype(str)]
        try:
            _, pca_outlier = _pca_diagnostics(all_rows, all_meta)
        except Exception:  # a tiny/degenerate population remains auditable
            pca_outlier = False

        for numerator, denominator in CONTRASTS:
            spec = ContrastSpec(numerator, denominator)
            row: dict[str, Any] = {
                "population": population,
                "contrast": spec.name,
                "group_A": denominator,
                "group_B": numerator,
                "pca_outlier_candidate": bool(pca_outlier),
            }
            concern: list[str] = []
            group_values: dict[str, list[int]] = {}
            for prefix, group in (("A", denominator), ("B", numerator)):
                vals: list[int] = []
                for i, library in enumerate(LIBRARIES_BY_GROUP[group], start=1):
                    key = (population, library)
                    if key in qc_idx.index:
                        item = qc_idx.loc[key]
                        n_cells = int(item["n_cells"])
                        total_umi = int(item["total_umi"])
                        detected = int(item["n_expressed_genes"])
                    else:
                        n_cells = total_umi = detected = 0
                        concern.append(f"missing_{library}")
                    row[f"{prefix}_lib{i}_cells"] = n_cells
                    row[f"{prefix}_lib{i}_total_umi"] = total_umi
                    row[f"{prefix}_lib{i}_detected_genes"] = detected
                    vals.append(n_cells)
                    if total_umi <= 0 or detected <= 0:
                        concern.append(f"zero_pseudobulk_{library}")
                group_values[prefix] = vals
            cells = group_values["A"] + group_values["B"]
            total_umis = [row[f"{prefix}_lib{i}_total_umi"] for prefix in ("A", "B") for i in range(1, 4)]
            detected = [row[f"{prefix}_lib{i}_detected_genes"] for prefix in ("A", "B") for i in range(1, 4)]
            row["min_cells"] = int(min(cells))
            row["median_cells"] = float(np.median(cells))
            row["min_total_UMI"] = int(min(total_umis))
            row["median_total_UMI"] = float(np.median(total_umis))
            row["min_detected_genes"] = int(min(detected))
            row["eligibility_20"] = bool(all(v >= 20 for v in group_values["A"]) and all(v >= 20 for v in group_values["B"]))
            row["eligibility_50"] = bool(all(v >= 50 for v in group_values["A"]) and all(v >= 50 for v in group_values["B"]))
            row["eligibility_100"] = bool(all(v >= 100 for v in group_values["A"]) and all(v >= 100 for v in group_values["B"]))
            if pca_outlier:
                concern.append("pca_outlier_candidate")
            if any(v < 20 for v in cells):
                concern.append("library_below_20_cells")
            elif any(v < min_primary_cells for v in cells):
                concern.append("library_below_50_cells")
            row["pseudobulk_qc_concern"] = ";".join(sorted(set(concern))) or "No concern"
            if population in DESCRIPTIVE_POPULATIONS:
                status = "Descriptive_only"
            elif not row["eligibility_20"]:
                status = "Not_DE_ready"
            elif not row["eligibility_50"] or row["pseudobulk_qc_concern"] != "No concern":
                status = "Sensitivity_only"
            else:
                status = "Primary_DE_ready"
            row["final_de_status"] = status
            row["numerator"] = numerator
            row["denominator"] = denominator
            row["genes_before"] = int(counts.shape[1])
            pop_counts = counts.loc[population]
            contrast_libraries = LIBRARIES_BY_GROUP[denominator] + LIBRARIES_BY_GROUP[numerator]
            row["genes_after"] = int(prefilter_genes(pop_counts.reindex(contrast_libraries), 10, 3).shape[1]) if row["eligibility_20"] else 0
            records.append(row)
    return pd.DataFrame(records)


def _capture_warnings() -> tuple[list[str], Any]:
    return [], warnings.catch_warnings(record=True)


def _run_one_deseq(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    spec: ContrastSpec,
    *,
    alpha: float,
    n_cpus: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run one two-group PyDESeq2 model and return results, diagnostics and PCA data."""

    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    libraries = LIBRARIES_BY_GROUP[spec.denominator] + LIBRARIES_BY_GROUP[spec.numerator]
    counts = counts.reindex(libraries).astype(np.int64)
    metadata = metadata.reindex(libraries).copy()
    metadata["group"] = pd.Categorical(metadata["group"].astype(str), categories=[spec.denominator, spec.numerator])
    filtered = prefilter_genes(counts, min_count=10, min_samples=3)
    genes_before, genes_after = counts.shape[1], filtered.shape[1]
    if genes_after == 0:
        raise ValueError("No genes passed the raw-count prefilter")
    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        counts=filtered,
        metadata=metadata,
        design="~group",
        refit_cooks=True,
        inference=inference,
        low_memory=True,
    )
    dds.deseq2()
    stats = DeseqStats(
        dds,
        contrast=["group", spec.numerator, spec.denominator],
        alpha=alpha,
        cooks_filter=True,
        independent_filter=True,
        inference=inference,
        quiet=True,
    )
    stats.summary()
    raw = stats.results_df.copy()
    raw.index.name = "gene"
    raw = raw.rename(columns={"log2FoldChange": "log2FC_raw"})
    coeffs = [str(c) for c in stats.LFC.columns if str(c) != "Intercept" and spec.numerator in str(c)]
    shrink_warning = ""
    shrunk = raw["log2FC_raw"].copy()
    if coeffs:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                stats.lfc_shrink(coeff=coeffs[0])
                shrink_warning = ";".join(str(w.message) for w in caught)
            shrunk = stats.results_df["log2FoldChange"].copy()
        except Exception as exc:  # Keep the raw Wald result if apeGLM cannot converge.
            shrink_warning = f"lfc_shrink_failed:{type(exc).__name__}:{exc}"
    result = raw.copy()
    result["log2FC_shrunk"] = shrunk.reindex(result.index)
    result["numerator"] = spec.numerator
    result["denominator"] = spec.denominator
    result["FDR_status"] = np.where(result["padj"] < alpha, "padj < 0.05", "NS")
    result["effect_status"] = np.where(result["log2FC_shrunk"].abs() >= 0.5, "abs(log2FC_shrunk) >= 0.5", "small_effect")
    result["gene"] = result.index.astype(str)

    normalized = filtered.div(dds.obs["size_factors"].reindex(filtered.index), axis=0)
    pca, pca_outlier = _pca_diagnostics(filtered, metadata)
    cooks_outliers = 0
    try:
        cooks_outliers = int(np.asarray(dds.cooks_outlier()).sum())
    except Exception:
        pass
    shrink_converged = getattr(stats, "_LFC_shrink_converged", None)
    shrink_failed = int((shrink_converged == False).sum()) if shrink_converged is not None else 0  # noqa: E712
    diagnostics = {
        "genes_before": genes_before,
        "genes_after": genes_after,
        "model_converged": True,
        "cooks_outlier_genes": cooks_outliers,
        "padj_na": int(result["padj"].isna().sum()),
        "shrink_failed": shrink_failed,
        "shrink_warning": shrink_warning,
        "runtime_warnings": "",
        "pca_outlier_candidate": bool(pca_outlier),
        "normalized": normalized,
    }
    return result.reset_index(drop=True), diagnostics, pca, normalized


def _replicate_audit(
    result: pd.DataFrame,
    normalized: pd.DataFrame,
    metadata: pd.DataFrame,
    spec: ContrastSpec,
    top_n: int = 50,
) -> tuple[pd.DataFrame, float, float, float]:
    """Export top-gene normalized expression and quantify replicate consistency."""

    ranked = result.sort_values(["padj", "pvalue", "log2FC_shrunk"], na_position="last").head(top_n)
    rows: list[dict[str, Any]] = []
    consistency_values: list[bool] = []
    top_genes = ranked["gene"].tolist()
    for rank, gene in enumerate(top_genes, start=1):
        expected = float(result.loc[result["gene"] == gene, "log2FC_shrunk"].iloc[0])
        values = normalized[gene]
        a = values.reindex(LIBRARIES_BY_GROUP[spec.denominator]).to_numpy(dtype=float)
        b = values.reindex(LIBRARIES_BY_GROUP[spec.numerator]).to_numpy(dtype=float)
        diffs = b - a
        matched = int((np.sign(diffs) == np.sign(expected)).sum()) if expected != 0 else int((np.abs(diffs) < 1e-12).sum())
        consistent = matched >= 2
        consistency_values.append(consistent)
        for library in normalized.index:
            rows.append({
                "library_id": library,
                "group": str(metadata.loc[library, "group"]),
                "gene": gene,
                "normalized_expression": float(normalized.loc[library, gene]),
                "rank": rank,
                "direction_consistency": f"{matched}/3",
                "replicate_consistency_concern": not consistent,
            })
    audit = pd.DataFrame(rows)
    finite_raw = result["log2FC_raw"].notna() & result["log2FC_shrunk"].notna()
    if finite_raw.sum() >= 2:
        correlation = float(result.loc[finite_raw, ["log2FC_raw", "log2FC_shrunk"]].corr(method="spearman").iloc[0, 1])
    else:
        correlation = float("nan")
    sign_agreement = float((np.sign(result.loc[finite_raw, "log2FC_raw"]) == np.sign(result.loc[finite_raw, "log2FC_shrunk"])).mean()) if finite_raw.any() else float("nan")
    median_abs_delta = float((result.loc[finite_raw, "log2FC_raw"] - result.loc[finite_raw, "log2FC_shrunk"]).abs().median()) if finite_raw.any() else float("nan")
    return audit, correlation, sign_agreement, median_abs_delta


def _save_qc_plots(
    result: pd.DataFrame,
    normalized: pd.DataFrame,
    metadata: pd.DataFrame,
    pca: pd.DataFrame,
    *,
    population: str,
    spec: ContrastSpec,
    figure_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    import seaborn as sns

    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {spec.denominator: "#4C78A8", spec.numerator: "#E45756"}
    groups = metadata.loc[normalized.index, "group"].astype(str)
    prefix = figure_dir / f"{safe_name(population)}_{spec.name}"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if "PC2" in pca:
        axes[0].scatter(pca["PC1"], pca["PC2"], c=[colors[g] for g in pca["group"]], s=70)
        axes[0].set_ylabel("PC2")
    else:
        axes[0].scatter(pca["PC1"], np.zeros(len(pca)), c=[colors[g] for g in pca["group"]], s=70)
        axes[0].set_ylabel("")
    for _, row in pca.iterrows():
        axes[0].text(row["PC1"], row.get("PC2", 0), row["library_id"], fontsize=8)
    axes[0].set_xlabel("PC1")
    axes[0].set_title("Pseudobulk PCA")
    corr = normalized.T.corr(method="spearman")
    sns.heatmap(corr, cmap="vlag", vmin=-1, vmax=1, annot=True, fmt=".2f", ax=axes[1], cbar=False)
    axes[1].set_title("Sample correlation")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[2].bar(normalized.index, normalized.sum(axis=1), color=[colors[g] for g in groups])
    axes[2].set_title("Library size (normalized sum)")
    axes[2].tick_params(axis="x", rotation=60)
    axes[2].set_ylabel("Sum normalized counts")
    fig.suptitle(f"{population} — {spec.name}")
    fig.tight_layout()
    fig.savefig(str(prefix) + "_pseudobulk_qc.png", dpi=180)
    plt.close(fig)

    plot = result.copy()
    plot["neg_log10_padj"] = -np.log10(plot["padj"].clip(lower=np.finfo(float).tiny))
    sig = plot["padj"] < 0.05
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(np.log10(plot["baseMean"].clip(lower=1)), plot["log2FC_shrunk"], c=np.where(sig, "#D62728", "#AAAAAA"), s=8, alpha=0.55)
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set_xlabel("log10(baseMean)")
    axes[0].set_ylabel("Shrunk log2FC")
    axes[0].set_title("MA plot")
    axes[1].scatter(plot["log2FC_shrunk"], plot["neg_log10_padj"], c=np.where(sig, "#D62728", "#AAAAAA"), s=8, alpha=0.55)
    axes[1].axhline(-np.log10(0.05), color="black", ls="--", lw=0.7)
    axes[1].set_xlabel("Shrunk log2FC")
    axes[1].set_ylabel("-log10(padj)")
    axes[1].set_title("Volcano plot")
    labels = plot.loc[sig].sort_values("padj").head(10)
    for _, row in labels.iterrows():
        axes[1].text(row["log2FC_shrunk"], row["neg_log10_padj"], row["gene"], fontsize=7)
    fig.suptitle(f"{population} — {spec.name}")
    fig.tight_layout()
    fig.savefig(str(prefix) + "_ma_volcano.png", dpi=180)
    plt.close(fig)


def _plain_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "(none)"
    return df.head(max_rows).to_string(index=False)


def run_de_stage1(config: dict[str, Any], *, allow_low_memory: bool = False) -> Path:
    """Run eligibility audit and primary broad-population pseudobulk DE."""

    require_compute_resources(config, allow_low_memory=allow_low_memory)
    paths = project_paths(config)
    logger = setup_logging("08_de_stage1", config)
    out_dir = paths["results"] / "de_stage1"
    fig_root = paths["figures"] / "de_stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)

    inputs = load_broad_inputs(config)
    counts = inputs["counts"]  # type: ignore[assignment]
    metadata = inputs["metadata"]  # type: ignore[assignment]
    pseudobulk_qc = inputs["pseudobulk_qc"]  # type: ignore[assignment]
    eligibility = build_eligibility(counts, metadata, pseudobulk_qc)
    eligibility.to_csv(out_dir / "broad_population_eligibility.tsv", sep="\t", index=False)

    audit = pd.DataFrame(
        [
            {"check": "count_nonnegative_integer", "value": True},
            {"check": "aggregation_unit", "value": "library_id × cell_type_broad_v2"},
            {"check": "source", "value": "Tier1 primary cells; layers['counts']; raw UMI counts"},
            {"check": "uses_log_normalized_X", "value": False},
            {"check": "count_rows", "value": int(counts.shape[0])},
            {"check": "feature_columns", "value": int(counts.shape[1])},
            {"check": "metadata_rows", "value": int(metadata.shape[0])},
        ]
    )
    audit.to_csv(out_dir / "input_audit.tsv", sep="\t", index=False)

    summary_rows: list[dict[str, Any]] = []
    consistency_root = out_dir / "replicate_consistency"
    n_cpus = int(config.get("pseudobulk", {}).get("n_cpus", 4))
    alpha = float(config.get("pseudobulk", {}).get("alpha", 0.05))
    for _, elig in eligibility.iterrows():
        if elig["final_de_status"] != "Primary_DE_ready":
            continue
        population = str(elig["population"])
        spec = ContrastSpec(str(elig["numerator"]), str(elig["denominator"]))
        libraries = LIBRARIES_BY_GROUP[spec.denominator] + LIBRARIES_BY_GROUP[spec.numerator]
        pop_counts = counts.loc[population].reindex(libraries)
        pop_meta = metadata[metadata["population"].astype(str) == population].set_index("library").reindex(libraries)
        try:
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            # Capture both Python warnings and the OS-level stderr inherited by
            # PyDESeq2/NumPy worker processes.  ``redirect_stderr`` alone cannot
            # see warnings emitted after a worker is spawned.
            with tempfile.TemporaryFile(mode="w+b") as fd_stderr:
                os.set_inheritable(fd_stderr.fileno(), True)
                original_stderr = os.dup(2)
                try:
                    os.dup2(fd_stderr.fileno(), 2)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                            result, diagnostics, pca, normalized = _run_one_deseq(
                                pop_counts, pop_meta, spec, alpha=alpha, n_cpus=n_cpus
                            )
                        warning_messages = {f"{w.category.__name__}: {w.message}" for w in caught}
                finally:
                    os.dup2(original_stderr, 2)
                    os.close(original_stderr)
                fd_stderr.seek(0)
                os_stderr_text = fd_stderr.read().decode(errors="replace")
            # NumPy warnings may be emitted by worker processes and therefore reach
            # the redirected OS-level stderr rather than the local warnings registry.
            warning_messages.update(
                line.strip() for line in (stderr_buffer.getvalue() + "\n" + os_stderr_text).splitlines() if "Warning" in line
            )
            warning_messages = sorted(warning_messages)
            diagnostics["runtime_warnings"] = "; ".join(warning_messages)
            pop_dir = out_dir / safe_name(population)
            pop_dir.mkdir(parents=True, exist_ok=True)
            result.to_csv(pop_dir / f"{spec.name}.tsv.gz", sep="\t", index=False, compression="gzip")
            pca.to_csv(pop_dir / f"{spec.name}_pca.tsv", sep="\t", index=False)
            consistency, lfc_corr, sign_agree, median_delta = _replicate_audit(result, normalized, pop_meta, spec)
            consistency_root.mkdir(parents=True, exist_ok=True)
            consistency.to_csv(consistency_root / f"{safe_name(population)}_{spec.name}_top50.tsv", sep="\t", index=False)
            _save_qc_plots(result, normalized, pop_meta, pca, population=population, spec=spec, figure_dir=fig_root)
            padj_sig = result["padj"] < alpha
            effect_sig = padj_sig & (result["log2FC_shrunk"].abs() >= 0.5)
            summary_rows.append(
                {
                    "population": population,
                    "contrast": spec.name,
                    "n_cells_group_A": int(elig[[f"A_lib{i}_cells" for i in range(1, 4)]].sum()),
                    "n_cells_group_B": int(elig[[f"B_lib{i}_cells" for i in range(1, 4)]].sum()),
                    "n_libraries_A": 3,
                    "n_libraries_B": 3,
                    "genes_tested": int(len(result)),
                    "padj_lt_0.05": int(padj_sig.sum()),
                    "padj_lt_0.05_and_absLFC_gt_0.5": int(effect_sig.sum()),
                    "up_genes": int((effect_sig & (result["log2FC_shrunk"] > 0)).sum()),
                    "down_genes": int((effect_sig & (result["log2FC_shrunk"] < 0)).sum()),
                    "model_converged": bool(diagnostics["model_converged"]),
                    "cooks_outlier_genes": int(diagnostics["cooks_outlier_genes"]),
                    "padj_na": int(diagnostics["padj_na"]),
                    "genes_before": int(diagnostics["genes_before"]),
                    "genes_after": int(diagnostics["genes_after"]),
                    "pca_outlier_candidate": bool(diagnostics["pca_outlier_candidate"]),
                    "lfc_spearman_raw_vs_shrunk": lfc_corr,
                    "lfc_sign_agreement": sign_agree,
                    "median_abs_lfc_shrinkage_delta": median_delta,
                    "replicate_consistency_concern": bool(consistency["replicate_consistency_concern"].any()) if not consistency.empty else True,
                    "warnings": "; ".join(
                        value for value in [diagnostics["runtime_warnings"], diagnostics["shrink_warning"]] if value
                    ) or "No warnings",
                }
            )
            logger.info("DE complete: %s %s (%d genes)", population, spec.name, len(result))
        except Exception as exc:
            logger.exception("DE failed for %s %s", population, spec.name)
            summary_rows.append(
                {
                    "population": population,
                    "contrast": spec.name,
                    "n_cells_group_A": int(elig[[f"A_lib{i}_cells" for i in range(1, 4)]].sum()),
                    "n_cells_group_B": int(elig[[f"B_lib{i}_cells" for i in range(1, 4)]].sum()),
                    "n_libraries_A": 3,
                    "n_libraries_B": 3,
                    "genes_tested": 0,
                    "padj_lt_0.05": 0,
                    "padj_lt_0.05_and_absLFC_gt_0.5": 0,
                    "up_genes": 0,
                    "down_genes": 0,
                    "model_converged": False,
                    "cooks_outlier_genes": 0,
                    "padj_na": 0,
                    "genes_before": int(counts.shape[1]),
                    "genes_after": int(elig["genes_after"]),
                    "pca_outlier_candidate": bool(elig["pca_outlier_candidate"]),
                    "lfc_spearman_raw_vs_shrunk": np.nan,
                    "lfc_sign_agreement": np.nan,
                    "median_abs_lfc_shrinkage_delta": np.nan,
                    "replicate_consistency_concern": True,
                    "warnings": f"{type(exc).__name__}: {exc}",
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "broad_de_summary.tsv", sep="\t", index=False)

    primary = eligibility[eligibility["final_de_status"] == "Primary_DE_ready"]
    sensitivity = eligibility[eligibility["final_de_status"] == "Sensitivity_only"]
    not_ready = eligibility[eligibility["final_de_status"] == "Not_DE_ready"]
    report = [
        "# DE stage 1 report",
        "",
        "本阶段仅使用已经审计的 broad pseudobulk 原始整数 UMI counts；聚合单位为 `library_id × cell_type_broad_v2`，统计学重复为 library。设计为 `~ group`，log2FC 方向为 numerator（B）相对 denominator（A）。未运行 subtype DE、通路分析、rescue 分类或任何 cell-level DE。",
        "",
        f"- 输入：{counts.shape[0]} 个 population-library pseudobulk，{counts.shape[1]} 个 feature。",
        f"- Primary_DE_ready eligibility：{len(primary)} 个 population × contrast。",
        f"- Sensitivity_only：{len(sensitivity)} 个；Not_DE_ready：{len(not_ready)} 个。",
        f"- 实际成功运行模型：{len(summary[summary.get('model_converged', pd.Series(dtype=bool)) == True]) if not summary.empty else 0} 个。",
        "",
        "## Eligibility",
        "",
        _plain_table(eligibility[["population", "contrast", "group_A", "group_B", "min_cells", "eligibility_20", "eligibility_50", "eligibility_100", "pseudobulk_qc_concern", "final_de_status"]]),
        "",
        "## DE summary",
        "",
        _plain_table(summary),
        "",
        "## Interpretation guardrails",
        "",
        "PCA、相关性、library size、MA 和 volcano 图仅用于模型诊断。Shrunk LFC 只用于 effect-size 展示和排序，p 值/FDR 来自原始 Wald 统计。任何单个 library 驱动的 top genes 均在 `replicate_consistency` 表中标记，未据此删除样本或细胞。",
    ]
    (out_dir / "DE_STAGE1_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    logger.info("DE_STAGE1_OK: %s", out_dir)
    return out_dir
