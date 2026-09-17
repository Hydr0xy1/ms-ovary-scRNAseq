"""Stage 6: independent signature-level validation with GSE232309."""
# ruff: noqa: E501

from __future__ import annotations

import json
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .de_stage1 import prefilter_genes
from .pathway_stage2 import parse_gmt
from .project import project_paths, setup_logging
from .stage7_regulatory_activity import sha256_file

REFERENCE = {
    "title": "A single-cell atlas of the aging mouse ovary",
    "PMID": "38200272",
    "PMCID": "PMC10798902",
    "DOI": "10.1038/s43587-023-00552-5",
    "GEO": "GSE232309",
    "license": "CC BY 4.0",
    "young_age": "3 months",
    "aged_age": "9 months",
    "n_per_group": 4,
}


def signature_concordance(internal: pd.Series, external: pd.Series) -> dict[str, float]:
    joined = pd.concat([internal.rename("internal"), external.rename("external")], axis=1).dropna()
    if len(joined) < 3:
        return {"n_genes": len(joined), "spearman": np.nan, "cosine": np.nan, "same_direction_fraction": np.nan}
    a = joined["internal"].to_numpy(dtype=float)
    b = joined["external"].to_numpy(dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {
        "n_genes": len(joined),
        "spearman": float(joined["internal"].corr(joined["external"], method="spearman")),
        "cosine": float(np.dot(a, b) / denominator) if denominator > 0 else np.nan,
        "same_direction_fraction": float((np.sign(a) == np.sign(b)).mean()),
    }


def set_population_column(table: pd.DataFrame, population: str) -> pd.DataFrame:
    """Set a single population column and place it first."""
    result = table.copy()
    result["population"] = population
    return result[["population", *[column for column in result.columns if column != "population"]]]


def _write_skip(output_root: Path, reason: str) -> None:
    (output_root / "EXTERNAL_VALIDATION_NOT_AVAILABLE.md").write_text(
        "# Stage 6 external validation skipped\n\n"
        f"已尝试使用GEO官方processed RDS（GSE232309），但无法可靠完成：`{reason}`。"
        "没有下载FASTQ、没有使用不可信镜像，也没有强行跨数据集batch correction。后续Stage继续运行。\n",
        encoding="utf-8",
    )
    print("STAGE6_EXTERNAL_VALIDATION_SKIPPED")


def prepare_external_pseudobulk_counts(counts: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Round sample-aggregated SoupX-adjusted counts for the DESeq2 likelihood.

    GSE232309 stores SoupX ambient-RNA-corrected values in the Seurat counts
    slot.  They are non-negative count-like values but are fractional.  We
    preserve them through cell-to-sample aggregation and round only the final
    pseudobulk totals, avoiding the larger distortion from per-cell rounding.
    """
    values = counts.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("External pseudobulk contains non-finite values")
    if (values < 0).any():
        raise ValueError("External pseudobulk contains negative values")
    rounded = np.rint(values).astype(np.int64)
    delta = np.abs(values - rounded)
    audit: dict[str, float | int | str] = {
        "source_count_provenance": "SoupX ambient-RNA-corrected Seurat counts layer",
        "rounding_stage": "after biological-sample pseudobulk aggregation",
        "rounding_method": "nearest integer (numpy.rint)",
        "n_fractional_pseudobulk_entries": int((delta > 1e-8).sum()),
        "max_absolute_rounding_delta": float(delta.max(initial=0.0)),
    }
    return pd.DataFrame(rounded, index=counts.index, columns=counts.columns), audit


def _run_external_deseq(counts: pd.DataFrame, metadata: pd.DataFrame, n_cpus: int = 4) -> pd.DataFrame:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    metadata = metadata.set_index("sample_id").reindex(counts.index).copy()
    metadata["age_group"] = pd.Categorical(metadata["age_group"], categories=["Young", "Aged"])
    integer_counts, _ = prepare_external_pseudobulk_counts(counts)
    filtered = prefilter_genes(integer_counts, min_count=10, min_samples=3)
    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        counts=filtered,
        metadata=metadata,
        design="~age_group",
        refit_cooks=True,
        inference=inference,
        low_memory=True,
    )
    dds.deseq2()
    stats = DeseqStats(
        dds,
        contrast=["age_group", "Aged", "Young"],
        alpha=0.05,
        cooks_filter=True,
        independent_filter=True,
        inference=inference,
        quiet=True,
    )
    stats.summary()
    result = stats.results_df.reset_index().rename(columns={"index": "gene", "log2FoldChange": "external_aging_log2FC", "padj": "external_aging_FDR", "pvalue": "external_aging_pvalue", "stat": "external_aging_stat"})
    return result


def _run_gsea(ranking: pd.DataFrame, gene_sets: dict[str, list[str]], population: str) -> pd.DataFrame:
    import gseapy as gp

    ordered = ranking[["gene", "external_aging_stat"]].dropna().drop_duplicates("gene").sort_values(["external_aging_stat", "gene"], ascending=[False, True])
    pre = gp.prerank(
        rnk=ordered,
        gene_sets=gene_sets,
        permutation_num=1000,
        min_size=10,
        max_size=500,
        weight=1.0,
        ascending=False,
        threads=4,
        seed=20260810,
        outdir=None,
        no_plot=True,
        verbose=False,
    )
    result = pre.res2d.rename(columns={"Term": "pathway", "NES": "external_aging_NES", "FDR q-val": "external_aging_FDR_q", "NOM p-val": "external_aging_nominal_p"})
    result.insert(0, "population", population)
    return result[["population", "pathway", "external_aging_NES", "external_aging_FDR_q", "external_aging_nominal_p"]]


def run_stage6(config: Mapping[str, Any], *, rscript: str) -> None:
    paths = project_paths(config)
    output_root = paths["results"] / "stage6_external_validation"
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("15_stage6_external_validation", dict(config))
    data_root = paths["root"] / "external_data" / "GSE232309"
    files = {
        "Granulosa": data_root / "GSE232309_Granulosa_Cells.rds.gz",
        "Stromal_fibroblast": data_root / "GSE232309_Stroma_Cells.rds.gz",
    }
    missing = [str(path) for path in files.values() if not path.exists() or path.stat().st_size < 10_000_000]
    if missing or not Path(rscript).exists():
        _write_skip(output_root, f"missing processed files or R environment: {missing}; Rscript={rscript}")
        return
    extract_root = output_root / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    extractor = paths["root"] / "scripts" / "15_extract_gse232309.R"
    try:
        for population, rds in files.items():
            result = subprocess.run(
                [rscript, str(extractor), str(rds), population, str(extract_root)],
                cwd=paths["root"],
                text=True,
                capture_output=True,
                timeout=3600,
            )
            (extract_root / f"{population}__extract.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(f"{population} extraction failed: {result.stderr[-1500:]}")
    except Exception as exc:
        _write_skip(output_root, f"official processed RDS extraction failed: {type(exc).__name__}: {exc}")
        return

    metadata_frames: list[pd.DataFrame] = []
    signatures: list[pd.DataFrame] = []
    concordance_rows: list[dict[str, Any]] = []
    reversal_rows: list[pd.DataFrame] = []
    gsea_rows: list[pd.DataFrame] = []
    mapping = pd.DataFrame(
        [
            {"external_object": "Granulosa_Cells", "external_cell_type": "Granulosa", "internal_population": "Granulosa", "mapping_basis": "lineage object plus canonical markers"},
            {"external_object": "Stroma_Cells", "external_cell_type": "Stroma", "internal_population": "Stromal_fibroblast", "mapping_basis": "lineage object plus ECM/fibroblast markers"},
        ]
    )
    gene_sets = parse_gmt(paths["root"] / "resources" / "gene_sets" / "mh.all.v2026.1.Mm.symbols.gmt")
    for population in files:
        metadata = pd.read_csv(extract_root / f"{population}__sample_metadata.tsv", sep="\t")
        metadata_frames.append(metadata)
        group_counts = metadata.groupby("age_group", observed=True)["sample_id"].nunique().to_dict()
        if group_counts != {"Aged": 4, "Young": 4}:
            _write_skip(output_root, f"{population} biological-sample audit failed: {group_counts}")
            return
        counts = pd.read_csv(extract_root / f"{population}__pseudobulk_counts.tsv.gz", sep="\t").set_index("sample_id")
        _, count_audit = prepare_external_pseudobulk_counts(counts)
        for key, value in count_audit.items():
            metadata[key] = value
        result = _run_external_deseq(counts, metadata)
        result.insert(0, "population", population)
        signatures.append(result)
        internal = pd.read_csv(
            paths["results"]
            / "de_stage1_5"
            / population
            / "rescue_ready_effects.tsv.gz",
            sep="\t",
        )
        joined = internal[["gene", "aging_effect", "treatment_effect", "residual_effect", "aging_padj", "treatment_padj"]].merge(result, on="gene", how="inner")
        metrics = signature_concordance(joined.set_index("gene")["aging_effect"], joined.set_index("gene")["external_aging_log2FC"])
        concordance_rows.append({"population": population, "scope": "all_overlapping_tested_genes", **metrics, "external_reference": REFERENCE["GEO"]})
        significant = joined.loc[joined["external_aging_FDR"].lt(0.05) & joined["aging_padj"].lt(0.05)]
        metrics_sig = signature_concordance(significant.set_index("gene")["aging_effect"], significant.set_index("gene")["external_aging_log2FC"])
        concordance_rows.append({"population": population, "scope": "both_aging_FDR_lt_0.05", **metrics_sig, "external_reference": REFERENCE["GEO"]})
        joined["direction_concordant_external_aging"] = joined["aging_effect"] * joined["external_aging_log2FC"] > 0
        joined["external_aging_supported"] = joined["external_aging_FDR"].lt(0.05) & joined["direction_concordant_external_aging"]
        joined["treatment_opposes_external_aging"] = joined["treatment_effect"] * joined["external_aging_log2FC"] < 0
        joined["external_supported_reversal"] = joined["external_aging_supported"] & joined["treatment_opposes_external_aging"]
        # ``result`` already carries population, so assignment must be idempotent.
        joined = set_population_column(joined, population)
        reversal_rows.append(joined)
        gsea_rows.append(_run_gsea(result, gene_sets, population))
        logger.info("External DE complete: %s", population)

    reference_metadata = pd.concat(metadata_frames, ignore_index=True)
    for key, value in REFERENCE.items():
        reference_metadata[f"reference_{key}"] = value
    signature = pd.concat(signatures, ignore_index=True)
    concordance = pd.DataFrame(concordance_rows)
    reversal = pd.concat(reversal_rows, ignore_index=True)
    external_gsea = pd.concat(gsea_rows, ignore_index=True)
    reference_metadata.to_csv(output_root / "reference_metadata.tsv", sep="\t", index=False)
    mapping.to_csv(output_root / "celltype_mapping.tsv", sep="\t", index=False)
    signature.to_csv(output_root / "external_aging_signature.tsv", sep="\t", index=False)
    concordance.to_csv(output_root / "internal_external_concordance.tsv", sep="\t", index=False)
    reversal.to_csv(output_root / "external_signature_reversal.tsv", sep="\t", index=False)
    external_gsea.to_csv(output_root / "external_hallmark.tsv", sep="\t", index=False)

    stage2 = pd.read_csv(paths["results"] / "pathway_stage2" / "primary_pathway_reversal_summary.tsv", sep="\t")
    strong = stage2.loc[stage2["pathway_evidence_level"].astype(str).eq("Strong_multi_metric_reversal"), ["population", "pathway", "aging_NES", "treatment_NES"]]
    pathway_check = strong.merge(external_gsea, on=["population", "pathway"], how="left")
    pathway_check["external_internal_aging_same_direction"] = pathway_check["aging_NES"] * pathway_check["external_aging_NES"] > 0
    pathway_check["treatment_opposes_external_aging"] = pathway_check["treatment_NES"] * pathway_check["external_aging_NES"] < 0
    pathway_check.to_csv(output_root / "external_strong_pathway_validation.tsv", sep="\t", index=False)

    report = [
        "# Stage 6 外部卵巢衰老验证报告",
        "",
        "## 1. 为什么做？",
        "内部aging signature和MRJP1 reversal都来自同一批Y/OC/OT。独立公开数据可减少所有结论围绕同一批样本循环验证的问题。",
        "",
        "## 2. Reference",
        pd.DataFrame([REFERENCE]).to_markdown(index=False),
        "使用GEO官方processed Granulosa/Stroma RDS，不下载FASTQ。reference为3月龄vs9月龄，每组4个biological samples；license为CC BY 4.0。",
        "",
        "## 3. 方法",
        "公开RDS的counts层是论文所述SoupX去环境RNA后的非负小数count-like值。先在每个biological sample内求和，再仅对最终pseudobulk总数做最近整数舍入以满足DESeq2负二项模型；不把它误称为未经处理的原始UMI。随后使用~age_group的PyDESeq2；不把external与internal做Harmony或强制batch correction。比较层面是gene-signature Spearman/cosine、方向一致率和Mouse Hallmark preranked GSEA。",
        "",
        "## 4. Sample audit",
        reference_metadata[["population", "sample_id", "age_group", "n_cells", "total_umi"]].to_markdown(index=False),
        "",
        "## 5. Internal vs external aging",
        concordance.to_markdown(index=False),
        "",
        "## 6. Strong Hallmark复现",
        pathway_check.to_markdown(index=False),
        "",
        "## 7. 可以说明什么？",
        "正向concordance支持内部OC vs Y反映可在独立卵巢衰老atlas中观察到的方向；MRJP1 effect若与external aging相反，则提供独立signature层面的支持。",
        "",
        "## 8. 不能说明什么？",
        "不同年龄、平台、解离、注释与鼠品系会影响一致性。外部一致不能证明MRJP1导致表型年轻化，也不能替代本项目新增biological replicates。",
        "",
        "## 9. 局限与下一步",
        "只验证Granulosa和Stromal两个可可靠映射lineage；subtype名称不强行一一对应。公开对象只保留SoupX校正后的count-like层，最近整数舍入是外部验证的额外技术局限。未来应使用预注册的新实验队列和matched phenotype复现。",
    ]
    (output_root / "EXTERNAL_AGING_VALIDATION_REPORT_CN.md").write_text("\n".join(report), encoding="utf-8")

    outputs = [p for p in output_root.rglob("*") if p.is_file() and p.name not in {"manifest.tsv", "COMPLETE.json"}]
    manifest = pd.DataFrame([{"path": str(p.relative_to(paths["root"])), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in outputs])
    manifest.to_csv(output_root / "manifest.tsv", sep="\t", index=False)
    complete = {
        "stage": 6,
        "status": "COMPLETE",
        "reference": REFERENCE,
        "statistical_unit": "external biological sample",
        "n_per_group": 4,
        "software": {"pydeseq2": version("pydeseq2"), "gseapy": version("gseapy")},
        "source_checksums": {population: sha256_file(path) for population, path in files.items()},
        "manifest_sha256": sha256_file(output_root / "manifest.tsv"),
    }
    (output_root / "COMPLETE.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print("STAGE6_EXTERNAL_VALIDATION_COMPLETE")
