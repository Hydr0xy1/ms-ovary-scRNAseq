"""Stage 14: conventional scRNA-seq completeness and supplementary analyses.

The stage is read-only with respect to the annotated AnnData and Stage 1-13
outputs. It reuses the audited nine-library DE and composition results, then
adds conventional abundance displays, mouse-native functional annotation,
cell-cycle description, marker catalogues, and standard DE visualizations.
"""
# ruff: noqa: E501

from __future__ import annotations

import gc
import hashlib
import json
import re
import subprocess
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom

from .de_stage1_5 import ALL_LIBRARIES
from .pathway_stage2 import POPULATION_TIERS, deduplicate_rank_table, parse_gmt
from .project import load_yaml, project_paths, require_compute_resources, setup_logging
from .stage13_publication import (
    CELL_TYPE_COLORS,
    GROUP_COLORS,
    SUBTYPE_COLORS,
    configure_publication_style,
)

GROUPS = ("Y", "OC", "OT")
PHASES = ("G1", "S", "G2M")
FUNCTIONAL_DATABASES = ("GO_BP", "Reactome", "KEGG")
MSIGDB_RELEASE = "2026.1.Mm"
MSIGDB_BASE_URL = (
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2026.1.Mm"
)
RESOURCE_SPECS = {
    "GO_BP": {
        "filename": "m5.go.bp.v2026.1.Mm.symbols.gmt",
        "url": f"{MSIGDB_BASE_URL}/m5.go.bp.v2026.1.Mm.symbols.gmt",
        "source": "Mouse MSigDB M5 GO Biological Process",
        "release": MSIGDB_RELEASE,
    },
    "Reactome": {
        "filename": "m2.cp.reactome.v2026.1.Mm.symbols.gmt",
        "url": f"{MSIGDB_BASE_URL}/m2.cp.reactome.v2026.1.Mm.symbols.gmt",
        "source": "Mouse MSigDB M2 CP Reactome",
        "release": MSIGDB_RELEASE,
    },
}
KEGG_ENRICHR_URL = (
    "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=KEGG_2019_Mouse"
)

PRIMARY_CONTRAST_LABELS = {
    "OC_vs_Y": "Aging: OC versus Y",
    "OT_vs_OC": "Treatment: OT versus OC",
    "OT_vs_Y": "Residual: OT versus Y",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _read_tsv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", low_memory=False)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _write_tsv(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)
    return path


def assign_cell_cycle_phase(s_score: Sequence[float], g2m_score: Sequence[float]) -> np.ndarray:
    """Match Scanpy's phase rule while keeping it independently testable."""

    s = np.asarray(s_score, dtype=float)
    g = np.asarray(g2m_score, dtype=float)
    if s.shape != g.shape or not np.isfinite(s).all() or not np.isfinite(g).all():
        raise ValueError("Cell-cycle scores must be finite arrays with identical shape")
    phase = np.where(s > g, "S", "G2M").astype(object)
    phase[(s < 0) & (g < 0)] = "G1"
    return phase.astype(str)


def compute_broad_abundance(
    obs: pd.DataFrame,
    *,
    sample_key: str = "library_id",
    group_key: str = "group",
    broad_key: str = "cell_type_broad_v2",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return library-by-cell-type counts and an auditable long fraction table."""

    required = {sample_key, group_key, broad_key}
    missing = required - set(obs.columns)
    if missing:
        raise KeyError(f"Abundance metadata lacks columns: {sorted(missing)}")
    work = obs[[sample_key, group_key, broad_key]].astype(str).copy()
    groups_per_library = work.groupby(sample_key, observed=True)[group_key].nunique()
    if not groups_per_library.eq(1).all():
        raise ValueError("Each library must map to exactly one group")
    libraries = [library for library in ALL_LIBRARIES if library in set(work[sample_key])]
    cell_types = sorted(work[broad_key].unique())
    counts = pd.crosstab(work[sample_key], work[broad_key]).reindex(
        index=libraries, columns=cell_types, fill_value=0
    )
    counts.index.name = "library_id"
    long = counts.stack(future_stack=True).rename("n_cells").reset_index()
    long = long.rename(columns={broad_key: "cell_type"})
    totals = counts.sum(axis=1).rename("total_cells")
    long = long.merge(totals, left_on="library_id", right_index=True, validate="many_to_one")
    long["group"] = long["library_id"].str.split("_", n=1).str[0]
    long["fraction_of_total_cells"] = long["n_cells"] / long["total_cells"]
    if not np.allclose(
        long.groupby("library_id", observed=True)["fraction_of_total_cells"].sum(), 1.0
    ):
        raise AssertionError("Broad cell fractions do not sum to one within library")
    return counts.reset_index(), long


def compute_subtype_abundance(
    obs: pd.DataFrame,
    parent: str,
    *,
    sample_key: str = "library_id",
    group_key: str = "group",
    broad_key: str = "cell_type_broad_v2",
    subtype_key: str = "cell_type_subtype_v2",
) -> pd.DataFrame:
    """Calculate subtype/parent and subtype/whole-ovary fractions per library."""

    required = {sample_key, group_key, broad_key, subtype_key}
    missing = required - set(obs.columns)
    if missing:
        raise KeyError(f"Subtype abundance metadata lacks columns: {sorted(missing)}")
    work = obs[list(required)].astype(str).copy()
    whole_totals = work.groupby(sample_key, observed=True).size()
    parent_obs = work.loc[work[broad_key].eq(parent)].copy()
    subtypes = sorted(parent_obs[subtype_key].unique())
    counts = pd.crosstab(parent_obs[sample_key], parent_obs[subtype_key]).reindex(
        index=ALL_LIBRARIES, columns=subtypes, fill_value=0
    )
    parent_totals = counts.sum(axis=1)
    rows: list[dict[str, Any]] = []
    for library in ALL_LIBRARIES:
        for subtype in subtypes:
            n_cells = int(counts.loc[library, subtype])
            parent_n = int(parent_totals.loc[library])
            whole_n = int(whole_totals.loc[library])
            rows.append(
                {
                    "parent_broad": parent,
                    "library_id": library,
                    "group": library.split("_", 1)[0],
                    "subtype": subtype,
                    "n_cells": n_cells,
                    "parent_cells": parent_n,
                    "whole_ovary_cells": whole_n,
                    "fraction_of_parent": n_cells / parent_n if parent_n else np.nan,
                    "fraction_of_whole_ovary": n_cells / whole_n if whole_n else np.nan,
                }
            )
    result = pd.DataFrame(rows)
    sums = result.groupby("library_id", observed=True)["fraction_of_parent"].sum()
    if not np.allclose(sums, 1.0, equal_nan=False):
        raise AssertionError(f"{parent} subtype fractions do not sum to one")
    return result


def _abundance_summary(
    broad: pd.DataFrame,
    subtype_tables: Mapping[str, pd.DataFrame],
    stage4_composition: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell_type, sub in broad.groupby("cell_type", observed=True, sort=True):
        means = sub.groupby("group", observed=True)["fraction_of_total_cells"].mean()
        sds = sub.groupby("group", observed=True)["fraction_of_total_cells"].std()
        rows.append(
            {
                "level": "broad",
                "parent_broad": "whole_ovary",
                "component": cell_type,
                **{f"{group}_mean_fraction": float(means.get(group, np.nan)) for group in GROUPS},
                **{f"{group}_sd_fraction": float(sds.get(group, np.nan)) for group in GROUPS},
                "aging_delta_OC_minus_Y": float(means.get("OC", np.nan) - means.get("Y", np.nan)),
                "treatment_delta_OT_minus_OC": float(means.get("OT", np.nan) - means.get("OC", np.nan)),
                "formal_interpretation": "descriptive_library_level; no cell-level test",
                "stage4_source": "not_applicable_to_broad_fraction",
            }
        )
    for parent, table in subtype_tables.items():
        for subtype, sub in table.groupby("subtype", observed=True, sort=True):
            means = sub.groupby("group", observed=True)["fraction_of_parent"].mean()
            sds = sub.groupby("group", observed=True)["fraction_of_parent"].std()
            has_stage4 = bool(
                not stage4_composition.empty
                and stage4_composition["broad_population"].astype(str).eq(parent).any()
            )
            rows.append(
                {
                    "level": "subtype_within_parent",
                    "parent_broad": parent,
                    "component": subtype,
                    **{f"{group}_mean_fraction": float(means.get(group, np.nan)) for group in GROUPS},
                    **{f"{group}_sd_fraction": float(sds.get(group, np.nan)) for group in GROUPS},
                    "aging_delta_OC_minus_Y": float(means.get("OC", np.nan) - means.get("Y", np.nan)),
                    "treatment_delta_OT_minus_OC": float(means.get("OT", np.nan) - means.get("OC", np.nan)),
                    "formal_interpretation": "reuse_CLR_Aitchison; n=3/group; no scCODA claim" if has_stage4 else "descriptive_library_level_only",
                    "stage4_source": "results/stage4_composition_decomposition/composition_by_library.tsv" if has_stage4 else "not_available",
                }
            )
    return pd.DataFrame(rows)


def _initial_audit() -> pd.DataFrame:
    rows = [
        ("A", "Broad cell-type abundance/composition summary", "DONE", "DONE", "results/composition/library_broad_fractions.tsv;results/publication_stage13/source_data/atlas_library_composition.tsv", "Reformat into conventional library-level tables and standalone figures."),
        ("B", "Subtype abundance/composition summary", "PARTIAL", "DONE", "results/stage4_composition_decomposition/composition_by_library.tsv;results/annotation_v2/subtype_census.tsv", "Add Granulosa, Stromal and Immune parent-denominator tables and figures."),
        ("C", "Formal differential abundance interpretation", "DONE", "DONE", "results/stage4_composition_decomposition/clr_composition.tsv;results/stage4_composition_decomposition/aitchison_distance.tsv", "Reuse CLR/Aitchison; scCODA not added because n=3/group and Stage 4 is adequate."),
        ("D", "GO Biological Process enrichment", "MISSING", "DONE", "results/stage14_conventional/enrichment/go_bp_gsea.tsv", "Add mouse-native ranked GSEA and secondary ORA."),
        ("E", "KEGG enrichment", "MISSING", "DONE", "results/stage14_conventional/enrichment/kegg_gsea.tsv", "Add official KEGG mouse ranked GSEA and secondary ORA."),
        ("F", "Reactome enrichment", "MISSING", "DONE", "results/stage14_conventional/enrichment/reactome_gsea.tsv", "Add Mouse MSigDB Reactome ranked GSEA and secondary ORA."),
        ("G", "Cell-cycle phase scoring/audit", "PARTIAL", "DONE", "results/stage14_conventional/cell_cycle/cell_cycle_interpretation.tsv", "Score without regression and summarize by library/subtype."),
        ("H", "Broad cell-type marker catalogue", "PARTIAL", "DONE", "results/stage14_conventional/markers/broad_cell_marker_catalogue.tsv", "Convert earlier marker evidence into a unified descriptive catalogue."),
        ("I", "Subtype marker catalogue", "PARTIAL", "DONE", "results/stage14_conventional/markers/subtype_marker_catalogue.tsv", "Add within-parent subtype specificity catalogues."),
        ("J", "Standard pseudobulk volcano plots", "MISSING", "DONE", "figures/stage14_conventional/volcano", "Render from existing unified DE only."),
        ("K", "Standard pseudobulk DEG heatmaps", "MISSING", "DONE", "figures/stage14_conventional/heatmap", "Render nine-library pseudobulk heatmaps from existing counts."),
        ("L", "Top DEG tables", "PARTIAL", "DONE", "results/stage14_conventional/de_visualization/top_deg_tables.tsv", "Apply fixed FDR then absolute-LFC ordering."),
        ("M", "Annotation marker dotplot/heatmap", "PARTIAL", "DONE", "figures/stage14_conventional/markers", "Add broad, Granulosa, Stromal and Immune dotplots."),
        ("N", "Trajectory/pseudotime", "NOT_APPLICABLE", "NOT_APPLICABLE", "results/stage14_conventional/NOT_RECOMMENDED_CONVENTIONAL_ANALYSES.md", "Cross-sectional adult ovary; not required for the stated question."),
        ("O", "RNA velocity", "NOT_APPLICABLE", "NOT_APPLICABLE", "results/stage14_conventional/NOT_RECOMMENDED_CONVENTIONAL_ANALYSES.md", "No reliable spliced/unspliced input."),
        ("P", "Ambient RNA correction", "NOT_APPLICABLE", "NOT_APPLICABLE", "results/stage14_conventional/NOT_RECOMMENDED_CONVENTIONAL_ANALYSES.md", "Only filtered matrices; no raw droplets and no audit evidence requiring late matrix replacement."),
        ("Q", "CNV inference", "NOT_APPLICABLE", "NOT_APPLICABLE", "results/stage14_conventional/NOT_RECOMMENDED_CONVENTIONAL_ANALYSES.md", "Non-tumor study without a malignant-cell question."),
    ]
    return pd.DataFrame(
        rows,
        columns=["module_id", "module", "status_before_stage14", "status", "source_files", "stage14_action_or_rationale"],
    )


def parse_generic_gmt(path: str | Path) -> dict[str, list[str]]:
    """Parse a GMT while preserving mouse symbols and rejecting malformed sets."""

    gene_sets: dict[str, list[str]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"Malformed GMT line {line_number} in {path}")
            name = fields[0].strip()
            genes = list(dict.fromkeys(g.strip() for g in fields[2:] if g.strip()))
            if not name or name in gene_sets:
                raise ValueError(f"Missing or duplicate gene-set name at line {line_number}")
            gene_sets[name] = genes
    if not gene_sets:
        raise ValueError(f"No gene sets parsed from {path}")
    return gene_sets


def _download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ms-ovary-scrna-stage14/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    if len(payload) < 100:
        raise RuntimeError(f"Downloaded resource is unexpectedly small: {url}")
    path.write_bytes(payload)
    return path


def _build_enrichr_mouse_kegg_gmt(
    resource_dir: Path,
    canonical_mouse_symbols: Sequence[str],
) -> Path:
    """Map the versioned Enrichr mouse KEGG library to audited mouse symbols."""

    raw_path = resource_dir / "KEGG_2019_Mouse.enrichr.txt"
    if not raw_path.exists():
        _download(KEGG_ENRICHR_URL, raw_path)
    candidates: dict[str, list[str]] = defaultdict(list)
    for symbol in map(str, canonical_mouse_symbols):
        candidates[symbol.upper()].append(symbol)
    lookup = {key: values[0] for key, values in candidates.items() if len(set(values)) == 1}
    output = resource_dir / "kegg_2019_mouse.canonical_symbols.gmt"
    with raw_path.open(encoding="utf-8") as source, output.open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for line_number, line in enumerate(source, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 2:
                raise ValueError(f"Malformed Enrichr KEGG mouse line {line_number}")
            name = "KEGG_MOUSE_2019_" + _safe_name(fields[0]).upper()
            mapped = sorted(
                {lookup[gene.strip().upper()] for gene in fields[1:] if gene.strip().upper() in lookup},
                key=str.upper,
            )
            if mapped:
                handle.write("\t".join([name, KEGG_ENRICHR_URL, *mapped]) + "\n")
    if len(parse_generic_gmt(output)) < 100:
        raise RuntimeError("KEGG_2019_Mouse yielded fewer than 100 mapped pathways")
    return output


def prepare_mouse_gene_sets(
    resource_dir: Path,
    canonical_mouse_symbols: Sequence[str],
) -> tuple[dict[str, Path], pd.DataFrame]:
    """Download versioned mouse-native GO/Reactome and official KEGG mouse sets."""

    resource_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    for database, spec in RESOURCE_SPECS.items():
        path = resource_dir / str(spec["filename"])
        if not path.exists():
            _download(str(spec["url"]), path)
        gene_sets = parse_generic_gmt(path)
        paths[database] = path
        rows.append(
            {
                "database": database,
                "species": "Mus musculus",
                "release_version": spec["release"],
                "source": spec["source"],
                "download_url": spec["url"],
                "download_date": date.today().isoformat(),
                "file_path": str(path),
                "sha256": sha256_file(path),
                "gene_id_type": "official mouse gene symbol",
                "mapping_rule": "direct case-sensitive canonical mouse symbol match",
                "n_gene_sets": len(gene_sets),
            }
        )
    kegg_path = _build_enrichr_mouse_kegg_gmt(resource_dir, canonical_mouse_symbols)
    paths["KEGG"] = kegg_path
    rows.append(
        {
            "database": "KEGG",
            "species": "Mus musculus",
            "release_version": "KEGG_2019_Mouse",
            "source": "Enrichr KEGG 2019 Mouse library",
            "download_url": KEGG_ENRICHR_URL,
            "download_date": date.today().isoformat(),
            "file_path": str(kegg_path),
            "sha256": sha256_file(kegg_path),
            "gene_id_type": "mouse gene symbol",
            "mapping_rule": "case-insensitive exact lookup to audited Cell Ranger canonical mouse symbols; no human ortholog conversion",
            "n_gene_sets": len(parse_generic_gmt(kegg_path)),
        }
    )
    return paths, pd.DataFrame(rows)


def validate_gene_set_mapping(
    gene_sets: Mapping[str, Sequence[str]],
    tested_genes: Iterable[str],
    *,
    min_total_overlap: int = 100,
) -> dict[str, Any]:
    tested = set(map(str, tested_genes))
    if len(tested) != len(list(map(str, tested_genes))):
        raise ValueError("Tested-gene universe contains duplicate identifiers")
    members = set().union(*(set(map(str, values)) for values in gene_sets.values()))
    overlap = members & tested
    if len(overlap) < min_total_overlap:
        raise ValueError(f"Only {len(overlap)} mouse gene symbols overlap the tested universe")
    return {
        "n_gene_sets": len(gene_sets),
        "n_unique_gene_set_members": len(members),
        "n_tested_genes": len(tested),
        "n_overlapping_genes": len(overlap),
        "tested_gene_overlap_fraction": len(overlap) / len(tested) if tested else np.nan,
    }


def validate_enrichment_schema(table: pd.DataFrame) -> None:
    required = {
        "population",
        "contrast",
        "database",
        "term",
        "NES",
        "nominal_p",
        "FDR_q",
        "leading_edge",
        "ranking_metric",
        "statistical_unit",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Enrichment result lacks columns: {sorted(missing)}")
    if not table["database"].isin(FUNCTIONAL_DATABASES).all():
        raise ValueError("Unknown functional database in enrichment result")
    if not table["statistical_unit"].astype(str).eq("library").all():
        raise ValueError("Enrichment result does not preserve library as the statistical unit")


def validate_output_manifest(manifest: pd.DataFrame, root: Path) -> None:
    required = {"path", "bytes", "sha256"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest lacks columns: {sorted(missing)}")
    if manifest["path"].duplicated().any():
        raise ValueError("Manifest contains duplicate paths")
    for row in manifest.itertuples(index=False):
        path = root / str(row.path)
        if not path.exists() or path.stat().st_size != int(row.bytes):
            raise FileNotFoundError(f"Manifest size/path mismatch: {path}")
        if sha256_file(path) != str(row.sha256):
            raise ValueError(f"Manifest checksum mismatch: {path}")


def _run_gsea_task(task: Mapping[str, Any]) -> pd.DataFrame:
    import gseapy as gp

    ranking = pd.read_csv(task["rank_path"], sep="\t", header=None, names=["gene", "stat"])
    ranking["gene"] = ranking["gene"].astype(str)
    ranking["stat"] = pd.to_numeric(ranking["stat"], errors="raise")
    if ranking["gene"].duplicated().any() or not np.isfinite(ranking["stat"]).all():
        raise ValueError(f"Invalid ranked list: {task['rank_path']}")
    gene_sets = parse_generic_gmt(task["gmt_path"])
    pre = gp.prerank(
        rnk=ranking,
        gene_sets=gene_sets,
        permutation_num=int(task["permutations"]),
        min_size=int(task["min_size"]),
        max_size=int(task["max_size"]),
        weight=1.0,
        ascending=False,
        threads=int(task["threads"]),
        seed=int(task["seed"]),
        outdir=None,
        no_plot=True,
        verbose=False,
    )
    result = pre.res2d.rename(
        columns={
            "Term": "term",
            "NOM p-val": "nominal_p",
            "FDR q-val": "FDR_q",
            "FWER p-val": "FWER_p",
            "Lead_genes": "leading_edge",
        }
    ).copy()
    required = {"term", "ES", "NES", "nominal_p", "FDR_q", "FWER_p"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Unexpected GSEA schema, missing {sorted(missing)}")
    result.insert(0, "database", task["database"])
    result.insert(0, "contrast", task["contrast"])
    result.insert(0, "population_evidence_tier", POPULATION_TIERS[task["population"]])
    result.insert(0, "population", task["population"])
    overlap_rows = []
    universe = set(ranking["gene"])
    for term, genes in gene_sets.items():
        overlap_rows.append(
            {
                "term": term,
                "gene_set_size": len(set(genes)),
                "n_matched_genes": len(set(genes) & universe),
            }
        )
    result = result.merge(pd.DataFrame(overlap_rows), on="term", how="left", validate="one_to_one")
    for column in ("ES", "NES", "nominal_p", "FDR_q", "FWER_p"):
        result[column] = pd.to_numeric(result[column], errors="raise")
    result["gsea_permutations"] = int(task["permutations"])
    result["ranking_metric"] = "unified_9_library_DE_Wald_statistic"
    result["statistical_unit"] = "library"
    result["n_libraries_per_group"] = 3
    return result


def run_functional_gsea(
    ranked_dir: Path,
    resource_paths: Mapping[str, Path],
    *,
    populations: Sequence[str],
    contrasts: Sequence[str],
    permutations: int,
    min_size: int,
    max_size: int,
    workers: int,
    threads: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    tasks: list[dict[str, Any]] = []
    for population in populations:
        for contrast in contrasts:
            rank_path = ranked_dir / f"{population}__{contrast}.rnk"
            if not rank_path.exists():
                raise FileNotFoundError(rank_path)
            for database, gmt_path in resource_paths.items():
                tasks.append(
                    {
                        "population": population,
                        "contrast": contrast,
                        "database": database,
                        "rank_path": str(rank_path),
                        "gmt_path": str(gmt_path),
                        "permutations": permutations,
                        "min_size": min_size,
                        "max_size": max_size,
                        "threads": threads,
                        "seed": seed,
                    }
                )
    collected: dict[str, list[pd.DataFrame]] = defaultdict(list)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_gsea_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            collected[str(task["database"])].append(result)
    return {
        database: pd.concat(collected[database], ignore_index=True).sort_values(
            ["population", "contrast", "FDR_q", "term"], kind="stable"
        )
        for database in FUNCTIONAL_DATABASES
    }


def ora_enrichment(
    selected_genes: Iterable[str],
    tested_genes: Iterable[str],
    gene_sets: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Hypergeometric ORA with the population-specific tested genes as background."""

    background = set(map(str, tested_genes))
    selected = set(map(str, selected_genes)) & background
    if not selected:
        return pd.DataFrame(
            columns=["term", "overlap", "selected_size", "term_size", "background_size", "pvalue", "padj", "overlap_genes"]
        )
    rows: list[dict[str, Any]] = []
    for term, members in gene_sets.items():
        eligible = set(map(str, members)) & background
        overlap = selected & eligible
        if not overlap:
            continue
        pvalue = float(hypergeom.sf(len(overlap) - 1, len(background), len(eligible), len(selected)))
        rows.append(
            {
                "term": term,
                "overlap": len(overlap),
                "selected_size": len(selected),
                "term_size": len(eligible),
                "background_size": len(background),
                "pvalue": pvalue,
                "overlap_genes": ";".join(sorted(overlap)),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        result["padj"] = pd.Series(dtype=float)
        return result
    order = np.argsort(result["pvalue"].to_numpy(), kind="stable")
    ranked = result["pvalue"].to_numpy()[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    padj = np.empty(len(result), dtype=float)
    padj[order] = np.clip(adjusted, 0, 1)
    result["padj"] = padj
    return result.sort_values(["padj", "pvalue", "term"], kind="stable")


def run_all_ora(
    de_root: Path,
    mapping: pd.DataFrame,
    resources: Mapping[str, Path],
    *,
    populations: Sequence[str],
    contrasts: Sequence[str],
    alpha: float,
    effect_cutoff: float,
) -> dict[tuple[str, str], pd.DataFrame]:
    gene_sets = {database: parse_generic_gmt(path) for database, path in resources.items()}
    outputs: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)
    for population in populations:
        table = _read_tsv(de_root / population / "unified_all_genes.tsv.gz")
        for contrast in contrasts:
            sub = table.loc[table["contrast"].astype(str).eq(contrast)].copy()
            ranked, _ = deduplicate_rank_table(sub, mapping)
            tested = ranked["canonical_mouse_symbol"].astype(str)
            for analysis_set, effect_required in [("FDR_only", False), ("FDR_and_abs_LFC_ge_0.5", True)]:
                supported = ranked["padj"].notna() & ranked["padj"].lt(alpha)
                if effect_required:
                    supported &= ranked["log2FoldChange"].abs().ge(effect_cutoff)
                for direction, direction_mask in [
                    ("up", ranked["log2FoldChange"].gt(0)),
                    ("down", ranked["log2FoldChange"].lt(0)),
                ]:
                    selected = ranked.loc[supported & direction_mask, "canonical_mouse_symbol"]
                    for database, sets in gene_sets.items():
                        result = ora_enrichment(selected, tested, sets)
                        if result.empty:
                            continue
                        result.insert(0, "analysis_set", analysis_set)
                        result.insert(0, "direction", direction)
                        result.insert(0, "database", database)
                        result.insert(0, "contrast", contrast)
                        result.insert(0, "population", population)
                        result["selection_rule"] = (
                            f"padj<{alpha}; direction by log2FoldChange"
                            + (f"; abs(log2FoldChange)>={effect_cutoff}" if effect_required else "")
                        )
                        result["background_rule"] = "canonical symbols among genes tested in the same population unified DE model"
                        result["statistical_unit"] = "library"
                        outputs[(database, direction)].append(result)
    return {
        key: pd.concat(parts, ignore_index=True).sort_values(
            ["population", "contrast", "analysis_set", "padj", "term"], kind="stable"
        )
        if parts
        else pd.DataFrame()
        for key, parts in outputs.items()
    }


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            next_value = self.parent[value]
            self.parent[value] = root
            value = next_value
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def cluster_go_redundancy(
    go_results: pd.DataFrame,
    gene_sets: Mapping[str, Sequence[str]],
    *,
    jaccard_threshold: float = 0.25,
    max_terms_per_context: int = 500,
) -> pd.DataFrame:
    """Cluster significant GO terms by gene-set Jaccard within each result context."""

    rows: list[dict[str, Any]] = []
    significant = go_results.loc[go_results["FDR_q"].lt(0.05)].copy()
    significant["direction"] = np.where(significant["NES"].ge(0), "positive", "negative")
    for keys, block in significant.groupby(
        ["population", "contrast", "direction"], observed=True, sort=True
    ):
        block = block.sort_values(["FDR_q", "nominal_p", "term"], kind="stable").head(max_terms_per_context)
        terms = block["term"].astype(str).tolist()
        if not terms:
            continue
        sets = {term: set(map(str, gene_sets.get(term, []))) for term in terms}
        candidate_intersections: dict[tuple[str, str], int] = defaultdict(int)
        inverted: dict[str, list[str]] = defaultdict(list)
        for term, genes in sets.items():
            for gene in genes:
                inverted[gene].append(term)
        for containing_terms in inverted.values():
            ordered = sorted(containing_terms)
            for i, left in enumerate(ordered):
                for right in ordered[i + 1 :]:
                    candidate_intersections[(left, right)] += 1
        uf = _UnionFind(terms)
        for (left, right), intersection in candidate_intersections.items():
            union = len(sets[left]) + len(sets[right]) - intersection
            if union and intersection / union >= jaccard_threshold:
                uf.union(left, right)
        clusters: dict[str, list[str]] = defaultdict(list)
        for term in terms:
            clusters[uf.find(term)].append(term)
        block_index = block.set_index("term")
        for cluster_number, members in enumerate(
            sorted(clusters.values(), key=lambda values: min(values)), start=1
        ):
            representative = (
                block_index.loc[members]
                .assign(abs_NES=lambda frame: frame["NES"].abs())
                .reset_index()
                .sort_values(["FDR_q", "abs_NES", "term"], ascending=[True, False, True], kind="stable")
                .iloc[0]["term"]
            )
            rows.append(
                {
                    "population": keys[0],
                    "contrast": keys[1],
                    "direction": keys[2],
                    "cluster_id": f"{keys[0]}__{keys[1]}__{keys[2]}__GO{cluster_number:03d}",
                    "representative_term": representative,
                    "representative_NES": float(block_index.loc[representative, "NES"]),
                    "representative_FDR": float(block_index.loc[representative, "FDR_q"]),
                    "n_cluster_members": len(members),
                    "cluster_members": ";".join(sorted(members)),
                    "clustering_rule": f"gene-set Jaccard >= {jaccard_threshold}; connected components",
                    "selection_scope": f"FDR<0.05; at most {max_terms_per_context} terms per population/contrast/direction",
                }
            )
    return pd.DataFrame(rows)


def _figure_qa(fig: plt.Figure, paths: Mapping[str, Path]) -> dict[str, Any]:
    from matplotlib import image as mpimg

    png = paths["png"]
    pixels = mpimg.imread(png)
    nonblank = bool(np.nanstd(pixels[..., :3]) > 0.005)
    svg_text = paths["svg"].read_text(encoding="utf-8", errors="ignore")
    return {
        "n_axes": len(fig.axes),
        "png_nonblank": nonblank,
        "svg_editable_text": "<text" in svg_text,
        "pdf_valid": paths["pdf"].read_bytes().startswith(b"%PDF"),
        "all_files_nonempty": all(path.stat().st_size > 1000 for path in paths.values()),
        "qa_pass": bool(
            len(fig.axes) == 1
            and nonblank
            and "<text" in svg_text
            and paths["pdf"].read_bytes().startswith(b"%PDF")
            and all(path.stat().st_size > 1000 for path in paths.values())
        ),
    }


def save_standalone_figure(
    fig: plt.Figure,
    plot_id: str,
    figure_root: Path,
    source_data: pd.DataFrame,
    source_root: Path,
    *,
    scientific_role: str,
    source_file: str,
    dpi: int = 600,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Save one single-axis plot with source data and programmatic QA."""

    if len(fig.axes) != 1:
        raise ValueError(f"{plot_id} must contain exactly one axes, found {len(fig.axes)}")
    figure_root.mkdir(parents=True, exist_ok=True)
    source_root.mkdir(parents=True, exist_ok=True)
    source_path = source_root / f"{plot_id}.tsv"
    source_data.to_csv(source_path, sep="\t", index=False)
    paths = {extension: figure_root / f"{plot_id}.{extension}" for extension in ("svg", "pdf", "png")}
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=dpi, bbox_inches="tight")
    qa = _figure_qa(fig, paths)
    plt.close(fig)
    manifest = {
        "plot_id": plot_id,
        "scientific_role": scientific_role,
        "source_file": source_file,
        "source_data": str(source_path),
        "svg": str(paths["svg"]),
        "pdf": str(paths["pdf"]),
        "png_600dpi": str(paths["png"]),
        "assembled_multipanel": False,
        "statistical_unit": "library" if scientific_role not in {"annotation_marker_support"} else "descriptive_cells",
    }
    return manifest, {"plot_id": plot_id, **qa}


def _plot_stacked_fraction(
    table: pd.DataFrame,
    *,
    category_col: str,
    fraction_col: str,
    title: str,
    color_registry: Mapping[str, str],
) -> plt.Figure:
    matrix = table.pivot(index="library_id", columns=category_col, values=fraction_col).reindex(ALL_LIBRARIES)
    categories = matrix.mean().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    bottom = np.zeros(len(matrix), dtype=float)
    fallback = plt.get_cmap("tab20")(np.linspace(0, 1, max(1, len(categories))))
    for index, category in enumerate(categories):
        values = matrix[category].fillna(0).to_numpy(dtype=float)
        ax.bar(
            np.arange(len(matrix)),
            values,
            bottom=bottom,
            width=0.78,
            color=color_registry.get(category, fallback[index]),
            edgecolor="white",
            linewidth=0.25,
            label=category.replace("_", " "),
        )
        bottom += values
    ax.set_xticks(np.arange(len(matrix)), matrix.index, rotation=45, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), ncol=1, title=None)
    return fig


def _plot_group_library_points(
    table: pd.DataFrame,
    *,
    category_col: str,
    value_col: str,
    title: str,
    xlabel: str,
) -> plt.Figure:
    categories = (
        table.groupby(category_col, observed=True)[value_col].mean().sort_values().index.tolist()
    )
    fig_height = max(2.8, 0.28 * len(categories) + 1.2)
    fig, ax = plt.subplots(figsize=(6.0, fig_height))
    offsets = {"Y": -0.18, "OC": 0.0, "OT": 0.18}
    for group in GROUPS:
        group_table = table.loc[table["group"].astype(str).eq(group)]
        for y_index, category in enumerate(categories):
            values = group_table.loc[
                group_table[category_col].astype(str).eq(category), value_col
            ].to_numpy(dtype=float)
            y = y_index + offsets[group]
            ax.scatter(values, np.full(len(values), y), s=15, color=GROUP_COLORS[group], alpha=0.8, linewidths=0, zorder=3)
            if len(values):
                mean = float(np.mean(values))
                sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ax.errorbar(mean, y, xerr=sd, fmt="D", markersize=3.5, color=GROUP_COLORS[group], capsize=2, linewidth=0.8, zorder=4)
    ax.set_yticks(np.arange(len(categories)), [str(value).replace("_", " ") for value in categories])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=GROUP_COLORS[group], label=group) for group in GROUPS]
    ax.legend(handles=handles, loc="lower right", title="Group")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.5)
    ax.set_axisbelow(True)
    return fig


def _representative_feature_indices(
    matrix: sparse.spmatrix,
    mapping: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose one expressed feature per canonical symbol without abs-stat selection."""

    totals = np.asarray(matrix.sum(axis=0)).ravel()
    work = mapping[["feature_order", "canonical_mouse_symbol", "valid_symbol"]].copy()
    work["total_log_expression"] = totals[work["feature_order"].to_numpy(dtype=int)]
    work = work.loc[work["valid_symbol"].astype(bool)].sort_values(
        ["canonical_mouse_symbol", "total_log_expression", "feature_order"],
        ascending=[True, False, True],
        kind="stable",
    )
    work = work.drop_duplicates("canonical_mouse_symbol", keep="first")
    return (
        work["feature_order"].to_numpy(dtype=int),
        work["canonical_mouse_symbol"].astype(str).to_numpy(),
    )


def _marker_rows_for_labels(
    matrix: sparse.csr_matrix,
    labels: pd.Series,
    symbols: np.ndarray,
    canonical_map: Mapping[str, set[str]],
    *,
    top_n: int,
    parent_labels: pd.Series | None = None,
) -> pd.DataFrame:
    """Rank descriptive markers by detection specificity then mean-log effect."""

    labels = labels.astype(str).reset_index(drop=True)
    parent_values = parent_labels.astype(str).reset_index(drop=True) if parent_labels is not None else None
    rows: list[pd.DataFrame] = []
    if parent_values is None:
        label_pairs = [("", label) for label in sorted(labels.unique())]
    else:
        label_pairs = sorted(
            set(zip(parent_values.astype(str), labels.astype(str), strict=False))
        )
    for parent, label in label_pairs:
        if parent_values is None:
            in_mask = labels.eq(label).to_numpy()
            universe_mask = np.ones(len(labels), dtype=bool)
        else:
            parent_mask = parent_values.eq(parent).to_numpy()
            in_mask = parent_mask & labels.eq(label).to_numpy()
            universe_mask = parent_mask
        out_mask = universe_mask & ~in_mask
        if not in_mask.any() or not out_mask.any():
            continue
        in_matrix = matrix[in_mask]
        out_matrix = matrix[out_mask]
        mean_in = np.asarray(in_matrix.mean(axis=0)).ravel()
        mean_out = np.asarray(out_matrix.mean(axis=0)).ravel()
        pct_in = in_matrix.getnnz(axis=0) / int(in_mask.sum())
        pct_out = out_matrix.getnnz(axis=0) / int(out_mask.sum())
        effect = mean_in - mean_out
        specificity = pct_in - pct_out
        frame = pd.DataFrame(
            {
                "gene": symbols,
                "effect_mean_log_expression": effect,
                "pct_in": pct_in,
                "pct_out": pct_out,
                "specificity_metric": specificity,
            }
        )
        frame = frame.loc[(frame["pct_in"] > 0) & (frame["effect_mean_log_expression"] > 0)].copy()
        frame = frame.sort_values(
            ["specificity_metric", "effect_mean_log_expression", "pct_in", "gene"],
            ascending=[False, False, False, True],
            kind="stable",
        )
        canonical = set(canonical_map.get(label, set()))
        keep = frame.head(top_n).copy()
        if canonical:
            keep = pd.concat([keep, frame.loc[frame["gene"].isin(canonical)]], ignore_index=True)
        keep = keep.drop_duplicates("gene", keep="first").reset_index(drop=True)
        keep["rank"] = np.arange(1, len(keep) + 1)
        keep["canonical_marker_flag"] = keep["gene"].isin(canonical)
        keep["annotation_note"] = np.where(
            keep["canonical_marker_flag"],
            "curated canonical marker retained even if outside data-driven top N",
            "descriptive one-vs-rest ranking; cells are not biological replicates",
        )
        keep.insert(0, "label", label)
        keep.insert(0, "parent_label", parent)
        keep["n_cells_in"] = int(in_mask.sum())
        keep["n_cells_out"] = int(out_mask.sum())
        keep["comparison_scope"] = "all_other_broad_types" if parent_values is None else "other_subtypes_within_parent_broad"
        rows.append(keep)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _canonical_marker_maps(marker_config: Mapping[str, Any]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    broad: dict[str, set[str]] = {}
    major = marker_config.get("major_annotation", {})
    aliases = {"Ovarian_epithelium": "Ovarian_epithelial", "Luteal_candidate": "Luteal"}
    for label, details in major.items():
        broad[aliases.get(label, label)] = set(map(str, details.get("positive", [])))

    subtype: dict[str, set[str]] = {}
    follicular = marker_config.get("follicular_annotation_review", {})
    follicular_aliases = {
        "Granulosa_preantral_like": "preantral",
        "Granulosa_antral_like": "antral",
        "Granulosa_atretic_like": "atretic",
        "Granulosa_cycling": "cycling",
        "Granulosa_low_complexity_candidate": "granulosa",
        "Theca": "theca",
        "Theca_cycling": "theca",
        "Luteal_like": "luteal",
        "Rare_luteal_candidate": "luteal",
    }
    for label, key in follicular_aliases.items():
        subtype[label] = set(map(str, follicular.get(key, {}).get("positive", [])))
    compartments = marker_config.get("compartment_subclustering", {})
    aliases_by_compartment = {
        "stromal": {
            "Stromal_fibroblast_candidate": "fibroblast",
            "ECM_high_candidate": "ecm_fibrotic",
            "Smooth_muscle_candidate": "smooth_muscle",
        },
        "immune": {
            "Macrophage_C1qc_high": "macrophage_c1qc",
            "Macrophage_inflammatory_candidate": "macrophage_inflammatory",
            "Macrophage_lipid_associated_candidate": "macrophage_lipid",
            "Dendritic_candidate": "dendritic",
            "T_cell_candidate": "t_cell",
            "NK_candidate": "nk",
            "B_cell_candidate": "b_cell",
            "Plasma_like_candidate": "plasma",
            "Neutrophil_candidate": "neutrophil",
        },
        "epithelial": {
            "Ciliated_epithelial_candidate": "ciliated",
            "Surface_epithelial_candidate": "surface",
            "Secretory_epithelial_candidate": "secretory",
        },
        "endothelial": {
            "Lymphatic_endothelial_candidate": "lymphatic",
            "Vascular_endothelial_candidate": "vascular",
        },
    }
    for compartment, aliases_for_section in aliases_by_compartment.items():
        section = compartments.get(compartment, {})
        for label, key in aliases_for_section.items():
            subtype[label] = set(map(str, section.get(key, [])))
    return broad, subtype


def _dotplot_source(
    matrix: sparse.csr_matrix,
    labels: pd.Series,
    symbols: np.ndarray,
    genes: Sequence[str],
) -> pd.DataFrame:
    symbol_to_index = {gene: index for index, gene in enumerate(symbols)}
    present = [gene for gene in genes if gene in symbol_to_index]
    rows: list[dict[str, Any]] = []
    label_values = labels.astype(str).reset_index(drop=True)
    for label in sorted(label_values.unique()):
        mask = label_values.eq(label).to_numpy()
        sub = matrix[mask][:, [symbol_to_index[gene] for gene in present]]
        means = np.asarray(sub.mean(axis=0)).ravel()
        fractions = sub.getnnz(axis=0) / int(mask.sum())
        for gene, mean, fraction in zip(present, means, fractions, strict=True):
            rows.append(
                {
                    "label": label,
                    "gene": gene,
                    "mean_log_expression": float(mean),
                    "fraction_expressed": float(fraction),
                    "n_cells": int(mask.sum()),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["mean_scaled_within_gene"] = result.groupby("gene", observed=True)["mean_log_expression"].transform(
        lambda values: (values - values.min()) / (values.max() - values.min()) if values.max() > values.min() else 0.0
    )
    return result


def _plot_marker_dotplot(source: pd.DataFrame, title: str) -> plt.Figure:
    labels = source["label"].drop_duplicates().tolist()
    genes = source["gene"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(max(5.2, 0.22 * len(genes) + 1.8), max(2.5, 0.28 * len(labels) + 1.2)))
    label_pos = {label: index for index, label in enumerate(labels)}
    gene_pos = {gene: index for index, gene in enumerate(genes)}
    x = source["gene"].map(gene_pos).to_numpy()
    y = source["label"].map(label_pos).to_numpy()
    sizes = 7 + 45 * source["fraction_expressed"].to_numpy(dtype=float)
    ax.scatter(
        x,
        y,
        s=sizes,
        c=source["mean_scaled_within_gene"],
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidths=0,
    )
    ax.set_xticks(np.arange(len(genes)), genes, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(labels)), [label.replace("_", " ") for label in labels])
    ax.set_xlim(-0.7, len(genes) - 0.3)
    ax.set_ylim(len(labels) - 0.3, -0.7)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Marker gene (color: scaled mean; size: fraction expressed)")
    ax.grid(color="#ECECEC", linewidth=0.35)
    ax.set_axisbelow(True)
    for fraction in (0.25, 0.50, 0.75):
        ax.scatter([], [], s=7 + 45 * fraction, color="#666666", label=f"{int(100*fraction)}%")
    ax.legend(title="Expressed", loc="upper left", bbox_to_anchor=(1.01, 1), labelspacing=0.8)
    return fig


def _prepare_de_tables(
    result_root: Path,
    mapping: pd.DataFrame,
    *,
    populations: Sequence[str],
    contrasts: Sequence[str],
    alpha: float,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    top_rows: list[pd.DataFrame] = []
    reversal_rows: list[pd.DataFrame] = []
    all_tables: dict[tuple[str, str], pd.DataFrame] = {}
    for population in populations:
        de = _read_tsv(result_root / "de_stage1_5" / population / "unified_all_genes.tsv.gz")
        for contrast in contrasts:
            selected, _ = deduplicate_rank_table(
                de.loc[de["contrast"].astype(str).eq(contrast)].copy(), mapping
            )
            selected["population"] = population
            selected["contrast"] = contrast
            selected["fdr_supported"] = selected["padj"].notna() & selected["padj"].lt(alpha)
            selected["effect_size_supported"] = selected["log2FoldChange"].abs().ge(0.5)
            all_tables[(population, contrast)] = selected
            top = selected.loc[selected["fdr_supported"]].copy()
            top["abs_log2FoldChange"] = top["log2FoldChange"].abs()
            top = top.sort_values(
                ["abs_log2FoldChange", "padj", "canonical_mouse_symbol"],
                ascending=[False, True, True],
                kind="stable",
            ).head(top_n)
            top["selection_rule"] = f"padj<{alpha}; then absolute raw log2FoldChange descending; top {top_n}"
            top["rank"] = np.arange(1, len(top) + 1)
            top_rows.append(top)
        reversal = _read_tsv(
            result_root / "de_stage1_6" / population / "observed_evidence_levels.tsv.gz"
        )
        boolean = reversal["Level_2_DE_supported_candidate"]
        if boolean.dtype != bool:
            boolean = boolean.astype(str).str.lower().isin({"true", "1", "yes"})
        keep = reversal.loc[boolean].copy()
        keep["selection_rule"] = "existing Stage 1.6 Level_2_DE_supported_candidate; no new score"
        reversal_rows.append(keep)
    return (
        pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame(),
        pd.concat(reversal_rows, ignore_index=True) if reversal_rows else pd.DataFrame(),
        all_tables,
    )


def _plot_volcano(table: pd.DataFrame, population: str, contrast: str) -> tuple[plt.Figure, pd.DataFrame]:
    data = table.copy()
    data["neg_log10_pvalue"] = np.where(
        data["pvalue"].notna() & data["pvalue"].gt(0),
        -np.log10(data["pvalue"].clip(lower=np.nextafter(0, 1))),
        np.nan,
    )
    data["support_class"] = "Not supported"
    data.loc[data["effect_size_supported"], "support_class"] = "|LFC| >= 0.5 only"
    data.loc[data["fdr_supported"], "support_class"] = "FDR < 0.05 only"
    data.loc[data["fdr_supported"] & data["effect_size_supported"], "support_class"] = "FDR < 0.05 and |LFC| >= 0.5"
    colors = {
        "Not supported": "#BDBDBD",
        "|LFC| >= 0.5 only": "#56B4E9",
        "FDR < 0.05 only": "#E69F00",
        "FDR < 0.05 and |LFC| >= 0.5": "#D55E00",
    }
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for support_class in colors:
        sub = data.loc[data["support_class"].eq(support_class)]
        ax.scatter(
            sub["log2FoldChange"],
            sub["neg_log10_pvalue"],
            s=5 if support_class == "Not supported" else 8,
            color=colors[support_class],
            alpha=0.35 if support_class == "Not supported" else 0.75,
            linewidths=0,
            label=support_class,
            rasterized=False,
        )
    ax.axvline(0, color="#444444", linewidth=0.6)
    ax.set_xlabel("Unified-model log2 fold change")
    ax.set_ylabel("-log10 nominal p-value")
    ax.set_title(f"{population.replace('_', ' ')}: {PRIMARY_CONTRAST_LABELS[contrast]}", loc="left", fontweight="bold")
    ax.legend(loc="upper right", fontsize=5.4)
    supported = data.loc[data["fdr_supported"]].copy()
    supported["abs_lfc"] = supported["log2FoldChange"].abs()
    labels = supported.sort_values(["abs_lfc", "padj"], ascending=[False, True]).head(6)
    for row in labels.itertuples(index=False):
        ax.annotate(
            str(row.canonical_mouse_symbol),
            (float(row.log2FoldChange), float(row.neg_log10_pvalue)),
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=5.2,
        )
    return fig, data[
        [
            "population",
            "contrast",
            "canonical_mouse_symbol",
            "log2FoldChange",
            "pvalue",
            "padj",
            "neg_log10_pvalue",
            "fdr_supported",
            "effect_size_supported",
            "support_class",
        ]
    ]


def _read_broad_counts(path: Path) -> pd.DataFrame:
    table = _read_tsv(path)
    population_col = "population" if "population" in table.columns else "cell_type_broad_v2"
    library_col = "library" if "library" in table.columns else "library_id"
    metadata = [column for column in [population_col, library_col, "group"] if column in table.columns]
    counts = table.set_index([population_col, library_col]).drop(columns=[column for column in metadata if column not in {population_col, library_col}], errors="ignore")
    counts.index.names = ["population", "library_id"]
    return counts.apply(pd.to_numeric, errors="raise")


def _plot_pseudobulk_heatmap(
    counts: pd.DataFrame,
    selected: pd.DataFrame,
    population: str,
    contrast: str,
) -> tuple[plt.Figure, pd.DataFrame]:
    feature_ids = selected["feature_id"].astype(str).tolist()
    labels = selected.set_index("feature_id")["canonical_mouse_symbol"].astype(str).to_dict()
    block = counts.xs(population, level="population").reindex(ALL_LIBRARIES)
    feature_ids = [feature for feature in feature_ids if feature in block.columns][:20]
    if not feature_ids:
        raise RuntimeError(f"No selected heatmap genes found for {population}/{contrast}")
    values = block[feature_ids].to_numpy(dtype=float)
    library_sums = block.sum(axis=1).to_numpy(dtype=float)
    log_cpm = np.log1p(values / np.maximum(library_sums[:, None], 1) * 1e6)
    means = log_cpm.mean(axis=0, keepdims=True)
    sds = log_cpm.std(axis=0, ddof=1, keepdims=True)
    z = (log_cpm - means) / np.where(sds > 0, sds, 1)
    fig, ax = plt.subplots(figsize=(5.0, max(3.2, 0.24 * len(feature_ids) + 1.4)))
    ax.imshow(z.T, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(ALL_LIBRARIES)), ALL_LIBRARIES, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(feature_ids)), [labels.get(feature, feature) for feature in feature_ids])
    ax.set_xlabel("Library (row z-score of log CPM)")
    ax.set_title(f"{population.replace('_', ' ')} top DE genes: {PRIMARY_CONTRAST_LABELS[contrast]}", loc="left", fontweight="bold")
    source = pd.DataFrame(z, index=ALL_LIBRARIES, columns=[labels.get(feature, feature) for feature in feature_ids])
    source.index.name = "library_id"
    source = source.reset_index().melt(id_vars="library_id", var_name="gene", value_name="z_log_CPM")
    source["population"] = population
    source["contrast"] = contrast
    source["group"] = source["library_id"].str.split("_", n=1).str[0]
    return fig, source


def _plot_gsea_terms(table: pd.DataFrame, database: str, population: str, contrast: str) -> tuple[plt.Figure, pd.DataFrame]:
    sub = table.loc[
        table["population"].astype(str).eq(population)
        & table["contrast"].astype(str).eq(contrast)
    ].copy()
    supported = sub.loc[sub["FDR_q"].lt(0.05)].copy()
    if supported.empty:
        supported = sub.sort_values(["FDR_q", "nominal_p", "term"], kind="stable").head(15)
        selection = "top_15_by_FDR_no_FDR_supported_terms"
    else:
        supported["abs_NES"] = supported["NES"].abs()
        supported = supported.sort_values(
            ["FDR_q", "abs_NES", "term"], ascending=[True, False, True], kind="stable"
        ).head(15)
        selection = "FDR<0.05_then_top_15_by_FDR_and_abs_NES"
    supported = supported.sort_values("NES")
    labels = [re.sub(r"^(GOBP_|REACTOME_|KEGG_MMU_\d+_)", "", term).replace("_", " ").title() for term in supported["term"].astype(str)]
    fig, ax = plt.subplots(figsize=(5.6, max(3.2, 0.26 * len(supported) + 1.3)))
    colors = np.where(supported["NES"].ge(0), "#D55E00", "#0072B2")
    sizes = 16 + 18 * np.clip(-np.log10(supported["FDR_q"].clip(lower=1e-12)), 0, 5)
    ax.scatter(supported["NES"], np.arange(len(supported)), s=sizes, c=colors, alpha=0.85, linewidths=0)
    ax.axvline(0, color="#555555", linewidth=0.6)
    ax.set_yticks(np.arange(len(supported)), labels)
    ax.set_xlabel("Normalized enrichment score")
    ax.set_title(f"{database}: {population.replace('_', ' ')} {contrast}", loc="left", fontweight="bold")
    supported["figure_selection_rule"] = selection
    return fig, supported


def _cell_cycle_outputs(
    adata: Any,
    s_genes: Sequence[str],
    g2m_genes: Sequence[str],
    subtype_abundance: pd.DataFrame,
    stage3_hallmark: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import scanpy as sc

    present_s = [gene for gene in s_genes if gene in adata.var_names]
    present_g2m = [gene for gene in g2m_genes if gene in adata.var_names]
    if len(present_s) < 20 or len(present_g2m) < 20:
        raise RuntimeError(
            f"Too few mouse cell-cycle genes found: S={len(present_s)}, G2M={len(present_g2m)}"
        )
    sc.tl.score_genes_cell_cycle(
        adata,
        s_genes=present_s,
        g2m_genes=present_g2m,
        use_raw=False,
        copy=False,
    )
    assigned = assign_cell_cycle_phase(adata.obs["S_score"], adata.obs["G2M_score"])
    if not np.array_equal(adata.obs["phase"].astype(str).to_numpy(), assigned):
        raise AssertionError("Independent phase assignment does not match Scanpy")
    obs = adata.obs[
        ["library_id", "group", "cell_type_broad_v2", "cell_type_subtype_v2", "S_score", "G2M_score", "phase"]
    ].copy()
    for column in ["library_id", "group", "cell_type_broad_v2", "cell_type_subtype_v2", "phase"]:
        obs[column] = obs[column].astype(str)
    targets = obs.loc[
        obs["cell_type_broad_v2"].eq("Granulosa")
        | obs["cell_type_subtype_v2"].eq("Theca_cycling")
    ].copy()
    summary_rows: list[dict[str, Any]] = []
    scopes = [("Broad_Granulosa", targets["cell_type_broad_v2"].eq("Granulosa"))]
    scopes.extend(
        (subtype, targets["cell_type_subtype_v2"].eq(subtype))
        for subtype in sorted(targets["cell_type_subtype_v2"].unique())
    )
    for scope, mask in scopes:
        block = targets.loc[mask]
        if block.empty:
            continue
        summary_rows.append(
            {
                "scope": scope,
                "n_cells": len(block),
                "S_score_mean": float(block["S_score"].mean()),
                "S_score_median": float(block["S_score"].median()),
                "G2M_score_mean": float(block["G2M_score"].mean()),
                "G2M_score_median": float(block["G2M_score"].median()),
                "n_S_genes_used": len(present_s),
                "n_G2M_genes_used": len(present_g2m),
                "scoring_method": "scanpy.tl.score_genes_cell_cycle on existing log-normalized X; descriptive only",
            }
        )
    score_summary = pd.DataFrame(summary_rows)

    broad = targets.loc[targets["cell_type_broad_v2"].eq("Granulosa")]
    phase_by_library = (
        broad.groupby(["library_id", "group", "phase"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    grid = pd.MultiIndex.from_product([ALL_LIBRARIES, PHASES], names=["library_id", "phase"])
    phase_by_library = phase_by_library.set_index(["library_id", "phase"]).reindex(grid, fill_value=0).reset_index()
    phase_by_library["group"] = phase_by_library["library_id"].str.split("_", n=1).str[0]
    phase_by_library["total_cells"] = phase_by_library.groupby("library_id", observed=True)["n_cells"].transform("sum")
    phase_by_library["phase_fraction"] = phase_by_library["n_cells"] / phase_by_library["total_cells"]

    phase_by_subtype = (
        targets.groupby(["cell_type_broad_v2", "cell_type_subtype_v2", "library_id", "group", "phase"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    phase_by_subtype["subtype_total_cells"] = phase_by_subtype.groupby(
        ["cell_type_subtype_v2", "library_id"], observed=True
    )["n_cells"].transform("sum")
    phase_by_subtype["phase_fraction"] = phase_by_subtype["n_cells"] / phase_by_subtype["subtype_total_cells"]

    interpretation_rows: list[dict[str, Any]] = []
    cycling = subtype_abundance.loc[subtype_abundance["subtype"].eq("Granulosa_cycling")]
    cycling_means = cycling.groupby("group", observed=True)["fraction_of_parent"].mean()
    interpretation_rows.append(
        {
            "evidence_type": "cycling_subtype_abundance",
            "broad_population": "Granulosa",
            "subtype": "Granulosa_cycling",
            "metric": "fraction_of_parent",
            "Y_value": float(cycling_means.get("Y", np.nan)),
            "OC_value": float(cycling_means.get("OC", np.nan)),
            "OT_value": float(cycling_means.get("OT", np.nan)),
            "aging_delta": float(cycling_means.get("OC", np.nan) - cycling_means.get("Y", np.nan)),
            "treatment_delta": float(cycling_means.get("OT", np.nan) - cycling_means.get("OC", np.nan)),
            "contrast": "library_level_descriptive",
            "NES": np.nan,
            "FDR_q": np.nan,
            "interpretation": "Cycling-state abundance is descriptive and does not establish a condition effect with n=3/group.",
        }
    )
    cycle_terms = {"HALLMARK_G2M_CHECKPOINT", "HALLMARK_MITOTIC_SPINDLE", "HALLMARK_E2F_TARGETS"}
    hall = stage3_hallmark.loc[
        stage3_hallmark["broad_population"].astype(str).eq("Granulosa")
        & stage3_hallmark["pathway"].astype(str).isin(cycle_terms)
        & stage3_hallmark["contrast"].astype(str).isin(["OC_vs_Y", "OT_vs_OC"])
    ].copy()
    for row in hall.itertuples(index=False):
        interpretation_rows.append(
            {
                "evidence_type": "existing_subtype_Hallmark_GSEA",
                "broad_population": "Granulosa",
                "subtype": str(row.subtype),
                "metric": str(row.pathway),
                "Y_value": np.nan,
                "OC_value": np.nan,
                "OT_value": np.nan,
                "aging_delta": np.nan,
                "treatment_delta": np.nan,
                "contrast": str(row.contrast),
                "NES": float(row.NES),
                "FDR_q": float(row.FDR_q),
                "interpretation": "Existing Stage 3 subtype GSEA; enrichment is not a cell-cycle phase proportion.",
            }
        )
    noncycling_supported = hall.loc[
        ~hall["subtype"].astype(str).eq("Granulosa_cycling") & hall["FDR_q"].lt(0.05)
    ]
    interpretation_rows.append(
        {
            "evidence_type": "integrated_interpretation",
            "broad_population": "Granulosa",
            "subtype": "all_reviewed",
            "metric": "cycling_abundance_vs_within_state_expression",
            "Y_value": np.nan,
            "OC_value": np.nan,
            "OT_value": np.nan,
            "aging_delta": np.nan,
            "treatment_delta": np.nan,
            "contrast": "Stage3_plus_Stage4_plus_Stage14",
            "NES": np.nan,
            "FDR_q": np.nan,
            "interpretation": (
                "Cell-cycle Hallmark support occurs in non-cycling Granulosa subtypes, so the broad G2M/Mitotic signal is not explained by cycling-cell abundance alone."
                if len(noncycling_supported)
                else "The available evidence does not clearly separate cycling-state abundance from within-state cell-cycle expression."
            ),
        }
    )
    return score_summary, phase_by_library, phase_by_subtype, pd.DataFrame(interpretation_rows)


def _plot_phase_stacked(phase_table: pd.DataFrame) -> plt.Figure:
    matrix = phase_table.pivot(index="library_id", columns="phase", values="phase_fraction").reindex(
        index=ALL_LIBRARIES, columns=PHASES, fill_value=0
    )
    colors = {"G1": "#999999", "S": "#E69F00", "G2M": "#CC79A7"}
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    bottom = np.zeros(len(matrix))
    for phase in PHASES:
        values = matrix[phase].to_numpy(dtype=float)
        ax.bar(np.arange(len(matrix)), values, bottom=bottom, color=colors[phase], width=0.78, label=phase)
        bottom += values
    ax.set_xticks(np.arange(len(matrix)), matrix.index, rotation=45, ha="right")
    ax.set_ylabel("Granulosa phase fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Granulosa cell-cycle phase by library", loc="left", fontweight="bold")
    ax.legend(loc="upper right", ncol=3)
    return fig


def _hallmark_alignment(
    functional: pd.DataFrame,
    functional_sets: Mapping[str, Sequence[str]],
    hallmark_results: pd.DataFrame,
    hallmark_sets: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Compare significant functional terms to overlapping significant Hallmarks."""

    rows: list[dict[str, Any]] = []
    hall = hallmark_results.loc[hallmark_results["FDR_q"].lt(0.05)].copy()
    for row in functional.loc[functional["FDR_q"].lt(0.05)].itertuples(index=False):
        candidates = hall.loc[
            hall["population"].astype(str).eq(str(row.population))
            & hall["contrast"].astype(str).eq(str(row.contrast))
        ]
        source_members = set(functional_sets.get(str(row.term), []))
        best_name = ""
        best_jaccard = 0.0
        best_nes = np.nan
        for candidate in candidates.itertuples(index=False):
            members = set(hallmark_sets.get(str(candidate.pathway), []))
            union = source_members | members
            score = len(source_members & members) / len(union) if union else 0.0
            if score > best_jaccard:
                best_name = str(candidate.pathway)
                best_jaccard = score
                best_nes = float(candidate.NES)
        if best_jaccard >= 0.10:
            status = "aligned_same_direction" if np.sign(float(row.NES)) == np.sign(best_nes) else "discordant_direction"
        else:
            status = "no_significant_overlapping_hallmark"
        rows.append(
            {
                "population": row.population,
                "contrast": row.contrast,
                "database": row.database,
                "term": row.term,
                "NES": row.NES,
                "FDR_q": row.FDR_q,
                "best_overlapping_significant_hallmark": best_name,
                "gene_set_jaccard": best_jaccard,
                "hallmark_NES": best_nes,
                "hallmark_alignment": status,
                "alignment_rule": "maximum gene-set Jaccard; comparable when Jaccard>=0.10; direction from NES sign",
            }
        )
    return pd.DataFrame(rows)


def _functional_summary(
    results: Mapping[str, pd.DataFrame],
    resource_paths: Mapping[str, Path],
    result_root: Path,
) -> pd.DataFrame:
    hallmark = _read_tsv(result_root / "pathway_stage2" / "hallmark_all_results.tsv")
    hallmark_sets = parse_gmt(result_root.parent / "resources" / "gene_sets" / "mh.all.v2026.1.Mm.symbols.gmt")
    rows: list[pd.DataFrame] = []
    for database, table in results.items():
        aligned = _hallmark_alignment(
            table,
            parse_generic_gmt(resource_paths[database]),
            hallmark,
            hallmark_sets,
        )
        if aligned.empty:
            continue
        aligned["primary_or_secondary"] = aligned["population"].map(POPULATION_TIERS)
        aligned["is_primary_population"] = aligned["population"].isin(["Granulosa", "Stromal_fibroblast"])
        rows.append(aligned)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if result.empty:
        return result
    result["abs_NES"] = result["NES"].abs()
    result["within_context_rank"] = result.groupby(
        ["population", "contrast", "database"], observed=True
    )["FDR_q"].rank(method="first")
    return result.sort_values(
        ["is_primary_population", "population", "contrast", "database", "FDR_q", "abs_NES"],
        ascending=[False, True, True, True, True, False],
        kind="stable",
    )


def _not_recommended_text() -> str:
    return """# 本项目当前不推荐补做的传统分析

## RNA velocity：NOT_APPLICABLE

当前输入没有经过审计的 spliced/unspliced 矩阵。仅凭 filtered gene-count matrix 无法估计可靠 RNA velocity，因此不应为了流程完整而补做。

## inferCNV / CopyKAT：NOT_NEEDED

本项目是非肿瘤小鼠卵巢研究，不存在恶性细胞识别问题。CNV 推断不会回答当前衰老/MRJP1问题。

## Ambient RNA 重新校正：NOT_APPLICABLE

项目只有 filtered count matrix，没有完整 raw droplet input。现有 annotation audit 也没有发现足以推翻全流程的严重环境 RNA 污染，因此不在项目末期修改 count matrix。

## 重新 batch correction：NOT_NEEDED

Harmony 已完成并经过批次审计。不再为了增加方法数量重复运行 Seurat integration、scVI 或 BBKNN。

## trajectory / pseudotime：NOT_REQUIRED

该数据来自横断面的成年卵巢。Granulosa 状态之间可以存在表达几何关系，但当前数据不能证明真实 lineage 或时间方向。本阶段不运行 trajectory；未来若问题改变，可将 PAGA/diffusion map 限定为 state geometry。

## 统计边界

所有条件比较的正式重复单位是 library，n=3/group。单细胞数量不能替代生物学重复，也不能用于扩大 differential abundance 或 condition-level DE 的显著性。
"""


def _supplement_plan() -> pd.DataFrame:
    rows = [
        ("Supplementary Figure S1", "QC", "Reuse", "Existing QC and QC-sensitivity figures"),
        ("Supplementary Figure S2", "Batch and integration audit", "Reuse", "Existing unintegrated/Harmony diagnostics"),
        ("Supplementary Figure S3", "Broad annotation markers", "Stage14", "Broad marker dotplot and catalogue"),
        ("Supplementary Figure S4", "Subtype annotation markers", "Stage14", "Granulosa, Stromal and Immune marker dotplots/catalogues"),
        ("Supplementary Figure S5", "Broad composition", "Stage14", "Library stacked bars and library-level points"),
        ("Supplementary Figure S6", "Subtype composition", "Stage14", "Parent-denominator Granulosa/Stromal/Immune figures; Stage4 CLR/Aitchison cited"),
        ("Supplementary Figure S7", "Cell-cycle audit", "Stage14", "Granulosa phase fractions and Stage3 subtype GSEA interpretation"),
        ("Supplementary Figure S8", "Standard pseudobulk DE", "Stage14", "Volcano plots and nine-library heatmaps"),
        ("Supplementary Figure S9", "GO BP GSEA and redundancy reduction", "Stage14", "Full table plus primary-population standalone plots"),
        ("Supplementary Figure S10", "Reactome GSEA", "Stage14", "Full table plus primary-population standalone plots"),
        ("Supplementary Figure S11", "KEGG mouse GSEA", "Stage14", "Full table plus primary-population standalone plots"),
        ("Supplementary Table S1", "Broad and subtype marker catalogue", "Stage14", "Effect, detection, specificity and curated-marker flags"),
        ("Supplementary Table S2", "Full conventional functional enrichment", "Stage14", "Ranked GSEA and directional ORA with provenance"),
        ("Supplementary Table S3", "Top DE and reversal genes", "Stage14", "Existing unified model and Stage1.6 definitions"),
        ("Supplementary Table S4", "Sensitivity and negative results", "Reuse", "QC, permutation and Stage13 negative-result ledgers"),
    ]
    return pd.DataFrame(rows, columns=["supplement_item", "content", "source", "notes"])


def _format_change_rows(table: pd.DataFrame, parent: str, n: int = 4) -> str:
    subset = table.loc[table["parent_broad"].astype(str).eq(parent)].copy()
    if subset.empty:
        return "No eligible rows."
    subset["max_absolute_change"] = subset[
        ["aging_delta_OC_minus_Y", "treatment_delta_OT_minus_OC"]
    ].abs().max(axis=1)
    top = subset.sort_values("max_absolute_change", ascending=False).head(n)
    return "; ".join(
        f"{row.component}: OC-Y={row.aging_delta_OC_minus_Y:+.3f}, OT-OC={row.treatment_delta_OT_minus_OC:+.3f}"
        for row in top.itertuples(index=False)
    )


def _format_top_terms(summary: pd.DataFrame, database: str, population: str, contrast: str, n: int = 3) -> str:
    sub = summary.loc[
        summary["database"].astype(str).eq(database)
        & summary["population"].astype(str).eq(population)
        & summary["contrast"].astype(str).eq(contrast)
    ].sort_values(["FDR_q", "abs_NES"], ascending=[True, False]).head(n)
    if sub.empty:
        return "No FDR-supported term."
    return "; ".join(
        f"{str(row.term).replace('_', ' ')} (NES={row.NES:.2f}, FDR={row.FDR_q:.3g}, {row.hallmark_alignment})"
        for row in sub.itertuples(index=False)
    )


def _write_reports(
    output_root: Path,
    audit: pd.DataFrame,
    abundance_summary: pd.DataFrame,
    functional_summary: pd.DataFrame,
    cell_cycle_interpretation: pd.DataFrame,
    broad_markers: pd.DataFrame,
    subtype_markers: pd.DataFrame,
    figure_manifest: pd.DataFrame,
) -> None:
    broad_change = _format_change_rows(abundance_summary, "whole_ovary")
    granulosa_change = _format_change_rows(abundance_summary, "Granulosa")
    stromal_change = _format_change_rows(abundance_summary, "Stromal_fibroblast")
    immune_change = _format_change_rows(abundance_summary, "Immune")
    cell_cycle_final = cell_cycle_interpretation.loc[
        cell_cycle_interpretation["evidence_type"].eq("integrated_interpretation"), "interpretation"
    ].iloc[0]
    discordant = int(functional_summary["hallmark_alignment"].eq("discordant_direction").sum())
    supported_functional = int(len(functional_summary))
    report = f"""# Stage 14 常规 scRNA-seq 完整性报告

## 1 为什么进行 Stage 14

Stage 1-13 已完成 QC、整合、注释、library-level pseudobulk、exact permutation、Hallmark 及高级机制候选分析。本阶段只补传统论文常见但尚未统一整理的结果，并冻结常规计算部分。

## 2 常规 scRNA-seq 流程完整性审计

审计覆盖 17 个模块。Stage 14 前 DONE={int(audit['status_before_stage14'].eq('DONE').sum())}、PARTIAL={int(audit['status_before_stage14'].eq('PARTIAL').sum())}、MISSING={int(audit['status_before_stage14'].eq('MISSING').sum())}、NOT_APPLICABLE={int(audit['status_before_stage14'].eq('NOT_APPLICABLE').sum())}。补充后所有适用模块均为 DONE。

## 3 已经完成的核心模块

QC、Harmony、UMAP/Leiden、broad/local annotation、统一9-library DE、exact permutation、Hallmark GSEA、subtype localization、composition/intrinsic decomposition及Stage 5-13结果均直接复用，没有重算。

## 4 Cell abundance

每个library单独计算细胞数和比例，正式展示保留Y/OC/OT各3个library点，不把105,763个细胞当成n。最大描述性变化为：{broad_change}。这些是捕获细胞比例，不等同于组织内绝对细胞数量。

## 5 Granulosa subtype abundance

分母为同一library内Granulosa总数，同时保留whole-ovary fraction。最大描述性变化为：{granulosa_change}。Stage4 CLR/Aitchison继续作为正式composition证据。

## 6 Stromal subtype abundance

分母为同一library内Stromal总数。最大描述性变化为：{stromal_change}。Immune补充结果为：{immune_change}。

## 7 GO Biological Process

使用Mouse MSigDB {MSIGDB_RELEASE} GO BP、统一模型Wald statistic、全部finite tested genes、10,000 permutations。Granulosa aging：{_format_top_terms(functional_summary, 'GO_BP', 'Granulosa', 'OC_vs_Y')}。Stromal aging：{_format_top_terms(functional_summary, 'GO_BP', 'Stromal_fibroblast', 'OC_vs_Y')}。GO显著term另按gene-set Jaccard聚类，避免把高度重叠term当成独立机制。

## 8 Reactome

Granulosa treatment：{_format_top_terms(functional_summary, 'Reactome', 'Granulosa', 'OT_vs_OC')}。Stromal treatment：{_format_top_terms(functional_summary, 'Reactome', 'Stromal_fibroblast', 'OT_vs_OC')}。

## 9 KEGG

KEGG gene sets来自版本明确的Enrichr `KEGG_2019_Mouse`集合；大写条目通过项目内经Cell Ranger/Ensembl审计的mouse canonical symbol表做case-insensitive exact mapping，没有转成人类同源基因。Granulosa treatment：{_format_top_terms(functional_summary, 'KEGG', 'Granulosa', 'OT_vs_OC')}。Stromal treatment：{_format_top_terms(functional_summary, 'KEGG', 'Stromal_fibroblast', 'OT_vs_OC')}。

## 10 GSEA 与 ORA 区别

GSEA使用所有tested genes的Wald排序，是主补充分析。ORA只使用现有DE定义的up/down基因，background严格限制为对应population进入统一DE模型的tested genes；另给出abs(LFC)>=0.5 sensitivity。ORA是secondary，不取代Hallmark或GSEA。

## 11 Cell-cycle audit

使用config中经过项目审核的mouse S/G2M基因，在既有log-normalized X上做Scanpy phase scoring，不regress out。结论：{cell_cycle_final}

## 12 Marker catalogue

Broad catalogue {len(broad_markers):,} 行，subtype catalogue {len(subtype_markers):,} 行。marker为cell identity的描述性one-vs-rest排名，重点是mean-log effect、pct expression、specificity及canonical marker保留；cell-level marker统计不作为condition-level biological replication。

## 13 Standard DE visualization

Volcano全部读取既有统一9-library模型；NA padj没有替换为极小值。Heatmap使用library-level raw pseudobulk counts转换log-CPM并逐基因z-score，明确显示9个library。

## 14 哪些传统分析没有做以及原因

RNA velocity缺少spliced/unspliced，inferCNV不适用于非肿瘤问题，ambient RNA缺少raw droplets，Harmony无需重复，trajectory对横断面成年卵巢不是必需。详见`NOT_RECOMMENDED_CONVENTIONAL_ANALYSES.md`。

## 15 常规分析目前是否完整

适用于本项目的传统模块已经完整。新增结果用于supplementary和审稿完整性，不改变Stage13 Figure 1-7结构。

## 16 主要限制

每组只有3个library；捕获比例受解离和捕获偏倚影响；marker为描述性cell-level ranking；GSEA/ORA表示转录富集而非生化激活；功能数据库之间允许出现不一致。当前共记录{supported_functional}条FDR-supported功能term与Hallmark的重叠审计，其中方向不一致{discordant}条，均保留。

## 17 是否还有必须补跑的基础计算

没有。常规scRNA-seq计算分析可在Stage14后冻结。未来工作应优先补matched phenotype和独立实验验证，而不是继续增加计算方法。
"""
    (output_root / "STAGE14_CONVENTIONAL_SCRNA_REPORT_CN.md").write_text(report, encoding="utf-8")

    explained = f"""# Stage 14 给用户的解释

## 1 细胞比例分析和DE有什么不同？

比例分析问“每个library捕获到的各类细胞占多少”；DE问“同一种细胞内哪些基因表达不同”。两者可能同时发生，也可能只有一种。这里所有组间判断都以library为重复，而不是把每个cell当成独立小鼠。

## 2 为什么已有Hallmark还补GO/KEGG/Reactome？

Hallmark集合小、去冗余好，适合文章主线；GO BP更细，Reactome强调反应和过程层级，KEGG强调经典通路图。它们用于补充解释，不会推翻Hallmark主线。

## 3 GO/KEGG/Reactome各回答什么？

GO BP描述生物过程；Reactome细化分子反应链；KEGG提供mouse signaling/metabolism pathway框架。任何富集都不等于通路已经在蛋白或功能层面被激活。

## 4 为什么GSEA和ORA可能不同？

GSEA看整个排序是否整体偏移；ORA只看超过DE阈值的基因是否集中。边缘但一致的变化可被GSEA发现，却可能进不了ORA。ORA还更受阈值和基因数影响。

## 5 为什么单独检查Granulosa cell cycle？

此前Granulosa出现G2M、Mitotic spindle和E2F信号。现在区分了cycling细胞比例和非cycling亚型内部的表达富集。当前综合结论是：{cell_cycle_final}

## 6 Marker gene和condition DEG有什么区别？

Marker用于区分细胞身份，例如Granulosa与Stromal；condition DEG是在同一细胞群内比较Y、OC、OT。marker的cell-level排序不能替代library-level condition DE。

## 7 为什么RNA velocity不是每篇文章都必须做？

velocity要求可靠spliced/unspliced counts，并依赖动力学假设。本项目没有该输入，而且研究问题不是胚胎发育或连续谱系，因此不做反而更严谨。

## 8 现在基础流程还有明显缺口吗？

没有必须补跑的基础计算。Stage14新增了传统composition展示、GO/Reactome/KEGG、cell-cycle audit、marker catalogue、volcano和九文库heatmap，共{len(figure_manifest)}张独立图；主分析仍以Stage1-13结论为准。
"""
    (output_root / "STAGE14_EXPLAINED_FOR_USER_CN.md").write_text(explained, encoding="utf-8")


def _relative_paths(table: pd.DataFrame, root: Path, columns: Sequence[str]) -> pd.DataFrame:
    result = table.copy()
    for column in columns:
        result[column] = result[column].map(lambda value: str(Path(value).resolve().relative_to(root.resolve())))
    return result


def run_stage14(config: Mapping[str, Any], *, allow_low_memory: bool = False) -> None:
    paths = project_paths(dict(config))
    settings = config["stage14_conventional"]
    require_compute_resources(dict(config), allow_low_memory=allow_low_memory)
    logger = setup_logging("23_stage14_conventional", dict(config))
    root = paths["root"]
    result_root = paths["results"]
    output_root = root / settings["output_dir"]
    figure_root = root / settings["figure_dir"]
    source_root = output_root / "figure_source_data"
    for directory in [output_root, figure_root, source_root]:
        directory.mkdir(parents=True, exist_ok=True)
    configure_publication_style()

    required_sources = [
        result_root / "06_annotation_v2.h5ad",
        result_root / "pathway_stage2" / "gene_identifier_mapping.tsv",
        result_root / "pathway_stage2" / "hallmark_all_results.tsv",
        result_root / "stage3_subtype_localization" / "subtype_hallmark.tsv",
        result_root / "stage4_composition_decomposition" / "composition_by_library.tsv",
        result_root / "stage4_composition_decomposition" / "decomposition_summary.tsv",
        result_root / "pseudobulk_ready" / "broad_counts.tsv.gz",
    ]
    missing = [str(path) for path in required_sources if not path.exists()]
    if missing:
        raise FileNotFoundError("Stage 14 required sources are missing: " + "; ".join(missing))

    audit = _initial_audit()
    _write_tsv(audit, output_root / "CONVENTIONAL_ANALYSIS_AUDIT.tsv")
    abundance_root = output_root / "abundance"
    enrichment_root = output_root / "enrichment"
    cycle_root = output_root / "cell_cycle"
    marker_root = output_root / "markers"
    de_output_root = output_root / "de_visualization"
    for directory in [abundance_root, enrichment_root, cycle_root, marker_root, de_output_root]:
        directory.mkdir(parents=True, exist_ok=True)

    figure_records: list[dict[str, Any]] = []
    qa_records: list[dict[str, Any]] = []

    def add_plot(
        fig: plt.Figure,
        plot_id: str,
        subdir: str,
        source: pd.DataFrame,
        role: str,
        source_file: str,
    ) -> None:
        manifest, qa = save_standalone_figure(
            fig,
            plot_id,
            figure_root / subdir,
            source,
            source_root,
            scientific_role=role,
            source_file=source_file,
            dpi=int(settings["png_dpi"]),
        )
        figure_records.append(manifest)
        qa_records.append(qa)

    import anndata as ad

    h5ad_path = root / settings["input_object"]
    adata = ad.read_h5ad(h5ad_path)
    if adata.shape != (105763, 57132):
        raise ValueError(f"Unexpected Stage14 input shape: {adata.shape}")
    if settings["counts_layer"] not in adata.layers or not sparse.issparse(adata.layers[settings["counts_layer"]]):
        raise ValueError("Stage14 requires the unchanged sparse raw counts layer")
    obs = adata.obs.copy()

    broad_counts, broad_long = compute_broad_abundance(obs)
    _write_tsv(broad_counts, abundance_root / "broad_cell_abundance.tsv")
    _write_tsv(broad_long, abundance_root / "broad_cell_fraction_by_library.tsv")
    subtype_tables = {
        parent: compute_subtype_abundance(obs, parent)
        for parent in ["Granulosa", "Stromal_fibroblast", "Immune"]
    }
    output_names = {
        "Granulosa": "granulosa_subtype_abundance.tsv",
        "Stromal_fibroblast": "stromal_subtype_abundance.tsv",
        "Immune": "immune_subtype_abundance.tsv",
    }
    for parent, table in subtype_tables.items():
        _write_tsv(table, abundance_root / output_names[parent])
    stage4_composition = _read_tsv(
        result_root / "stage4_composition_decomposition" / "composition_by_library.tsv"
    )
    abundance_summary = _abundance_summary(broad_long, subtype_tables, stage4_composition)
    _write_tsv(abundance_summary, abundance_root / "abundance_summary.tsv")

    add_plot(
        _plot_stacked_fraction(
            broad_long,
            category_col="cell_type",
            fraction_col="fraction_of_total_cells",
            title="Broad cell composition by library",
            color_registry=CELL_TYPE_COLORS,
        ),
        "broad_cell_fraction_stacked_by_library",
        "abundance",
        broad_long,
        "cell_abundance",
        "results/06_annotation_v2.h5ad::obs",
    )
    add_plot(
        _plot_group_library_points(
            broad_long,
            category_col="cell_type",
            value_col="fraction_of_total_cells",
            title="Broad cell fractions: biological libraries",
            xlabel="Fraction of all captured cells (points: libraries; diamonds: mean +/- SD)",
        ),
        "broad_cell_fraction_group_library_points",
        "abundance",
        broad_long,
        "cell_abundance",
        "results/06_annotation_v2.h5ad::obs",
    )
    for parent, table in subtype_tables.items():
        plot_parent = _safe_name(parent).lower()
        add_plot(
            _plot_stacked_fraction(
                table,
                category_col="subtype",
                fraction_col="fraction_of_parent",
                title=f"{parent.replace('_', ' ')} subtype composition by library",
                color_registry=SUBTYPE_COLORS,
            ),
            f"{plot_parent}_subtype_fraction_stacked_by_library",
            "abundance",
            table,
            "subtype_abundance",
            "results/06_annotation_v2.h5ad::obs",
        )
        add_plot(
            _plot_group_library_points(
                table,
                category_col="subtype",
                value_col="fraction_of_parent",
                title=f"{parent.replace('_', ' ')} subtype fractions",
                xlabel="Fraction within parent broad type (points: libraries; diamonds: mean +/- SD)",
            ),
            f"{plot_parent}_subtype_fraction_group_library_points",
            "abundance",
            table,
            "subtype_abundance",
            "results/06_annotation_v2.h5ad::obs",
        )

    cycle_config = config["follicular_subclustering"]["cell_cycle_genes"]
    score_summary, phase_by_library, phase_by_subtype, cycle_interpretation = _cell_cycle_outputs(
        adata,
        cycle_config["s_phase"],
        cycle_config["g2m_phase"],
        subtype_tables["Granulosa"],
        _read_tsv(result_root / "stage3_subtype_localization" / "subtype_hallmark.tsv"),
    )
    _write_tsv(score_summary, cycle_root / "cell_cycle_scores_summary.tsv")
    _write_tsv(phase_by_library, cycle_root / "cell_cycle_phase_by_library.tsv")
    _write_tsv(phase_by_subtype, cycle_root / "cell_cycle_phase_by_subtype.tsv")
    _write_tsv(cycle_interpretation, cycle_root / "cell_cycle_interpretation.tsv")
    add_plot(
        _plot_phase_stacked(phase_by_library),
        "granulosa_cell_cycle_phase_by_library",
        "cell_cycle",
        phase_by_library,
        "cell_cycle_description",
        "results/06_annotation_v2.h5ad::X + obs",
    )
    phase_points = phase_by_subtype.copy()
    phase_points["subtype_phase"] = phase_points["cell_type_subtype_v2"] + " / " + phase_points["phase"]
    add_plot(
        _plot_group_library_points(
            phase_points.loc[phase_points["cell_type_broad_v2"].eq("Granulosa")],
            category_col="subtype_phase",
            value_col="phase_fraction",
            title="Granulosa phase fractions within subtype",
            xlabel="Phase fraction within subtype (points: libraries; diamonds: mean +/- SD)",
        ),
        "granulosa_cell_cycle_phase_by_subtype",
        "cell_cycle",
        phase_points,
        "cell_cycle_description",
        "results/06_annotation_v2.h5ad::X + obs",
    )

    mapping = _read_tsv(result_root / "pathway_stage2" / "gene_identifier_mapping.tsv")
    x = adata.X
    if not sparse.issparse(x):
        raise ValueError("Marker catalogue requires sparse existing log-normalized X")
    selected_indices, symbols = _representative_feature_indices(x, mapping)
    marker_matrix = x[:, selected_indices].tocsr()
    marker_config = load_yaml(paths["markers"])
    broad_canonical, subtype_canonical = _canonical_marker_maps(marker_config)
    broad_labels = obs[settings["broad_key"]].astype(str).reset_index(drop=True)
    subtype_labels = obs[settings["subtype_key"]].astype(str).reset_index(drop=True)
    broad_markers = _marker_rows_for_labels(
        marker_matrix,
        broad_labels,
        symbols,
        broad_canonical,
        top_n=int(settings["marker_top_n"]),
    ).drop(columns="parent_label").rename(columns={"label": "cell_type"})
    broad_markers.insert(1, "subtype", "")
    subtype_markers = _marker_rows_for_labels(
        marker_matrix,
        subtype_labels,
        symbols,
        subtype_canonical,
        top_n=int(settings["marker_top_n"]),
        parent_labels=broad_labels,
    ).rename(columns={"label": "subtype"})
    subtype_markers = subtype_markers.rename(columns={"parent_label": "cell_type"})
    if broad_markers.duplicated(["cell_type", "gene"]).any():
        raise AssertionError("Broad marker catalogue contains duplicate cell-type/gene rows")
    if subtype_markers.duplicated(["cell_type", "subtype", "gene"]).any():
        raise AssertionError("Subtype marker catalogue contains duplicate parent/subtype/gene rows")
    _write_tsv(broad_markers, marker_root / "broad_cell_marker_catalogue.tsv")
    _write_tsv(subtype_markers, marker_root / "subtype_marker_catalogue.tsv")
    for parent, filename in [
        ("Granulosa", "granulosa_marker_catalogue.tsv"),
        ("Stromal_fibroblast", "stromal_marker_catalogue.tsv"),
        ("Immune", "immune_marker_catalogue.tsv"),
    ]:
        _write_tsv(
            subtype_markers.loc[subtype_markers["cell_type"].astype(str).eq(parent)].copy(),
            marker_root / filename,
        )

    marker_plot_specs = [("broad", None, broad_markers, broad_labels)] + [
        (parent.lower(), parent, subtype_markers.loc[subtype_markers["cell_type"].astype(str).eq(parent)], subtype_labels)
        for parent in ["Granulosa", "Stromal_fibroblast", "Immune"]
    ]
    for plot_name, parent, catalogue, labels in marker_plot_specs:
        if parent is None:
            mask = ~broad_labels.isin(["Mixed", "Uncertain", "Uncertain_immune_related"])
            use_labels = broad_labels.loc[mask].reset_index(drop=True)
            use_matrix = marker_matrix[mask.to_numpy()]
            group_col = "cell_type"
        else:
            mask = broad_labels.eq(parent)
            use_labels = labels.loc[mask].reset_index(drop=True)
            use_matrix = marker_matrix[mask.to_numpy()]
            group_col = "subtype"
        genes: list[str] = []
        for _, block in catalogue.groupby(group_col, observed=True, sort=True):
            canonical = block.loc[block["canonical_marker_flag"], "gene"].head(3).tolist()
            selected_genes = canonical + block.loc[~block["gene"].isin(canonical), "gene"].head(max(0, 3 - len(canonical))).tolist()
            genes.extend(selected_genes)
        genes = list(dict.fromkeys(genes))
        dot_source = _dotplot_source(use_matrix, use_labels, symbols, genes)
        add_plot(
            _plot_marker_dotplot(dot_source, f"{plot_name.replace('_', ' ').title()} annotation markers"),
            f"{plot_name}_annotation_marker_dotplot",
            "markers",
            dot_source,
            "annotation_marker_support",
            "results/06_annotation_v2.h5ad::X + obs; resources/markers/ovary_markers.yaml",
        )

    del marker_matrix, x, adata
    gc.collect()

    populations = [
        *settings["primary_populations"],
        *settings["secondary_populations"],
        *settings["exploratory_populations"],
    ]
    contrasts = list(settings["enrichment_contrasts"])
    top_de, top_reversal, de_tables = _prepare_de_tables(
        result_root,
        mapping,
        populations=populations,
        contrasts=contrasts,
        alpha=float(settings["deg_fdr"]),
        top_n=int(settings["deg_top_n"]),
    )
    _write_tsv(top_de, de_output_root / "top_deg_tables.tsv")
    _write_tsv(top_reversal, de_output_root / "top_reversal_genes.tsv")
    for population in populations:
        for contrast in settings["primary_figure_contrasts"]:
            fig, volcano_source = _plot_volcano(de_tables[(population, contrast)], population, contrast)
            add_plot(
                fig,
                f"volcano_{_safe_name(population).lower()}_{contrast.lower()}",
                "volcano",
                volcano_source,
                "standard_pseudobulk_DE",
                f"results/de_stage1_5/{population}/unified_all_genes.tsv.gz",
            )
    broad_pseudobulk = _read_broad_counts(result_root / "pseudobulk_ready" / "broad_counts.tsv.gz")
    for population in settings["primary_populations"]:
        for contrast in settings["primary_figure_contrasts"]:
            selected = top_de.loc[
                top_de["population"].astype(str).eq(population)
                & top_de["contrast"].astype(str).eq(contrast)
            ]
            fig, heat_source = _plot_pseudobulk_heatmap(
                broad_pseudobulk, selected, population, contrast
            )
            add_plot(
                fig,
                f"pseudobulk_heatmap_{_safe_name(population).lower()}_{contrast.lower()}",
                "heatmap",
                heat_source,
                "pseudobulk_replicate_consistency",
                "results/pseudobulk_ready/broad_counts.tsv.gz; results/de_stage1_5",
            )

    resource_dir = root / "resources" / "gene_sets" / "stage14"
    resource_paths, provenance = prepare_mouse_gene_sets(
        resource_dir,
        mapping.loc[mapping["valid_symbol"].astype(bool), "canonical_mouse_symbol"].astype(str).tolist(),
    )
    provenance["file_path"] = provenance["file_path"].map(
        lambda value: str(Path(value).resolve().relative_to(root.resolve()))
    )
    _write_tsv(provenance, enrichment_root / "gene_set_provenance.tsv")
    mapping_audit_rows = []
    example_rank = pd.read_csv(
        result_root / "pathway_stage2" / "ranked_lists" / "Granulosa__OC_vs_Y.rnk",
        sep="\t",
        header=None,
        names=["gene", "stat"],
    )
    for database, resource_path in resource_paths.items():
        mapping_row = validate_gene_set_mapping(
            parse_generic_gmt(resource_path), example_rank["gene"].astype(str).tolist()
        )
        mapping_audit_rows.append({"database": database, **mapping_row, "status": "PASS"})
    _write_tsv(pd.DataFrame(mapping_audit_rows), enrichment_root / "gene_set_mapping_audit.tsv")

    gsea_results = run_functional_gsea(
        result_root / "pathway_stage2" / "ranked_lists",
        resource_paths,
        populations=populations,
        contrasts=contrasts,
        permutations=int(settings["gsea_permutations"]),
        min_size=int(settings["gsea_min_size"]),
        max_size=int(settings["gsea_max_size"]),
        workers=int(settings["gsea_workers"]),
        threads=int(settings["gsea_threads_per_worker"]),
        seed=int(settings["random_seed"]),
    )
    gsea_names = {"GO_BP": "go_bp_gsea.tsv", "Reactome": "reactome_gsea.tsv", "KEGG": "kegg_gsea.tsv"}
    for database, table in gsea_results.items():
        validate_enrichment_schema(table)
        _write_tsv(table, enrichment_root / gsea_names[database])
    ora_results = run_all_ora(
        result_root / "de_stage1_5",
        mapping,
        resource_paths,
        populations=populations,
        contrasts=contrasts,
        alpha=float(settings["deg_fdr"]),
        effect_cutoff=float(settings["deg_effect_size_sensitivity"]),
    )
    ora_names = {
        ("GO_BP", "up"): "go_bp_ora_up.tsv",
        ("GO_BP", "down"): "go_bp_ora_down.tsv",
        ("Reactome", "up"): "reactome_ora_up.tsv",
        ("Reactome", "down"): "reactome_ora_down.tsv",
        ("KEGG", "up"): "kegg_ora_up.tsv",
        ("KEGG", "down"): "kegg_ora_down.tsv",
    }
    for key, filename in ora_names.items():
        _write_tsv(ora_results.get(key, pd.DataFrame()), enrichment_root / filename)
    go_clusters = cluster_go_redundancy(
        gsea_results["GO_BP"],
        parse_generic_gmt(resource_paths["GO_BP"]),
        jaccard_threshold=float(settings["go_redundancy_jaccard"]),
        max_terms_per_context=int(settings["go_redundancy_max_terms_per_context"]),
    )
    _write_tsv(go_clusters, enrichment_root / "go_redundancy_clusters.tsv")
    functional_summary = _functional_summary(gsea_results, resource_paths, result_root)
    _write_tsv(functional_summary, enrichment_root / "functional_enrichment_summary.tsv")
    for database, table in gsea_results.items():
        for population in settings["primary_populations"]:
            for contrast in settings["primary_figure_contrasts"]:
                fig, plot_source = _plot_gsea_terms(table, database, population, contrast)
                add_plot(
                    fig,
                    f"{database.lower()}_{_safe_name(population).lower()}_{contrast.lower()}",
                    "enrichment",
                    plot_source,
                    "functional_annotation",
                    f"results/stage14_conventional/enrichment/{gsea_names[database]}",
                )

    (output_root / "NOT_RECOMMENDED_CONVENTIONAL_ANALYSES.md").write_text(
        _not_recommended_text(), encoding="utf-8"
    )
    _write_tsv(_supplement_plan(), output_root / "CONVENTIONAL_SUPPLEMENT_PLAN.tsv")
    figure_manifest = pd.DataFrame(figure_records)
    figure_qa = pd.DataFrame(qa_records)
    if not figure_qa["qa_pass"].all():
        failed = figure_qa.loc[~figure_qa["qa_pass"], "plot_id"].tolist()
        raise RuntimeError(f"Stage14 figure QA failed: {failed}")
    figure_manifest = _relative_paths(
        figure_manifest,
        root,
        ["source_data", "svg", "pdf", "png_600dpi"],
    )
    _write_tsv(figure_manifest, output_root / "STAGE14_FIGURE_MANIFEST.tsv")
    _write_tsv(figure_qa, output_root / "STAGE14_FIGURE_QA.tsv")
    _write_reports(
        output_root,
        audit,
        abundance_summary,
        functional_summary,
        cycle_interpretation,
        broad_markers,
        subtype_markers,
        figure_manifest,
    )

    outputs = [
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name not in {"manifest.tsv", "COMPLETE.json"}
    ]
    outputs += [path for path in figure_root.rglob("*") if path.is_file()]
    manifest = pd.DataFrame(
        [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(outputs)
        ]
    )
    _write_tsv(manifest, output_root / "manifest.tsv")
    validate_output_manifest(manifest, root)
    complete = {
        "stage": 14,
        "status": "COMPLETE",
        "analysis_code_commit": _git_commit(root),
        "input_object": str(h5ad_path.relative_to(root)),
        "input_shape": list((105763, 57132)),
        "annotated_object_modified": False,
        "statistical_unit": "library",
        "n_libraries_per_group": 3,
        "gsea_permutations": int(settings["gsea_permutations"]),
        "n_standalone_plots": len(figure_manifest),
        "assembled_multipanel_figures": 0,
        "n_broad_marker_rows": len(broad_markers),
        "n_subtype_marker_rows": len(subtype_markers),
        "n_hallmark_discordant_functional_terms": int(
            functional_summary["hallmark_alignment"].eq("discordant_direction").sum()
        ),
        "mandatory_conventional_gaps_remaining": 0,
        "manifest_sha256": sha256_file(output_root / "manifest.tsv"),
    }
    (output_root / "COMPLETE.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Stage14 complete: %d independent plots", len(figure_manifest))
    print_stage14_summary(config)


def print_stage14_summary(config: Mapping[str, Any]) -> None:
    paths = project_paths(dict(config))
    output_root = paths["root"] / config["stage14_conventional"]["output_dir"]
    complete_path = output_root / "COMPLETE.json"
    if not complete_path.exists():
        raise FileNotFoundError(complete_path)
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    abundance = _read_tsv(output_root / "abundance" / "abundance_summary.tsv")
    functional = _read_tsv(output_root / "enrichment" / "functional_enrichment_summary.tsv")
    cycle = _read_tsv(output_root / "cell_cycle" / "cell_cycle_interpretation.tsv")
    cell_cycle_text = cycle.loc[cycle["evidence_type"].eq("integrated_interpretation"), "interpretation"].iloc[0]
    broad = _format_change_rows(abundance, "whole_ovary", n=2)
    granulosa = _format_change_rows(abundance, "Granulosa", n=2)
    stromal = _format_change_rows(abundance, "Stromal_fibroblast", n=2)
    n_tables = len([path for path in output_root.rglob("*.tsv") if "figure_source_data" not in path.parts])
    print("========================================")
    print("STAGE14_CONVENTIONAL_SCRNA_COMPLETE")
    print("========================================")
    print("1_EXISTING=QC;Harmony;annotation;pseudobulk_DE;Hallmark;subtype;composition;Stage3-13")
    print("2_ADDED=conventional_abundance;GO_BP;Reactome;KEGG;ORA;cell_cycle;marker_catalogues;volcano;heatmap")
    print(f"3_BROAD_ABUNDANCE={broad}")
    print(f"4_GRANULOSA_SUBTYPE_ABUNDANCE={granulosa}")
    print(f"5_STROMAL_SUBTYPE_ABUNDANCE={stromal}")
    print(f"6_GO_BP={_format_top_terms(functional, 'GO_BP', 'Stromal_fibroblast', 'OT_vs_OC')}")
    print(f"7_REACTOME={_format_top_terms(functional, 'Reactome', 'Stromal_fibroblast', 'OT_vs_OC')}")
    print(f"8_KEGG={_format_top_terms(functional, 'KEGG', 'Stromal_fibroblast', 'OT_vs_OC')}")
    print(f"9_CELL_CYCLE={cell_cycle_text}")
    print(f"10_MARKERS=broad:{complete['n_broad_marker_rows']};subtype:{complete['n_subtype_marker_rows']}")
    print(f"11_HALLMARK_DISCORDANCE={complete['n_hallmark_discordant_functional_terms']}")
    print(f"12_MANDATORY_GAPS_REMAINING={complete['mandatory_conventional_gaps_remaining']}")
    print(f"13_NEW_STANDALONE_PLOTS={complete['n_standalone_plots']}")
    print(f"14_NEW_TABLES={n_tables}")
    print("15_REPORT=results/stage14_conventional/STAGE14_CONVENTIONAL_SCRNA_REPORT_CN.md")
    print("16_EXPLAINED=results/stage14_conventional/STAGE14_EXPLAINED_FOR_USER_CN.md")
    print(f"17_FINAL_GIT_COMMIT={_git_commit(paths['root'])}")
