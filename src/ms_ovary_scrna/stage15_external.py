"""Stage 15 external mouse-ovary age/cycle reference.

The external program is defined only from public GSE267729 samples.  Internal
Y/OC/OT data are used only after the external ranking is frozen for projection.
No existing AnnData object is modified.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import requests
from anndata import AnnData
from scipy.io import mmread
from scipy.sparse import csr_matrix
import yaml

from .project import project_paths, setup_logging


GEO_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
SAMPLE_IDS = [f"GSM{n}" for n in range(8274677, 8274694)]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _fetch_sample_metadata(sample_id: str) -> dict[str, Any]:
    response = requests.get(
        GEO_URL,
        params={"acc": sample_id, "targ": "self", "form": "text", "view": "quick"},
        timeout=60,
    )
    response.raise_for_status()
    text = response.content.decode("utf-8", "replace")
    lines = text.splitlines()
    title = next((line.split(" = ", 1)[1] for line in lines if line.startswith("!Sample_title = ")), "")
    characteristics = [line.split(" = ", 1)[1] for line in lines if line.startswith("!Sample_characteristics_ch1")]
    urls = [line.split(" = ", 1)[1].replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov") for line in lines if line.startswith("!Sample_supplementary_file")]
    joined = " ".join([title, *characteristics]).lower()
    if "young_regular" in joined:
        age_group, cycle_state = "young", "regular"
    elif "peri-estropause_regular" in joined:
        age_group, cycle_state = "peri", "regular"
    elif "peri-estropause_irregular" in joined:
        age_group, cycle_state = "peri", "irregular"
    elif "post-estropause" in joined or "acylic" in joined or "acyclic" in joined:
        age_group, cycle_state = "post", "acyclic"
    else:
        age_group, cycle_state = "unknown", "unknown"
    age_match = re.search(r"age:\s*([0-9.]+)month", joined)
    return {
        "sample_id": sample_id,
        "title": title,
        "age_months": float(age_match.group(1)) if age_match else np.nan,
        "age_group": age_group,
        "cycle_state": cycle_state,
        "matrix_urls": urls,
        "metadata_source": f"NCBI GEO {sample_id}",
    }


def fetch_registry(output_root: Path) -> pd.DataFrame:
    rows = [_fetch_sample_metadata(sample_id) for sample_id in SAMPLE_IDS]
    table = pd.DataFrame(rows)
    table["matrix_url_1"] = table["matrix_urls"].map(lambda x: x[0] if len(x) > 0 else "")
    table["matrix_url_2"] = table["matrix_urls"].map(lambda x: x[1] if len(x) > 1 else "")
    table["matrix_url_3"] = table["matrix_urls"].map(lambda x: x[2] if len(x) > 2 else "")
    table = table.drop(columns=["matrix_urls"])
    table.to_csv(output_root / "GSE267729_sample_registry.tsv", sep="\t", index=False)
    return table


def _download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            return
        except Exception as exc:  # pragma: no cover - network retry path
            last_error = exc
            if path.exists():
                path.unlink()
            time.sleep(2**attempt)
    raise RuntimeError(f"Download failed: {url}: {last_error}")


def _validate_gzip(path: Path) -> None:
    """Read a gzip file to EOF so truncated downloads cannot enter analysis."""

    with gzip.open(path, "rb") as handle:
        while handle.read(1024 * 1024):
            pass


def _validate_downloads(output_root: Path, registry: pd.DataFrame) -> None:
    bad: list[str] = []
    for row in registry.itertuples(index=False):
        if row.age_group == "unknown":
            continue
        sample_root = output_root / "processed_10x" / str(row.sample_id)
        for url in [row.matrix_url_1, row.matrix_url_2, row.matrix_url_3]:
            if not url:
                continue
            path = sample_root / Path(url.split("/", 1)[-1]).name
            try:
                _validate_gzip(path)
            except (OSError, EOFError, FileNotFoundError) as exc:
                bad.append(f"{path}: {exc}")
    if bad:
        raise RuntimeError("Gzip integrity check failed:\n" + "\n".join(bad))


def download_gse267729(output_root: Path, logger: Any) -> pd.DataFrame:
    registry_path = output_root / "GSE267729_sample_registry.tsv"
    registry = pd.read_csv(registry_path, sep="\t") if registry_path.exists() else fetch_registry(output_root)
    raw_root = output_root / "processed_10x"
    manifest_rows: list[dict[str, Any]] = []
    for row in registry.itertuples(index=False):
        if row.age_group == "unknown":
            continue
        sample_root = raw_root / str(row.sample_id)
        urls = [row.matrix_url_1, row.matrix_url_2, row.matrix_url_3]
        for url in urls:
            if not url:
                continue
            filename = Path(url.split("/", 1)[-1]).name
            path = sample_root / filename
            _download(url, path)
            manifest_rows.append({"sample_id": row.sample_id, "path": str(path.relative_to(output_root)), "bytes": path.stat().st_size, "sha256": _sha256(path)})
        logger.info("Downloaded/verified %s", row.sample_id)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_root / "GSE267729_download_manifest.tsv", sep="\t", index=False)
    return registry


def _marker_panels(marker_file: Path) -> dict[str, dict[str, list[str]]]:
    payload = yaml.safe_load(marker_file.read_text(encoding="utf-8"))
    return payload["major_annotation"]


def _read_external_10x(sample_root: Path) -> AnnData:
    matrix_files = list(sample_root.glob("*_matrix.mtx.gz"))
    feature_files = list(sample_root.glob("*_features.tsv.gz"))
    barcode_files = list(sample_root.glob("*_barcodes.tsv.gz"))
    if len(matrix_files) != 1 or len(feature_files) != 1 or len(barcode_files) != 1:
        raise FileNotFoundError(f"Expected one matrix/features/barcodes file in {sample_root}")
    matrix = csr_matrix(mmread(matrix_files[0]).tocsr().T)
    features = pd.read_csv(feature_files[0], sep="\t", header=None, compression="gzip")
    barcodes = pd.read_csv(barcode_files[0], sep="\t", header=None, compression="gzip")[0].astype(str).tolist()
    symbols = features.iloc[:, 1 if features.shape[1] >= 2 else 0].astype(str).tolist()
    # Cell Ranger feature symbols can repeat.  Keep the first occurrence in a
    # deterministic way rather than silently collapsing external features.
    seen: dict[str, int] = {}
    unique_symbols: list[str] = []
    for symbol in symbols:
        count = seen.get(symbol, 0)
        unique_symbols.append(symbol if count == 0 else f"{symbol}-{count}")
        seen[symbol] = count + 1
    return AnnData(X=matrix, obs=pd.DataFrame(index=barcodes), var=pd.DataFrame(index=unique_symbols))


def _score_labels(adata: AnnData, panels: Mapping[str, Mapping[str, list[str]]]) -> tuple[np.ndarray, pd.DataFrame]:
    totals = np.asarray(adata.X.sum(axis=1)).ravel().astype(float)
    totals[totals <= 0] = 1.0
    # scipy's sparse multiply may return COO; normalize back to CSR before
    # column indexing below.
    x = adata.X.tocsr().multiply((1e4 / totals)[:, None]).tocsr()
    x.data = np.log1p(x.data)
    genes = pd.Index(adata.var_names.astype(str))
    scores: dict[str, np.ndarray] = {}
    evidence: dict[str, int] = {}
    for label, spec in panels.items():
        pos = [genes.get_loc(g) for g in spec.get("positive", []) if g in genes]
        neg = [genes.get_loc(g) for g in spec.get("exclude", []) if g in genes]
        if len(pos) < 2:
            continue
        score = np.asarray(x[:, pos].mean(axis=1)).ravel()
        if neg:
            score -= np.asarray(x[:, neg].mean(axis=1)).ravel()
        scores[label] = score
        evidence[label] = len(pos)
    score_table = pd.DataFrame(scores)
    labels = score_table.idxmax(axis=1).to_numpy(dtype=object)
    best = score_table.max(axis=1).to_numpy()
    labels[best <= 0.05] = "Uncertain"
    return labels, pd.DataFrame([{"label": k, "n_markers_found": v} for k, v in evidence.items()])


def _sample_pseudobulk(output_root: Path, registry: pd.DataFrame, marker_file: Path, logger: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    panels = _marker_panels(marker_file)
    rows: list[dict[str, Any]] = []
    count_rows: list[pd.DataFrame] = []
    gene_order: pd.Index | None = None
    for row in registry.itertuples(index=False):
        if row.age_group == "unknown":
            continue
        sample_root = output_root / "processed_10x" / str(row.sample_id)
        adata = _read_external_10x(sample_root)
        labels, evidence = _score_labels(adata, panels)
        gene_order = pd.Index(adata.var_names.astype(str)) if gene_order is None else gene_order
        if not gene_order.equals(pd.Index(adata.var_names.astype(str))):
            raise ValueError("GSE267729 feature order differs across samples")
        for population in ["Granulosa", "Stromal_fibroblast"]:
            mask = labels == population
            n_cells = int(mask.sum())
            if n_cells < 30:
                logger.warning("Low external cells: %s %s n=%s", row.sample_id, population, n_cells)
                continue
            counts = np.asarray(adata.X[mask].sum(axis=0)).ravel().astype(float)
            count_rows.append(pd.DataFrame([counts], index=[f"{row.sample_id}::{population}"], columns=gene_order))
            rows.append({"sample_id": row.sample_id, "age_group": row.age_group, "cycle_state": row.cycle_state, "age_months": row.age_months, "population": population, "n_cells": n_cells, "n_uncertain": int((labels == "Uncertain").sum()), "n_total_cells": int(adata.n_obs)})
        logger.info("Processed %s", row.sample_id)
    counts = pd.concat(count_rows, axis=0) if count_rows else pd.DataFrame()
    metadata = pd.DataFrame(rows)
    counts.to_csv(output_root / "GSE267729_broad_pseudobulk_counts.tsv.gz", sep="\t", compression="gzip")
    metadata.to_csv(output_root / "GSE267729_broad_pseudobulk_metadata.tsv", sep="\t", index=False)
    return counts, metadata


def _age_programs(output_root: Path, counts: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if counts.empty or metadata.empty:
        raise ValueError("No pseudobulk counts available")
    rows: list[pd.DataFrame] = []
    meta = metadata.set_index(metadata["sample_id"].astype(str) + "::" + metadata["population"].astype(str))
    for population in ["Granulosa", "Stromal_fibroblast"]:
        for contrast, aged_mask in [("peri_regular_vs_young", (meta["age_group"] == "peri") & (meta["cycle_state"] == "regular")), ("peri_irregular_vs_young", (meta["age_group"] == "peri") & (meta["cycle_state"] == "irregular")), ("post_acyclic_vs_young", meta["age_group"] == "post")]:
            keys = meta.index[meta["population"].eq(population)]
            young = keys[meta.loc[keys, "age_group"].eq("young")]
            aged = keys[meta.loc[keys].index.map(lambda x: bool(aged_mask.loc[x]))]
            if len(young) < 3 or len(aged) < 3:
                continue
            lib = counts.loc[young.tolist() + aged.tolist()]
            log_cpm = np.log2(lib.div(lib.sum(axis=1), axis=0) * 1e6 + 0.5)
            effect = log_cpm.loc[aged].mean(axis=0) - log_cpm.loc[young].mean(axis=0)
            result = pd.DataFrame({"gene": effect.index.astype(str), "external_age_log2fc": effect.to_numpy(), "population": population, "contrast": contrast, "n_young": len(young), "n_aged": len(aged)})
            result["abs_effect"] = result["external_age_log2fc"].abs()
            result = result.sort_values(["external_age_log2fc", "gene"], ascending=[False, True], kind="stable")
            rows.append(result)
    programs = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    programs.to_csv(output_root / "GSE267729_external_age_programs.tsv.gz", sep="\t", index=False, compression="gzip")
    return programs


def run_stage15_external(config: Mapping[str, Any], *, download_only: bool = False) -> Path:
    paths = project_paths(dict(config))
    settings = config["deep_dive_stage15"]
    root = paths["root"]
    output_root = root / settings["output_dir"] / "external_gse267729"
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("25_stage15_external", dict(config))
    registry = fetch_registry(output_root)
    registry = download_gse267729(output_root, logger)
    if download_only:
        status = {"status": "DOWNLOAD_COMPLETE", "next_stage": "external_pseudobulk_and_frozen_programs", "n_samples": int(len(registry)), "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        (output_root / "CHECKPOINT.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print("STAGE15_EXTERNAL_DOWNLOAD_COMPLETE")
        return output_root
    _validate_downloads(output_root, registry)
    counts, metadata = _sample_pseudobulk(output_root, registry, root / config["project"]["marker_file"], logger)
    programs = _age_programs(output_root, counts, metadata)
    status = {"status": "EXTERNAL_PROGRAMS_COMPLETE", "n_samples": int(registry["sample_id"].nunique()), "n_pseudobulk_rows": int(len(counts)), "n_program_rows": int(len(programs)), "program_definition": "external-only log2CPM age contrasts; no internal OT data used for selection", "next_stage": "internal_projection_and_composition_state_decomposition", "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    (output_root / "CHECKPOINT.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("STAGE15_EXTERNAL_PROGRAMS_COMPLETE")
    print(f"OUTPUT={output_root.relative_to(root)}")
    return output_root
