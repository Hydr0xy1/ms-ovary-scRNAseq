"""Low-memory, all-cell cell-cycle scoring for the frozen mouse ovary atlas.

The formal AnnData is too large for a 2 GiB no-GPU instance.  This module
therefore reads the CSR-encoded log-normalized ``X`` directly from HDF5 and
reproduces Scanpy's ``score_genes_cell_cycle`` calculation in bounded chunks.
It never rewrites the source H5AD.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from .project import project_paths

PHASES = ("G1", "S", "G2M")
OBS_COLUMNS = (
    "library_id",
    "group",
    "cell_type_broad_v2",
    "cell_type_subtype_v2",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(value: Any, path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _decode_strings(dataset: h5py.Dataset) -> np.ndarray:
    """Read an AnnData string-array dataset without loading the full object."""
    values = dataset.asstr()[:]
    return np.asarray(values, dtype=object)


def _read_obs_column(obs: h5py.Group, name: str) -> np.ndarray:
    """Read either a string array or an AnnData categorical column."""
    node = obs[name]
    if isinstance(node, h5py.Dataset):
        return _decode_strings(node)
    categories = _decode_strings(node["categories"])
    codes = np.asarray(node["codes"][:], dtype=np.int64)
    result = np.full(len(codes), "", dtype=object)
    valid = codes >= 0
    result[valid] = categories[codes[valid]]
    return result


def _file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_gene_means(
    matrix: h5py.Group,
    *,
    n_obs: int,
    n_vars: int,
    chunk_nnz: int,
) -> np.ndarray:
    """Compute all-gene means from a CSR HDF5 group with bounded memory."""
    data = matrix["data"]
    indices = matrix["indices"]
    totals = np.zeros(n_vars, dtype=np.float64)
    total_nnz = len(data)
    next_report = 0.1
    for start in range(0, total_nnz, chunk_nnz):
        stop = min(start + chunk_nnz, total_nnz)
        cols = np.asarray(indices[start:stop], dtype=np.int64)
        values = np.asarray(data[start:stop], dtype=np.float64)
        totals += np.bincount(cols, weights=values, minlength=n_vars)
        progress = stop / total_nnz
        if progress + 1e-12 >= next_report:
            print(f"GENE_MEANS_PROGRESS={progress:.0%}", flush=True)
            next_report += 0.1
    return totals / float(n_obs)


def scanpy_control_genes(
    var_names: Sequence[str],
    gene_means: Sequence[float],
    gene_list: Sequence[str],
    *,
    ctrl_size: int,
    n_bins: int = 25,
    random_state: int = 0,
    ctrl_as_ref: bool = True,
) -> list[str]:
    """Reproduce Scanpy 1.12 ``_score_genes_bins`` control selection."""
    gene_pool = pd.Index(var_names, dtype="string")
    genes = pd.Index(gene_list, dtype="string")
    obs_avg = pd.Series(np.asarray(gene_means, dtype=float), index=gene_pool)
    obs_avg = obs_avg[np.isfinite(obs_avg)]
    n_items = int(np.round(len(obs_avg) / (n_bins - 1)))
    obs_cut = obs_avg.rank(method="min") // n_items
    keep_ctrl_in_obs_cut: np.bool_ | np.ndarray
    keep_ctrl_in_obs_cut = np.False_ if ctrl_as_ref else obs_cut.index.isin(genes)

    np.random.seed(random_state)
    control_genes = pd.Index([], dtype="string")
    for cut in np.unique(obs_cut.loc[genes]):
        candidates = obs_cut[(obs_cut == cut) & ~keep_ctrl_in_obs_cut].index
        if ctrl_size < len(candidates):
            candidates = candidates.to_series().sample(ctrl_size).index
        if ctrl_as_ref:
            candidates = candidates.difference(genes)
        control_genes = control_genes.union(candidates)
    if len(control_genes) == 0:
        raise RuntimeError("No control genes were selected")
    return control_genes.astype(str).tolist()


def _score_weight(
    lookup: Mapping[str, int], genes: Sequence[str], controls: Sequence[str], n_vars: int
) -> np.ndarray:
    weight = np.zeros(n_vars, dtype=np.float64)
    weight[[lookup[gene] for gene in genes]] += 1.0 / len(genes)
    weight[[lookup[gene] for gene in controls]] -= 1.0 / len(controls)
    return weight


def stream_cell_scores(
    matrix: h5py.Group,
    *,
    n_obs: int,
    n_vars: int,
    s_weight: np.ndarray,
    g2m_weight: np.ndarray,
    cells_per_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate per-cell S and G2M scores from bounded CSR row chunks."""
    data = matrix["data"]
    indices = matrix["indices"]
    indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
    if len(indptr) != n_obs + 1:
        raise ValueError("CSR indptr length does not match n_obs")
    s_score = np.empty(n_obs, dtype=np.float64)
    g2m_score = np.empty(n_obs, dtype=np.float64)
    next_report = 0.1
    for row_start in range(0, n_obs, cells_per_chunk):
        row_stop = min(row_start + cells_per_chunk, n_obs)
        value_start = int(indptr[row_start])
        value_stop = int(indptr[row_stop])
        local_data = np.asarray(data[value_start:value_stop], dtype=np.float32)
        local_indices = np.asarray(indices[value_start:value_stop], dtype=np.int32)
        local_indptr = indptr[row_start : row_stop + 1] - value_start
        block = sparse.csr_matrix(
            (local_data, local_indices, local_indptr),
            shape=(row_stop - row_start, n_vars),
        )
        s_score[row_start:row_stop] = np.asarray(block @ s_weight).ravel()
        g2m_score[row_start:row_stop] = np.asarray(block @ g2m_weight).ravel()
        progress = row_stop / n_obs
        if progress + 1e-12 >= next_report:
            print(f"CELL_SCORE_PROGRESS={progress:.0%}", flush=True)
            next_report += 0.1
    return s_score, g2m_score


def assign_phase(s_score: Sequence[float], g2m_score: Sequence[float]) -> np.ndarray:
    """Apply the exact phase decision rule used by Scanpy."""
    s = np.asarray(s_score, dtype=float)
    g2m = np.asarray(g2m_score, dtype=float)
    if s.shape != g2m.shape or not np.isfinite(s).all() or not np.isfinite(g2m).all():
        raise ValueError("Cell-cycle scores must be finite arrays with identical shape")
    phase = np.full(len(s), "S", dtype=object)
    phase[g2m > s] = "G2M"
    phase[(s < 0) & (g2m < 0)] = "G1"
    return phase


def _phase_summary(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    counts = (
        frame.groupby([*keys, "phase"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    totals = counts.groupby(list(keys), observed=True)["n_cells"].transform("sum")
    counts["total_cells"] = totals
    counts["phase_fraction"] = counts["n_cells"] / totals
    return counts.sort_values([*keys, "phase"], kind="stable").reset_index(drop=True)


def _score_summary(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    return (
        frame.groupby(list(keys), observed=True)
        .agg(
            n_cells=("phase", "size"),
            S_score_mean=("S_score", "mean"),
            S_score_median=("S_score", "median"),
            G2M_score_mean=("G2M_score", "mean"),
            G2M_score_median=("G2M_score", "median"),
            cycle_score_max_median=("cycle_score_max", "median"),
            phase_margin_median=("phase_margin", "median"),
        )
        .reset_index()
    )


def _stage14_parity(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Validate the streaming implementation against the frozen Stage 14 run."""
    stage14 = root / "results/stage14_conventional/cell_cycle"
    rows: list[dict[str, Any]] = []
    expected_phase_path = stage14 / "cell_cycle_phase_by_library.tsv"
    if expected_phase_path.exists():
        expected = pd.read_csv(expected_phase_path, sep="\t")
        actual = _phase_summary(
            frame.loc[frame["cell_type_broad_v2"].eq("Granulosa")], ["library_id"]
        )
        actual = actual[["library_id", "phase", "n_cells"]]
        merged = expected.merge(
            actual,
            on=["library_id", "phase"],
            how="outer",
            suffixes=("_stage14", "_streaming"),
        ).fillna(-1)
        for row in merged.itertuples(index=False):
            difference = int(row.n_cells_streaming) - int(row.n_cells_stage14)
            rows.append(
                {
                    "audit_type": "phase_count",
                    "scope": f"Granulosa/{row.library_id}/{row.phase}",
                    "stage14_value": int(row.n_cells_stage14),
                    "streaming_value": int(row.n_cells_streaming),
                    "absolute_difference": abs(difference),
                    "tolerance": 0,
                    "pass": difference == 0,
                }
            )

    expected_score_path = stage14 / "cell_cycle_scores_summary.tsv"
    if expected_score_path.exists():
        expected = pd.read_csv(expected_score_path, sep="\t")
        targets = frame.loc[
            frame["cell_type_broad_v2"].eq("Granulosa")
            | frame["cell_type_subtype_v2"].eq("Theca_cycling")
        ]
        actual_rows = []
        scopes = [("Broad_Granulosa", targets["cell_type_broad_v2"].eq("Granulosa"))]
        scopes.extend(
            (subtype, targets["cell_type_subtype_v2"].eq(subtype))
            for subtype in sorted(targets["cell_type_subtype_v2"].unique())
        )
        for scope, mask in scopes:
            block = targets.loc[mask]
            if block.empty:
                continue
            actual_rows.append(
                {
                    "scope": scope,
                    "n_cells": len(block),
                    "S_score_mean": block["S_score"].mean(),
                    "S_score_median": block["S_score"].median(),
                    "G2M_score_mean": block["G2M_score"].mean(),
                    "G2M_score_median": block["G2M_score"].median(),
                }
            )
        actual = pd.DataFrame(actual_rows).set_index("scope")
        expected = expected.set_index("scope")
        for scope in expected.index.intersection(actual.index):
            for metric in [
                "n_cells",
                "S_score_mean",
                "S_score_median",
                "G2M_score_mean",
                "G2M_score_median",
            ]:
                left = float(expected.loc[scope, metric])
                right = float(actual.loc[scope, metric])
                tolerance = 0.0 if metric == "n_cells" else 1e-6
                rows.append(
                    {
                        "audit_type": "score_summary",
                        "scope": f"{scope}/{metric}",
                        "stage14_value": left,
                        "streaming_value": right,
                        "absolute_difference": abs(right - left),
                        "tolerance": tolerance,
                        "pass": abs(right - left) <= tolerance,
                    }
                )
    return pd.DataFrame(rows)


def _write_report(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    input_path: Path,
    present_s: Sequence[str],
    present_g2m: Sequence[str],
    parity: pd.DataFrame,
) -> None:
    overall = _phase_summary(frame, ["group"])
    pivot = (
        overall.pivot(index="group", columns="phase", values="phase_fraction")
        .reindex(columns=PHASES, fill_value=0)
        .fillna(0)
    )
    table_lines = ["| group | G1 | S | G2M |", "|---|---:|---:|---:|"]
    for group in ["Y", "OC", "OT"]:
        if group in pivot.index:
            table_lines.append(
                f"| {group} | {pivot.loc[group, 'G1']:.1%} | "
                f"{pivot.loc[group, 'S']:.1%} | {pivot.loc[group, 'G2M']:.1%} |"
            )
    counts = frame["phase"].value_counts().reindex(PHASES, fill_value=0)
    report = f"""# 全细胞周期分析报告

## 运行结论

- 输入：`{input_path}`，{len(frame):,} 个细胞。
- 使用已审核的小鼠细胞周期基因：S 期 {len(present_s)} 个，G2M 期 {len(present_g2m)} 个。
- 按 Scanpy 规则分配：G1 {counts['G1']:,}，S {counts['S']:,}，G2M {counts['G2M']:,}。
- 低内存流式算法与已冻结 Stage 14 Granulosa 结果的一致性校验：{'PASS' if len(parity) and parity['pass'].all() else 'NOT CONFIRMED'}。
- 主 H5AD 仅以只读方式打开，未覆盖、未回归掉细胞周期信号。

## 三组细胞级描述

{chr(10).join(table_lines)}

> 上表为细胞级描述，不能把单个细胞当作生物学重复。正式比较应使用每个 library/pool 的周期比例（每组 n=3）。

## 解读边界

- `phase` 是基于转录组 S/G2M 模块的推定，不等同于 EdU/BrdU、Ki67 或 DNA 含量的实验测量。
- 对差异化程度高、RNA 含量低或应激较强的细胞，G1 更准确的含义是“未检出明确 S/G2M 转录信号”。
- 请同时查看连续的 `S_score`、`G2M_score`、`cycle_score_max` 和 `phase_margin`，不要只依赖三分类标签。
- 本阶段不使用细胞周期回归，也不改变已有降维、聚类或注释。
"""
    (output_dir / "CELL_CYCLE_REPORT_CN.md").write_text(report, encoding="utf-8")


def run_all_cell_cycle(
    config: Mapping[str, Any],
    *,
    chunk_nnz: int = 2_000_000,
    cells_per_chunk: int = 500,
    reuse_gene_means: bool = True,
) -> None:
    root = project_paths(dict(config))["root"]
    input_path = root / "results/06_annotation_v2.h5ad"
    output_dir = root / "results/deep_dive_stage26_all_cell_cycle"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    settings = config["follicular_subclustering"]["cell_cycle_genes"]
    requested_s = [str(gene) for gene in settings["s_phase"]]
    requested_g2m = [str(gene) for gene in settings["g2m_phase"]]
    before = input_path.stat()
    state_path = output_dir / "RUN_STATE.json"
    state = {
        "stage": "26_all_cell_cycle",
        "status": "running",
        "started_at": _now(),
        "input": str(input_path.relative_to(root)),
        "input_bytes": before.st_size,
        "input_mtime_ns": before.st_mtime_ns,
        "python": platform.python_version(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "chunk_nnz": chunk_nnz,
        "cells_per_chunk": cells_per_chunk,
        "source_matrix": "X: log1p library-size normalized expression (target_sum=10000)",
        "method": "bounded-memory reproduction of scanpy.tl.score_genes_cell_cycle",
    }
    _write_json(state, state_path)

    try:
        with h5py.File(input_path, "r") as handle:
            matrix = handle["X"]
            n_obs, n_vars = [int(value) for value in matrix.attrs["shape"]]
            if (n_obs, n_vars) != (105_763, 57_132):
                raise ValueError(f"Unexpected frozen object shape: {(n_obs, n_vars)}")
            if matrix.attrs.get("encoding-type") != "csr_matrix":
                raise ValueError("Stage 26 requires CSR-encoded X")
            var_names = _decode_strings(handle["var"]["_index"]).astype(str)
            if len(set(var_names)) != n_vars:
                raise ValueError("Feature names must be unique")
            lookup = {gene: index for index, gene in enumerate(var_names)}
            present_s = [gene for gene in requested_s if gene in lookup]
            present_g2m = [gene for gene in requested_g2m if gene in lookup]
            if len(present_s) < 20 or len(present_g2m) < 20:
                raise ValueError(
                    f"Too few cell-cycle genes: S={len(present_s)}, G2M={len(present_g2m)}"
                )
            obs = {"cell_barcode": _read_obs_column(handle["obs"], "_index")}
            obs.update(
                {column: _read_obs_column(handle["obs"], column) for column in OBS_COLUMNS}
            )
            if any(len(values) != n_obs for values in obs.values()):
                raise ValueError("Observation metadata length mismatch")

            marker_rows = []
            for phase, genes in [("S", requested_s), ("G2M", requested_g2m)]:
                marker_rows.extend(
                    {"phase": phase, "gene": gene, "present": gene in lookup} for gene in genes
                )
            pd.DataFrame(marker_rows).to_csv(
                output_dir / "CELL_CYCLE_MARKER_COVERAGE.tsv", sep="\t", index=False
            )

            means_path = output_dir / "gene_means_log_normalized.npy"
            means_meta_path = output_dir / "gene_means_metadata.json"
            means_meta = {
                "input_bytes": before.st_size,
                "input_mtime_ns": before.st_mtime_ns,
                "n_obs": n_obs,
                "n_vars": n_vars,
            }
            can_reuse = False
            if reuse_gene_means and means_path.exists() and means_meta_path.exists():
                stored = json.loads(means_meta_path.read_text(encoding="utf-8"))
                can_reuse = stored == means_meta
            if can_reuse:
                gene_means = np.load(means_path)
                print("GENE_MEANS_REUSED=true", flush=True)
            else:
                gene_means = stream_gene_means(
                    matrix, n_obs=n_obs, n_vars=n_vars, chunk_nnz=chunk_nnz
                )
                np.save(means_path, gene_means)
                _write_json(means_meta, means_meta_path)

            ctrl_size = min(len(present_s), len(present_g2m))
            s_controls = scanpy_control_genes(
                var_names, gene_means, present_s, ctrl_size=ctrl_size
            )
            g2m_controls = scanpy_control_genes(
                var_names, gene_means, present_g2m, ctrl_size=ctrl_size
            )
            pd.DataFrame(
                [
                    *({"score": "S_score", "control_gene": gene} for gene in s_controls),
                    *(
                        {"score": "G2M_score", "control_gene": gene}
                        for gene in g2m_controls
                    ),
                ]
            ).to_csv(output_dir / "CONTROL_GENES.tsv", sep="\t", index=False)

            s_weight = _score_weight(lookup, present_s, s_controls, n_vars)
            g2m_weight = _score_weight(lookup, present_g2m, g2m_controls, n_vars)
            s_score, g2m_score = stream_cell_scores(
                matrix,
                n_obs=n_obs,
                n_vars=n_vars,
                s_weight=s_weight,
                g2m_weight=g2m_weight,
                cells_per_chunk=cells_per_chunk,
            )

        phase = assign_phase(s_score, g2m_score)
        frame = pd.DataFrame(obs)
        frame["S_score"] = s_score
        frame["G2M_score"] = g2m_score
        frame["cycle_score_max"] = np.maximum(s_score, g2m_score)
        frame["phase_margin"] = np.abs(s_score - g2m_score)
        frame["phase"] = phase
        per_cell_path = output_dir / "CELL_CYCLE_SCORES_PER_CELL.tsv.gz"
        with gzip.open(per_cell_path, "wt", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, sep="\t", index=False)

        summaries = {
            "PHASE_BY_LIBRARY.tsv": _phase_summary(frame, ["library_id", "group"]),
            "PHASE_BY_GROUP_CELL_LEVEL.tsv": _phase_summary(frame, ["group"]),
            "PHASE_BY_BROAD_CELL_TYPE.tsv": _phase_summary(frame, ["cell_type_broad_v2"]),
            "PHASE_BY_BROAD_CELL_TYPE_AND_LIBRARY.tsv": _phase_summary(
                frame, ["cell_type_broad_v2", "library_id", "group"]
            ),
            "SCORE_BY_BROAD_CELL_TYPE.tsv": _score_summary(
                frame, ["cell_type_broad_v2"]
            ),
            "SCORE_BY_SUBTYPE.tsv": _score_summary(
                frame, ["cell_type_broad_v2", "cell_type_subtype_v2"]
            ),
        }
        for filename, table in summaries.items():
            table.to_csv(output_dir / filename, sep="\t", index=False)

        parity = _stage14_parity(frame, root)
        parity.to_csv(output_dir / "STAGE14_PARITY_AUDIT.tsv", sep="\t", index=False)
        if parity.empty or not parity["pass"].all():
            raise RuntimeError("Low-memory scores did not reproduce the frozen Stage 14 audit")

        after = input_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Input H5AD changed during read-only scoring")
        _write_report(
            frame,
            output_dir,
            input_path=input_path,
            present_s=present_s,
            present_g2m=present_g2m,
            parity=parity,
        )
        state.update(
            status="complete",
            finished_at=_now(),
            n_cells=n_obs,
            n_features=n_vars,
            n_s_genes=len(present_s),
            n_g2m_genes=len(present_g2m),
            n_s_controls=len(s_controls),
            n_g2m_controls=len(g2m_controls),
            phase_counts={key: int(value) for key, value in frame["phase"].value_counts().items()},
            stage14_parity_pass=True,
            source_h5ad_unchanged=True,
            per_cell_sha256=_file_sha256(per_cell_path),
        )
        _write_json(state, state_path)
        print("STAGE26_ALL_CELL_CYCLE_COMPLETE", flush=True)
    except Exception as exc:
        state.update(status="failed", finished_at=_now(), error=repr(exc))
        _write_json(state, state_path)
        raise
