from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_CONFIG = Path("/root/autodl-tmp/ovary_scRNAseq/config/analysis_config.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key in ("project", "resources", "ingest", "qc", "preprocess"):
        if key not in config:
            raise KeyError(f"Missing configuration section: {key}")
    return config


def project_paths(config: dict[str, Any]) -> dict[str, Path]:
    project = config["project"]
    paths = {
        "root": Path(project["root"]),
        "input": Path(project["input_dir"]),
        "results": Path(project["result_dir"]),
        "figures": Path(project["figure_dir"]),
        "logs": Path(project["log_dir"]),
        "metadata": Path(project["metadata"]),
        "markers": Path(project["marker_file"]),
        "pathways": Path(project["pathway_file"]),
    }
    for key in ("results", "figures", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def read_metadata(config: dict[str, Any]) -> pd.DataFrame:
    path = project_paths(config)["metadata"]
    metadata = pd.read_csv(path, sep="\t", dtype=str).set_index("library_id", drop=False)
    if metadata.index.has_duplicates:
        duplicates = metadata.index[metadata.index.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate library_id values: {duplicates}")
    return metadata


def setup_logging(name: str, config: dict[str, Any]) -> logging.Logger:
    paths = project_paths(config)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)
        file_handler = logging.FileHandler(paths["logs"] / f"{name}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def cgroup_memory_limit_gib() -> float | None:
    candidates = [
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ]
    for path in candidates:
        if path.exists():
            value = path.read_text().strip()
            if value != "max":
                return int(value) / 1024**3
    return None


def require_compute_resources(
    config: dict[str, Any], *, allow_low_memory: bool = False
) -> None:
    if allow_low_memory or os.environ.get("OVARY_ALLOW_LOW_MEMORY") == "1":
        return
    if not config["resources"].get("guard_heavy_jobs", True):
        return
    limit = cgroup_memory_limit_gib()
    recommended = float(config["resources"]["recommended_compute_memory_gib"])
    if limit is not None and limit < recommended:
        raise RuntimeError(
            f"Heavy job blocked: cgroup memory limit is {limit:.1f} GiB; "
            f"configuration recommends at least {recommended:.0f} GiB. "
            "Switch to compute mode, or explicitly pass --allow-low-memory only "
            "for a synthetic/small test."
        )


def write_json(value: Any, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
