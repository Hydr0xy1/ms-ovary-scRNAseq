from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config, project_paths


GENES = [
    "Fshr", "Foxl2", "Amh", "Cyp19a1", "Inha", "Inhbb",
    "Col1a1", "Col1a2", "Col3a1", "Dcn", "Lum", "Pdgfra",
    "Ptprc", "Lcp1", "Tyrobp", "Adgre1", "Csf1r", "Lyz2",
    "Cdkn2a", "Cdkn1a", "Trp53", "Serpine1", "Il6", "Il1b", "Tnf", "Ccl2",
    "Nfe2l2", "Keap1", "Hmox1", "Nqo1", "Sod1", "Sod2", "Cat", "Gpx1",
    "Star", "Cyp11a1", "Cyp17a1", "Hsd3b1", "Lhcgr", "Scarb1",
    "Bax", "Bcl2", "Casp3", "Casp9", "Pecam1", "Cdh5", "Kdr", "Emcn",
] + [f"Gene{i}" for i in range(72)]


def make_synthetic(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    libraries = [f"{group}_{rep}" for group in ("Y", "OC", "OT") for rep in range(1, 4)]
    matrices = []
    obs_records = []
    for library in libraries:
        group = library.split("_")[0]
        for cell_type in ("Granulosa", "Stromal"):
            n_cells = 18
            x = rng.poisson(1.0, size=(n_cells, len(GENES))).astype(np.int32)
            if cell_type == "Granulosa":
                x[:, 0:6] += rng.poisson(3.0, size=(n_cells, 6))
            else:
                x[:, 6:12] += rng.poisson(3.0, size=(n_cells, 6))
            if group == "OC":
                x[:, 18:26] += rng.poisson(3.0, size=(n_cells, 8))
            elif group == "OT":
                x[:, 18:26] += rng.poisson(1.0, size=(n_cells, 8))
            matrices.append(sparse.csr_matrix(x))
            for index in range(n_cells):
                obs_records.append(
                    {
                        "library_id": library,
                        "group": group,
                        "treatment": "MRJP1" if group == "OT" else "vehicle",
                        "batch": "batch1",
                        "estrous_stage": "unknown",
                        "synthetic_cell_type": cell_type,
                        "barcode": f"{library}-{cell_type}-{index}",
                    }
                )
    matrix = sparse.vstack(matrices, format="csr")
    obs = pd.DataFrame(obs_records).set_index("barcode")
    var = pd.DataFrame(index=pd.Index(GENES, name="gene"))
    adata = ad.AnnData(X=matrix.copy(), obs=obs, var=var)
    adata.layers["counts"] = matrix.copy()
    adata.write_h5ad(path, compression="gzip")


def run(command: list[str], env: dict[str, str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny end-to-end test safe for no-card mode.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    source_config = load_config(args.config)
    source_paths = project_paths(source_config)
    script_dir = Path(__file__).resolve().parent
    tmp_root = Path(tempfile.mkdtemp(prefix="ovary_smoke_"))
    try:
        for name in (
            "results",
            "figures",
            "logs",
            "config",
            "data",
            "metadata",
            "resources/markers",
            "resources/gene_sets",
        ):
            (tmp_root / name).mkdir(parents=True, exist_ok=True)
        synthetic = tmp_root / "synthetic_counts.h5ad"
        make_synthetic(synthetic, int(source_config["project"]["random_seed"]))

        config = source_config.copy()
        config["project"] = dict(source_config["project"])
        config["project"].update(
            {
                "root": str(tmp_root),
                "input_dir": "data",
                "result_dir": "results",
                "figure_dir": "figures",
                "log_dir": "logs",
                "metadata": "metadata/sample_metadata.tsv",
                "marker_file": "resources/markers/ovary_markers.yaml",
                "pathway_file": "resources/gene_sets/pathway_gene_sets.yaml",
                "cluster_labels": "metadata/cluster_labels.tsv",
            }
        )
        config["qc"] = dict(source_config["qc"])
        config["qc"].update({"min_genes_initial": 5, "min_cells_per_gene": 2})
        config["preprocess"] = dict(source_config["preprocess"])
        config["preprocess"].update(
            {"n_top_hvg": 50, "n_pcs": 20, "use_n_pcs": 15, "n_neighbors": 10}
        )
        config["pseudobulk"] = dict(source_config["pseudobulk"])
        config["pseudobulk"].update(
            {"cell_type_key": "synthetic_cell_type", "min_cells_per_sample_cell_type": 5, "n_cpus": 1}
        )
        config["pathway"] = dict(source_config["pathway"])
        config["pathway"].update({"permutations": 20, "min_size": 3})
        config_path = tmp_root / "config" / "smoke_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        shutil.copy2(
            source_paths["markers"],
            tmp_root / "resources" / "markers" / "ovary_markers.yaml",
        )
        shutil.copy2(
            source_paths["pathways"],
            tmp_root / "resources" / "gene_sets" / "pathway_gene_sets.yaml",
        )
        pd.DataFrame(
            {
                "library_id": [f"{group}_{rep}" for group in ("Y", "OC", "OT") for rep in range(1, 4)],
                "group": [group for group in ("Y", "OC", "OT") for _ in range(3)],
            }
        ).to_csv(tmp_root / "metadata" / "sample_metadata.tsv", sep="\t", index=False)
        (tmp_root / "metadata" / "cluster_labels.tsv").write_text(
            "cluster\tcell_type_final\tannotation_notes\n", encoding="utf-8"
        )

        env = os.environ.copy()
        env.update(
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMBA_NUM_THREADS": "1",
                "NUMBA_THREADING_LAYER": "omp",
                "OVARY_ALLOW_LOW_MEMORY": "1",
            }
        )
        python = sys.executable
        run(
            [python, str(script_dir / "02_qc.py"), str(synthetic), "--config", str(config_path), "--skip-scrublet", "--apply-filter", "--allow-low-memory"],
            env,
        )
        run(
            [python, str(script_dir / "03_preprocess_cluster.py"), str(tmp_root / "results" / "02_qc_filtered.h5ad"), "--config", str(config_path), "--integration", "harmony", "--allow-low-memory"],
            env,
        )
        run(
            [python, str(script_dir / "04_annotate.py"), str(tmp_root / "results" / "03_clustered.h5ad"), "--config", str(config_path), "--allow-low-memory"],
            env,
        )
        run(
            [python, str(script_dir / "05_pseudobulk_de.py"), str(tmp_root / "results" / "04_annotated.h5ad"), "--config", str(config_path), "--allow-low-memory"],
            env,
        )
        run([python, str(script_dir / "06_pathway_rescue.py"), "--config", str(config_path)], env)
        required = [
            tmp_root / "results" / "02_qc_filtered.h5ad",
            tmp_root / "results" / "03_clustered.h5ad",
            tmp_root / "results" / "04_annotated.h5ad",
            tmp_root / "results" / "05_pseudobulk" / "Granulosa" / "gene_rescue_summary.tsv",
            tmp_root / "results" / "06_pathways" / "pathway_rescue_summary.tsv",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise AssertionError(f"Smoke test outputs missing: {missing}")
        print(f"SMOKE_TEST_OK={tmp_root}")
    finally:
        if not args.keep:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
