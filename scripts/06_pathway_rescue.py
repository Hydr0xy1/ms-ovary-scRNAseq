from __future__ import annotations

import argparse
from pathlib import Path

import gseapy as gp
import pandas as pd

from _project import DEFAULT_CONFIG, load_config, load_yaml, project_paths, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom ovarian pathway GSEA from pseudobulk ranks.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config = load_config(args.config)
    paths = project_paths(config)
    logger = setup_logging("06_pathway_rescue", config)
    settings = config["pathway"]
    gene_sets = load_yaml(paths["pathways"])
    input_root = paths["results"] / "05_pseudobulk"
    output_root = paths["results"] / "06_pathways"
    output_root.mkdir(parents=True, exist_ok=True)

    result_files = sorted(input_root.glob("*/*_vs_*.tsv"))
    if not result_files:
        raise FileNotFoundError(f"No pseudobulk contrast tables found under {input_root}")
    summaries = []
    for path in result_files:
        table = pd.read_csv(path, sep="\t")
        ranking_column = settings["ranking_column"]
        ranking = table[["gene", ranking_column]].dropna().sort_values(ranking_column, ascending=False)
        ranking = ranking.drop_duplicates("gene")
        pre = gp.prerank(
            rnk=ranking,
            gene_sets=gene_sets,
            permutation_num=int(settings["permutations"]),
            min_size=int(settings["min_size"]),
            max_size=int(settings["max_size"]),
            seed=int(config["project"]["random_seed"]),
            outdir=None,
            no_plot=True,
            verbose=False,
        )
        result = pre.res2d.copy()
        result.insert(0, "cell_type", path.parent.name)
        result.insert(1, "contrast", path.stem)
        summaries.append(result)
        logger.info("GSEA: %s/%s (%d pathways)", path.parent.name, path.stem, len(result))
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(output_root / "custom_pathway_gsea.tsv", sep="\t", index=False)

    pivot = combined.pivot_table(
        index=["cell_type", "Term"], columns="contrast", values="NES", aggfunc="first"
    ).reset_index()
    if {"OC_vs_Y", "OT_vs_OC", "OT_vs_Y"}.issubset(pivot.columns):
        pivot["opposite_aging_treatment"] = (
            pivot["OC_vs_Y"] * pivot["OT_vs_OC"] < 0
        )
        pivot["closer_to_y"] = pivot["OT_vs_Y"].abs() < pivot["OC_vs_Y"].abs()
        pivot["pathway_rescued"] = pivot["opposite_aging_treatment"] & pivot["closer_to_y"]
    pivot.to_csv(output_root / "pathway_rescue_summary.tsv", sep="\t", index=False)
    logger.info("PATHWAY_RESCUE_OK: %s", output_root)


if __name__ == "__main__":
    main()
