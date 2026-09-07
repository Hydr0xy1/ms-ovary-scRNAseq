from __future__ import annotations

import argparse

from ms_ovary_scrna.pathway_stage2 import run_pathway_stage2
from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Broad-cell Mouse Hallmark pathway reversal with exact label permutation."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-workers", type=int, default=7)
    parser.add_argument("--model-cpus", type=int, default=4)
    parser.add_argument("--gsea-workers", type=int, default=7)
    parser.add_argument("--gsea-threads", type=int, default=4)
    parser.add_argument("--allow-low-memory", action="store_true")
    args = parser.parse_args()
    run_pathway_stage2(
        load_config(args.config),
        model_workers=args.model_workers,
        model_cpus=args.model_cpus,
        gsea_workers=args.gsea_workers,
        gsea_threads=args.gsea_threads,
        allow_low_memory=args.allow_low_memory,
    )


if __name__ == "__main__":
    main()
