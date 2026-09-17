#!/usr/bin/env python
from __future__ import annotations

import argparse

from ms_ovary_scrna.project import load_config
from ms_ovary_scrna.stage3_subtype_localization import run_stage3


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 subtype localization")
    parser.add_argument("--config", default="config/analysis_config.yaml")
    parser.add_argument("--allow-low-memory", action="store_true")
    parser.add_argument("--skip-gsea", action="store_true")
    parser.add_argument("--skip-permutation", action="store_true")
    args = parser.parse_args()
    run_stage3(
        load_config(args.config),
        allow_low_memory=args.allow_low_memory,
        skip_gsea=args.skip_gsea,
        skip_permutation=args.skip_permutation,
    )


if __name__ == "__main__":
    main()
