#!/usr/bin/env python
from __future__ import annotations

import argparse

from ms_ovary_scrna.project import load_config
from ms_ovary_scrna.stage14_conventional import print_stage14_summary, run_stage14


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 14 conventional scRNA-seq completeness and supplementary analyses"
    )
    parser.add_argument("--config", default="config/analysis_config.yaml")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print the completed Stage 14 summary without recomputing outputs.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.summary_only:
        print_stage14_summary(config)
    else:
        run_stage14(config)


if __name__ == "__main__":
    main()
