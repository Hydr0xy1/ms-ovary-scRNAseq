#!/usr/bin/env python
from __future__ import annotations

import argparse

from ms_ovary_scrna.project import load_config
from ms_ovary_scrna.stage16_ml import run_stage16_ml


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 16-22 deep-dive ML and external validation")
    parser.add_argument("--config", default="config/analysis_config.yaml")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="Optional stage names for a checkpointed partial run",
    )
    parser.add_argument(
        "--force",
        nargs="+",
        default=None,
        help="Re-run named stages even when their checkpoint is complete",
    )
    args = parser.parse_args()
    run_stage16_ml(
        load_config(args.config),
        selected_stages=args.stages,
        force_stages=args.force,
    )


if __name__ == "__main__":
    main()
