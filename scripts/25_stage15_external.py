#!/usr/bin/env python
from __future__ import annotations

import argparse

from ms_ovary_scrna.project import load_config
from ms_ovary_scrna.stage15_external import run_stage15_external


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 15 external GSE267729 age/cycle reference")
    parser.add_argument("--config", default="config/analysis_config.yaml")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    run_stage15_external(load_config(args.config), download_only=args.download_only)


if __name__ == "__main__":
    main()
