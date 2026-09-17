#!/usr/bin/env python
from __future__ import annotations

import argparse

from ms_ovary_scrna.project import load_config
from ms_ovary_scrna.stage4_composition_decomposition import run_stage4


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 composition/intrinsic decomposition")
    parser.add_argument("--config", default="config/analysis_config.yaml")
    parser.add_argument("--allow-low-memory", action="store_true")
    args = parser.parse_args()
    run_stage4(load_config(args.config), allow_low_memory=args.allow_low_memory)


if __name__ == "__main__":
    main()
