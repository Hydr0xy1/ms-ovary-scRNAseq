#!/usr/bin/env python
from __future__ import annotations

import argparse

from ms_ovary_scrna.project import load_config
from ms_ovary_scrna.stage15_projection_state import run_state_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit multidimensional ovarian cell-state modules")
    parser.add_argument("--config", default="config/analysis_config.yaml")
    args = parser.parse_args()
    run_state_audit(load_config(args.config))


if __name__ == "__main__":
    main()
