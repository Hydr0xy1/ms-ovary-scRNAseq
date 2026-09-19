#!/usr/bin/env python
from __future__ import annotations

import argparse

from ms_ovary_scrna.project import load_config
from ms_ovary_scrna.stage15_followup_validation import run_followup_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 15 follow-up validation analyses")
    parser.add_argument("--config", default="config/analysis_config.yaml")
    args = parser.parse_args()
    run_followup_validation(load_config(args.config))


if __name__ == "__main__":
    main()
