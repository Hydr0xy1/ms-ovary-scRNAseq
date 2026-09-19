#!/usr/bin/env python
"""Render independent Stage 16-22 publication figure drafts with Python."""

from __future__ import annotations

import argparse
from pathlib import Path

from ms_ovary_scrna.stage16_figures import generate_stage16_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing results/deep_dive_stage16_ml",
    )
    args = parser.parse_args()
    outputs = generate_stage16_figures(args.project_root.resolve())
    print(f"STAGE16_FIGURES_COMPLETE n={len(outputs)}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
