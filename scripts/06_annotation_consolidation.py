from __future__ import annotations

import argparse

from ms_ovary_scrna.annotation_consolidation import run_annotation_consolidation
from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate reviewed local annotations into a new main AnnData object."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--allow-low-memory", action="store_true")
    args = parser.parse_args()
    run_annotation_consolidation(load_config(args.config), allow_low_memory=args.allow_low_memory)


if __name__ == "__main__":
    main()
