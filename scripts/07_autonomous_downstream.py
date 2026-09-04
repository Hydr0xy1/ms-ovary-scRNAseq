from __future__ import annotations

import argparse

from ms_ovary_scrna.autonomous_downstream import run_autonomous_pipeline
from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run conservative stromal/immune/epithelial review, v2 consolidation "
            "and pseudobulk readiness."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--allow-low-memory", action="store_true")
    args = parser.parse_args()
    run_autonomous_pipeline(load_config(args.config), allow_low_memory=args.allow_low_memory)


if __name__ == "__main__":
    main()
