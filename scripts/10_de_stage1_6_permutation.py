from __future__ import annotations

import argparse

from ms_ovary_scrna.de_stage1_6 import run_de_stage1_6
from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact library-label permutation validation for Stage 1.5 effect geometry."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--allow-low-memory", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore complete population-level checkpoints and recompute all models.",
    )
    args = parser.parse_args()
    run_de_stage1_6(
        load_config(args.config),
        allow_low_memory=args.allow_low_memory,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

