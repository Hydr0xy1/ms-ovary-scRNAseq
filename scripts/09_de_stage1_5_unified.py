from __future__ import annotations

import argparse

from ms_ovary_scrna.de_stage1_5 import run_de_stage1_5
from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified nine-library broad pseudobulk DE and rescue-ready effect geometry."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--allow-low-memory", action="store_true")
    args = parser.parse_args()
    run_de_stage1_5(load_config(args.config), allow_low_memory=args.allow_low_memory)


if __name__ == "__main__":
    main()

