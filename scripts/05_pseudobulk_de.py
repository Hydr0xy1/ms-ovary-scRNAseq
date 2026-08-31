from __future__ import annotations

import argparse

from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config
from ms_ovary_scrna.pseudobulk import run_pseudobulk


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample-level pseudobulk DE and rescue analysis.")
    parser.add_argument("input")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--skip-deseq2", action="store_true")
    parser.add_argument("--allow-low-memory", action="store_true")
    args = parser.parse_args()
    run_pseudobulk(
        load_config(args.config),
        args.input,
        skip_deseq2=args.skip_deseq2,
        allow_low_memory=args.allow_low_memory,
    )


if __name__ == "__main__":
    main()
