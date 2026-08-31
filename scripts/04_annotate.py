from __future__ import annotations

import argparse

from ms_ovary_scrna.annotation import run_annotation
from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Marker-guided and CellTypist annotation.")
    parser.add_argument("input")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-celltypist", action="store_true")
    parser.add_argument("--allow-low-memory", action="store_true")
    args = parser.parse_args()
    run_annotation(
        load_config(args.config),
        args.input,
        use_celltypist=args.run_celltypist,
        allow_low_memory=args.allow_low_memory,
    )


if __name__ == "__main__":
    main()
