from __future__ import annotations

import argparse

from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config
from ms_ovary_scrna.qc import run_qc


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-library adaptive QC and Scrublet.")
    parser.add_argument("input")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--apply-filter", action="store_true")
    parser.add_argument("--skip-scrublet", action="store_true")
    parser.add_argument("--allow-low-memory", action="store_true")
    args = parser.parse_args()
    run_qc(
        load_config(args.config),
        args.input,
        apply_filter=args.apply_filter,
        skip_scrublet=args.skip_scrublet,
        allow_low_memory=args.allow_low_memory,
    )


if __name__ == "__main__":
    main()
