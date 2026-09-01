from __future__ import annotations

import argparse

from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config
from ms_ovary_scrna.qc_sensitivity import run_qc_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only QC threshold and Scrublet sensitivity analysis."
    )
    parser.add_argument("input")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--allow-low-memory", action="store_true")
    args = parser.parse_args()
    run_qc_sensitivity(
        load_config(args.config),
        args.input,
        allow_low_memory=args.allow_low_memory,
    )


if __name__ == "__main__":
    main()
