from __future__ import annotations

import argparse

from ms_ovary_scrna.preprocessing import run_preprocess_cluster
from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize, select HVGs, run PCA, integrate, and cluster."
    )
    parser.add_argument("input")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--integration", choices=["none", "harmony", "bbknn"], default=None)
    parser.add_argument("--allow-low-memory", action="store_true")
    args = parser.parse_args()
    run_preprocess_cluster(
        load_config(args.config),
        args.input,
        integration=args.integration,
        allow_low_memory=args.allow_low_memory,
    )


if __name__ == "__main__":
    main()
