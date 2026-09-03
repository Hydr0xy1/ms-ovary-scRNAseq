from __future__ import annotations

import argparse

from ms_ovary_scrna.follicular_subclustering import run_follicular_subclustering
from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Subcluster the follicular/steroidogenic compartment without changing the atlas."
        )
    )
    parser.add_argument("input")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--primary-resolution", type=float)
    parser.add_argument("--use-n-pcs", type=int)
    parser.add_argument("--allow-low-memory", action="store_true")
    args = parser.parse_args()
    run_follicular_subclustering(
        load_config(args.config),
        args.input,
        primary_resolution=args.primary_resolution,
        use_n_pcs=args.use_n_pcs,
        allow_low_memory=args.allow_low_memory,
    )


if __name__ == "__main__":
    main()
