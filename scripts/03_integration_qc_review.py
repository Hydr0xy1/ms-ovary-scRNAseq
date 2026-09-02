from __future__ import annotations

import argparse

from ms_ovary_scrna.integration_qc_review import run_integration_qc_review
from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only Harmony, resolution, and secondary-QC review."
    )
    parser.add_argument("input")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--unintegrated-reference")
    parser.add_argument("--cluster-marker-scores")
    parser.add_argument("--primary-markers")
    parser.add_argument("--allow-low-memory", action="store_true")
    args = parser.parse_args()
    run_integration_qc_review(
        load_config(args.config),
        args.input,
        unintegrated_reference=args.unintegrated_reference,
        cluster_marker_scores=args.cluster_marker_scores,
        primary_markers=args.primary_markers,
        allow_low_memory=args.allow_low_memory,
    )


if __name__ == "__main__":
    main()
