from __future__ import annotations

import argparse

from ms_ovary_scrna.broad_annotation import run_broad_annotation_review

from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Curate major ovarian lineages with marker, exclusion, QC and reference evidence."
        )
    )
    parser.add_argument("input")
    parser.add_argument("--output")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--allow-low-memory", action="store_true")
    args = parser.parse_args()
    run_broad_annotation_review(
        load_config(args.config),
        args.input,
        output_path=args.output,
        allow_low_memory=args.allow_low_memory,
    )


if __name__ == "__main__":
    main()
