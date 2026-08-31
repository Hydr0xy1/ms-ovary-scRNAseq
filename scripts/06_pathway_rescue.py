from __future__ import annotations

import argparse

from ms_ovary_scrna.pathways import run_pathway_rescue
from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom ovarian pathway GSEA.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    run_pathway_rescue(load_config(args.config))


if __name__ == "__main__":
    main()
