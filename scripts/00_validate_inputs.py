from __future__ import annotations

import argparse

from ms_ovary_scrna.io import inspect_filtered_inputs, validate_inputs
from ms_ovary_scrna.project import DEFAULT_CONFIG, load_config

EXPECTED_CELLS = {
    "Y_1": 13_711,
    "Y_2": 11_599,
    "Y_3": 11_612,
    "OC_1": 12_192,
    "OC_2": 10_207,
    "OC_3": 11_902,
    "OT_1": 12_781,
    "OT_2": 13_005,
    "OT_3": 9_844,
}
EXPECTED_FEATURES = 57_132
EXPECTED_TOTAL_CELLS = 106_853


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-memory audit of 10x MTX inputs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--full-gzip-test",
        action="store_true",
        help="Read every compressed byte and verify CRC (safe but slow).",
    )
    parser.add_argument(
        "--scanpy-audit",
        action="store_true",
        help="Read each matrix sequentially as AnnData and report its unmodified structure.",
    )
    parser.add_argument("--focus-sample", default="Y_1")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_inputs(config, full_gzip_test=args.full_gzip_test)
    if args.scanpy_audit:
        inspect_filtered_inputs(
            config,
            expected_cells=EXPECTED_CELLS,
            expected_features=EXPECTED_FEATURES,
            expected_total_cells=EXPECTED_TOTAL_CELLS,
            focus_sample=args.focus_sample,
        )


if __name__ == "__main__":
    main()
