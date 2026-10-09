#!/usr/bin/env python
"""专门解析颗粒/基质：复用冻结结果，不修改主对象。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ms_ovary_scrna.stage27_granulosa_stromal import export_metadata, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--export-obs", type=Path)
    args = parser.parse_args()
    if args.export_obs:
        export_metadata(args.export_obs, args.root)
    else:
        run(args.root)


if __name__ == "__main__":
    main()
