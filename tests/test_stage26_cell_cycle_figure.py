from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/42_stage26_cell_cycle_figure.py"
SPEC = importlib.util.spec_from_file_location("stage26_cell_cycle_figure", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
prepare_source = MODULE.prepare_source


def test_prepare_source_uses_library_as_row_unit() -> None:
    table = pd.DataFrame(
        {
            "cell_type_broad_v2": ["Granulosa"] * 6,
            "library_id": ["Y_1"] * 3 + ["OC_1"] * 3,
            "group": ["Y"] * 3 + ["OC"] * 3,
            "phase": ["G1", "S", "G2M"] * 2,
            "n_cells": [60, 20, 20, 80, 10, 10],
            "total_cells": [100] * 6,
        }
    )
    points, summary, order = prepare_source(table)
    assert order == ["Granulosa"]
    assert len(points) == 2
    assert points.set_index("library_id").loc["Y_1", "cycling_fraction"] == 0.4
    assert points.set_index("library_id").loc["OC_1", "cycling_fraction"] == 0.2
    assert summary.set_index("group").loc["Y", "n_libraries"] == 1
