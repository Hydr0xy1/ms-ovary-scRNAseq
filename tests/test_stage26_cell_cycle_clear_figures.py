from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/43_stage26_cell_cycle_clear_figures.py"
SPEC = importlib.util.spec_from_file_location("stage26_clear_figures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_granulosa_source_preserves_phase_fractions() -> None:
    rows = []
    for library in MODULE.LIBRARIES:
        group = library.split("_")[0]
        for phase, count in [("G1", 60), ("S", 15), ("G2M", 25)]:
            rows.append(
                {
                    "cell_type_broad_v2": "Granulosa",
                    "library_id": library,
                    "group": group,
                    "phase": phase,
                    "n_cells": count,
                    "total_cells": 100,
                    "phase_fraction": count / 100,
                }
            )
    source = MODULE.granulosa_source(MODULE.prepare_phase_table(pd.DataFrame(rows)))
    assert len(source) == 27
    assert source["cycling_percent"].eq(40).all()
    assert source.groupby("library_id", observed=True)["phase_fraction"].sum().eq(1).all()

