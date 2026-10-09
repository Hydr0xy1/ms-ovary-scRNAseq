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


def test_heatmap_observed_zero_is_not_missing() -> None:
    table = pd.DataFrame([
        {"cell_type_broad_v2": "Luteal", "library_id": "Y_2", "group": "Y",
         "phase": "G1", "n_cells": 123, "total_cells": 123, "phase_fraction": 1.0},
        {"cell_type_broad_v2": "Luteal", "library_id": "OT_2", "group": "OT",
         "phase": "G1", "n_cells": 26, "total_cells": 26, "phase_fraction": 1.0},
    ])
    source = MODULE.cycling_heatmap_source(MODULE.prepare_phase_table(table))
    assert source["cycling_percent"].eq(0).all()
    assert source["total_cells"].tolist() == [123, 26]
    assert "OC_1" not in source["library_id"].astype(str).tolist()
