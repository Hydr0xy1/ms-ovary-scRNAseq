from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/45_stage27_focused_figures.py"
SPEC = importlib.util.spec_from_file_location("focused_figures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_short_heatmap_group_headers_do_not_overlap_title() -> None:
    MODULE.style()
    matrix = pd.DataFrame([[1] * 9] * 4, columns=MODULE.LIBRARIES)
    figure = MODULE.heatmap(matrix, "Stromal fibroblast: weak-score sensitivity", "Percent", "Test")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    ax = figure.axes[0]
    # 左对齐 title 用 _left_title；检查分组标题与主标题的实际渲染边界。
    title_box = ax._left_title.get_window_extent(renderer)
    for text in ax.texts:
        if text.get_text() in ["Y", "OC", "OT"]:
            assert not title_box.overlaps(text.get_window_extent(renderer))
    plt.close(figure)
