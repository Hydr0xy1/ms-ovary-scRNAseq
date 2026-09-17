from __future__ import annotations

import ast
from pathlib import Path


def test_advanced_stage_logging_calls_match_project_api() -> None:
    package_root = Path(__file__).parents[1] / "src" / "ms_ovary_scrna"
    for stage in range(6, 13):
        path = next(package_root.glob(f"stage{stage}_*.py"))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setup_logging"
        ]
        assert calls, f"{path.name} must initialize a stage log"
        assert all(len(call.args) == 2 for call in calls), (
            f"{path.name} must call setup_logging(name, config)"
        )
