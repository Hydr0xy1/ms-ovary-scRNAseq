# Development guide

## Where new code belongs

- Reusable calculations: `src/ms_ovary_scrna/<domain>.py`
- Terminal entry point: `scripts/<stage>_<name>.py`
- Unit/regression test: `tests/test_<domain>.py`
- Exploratory review: `notebooks/<stage>_<topic>_review.ipynb`
- Scientific parameters: `config/analysis_config.yaml`
- Sample and curated labels: `metadata/`
- Marker/pathway definitions: `resources/`

Numbered scripts should parse arguments and call package functions. They should
not contain reusable Scanpy, plotting, annotation, or statistical logic.

## Examples

Cell-composition analysis:

```text
src/ms_ovary_scrna/composition.py
scripts/07_cell_composition.py
tests/test_composition.py
notebooks/05_cell_composition_review.ipynb
```

Trajectory analysis:

```text
src/ms_ovary_scrna/trajectory.py
scripts/08_trajectory.py
tests/test_trajectory.py
notebooks/06_trajectory_review.ipynb
```

## Local development loop

```powershell
python -m pip install --no-deps -e .
python -m compileall -q -f src scripts tests
ruff check src scripts tests
pytest -q -m "not integration"
git diff --check
git status --short
```

Do not commit data, H5AD objects, generated results, environment secrets, or
machine-specific absolute paths.
