# Mouse ovary scRNA-seq environment

The server Conda environment is named `ovary_sc`, uses Python 3.12, and is
registered as the Jupyter kernel `Python (ovary_sc)`.

## Activate

```bash
cd /root/autodl-tmp/ovary_scRNAseq
source environment/activate.sh
```

The activation script validates the OpenMP, MKL, OpenBLAS, and Numba thread
limits. It replaces invalid values such as AutoDL's `OMP_NUM_THREADS=0` with 16.
For a smaller job, set a fallback before activation:

```bash
export OVARY_THREADS=4
source environment/activate.sh
```

An already valid library-specific value is preserved, so a one-command limit
such as `OMP_NUM_THREADS=1 python scripts/99_smoke_test.py` still takes priority.
The Numba threading backend is set to the server-tested `omp`; inherited `tbb`
is not used because this image contains an incompatible TBB runtime. Only set
`OVARY_NUMBA_THREADING_LAYER` if you have deliberately installed and tested a
different backend.

## Editable project installation

After the first Git pull that adds or changes the Python package layout, run:

```bash
python -m pip install --no-deps -e .
python -c "import ms_ovary_scrna; print(ms_ovary_scrna.__version__)"
```

## Start JupyterLab

```bash
source environment/activate.sh
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Use an SSH tunnel rather than exposing port 8888 publicly.

## Recreate

Direct dependencies are in `environment/requirements.in`. The resolved snapshot
is `environment/requirements-lock.txt`.

```bash
bash environment/bootstrap_env.sh
```

`bootstrap_env.sh` regenerates the resolved lock from the server environment and
excludes the editable local checkout. If the lock changes, copy it back to the
local repository, review it, and commit it locally; do not commit directly from
the server checkout.
