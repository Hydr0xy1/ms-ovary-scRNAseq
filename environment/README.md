# Mouse ovary scRNA-seq environment

The server Conda environment is named `ovary_sc`, uses Python 3.12, and is
registered as the Jupyter kernel `Python (ovary_sc)`.

## Activate

```bash
cd /root/autodl-tmp/ovary_scRNAseq
source environment/activate.sh
```

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
