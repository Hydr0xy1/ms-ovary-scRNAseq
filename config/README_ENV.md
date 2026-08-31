# Mouse ovary scRNA-seq environment

Project root: `/root/autodl-tmp/ovary_scRNAseq`

## Activate

```bash
source /root/autodl-tmp/ovary_scRNAseq/config/activate.sh
```

The Conda environment is named `ovary_sc`, uses Python 3.12, and is registered as
the Jupyter kernel `Python (ovary_sc)`.

## Project layout

- `data/Summary/`: extracted, filtered 10x matrices (read-only source data)
- `notebooks/`: interactive analyses
- `scripts/`: reproducible analysis scripts
- `results/`: processed AnnData objects and tabular outputs
- `figures/`: plots
- `logs/`: run logs and validation artifacts
- `config/`: environment requirements, lock file, and activation helper

## Start JupyterLab

```bash
source /root/autodl-tmp/ovary_scRNAseq/config/activate.sh
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Use an SSH tunnel from the local computer rather than opening port 8888 publicly.

## Recreate

The direct requested packages are in `config/requirements.in`. The fully resolved
snapshot generated after installation is `config/requirements-lock.txt`.
