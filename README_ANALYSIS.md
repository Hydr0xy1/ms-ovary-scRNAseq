# Mouse ovary scRNA-seq analysis

Reproducible Scanpy workflow for the following biological groups:

- `Y`: 4-month vehicle control
- `OC`: 10-month vehicle control
- `OT`: 10-month MRJP1, 200 mg/kg

Primary contrasts:

- `OC vs Y`: reproductive-aging effect
- `OT vs OC`: MRJP1 treatment effect
- `OT vs Y`: residual deviation from the young state

## Code organization

- `src/ms_ovary_scrna/`: reusable, tested Python functions
- `scripts/`: thin command-line entry points and ordered workflow stages
- `config/`: scientific parameters only
- `metadata/`: sample metadata, inventory, and curated cluster labels
- `resources/`: ovarian marker panels and pathway gene sets
- `environment/`: dependency specifications and activation/bootstrap helpers
- `notebooks/`: interactive review of small result tables and figures
- `tests/`: unit, integration, and synthetic regression tests
- `data/`, `results/`, `figures/`, `logs/`: runtime files excluded from Git

See `docs/DEVELOPMENT.md` before adding new code.

## Environment

On the configured server:

```bash
cd /root/autodl-tmp/ovary_scRNAseq
source environment/activate.sh
python -m pip install --no-deps -e .
```

The editable installation makes `ms_ovary_scrna` importable while keeping the
source of truth in this Git checkout.

## Before analysis

Edit `metadata/sample_metadata.tsv` and replace every recoverable `TODO` or
`unknown` value. If four mice were pooled into each library, the formal
single-cell sample size is three pools per group, not twelve mice per group.

## Safe preflight and regression tests

```bash
source environment/activate.sh

python -m compileall -q -f src scripts tests
pytest -q -m "not integration"
python scripts/00_validate_inputs.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 python scripts/99_smoke_test.py
```

## Compute workflow

Run each stage separately and review its output before continuing.

```bash
source environment/activate.sh

# 1. Merge all libraries and preserve raw counts.
python scripts/01_build_merged.py

# 2a. Annotate QC failures without deleting cells.
python scripts/02_qc.py results/01_merged_counts.h5ad

# Review results/02_qc_summary.tsv and figures/02_qc_before_filtering.png.
# Only after biological review, apply the proposed filters.
python scripts/02_qc.py results/01_merged_counts.h5ad --apply-filter

# 3. Normalize, select batch-aware HVGs, compare unintegrated/Harmony, cluster.
python scripts/03_preprocess_cluster.py results/02_qc_filtered.h5ad

# 4. Marker-guided annotation draft; CellTypist is off by default.
python scripts/04_annotate.py results/03_clustered.h5ad

# Review marker tables, fill metadata/cluster_labels.tsv, then rerun stage 4.
python scripts/04_annotate.py results/03_clustered.h5ad --run-celltypist

# 5. Sample-level pseudobulk DE and gene rescue classification.
python scripts/05_pseudobulk_de.py results/04_annotated.h5ad

# 6. Custom ovarian pathway GSEA and pathway rescue summary.
python scripts/06_pathway_rescue.py
```

## Checkpoints

- `results/01_merged_counts.h5ad`
- `results/02_qc_annotated.h5ad`
- `results/02_qc_filtered.h5ad`
- `results/03_clustered.h5ad`
- `results/04_annotated.h5ad`
- `results/05_pseudobulk/`
- `results/06_pathways/`

Integrated embeddings are used only for graph construction, visualization, and
clustering. Differential expression always uses the untouched integer UMI layer
`counts`, aggregated by library and curated cell type.
