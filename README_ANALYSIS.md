# Mouse ovary scRNA-seq analysis scaffold

Biological groups:

- `Y`: 4-month vehicle control
- `OC`: 10-month vehicle control
- `OT`: 10-month MRJP1, 200 mg/kg

Primary contrasts:

- `OC vs Y`: reproductive-aging effect
- `OT vs OC`: MRJP1 treatment effect
- `OT vs Y`: residual deviation from the young state

## Resource policy

The current no-card container is limited to 2 GiB RAM and 0.5 CPU. Only the
commands marked **NO-CARD SAFE** should be run in that mode. Heavy scripts have
a cgroup-memory guard and will refuse to run until compute mode is enabled.

Recommended compute mode: at least 64 GiB RAM and 8-16 CPU cores.

## Before analysis

Edit `config/sample_metadata.tsv` and replace every `TODO` or `unknown` value
that can be recovered. In particular, record the mouse IDs contributing to each
library. If four mice were pooled into each library, the formal scRNA-seq sample
size is three pools per group, not twelve mice per group.

## NO-CARD SAFE commands

```bash
source /root/autodl-tmp/ovary_scRNAseq/config/activate.sh

# Syntax only
python -m py_compile scripts/*.py

# Input dimensions, barcode/feature consistency, and feature hash
python scripts/00_validate_inputs.py

# Optional full CRC check; safe but slow with 0.5 CPU
python scripts/00_validate_inputs.py --full-gzip-test

# Tiny synthetic end-to-end test; never opens the real matrices
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 python scripts/99_smoke_test.py
```

## COMPUTE MODE commands

Run each stage separately and review its output before continuing.

```bash
source /root/autodl-tmp/ovary_scRNAseq/config/activate.sh

# 1. Merge all libraries and preserve raw counts
python scripts/01_build_merged.py

# 2a. Annotate QC failures without deleting cells
python scripts/02_qc.py results/01_merged_counts.h5ad

# Review results/02_qc_summary.tsv and figures/02_qc_before_filtering.png.
# Adjust config/analysis_config.yaml if necessary, then apply filters:
python scripts/02_qc.py results/01_merged_counts.h5ad --apply-filter

# 3. Normalize, select batch-aware HVGs, compare unintegrated/Harmony, cluster
python scripts/03_preprocess_cluster.py results/02_qc_filtered.h5ad

# 4. Marker-guided annotation draft; CellTypist is off by default
python scripts/04_annotate.py results/03_clustered.h5ad

# Review marker tables, fill config/cluster_labels.tsv, then rerun stage 4.
# Optional immune-only CellTypist pass:
python scripts/04_annotate.py results/03_clustered.h5ad --run-celltypist

# 5. Sample-level pseudobulk DE and gene rescue classification
python scripts/05_pseudobulk_de.py results/04_annotated.h5ad

# 6. Custom ovarian pathway GSEA and pathway rescue summary
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
`counts`, aggregated by library and final cell type.
