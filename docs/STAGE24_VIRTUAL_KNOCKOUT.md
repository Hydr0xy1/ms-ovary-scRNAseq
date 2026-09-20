# Stage 24 virtual-knockout pilot

Stage 24 is isolated from the previous analysis and never modifies
`results/06_annotation_v2.h5ad`. It uses the raw UMI counts in `layers["counts"]`
to prepare a bounded OC Granulosa gene universe, then performs CPM normalization
before scTenifold network construction.

## Biological scope

- Reference state: OC Granulosa cells.
- Primary targets: `Hif1a`, `Smad3`, `Igfbp2`, `Pak3`, and `Abca1`.
- Benchmark: one uniquely selected expression-matched reference gene per target.
- Primary analysis: 300 Tier1 cells per OC library, three sampling seeds.
- Sensitivity analysis: 1,200 Tier1+Tier2 cells per OC library, three seeds.
- Library robustness: three leave-one-OC-library-out networks.
- Network size: 3,000 frozen genes; 10 PC-regression networks per run.

Cells are balanced before each network is constructed so a large library cannot
dominate the model. A WT network is constructed once per scenario and seed, then
reused for every virtual knockout in that run.

## Environment setup

```bash
cd /root/autodl-tmp/ovary_scRNAseq
bash environment/bootstrap_stage24.sh
bash environment/bootstrap_stage24_r.sh
```

The Python and R environments are placed under `/root/autodl-tmp/envs`, not on
the smaller system disk. The official R implementation is used for a focused
`Hif1a` cross-implementation check before the multi-seed Python queue.

## Run

```bash
/root/autodl-tmp/envs/ovary_stage24/bin/python \
  scripts/35_stage24_virtual_knockout.py
```

Individual resumable steps can be run with `--steps`, for example:

```bash
/root/autodl-tmp/envs/ovary_stage24/bin/python \
  scripts/35_stage24_virtual_knockout.py --steps prepare r_crosscheck
```

Every scenario/seed stores its WT tensor and each KO result independently.
Completed units are skipped when the command is restarted.

## Outputs

- `results/deep_dive_stage24_virtual_knockout/00_preparation/`: input audit,
  candidate coverage, frozen gene universe, and matched references.
- `results/deep_dive_stage24_virtual_knockout/01_runs/`: sampled-cell manifests,
  WT tensor checkpoints, and per-gene perturbation rankings.
- `results/deep_dive_stage24_virtual_knockout/02_summary/`: multi-seed stability,
  pathway enrichment, real-transcriptome association, and LOO sensitivity.
- `results/deep_dive_stage24_virtual_knockout/03_r_crosscheck/`: official R versus
  Python implementation comparison.
- `figures/deep_dive_stage24_virtual_knockout/`: independent Nature-style figures
  plus source-data tables; no assembled multi-panel figure is generated.

## Interpretation boundary

scTenifold perturbation distance is unsigned. Its model-internal gene p-values
do not use the three libraries as biological replicates. Results therefore rank
candidate mechanisms and experimental follow-ups, but do not establish that a
gene mediates MRJP1 action and do not substitute for a real knockout experiment.
