# Stage 25: bee-inspired functional resilience bridge

This independent stage reads the frozen annotated mouse object and Stage 24 outputs.
It never changes H5AD files or Stage 1–24 results. Formal calculations run only on the
main mouse server. Project A contributes small frozen evidence/mapping tables.

## Prespecified analysis contract

- Biological unit: library/pool, Y/OC/OT with three independent units per group.
- Populations: Granulosa and Stromal_fibroblast. Existing subtype and Tier labels are reused.
- Cell Tier1 is the primary analysis; Tier1+Tier2 is a sensitivity analysis. Source evidence
  tiers are separate fields and must never be confused with mouse cell tiers.
- Eight functional modules: mitochondrial_energy, RNA_processing_translation,
  proteostasis_autophagy, lipid_sterol_redox, membrane_structure_transport,
  ECM_stromal_support, stress_inflammation, reproductive_support_secretion.
- Gene membership is frozen using Project A evidence and annotation rules before inspecting
  mouse treatment effects. The table records exact source rows and annotation matches.
- Direct bee evidence, empirical mammalian readouts, and hypothesis-only entries are distinct.
  The primary bridge excludes hypothesis-only entries. Separate source strata test whether
  a result actually derives from bees. Membership does not imply conserved regulation.
- Primary mapping requires an unambiguous one-to-one chain. Bee→fly→human→mouse and
  Bombus→Apis annotation bridges are explicitly marked indirect. Retain all alternative
  mappings in the audit; never select the first hit or manufacture unique feature names.
- Human–mouse mapping uses the frozen reciprocal-unique NCBI table. Entries absent from
  this strict table are unmapped in the frozen reference, not proven evolutionary absences.
- An existing reciprocal protein-best-hit reference is evaluated separately as
  `bee_RBH_sensitivity`. Its 1:1 cardinality is algorithmic and is not equated with
  evolutionarily proven one-to-one orthology. This reference was added after the strict
  mapping-coverage audit and before inspection of any mouse effect estimates.

## Library inference

Use library pseudobulk log-CPM module means, plus library means, medians and 90th percentiles
of per-cell log-normalized module scores as sensitivity readouts. Cells are never used as
independent inferential replicates. Equal gene weighting is used within modules and equal
library weighting between libraries. Require at least three evaluable genes for a module.

OC−Y, OT−OC and OT−Y use exhaustive two-sided label permutation (20 assignments for 3+3;
minimum attainable two-sided P=0.10). BH correction is reported separately from descriptive
support. Report every library omission, pairwise direction consistency and Tier/center/tail
sensitivity. Missing age significance does not demonstrate orthogonality.

Age/treatment geometry is computed in each module's multi-gene library pseudobulk space:
age A=OC−Y, treatment T=OT−OC, T_parallel=(T·A/A·A)A, T_orthogonal=T−T_parallel.
Report cosine, projected fraction and orthogonal norm fraction; assess library sensitivity
by recomputing all group means after each library omission. These estimates share OC and
may have regression-to-mean bias; a split-OC descriptive sensitivity separates the OC unit
used to estimate A from those used to estimate T. The phrase protective is not inferred
from orthogonality. Existing subtype labels support composition-standardized sensitivity.

## Virtual knockout comparison

Use stored Stage 24 rankings and WT tensors; do not construct new networks. Test strict
all-seed intersection and >=2/3-seed recurrent top-5% hits, individual seeds, both cell Tiers,
and three existing leave-one-OC-library-out networks. Exclude each knocked-out node from
its own enrichment universe. Limit enrichment to genes modeled in the corresponding run.

Compare candidates with their frozen expression-matched references. Use seeded random
gene sets matched jointly on OC mean expression, detection fraction, WT weighted degree
and edge degree; export matching diagnostics and all null summaries. Random-set P values
are competitive model statistics, not library-level biological P values. Unsigned perturbation
distances cannot predict up/down transcript changes. Compare real age/treatment directions
only as observed transcriptomic readouts and mark direction as unavailable for VKO itself.

## Figures and stop rule

Every figure is standalone (no panel assembly), with SVG, PDF, PNG and source data.
The visual evidence must show mapping attrition, individual libraries, treatment–age geometry,
and candidate specificity/robustness. Editable SVG text and embedded PDF fonts are required.
The plot backend will follow the user's choice. No new public expression datasets or new
machine-learning models are introduced. Stop after the requested report, ledger, model and QA.

## Execution and audit notes

The audited existing project plotting workflow is Python/matplotlib; this backend is used
for every figure and exported preview. Formal entry points:

```bash
cd /root/autodl-tmp/ovary_scRNAseq
/root/autodl-tmp/envs/ovary_stage24/bin/python scripts/38_stage25_bee_mouse_bridge.py
/root/autodl-tmp/envs/ovary_stage24/bin/python scripts/39_stage25_report.py
```

Human empirical readouts trace to GSE202601 fibro-like states (verified against the
upstream run configuration), not to a universal mammalian ageing signature. The Bombus
source table lacks an explicit species accession; genus-level `Bombus_sp` is retained
instead of inventing a species. The archive is sufficient as the bee source; no task
or data migration is submitted to the old bee server.

Geometry uses |cosine|<=0.30 for a descriptive orthogonal classification, negative cosine
below -0.30 plus opposite scalar effects for age_opposite, and positive cosine above 0.30
plus concordant scalar effects for age_parallel. The geometric label is library_sensitive
when fewer than 7/9 library omissions retain its class or fewer than 8/9 retain treatment
direction. Synthesis is stricter: any sign flip in the six contrast-specific omissions
triggers library_sensitive, and Tier or cell-center direction disagreement triggers
inconsistent. Split-OC sensitivity remains explicit rather than being treated as proof.

VKO random controls use 2,000 draws from each module gene's 40 nearest eligible neighbors
in standardized log-expression, detection, log-weighted-degree and log-edge-degree space.
Each draw samples without replacement and excludes the tested module genes. All ten KO
nodes (five candidates, five references) are excluded from the common background. A
comparison fails matching quality when the largest absolute mean standardized feature
difference exceeds 0.25. Null histograms and all diagnostic values are exported.

`MODULE_SYNTHESIS.tsv` integrates the primary and sensitivity readouts; `VKO_CANDIDATE_ROBUSTNESS.tsv`
links individual seeds, cell Tiers and OC-library omissions. Coupling and technical-depth
correlations use nine library summaries descriptively and do not create new cell-level tests.

The final report must retain null and non-evaluable findings. In particular, a stable
Stage24 node is not automatically a Stage25 module-specific node, and lack of gene coverage
is not evidence that a biological function is absent.
