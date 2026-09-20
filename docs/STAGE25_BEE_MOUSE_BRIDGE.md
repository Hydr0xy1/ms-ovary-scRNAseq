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
