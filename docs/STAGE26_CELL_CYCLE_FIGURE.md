# Stage 26 cell-cycle figure and literature context

## Figure contract

- Core conclusion: cycling transcriptional states are concentrated in Granulosa cells; apparent condition shifts must be interpreted at the level of the nine independent libraries rather than by treating cells as replicates.
- Archetype: single-panel quantitative comparison.
- Backend: Python/matplotlib only.
- Final size: 183 x 120 mm.
- Hero evidence: S + G2M fraction for each broad cell type in each library.
- Robustness: all available libraries are displayed; diamonds are unweighted group means and lines are observed library ranges.
- Statistics: descriptive only; `n=3` independent library/pools per experimental group. No per-cell hypothesis test is shown.
- Reviewer risks: phase is inferred from transcriptomic S/G2M modules; captured-cell fractions are not absolute in situ proliferation rates; several rare lineages are absent from individual libraries; oocytes, Mixed and uncertain labels are excluded from this mitotic comparison.

## Literature search

Search date: 2026-10-09. PubMed was searched first and key records were verified against Crossref using DOI. This was a targeted contextual search, not a systematic review.

1. Isola JVV et al. *A single-cell atlas of the aging mouse ovary*. Nature Aging (2024). PMID: 38200272. DOI: [10.1038/s43587-023-00552-5](https://doi.org/10.1038/s43587-023-00552-5).
   - Supports cell-type-resolved interpretation of ovarian aging and reports stress, immunogenic and fibrotic signaling changes in follicular cells.
2. Wang S et al. *Single-Cell Transcriptomic Atlas of Primate Ovarian Aging*. Cell (2020). PMID: 32004457. DOI: [10.1016/j.cell.2020.01.009](https://doi.org/10.1016/j.cell.2020.01.009).
   - Supports Granulosa as an aging-sensitive somatic compartment and links aging to oxidative stress and apoptosis, not simply proliferation loss.
3. Wu M et al. *Spatiotemporal transcriptomic changes of human ovarian aging and the regulatory role of FOXP1*. Nature Aging (2024). PMID: 38594460. DOI: [10.1038/s43587-024-00607-1](https://doi.org/10.1038/s43587-024-00607-1).
   - Provides a plausible cell-cycle-arrest link through the age-associated FOXP1–CDKN1A axis, but does not establish that MRJP1 acts through this pathway.
4. Zhang J et al. *A multi-omic single-cell landscape of the aging mouse ovary*. GeroScience (2025). PMID: 39934558. DOI: [10.1007/s11357-025-01556-2](https://doi.org/10.1007/s11357-025-01556-2).
   - Reinforces cell-type specificity and identifies endoplasmic-reticulum stress in aged Granulosa cells.
5. Johnson J, Emerson JW & Lawley SD. *Recapitulating human ovarian aging using random walks*. PeerJ (2022). PMID: 36032944. DOI: [10.7717/peerj.13941](https://doi.org/10.7717/peerj.13941).
   - Motivates the biological relevance of pre-Granulosa cell-cycle entry and stress-responsive checkpoint control during follicle activation.
6. Satija R et al. *Spatial reconstruction of single-cell gene expression data*. Nature Biotechnology (2015). PMID: 25867923. DOI: [10.1038/nbt.3192](https://doi.org/10.1038/nbt.3192).
7. Tirosh I et al. *Dissecting the multicellular ecosystem of metastatic melanoma by single-cell RNA-seq*. Science (2016). PMID: 27124452. DOI: [10.1126/science.aad0501](https://doi.org/10.1126/science.aad0501).
   - The Satija/Tirosh gene-set framework underlies widely used transcriptomic S/G2M scoring; it assigns a transcriptional state and is not a direct assay of DNA synthesis or mitosis.

## Interpretation for the current dataset

- Granulosa has the largest library-resolved S + G2M fraction among curated broad ovarian cell types.
- OC, OT and Y differences remain variable across the three libraries; the figure therefore makes no claim of a statistically resolved MRJP1-induced proliferation change.
- A defensible conclusion is that MRJP1-treated ovaries retain a Granulosa-enriched cycling transcriptional compartment. EdU/BrdU, Ki67 and/or DNA-content measurements would be required to establish altered proliferation in situ.

## How related studies present cell-cycle evidence

- The goat ovarian-aging atlas (PMCID: PMC10699009, Fig. 3D) displays cell-cycle phase on a t-SNE embedding and separately reports the proportion of Granulosa cells in each phase. The essential quantitative component is therefore phase composition, not a score-range scatter plot.
- The human ovarian-aging/FOXP1 study (PMCID: PMC11031396, Fig. 6 and Extended Data Fig. 8) does not treat transcriptomic phase assignment as sufficient evidence of proliferation. It supports the mechanism using EdU incorporation and Ki67 immunofluorescence.
- The aging mouse-ovary atlas (PMCID: PMC10798902, Fig. 6) presents cell-type-specific pathway or upstream-regulator changes and then adds protein/imaging validation. This supports separating transcriptomic localization from functional validation.

Accordingly, the revised presentation uses two independent figures:

1. A 100% stacked bar chart of G1/S/G2M composition in every Granulosa library. This is the primary figure because it shows the biological replicate and the inferred phase directly.
2. A cell-type-by-library heatmap of S+G2M percentages. This is the atlas overview and makes the Granulosa localization immediately visible.

The earlier range-scatter figure is retained only for provenance and is no longer the recommended main display.
