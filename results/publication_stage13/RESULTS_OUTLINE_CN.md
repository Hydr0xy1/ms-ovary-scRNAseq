# 论文Results结构与证据索引

## Result 1 A single-cell atlas reveals cell-type-specific remodeling during ovarian aging.

主结论：9个library构成可重复的卵巢单细胞图谱，衰老效应具有cell-type specificity。图：Figure 1独立atlas图；Figure 2 aging summary。数据：`results/06_annotation_v2.h5ad`、`results/composition/library_broad_fractions.tsv`、`results/de_stage1/broad_de_summary.tsv`。统计：library-level pseudobulk；n=3/group。限制：cell数量不是生物学重复数。

## Result 2 MRJP1 treatment is associated with cell-type-specific reversal of aging-related transcriptional changes.

主结论：Granulosa和Stromal的aging-treatment effect最稳定地反向。图：Figure 3 reversal scatters、permutation comparison和aging-axis plots。数据：`results/de_stage1_6/*/observed_evidence_levels.tsv.gz`、`population_reversal_evidence.tsv`、`stage5_rejuvenation_geometry/aging_axis_projection.tsv`。统计：Spearman、cosine、20 exact assignments。限制：只称transcriptomic reversal/young-directed projection。

## Result 3 Stromal fibroblasts show the strongest and most reproducible program-level reversal.

主结论：Stromal有12条strong pathways、广泛亚型定位和9/12外部方向支持。图：Figure 4 standalone pathway、subtype、external、regulatory和NMF图。数据：`hallmark_pathway_evidence.tsv`、`broad_to_subtype_localization.tsv`、`external_strong_pathway_validation.tsv`、`candidate_regulators.tsv`、`program_reversal.tsv`。统计：GSEA FDR、exact permutation ranks、Spearman/cosine。限制：pathway activity不等于生化功能。

## Result 4 Granulosa responses are localized to defined cellular states and are primarily cell-intrinsic.

主结论：Granulosa Protein secretion及相关程序定位于antral/preantral-like状态。图：Figure 5 standalone subtype、pathway、decomposition、regulatory和NMF图。数据：Stage 2-4、7-8 TSV。统计：subtype pseudobulk GSEA与linear CPM Shapley decomposition。限制：最强Granulosa通路没有外部通路方向复现。

## Result 5 MRJP1-associated responses are driven predominantly by within-subtype transcriptional changes rather than compositional restoration.

主结论：Granulosa、Stromal和Immune均以within-subtype effect为主。图：`composition_intrinsic_decomposition.svg`。数据：`results/stage4_composition_decomposition/decomposition_summary.tsv`。统计：two-factor exact Shapley decomposition in linear CPM space。限制：projection fraction不是causal percentage。

## Result 6 Independent aging references support a substantial portion of the stromal aging/reversal signature.

主结论：Stromal显著衰老基因Spearman约0.645，9/12强通路获得方向支持。图：`external_pathway_validation.svg`。数据：Stage 6 concordance/pathway validation TSV。统计：Spearman、cosine、direction concordance。限制：外部参考不是MRJP1治疗重复。

## Result 7 Regulatory and ligand-target analyses nominate candidate mechanisms linking stromal and granulosa responses.

主结论：PI3K/Arnt/Hoxa5及IL6/FGF2/BDNF/BMP4形成可测试假说。图：Figure 4/5 regulatory standalone图与Figure 6 ligand hypothesis图。数据：Stage 7-9 TSV。统计：activity permutation、表达门槛、ligand-target支持。限制：均为hypothesis-generating，不证明直接机制。
