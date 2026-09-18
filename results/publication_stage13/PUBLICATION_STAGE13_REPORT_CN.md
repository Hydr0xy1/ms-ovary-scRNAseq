# Stage 13 论文级结果收敛报告

## 1 当前项目状态

Stage 1-12已完成。本阶段没有重跑Harmony、UMAP、DE、GSEA、注释或改变任何统计阈值，只整理已有证据并生成独立单图。所有正式比较仍以library为生物学重复（每组n=3）。

## 2 最强结论

Granulosa与Stromal_fibroblast是最稳定的MRJP1-associated transcriptomic reversal populations。Stromal证据链更完整：broad exact permutation、12条strong Hallmark、subtype定位、外部衰老一致性、regulatory activity与NMF相互支持。

## 3 证据等级

- LEVEL A：当前数据直接支持的观察结果。
- LEVEL B：有统计支持的转录组推断。
- LEVEL C：多层结果支持的机制假说。
- LEVEL D：必须经过功能实验验证的具体机制。

详见`MASTER_EVIDENCE_LEDGER.tsv`。

## 4 Figure 1-7设计

Figure 1-7的科学问题与panel选择已经完成，但遵照用户要求，没有输出任何拼接组图。当前生成22个独立、单轴、可追溯图，每个图均有SVG、PDF和600 dpi PNG。

## 5 Supplementary设计

补充材料按QC、批次、注释、pseudobulk、置换、Hallmark、亚型、分解、几何、外部验证、调控、NMF、通讯和负结果组织为S1-S17。原有223张图不删除，统一登记在`FIGURE_INVENTORY.tsv`。

## 6 Stromal主线

Stromal的aging-treatment全转录组Spearman为-0.763，真实标签rank 1/20；12条strong Hallmark覆盖炎症、TNFA/NFkB、OXPHOS、protein secretion、apoptosis和DNA repair等程序。外部参考中9/12强通路方向一致且治疗方向相反。PI3K、Arnt和Hoxa5及NMF程序提供机制假说，但不是直接机制证据。

## 7 Granulosa主线

Granulosa的aging-treatment全转录组Spearman为-0.740，真实标签rank 1/20；1条strong Hallmark为Protein secretion，并定位到antral/preantral-like状态。显著衰老基因有外部一致性，但最强Protein secretion通路在外部数据中方向不一致，必须保留这一限制。

## 8 Composition vs intrinsic

Granulosa、Stromal和Immune的总体变化主要由within-subtype expression change构成。这里使用linear CPM Shapley双因素分解；projection fraction是模型空间的几何投影，不是因果贡献百分比。

## 9 External validation

内部与外部显著衰老LFC的Spearman约为Granulosa 0.615、Stromal 0.645。外部数据提升了Stromal主线的可信度，但不是MRJP1处理的外部复现。

## 10 Regulatory/NMF

Stromal PI3K、Arnt、Hoxa5和Granulosa PI3K为优先activity hypotheses。两条lineage均选择K=5，10个程序中9个方向反向并更接近年轻组。activity和co-expression均不能证明直接靶点或因果通路。

## 11 Communication hypotheses

IL6、FGF2、BDNF、BMP4只作为Stromal->Granulosa定向通讯假说；现有结果不能证明分泌、受体激活或跨细胞因果链。

## 12 Candidate shortlist

60条候选被透明压缩为Tier A=9、Tier B=16、Tier C=35。Tier A为：Il6st, Hif1a, Fn1, Tnc, Zfpm2, Prkdc, Bard1, Abi1, Abca1。分层使用明确证据门槛和人工可解释锚点，没有加权黑箱分数。

## 13 Experimental priorities

Tier 1优先验证Stromal炎症/TNFA-NFkB/OXPHOS/protein secretion及Granulosa protein secretion和cell-state response；Tier 2验证PI3K、Arnt、Hoxa5以及四个通讯候选；Tier 3补充AMH/FSH/E2、卵泡计数、ROS/MDA/SOD/GSH-Px/ATP/MMP、p16/p21/gamma-H2AX、Ki67和TUNEL。

## 14 Negative results

Theca缺少稳定population-level exact permutation支持；Granulosa最强Hallmark缺少外部通路方向复现；次级population没有与Granulosa/Stromal相当的多终点证据；当前没有matched phenotype。详见`NEGATIVE_RESULTS_LEDGER.tsv`。

## 15 Statistical limitations

每组只有3个library；20种exact assignments使经验p值最小为0.05；external atlas存在年龄、平台和预处理差异；cell-level数量不等于生物学重复数。

## 16 推荐论文叙事

自然卵巢衰老伴随cell-type-specific transcriptional remodeling。MRJP1处理与部分衰老相关程序的反向移动相关，主要集中在Stromal fibroblast和Granulosa，并主要反映亚型内表达变化。外部衰老数据更强地支持Stromal主线；调控、NMF和通讯结果用于提出后续机制假说。

## 17 仍缺少哪些实验

缺少同一动物匹配的卵巢储备、激素、线粒体/氧化应激、炎症、衰老、凋亡和生育力数据，也缺少候选调控因子和配体-受体的扰动实验。

## 18 下一步建议

先人工审核独立图和Tier A，再完成最小正交验证。未经功能数据支持，不使用“rejuvenates”“restores youth”“reverses ovarian aging”或确定性通路/通讯机制表述。
