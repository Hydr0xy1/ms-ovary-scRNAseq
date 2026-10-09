# 颗粒细胞和成纤维细胞专门解析

本轮目的：拆开既有细胞周期变化的来源，并与冻结的 pseudobulk/GSEA、状态模块证据对照；不是重新进行常规单细胞分析。

## 范围与分析约定

- 统计单位是独立 library/pool，Y、OC、OT 各 3 个；不将细胞或基因作为生物学重复。
- 复用 Stage26 已有周期分数、v2 注释/质量 Tier、Stage14 亚型组成及标记、Stage13 Hallmark 和 Stage15 状态模块。
- 主对象只读；不删除细胞、不重新标准化/聚类、不改注释、不开展新的 DEG。
- 所有已注释细胞是描述性全景；Tier1_primary 是更严格的敏感性视图，不能把两者互换。
- Hallmark必须读取完整冻结结果（含未显著条目），不能用仅包含已选反向通路的论文图源代替。服务器两张小表缓存到新阶段input_cache，不改变历史目录。
- 颗粒细胞另比较排除 low-complexity、排除已用周期标记注释的 cycling 亚型以及同时排除两者。
- 基质另比较 fibroblast+ECM、fibroblast_candidate、ECM_high_candidate；不把 smooth-muscle 候选直接当成纤维细胞。
- 0、0.05、0.10、0.20 的最大周期分数用于弱信号敏感性诊断，不是新 phase 定义或最终过滤阈值。
- 亚型标准化只用每库至少 10 个细胞的共同亚型，固定权重取 9 库亚型比例的等权均值。报告共同亚型覆盖度。
- 两组精确置换沿用 Stage25 穷举方法；3 对 3 共 20 种分配，双侧最小 P 为 0.10。BH 在 family×metric×contrast 内跨 population/view 校正。LOO 只检查方向，不能增加生物学重复。
- 组成/状态分解为对称描述性分解，不证明细胞自主改变、因果比例或选择性存活。

## 图件约定（nature-figure）

- 暂定结论：颗粒与基质的周期变化具有不同亚型和功能来源，不能简单等同于 MRJP1 增殖恢复。
- 类型：单张定量图，不拼接组图；Python/matplotlib 为唯一绘图后端。
- 证据链：每库亚型组成 → 每库周期敏感性 → 既有 pathway 和功能模块。
- 每张约 183 mm 宽；SVG 可编辑文字、PDF TrueType、600 dpi PNG；逐张输出 source data。
- 不用星号暗示本轮文库级显著性；Hallmark 图只注明既有 GSEA 的 FDR 支持，不代表新置换结果。
- 主要风险：n=3、解离捕获偏倚、发情阶段未知、低复杂度弱分数、周期注释与周期评分共享基因、G0/G1 与分裂/阻滞不能由 RNA phase 区分。
- 共10张独立图（每群体5张）：亚型组成、周期分层、弱分数敏感性、完整预定义Hallmark、已有状态模块。基质的罕见Unresolved亚型保留在结果/图源表，不用于拉伸主图色阶。
- 绘图前检查文字在画布内；短热图的标题/分组标题重叠有回归测试。SVG保留text元素，PNG为600 dpi。

## 运行

服务器只导出小型元数据（不载入 X/raw/counts）：

```powershell
python scripts/44_stage27_granulosa_stromal.py --export-obs results/06_annotation_v2.h5ad
```

将输出 QC_METADATA.tsv.gz 放在本地对应结果目录，再运行：

```powershell
python scripts/44_stage27_granulosa_stromal.py
python scripts/45_stage27_focused_figures.py
```

报告、表格和图件分别存放在独立 stage27 结果/图件目录中；大型输出不进入 Git。
