"""Conservative, file-grounded synthesis for the Stage 15 deep dive."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .project import project_paths, setup_logging


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t", compression="infer")
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def run_stage15_synthesis(config: Mapping[str, Any]) -> Path:
    paths = project_paths(dict(config))
    settings = config["deep_dive_stage15"]
    root = paths["root"]
    output_root = root / settings["output_dir"] / "synthesis"
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("29_stage15_synthesis", dict(config))
    stage_root = root / settings["output_dir"]

    status = _read_json(stage_root / "status_audit" / "CHECKPOINT.json")
    external = _read_json(stage_root / "external_gse267729" / "CHECKPOINT.json")
    projection = _read_json(stage_root / "external_projection" / "CHECKPOINT.json")
    program = _read_json(stage_root / "program_stability" / "CHECKPOINT.json")
    state = _read_tsv(stage_root / "state_audit" / "state_effect_summary.tsv")
    projection_summary = _read_tsv(stage_root / "external_projection" / "external_age_projection_summary.tsv")
    stability = _read_tsv(stage_root / "program_stability" / "program_seed_stability.tsv")
    loo = _read_tsv(stage_root / "program_stability" / "program_leave_one_library_projection.tsv")

    lines = [
        "# 小鼠卵巢衰老/MRJP1：Stage 15 深入挖掘阶段性汇总",
        "",
        "本报告只汇总已经写入文件的结果，不重新进行常规 QC、聚类、基础注释、普通 DEG、普通 GO/KEGG/GSEA，也不修改主 H5AD。正式统计单位仍为 library / biological pool（每组 n=3）。",
        "",
        "## A. 已完成的分析",
        "",
        f"- 项目状态审查：{status.get('status', '未找到完成标记')}；输入对象 shape={status.get('input_shape', 'NA')}。",
        f"- GSE267729 外部矩阵与外部程序：{external.get('status', '尚未完成')}。",
        f"- 外部年龄轴投影：{projection.get('status', '尚未完成')}。",
        "- 多维状态模块审计：已独立计算 p53/p21、DNA damage、SASP/inflammation、apoptosis、mitochondria/OXPHOS、ROS、follicle maturation、atresia、steroidogenesis、ECM/fibrosis；没有合成总衰老分数。",
        f"- 潜在程序稳定性：{program.get('status', '尚未完成')}。",
        "",
        "## B. 关键结果",
        "",
    ]
    if not state.empty and {"population", "module", "metric", "closer_to_young"}.issubset(state.columns):
        for (population, module, metric), sub in state.groupby(["population", "module", "metric"], observed=True):
            if metric != "mean":
                continue
            closer = bool(pd.Series(sub["closer_to_young"]).iloc[0])
            direction = "是" if closer else "否"
            lines.append(
                f"- {population} / {module}：按library均值汇总，OT相对OC是否更接近Y={direction}；"
                "这只是状态描述，不等同于功能恢复。"
            )
    else:
        lines.append("- 状态效应表尚未可用。")
    if not projection_summary.empty:
        for row in projection_summary.itertuples(index=False):
            lines.append(
                f"- 外部轴 {row.population}/{row.external_contrast}：OT-OC轴差={_fmt(row.treatment_minus_aging_axis)}，"
                f"OT相对OC的Y距离变化={_fmt(row.distance_change_OT_vs_OC)}；需同时看方向和距离。"
            )
    else:
        lines.append("- 外部年龄轴投影摘要尚未可用。")
    if not stability.empty and {"population", "k", "mean_component_cosine"}.issubset(stability.columns):
        best = stability.groupby(["population", "k"], observed=True)["mean_component_cosine"].mean().reset_index()
        for row in best.itertuples(index=False):
            lines.append(f"- 程序稳定性 {row.population}, K={row.k}：跨种子平均component cosine={_fmt(row.mean_component_cosine)}。")
    if not loo.empty and {"population", "k", "held_out_rmse"}.issubset(loo.columns):
        best_loo = loo.groupby(["population", "k"], observed=True)["held_out_rmse"].median().reset_index()
        for row in best_loo.itertuples(index=False):
            lines.append(f"- 留一library投影 {row.population}, K={row.k}：RMSE中位数={_fmt(row.held_out_rmse)}。")

    lines.extend(
        [
            "",
            "## C. 对 MRJP1 作用的最高强度结论",
            "",
            "在当前数据和 n=3/group 的统计分辨率下，最高强度结论只能是：MRJP1 与部分 Granulosa/Stromal 细胞状态的转录重塑相伴，部分指标可能沿外部或内部年龄方向回移；是否接近年轻状态、是否主要为细胞内改变、以及是否存在治疗特异的正交成分，必须分别依据投影、距离、组成分解和稳定性结果判断。不能将反向表达直接称为年轻化或功能恢复。",
            "",
            "## D. 仍然只是机制假说的部分",
            "",
            "TF活性、NMF程序、配体–受体/通讯以及候选细胞间链条只属于机制假说；它们需要受体表达、下游靶程序和实验验证共同支持，不能单凭算法排名表述为因果机制。",
            "",
            "## E. 与预期不一致或不支持的结果",
            "",
            "- Theca方向逆转不能因为比例高就视为稳定响应；前序置换支持较弱。",
            "- SASP、ROS、apoptosis等模块不必同步下降；若只改变高分位数尾部，也不能解释为整体衰老逆转。",
            "- 元数据中 batch、estrous_stage、pool_mouse_ids 缺失时，当前数据不能区分周期、批次、pool内动物和选择性存活解释。",
            "",
            "## F. 最稳定的候选程序、TF或配体–受体链",
            "",
            "本报告不在缺少稳定性门槛时强行指定候选。只有跨library、跨随机种子并且留一library投影可接受的程序，才可进入后续调控/微环境验证；现有 Stage 7–9 候选仍应按机制假说层处理。",
            "",
            "## G. 需要实验验证的优先级",
            "",
            "1. Granulosa 与 Stromal 中最稳定的年龄/治疗相关程序及其代表蛋白；2. DNA damage、ECM/fibrosis、炎症和线粒体表型的组织学/生化验证；3. 只有在配体–受体–靶基因链通过稳定性门槛后，才验证少数旁分泌候选；4. 若条件允许，补充动情周期、批次和pool内动物元数据。",
            "",
            "## H. 失败、未完成及其原因",
            "",
            f"- 外部下载/解压完整性以下载日志为准；SCP1914 当前不可可靠获取，GSE202601 仅作为跨物种支持。",
            f"- 外部阶段状态：{external.get('status', '未完成')}；投影状态：{projection.get('status', '未完成')}；程序稳定性状态：{program.get('status', '未完成')}。",
            "",
            "## I. 结果文件和日志位置",
            "",
            "- `results/deep_dive_stage15/status_audit/`：元数据与项目状态审查",
            "- `results/deep_dive_stage15/state_audit/`：多维状态模块结果",
            "- `results/deep_dive_stage15/external_gse267729/`：外部矩阵、注册表、外部年龄程序和下载日志",
            "- `results/deep_dive_stage15/external_projection/`：年龄轴投影和距离摘要",
            "- `results/deep_dive_stage15/program_stability/`：NMF稳定性、top genes和留一library投影",
            "",
            "## J. 下一次继续运行时应从哪个阶段开始",
            "",
            "先审核本报告与各阶段 CHECKPOINT；若程序稳定性不足，停止扩展 NMF，将主线保持在外部年龄轴、组成/内在分解和状态模块；若稳定性通过，再仅对稳定程序进行验证性 TF/微环境整合。",
        ]
    )
    report_path = output_root / "DEEP_DIVE_STAGE15_REPORT_CN.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    checkpoint = {
        "status": "SYNTHESIS_COMPLETE",
        "inputs": {
            "external": bool(external),
            "projection": bool(projection),
            "state_audit": not state.empty,
            "program_stability": bool(program),
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "report": str(report_path.relative_to(root)),
        "next_stage": "manual_review_then_validation_followup",
    }
    (output_root / "CHECKPOINT.json").write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Stage 15 synthesis written: %s", report_path)
    print("STAGE15_SYNTHESIS_COMPLETE")
    print(f"OUTPUT={output_root.relative_to(root)}")
    return output_root
