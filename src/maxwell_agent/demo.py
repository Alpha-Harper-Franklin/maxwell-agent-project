from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from .agent import MaxwellAgent
from .models import (
    CaseDeliveryReport,
    ElectromagnetDesign,
    RequirementEvaluation,
    RequirementIntake,
    SimulationResult,
)


ProgressCallback = Callable[[int, str], None]


STATUS_LABELS = {
    "completed": "已完成",
    "blocked": "已阻塞",
    "failed": "失败",
}

EVALUATION_STATUS_LABELS = {
    "passed": "已满足",
    "failed": "未满足",
    "partial": "部分验证",
    "unverified": "待验证",
}

DESIGN_LABELS = {
    "summary": "需求解释",
    "objective": "优化目标",
    "size_preference": "尺寸偏好",
    "supply_voltage_v": "供电电压 (V)",
    "current_min_a": "电流下限 (A)",
    "current_limit_a": "电流上限 (A)",
    "target_force_n": "目标吸力 (N)",
    "air_gap_mm": "气隙 (mm)",
    "current_a": "电流 (A)",
    "coil_turns": "线圈匝数",
    "core_width_mm": "铁芯宽度 (mm)",
    "core_height_mm": "铁芯高度 (mm)",
    "core_thickness_mm": "铁芯厚度 (mm)",
    "coil_width_mm": "线圈宽度 (mm)",
    "coil_height_mm": "线圈高度 (mm)",
    "region_padding_mm": "空气域边距 (mm)",
    "core_material": "铁芯材料",
    "coil_material": "线圈材料",
}

OUTPUT_LABELS = {
    "max_flux_density_t": "最大磁密 (T)",
    "global_max_flux_density_t": "全局最大磁密 (T)",
    "project_name": "项目名",
    "design_name": "设计名",
    "flux_density_note": "磁密提取说明",
    "force_n": "吸力 (N)",
    "estimated_coil_resistance_ohm": "估算线圈电阻 (ohm)",
    "estimated_current_at_supply_a": "估算供电电流 (A)",
    "cross_section_area_mm2": "截面积 (mm^2)",
    "estimated_current_density_a_per_mm2": "估算电流密度 (A/mm^2)",
    "avg_current_density_a_per_mm2": "平均电流密度 (A/mm^2)",
    "capacitance_pf": "电容 (pF)",
    "capacitance_f": "电容 (F)",
    "capacitance_per_unit_length_pf_per_m": "单位长度电容 (pF/m)",
    "max_electric_field_v_per_m": "最大电场 (V/m)",
    "max_electric_field_note": "最大电场说明",
    "reference_average_field_v_per_m": "参考平均电场 (V/m)",
    "reference_capacitance_f_for_1m_depth": "参考电容 (1m 深度, F)",
    "matrix_export_path": "矩阵导出文件",
    "estimated_secondary_voltage_v": "估算次级电压 (V)",
    "turns_ratio": "匝比",
    "estimated_inductance_h": "估算电感 (H)",
    "center_flux_density_t": "中心磁密 (T)",
    "estimated_force_n": "估算力 (N)",
}

OBJECTIVE_LABELS = {
    "maximize_force": "优先提高吸力",
    "balance_force_and_size": "平衡吸力和尺寸",
    "maximize_inductance": "优先提高电感",
}

SIZE_LABELS = {
    "compact": "紧凑",
    "balanced": "均衡",
    "performance": "性能优先",
}


@dataclass(slots=True)
class DisplayRow:
    label: str
    value: str


@dataclass(slots=True)
class DemoBundle:
    requirement: str
    generated_at: str
    status: str
    status_label: str
    message: str
    run_directory: Path
    project_file: Path | None
    design_rows: list[DisplayRow] = field(default_factory=list)
    output_rows: list[DisplayRow] = field(default_factory=list)
    evaluation_rows: list[DisplayRow] = field(default_factory=list)
    evaluation_summary: str = ""
    evaluation_status_label: str = ""
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifact_paths: list[Path] = field(default_factory=list)
    summary_text_path: Path | None = None
    summary_html_path: Path | None = None
    case_report_json_path: Path | None = None
    case_report_markdown_path: Path | None = None
    case_report_html_path: Path | None = None
    delivery_report: CaseDeliveryReport | None = None

    def to_text_report(self) -> str:
        lines: list[str] = [
            "Maxwell 智能体演示结果",
            f"生成时间: {self.generated_at}",
            f"状态: {self.status_label}",
            f"说明: {self.message}",
            "",
            "原始需求:",
            self.requirement.strip(),
            "",
            "需求判定:",
            f"- 总体结论: {self.evaluation_status_label or '待验证'}",
        ]
        if self.evaluation_summary:
            lines.append(f"- 判定说明: {self.evaluation_summary}")
        for row in self.evaluation_rows:
            lines.append(f"- {row.label}: {row.value}")

        lines.extend(["", "设计参数 / 结构化规格:"])
        if self.design_rows:
            lines.extend(f"- {row.label}: {row.value}" for row in self.design_rows)
        else:
            lines.append("- 当前任务没有传统电磁铁参数表，已输出结构化仿真规格和执行结果。")

        if self.output_rows:
            lines.extend(["", "仿真输出:"])
            lines.extend(f"- {row.label}: {row.value}" for row in self.output_rows)

        if self.delivery_report:
            report = self.delivery_report
            lines.extend(["", "单案例交付报告摘要:"])
            lines.append(f"- 物理类型: {report.insight.physics_type}")
            lines.append(f"- 执行构型: {report.insight.builder_hint}")
            lines.append(f"- 迭代轮数: {len(report.iterations)}")
            for item in report.iterations:
                feedback = "是" if item.feedback_required else "否"
                lines.append(f"- 第 {item.index} 轮: {item.evaluation_status}，需要反馈修正: {feedback}")
            if report.insight.geometry_objects:
                lines.append("- 识别到的几何对象:")
                lines.extend(
                    f"  {item.get('name', '')}: {item.get('role', '')} {item.get('detail', '')}".rstrip()
                    for item in report.insight.geometry_objects
                )
            if report.insight.knowledge_items:
                lines.append("- 知识沉淀:")
                lines.extend(f"  {item}" for item in report.insight.knowledge_items)

        if self.assumptions:
            lines.extend(["", "关键假设:"])
            lines.extend(f"- {item}" for item in self.assumptions)

        if self.warnings:
            lines.extend(["", "注意事项:"])
            lines.extend(f"- {item}" for item in self.warnings)

        lines.extend(["", f"运行目录: {self.run_directory}"])
        if self.project_file:
            lines.append(f"项目文件: {self.project_file}")
        if self.case_report_html_path:
            lines.append(f"单案例交付报告: {self.case_report_html_path}")
        if self.artifact_paths:
            lines.append("产物文件:")
            lines.extend(f"- {artifact}" for artifact in self.artifact_paths)
        return "\n".join(lines)

    def to_html_document(self, page_title: str = "Maxwell 智能体运行结果") -> str:
        artifact_list = "".join(
            f'<li><a href="{escape(path.resolve().as_uri())}">{escape(path.name)}</a>'
            f'<div class="subtle">{escape(str(path))}</div></li>'
            for path in self.artifact_paths
        )
        design_rows = _rows_to_html(self.design_rows) or (
            "<tr><th>当前状态</th><td>当前任务没有传统电磁铁参数表，已输出结构化仿真规格和执行结果。</td></tr>"
        )
        output_rows = _rows_to_html(self.output_rows) or "<tr><th>暂无输出</th><td>本次运行没有生成额外输出。</td></tr>"
        evaluation_rows = _rows_to_html(self.evaluation_rows) or "<tr><th>暂无判定</th><td>当前没有生成需求判定明细。</td></tr>"
        assumption_items = _items_to_html(self.assumptions or ["无"])
        warning_items = _items_to_html(self.warnings or ["无"])
        project_line = (
            f'<a href="{escape(self.project_file.resolve().as_uri())}">{escape(self.project_file.name)}</a>'
            if self.project_file
            else "尚未生成"
        )
        delivery_section = _render_delivery_report_section(self.delivery_report, self.case_report_html_path)

        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(page_title)}</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --text: #18201b;
      --muted: #58645e;
      --line: #d7d0c1;
      --accent: #245c4a;
      --accent-strong: #173b30;
      --warn: #9f4f2b;
      --shadow: 0 18px 50px rgba(36, 43, 39, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei UI", "PingFang SC", "Segoe UI", sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #f7f2e7 0%, #efe7d9 100%);
      min-height: 100vh;
    }}
    .shell {{ width: min(1120px, calc(100vw - 32px)); margin: 24px auto 40px; }}
    .hero {{
      background: linear-gradient(135deg, rgba(36, 92, 74, 0.96), rgba(23, 59, 48, 0.98));
      color: #f4f0e8;
      border-radius: 14px;
      padding: 28px 30px;
      box-shadow: var(--shadow);
    }}
    .status {{
      display: inline-flex;
      margin-top: 18px;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid rgba(255, 255, 255, 0.18);
      font-weight: 700;
    }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; margin-top: 18px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 22px;
      box-shadow: var(--shadow);
    }}
    .panel h2 {{ margin: 0 0 14px; font-size: 18px; color: var(--accent-strong); }}
    .subtle {{ margin-top: 6px; color: var(--muted); font-size: 13px; line-height: 1.5; word-break: break-all; }}
    .mono {{
      font-family: "Consolas", "SFMono-Regular", monospace;
      background: rgba(36, 92, 74, 0.06);
      border-radius: 8px;
      padding: 14px;
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; vertical-align: top; border-bottom: 1px solid var(--line); padding: 10px 0; font-size: 14px; }}
    th {{ width: 42%; color: var(--muted); font-weight: 600; padding-right: 16px; }}
    ul {{ margin: 0; padding-left: 20px; line-height: 1.7; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .warn li {{ color: var(--warn); }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>{escape(page_title)}</h1>
      <p>{escape(self.message)}</p>
      <div class="status">状态: {escape(self.status_label)}</div>
      <p class="subtle">生成时间: {escape(self.generated_at)} | 运行目录: {escape(str(self.run_directory))}</p>
    </section>

    <div class="grid">
      <section class="panel">
        <h2>原始需求</h2>
        <div class="mono">{escape(self.requirement.strip())}</div>
      </section>
      <section class="panel">
        <h2>项目文件</h2>
        <div>{project_line}</div>
        <div class="subtle">{escape(str(self.project_file)) if self.project_file else "尚未生成 AEDT 项目文件"}</div>
      </section>
    </div>

    <div class="grid">
      <section class="panel">
        <h2>需求判定</h2>
        <div class="mono">总体结论: {escape(self.evaluation_status_label or "待验证")}
{escape(self.evaluation_summary or "当前没有形成需求判定。")}</div>
        <table>{evaluation_rows}</table>
      </section>
      <section class="panel">
        <h2>仿真输出</h2>
        <table>{output_rows}</table>
      </section>
    </div>

    {delivery_section}

    <div class="grid">
      <section class="panel">
        <h2>设计参数 / 结构化规格</h2>
        <table>{design_rows}</table>
      </section>
      <section class="panel warn">
        <h2>注意事项</h2>
        <ul>{warning_items}</ul>
      </section>
    </div>

    <div class="grid">
      <section class="panel">
        <h2>关键假设</h2>
        <ul>{assumption_items}</ul>
      </section>
      <section class="panel">
        <h2>产物文件</h2>
        <ul>{artifact_list}</ul>
      </section>
    </div>
  </div>
</body>
</html>"""


def execute_demo(
    agent: MaxwellAgent,
    requirement: str,
    progress_callback: ProgressCallback | None = None,
) -> DemoBundle:
    result = agent.run(requirement, progress_callback=progress_callback)
    bundle = build_demo_bundle(requirement=requirement, result=result)
    persist_demo_bundle(bundle)
    return bundle


def build_demo_bundle(requirement: str, result: SimulationResult) -> DemoBundle:
    design = result.design
    intake = result.intake
    artifact_paths = _dedupe_paths(result.artifacts)
    evaluation = result.evaluation
    report = result.delivery_report
    return DemoBundle(
        requirement=requirement,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        status=result.status,
        status_label=STATUS_LABELS.get(result.status, result.status),
        message=result.message,
        run_directory=result.run_directory,
        project_file=result.project_file,
        design_rows=_build_design_rows(design, intake),
        output_rows=_build_output_rows(result.outputs),
        evaluation_rows=_build_evaluation_rows(evaluation),
        evaluation_summary=evaluation.summary if evaluation else "",
        evaluation_status_label=EVALUATION_STATUS_LABELS.get(evaluation.overall_status, "") if evaluation else "",
        assumptions=list(design.assumptions) if design else list(intake.assumptions) if intake else [],
        warnings=list(design.warnings) if design else list(intake.warnings) if intake else [result.message],
        artifact_paths=artifact_paths,
        delivery_report=report,
        case_report_json_path=result.run_directory / "case_delivery_report.json" if report else None,
        case_report_markdown_path=result.run_directory / "case_delivery_report.md" if report else None,
        case_report_html_path=result.run_directory / "case_delivery_report.html" if report else None,
    )


def persist_demo_bundle(bundle: DemoBundle) -> None:
    run_directory = bundle.run_directory
    run_directory.mkdir(parents=True, exist_ok=True)

    summary_text_path = run_directory / "demo_summary.txt"
    summary_html_path = run_directory / "demo_summary.html"

    summary_text_path.write_text(bundle.to_text_report(), encoding="utf-8")
    summary_html_path.write_text(bundle.to_html_document(), encoding="utf-8")

    bundle.summary_text_path = summary_text_path
    bundle.summary_html_path = summary_html_path
    bundle.artifact_paths = _dedupe_paths([*bundle.artifact_paths, summary_text_path, summary_html_path])


def _build_design_rows(
    design: ElectromagnetDesign | None,
    intake: RequirementIntake | None,
) -> list[DisplayRow]:
    if not design:
        if intake:
            return [
                DisplayRow(label=f"结构化参数 / {key}", value=_format_value(value))
                for key, value in intake.extracted_parameters.items()
            ]
        return []

    payload = design.model_dump(mode="json")
    rows: list[DisplayRow] = []
    for key, label in DESIGN_LABELS.items():
        value = payload.get(key)
        if key == "objective" and value in OBJECTIVE_LABELS:
            formatted = OBJECTIVE_LABELS[value]
        elif key == "size_preference" and value in SIZE_LABELS:
            formatted = SIZE_LABELS[value]
        else:
            formatted = _format_value(value)
        rows.append(DisplayRow(label=label, value=formatted))
    return rows


def _build_output_rows(outputs: dict[str, Any]) -> list[DisplayRow]:
    rows: list[DisplayRow] = []
    for key, value in outputs.items():
        if key.startswith("param_"):
            label = f"结构化参数 / {key[6:]}"
        elif key.startswith("spec_"):
            label = f"仿真规格 / {key[5:]}"
        elif key == "task_family":
            label = "任务辅助标签"
        elif key == "support_message":
            label = "AI 结构化结论"
        else:
            label = OUTPUT_LABELS.get(key, key)
        rows.append(DisplayRow(label=label, value=_format_value(value)))
    return rows


def _build_evaluation_rows(evaluation: RequirementEvaluation | None) -> list[DisplayRow]:
    if not evaluation:
        return []
    return [
        DisplayRow(
            label=f"{item.name} [{EVALUATION_STATUS_LABELS.get(item.status, item.status)}]",
            value=item.detail,
        )
        for item in evaluation.checks
    ]


def _render_delivery_report_section(report: CaseDeliveryReport | None, html_path: Path | None) -> str:
    if report is None:
        return ""
    geometry_rows = _named_items_to_rows(report.insight.geometry_objects) or "<tr><th>暂无</th><td>本次没有识别到几何对象。</td></tr>"
    constraint_rows = _named_items_to_rows(report.insight.constraint_items) or "<tr><th>暂无</th><td>本次没有抽取到可验证约束。</td></tr>"
    capability_rows = _capability_items_to_rows(report.insight.capability_items) or "<tr><th>暂无</th><td>本次没有形成能力图。</td></tr>"
    residual_rows = _residual_items_to_rows(report.insight.residual_items) or "<tr><th>暂无</th><td>当前没有未满足约束残差。</td></tr>"
    iteration_rows = "".join(
        f"<tr><th>第 {item.index} 轮</th><td>状态 {escape(str(item.status))}，需求判定 {escape(str(item.evaluation_status))}，"
        f"反馈修正 {escape('是' if item.feedback_required else '否')}<div class=\"subtle\">"
        f"{escape(item.feedback_reason or item.ir_patch_summary or item.message)}</div></td></tr>"
        for item in report.iterations
    ) or "<tr><th>暂无</th><td>本次没有记录到迭代过程。</td></tr>"
    knowledge = _items_to_html(report.insight.knowledge_items or ["本次没有新增原语规则。"])
    link = (
        f'<a href="{escape(html_path.resolve().as_uri())}">打开完整单案例交付报告</a>'
        if html_path and html_path.exists()
        else "完整报告已写入运行目录。"
    )
    return f"""
    <div class="grid">
      <section class="panel">
        <h2>闭环迭代过程</h2>
        <table>{iteration_rows}</table>
        <div class="subtle">{link}</div>
      </section>
      <section class="panel">
        <h2>几何感知</h2>
        <table>{geometry_rows}</table>
      </section>
    </div>
    <div class="grid">
      <section class="panel">
        <h2>能力图</h2>
        <table>{capability_rows}</table>
      </section>
      <section class="panel">
        <h2>残差反馈</h2>
        <table>{residual_rows}</table>
      </section>
    </div>
    <div class="grid">
      <section class="panel">
        <h2>约束解释</h2>
        <table>{constraint_rows}</table>
      </section>
      <section class="panel">
        <h2>知识沉淀</h2>
        <ul>{knowledge}</ul>
      </section>
    </div>
"""


def _rows_to_html(rows: list[DisplayRow]) -> str:
    return "".join(f"<tr><th>{escape(row.label)}</th><td>{escape(row.value)}</td></tr>" for row in rows)


def _items_to_html(items: list[str]) -> str:
    return "".join(f"<li>{escape(item)}</li>" for item in items)


def _named_items_to_rows(items: list[dict[str, str]]) -> str:
    rows = []
    for item in items:
        label = item.get("name") or "未命名"
        value = "，".join(part for part in (item.get("role", ""), item.get("detail", "")) if part)
        rows.append(f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>")
    return "".join(rows)


def _capability_items_to_rows(items: list[dict[str, Any]]) -> str:
    rows = []
    for item in items:
        label = str(item.get("label") or item.get("key") or "能力")
        count = item.get("count")
        value = f"{item.get('layer', '')}"
        if count:
            value = f"{value}，数量 {count}"
        rows.append(f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>")
    return "".join(rows)


def _residual_items_to_rows(items: list[dict[str, Any]]) -> str:
    rows = []
    for item in items:
        label = str(item.get("name") or "约束")
        actual = item.get("actual")
        target = item.get("target")
        relation = item.get("relation") or ""
        residual = item.get("residual")
        if actual is not None and target is not None:
            value = f"实际 {actual} {relation} 目标 {target}，残差 {residual}"
        else:
            value = str(item.get("detail") or "未能量化，但已记录为待反馈约束。")
        rows.append(f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>")
    return "".join(rows)


def _format_value(value: Any) -> str:
    if value is None:
        return "未设置"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) >= 1000 or abs(value) < 0.001:
            return f"{value:.6g}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return ", ".join(_format_value(item) for item in value) or "无"
    if isinstance(value, dict):
        return ", ".join(f"{key}={_format_value(item)}" for key, item in value.items()) or "无"
    return str(value)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(Path(path).resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(Path(path))
    return unique
