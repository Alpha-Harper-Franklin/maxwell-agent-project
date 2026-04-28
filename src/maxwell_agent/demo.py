from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from .agent import MaxwellAgent
from .models import ElectromagnetDesign, RequirementEvaluation, RequirementIntake, SimulationResult


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
    "project_name": "项目名",
    "design_name": "设计名",
    "flux_density_note": "磁密提取说明",
    "force_n": "吸力 (N)",
    "estimated_coil_resistance_ohm": "估算线圈电阻 (Ω)",
    "estimated_current_at_supply_a": "估算供电电流 (A)",
}

OUTPUT_LABELS.update(
    {
        "max_flux_density_t": "最大磁密 (T)",
        "project_name": "项目名",
        "design_name": "设计名",
        "flux_density_note": "磁密提取说明",
        "force_n": "吸力 (N)",
        "estimated_coil_resistance_ohm": "估算线圈电阻 (Ω)",
        "estimated_current_at_supply_a": "估算供电电流 (A)",
        "capacitance_pf": "电容 (pF)",
        "capacitance_f": "电容 (F)",
        "max_electric_field_note": "最大电场结果",
        "reference_average_field_v_per_m": "参考平均电场 (V/m)",
        "reference_capacitance_f_for_1m_depth": "参考电容 (1m 深度, F)",
        "matrix_export_path": "电容矩阵文件",
    }
)

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

    def to_text_report(self) -> str:
        lines: list[str] = [
            "Maxwell 智能体演示结果",
            f"生成时间: {self.generated_at}",
            f"状态: {self.status_label}",
            f"说明: {self.message}",
            "",
            "原始需求",
            self.requirement.strip(),
            "",
            "需求判定",
            f"- 总体结论: {self.evaluation_status_label or '待验证'}",
        ]
        if self.evaluation_summary:
            lines.append(f"- 判定说明: {self.evaluation_summary}")
        for row in self.evaluation_rows:
            lines.append(f"- {row.label}: {row.value}")

        lines.extend(["", "设计参数"])
        if self.design_rows:
            for row in self.design_rows:
                lines.append(f"- {row.label}: {row.value}")
        else:
            lines.append("- 当前任务未进入 Maxwell 2D 参数化模板。")

        if self.output_rows:
            lines.extend(["", "仿真输出"])
            for row in self.output_rows:
                lines.append(f"- {row.label}: {row.value}")

        if self.assumptions:
            lines.extend(["", "关键假设"])
            for item in self.assumptions:
                lines.append(f"- {item}")

        if self.warnings:
            lines.extend(["", "注意事项"])
            for item in self.warnings:
                lines.append(f"- {item}")

        lines.extend(["", f"运行目录: {self.run_directory}"])
        if self.project_file:
            lines.append(f"项目文件: {self.project_file}")
        if self.artifact_paths:
            lines.append("产物文件")
            for artifact in self.artifact_paths:
                lines.append(f"- {artifact}")
        return (
            "\n".join(lines)
            .replace("Maxwell 2D", "通用 Maxwell")
            .replace("参数化模板", "仿真规格和执行结果")
        )

    def to_html_document(self, page_title: str = "Maxwell 演示运行结果") -> str:
        artifact_list = "".join(
            f'<li><a href="{escape(path.resolve().as_uri())}">{escape(path.name)}</a>'
            f'<div class="subtle">{escape(str(path))}</div></li>'
            for path in self.artifact_paths
        )
        design_rows = "".join(
            f"<tr><th>{escape(row.label)}</th><td>{escape(row.value)}</td></tr>"
            for row in self.design_rows
        ) or "<tr><th>当前状态</th><td>当前任务未进入 Maxwell 2D 参数化模板。</td></tr>"
        design_rows = design_rows.replace("Maxwell 2D", "通用 Maxwell")
        design_rows = design_rows.replace("参数化模板", "仿真规格和执行结果")
        output_rows = "".join(
            f"<tr><th>{escape(row.label)}</th><td>{escape(row.value)}</td></tr>"
            for row in self.output_rows
        ) or "<tr><th>暂无输出</th><td>本次运行没有生成额外输出。</td></tr>"
        evaluation_rows = "".join(
            f"<tr><th>{escape(row.label)}</th><td>{escape(row.value)}</td></tr>"
            for row in self.evaluation_rows
        ) or "<tr><th>暂无判定</th><td>当前没有生成需求判定明细。</td></tr>"
        assumption_items = "".join(f"<li>{escape(item)}</li>" for item in self.assumptions) or "<li>无</li>"
        warning_items = "".join(f"<li>{escape(item)}</li>" for item in self.warnings) or "<li>无</li>"
        project_line = (
            f'<a href="{escape(self.project_file.resolve().as_uri())}">{escape(self.project_file.name)}</a>'
            if self.project_file
            else "尚未生成"
        )

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
      background:
        radial-gradient(circle at top left, rgba(36, 92, 74, 0.12), transparent 30%),
        linear-gradient(180deg, #f7f2e7 0%, #efe7d9 100%);
      min-height: 100vh;
    }}
    .shell {{
      width: min(1120px, calc(100vw - 32px));
      margin: 24px auto 40px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(36, 92, 74, 0.96), rgba(23, 59, 48, 0.98));
      color: #f4f0e8;
      border-radius: 24px;
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
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
      margin-top: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 22px;
      box-shadow: var(--shadow);
    }}
    .panel h2 {{
      margin: 0 0 14px;
      font-size: 18px;
      color: var(--accent-strong);
    }}
    .subtle {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      word-break: break-all;
    }}
    .mono {{
      font-family: "Consolas", "SFMono-Regular", monospace;
      background: rgba(36, 92, 74, 0.06);
      border-radius: 12px;
      padding: 14px;
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      text-align: left;
      vertical-align: top;
      border-bottom: 1px solid var(--line);
      padding: 10px 0;
      font-size: 14px;
    }}
    th {{
      width: 42%;
      color: var(--muted);
      font-weight: 600;
      padding-right: 16px;
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
      line-height: 1.7;
    }}
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

    <div class="grid">
      <section class="panel">
        <h2>设计参数</h2>
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
            label = "任务类型"
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
