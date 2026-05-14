from __future__ import annotations

from datetime import datetime
from html import escape
import json
from pathlib import Path
from typing import Any

from .capability_graph import capability_graph_for_intake
from .models import (
    CaseDeliveryReport,
    CaseInsight,
    IterationRecord,
    RequirementCheck,
    RequirementEvaluation,
    RequirementIntake,
    SimulationResult,
)
from .residuals import analyze_requirement_residuals
from .semantics import infer_builder_hint, infer_physics_type


def build_case_insight(
    intake: RequirementIntake | None,
    outputs: dict[str, Any] | None = None,
    evaluation: RequirementEvaluation | None = None,
    artifacts: list[Path] | None = None,
    requirement: str = "",
) -> CaseInsight:
    spec = intake.simulation_spec if intake else {}
    plan = intake.execution_plan if intake else {}
    physics_type = infer_physics_type(spec, plan, requirement=requirement)
    builder_hint = infer_builder_hint(intake, requirement=requirement) if intake else "unknown"
    helper_label = intake.task_family if intake else "unknown"
    if builder_hint == "unknown" and helper_label not in {"", "unknown"}:
        builder_hint = helper_label
    knowledge_items = _knowledge_items(intake, artifacts or [])
    residual_items = analyze_requirement_residuals(outputs or {}, evaluation)

    return CaseInsight(
        physics_type=physics_type,
        helper_label=helper_label,
        builder_hint=builder_hint,
        capability_items=capability_graph_for_intake(intake, requirement=requirement),
        geometry_objects=_geometry_objects(intake),
        constraint_items=_constraint_items(intake),
        output_items=_output_items(intake, outputs or {}),
        residual_items=residual_items,
        knowledge_items=knowledge_items,
        engineering_explanations=_engineering_explanations(evaluation),
    )


def build_iteration_record(
    index: int,
    result: SimulationResult,
    intake: RequirementIntake | None,
    requirement: str = "",
    feedback_required: bool = False,
    feedback_reason: str = "",
) -> IterationRecord:
    checks = result.evaluation.checks if result.evaluation else []
    insight = build_case_insight(
        intake=intake or result.intake,
        outputs=result.outputs,
        evaluation=result.evaluation,
        artifacts=result.artifacts,
        requirement=requirement,
    )
    return IterationRecord(
        index=index,
        status=result.status,
        evaluation_status=result.evaluation.overall_status if result.evaluation else "unverified",
        message=result.message,
        feedback_required=feedback_required,
        feedback_reason=feedback_reason,
        design_snapshot=_design_snapshot(intake or result.intake),
        output_snapshot=_compact_outputs(result.outputs),
        failed_checks=[check.detail for check in checks if check.status == "failed"],
        passed_checks=[check.detail for check in checks if check.status == "passed"],
        unverified_checks=[check.detail for check in checks if check.status == "unverified"],
        residual_items=insight.residual_items,
        ir_patch_summary=str(((intake or result.intake).extracted_parameters or {}).get("last_ir_patch_summary", ""))
        if (intake or result.intake)
        else "",
        insight=insight,
    )


def build_case_delivery_report(
    requirement: str,
    result: SimulationResult,
    iterations: list[IterationRecord] | None = None,
) -> CaseDeliveryReport:
    intake = result.intake
    evaluation = result.evaluation
    insight = build_case_insight(
        intake=intake,
        outputs=result.outputs,
        evaluation=evaluation,
        artifacts=result.artifacts,
        requirement=requirement,
    )
    assumptions: list[str] = []
    warnings: list[str] = []
    if intake:
        assumptions.extend(str(item) for item in intake.assumptions)
        warnings.extend(str(item) for item in intake.warnings)
    if result.design:
        assumptions.extend(str(item) for item in result.design.assumptions)
        warnings.extend(str(item) for item in result.design.warnings)
    return CaseDeliveryReport(
        requirement=requirement,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        run_directory=result.run_directory,
        project_file=result.project_file,
        final_status=result.status,
        final_evaluation_status=evaluation.overall_status if evaluation else "unverified",
        final_summary=evaluation.summary if evaluation else result.message,
        final_outputs=dict(result.outputs),
        assumptions=_unique_text(assumptions),
        warnings=_unique_text(warnings),
        insight=insight,
        iterations=iterations or list(result.iterations),
        artifact_paths=list(result.artifacts),
    )


def persist_case_delivery_report(report: CaseDeliveryReport, run_dir: Path | None = None) -> list[Path]:
    target_dir = run_dir or report.run_directory
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "case_delivery_report.json"
    md_path = target_dir / "case_delivery_report.md"
    html_path = target_dir / "case_delivery_report.html"
    report.artifact_paths = _unique_paths([*report.artifact_paths, json_path, md_path, html_path])
    json_path.write_text(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(case_report_to_markdown(report), encoding="utf-8")
    html_path.write_text(case_report_to_html(report), encoding="utf-8")
    return [json_path, md_path, html_path]


def case_report_to_markdown(report: CaseDeliveryReport) -> str:
    lines = [
        "# Maxwell Agent 单案例交付报告",
        "",
        f"- 生成时间: {report.generated_at}",
        f"- 运行状态: {report.final_status}",
        f"- 需求判定: {report.final_evaluation_status}",
        f"- 运行目录: {report.run_directory}",
    ]
    if report.project_file:
        lines.append(f"- Maxwell 工程: {report.project_file}")
    lines.extend(["", "## 原始需求", "", report.requirement.strip(), "", "## 系统识别结果"])
    lines.extend(
        [
            f"- 物理类型: {report.insight.physics_type}",
            f"- 执行构型: {report.insight.builder_hint}",
            f"- 辅助标签: {report.insight.helper_label}",
        ]
    )
    lines.extend(_markdown_items("几何对象", _format_named_items(report.insight.geometry_objects)))
    lines.extend(_markdown_items("能力图", _format_capability_items(report.insight.capability_items)))
    lines.extend(_markdown_items("约束", _format_named_items(report.insight.constraint_items)))
    lines.extend(_markdown_items("输出", _format_named_items(report.insight.output_items)))
    lines.extend(_markdown_items("残差反馈", _format_residual_items(report.insight.residual_items)))
    lines.extend(["", "## 迭代过程"])
    if report.iterations:
        for item in report.iterations:
            lines.append(
                f"- 第 {item.index} 轮: 状态 {item.status}, 需求判定 {item.evaluation_status}, "
                f"需要反馈修正: {'是' if item.feedback_required else '否'}"
            )
            if item.feedback_reason:
                lines.append(f"  原因: {item.feedback_reason}")
            if item.ir_patch_summary:
                lines.append(f"  IR 修正: {item.ir_patch_summary}")
            for detail in item.failed_checks:
                lines.append(f"  未满足: {detail}")
            for detail in item.passed_checks[:6]:
                lines.append(f"  已满足: {detail}")
    else:
        lines.append("- 本次没有记录到多轮迭代。")
    lines.extend(["", "## 最终工程解释", "", report.final_summary or "无"])
    lines.extend(_markdown_items("指标解释", report.insight.engineering_explanations))
    if report.insight.knowledge_items:
        lines.extend(_markdown_items("知识库记录", report.insight.knowledge_items))
    if report.assumptions:
        lines.extend(_markdown_items("工程假设", report.assumptions))
    if report.warnings:
        lines.extend(_markdown_items("注意事项", report.warnings))
    return "\n".join(lines).strip() + "\n"


def case_report_to_html(report: CaseDeliveryReport) -> str:
    geometry = _html_list(_format_named_items(report.insight.geometry_objects))
    capabilities = _html_list(_format_capability_items(report.insight.capability_items))
    constraints = _html_list(_format_named_items(report.insight.constraint_items))
    outputs = _html_list(_format_named_items(report.insight.output_items))
    residuals = _html_list(_format_residual_items(report.insight.residual_items))
    knowledge = _html_list(report.insight.knowledge_items or ["本次没有新增原语规则。"])
    explanations = _html_list(report.insight.engineering_explanations or [report.final_summary or "本次没有形成工程解释。"])
    assumptions = _html_list(report.assumptions or ["无"])
    warnings = _html_list(report.warnings or ["无"])
    iterations = _html_iterations(report.iterations)
    project = escape(str(report.project_file)) if report.project_file else "未生成"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Maxwell Agent 单案例交付报告</title>
  <style>
    body {{ margin: 0; font-family: "Microsoft YaHei UI", "PingFang SC", "Segoe UI", sans-serif; background: #f5f1e8; color: #162019; }}
    main {{ width: min(1120px, calc(100vw - 32px)); margin: 24px auto 48px; }}
    header {{ background: #173b30; color: #fff8ec; padding: 26px 30px; border-radius: 14px; }}
    section {{ background: #fffdf8; border: 1px solid #d8cfbc; border-radius: 10px; padding: 20px; margin-top: 16px; }}
    h1, h2 {{ margin: 0 0 12px; }}
    p {{ line-height: 1.7; }}
    ul {{ margin: 0; padding-left: 20px; line-height: 1.7; }}
    .meta {{ color: #d7e6dc; line-height: 1.7; }}
    .mono {{ white-space: pre-wrap; background: #f0eadf; border-radius: 8px; padding: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Maxwell Agent 单案例交付报告</h1>
    <div class="meta">生成时间: {escape(report.generated_at)} | 状态: {escape(str(report.final_status))} | 需求判定: {escape(str(report.final_evaluation_status))}</div>
    <div class="meta">运行目录: {escape(str(report.run_directory))}</div>
    <div class="meta">Maxwell 工程: {project}</div>
  </header>
  <section><h2>原始需求</h2><div class="mono">{escape(report.requirement.strip())}</div></section>
  <section>
    <h2>系统识别结果</h2>
    <ul>
      <li>物理类型: {escape(report.insight.physics_type)}</li>
      <li>执行构型: {escape(report.insight.builder_hint)}</li>
      <li>辅助标签: {escape(report.insight.helper_label)}</li>
    </ul>
  </section>
  <div class="grid">
    <section><h2>几何对象</h2>{geometry}</section>
    <section><h2>能力图</h2>{capabilities}</section>
    <section><h2>约束</h2>{constraints}</section>
    <section><h2>输出</h2>{outputs}</section>
    <section><h2>残差反馈</h2>{residuals}</section>
  </div>
  <section><h2>迭代过程</h2>{iterations}</section>
  <section><h2>最终工程解释</h2><p>{escape(report.final_summary or "无")}</p>{explanations}</section>
  <div class="grid">
    <section><h2>知识库记录</h2>{knowledge}</section>
    <section><h2>工程假设</h2>{assumptions}</section>
    <section><h2>注意事项</h2>{warnings}</section>
  </div>
</main>
</body>
</html>"""


def _geometry_objects(intake: RequirementIntake | None) -> list[dict[str, str]]:
    if intake is None:
        return []
    if intake.design:
        design = intake.design
        return [
            {"name": "铁芯", "role": "导磁结构", "detail": f"{design.core_width_mm}mm x {design.core_height_mm}mm"},
            {"name": "线圈", "role": "电流激励", "detail": f"{design.coil_turns} 匝, {design.current_a}A"},
            {"name": "气隙", "role": "主要工作间隙", "detail": f"{design.air_gap_mm}mm"},
            {"name": "空气域", "role": "外部边界区域", "detail": f"边距 {design.region_padding_mm}mm"},
        ]
    geometry = _as_mapping(_as_mapping(intake.simulation_spec).get("geometry"))
    objects = geometry.get("objects") or geometry.get("cross_section_objects") or geometry.get("entities")
    rows: list[dict[str, str]] = []
    if isinstance(objects, list):
        for index, item in enumerate(objects, start=1):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("id") or f"object_{index}")
            primitive = str(item.get("primitive") or item.get("shape") or item.get("type") or "unknown")
            material = str(item.get("material") or item.get("role") or "")
            rows.append({"name": name, "role": primitive, "detail": material or _short_json(item)})
    elif geometry:
        rows.append(
            {
                "name": str(geometry.get("type") or "geometry"),
                "role": "参数化几何",
                "detail": _short_json({key: value for key, value in geometry.items() if key != "objects"}),
            }
        )
    return rows


def _constraint_items(intake: RequirementIntake | None) -> list[dict[str, str]]:
    if intake is None:
        return []
    rows: list[dict[str, str]] = []
    if intake.design:
        design = intake.design
        candidates = {
            "供电电压": design.supply_voltage_v,
            "电流下限": design.current_min_a,
            "电流上限": design.current_limit_a,
            "目标吸力": design.target_force_n,
            "气隙": design.air_gap_mm,
        }
        for name, value in candidates.items():
            if value is not None:
                rows.append({"name": name, "role": "硬约束", "detail": _format_value(value)})
    constraints = _as_mapping(_as_mapping(intake.simulation_spec).get("constraints"))
    for key, value in constraints.items():
        rows.append({"name": str(key), "role": "硬约束", "detail": _format_value(value)})
    return _unique_named(rows)


def _output_items(intake: RequirementIntake | None, outputs: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    spec_outputs = _as_mapping(intake.simulation_spec).get("required_outputs") if intake else []
    if isinstance(spec_outputs, list):
        for item in spec_outputs:
            if isinstance(item, dict):
                rows.append(
                    {
                        "name": str(item.get("name") or item.get("output_key") or "output"),
                        "role": "需求输出",
                        "detail": _short_json(item),
                    }
                )
            elif isinstance(item, str):
                rows.append({"name": item, "role": "需求输出", "detail": ""})
    for key, value in outputs.items():
        if key.startswith(("param_", "spec_", "plan_")) or key in {"status", "notes"}:
            continue
        rows.append({"name": str(key), "role": "仿真/估算输出", "detail": _format_value(value)})
    return _unique_named(rows)


def _knowledge_items(intake: RequirementIntake | None, artifacts: list[Path]) -> list[str]:
    items: list[str] = []
    spec = _as_mapping(intake.simulation_spec) if intake else {}
    learned = spec.get("learned_primitives") or spec.get("primitive_library_updates")
    if isinstance(learned, list):
        for item in learned:
            items.append(f"本次记录原语规则: {_format_value(item)}")
    for path in artifacts:
        name = path.name.lower()
        if name.startswith("learned_primitive_") and name.endswith(".json"):
            items.append(f"本次学习并落盘原语: {path.name}")
    if intake and _geometry_objects(intake):
        items.append("本次几何已整理为可复用对象图，后续同类任务可直接复用。")
    return _unique_text(items)


def _engineering_explanations(evaluation: RequirementEvaluation | None) -> list[str]:
    if evaluation is None:
        return []
    rows = []
    for check in evaluation.checks:
        prefix = {"passed": "满足", "failed": "未满足", "unverified": "待验证"}.get(check.status, check.status)
        rows.append(f"{prefix}: {check.name}，{check.detail}")
    return rows


def _design_snapshot(intake: RequirementIntake | None) -> dict[str, Any]:
    if intake is None:
        return {}
    if intake.design:
        payload = intake.design.model_dump(mode="json")
        keep = [
            "supply_voltage_v",
            "current_min_a",
            "current_limit_a",
            "current_a",
            "coil_turns",
            "air_gap_mm",
            "core_width_mm",
            "core_height_mm",
            "coil_width_mm",
            "coil_height_mm",
        ]
        return {key: payload.get(key) for key in keep if payload.get(key) is not None}
    snapshot: dict[str, Any] = {"task_family": intake.task_family}
    for source in (intake.extracted_parameters, _as_mapping(intake.simulation_spec).get("constraints") or {}):
        if isinstance(source, dict):
            for key, value in source.items():
                if len(snapshot) >= 16:
                    break
                if isinstance(value, (str, int, float, bool)):
                    snapshot[str(key)] = value
    return snapshot


def _compact_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in outputs.items():
        if key.startswith(("param_", "spec_", "plan_")):
            continue
        if isinstance(value, (str, int, float, bool)):
            compact[key] = value
        if len(compact) >= 20:
            break
    return compact


def _markdown_items(title: str, items: list[str]) -> list[str]:
    lines = ["", f"## {title}"]
    if items:
        lines.extend(f"- {item}" for item in items)
    else:
        lines.append("- 无")
    return lines


def _format_named_items(items: list[dict[str, str]]) -> list[str]:
    rows: list[str] = []
    for item in items:
        name = item.get("name", "")
        role = item.get("role", "")
        detail = item.get("detail", "")
        rows.append("，".join(part for part in (name, role, detail) if part))
    return rows


def _format_capability_items(items: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for item in items:
        label = str(item.get("label") or item.get("key") or "能力")
        layer = str(item.get("layer") or "")
        count = item.get("count")
        suffix = f"，数量 {count}" if count else ""
        rows.append(f"{label}，{layer}{suffix}")
    return rows


def _format_residual_items(items: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for item in items:
        name = str(item.get("name") or "约束")
        actual = item.get("actual")
        target = item.get("target")
        relation = item.get("relation") or ""
        residual = item.get("residual")
        if actual is not None and target is not None:
            rows.append(f"{name}: 实际 {actual} {relation} 目标 {target}，残差 {residual}")
        else:
            rows.append(f"{name}: {item.get('detail') or '未能量化，但已记录为待反馈约束。'}")
    return rows


def _html_list(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{escape(str(item))}</li>" for item in items) + "</ul>"


def _html_iterations(iterations: list[IterationRecord]) -> str:
    if not iterations:
        return "<p>本次没有记录到多轮迭代。</p>"
    parts = ["<ul>"]
    for item in iterations:
        feedback = "是" if item.feedback_required else "否"
        detail = item.feedback_reason or item.message
        parts.append(
            f"<li>第 {item.index} 轮: 状态 {escape(str(item.status))}, "
            f"需求判定 {escape(str(item.evaluation_status))}, 需要反馈修正: {feedback}"
            f"<br>{escape(detail)}"
            f"{'<br>IR 修正: ' + escape(item.ir_patch_summary) if item.ir_patch_summary else ''}</li>"
        )
    parts.append("</ul>")
    return "".join(parts)


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _short_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) >= 1000 or abs(value) < 0.001:
            return f"{value:.6g}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return ", ".join(_format_value(item) for item in value)
    if isinstance(value, dict):
        return _short_json(value)
    return str(value)


def _unique_text(items: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def _unique_named(items: list[dict[str, str]]) -> list[dict[str, str]]:
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (item.get("name", ""), item.get("role", ""), item.get("detail", ""))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(Path(path).resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(Path(path))
    return unique
