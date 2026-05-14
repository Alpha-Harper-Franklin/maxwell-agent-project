from __future__ import annotations

from typing import Any

from .maxwell_ir import MaxwellIRPlan
from .models import RequirementIntake
from .semantics import infer_physics_type


CAPABILITY_CATALOG: dict[str, dict[str, str]] = {
    "physics.magnetostatic_2d": {"layer": "physics", "label": "二维静磁"},
    "physics.electrostatic_2d": {"layer": "physics", "label": "二维静电"},
    "geometry.rectangle": {"layer": "geometry", "label": "矩形原语"},
    "geometry.circle": {"layer": "geometry", "label": "圆形原语"},
    "geometry.region": {"layer": "geometry", "label": "空气/边界区域"},
    "geometry.subtract": {"layer": "geometry", "label": "布尔减"},
    "assignment.current": {"layer": "excitation", "label": "电流激励"},
    "assignment.voltage": {"layer": "excitation", "label": "电压激励"},
    "assignment.balloon": {"layer": "boundary", "label": "开放边界"},
    "assignment.matrix": {"layer": "solver", "label": "矩阵参数"},
    "output.field_scalar": {"layer": "postprocess", "label": "场量提取"},
    "output.matrix_export_value": {"layer": "postprocess", "label": "矩阵导出"},
    "output.derived": {"layer": "postprocess", "label": "解析派生输出"},
}


def capability_graph_for_intake(intake: RequirementIntake | None, requirement: str = "") -> list[dict[str, Any]]:
    if intake is None:
        return []
    items: dict[str, dict[str, Any]] = {}
    physics = infer_physics_type(intake.simulation_spec, intake.execution_plan, requirement=requirement)
    if physics != "unknown":
        _add(items, f"physics.{physics}")

    ir_payload = _ir_payload(intake)
    if ir_payload:
        try:
            plan = MaxwellIRPlan.model_validate(ir_payload)
            _collect_from_ir_plan(items, plan)
        except Exception:
            _collect_from_spec(items, intake.simulation_spec)
    else:
        _collect_from_spec(items, intake.simulation_spec)

    return sorted(items.values(), key=lambda item: (str(item["layer"]), str(item["key"])))


def _collect_from_ir_plan(items: dict[str, dict[str, Any]], plan: MaxwellIRPlan) -> None:
    for obj in plan.objects:
        _add(items, f"geometry.{obj.kind}", count=1)
    for op in plan.operations:
        _add(items, f"geometry.{op.kind}", count=1)
    for assignment in plan.assignments:
        _add(items, f"assignment.{assignment.kind}", count=1)
    for _ in plan.derived_outputs:
        _add(items, "output.derived", count=1)
    for post in plan.postprocess:
        _add(items, f"output.{post.kind}", count=1)


def _collect_from_spec(items: dict[str, dict[str, Any]], spec: dict[str, Any] | None) -> None:
    spec = spec if isinstance(spec, dict) else {}
    geometry = spec.get("geometry")
    if isinstance(geometry, dict):
        objects = geometry.get("objects") or geometry.get("cross_section_objects") or geometry.get("entities")
        if isinstance(objects, list):
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                kind = str(obj.get("primitive") or obj.get("shape") or obj.get("type") or "").lower()
                if "rect" in kind or "bar" in kind or "plate" in kind:
                    _add(items, "geometry.rectangle", count=1)
                if "circle" in kind or "annulus" in kind or "ring" in kind or "圆" in kind or "环" in kind:
                    _add(items, "geometry.circle", count=1)
                if obj.get("operation") == "subtract" or obj.get("subtract"):
                    _add(items, "geometry.subtract", count=1)
        else:
            blob = str(geometry).lower()
            if "rect" in blob or "plate" in blob or "bar" in blob or "矩形" in blob or "板" in blob:
                _add(items, "geometry.rectangle")
            if "circle" in blob or "radius" in blob or "annulus" in blob or "ring" in blob or "圆" in blob or "环" in blob:
                _add(items, "geometry.circle")
            if "inner" in blob and "outer" in blob:
                _add(items, "geometry.subtract")
            if "内" in blob and "外" in blob and ("圆" in blob or "环" in blob):
                _add(items, "geometry.subtract")

    excitations = spec.get("excitations")
    blob = str(excitations).lower()
    if "current" in blob or "电流" in blob:
        _add(items, "assignment.current")
    if "voltage" in blob or "电压" in blob:
        _add(items, "assignment.voltage")
    if spec.get("boundaries"):
        _add(items, "assignment.balloon")

    required_outputs = str(spec.get("required_outputs") or "").lower()
    if any(token in required_outputs for token in ("field", "bmax", "flux", "磁", "场")):
        _add(items, "output.field_scalar")
    if "capacitance" in required_outputs or "电容" in required_outputs:
        _add(items, "assignment.matrix")
        _add(items, "output.matrix_export_value")


def _add(items: dict[str, dict[str, Any]], key: str, count: int = 0) -> None:
    meta = CAPABILITY_CATALOG.get(key, {"layer": "unknown", "label": key})
    current = items.setdefault(
        key,
        {
            "key": key,
            "layer": meta["layer"],
            "label": meta["label"],
            "count": 0,
        },
    )
    current["count"] = int(current.get("count") or 0) + count


def _ir_payload(intake: RequirementIntake) -> dict[str, Any] | None:
    for source in (intake.simulation_spec, intake.execution_plan):
        if isinstance(source, dict) and isinstance(source.get("ir_plan"), dict):
            return source["ir_plan"]
    return None
