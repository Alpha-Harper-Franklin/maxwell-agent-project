from __future__ import annotations

import re
from typing import Any

from .models import RequirementEvaluation


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def analyze_requirement_residuals(
    outputs: dict[str, Any] | None,
    evaluation: RequirementEvaluation | None,
) -> list[dict[str, Any]]:
    if evaluation is None:
        return []
    residuals: list[dict[str, Any]] = []
    outputs = outputs or {}
    for check in evaluation.checks:
        item = _residual_from_check(check.name, check.status, check.detail, outputs)
        if item:
            residuals.append(item)
    return residuals


def compact_residual_payload(residuals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in residuals:
        rows.append(
            {
                "name": item.get("name"),
                "status": item.get("status"),
                "actual": item.get("actual"),
                "target": item.get("target"),
                "relation": item.get("relation"),
                "residual": item.get("residual"),
                "relative_error": item.get("relative_error"),
                "suggested_ir_targets": item.get("suggested_ir_targets", []),
            }
        )
    return rows


def _residual_from_check(
    name: str,
    status: str,
    detail: str,
    outputs: dict[str, Any],
) -> dict[str, Any] | None:
    actual, relation, target = _parse_relation(detail)
    if actual is None or target is None:
        actual, target, relation = _fallback_from_outputs(name, outputs)
    if actual is None or target is None:
        return {
            "name": name,
            "status": status,
            "detail": detail,
            "relation": "unknown",
            "suggested_ir_targets": _suggested_targets(name),
        }

    if relation in {"<=", "max"}:
        residual = actual - target
        passed_margin = target - actual
    elif relation in {">=", "min"}:
        residual = target - actual
        passed_margin = actual - target
    else:
        residual = abs(actual - target)
        passed_margin = -residual
    denominator = abs(target) if abs(target) > 1e-12 else 1.0
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "actual": actual,
        "target": target,
        "relation": relation,
        "residual": residual,
        "passed_margin": passed_margin,
        "relative_error": residual / denominator,
        "suggested_ir_targets": _suggested_targets(name),
    }


def _parse_relation(detail: str) -> tuple[float | None, str, float | None]:
    text = str(detail or "")
    patterns = [
        (rf"({FLOAT_RE})\s*(?:[A-Za-z/%^0-9.]+)?\s*<=\s*({FLOAT_RE})", "<="),
        (rf"({FLOAT_RE})\s*(?:[A-Za-z/%^0-9.]+)?\s*≤\s*({FLOAT_RE})", "<="),
        (rf"({FLOAT_RE})\s*(?:[A-Za-z/%^0-9.]+)?\s*>\s*({FLOAT_RE})", "<="),
        (rf"({FLOAT_RE})\s*(?:[A-Za-z/%^0-9.]+)?\s*>=\s*({FLOAT_RE})", ">="),
        (rf"({FLOAT_RE})\s*(?:[A-Za-z/%^0-9.]+)?\s*≥\s*({FLOAT_RE})", ">="),
        (rf"({FLOAT_RE})\s*(?:[A-Za-z/%^0-9.]+)?\s*<\s*({FLOAT_RE})", ">="),
        (rf"约\s*({FLOAT_RE}).*?不超过\s*({FLOAT_RE})", "<="),
        (rf"({FLOAT_RE}).*?不超过\s*({FLOAT_RE})", "<="),
        (rf"({FLOAT_RE}).*?超过\s*({FLOAT_RE})", "<="),
        (rf"约\s*({FLOAT_RE}).*?至少\s*({FLOAT_RE})", ">="),
        (rf"({FLOAT_RE}).*?低于\s*({FLOAT_RE})", ">="),
        (rf"({FLOAT_RE}).*?小于目标\s*({FLOAT_RE})", ">="),
    ]
    for pattern, relation in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)), relation, float(match.group(2))
    return None, "unknown", None


def _fallback_from_outputs(name: str, outputs: dict[str, Any]) -> tuple[float | None, float | None, str]:
    lower = name.lower()
    text = name.lower()
    if "电流密度" in text or "current_density" in lower:
        return _as_float(outputs.get("estimated_current_density_a_per_mm2")), None, "<="
    if "电流" in text or "current" in lower:
        return _as_float(outputs.get("estimated_current_at_supply_a") or outputs.get("current_a")), None, "<="
    if "磁密" in text or "磁场" in text or "flux" in lower or "field" in lower:
        return _as_float(outputs.get("max_flux_density_t") or outputs.get("global_max_flux_density_t")), None, "<="
    if "电容" in text or "capacitance" in lower:
        return _as_float(outputs.get("capacitance_pf")), None, ">="
    if "电感" in text or "inductance" in lower:
        return _as_float(outputs.get("estimated_inductance_h")), None, ">="
    return None, None, "unknown"


def _suggested_targets(name: str) -> list[str]:
    lower = name.lower()
    text = name.lower()
    if "电流密度" in text or "current_density" in lower:
        return ["geometry.width_mm", "geometry.thickness_mm", "excitations.current_a"]
    if "供电" in text or "电流" in text or "current" in lower:
        return ["excitations.current_a", "design.coil_turns", "design.coil_width_mm", "design.coil_height_mm"]
    if "磁密" in text or "磁场" in text or "flux" in lower:
        return ["geometry.core_width_mm", "geometry.air_gap_mm", "excitations.current_a", "postprocess.object_name"]
    if "电容" in text or "capacitance" in lower:
        return ["geometry.plate_width_mm", "geometry.plate_spacing_mm", "geometry.outer_radius_mm", "geometry.inner_radius_mm"]
    if "电场" in text or "electric" in lower:
        return ["geometry.plate_spacing_mm", "excitations.voltage_v", "boundaries.air_pad_mm"]
    if "电感" in text or "inductance" in lower:
        return ["geometry.air_gap_mm", "excitations.coil_turns", "geometry.core_width_mm"]
    return []


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
