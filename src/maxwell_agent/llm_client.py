from __future__ import annotations

import json
import math
import re
import time
from textwrap import dedent
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from .config import Settings
from .errors import RequirementPlanningError, UnknownPrimitiveError, UnsupportedRequirementError
from .maxwell_ir import (
    GeneratedIRPlan,
    IRPatch,
    IRAssignment,
    IRDerivedOutput,
    IRLocalValue,
    IRObject,
    IRParameterBinding,
    IRPostprocess,
    MaxwellIRPlan,
    IROperation,
    apply_ir_patch,
    render_script_from_ir,
    validate_ir_plan,
)
from .models import ElectromagnetDesign, ElectromagnetDesignPatch, GeneratedMaxwellScript, RequirementIntake
from .prompting import (
    build_design_feedback_instructions,
    build_intake_feedback_instructions,
    build_ir_feedback_instructions,
    build_ir_patch_feedback_instructions,
    build_ir_generation_instructions,
    build_ir_repair_instructions,
    build_primitive_template_generation_instructions,
    build_primitive_template_repair_instructions,
    build_requirement_structuring_instructions,
    build_spec_refinement_instructions,
)
from .primitive_library import (
    PrimitiveLibrary,
    PrimitiveTemplate,
    PrimitiveTemplateArtifact,
    PrimitiveTemplateObject,
    validate_primitive_template,
)
from .residuals import analyze_requirement_residuals, compact_residual_payload
from .semantics import enrich_intake_semantics, infer_builder_hint, intake_has_generic_object_graph


FLOAT_CORE = r"[0-9]+(?:\.[0-9]+)?"
FLOAT_PATTERN = rf"({FLOAT_CORE})"


def _extract_json_blob(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return text[start : end + 1]


def _as_response_input(input_payload: Any) -> list[dict[str, str]]:
    content = input_payload if isinstance(input_payload, str) else json.dumps(input_payload, ensure_ascii=False)
    return [{"role": "user", "content": content}]


def _normalize_task_family(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = [chunk.strip(" -\n\r\t") for chunk in re.split(r"[;\n]+", value) if chunk.strip(" -\n\r\t")]
        return parts or [value.strip()]
    return [str(value).strip()]


def _normalize_json_like(value: Any) -> Any:
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    if isinstance(value, list):
        return [_normalize_json_like(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_json_like(item) for key, item in value.items()}
    return str(value)


def _normalize_primitive_artifact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["summary"] = str(normalized.get("summary") or "已生成可复用二维原语模板。").strip()
    normalized["assumptions"] = _ensure_list(normalized.get("assumptions"))
    normalized["warnings"] = _ensure_list(normalized.get("warnings"))
    template_payload = normalized.get("template")
    if not isinstance(template_payload, dict):
        raise RequirementPlanningError("AI 原语学习结果缺少 template 对象。")
    normalized["template"] = template_payload
    return normalized


def _validate_primitive_artifact_payload(payload: dict[str, Any]) -> PrimitiveTemplateArtifact:
    normalized = _normalize_primitive_artifact_payload(payload)
    try:
        artifact = PrimitiveTemplateArtifact.model_validate(normalized)
        artifact.template = validate_primitive_template(artifact.template)
        return artifact
    except ValidationError as exc:
        raise RequirementPlanningError("AI 生成的原语模板不符合结构约定。") from exc
    except ValueError as exc:
        raise RequirementPlanningError(f"AI 生成的原语模板语义无效: {exc}") from exc


def _merge_unique_strings(*groups: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in _ensure_list(group):
            key = item.strip()
            if not key:
                continue
            lowered = key.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            merged.append(key)
    return merged


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _coerce_measure_value(value: Any) -> float | None:
    if isinstance(value, dict):
        if "value" in value:
            return _coerce_float(value.get("value"))
        if "magnitude" in value:
            return _coerce_float(value.get("magnitude"))
    return _coerce_float(value)


def _coerce_xy_pair(value: Any) -> tuple[float | None, float | None]:
    if isinstance(value, dict):
        return _coerce_measure_value(value.get("x")), _coerce_measure_value(value.get("y"))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _coerce_measure_value(value[0]), _coerce_measure_value(value[1])
    return None, None


def _normalize_generic_primitives_to_objects(geometry: dict[str, Any]) -> list[dict[str, Any]]:
    primitives = _as_mapping_list(geometry.get("primitives"))
    if not primitives:
        return []
    operations = _as_mapping_list(geometry.get("operations"))
    primitive_and_operations = [*primitives, *operations]

    by_name = {str(item.get("name") or "").strip(): item for item in primitives if str(item.get("name") or "").strip()}
    objects: list[dict[str, Any]] = []
    consumed: set[str] = set()

    for item in primitive_and_operations:
        item_type = str(item.get("type") or item.get("kind") or item.get("shape") or "").strip().lower()
        if item_type != "subtract":
            continue
        blank_name = str(item.get("blank") or item.get("blank_part") or item.get("outer") or "").strip()
        tool_name = str(item.get("tool") or item.get("tool_part") or item.get("inner") or "").strip()
        blank = by_name.get(blank_name)
        tool = by_name.get(tool_name)
        if not blank or not tool:
            continue
        blank_type = str(blank.get("type") or blank.get("kind") or blank.get("shape") or "").strip().lower()
        tool_type = str(tool.get("type") or tool.get("kind") or tool.get("shape") or "").strip().lower()
        if blank_type != "circle" or tool_type != "circle":
            continue
        outer_radius = _coerce_measure_value(blank.get("radius") or blank.get("radius_mm"))
        inner_radius = _coerce_measure_value(tool.get("radius") or tool.get("radius_mm"))
        if outer_radius is None or inner_radius is None:
            continue
        center_x, center_y = _coerce_xy_pair(blank.get("center") or blank.get("center_mm"))
        if center_x is None or center_y is None:
            center_x, center_y = _coerce_xy_pair(tool.get("center") or tool.get("center_mm"))
        if center_x is None or center_y is None:
            center_x, center_y = 0.0, 0.0
        objects.append(
            {
                "name": str(item.get("result_name") or item.get("name") or "annular_conductor"),
                "type": "annulus",
                "center": {"x": center_x, "y": center_y},
                "inner_radius": min(inner_radius, outer_radius),
                "outer_radius": max(inner_radius, outer_radius),
            }
        )
        consumed.update({blank_name, tool_name, str(item.get("name") or "").strip(), str(item.get("result_name") or "").strip()})

    for item in primitives:
        name = str(item.get("name") or "").strip()
        if name in consumed:
            continue
        item_type = str(item.get("type") or item.get("kind") or item.get("shape") or "").strip().lower()
        if item_type in {"circle", "rectangle", "annulus", "region", "air_region"}:
            normalized = dict(item)
            normalized.setdefault("type", item_type)
            if item_type == "annulus":
                normalized.setdefault("center", item.get("center") or item.get("center_mm") or {"x": 0.0, "y": 0.0})
                normalized.setdefault("inner_radius", item.get("inner_radius") or item.get("inner_radius_mm"))
                normalized.setdefault("outer_radius", item.get("outer_radius") or item.get("outer_radius_mm"))
            objects.append(normalized)
    return objects


def _sanitize_identifier(value: Any, fallback: str) -> str:
    text = _normalize_task_family(value)
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"{fallback}_{text}"
    return text


def _replace_expression_tokens(expression: str, mapping: dict[str, str]) -> str:
    result = str(expression or "")
    for key, value in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        result = re.sub(rf"\b{re.escape(key)}\b", value, result)
    return result


def _flatten_generic_object_payload(raw_object: dict[str, Any]) -> dict[str, Any]:
    flattened = dict(raw_object)
    center_x, center_y = _coerce_xy_pair(raw_object.get("center"))
    if center_x is None or center_y is None:
        center_x, center_y = _coerce_xy_pair(raw_object.get("center_mm"))
    if center_x is not None and center_y is not None:
        flattened.setdefault("center_x_mm", center_x)
        flattened.setdefault("center_y_mm", center_y)
    origin_x, origin_y = _coerce_xy_pair(raw_object.get("origin") or raw_object.get("corner") or raw_object.get("lower_left"))
    if origin_x is not None and origin_y is not None:
        flattened.setdefault("origin_x_mm", origin_x)
        flattened.setdefault("origin_y_mm", origin_y)
    width_mm = _coerce_measure_value(raw_object.get("width")) or _coerce_measure_value(raw_object.get("width_mm"))
    height_mm = _coerce_measure_value(raw_object.get("height")) or _coerce_measure_value(raw_object.get("height_mm"))
    if width_mm is None or height_mm is None:
        size = raw_object.get("size") or raw_object.get("sizes")
        if isinstance(size, dict):
            width_mm = width_mm or _coerce_measure_value(size.get("x") or size.get("width"))
            height_mm = height_mm or _coerce_measure_value(size.get("y") or size.get("height"))
        elif isinstance(size, (list, tuple)) and len(size) >= 2:
            width_mm = width_mm or _coerce_measure_value(size[0])
            height_mm = height_mm or _coerce_measure_value(size[1])
    if width_mm is not None:
        flattened.setdefault("width_mm", width_mm)
    if height_mm is not None:
        flattened.setdefault("height_mm", height_mm)
    radius_mm = _coerce_measure_value(raw_object.get("radius")) or _coerce_measure_value(raw_object.get("radius_mm"))
    if radius_mm is not None:
        flattened.setdefault("radius_mm", radius_mm)
    inner_radius_mm = _coerce_measure_value(raw_object.get("inner_radius")) or _coerce_measure_value(raw_object.get("inner_radius_mm"))
    if inner_radius_mm is not None:
        flattened.setdefault("inner_radius_mm", inner_radius_mm)
    outer_radius_mm = _coerce_measure_value(raw_object.get("outer_radius")) or _coerce_measure_value(raw_object.get("outer_radius_mm"))
    if outer_radius_mm is not None:
        flattened.setdefault("outer_radius_mm", outer_radius_mm)
    outer_width_mm = _coerce_measure_value(raw_object.get("outer_width")) or _coerce_measure_value(raw_object.get("outer_width_mm"))
    if outer_width_mm is not None:
        flattened.setdefault("outer_width_mm", outer_width_mm)
    outer_height_mm = _coerce_measure_value(raw_object.get("outer_height")) or _coerce_measure_value(raw_object.get("outer_height_mm"))
    if outer_height_mm is not None:
        flattened.setdefault("outer_height_mm", outer_height_mm)
    inner_width_mm = _coerce_measure_value(raw_object.get("inner_width")) or _coerce_measure_value(raw_object.get("inner_width_mm"))
    if inner_width_mm is not None:
        flattened.setdefault("inner_width_mm", inner_width_mm)
    inner_height_mm = _coerce_measure_value(raw_object.get("inner_height")) or _coerce_measure_value(raw_object.get("inner_height_mm"))
    if inner_height_mm is not None:
        flattened.setdefault("inner_height_mm", inner_height_mm)
    return flattened


def _extract_first_float(text: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _extract_voltage_mentions(text: str) -> list[float]:
    values: list[float] = []
    for raw_value, unit in re.findall(rf"{FLOAT_PATTERN}\s*(k?v)(?=$|[^A-Za-z])", text, re.IGNORECASE):
        scale = 1000.0 if unit.lower() == "kv" else 1.0
        values.append(float(raw_value) * scale)
    return values


def _extract_transformer_voltage_pair(text: str) -> tuple[float | None, float | None]:
    pair_match = re.search(
        rf"{FLOAT_PATTERN}\s*(k?v)(?=$|[^A-Za-z])\s*(?:到|变到|变成|降到|to|->)\s*{FLOAT_PATTERN}(?:\s*(k?v)(?=$|[^A-Za-z]))?",
        text,
        re.IGNORECASE,
    )
    if pair_match:
        primary_value = float(pair_match.group(1))
        primary_unit = pair_match.group(2).lower()
        secondary_value = float(pair_match.group(3))
        secondary_unit = (pair_match.group(4) or "v").lower()
        primary_v = primary_value * (1000.0 if primary_unit == "kv" else 1.0)
        secondary_v = secondary_value * (1000.0 if secondary_unit == "kv" else 1.0)
        return primary_v, secondary_v

    mentions = _extract_voltage_mentions(text)
    primary_v = mentions[0] if len(mentions) >= 1 else None
    secondary_v = mentions[1] if len(mentions) >= 2 else None
    return primary_v, secondary_v


def _extract_current_range(text: str) -> tuple[float | None, float | None]:
    match = re.search(
        rf"(?:电流(?!密度)|current(?!\s*density))\s*{FLOAT_PATTERN}\s*A\s*(?:-|~|～|到|至)\s*{FLOAT_PATTERN}\s*A?",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None, None
    low = float(match.group(1))
    high = float(match.group(2))
    if low > high:
        low, high = high, low
    return low, high


def _extract_plate_spacing_mm(text: str) -> float | None:
    return _extract_first_float(
        text,
        [
            rf"(?:板间距|极板间距|板距|plate\s*spacing|gap)\s*{FLOAT_PATTERN}\s*mm\b",
        ],
    )


def _extract_plate_width_mm(text: str) -> float | None:
    return _extract_first_float(
        text,
        [
            rf"(?:板宽|极板宽度|plate\s*width)\s*{FLOAT_PATTERN}\s*mm\b",
        ],
    )


def _extract_capacitance_f(text: str) -> float | None:
    patterns = [
        rf"(?:电容|capacit(?:ance|or)?).{{0,20}}?{FLOAT_PATTERN}\s*pF\b",
        rf"{FLOAT_PATTERN}\s*pF\b",
    ]
    value = _extract_first_float(text, patterns)
    if value is not None:
        return value * 1e-12
    patterns = [
        rf"(?:电容|capacit(?:ance|or)?).{{0,20}}?{FLOAT_PATTERN}\s*nF\b",
        rf"{FLOAT_PATTERN}\s*nF\b",
    ]
    value = _extract_first_float(text, patterns)
    if value is not None:
        return value * 1e-9
    patterns = [
        rf"(?:电容|capacit(?:ance|or)?).{{0,20}}?{FLOAT_PATTERN}\s*(?:uF|μF)\b",
        rf"{FLOAT_PATTERN}\s*(?:uF|μF)\b",
    ]
    value = _extract_first_float(text, patterns)
    if value is not None:
        return value * 1e-6
    return _extract_first_float(
        text,
        [
            rf"(?:电容|capacit(?:ance|or)?).{{0,20}}?{FLOAT_PATTERN}\s*F\b",
            rf"{FLOAT_PATTERN}\s*F\b",
        ],
    )


def _extract_target_capacitance_f(text: str) -> float | None:
    for unit_scale, unit_pattern in (
        (1e-12, r"pF"),
        (1e-9, r"nF"),
        (1e-6, r"(?:uF|μF)"),
        (1.0, r"F"),
    ):
        value = _extract_first_float(
            text,
            [
                rf"(?:电容|capacit(?:ance|or)?).{{0,20}}?(?:至少|不低于|不小于|达到|大于等于|>=)\s*{FLOAT_PATTERN}\s*{unit_pattern}\b",
            ],
        )
        if value is not None:
            return value * unit_scale
    for unit_scale, unit_pattern in (
        (1e-12, r"pF"),
        (1e-9, r"nF"),
        (1e-6, r"(?:uF|μF)"),
        (1.0, r"F"),
    ):
        value = _extract_first_float(
            text,
            [
                rf"(?:电容|capacit(?:ance|or)?).{{0,20}}?(?:目标|为|取值为|设为|=|约|大约)\s*{FLOAT_PATTERN}\s*{unit_pattern}\b",
            ],
        )
        if value is not None:
            return value * unit_scale
    return None


def _extract_max_electric_field_v_per_m(text: str) -> float | None:
    patterns = [
        (1e6, rf"(?:电场|电场强度|electric\s*field).{{0,20}}?(?:不超过|不大于|小于等于|上限|最大|max(?:imum)?)\s*{FLOAT_PATTERN}\s*kV\s*/\s*mm\b"),
        (1e3, rf"(?:电场|电场强度|electric\s*field).{{0,20}}?(?:不超过|不大于|小于等于|上限|最大|max(?:imum)?)\s*{FLOAT_PATTERN}\s*V\s*/\s*mm\b"),
        (1e3, rf"(?:电场|电场强度|electric\s*field).{{0,20}}?(?:不超过|不大于|小于等于|上限|最大|max(?:imum)?)\s*{FLOAT_PATTERN}\s*kV\s*/\s*m\b"),
        (1.0, rf"(?:电场|电场强度|electric\s*field).{{0,20}}?(?:不超过|不大于|小于等于|上限|最大|max(?:imum)?)\s*{FLOAT_PATTERN}\s*V\s*/\s*m\b"),
    ]
    for scale, pattern in patterns:
        value = _extract_first_float(text, [pattern])
        if value is not None:
            return value * scale
    return None


def _looks_like_capacitor_requirement(text: str) -> bool:
    lower = text.lower()
    return any(keyword in text for keyword in ("电容器", "平行板", "平板电容")) or "capacitor" in lower


def _looks_like_coaxial_capacitor_requirement(text: str) -> bool:
    lower = text.lower()
    has_coaxial = "\u540c\u8f74" in text or "coaxial" in lower
    has_capacitive_task = any(keyword in text for keyword in ("\u7535\u5bb9", "\u7535\u7f06")) or any(
        keyword in lower for keyword in ("capacitance", "capacitor", "cable")
    )
    return has_coaxial and has_capacitive_task


def _looks_like_transformer_requirement(text: str) -> bool:
    lower = text.lower()
    return any(keyword in text for keyword in ("\u53d8\u538b\u5668", "鍙樺帇鍣?")) or "transformer" in lower


def _looks_like_inductor_requirement(text: str) -> bool:
    lower = text.lower()
    if any(
        keyword in text
        for keyword in (
            "\u7535\u611f",
            "\u7535\u6297\u5668",
            "\u7ebf\u5708\u7535\u611f",
        )
    ) or "inductor" in lower:
        return True
    has_inductance_unit = bool(re.search(rf"{FLOAT_PATTERN}\s*(?:mH|uH|H)\b", text, flags=re.IGNORECASE))
    has_current = bool(re.search(rf"{FLOAT_PATTERN}\s*A\b", text, flags=re.IGNORECASE))
    has_turn_like_number = any(keyword in text for keyword in ("\u530d", "\u5305", "turn"))
    corrupted_text = "?" in text
    return has_inductance_unit and has_current and (has_turn_like_number or corrupted_text)


def _looks_like_solenoid_requirement(text: str) -> bool:
    lower = text.lower()
    return any(
        keyword in text
        for keyword in (
            "\u87ba\u7ebf\u7ba1",
            "\u7a7a\u5fc3\u7ebf\u5708",
            "\u7a7a\u6c14\u82af\u7ebf\u5708",
        )
    ) or any(keyword in lower for keyword in ("solenoid", "air core coil", "air-core coil"))


def _looks_like_busbar_requirement(text: str) -> bool:
    lower = text.lower()
    return any(
        keyword in text
        for keyword in (
            "\u6bcd\u6392",
            "\u94dc\u6392",
            "\u6c47\u6d41\u6392",
        )
    ) or "busbar" in lower


def _looks_like_electromagnet_requirement(text: str) -> bool:
    lower = text.lower()
    return any(
        keyword in text
        for keyword in (
            "\u7535\u78c1\u94c1",
            "\u7ebf\u5708",
            "\u78c1\u8def",
            "\u6267\u884c\u5668",
            "\u884e\u94c1",
            "\u5438\u529b",
        )
    ) or any(keyword in lower for keyword in ("electromagnet", "coil", "magnetic circuit", "actuator"))


def _extract_frequency_hz(text: str) -> float | None:
    if "\u5de5\u9891" in text:
        return 50.0
    khz = _extract_first_float(
        text,
        [
            rf"(?:\u9891\u7387|frequency)\s*{FLOAT_PATTERN}\s*khz\b",
            rf"{FLOAT_PATTERN}\s*khz\b",
        ],
    )
    if khz is not None:
        return khz * 1000.0
    return _extract_first_float(
        text,
        [
            rf"(?:\u9891\u7387|frequency)\s*{FLOAT_PATTERN}\s*hz\b",
            rf"{FLOAT_PATTERN}\s*hz\b",
        ],
    )


def _extract_inductance_h(text: str) -> float | None:
    mh_value = _extract_first_float(
        text,
        [
            rf"(?:\u7535\u611f|inductance).{{0,10}}?{FLOAT_PATTERN}\s*mh\b",
            rf"{FLOAT_PATTERN}\s*mh\b",
        ],
    )
    if mh_value is not None:
        return mh_value * 1e-3
    uh_value = _extract_first_float(
        text,
        [
            rf"(?:\u7535\u611f|inductance).{{0,10}}?{FLOAT_PATTERN}\s*uh\b",
            rf"{FLOAT_PATTERN}\s*uh\b",
        ],
    )
    if uh_value is not None:
        return uh_value * 1e-6
    return _extract_first_float(
        text,
        [
            rf"(?:\u7535\u611f|inductance).{{0,10}}?{FLOAT_PATTERN}\s*h\b",
            rf"{FLOAT_PATTERN}\s*h\b",
        ],
    )


def _extract_mm_value(text: str, keywords: list[str]) -> float | None:
    alternatives = "|".join(keywords)
    return _extract_first_float(
        text,
        [
            rf"(?:{alternatives}).{{0,12}}?{FLOAT_PATTERN}\s*mm\b",
        ],
    )


def _extract_turns(text: str, default: int) -> int:
    turns = _extract_first_float(
        text,
        [
            rf"(?:\u530d\u6570|\u7ebf\u5708\u530d\u6570|turns?)\s*(?:=|\u8bbe\u4e3a|\u53d6\u503c\u4e3a)?\s*{FLOAT_PATTERN}\b",
            rf"(?:\u5305\u6570|\u7ebf\u5708\u5305\u6570|turns?)\s*(?:=|\u8bbe\u4e3a|\u53d6\u503c\u4e3a)?\s*{FLOAT_PATTERN}\b",
            rf"{FLOAT_PATTERN}\s*(?:\u530d|turns?)\b",
            rf"{FLOAT_PATTERN}\s*(?:\u5305|turns?)\b",
        ],
    )
    return int(round(turns or float(default)))


def _extract_current_a(text: str, default: float) -> float:
    current_min, current_max = _extract_current_range(text)
    if current_max is not None:
        return current_max
    current = _extract_first_float(
        text,
        [
            rf"(?:\u7535\u6d41(?!\u5bc6\u5ea6)|current(?!\s*density))\s*(?:=|\u8bbe\u4e3a|\u53d6\u503c\u4e3a)?\s*{FLOAT_PATTERN}\s*A\b",
            rf"{FLOAT_PATTERN}\s*A\b",
        ],
    )
    return current or default


def _extract_exact_current_a(text: str) -> float | None:
    current_min, current_max = _extract_current_range(text)
    if current_max is not None:
        return None
    return _extract_first_float(
        text,
        [
            rf"(?:电流(?!密度)|current(?!\s*density))\s*(?:为|取值为|设为|设置为|=|约|大约|控制在)?\s*{FLOAT_PATTERN}\s*A\b",
            rf"通以\s*{FLOAT_PATTERN}\s*A\b",
        ],
    )


def _extract_current_limit_a(text: str) -> float | None:
    current_min, current_max = _extract_current_range(text)
    current_limit = _extract_first_float(
        text,
        [
            rf"(?:电流(?!密度)|current(?!\s*density))[^0-9A-Za-z]{{0,16}}?(?:不超过|不大于|小于等于|上限|最大|max(?:imum)?)\s*{FLOAT_PATTERN}\s*A\b",
            rf"(?:<=|≤)\s*{FLOAT_PATTERN}\s*A\b",
        ],
    )
    return current_limit or current_max


def _extract_max_current_density_a_per_mm2(text: str) -> float | None:
    return _extract_first_float(
        text,
        [
            rf"(?:电流密度|current\s*density)[^0-9A-Za-z]{{0,20}}?(?:不超过|不大于|小于等于|上限|最大|max(?:imum)?)\s*{FLOAT_PATTERN}\s*A\s*/\s*mm(?:\^?2|²)\b",
            rf"(?:<=|≤)\s*{FLOAT_PATTERN}\s*A\s*/\s*mm(?:\^?2|²)\b",
        ],
    )


def _extract_max_flux_density_t(text: str) -> float | None:
    return _extract_first_float(
        text,
        [
            rf"(?:磁密|磁通密度|磁感应强度|flux\s*density)[^0-9A-Za-z]{{0,20}}?(?:不超过|不大于|小于等于|上限|最大|max(?:imum)?)\s*{FLOAT_PATTERN}\s*T\b",
            rf"(?:<=|≤)\s*{FLOAT_PATTERN}\s*T\b",
        ],
    )


def _fill_simple_constraints_from_requirement(payload: dict[str, Any], requirement: str) -> None:
    current_min, current_max = _extract_current_range(requirement)
    current_limit = _extract_first_float(
        requirement,
        [
            rf"(?:电流(?!密度)|current(?!\s*density))[^0-9A-Za-z]{{0,16}}?(?:不超过|不大于|小于等于|上限|最大|max(?:imum)?)\s*{FLOAT_PATTERN}\s*A\b",
            rf"(?:<=|≤)\s*{FLOAT_PATTERN}\s*A\b",
        ],
    )
    exact_current = _extract_first_float(
        requirement,
        [
            rf"(?:电流(?!密度)|current(?!\s*density))\s*(?:为|取值为|设为|设置为|=|约|大约|控制在)?\s*{FLOAT_PATTERN}\s*A\b",
            rf"通以\s*{FLOAT_PATTERN}\s*A\b",
        ],
    )
    air_gap = _extract_first_float(
        requirement,
        [
            rf"(?:气隙|air\s*gap)\s*{FLOAT_PATTERN}\s*mm\b",
        ],
    )
    turns = _extract_first_float(
        requirement,
        [
            rf"(?:匝数|线圈匝数|turns?)\s*(?:为|取值为|设为|=)?\s*{FLOAT_PATTERN}\b",
            rf"{FLOAT_PATTERN}\s*(?:匝|turns?)\b",
        ],
    )
    target_force = _extract_first_float(
        requirement,
        [
            rf"(?:吸力|推力|force)[^0-9A-Za-z]{{0,16}}?(?:至少|不低于|不小于|达到|大于等于|>=)\s*{FLOAT_PATTERN}\s*N\b",
        ],
    )
    voltage_mentions = _extract_voltage_mentions(requirement)
    supply_voltage = voltage_mentions[0] if voltage_mentions else None

    if payload.get("current_min_a") is None and current_min is not None:
        payload["current_min_a"] = current_min
    if payload.get("current_limit_a") is None and current_max is not None:
        payload["current_limit_a"] = current_max
    if payload.get("current_limit_a") is None and current_limit is not None:
        payload["current_limit_a"] = current_limit
    if payload.get("current_a") is None and exact_current is not None and current_max is None:
        payload["current_a"] = exact_current
    if payload.get("current_a") is None and current_max is not None:
        payload["current_a"] = current_max
    if payload.get("current_a") is None and exact_current is None and current_limit is not None:
        payload["current_a"] = current_limit
    if payload.get("air_gap_mm") is None and air_gap is not None:
        payload["air_gap_mm"] = air_gap
    if payload.get("coil_turns") is None and turns is not None:
        payload["coil_turns"] = int(round(turns))
    if payload.get("supply_voltage_v") is None and supply_voltage is not None:
        payload["supply_voltage_v"] = supply_voltage
    if payload.get("target_force_n") is None and target_force is not None:
        payload["target_force_n"] = target_force


def _build_busbar_constraints(requirement: str) -> dict[str, float]:
    constraints: dict[str, float] = {}
    current_min, current_max = _extract_current_range(requirement)
    current_limit = _extract_current_limit_a(requirement)
    exact_current = _extract_exact_current_a(requirement)
    max_current_density = _extract_max_current_density_a_per_mm2(requirement)
    max_flux_density = _extract_max_flux_density_t(requirement)
    if current_min is not None:
        constraints["current_min_a"] = current_min
    if current_limit is not None:
        constraints["current_limit_a"] = current_limit
    if exact_current is not None:
        constraints["required_current_a"] = exact_current
    if max_current_density is not None:
        constraints["max_current_density_a_per_mm2"] = max_current_density
    if max_flux_density is not None:
        constraints["max_flux_density_t"] = max_flux_density
    return constraints


def _build_capacitor_constraints(requirement: str) -> dict[str, float]:
    constraints: dict[str, float] = {}
    target_capacitance_f = _extract_target_capacitance_f(requirement)
    max_field_v_per_m = _extract_max_electric_field_v_per_m(requirement)
    if target_capacitance_f is not None:
        constraints["target_capacitance_f"] = target_capacitance_f
    if max_field_v_per_m is not None:
        constraints["max_electric_field_v_per_m"] = max_field_v_per_m
    return constraints


def _normalize_design_payload(payload: dict[str, Any], requirement: str) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.setdefault("problem_type", "electromagnet_2d")
    normalized["source_requirement"] = requirement.strip()
    normalized.setdefault("summary", requirement.strip())
    if not normalized.get("summary"):
        normalized["summary"] = requirement.strip() or "已生成电磁任务结构化结果。"
    normalized["assumptions"] = _ensure_list(normalized.get("assumptions"))
    normalized["warnings"] = _ensure_list(normalized.get("warnings"))
    normalized["required_outputs"] = _ensure_list(normalized.get("required_outputs")) or [
        "flux_density",
        "force",
        "inductance",
    ]
    _fill_simple_constraints_from_requirement(normalized, requirement)
    return normalized


def _build_minimal_spec_from_design(design_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "software": "ansys_maxwell",
        "task_family": "electromagnet_2d",
        "geometry": {
            "type": "core_coil_armature_2d",
            "air_gap_mm": design_payload.get("air_gap_mm"),
            "core_width_mm": design_payload.get("core_width_mm"),
            "core_height_mm": design_payload.get("core_height_mm"),
            "core_thickness_mm": design_payload.get("core_thickness_mm"),
            "coil_width_mm": design_payload.get("coil_width_mm"),
            "coil_height_mm": design_payload.get("coil_height_mm"),
            "region_padding_mm": design_payload.get("region_padding_mm"),
        },
        "materials": {
            "core": design_payload.get("core_material", "steel_1008"),
            "coil": design_payload.get("coil_material", "copper"),
        },
        "excitations": {
            "current_a": design_payload.get("current_a"),
            "coil_turns": design_payload.get("coil_turns"),
        },
        "constraints": {
            "supply_voltage_v": design_payload.get("supply_voltage_v"),
            "current_min_a": design_payload.get("current_min_a"),
            "current_limit_a": design_payload.get("current_limit_a"),
            "target_force_n": design_payload.get("target_force_n"),
        },
        "solver": {
            "design_type": "Maxwell 2D",
            "solution_type": "Magnetostatic",
        },
        "required_outputs": design_payload.get("required_outputs", ["flux_density", "force", "inductance"]),
        "execution_ready": True,
        "missing_inputs": [],
    }


def _build_default_execution_plan_from_design(design_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "software": "ansys_maxwell",
        "design_type": "Maxwell 2D",
        "solution_type": "Magnetostatic",
        "project_name": "electromagnet_2d",
        "design_name": "Electromagnet2D",
        "model_units": "mm",
        "variables": {
            "air_gap_mm": design_payload.get("air_gap_mm", 2.0),
            "core_width_mm": design_payload.get("core_width_mm", 20.0),
            "core_height_mm": design_payload.get("core_height_mm", 40.0),
            "core_thickness_mm": design_payload.get("core_thickness_mm", 10.0),
            "coil_width_mm": design_payload.get("coil_width_mm", 12.0),
            "coil_height_mm": design_payload.get("coil_height_mm", 20.0),
            "region_padding_mm": design_payload.get("region_padding_mm", 20.0),
            "coil_turns": design_payload.get("coil_turns", 400),
            "current_a": design_payload.get("current_a", 1.0),
        },
        "steps": [
            {"action": "build_geometry"},
            {"action": "assign_materials"},
            {"action": "assign_excitation"},
            {"action": "solve"},
            {"action": "extract_outputs"},
        ],
        "postprocess": ["max_flux_density_t"],
        "execution_ready": True,
        "missing_inputs": [],
    }


def _execution_plan_is_compilable(plan: dict[str, Any]) -> bool:
    if not isinstance(plan, dict):
        return False
    if bool(plan.get("execution_ready")):
        return True
    variables = plan.get("variables")
    steps = plan.get("steps")
    has_variables = isinstance(variables, dict) and bool(variables)
    has_steps = isinstance(steps, list) and bool(steps)
    has_solver = bool(plan.get("design_type") or plan.get("solution_type"))
    return has_solver and (has_variables or has_steps)


def _normalize_intake_payload(payload: dict[str, Any], requirement: str) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["summary"] = str(normalized.get("summary") or requirement.strip() or "已完成需求结构化。").strip()
    normalized["support_message"] = str(
        normalized.get("support_message") or "当前任务尚未落到可执行的 Maxwell 脚本生成链路。"
    ).strip()
    normalized["assumptions"] = _ensure_list(normalized.get("assumptions"))
    normalized["warnings"] = _ensure_list(normalized.get("warnings"))

    extracted = normalized.get("extracted_parameters")
    if not isinstance(extracted, dict):
        extracted = {}
    normalized["extracted_parameters"] = _normalize_json_like(extracted)

    simulation_spec = normalized.get("simulation_spec")
    if not isinstance(simulation_spec, dict):
        simulation_spec = {}
    normalized["simulation_spec"] = _normalize_json_like(simulation_spec)

    execution_plan = normalized.get("execution_plan")
    if not isinstance(execution_plan, dict):
        execution_plan = {}
    normalized["execution_plan"] = _normalize_json_like(execution_plan)

    design_payload = normalized.get("design")
    if isinstance(design_payload, dict):
        normalized["design"] = _normalize_design_payload(design_payload, requirement)
    else:
        normalized["design"] = None

    task_family = _normalize_task_family(normalized.get("task_family") or normalized["simulation_spec"].get("task_family"))
    if normalized["design"] and task_family in {"unknown", "generic_maxwell"}:
        task_family = "electromagnet_2d"
    normalized["task_family"] = task_family

    if normalized["design"] and not normalized["simulation_spec"]:
        normalized["simulation_spec"] = _build_minimal_spec_from_design(normalized["design"])
    if normalized["design"] and not normalized["execution_plan"]:
        normalized["execution_plan"] = _build_default_execution_plan_from_design(normalized["design"])

    supported_now = bool(normalized.get("supported_now"))
    spec_ready = bool(normalized["simulation_spec"].get("execution_ready"))
    plan_ready = _execution_plan_is_compilable(normalized["execution_plan"])
    if normalized["design"] or spec_ready or plan_ready:
        supported_now = True
    normalized["supported_now"] = supported_now

    normalized["simulation_spec"].setdefault("task_family", task_family)
    normalized["execution_plan"].setdefault("task_family", task_family)
    normalized["simulation_spec"].setdefault("execution_ready", spec_ready or plan_ready)
    normalized["execution_plan"].setdefault("execution_ready", plan_ready or spec_ready)
    normalized["simulation_spec"], normalized["execution_plan"] = enrich_intake_semantics(
        normalized["simulation_spec"],
        normalized["execution_plan"],
        requirement=requirement,
    )
    if isinstance(normalized["extracted_parameters"], dict):
        physics_type = normalized["simulation_spec"].get("physics_type")
        if physics_type and "physics_type" not in normalized["extracted_parameters"]:
            normalized["extracted_parameters"]["physics_type"] = physics_type

    if "missing_inputs" not in normalized["simulation_spec"]:
        normalized["simulation_spec"]["missing_inputs"] = []
    if "missing_inputs" not in normalized["execution_plan"]:
        normalized["execution_plan"]["missing_inputs"] = []

    return normalized


def _normalize_script_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["filename"] = str(normalized.get("filename") or "generated_maxwell_job.py").strip()
    normalized["entrypoint"] = str(normalized.get("entrypoint") or "run_job").strip()
    normalized["summary"] = str(normalized.get("summary") or "已生成 Maxwell 脚本。").strip()
    normalized["code"] = str(normalized.get("code") or "")
    normalized["assumptions"] = _ensure_list(normalized.get("assumptions"))
    normalized["warnings"] = _ensure_list(normalized.get("warnings"))
    return normalized


def _fallback_transformer_intake(requirement: str) -> RequirementIntake:
    primary_v, secondary_v = _extract_transformer_voltage_pair(requirement)
    missing_inputs = [
        "额定功率或额定电流",
        "频率",
        "相数",
        "铁芯材料",
        "铁芯几何尺寸",
        "绕组结构与绝缘要求",
        "冷却方式",
    ]
    return RequirementIntake(
        task_family="transformer",
        supported_now=False,
        support_message="AI 已将该需求结构化为变压器任务，但当前信息不足，无法直接生成可执行脚本。",
        summary="已识别为变压器需求。",
        extracted_parameters={
            "device_type": "transformer",
            "input_voltage_v": primary_v,
            "output_voltage_v": secondary_v,
        },
        simulation_spec={
            "software": "ansys_maxwell",
            "task_family": "transformer",
            "geometry": {"type": "core_winding_transformer"},
            "materials": {},
            "excitations": {"input_voltage_v": primary_v},
            "constraints": {"output_voltage_v": secondary_v},
            "solver": {"preferred": "Transient or EddyCurrent"},
            "required_outputs": ["secondary_voltage", "flux_density", "loss"],
            "execution_ready": False,
            "missing_inputs": missing_inputs,
        },
        execution_plan={
            "software": "ansys_maxwell",
            "design_type": "Maxwell 2D or 3D",
            "solution_type": "Transient or EddyCurrent",
            "project_name": "transformer_concept",
            "design_name": "TransformerConcept",
            "model_units": "mm",
            "variables": {},
            "steps": [],
            "postprocess": [],
            "execution_ready": False,
            "missing_inputs": missing_inputs,
        },
        assumptions=[
            "默认按工频交流变压器理解。",
        ],
        warnings=[
            "当前缺少容量、频率、铁芯和绕组细节，不能直接建模。",
        ],
    )


def _fallback_unknown_intake(requirement: str) -> RequirementIntake:
    return RequirementIntake(
        task_family="unknown",
        supported_now=False,
        support_message="AI 未能把当前需求整理成可直接执行的 Maxwell 任务，请补充几何、材料、激励和目标输出。",
        summary="已完成初步结构化，但当前信息不足以直接执行。",
        extracted_parameters={"original_requirement": requirement.strip()},
        simulation_spec={
            "software": "ansys_maxwell",
            "task_family": "unknown",
            "execution_ready": False,
            "missing_inputs": ["geometry", "materials", "excitations", "required_outputs"],
        },
        execution_plan={
            "software": "ansys_maxwell",
            "execution_ready": False,
            "steps": [],
            "postprocess": [],
            "missing_inputs": ["geometry", "materials", "excitations", "required_outputs"],
        },
        assumptions=[],
        warnings=["当前通用脚本执行器仍需要明确的工程语义输入。"],
    )


def _looks_like_annular_conductor_requirement(text: str) -> bool:
    lower = text.lower()
    has_ring = any(token in text for token in ("\u540c\u5fc3", "\u5706\u73af", "\u73af\u5f62", "\u73af\u5bfc\u4f53", "\u5185\u534a\u5f84", "\u5916\u534a\u5f84"))
    has_current = "\u7535\u6d41" in text or re.search(rf"{FLOAT_PATTERN}\s*A\b", text, flags=re.IGNORECASE) is not None
    return (has_ring and has_current) or ("annular" in lower and "current" in lower)


def _fallback_annular_conductor_intake(requirement: str) -> RequirementIntake:
    text = requirement.strip()
    inner_radius_mm = _extract_first_float(
        text,
        [
            rf"(?:\u5185\u534a\u5f84|inner\s*radius)\s*{FLOAT_PATTERN}\s*mm\b",
        ],
    ) or 3.0
    outer_radius_mm = _extract_first_float(
        text,
        [
            rf"(?:\u5916\u534a\u5f84|outer\s*radius)\s*{FLOAT_PATTERN}\s*mm\b",
        ],
    ) or 6.0
    current_a = _extract_current_a(text, default=100.0)
    return RequirementIntake(
        task_family="generic_maxwell",
        supported_now=True,
        support_message="\u5df2\u8bc6\u522b\u4e3a\u4e8c\u7ef4\u540c\u5fc3\u73af\u5bfc\u4f53\u9759\u78c1\u4efb\u52a1\uff0c\u5e76\u8f6c\u6210\u901a\u7528 Maxwell IR \u539f\u8bed\u56fe\u3002",
        summary="\u5df2\u6309\u4e8c\u7ef4\u73af\u5f62\u8f7d\u6d41\u5bfc\u4f53\u751f\u6210\u7ed3\u6784\u5316\u4eff\u771f\u89c4\u683c\u3002",
        extracted_parameters={
            "geometry_type": "annular_conductor_cross_section",
            "inner_radius_mm": inner_radius_mm,
            "outer_radius_mm": outer_radius_mm,
            "current_a": current_a,
        },
        simulation_spec={
            "software": "ansys_maxwell",
            "physics_type": "magnetostatic_2d",
            "task_family": "generic_maxwell",
            "geometry": {
                "model_dimensionality": "2D planar",
                "primitives": [
                    {
                        "name": "annular_conductor",
                        "type": "annulus",
                        "center": {"x": 0.0, "y": 0.0, "unit": "mm"},
                        "inner_radius": {"value": inner_radius_mm, "unit": "mm"},
                        "outer_radius": {"value": outer_radius_mm, "unit": "mm"},
                    }
                ],
            },
            "materials": [{"target": "annular_conductor", "material": "copper"}],
            "excitations": [{"target": "annular_conductor", "type": "total_current", "value_A": current_a}],
            "boundaries": [{"target": "outer_air_region", "type": "balloon"}],
            "solver": {"solution_type": "Magnetostatic"},
            "required_outputs": [{"name": "B_max_global", "unit": "T"}],
            "execution_ready": True,
            "missing_inputs": [],
        },
        execution_plan={
            "software": "ansys_maxwell",
            "task_family": "generic_maxwell",
            "design_type": "Maxwell 2D",
            "solution_type": "Magnetostatic",
            "project_name": "annular_conductor_2d",
            "design_name": "GenericAnnular2D",
            "model_units": "mm",
            "variables": {
                "inner_radius_mm": inner_radius_mm,
                "outer_radius_mm": outer_radius_mm,
                "current_a": current_a,
            },
            "steps": ["build_annulus", "assign_current", "solve", "extract_bmax"],
            "postprocess": ["max_flux_density"],
            "execution_ready": True,
            "missing_inputs": [],
        },
        assumptions=[
            "\u9ed8\u8ba4\u7535\u6d41\u65b9\u5411\u6cbf\u4e8c\u7ef4\u622a\u9762\u6cd5\u5411\uff0c\u6750\u6599\u4e3a\u94dc\uff0c\u5468\u56f4\u4e3a\u7a7a\u6c14\u57df\u3002",
        ],
        warnings=[
            "\u8be5\u6a21\u578b\u662f\u4e8c\u7ef4\u622a\u9762\u9759\u78c1\u9996\u7248\u9a8c\u8bc1\uff0c\u672a\u5305\u542b\u6709\u9650\u957f\u5ea6\u7aef\u90e8\u6548\u5e94\u3002",
        ],
    )


def _fallback_capacitor_intake(requirement: str) -> RequirementIntake:
    text = requirement.strip()
    lower = text.lower()
    plate_spacing_mm = _extract_plate_spacing_mm(text) or 1.0
    plate_width_mm = _extract_plate_width_mm(text) or 20.0
    voltage_mentions = _extract_voltage_mentions(text)
    voltage_v = voltage_mentions[0] if voltage_mentions else 100.0
    constraints = _build_capacitor_constraints(text)

    required_outputs: list[str] = []
    if "电容" in text or "capacit" in lower:
        required_outputs.append("capacitance")
    if "电场" in text or "electric field" in lower:
        required_outputs.append("electric_field")
    if not required_outputs:
        required_outputs = ["capacitance", "electric_field"]

    return RequirementIntake(
        task_family="capacitor_2d",
        supported_now=True,
        support_message="已识别为二维平行板电容器任务，并切换到本地可执行的 Maxwell 静电链路。",
        summary="已按二维平行板电容器生成结构化仿真规格。",
        extracted_parameters={
            "original_requirement": text,
            "plate_spacing_mm": plate_spacing_mm,
            "plate_width_mm": plate_width_mm,
            "voltage_v": voltage_v,
            **constraints,
        },
        simulation_spec={
            "software": "ansys_maxwell",
            "task_family": "capacitor_2d",
            "geometry": {
                "type": "parallel_plate_capacitor_2d",
                "plate_spacing_mm": plate_spacing_mm,
                "plate_width_mm": plate_width_mm,
                "air_region_margin_mm": 10.0,
            },
            "materials": {
                "plate": "pec",
                "dielectric": "vacuum",
            },
            "excitations": {
                "voltage_V": voltage_v,
            },
            "constraints": constraints,
            "boundaries": {
                "air_region_margin_mm": 10.0,
            },
            "solver": {
                "design_type": "Maxwell 2D",
                "solution_type": "ElectrostaticXY",
            },
            "required_outputs": required_outputs,
            "execution_ready": True,
            "missing_inputs": [],
        },
        execution_plan={
            "software": "ansys_maxwell",
            "task_family": "capacitor_2d",
            "design_type": "Maxwell 2D",
            "solution_type": "ElectrostaticXY",
            "project_name": "capacitor_2d",
            "design_name": "Capacitor2D",
            "model_units": "mm",
            "variables": {
                "plate_spacing_mm": plate_spacing_mm,
                "plate_width_mm": plate_width_mm,
                "voltage_V": voltage_v,
                "air_region_margin_mm": 10.0,
            },
            "steps": [
                {"action": "build_geometry"},
                {"action": "assign_voltage"},
                {"action": "solve"},
                {"action": "extract_outputs"},
            ],
            "postprocess": required_outputs,
            "execution_ready": True,
            "missing_inputs": [],
        },
        assumptions=[
            "未明确介质材料时默认按真空介质处理。",
            "未明确板厚时按二维理想导体极板处理。",
        ],
        warnings=[
            "当前本地可执行链路主要覆盖规则几何的二维平行板电容器任务。",
        ],
    )


def _fallback_electromagnet_intake(requirement: str) -> RequirementIntake:
    design_payload = _normalize_design_payload(
        {
            "problem_type": "electromagnet_2d",
            "summary": requirement.strip(),
            "objective": "maximize_force",
            "size_preference": "balanced",
            "core_material": "steel_1008",
            "coil_material": "copper",
        },
        requirement,
    )
    design = ElectromagnetDesign.model_validate(design_payload)
    return RequirementIntake(
        task_family="electromagnet_2d",
        supported_now=True,
        support_message="AI 响应异常，已按电磁铁需求回退到保守的二维 Maxwell 首版脚本链路。",
        summary="已按电磁铁/磁路执行器需求生成保守的首版结构化结果。",
        extracted_parameters={
            "original_requirement": requirement.strip(),
            "air_gap_mm": design.air_gap_mm,
            "current_a": design.current_a,
            "current_min_a": design.current_min_a,
            "supply_voltage_v": design.supply_voltage_v,
            "current_limit_a": design.current_limit_a,
        },
        simulation_spec=_build_minimal_spec_from_design(design.model_dump(mode="json")),
        execution_plan=_build_default_execution_plan_from_design(design.model_dump(mode="json")),
        assumptions=[
            "当云端结构化失败时，回退到保守的电磁铁二维首版模型。",
        ],
        warnings=[
            "当前结构化结果含有保守默认尺寸，后续应继续用 AI 或人工补全几何细节。",
        ],
        design=design,
    )


def _fallback_transformer_2d_intake(requirement: str) -> RequirementIntake:
    primary_v, secondary_v = _extract_transformer_voltage_pair(requirement)
    frequency_hz = _extract_frequency_hz(requirement) or 50.0
    primary_v = primary_v or 10000.0
    secondary_v = secondary_v or 220.0
    primary_turns = 500
    turns_ratio = secondary_v / primary_v if primary_v else 0.022
    secondary_turns = max(10, int(round(primary_turns * turns_ratio)))
    primary_current_a = 1.0
    return RequirementIntake(
        task_family="transformer_2d",
        supported_now=True,
        support_message="\u5df2\u6309\u4e8c\u7ef4\u53d8\u538b\u5668\u6982\u5ff5\u6a21\u578b\u751f\u6210\u53ef\u6267\u884c\u7684\u9996\u7248\u4eff\u771f\u89c4\u683c\u3002",
        summary="\u5df2\u6309\u53d8\u538b\u5668\u9700\u6c42\u751f\u6210\u4e8c\u7ef4\u6982\u5ff5\u9a8c\u8bc1\u6a21\u578b\u3002",
        extracted_parameters={
            "device_type": "transformer",
            "input_voltage_v": primary_v,
            "output_voltage_v": secondary_v,
            "frequency_hz": frequency_hz,
            "primary_turns": primary_turns,
            "secondary_turns": secondary_turns,
            "primary_current_a": primary_current_a,
        },
        simulation_spec={
            "software": "ansys_maxwell",
            "task_family": "transformer_2d",
            "geometry": {
                "type": "core_winding_transformer_2d",
                "core_width_mm": 60.0,
                "core_height_mm": 80.0,
                "core_thickness_mm": 20.0,
                "primary_coil_width_mm": 12.0,
                "secondary_coil_width_mm": 10.0,
                "coil_height_mm": 48.0,
                "window_gap_mm": 8.0,
                "region_padding_mm": 30.0,
            },
            "materials": {
                "core": "steel_1008",
                "primary": "copper",
                "secondary": "copper",
            },
            "excitations": {
                "primary_voltage_v": primary_v,
                "secondary_target_voltage_v": secondary_v,
                "frequency_hz": frequency_hz,
                "primary_current_a": primary_current_a,
                "primary_turns": primary_turns,
                "secondary_turns": secondary_turns,
            },
            "constraints": {
                "input_voltage_v": primary_v,
                "output_voltage_v": secondary_v,
                "frequency_hz": frequency_hz,
            },
            "solver": {"design_type": "Maxwell 2D", "solution_type": "Magnetostatic"},
            "required_outputs": ["secondary_voltage", "turns_ratio", "flux_density"],
            "execution_ready": True,
            "missing_inputs": [],
        },
        execution_plan={
            "software": "ansys_maxwell",
            "task_family": "transformer_2d",
            "design_type": "Maxwell 2D",
            "solution_type": "Magnetostatic",
            "project_name": "transformer_2d",
            "design_name": "Transformer2D",
            "model_units": "mm",
            "variables": {
                "primary_voltage_v": primary_v,
                "secondary_target_voltage_v": secondary_v,
                "frequency_hz": frequency_hz,
                "primary_turns": primary_turns,
                "secondary_turns": secondary_turns,
                "primary_current_a": primary_current_a,
            },
            "steps": [
                {"action": "build_geometry"},
                {"action": "assign_materials"},
                {"action": "assign_excitation"},
                {"action": "solve"},
                {"action": "extract_outputs"},
            ],
            "postprocess": ["secondary_voltage", "turns_ratio", "flux_density"],
            "execution_ready": True,
            "missing_inputs": [],
        },
        assumptions=[
            "\u9996\u7248\u9ed8\u8ba4\u630950Hz\u5de5\u9891\u6982\u5ff5\u53d8\u538b\u5668\u5904\u7406\u3002",
            "\u6b21\u7ea7\u7535\u538b\u9996\u7248\u6309\u531d\u6570\u6bd4\u4f30\u7b97\uff0c\u7528 Maxwell \u4e8c\u7ef4\u6a21\u578b\u9a8c\u8bc1\u78c1\u5bc6\u6c34\u5e73\u3002",
        ],
        warnings=[
            "\u8fd9\u662f\u53d8\u538b\u5668\u7684\u9996\u7248\u6982\u5ff5\u6a21\u578b\uff0c\u66f4\u7cbe\u786e\u7684\u6682\u6001\u7535\u78c1/\u7535\u8def\u8026\u5408\u4ecd\u9700\u540e\u7eed\u6269\u5c55\u3002",
        ],
    )


def _fallback_inductor_2d_intake(requirement: str) -> RequirementIntake:
    text = requirement.strip()
    air_gap_mm = _extract_first_float(
        text,
        [
            rf"(?:\u6c14\u9699|air\s*gap)\s*{FLOAT_PATTERN}\s*mm\b",
        ],
    ) or 0.5
    current_min_a, current_max_a = _extract_current_range(text)
    current_limit_a = current_max_a or _extract_first_float(
        text,
        [
            rf"(?:\u7535\u6d41|current).{{0,12}}?(?:\u4e0d\u8d85\u8fc7|\u4e0a\u9650|max)\s*{FLOAT_PATTERN}\s*A\b",
        ],
    )
    current_a = current_limit_a or _extract_first_float(
        text,
        [
            rf"(?:\u7535\u6d41|current)\s*(?:=|\u8bbe\u4e3a|\u53d6\u503c\u4e3a)?\s*{FLOAT_PATTERN}\s*A\b",
        ],
    ) or 2.0
    turns = _extract_turns(text, default=600)
    target_inductance_h = _extract_inductance_h(text)
    return RequirementIntake(
        task_family="inductor_2d",
        supported_now=True,
        support_message="\u5df2\u6309\u4e8c\u7ef4\u7535\u611f/\u7ebf\u5708\u78c1\u8def\u6a21\u578b\u751f\u6210\u53ef\u6267\u884c\u9996\u7248\u4eff\u771f\u89c4\u683c\u3002",
        summary="\u5df2\u6309\u7535\u611f\u7c7b\u9700\u6c42\u751f\u6210\u4e8c\u7ef4\u78c1\u9759\u6001\u9a8c\u8bc1\u6a21\u578b\u3002",
        extracted_parameters={
            "device_type": "inductor",
            "air_gap_mm": air_gap_mm,
            "current_a": current_a,
            "current_min_a": current_min_a,
            "current_limit_a": current_limit_a,
            "coil_turns": turns,
            "target_inductance_h": target_inductance_h,
        },
        simulation_spec={
            "software": "ansys_maxwell",
            "task_family": "inductor_2d",
            "geometry": {
                "type": "core_coil_inductor_2d",
                "air_gap_mm": air_gap_mm,
                "core_width_mm": 30.0,
                "core_height_mm": 45.0,
                "core_thickness_mm": 12.0,
                "coil_width_mm": 16.0,
                "coil_height_mm": 22.0,
                "region_padding_mm": 20.0,
            },
            "materials": {"core": "steel_1008", "coil": "copper"},
            "excitations": {"current_a": current_a, "coil_turns": turns},
            "constraints": {
                "current_min_a": current_min_a,
                "current_limit_a": current_limit_a,
                "target_inductance_h": target_inductance_h,
            },
            "solver": {"design_type": "Maxwell 2D", "solution_type": "Magnetostatic"},
            "required_outputs": ["inductance", "flux_density"],
            "execution_ready": True,
            "missing_inputs": [],
        },
        execution_plan={
            "software": "ansys_maxwell",
            "task_family": "inductor_2d",
            "design_type": "Maxwell 2D",
            "solution_type": "Magnetostatic",
            "project_name": "inductor_2d",
            "design_name": "Inductor2D",
            "model_units": "mm",
            "variables": {
                "air_gap_mm": air_gap_mm,
                "coil_turns": turns,
                "current_a": current_a,
            },
            "steps": [
                {"action": "build_geometry"},
                {"action": "assign_materials"},
                {"action": "assign_excitation"},
                {"action": "solve"},
                {"action": "extract_outputs"},
            ],
            "postprocess": ["inductance", "flux_density"],
            "execution_ready": True,
            "missing_inputs": [],
        },
        assumptions=[
            "\u9996\u7248\u9ed8\u8ba4\u4e3a\u5e26\u5c0f\u6c14\u9699\u7684\u4e8c\u7ef4\u78c1\u8def\u7535\u611f\u6a21\u578b\u3002",
        ],
        warnings=[
            "\u9996\u7248\u7535\u611f\u7ed3\u679c\u91c7\u7528 Maxwell \u78c1\u573a + \u7b80\u5316\u7535\u611f\u4f30\u7b97\u8054\u5408\u8f93\u51fa\u3002",
        ],
    )


def _fallback_solenoid_2d_intake(requirement: str) -> RequirementIntake:
    text = requirement.strip()
    lower = text.lower()
    length_mm = _extract_mm_value(text, ["\u957f\u5ea6", "length"]) or 50.0
    radius_mm = _extract_mm_value(text, ["\u534a\u5f84", "radius"]) or 8.0
    current_a = _extract_current_a(text, 1.0)
    turns = _extract_turns(text, 300)
    required_outputs = ["center_flux_density", "max_flux_density"]
    if "\u78c1\u573a" in text or "magnetic field" in lower:
        required_outputs.append("magnetic_field")

    return RequirementIntake(
        task_family="solenoid_2d",
        supported_now=True,
        support_message="\u5df2\u6309\u4e8c\u7ef4\u7a7a\u6c14\u82af\u87ba\u7ebf\u7ba1\u6a21\u578b\u751f\u6210\u53ef\u6267\u884c\u4eff\u771f\u89c4\u683c\u3002",
        summary="\u5df2\u6309\u87ba\u7ebf\u7ba1/\u7a7a\u5fc3\u7ebf\u5708\u9700\u6c42\u751f\u6210\u4e8c\u7ef4\u78c1\u9759\u6001\u9a8c\u8bc1\u4efb\u52a1\u3002",
        extracted_parameters={
            "device_type": "solenoid",
            "length_mm": length_mm,
            "radius_mm": radius_mm,
            "current_a": current_a,
            "coil_turns": turns,
        },
        simulation_spec={
            "software": "ansys_maxwell",
            "task_family": "solenoid_2d",
            "geometry": {
                "type": "air_core_solenoid_2d",
                "length_mm": length_mm,
                "radius_mm": radius_mm,
                "coil_thickness_mm": 2.0,
                "region_padding_mm": 25.0,
            },
            "materials": {"coil": "copper", "core": "vacuum"},
            "excitations": {"current_a": current_a, "coil_turns": turns},
            "constraints": {},
            "solver": {"design_type": "Maxwell 2D", "solution_type": "Magnetostatic"},
            "required_outputs": required_outputs,
            "execution_ready": True,
            "missing_inputs": [],
        },
        execution_plan={
            "software": "ansys_maxwell",
            "task_family": "solenoid_2d",
            "design_type": "Maxwell 2D",
            "solution_type": "Magnetostatic",
            "project_name": "solenoid_2d",
            "design_name": "Solenoid2D",
            "model_units": "mm",
            "variables": {
                "length_mm": length_mm,
                "radius_mm": radius_mm,
                "current_a": current_a,
                "coil_turns": turns,
            },
            "steps": [
                {"action": "build_geometry"},
                {"action": "assign_materials"},
                {"action": "assign_excitation"},
                {"action": "solve"},
                {"action": "extract_outputs"},
            ],
            "postprocess": required_outputs,
            "execution_ready": True,
            "missing_inputs": [],
        },
        assumptions=[
            "\u9996\u7248\u6309\u7a7a\u6c14\u82af\u4e8c\u7ef4\u7b49\u6548\u7ebf\u5708\u6a21\u578b\u5904\u7406\u3002",
            "\u7ebf\u5708\u531d\u6570\u7528\u4e8e\u7b49\u6548\u5b89\u531d\u6570\u548c\u4e2d\u5fc3\u78c1\u5bc6\u53c2\u8003\u4f30\u7b97\u3002",
        ],
        warnings=[
            "\u8be5\u6a21\u578b\u7528\u4e8e\u7a7a\u5fc3\u87ba\u7ebf\u7ba1\u9996\u7248\u9a8c\u8bc1\uff0c\u4e0d\u7b49\u540c\u4e8e\u590d\u6742\u591a\u5c42\u7ed5\u7ec4\u4e09\u7ef4\u6a21\u578b\u3002",
        ],
    )


def _fallback_coaxial_capacitor_2d_intake(requirement: str) -> RequirementIntake:
    text = requirement.strip()
    inner_radius_mm = _extract_mm_value(text, ["\u5185\u534a\u5f84", "inner radius"]) or 1.0
    outer_radius_mm = _extract_mm_value(text, ["\u5916\u534a\u5f84", "outer radius"]) or 5.0
    if outer_radius_mm <= inner_radius_mm:
        outer_radius_mm = inner_radius_mm * 3.0
    voltage_mentions = _extract_voltage_mentions(text)
    voltage_v = voltage_mentions[0] if voltage_mentions else 100.0
    constraints = _build_capacitor_constraints(text)

    return RequirementIntake(
        task_family="coaxial_capacitor_2d",
        supported_now=True,
        support_message="\u5df2\u6309\u4e8c\u7ef4\u540c\u8f74\u7535\u5bb9/\u7535\u7f06\u622a\u9762\u6a21\u578b\u751f\u6210\u53ef\u6267\u884c\u4eff\u771f\u89c4\u683c\u3002",
        summary="\u5df2\u6309\u540c\u8f74\u7535\u5bb9\u9700\u6c42\u751f\u6210\u4e8c\u7ef4\u9759\u7535\u573a\u4efb\u52a1\u3002",
        extracted_parameters={
            "original_requirement": text,
            "device_type": "coaxial_capacitor",
            "inner_radius_mm": inner_radius_mm,
            "outer_radius_mm": outer_radius_mm,
            "voltage_v": voltage_v,
            **constraints,
        },
        simulation_spec={
            "software": "ansys_maxwell",
            "task_family": "coaxial_capacitor_2d",
            "geometry": {
                "type": "coaxial_capacitor_2d",
                "inner_radius_mm": inner_radius_mm,
                "outer_radius_mm": outer_radius_mm,
                "region_padding_mm": 8.0,
            },
            "materials": {"inner": "pec", "outer": "pec", "dielectric": "vacuum"},
            "excitations": {"voltage_V": voltage_v},
            "constraints": constraints,
            "solver": {"design_type": "Maxwell 2D", "solution_type": "ElectrostaticXY"},
            "required_outputs": ["capacitance", "electric_field"],
            "execution_ready": True,
            "missing_inputs": [],
        },
        execution_plan={
            "software": "ansys_maxwell",
            "task_family": "coaxial_capacitor_2d",
            "design_type": "Maxwell 2D",
            "solution_type": "ElectrostaticXY",
            "project_name": "coaxial_capacitor_2d",
            "design_name": "CoaxialCapacitor2D",
            "model_units": "mm",
            "variables": {
                "inner_radius_mm": inner_radius_mm,
                "outer_radius_mm": outer_radius_mm,
                "voltage_V": voltage_v,
            },
            "steps": [
                {"action": "build_geometry"},
                {"action": "assign_voltage"},
                {"action": "solve"},
                {"action": "extract_outputs"},
            ],
            "postprocess": ["capacitance", "electric_field"],
            "execution_ready": True,
            "missing_inputs": [],
        },
        assumptions=[
            "\u9996\u7248\u6309\u771f\u7a7a\u4ecb\u8d28\u548c\u4e8c\u7ef4\u622a\u9762\u5904\u7406\uff0c\u7ed3\u679c\u7ed9\u51fa\u6bcf\u7c73\u957f\u5ea6\u53c2\u8003\u7535\u5bb9\u3002",
        ],
        warnings=[
            "\u5b9e\u9645\u7535\u7f06\u8bbe\u8ba1\u8fd8\u9700\u8981\u8865\u5145\u4ecb\u8d28\u6750\u6599\u3001\u5c4f\u853d\u5c42\u548c\u7edd\u7f18\u8981\u6c42\u3002",
        ],
    )


def _fallback_busbar_2d_intake(requirement: str) -> RequirementIntake:
    text = requirement.strip()
    lower = text.lower()
    width_mm = _extract_mm_value(text, ["\u5bbd", "\u5bbd\u5ea6", "width"]) or 10.0
    thickness_mm = _extract_mm_value(text, ["\u539a", "\u539a\u5ea6", "thickness"]) or 2.0
    current_a = _extract_current_a(text, 200.0)
    constraints = _build_busbar_constraints(text)
    required_outputs = ["flux_density", "current_density"]
    if "\u78c1\u573a" in text or "magnetic field" in lower:
        required_outputs.append("magnetic_field")

    return RequirementIntake(
        task_family="busbar_2d",
        supported_now=True,
        support_message="\u5df2\u6309\u4e8c\u7ef4\u8f7d\u6d41\u6bcd\u6392/\u94dc\u6392\u6a21\u578b\u751f\u6210\u53ef\u6267\u884c\u4eff\u771f\u89c4\u683c\u3002",
        summary="\u5df2\u6309\u8f7d\u6d41\u5bfc\u4f53/\u6bcd\u6392\u9700\u6c42\u751f\u6210\u4e8c\u7ef4\u78c1\u9759\u6001\u4efb\u52a1\u3002",
        extracted_parameters={
            "device_type": "busbar",
            "width_mm": width_mm,
            "thickness_mm": thickness_mm,
            "current_a": current_a,
            **constraints,
        },
        simulation_spec={
            "software": "ansys_maxwell",
            "task_family": "busbar_2d",
            "geometry": {
                "type": "rectangular_busbar_2d",
                "width_mm": width_mm,
                "thickness_mm": thickness_mm,
                "region_padding_mm": 20.0,
            },
            "materials": {"conductor": "copper", "ambient": "vacuum"},
            "excitations": {"current_a": current_a},
            "constraints": constraints,
            "solver": {"design_type": "Maxwell 2D", "solution_type": "Magnetostatic"},
            "required_outputs": required_outputs,
            "execution_ready": True,
            "missing_inputs": [],
        },
        execution_plan={
            "software": "ansys_maxwell",
            "task_family": "busbar_2d",
            "design_type": "Maxwell 2D",
            "solution_type": "Magnetostatic",
            "project_name": "busbar_2d",
            "design_name": "Busbar2D",
            "model_units": "mm",
            "variables": {
                "width_mm": width_mm,
                "thickness_mm": thickness_mm,
                "current_a": current_a,
            },
            "steps": [
                {"action": "build_geometry"},
                {"action": "assign_materials"},
                {"action": "assign_excitation"},
                {"action": "solve"},
                {"action": "extract_outputs"},
            ],
            "postprocess": required_outputs,
            "execution_ready": True,
            "missing_inputs": [],
        },
        assumptions=[
            "\u9996\u7248\u6309\u5355\u6839\u76f4\u7ebf\u77e9\u5f62\u94dc\u6392\u4e8c\u7ef4\u622a\u9762\u5904\u7406\u3002",
            "\u7535\u6d41\u5bc6\u5ea6\u9996\u7248\u6309\u51e0\u4f55\u622a\u9762\u79ef\u8fdb\u884c\u5747\u5300\u4f30\u7b97\u3002",
        ],
        warnings=[
            "\u8be5\u6a21\u578b\u9002\u5408\u6bcd\u6392/\u8f7d\u6d41\u5bfc\u4f53\u7684\u9996\u7248\u78c1\u573a\u9a8c\u8bc1\uff0c\u6682\u672a\u8026\u5408\u70ed\u6548\u5e94\u548c\u591a\u5bfc\u4f53\u76f8\u4e92\u4f5c\u7528\u3002",
        ],
    )




def _fallback_intake_from_requirement(requirement: str) -> RequirementIntake:
    text = requirement.strip()
    if _looks_like_annular_conductor_requirement(text):
        return _fallback_annular_conductor_intake(text)
    if _looks_like_coaxial_capacitor_requirement(text):
        return _fallback_coaxial_capacitor_2d_intake(text)
    if _looks_like_busbar_requirement(text):
        return _fallback_busbar_2d_intake(text)
    if _looks_like_solenoid_requirement(text):
        return _fallback_solenoid_2d_intake(text)
    if _looks_like_transformer_requirement(text):
        return _fallback_transformer_2d_intake(text)
    if _looks_like_capacitor_requirement(text):
        return _fallback_capacitor_intake(text)
    if _looks_like_inductor_requirement(text):
        return _fallback_inductor_2d_intake(text)
    if _looks_like_electromagnet_requirement(text):
        return _fallback_electromagnet_intake(text)
    return _fallback_unknown_intake(text)


def _rescue_supported_fallback_intake(requirement: str, intake: RequirementIntake) -> RequirementIntake:
    text = requirement.strip()
    if intake.supported_now and (
        bool(intake.simulation_spec.get("execution_ready")) or bool(intake.execution_plan.get("execution_ready"))
    ):
        return intake
    semantic_hint = infer_builder_hint(intake, requirement=text)
    if semantic_hint == "capacitor_2d":
        return _fallback_capacitor_intake(text)
    if semantic_hint == "transformer_2d":
        return _fallback_transformer_2d_intake(text)
    if semantic_hint == "inductor_2d":
        return _fallback_inductor_2d_intake(text)
    if semantic_hint == "solenoid_2d":
        return _fallback_solenoid_2d_intake(text)
    if semantic_hint == "coaxial_capacitor_2d":
        return _fallback_coaxial_capacitor_2d_intake(text)
    if semantic_hint == "busbar_2d":
        return _fallback_busbar_2d_intake(text)
    if _looks_like_coaxial_capacitor_requirement(text):
        return _fallback_coaxial_capacitor_2d_intake(text)
    if _looks_like_annular_conductor_requirement(text):
        return _fallback_annular_conductor_intake(text)
    if _looks_like_capacitor_requirement(text):
        return _fallback_capacitor_intake(text)
    if _looks_like_transformer_requirement(text):
        return _fallback_transformer_2d_intake(text)
    if _looks_like_inductor_requirement(text):
        return _fallback_inductor_2d_intake(text)
    if _looks_like_solenoid_requirement(text):
        return _fallback_solenoid_2d_intake(text)
    if _looks_like_busbar_requirement(text):
        return _fallback_busbar_2d_intake(text)
    if _looks_like_electromagnet_requirement(text):
        return _fallback_electromagnet_intake(text)
    return intake


def _task_family_matches_requirement_text(task_family: str, requirement: str) -> bool:
    text = requirement.strip()
    if task_family == "capacitor_2d":
        return _looks_like_capacitor_requirement(text)
    if task_family == "coaxial_capacitor_2d":
        return _looks_like_coaxial_capacitor_requirement(text)
    if task_family == "busbar_2d":
        return _looks_like_busbar_requirement(text)
    if task_family == "transformer_2d":
        return _looks_like_transformer_requirement(text)
    if task_family == "inductor_2d":
        return _looks_like_inductor_requirement(text)
    if task_family == "solenoid_2d":
        return _looks_like_solenoid_requirement(text)
    if task_family == "electromagnet_2d":
        return _looks_like_electromagnet_requirement(text)
    return True


def _validate_intake_payload(payload: dict[str, Any], requirement: str) -> RequirementIntake:
    normalized = _normalize_intake_payload(payload, requirement)
    try:
        return RequirementIntake.model_validate(normalized)
    except ValidationError as exc:
        raise RequirementPlanningError("当前需求的结构化结果不符合系统约定格式。") from exc


def _validate_design_payload(payload: dict[str, Any], requirement: str) -> ElectromagnetDesign:
    normalized = _normalize_design_payload(payload, requirement)
    try:
        return ElectromagnetDesign.model_validate(normalized)
    except ValidationError as exc:
        raise RequirementPlanningError("设计修正结果不符合电磁铁设计格式。") from exc


def _validate_design_patch_payload(payload: dict[str, Any]) -> ElectromagnetDesignPatch:
    normalized = dict(payload)
    normalized["assumptions"] = _ensure_list(normalized.get("assumptions"))
    normalized["warnings"] = _ensure_list(normalized.get("warnings"))
    if "summary" in normalized and normalized["summary"] is not None:
        normalized["summary"] = str(normalized["summary"]).strip() or None
    try:
        return ElectromagnetDesignPatch.model_validate(normalized)
    except ValidationError as exc:
        raise RequirementPlanningError("LLM 反馈修正结果不符合设计补丁格式。") from exc


def _estimate_design_supply_current(design: ElectromagnetDesign) -> float | None:
    if design.supply_voltage_v is None or design.coil_turns <= 0:
        return None
    window_area_mm2 = design.coil_width_mm * design.coil_height_mm
    if window_area_mm2 <= 0:
        return None
    conductor_area_m2 = (window_area_mm2 * 0.6 / design.coil_turns) * 1e-6
    if conductor_area_m2 <= 0:
        return None
    mean_turn_length_mm = 2.0 * (design.coil_height_mm + design.coil_width_mm + design.core_thickness_mm)
    total_length_m = design.coil_turns * mean_turn_length_mm * 1e-3
    coil_resistance = 1.724e-8 * total_length_m / conductor_area_m2
    if coil_resistance <= 0:
        return None
    return design.supply_voltage_v / coil_resistance


def _enforce_design_current_voltage_constraints(design: ElectromagnetDesign) -> ElectromagnetDesign:
    if design.supply_voltage_v is None or design.current_limit_a is None:
        return design
    limit = design.current_limit_a
    estimate = _estimate_design_supply_current(design)
    if estimate is None:
        return design

    turns = design.coil_turns
    coil_width = design.coil_width_mm
    coil_height = design.coil_height_mm
    for _ in range(16):
        if estimate <= limit + 1e-9:
            break
        if turns < 10000:
            ratio = max(1.2, min(2.0, (estimate / limit) ** 0.5))
            turns = min(10000, max(turns + 1, int(round(turns * ratio))))
        elif coil_width > 1.0:
            coil_width = max(1.0, coil_width * 0.85)
        elif coil_height > 1.0:
            coil_height = max(1.0, coil_height * 0.85)
        trial = design.model_copy(
            update={
                "coil_turns": turns,
                "coil_width_mm": coil_width,
                "coil_height_mm": coil_height,
                "current_a": min(design.current_a, limit),
            }
        )
        estimate = _estimate_design_supply_current(trial)
        if estimate is None:
            break

    return design.model_copy(
        update={
            "coil_turns": turns,
            "coil_width_mm": coil_width,
            "coil_height_mm": coil_height,
            "current_a": min(design.current_a, limit),
        }
    )


def _validate_script_payload(payload: dict[str, Any]) -> GeneratedMaxwellScript:
    normalized = _normalize_script_payload(payload)
    try:
        return GeneratedMaxwellScript.model_validate(normalized)
    except ValidationError as exc:
        raise RequirementPlanningError("脚本生成结果不符合系统约定格式。") from exc


def _build_local_script_from_design(design: ElectromagnetDesign) -> GeneratedMaxwellScript:
    code = dedent(
        """
        from ansys.aedt.core import Maxwell2d


        def run_job(job: dict) -> dict:
            design = dict(job.get("design") or {})
            project_file = str(job["project_file"])
            version = job.get("maxwell_version")
            non_graphical = bool(job.get("non_graphical", True))
            student_version = bool(job.get("student_version", False))

            air_gap = float(design.get("air_gap_mm", 2.0))
            core_width = float(design.get("core_width_mm", 20.0))
            core_height = float(design.get("core_height_mm", 40.0))
            core_thickness = float(design.get("core_thickness_mm", 10.0))
            coil_width = float(design.get("coil_width_mm", 12.0))
            coil_height = float(design.get("coil_height_mm", 20.0))
            region_padding = float(design.get("region_padding_mm", 20.0))
            current_a = float(design.get("current_a", 1.0))
            core_material = str(design.get("core_material", "steel_1008"))
            coil_material = str(design.get("coil_material", "copper"))

            outputs: dict[str, float | str] = {}
            with Maxwell2d(
                project=project_file,
                design="Electromagnet2D",
                solution_type="Magnetostatic",
                version=version,
                non_graphical=non_graphical,
                new_desktop=True,
                close_on_exit=True,
                student_version=student_version,
            ) as app:
                app.modeler.model_units = "mm"
                core = app.modeler.create_rectangle(
                    origin=["0mm", "0mm", "0mm"],
                    sizes=[f"{core_width}mm", f"{core_height}mm"],
                    name="core",
                    material=core_material,
                )
                armature = app.modeler.create_rectangle(
                    origin=[f"{core_width + air_gap}mm", "0mm", "0mm"],
                    sizes=[f"{core_thickness}mm", f"{core_height}mm"],
                    name="armature",
                    material=core_material,
                )
                coil = app.modeler.create_rectangle(
                    origin=[f"{-coil_width}mm", f"{(core_height - coil_height) / 2}mm", "0mm"],
                    sizes=[f"{coil_width}mm", f"{coil_height}mm"],
                    name="coil",
                    material=coil_material,
                )
                region = app.modeler.create_region(
                    pad_value=[
                        f"{region_padding}mm",
                        f"{region_padding}mm",
                        f"{region_padding}mm",
                        f"{region_padding}mm",
                    ],
                    pad_type="Absolute Offset",
                    name="Region",
                )
                app.assign_current("coil", amplitude=f"{current_a}A", name="Current1")
                app.assign_force(["armature"], force_name="Force1")
                try:
                    app.assign_balloon(region.edges, boundary="OuterRegion")
                except Exception:
                    pass
                app.create_setup(name="Setup1", setup_type="Magnetostatic")
                app.save_project()
                solve_ok = bool(app.analyze_setup("Setup1"))
                outputs["solve_status"] = "completed" if solve_ok else "failed"
                if not solve_ok:
                    outputs["status"] = "failed"
                    outputs["notes"] = "Maxwell solve failed."
                    outputs["project_name"] = app.project_name
                    outputs["design_name"] = app.design_name
                    app.save_project()
                    return outputs
                try:
                    outputs["max_flux_density_t"] = float(
                        app.post.get_scalar_field_value("Mag_B", "Maximum", object_name="AllObjects")
                    )
                except Exception as exc:
                    outputs["max_flux_density_note"] = f"Postprocess skipped: {exc}"
                outputs["project_name"] = app.project_name
                outputs["design_name"] = app.design_name
                app.save_project()
            return outputs
        """
    ).strip()
    return GeneratedMaxwellScript(
        filename="generated_maxwell_job.py",
        entrypoint="run_job",
        summary="根据电磁铁结构化参数生成本地 PyAEDT 脚本。",
        code=code,
        assumptions=["当云端脚本生成不可用时，回退到本地确定性脚本。"],
        warnings=[],
)


def _extract_ir_payload_from_intake(intake: RequirementIntake) -> dict[str, Any] | None:
    simulation_spec = intake.simulation_spec if isinstance(intake.simulation_spec, dict) else {}
    execution_plan = intake.execution_plan if isinstance(intake.execution_plan, dict) else {}
    ir_payload = simulation_spec.get("ir_plan")
    if isinstance(ir_payload, dict):
        return ir_payload
    ir_payload = execution_plan.get("ir_plan")
    if isinstance(ir_payload, dict):
        return ir_payload
    return None


def _render_ir_script_for_intake(
    intake: RequirementIntake,
    plan: MaxwellIRPlan,
    *,
    summary: str | None = None,
    assumptions: list[str] | None = None,
    warnings: list[str] | None = None,
) -> GeneratedMaxwellScript:
    merged_assumptions = _merge_unique_strings(
        intake.assumptions,
        assumptions or [],
        ["本脚本由统一 Maxwell IR 渲染器生成。"],
    )
    merged_warnings = _merge_unique_strings(intake.warnings, warnings or [])
    return render_script_from_ir(
        plan,
        summary=summary or intake.summary,
        assumptions=merged_assumptions,
        warnings=merged_warnings,
    )


def _build_local_script_from_ir_payload(intake: RequirementIntake) -> GeneratedMaxwellScript:
    ir_payload = _extract_ir_payload_from_intake(intake)
    if not ir_payload:
        raise RequirementPlanningError("当前任务未提供可渲染的 Maxwell IR。")
    try:
        plan = validate_ir_plan(MaxwellIRPlan.model_validate(ir_payload))
    except ValidationError as exc:
        raise RequirementPlanningError("当前任务提供的 Maxwell IR 不符合约定格式。") from exc
    except ValueError as exc:
        raise RequirementPlanningError(f"当前任务提供的 Maxwell IR 语义无效: {exc}") from exc
    return _render_ir_script_for_intake(intake, plan)


def _validate_ir_artifact_payload(payload: dict[str, Any]) -> GeneratedIRPlan:
    normalized = _normalize_ir_artifact_payload(payload)
    try:
        normalized["ir_plan"] = validate_ir_plan(MaxwellIRPlan.model_validate(normalized["ir_plan"]))
        return GeneratedIRPlan.model_validate(normalized)
    except ValidationError as exc:
        raise RequirementPlanningError("AI 生成的 Maxwell IR 不符合结构约定。") from exc
    except ValueError as exc:
        raise RequirementPlanningError(f"AI 生成的 Maxwell IR 语义无效: {exc}") from exc


def _validate_ir_patch_payload(payload: dict[str, Any]) -> IRPatch:
    if not isinstance(payload, dict):
        raise RequirementPlanningError("AI 生成的 IR 补丁不是 JSON 对象。")
    try:
        return IRPatch.model_validate(payload)
    except ValidationError as exc:
        raise RequirementPlanningError("AI 生成的 IR 补丁不符合结构约定。") from exc


def _normalize_ir_artifact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["summary"] = str(normalized.get("summary") or "已生成通用 Maxwell IR 方案。").strip()
    normalized["assumptions"] = _ensure_list(normalized.get("assumptions"))
    normalized["warnings"] = _ensure_list(normalized.get("warnings"))
    ir_payload = normalized.get("ir_plan")
    if not isinstance(ir_payload, dict):
        raise RequirementPlanningError("IR 结果缺少 ir_plan 对象。")
    normalized["ir_plan"] = _normalize_ir_plan_payload(ir_payload)
    if str(normalized["ir_plan"].get("driver") or "Maxwell2d") != "Maxwell2d":
        normalized["ir_plan"]["driver"] = "Maxwell2d"
        normalized["warnings"] = _merge_unique_strings(
            normalized["warnings"],
            ["当前执行器已将非二维 IR 自动降级为 Maxwell 2D 近似执行。"],
        )
    return normalized


def _normalize_ir_plan_payload(ir_payload: dict[str, Any]) -> dict[str, Any]:
    if "objects" in ir_payload and "assignments" in ir_payload:
        objects = ir_payload.get("objects") or []
        assignments = ir_payload.get("assignments") or []
        if objects and assignments and isinstance(objects, list) and isinstance(assignments, list):
            normalized_outputs = [
                item
                for item in (_normalize_ir_derived_output_item(entry) for entry in ir_payload.get("derived_outputs") or [])
                if item is not None
            ]
            solution_type = _normalize_ir_solution_type(ir_payload.get("solution_type"), "Magnetostatic")
            return {
                "driver": _normalize_ir_driver(ir_payload.get("driver")),
                "design_name": str(ir_payload.get("design_name") or "GenericIRDesign"),
                "solution_type": solution_type,
                "model_units": str(ir_payload.get("model_units") or "mm"),
                "setup_name": str(ir_payload.get("setup_name") or "Setup1"),
                "setup_type": _normalize_ir_setup_type(
                    ir_payload.get("setup_type"),
                    "Electrostatic" if solution_type == "ElectrostaticXY" else solution_type,
                ),
                "parameters": [_normalize_ir_parameter_item(item) for item in ir_payload.get("parameters") or []],
                "locals": [_normalize_ir_local_item(item) for item in ir_payload.get("locals") or []],
                "objects": [_normalize_ir_object_item(item) for item in objects],
                "operations": [_normalize_ir_operation_item(item) for item in ir_payload.get("operations") or []],
                "assignments": [_normalize_ir_assignment_item(item) for item in assignments],
                "derived_outputs": [
                    {key: value for key, value in item.items() if key != "__kind__"}
                    for item in normalized_outputs
                    if item.get("__kind__") == "derived"
                ],
                "postprocess": [
                    {key: value for key, value in item.items() if key != "__kind__"}
                    for item in normalized_outputs
                    if item.get("__kind__") == "postprocess"
                ]
                + [_normalize_ir_postprocess_item(item) for item in ir_payload.get("postprocess") or []],
                "failure_note": str(ir_payload.get("failure_note") or "Maxwell solve failed."),
            }
    return ir_payload


def _sanitize_identifier(value: Any, prefix: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = prefix
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = prefix
    if re.match(r"^[0-9]", text):
        text = f"{prefix}_{text}"
    return text


def _normalize_expression_token(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    text = str(value or "").strip()
    numeric_match = re.match(r"^\s*([-+]?[0-9]+(?:\.[0-9]+)?)\s*(?:[A-Za-z]+(?:/[A-Za-z]+)?|%)\s*$", text)
    if numeric_match:
        return numeric_match.group(1)
    return text


def _coerce_scalar_default(value: Any) -> float | int | str | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    numeric = _coerce_float(value)
    if numeric is not None:
        if abs(numeric - round(numeric)) < 1e-9:
            return int(round(numeric))
        return numeric
    text = str(value or "").strip()
    unit_match = re.match(r"^\s*([-+]?[0-9]+(?:\.[0-9]+)?)", text)
    if unit_match:
        numeric = float(unit_match.group(1))
        if abs(numeric - round(numeric)) < 1e-9:
            return int(round(numeric))
        return numeric
    return text


def _infer_cast_from_default(value: float | int | str | bool) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


def _map_ir_source(value: Any) -> str:
    normalized = _normalize_task_family(value)
    mapping = {
        "geometry": "geometry",
        "geom": "geometry",
        "excitation": "excitations",
        "excitations": "excitations",
        "boundary": "boundaries",
        "boundaries": "boundaries",
        "constraint": "constraints",
        "constraints": "constraints",
        "design": "design",
    }
    return mapping.get(normalized, "design")


def _normalize_ir_driver(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "3d" in text:
        return "Maxwell3d"
    return "Maxwell2d"


def _normalize_ir_solution_type(value: Any, default: str) -> str:
    text = str(value or "").strip().lower()
    if any(token in text for token in ("magnetostatic", "静磁", "闈欑")):
        return "Magnetostatic"
    if any(token in text for token in ("electrostaticxy", "electrostatic", "静电", "闈欑數")):
        return "ElectrostaticXY" if "xy" in text or "2d" in text or default == "ElectrostaticXY" else "Electrostatic"
    return default


def _normalize_ir_setup_type(value: Any, default: str) -> str:
    text = str(value or "").strip().lower()
    if any(token in text for token in ("magnetostatic", "静磁", "闈欑")):
        return "Magnetostatic"
    if any(token in text for token in ("electrostatic", "静电", "闈欑數")):
        return "Electrostatic"
    return default


def _normalize_ir_parameter_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise RequirementPlanningError("IR parameter item must be an object.")
    name = _sanitize_identifier(item.get("name") or item.get("field") or "param", "param")
    default = _coerce_scalar_default(item.get("default", item.get("value")))
    return {
        "name": name,
        "source": _map_ir_source(item.get("source") or item.get("category")),
        "field": str(item.get("field") or name),
        "default": default,
        "cast": str(item.get("cast") or _infer_cast_from_default(default)),
    }


def _normalize_ir_local_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise RequirementPlanningError("IR local item must be an object.")
    return {
        "name": _sanitize_identifier(item.get("name") or "local", "local"),
        "expression": _normalize_expression_token(item.get("expression") or "0"),
        "cast": str(item.get("cast") or "float"),
    }


def _normalize_ir_object_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise RequirementPlanningError("IR object item must be an object.")
    kind = str(item.get("kind") or item.get("type") or "").strip().lower()
    name = str(item.get("name") or kind or "object")
    if kind == "rectangle":
        origin = item.get("origin_exprs") or item.get("origin") or item.get("corner") or ["0", "0", "0"]
        sizes = item.get("sizes_exprs") or item.get("sizes") or item.get("size") or ["1", "1"]
        origin_list = [_normalize_expression_token(value) for value in list(origin)]
        sizes_list = [_normalize_expression_token(value) for value in list(sizes)]
        if len(origin_list) == 2:
            origin_list.append("0")
        return {
            "name": name,
            "kind": "rectangle",
            "material": str(item.get("material") or "vacuum"),
            "origin_exprs": origin_list,
            "sizes_exprs": sizes_list,
        }
    if kind == "circle":
        origin = item.get("origin_exprs") or item.get("origin") or item.get("center") or ["0", "0", "0"]
        origin_list = [_normalize_expression_token(value) for value in list(origin)]
        if len(origin_list) == 2:
            origin_list.append("0")
        return {
            "name": name,
            "kind": "circle",
            "material": str(item.get("material") or "vacuum"),
            "origin_exprs": origin_list,
            "radius_expr": _normalize_expression_token(item.get("radius_expr") or item.get("radius") or "1"),
        }
    if kind == "region":
        padding = item.get("pad_value_exprs") or item.get("pad_value")
        if padding is None:
            padding_obj = item.get("padding") or {}
            if isinstance(padding_obj, dict):
                padding = [
                    padding_obj.get("left", 20),
                    padding_obj.get("right", 20),
                    padding_obj.get("bottom", 20),
                    padding_obj.get("top", 20),
                ]
            else:
                padding = [20, 20, 20, 20]
        return {
            "name": name,
            "kind": "region",
            "pad_value_exprs": [_normalize_expression_token(value) for value in list(padding)],
            "pad_type": str(item.get("pad_type") or "Absolute Offset"),
        }
    raise RequirementPlanningError(f"当前 IR 对象类型暂不支持: {kind}")


def _normalize_ir_operation_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise RequirementPlanningError("IR operation item must be an object.")
    kind = str(item.get("kind") or item.get("type") or "").strip().lower()
    if kind != "subtract":
        raise RequirementPlanningError(f"当前 IR 操作类型暂不支持: {kind}")
    blank_parts = item.get("blank_parts") or item.get("blank_list") or item.get("blank") or []
    tool_parts = item.get("tool_parts") or item.get("tool_list") or item.get("tool") or []
    return {
        "kind": "subtract",
        "blank_parts": [str(value) for value in list(blank_parts)],
        "tool_parts": [str(value) for value in list(tool_parts)],
        "keep_originals": bool(item.get("keep_originals", False)),
    }


def _normalize_ir_assignment_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise RequirementPlanningError("IR assignment item must be an object.")
    kind = str(item.get("kind") or item.get("type") or "").strip().lower()
    name = str(item.get("name") or kind or "assignment")
    if kind in {"current", "voltage"}:
        targets = item.get("targets")
        if not targets and item.get("target") is not None:
            targets = [item.get("target")]
        return {
            "name": name,
            "kind": kind,
            "targets": [str(value) for value in list(targets or [])],
            "amplitude_expr": _normalize_expression_token(
                item.get("amplitude_expr") or item.get("amplitude") or item.get("magnitude") or "0"
            ),
        }
    if kind == "balloon":
        target = item.get("targets")
        if not target and item.get("target") is not None:
            target = [item.get("target")]
        return {
            "name": name,
            "kind": "balloon",
            "targets": [str(value) for value in list(target or [])],
            "boundary_name": str(item.get("boundary_name") or item.get("boundary") or name),
            "is_voltage": bool(item.get("is_voltage")) if item.get("is_voltage") is not None else None,
        }
    if kind == "matrix":
        return {
            "name": name,
            "kind": "matrix",
            "signal_assignments": [str(value) for value in list(item.get("signal_assignments") or item.get("signal_sources") or [])],
            "ground_assignments": [str(value) for value in list(item.get("ground_assignments") or item.get("ground_sources") or [])],
        }
    raise RequirementPlanningError(f"当前 IR 激励/边界类型暂不支持: {kind}")


def _normalize_ir_derived_output_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    method = str(item.get("method") or item.get("kind") or "").strip().lower()
    output_key = str(item.get("output_key") or item.get("name") or "output")
    if method in {"", "analytic", "expression"}:
        return {
            "__kind__": "derived",
            "output_key": output_key,
            "expression": _normalize_expression_token(item.get("expression") or "0"),
            "cast": str(item.get("cast") or "float"),
            "phase": str(item.get("phase") or "before_solve"),
        }
    if method == "field_scalar":
        return {
            "__kind__": "postprocess",
            "kind": "field_scalar",
            "output_key": output_key,
            "cast": str(item.get("cast") or "float"),
            "quantity": str(item.get("quantity") or item.get("field") or "Mag_B"),
            "scalar_function": str(item.get("scalar_function") or item.get("stat") or "Maximum"),
            "object_name": str(item.get("object_name") or item.get("target") or "AllObjects"),
        }
    return None


def _normalize_ir_postprocess_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise RequirementPlanningError("IR postprocess item must be an object.")
    kind = str(item.get("kind") or item.get("type") or item.get("method") or "").strip().lower()
    if kind == "field_scalar":
        return {
            "kind": "field_scalar",
            "output_key": str(item.get("output_key") or item.get("name") or "field_output"),
            "cast": str(item.get("cast") or "float"),
            "quantity": str(item.get("quantity") or item.get("field") or "Mag_B"),
            "scalar_function": str(item.get("scalar_function") or item.get("stat") or "Maximum"),
            "object_name": str(item.get("object_name") or item.get("target") or "AllObjects"),
            "solution_expr": item.get("solution_expr"),
            "object_type": item.get("object_type"),
            "error_output_key": item.get("error_output_key"),
        }
    if kind == "matrix_export_value":
        return {
            "kind": "matrix_export_value",
            "output_key": str(item.get("output_key") or item.get("name") or "matrix_value"),
            "cast": str(item.get("cast") or "float"),
            "setup_name": item.get("setup_name"),
            "matrix_assignment_name": str(item.get("matrix_assignment_name") or item.get("matrix_name") or ""),
            "output_filename": str(item.get("output_filename") or "matrix.txt"),
            "regex_pattern": str(item.get("regex_pattern") or ""),
            "scaled_output_key": item.get("scaled_output_key"),
            "scale": item.get("scale"),
            "error_output_key": item.get("error_output_key"),
        }
    raise RequirementPlanningError(f"当前 IR 后处理类型暂不支持: {kind}")


def _attach_ir_artifact_to_intake(intake: RequirementIntake, artifact: GeneratedIRPlan) -> RequirementIntake:
    simulation_spec = dict(intake.simulation_spec or {})
    execution_plan = dict(intake.execution_plan or {})
    simulation_spec["ir_plan"] = artifact.ir_plan.model_dump(mode="json")
    simulation_spec["execution_ready"] = True
    simulation_spec.setdefault("software", "ansys_maxwell")
    simulation_spec.setdefault("task_family", intake.task_family or "generic_maxwell")
    solver = dict(simulation_spec.get("solver") or {})
    solver["design_type"] = "Maxwell 2D" if artifact.ir_plan.driver == "Maxwell2d" else "Maxwell 3D"
    solver["solution_type"] = artifact.ir_plan.solution_type
    simulation_spec["solver"] = solver
    execution_plan["ir_plan"] = artifact.ir_plan.model_dump(mode="json")
    execution_plan["execution_ready"] = True
    execution_plan.setdefault("software", "ansys_maxwell")
    execution_plan.setdefault("task_family", intake.task_family or "generic_maxwell")
    execution_plan["design_type"] = "Maxwell 2D" if artifact.ir_plan.driver == "Maxwell2d" else "Maxwell 3D"
    execution_plan["solution_type"] = artifact.ir_plan.solution_type
    execution_plan["design_name"] = artifact.ir_plan.design_name
    execution_plan["model_units"] = artifact.ir_plan.model_units
    intake.simulation_spec = simulation_spec
    intake.execution_plan = execution_plan
    intake.supported_now = True
    intake.summary = artifact.summary or intake.summary
    intake.assumptions = _merge_unique_strings(intake.assumptions, artifact.assumptions)
    intake.warnings = _merge_unique_strings(intake.warnings, artifact.warnings)
    _apply_ir_parameter_defaults_to_intake(intake, artifact.ir_plan)
    return intake


def _apply_ir_parameter_defaults_to_intake(intake: RequirementIntake, plan: MaxwellIRPlan) -> RequirementIntake:
    simulation_spec = dict(intake.simulation_spec or {})
    execution_plan = dict(intake.execution_plan or {})
    extracted_parameters = dict(intake.extracted_parameters or {})
    execution_variables = dict(execution_plan.get("variables") or {})
    design_payload = intake.design.model_dump(mode="json") if intake.design is not None else None
    design_changed = False

    for binding in plan.parameters:
        value = _normalize_json_like(binding.default)
        execution_variables[binding.name] = value
        extracted_parameters[binding.name] = value
        if binding.source in {"geometry", "excitations", "boundaries", "constraints"}:
            existing_section = simulation_spec.get(binding.source)
            if isinstance(existing_section, dict):
                section = dict(existing_section)
            else:
                section = {}
                if existing_section not in (None, {}, []):
                    simulation_spec.setdefault(f"{binding.source}_notes", _normalize_json_like(existing_section))
            section[binding.field] = value
            simulation_spec[binding.source] = section
            extracted_parameters[binding.field] = value
        elif binding.source == "design" and isinstance(design_payload, dict):
            design_payload[binding.field] = value
            extracted_parameters[binding.field] = value
            design_changed = True

    execution_plan["variables"] = execution_variables
    intake.simulation_spec = simulation_spec
    intake.execution_plan = execution_plan
    intake.extracted_parameters = _normalize_json_like(extracted_parameters)
    if design_changed and isinstance(design_payload, dict):
        try:
            intake.design = ElectromagnetDesign.model_validate(design_payload)
        except ValidationError:
            pass
    return intake


def _build_ir_artifact_from_intake(intake: RequirementIntake) -> GeneratedIRPlan:
    ir_payload = _extract_ir_payload_from_intake(intake)
    if isinstance(ir_payload, dict):
        try:
            return GeneratedIRPlan(
                summary=intake.summary or "已生成通用 Maxwell IR 方案。",
                ir_plan=validate_ir_plan(MaxwellIRPlan.model_validate(ir_payload)),
                assumptions=list(intake.assumptions),
                warnings=list(intake.warnings),
            )
        except ValidationError as exc:
            raise RequirementPlanningError("当前 intake 中的 Maxwell IR 不符合结构约定。") from exc
        except ValueError as exc:
            raise RequirementPlanningError(f"当前 intake 中的 Maxwell IR 语义无效: {exc}") from exc

    local_builder = {
        "capacitor_2d": _build_capacitor_ir_plan,
        "busbar_2d": _build_busbar_ir_plan,
        "solenoid_2d": _build_solenoid_ir_plan,
        "coaxial_capacitor_2d": _build_coaxial_capacitor_ir_plan,
        "generic_maxwell": _build_generic_2d_ir_plan,
    }.get(infer_builder_hint(intake))
    if local_builder is None:
        raise RequirementPlanningError("当前 intake 暂时没有可直接修订的 Maxwell IR。")
    try:
        return GeneratedIRPlan(
            summary=intake.summary or "已根据当前任务合成本地 Maxwell IR。",
            ir_plan=validate_ir_plan(local_builder(intake)),
            assumptions=list(intake.assumptions),
            warnings=list(intake.warnings),
        )
    except ValueError as exc:
        raise RequirementPlanningError(f"当前任务合成的 Maxwell IR 语义无效: {exc}") from exc


def _compact_feedback_evaluation(evaluation: Any | None) -> dict[str, Any]:
    evaluation_payload = (evaluation.model_dump(mode="json") if hasattr(evaluation, "model_dump") else evaluation) or {}
    if not isinstance(evaluation_payload, dict):
        return {}
    compact_checks: list[dict[str, Any]] = []
    for check in evaluation_payload.get("checks", []):
        if not isinstance(check, dict):
            continue
        status = str(check.get("status") or "").strip().lower()
        if status not in {"failed", "unverified"}:
            continue
        compact_checks.append(
            {
                "name": check.get("name"),
                "status": status,
                "detail": check.get("detail"),
            }
        )
    return {
        "overall_status": evaluation_payload.get("overall_status"),
        "summary": evaluation_payload.get("summary"),
        "checks": compact_checks,
    }


def _compact_intake_for_ir_feedback(intake: RequirementIntake) -> dict[str, Any]:
    payload = intake.model_dump(mode="json")
    simulation_spec = dict(payload.get("simulation_spec") or {})
    execution_plan = dict(payload.get("execution_plan") or {})
    simulation_spec.pop("ir_plan", None)
    execution_plan.pop("ir_plan", None)
    payload["simulation_spec"] = simulation_spec
    payload["execution_plan"] = execution_plan
    return {
        "task_family": payload.get("task_family"),
        "supported_now": payload.get("supported_now"),
        "summary": payload.get("summary"),
        "extracted_parameters": payload.get("extracted_parameters"),
        "simulation_spec": simulation_spec,
        "execution_plan": execution_plan,
        "assumptions": payload.get("assumptions"),
        "warnings": payload.get("warnings"),
    }


def _build_capacitor_ir_plan(intake: RequirementIntake) -> MaxwellIRPlan:
    spec = intake.simulation_spec or {}
    geometry = spec.get("geometry") if isinstance(spec.get("geometry"), dict) else {}
    excitations = spec.get("excitations") if isinstance(spec.get("excitations"), dict) else {}
    boundaries = spec.get("boundaries") if isinstance(spec.get("boundaries"), dict) else {}

    plate_width = _coerce_float(geometry.get("plate_width_mm")) or _coerce_float(
        intake.extracted_parameters.get("板宽_mm")
    ) or 20.0
    plate_spacing = _coerce_float(geometry.get("plate_spacing_mm")) or _coerce_float(
        intake.extracted_parameters.get("板间距_mm")
    ) or 1.0
    air_margin = _coerce_float(geometry.get("air_region_margin_mm")) or _coerce_float(
        boundaries.get("air_region_margin_mm")
    ) or 10.0
    voltage_v = _coerce_float(excitations.get("voltage_V")) or _coerce_float(
        intake.extracted_parameters.get("施加电压_V")
    ) or 100.0

    return MaxwellIRPlan(
        design_name="Capacitor2D",
        solution_type="ElectrostaticXY",
        setup_type="Electrostatic",
        parameters=[
            IRParameterBinding(name="plate_width_mm", source="geometry", field="plate_width_mm", default=plate_width),
            IRParameterBinding(name="plate_spacing_mm", source="geometry", field="plate_spacing_mm", default=plate_spacing),
            IRParameterBinding(name="air_margin_mm", source="geometry", field="air_region_margin_mm", default=air_margin),
            IRParameterBinding(name="voltage_v", source="excitations", field="voltage_V", default=voltage_v),
        ],
        locals=[
            IRLocalValue(name="plate_thickness_mm", expression="0.01"),
        ],
        objects=[
            IRObject(
                name="lower_plate",
                kind="rectangle",
                material="pec",
                origin_exprs=["-plate_width_mm / 2", "0", "0"],
                sizes_exprs=["plate_width_mm", "plate_thickness_mm"],
            ),
            IRObject(
                name="upper_plate",
                kind="rectangle",
                material="pec",
                origin_exprs=["-plate_width_mm / 2", "plate_spacing_mm + plate_thickness_mm", "0"],
                sizes_exprs=["plate_width_mm", "plate_thickness_mm"],
            ),
            IRObject(
                name="Region",
                kind="region",
                pad_value_exprs=["air_margin_mm", "air_margin_mm", "air_margin_mm", "air_margin_mm"],
                pad_type="Absolute Offset",
            ),
        ],
        assignments=[
            IRAssignment(name="GND", kind="voltage", targets=["lower_plate"], amplitude_expr="0"),
            IRAssignment(name="SIG", kind="voltage", targets=["upper_plate"], amplitude_expr="voltage_v"),
            IRAssignment(
                name="OuterRegion",
                kind="balloon",
                targets=["Region"],
                boundary_name="OuterRegion",
                is_voltage=True,
            ),
            IRAssignment(
                name="CapMatrix",
                kind="matrix",
                signal_assignments=["SIG"],
                ground_assignments=["GND"],
            ),
        ],
        derived_outputs=[
            IRDerivedOutput(output_key="applied_voltage_v", expression="voltage_v"),
            IRDerivedOutput(output_key="plate_spacing_mm", expression="plate_spacing_mm"),
            IRDerivedOutput(output_key="plate_width_mm", expression="plate_width_mm"),
            IRDerivedOutput(
                output_key="reference_average_field_v_per_m",
                expression="voltage_v / max(plate_spacing_mm / 1000.0, 1e-12)",
            ),
            IRDerivedOutput(
                output_key="reference_capacitance_f_for_1m_depth",
                expression="8.854187817e-12 * (plate_width_mm / 1000.0) / max(plate_spacing_mm / 1000.0, 1e-12)",
            ),
        ],
        postprocess=[
            IRPostprocess(
                kind="field_scalar",
                output_key="max_electric_field_note",
                cast="float",
                quantity="Mag_E",
                scalar_function="Maximum",
                object_name="Region",
                object_type="surface",
                solution_expr='"Setup1 : LastAdaptive"',
                error_output_key="max_electric_field_note_error",
            ),
            IRPostprocess(
                kind="matrix_export_value",
                output_key="capacitance_pf",
                cast="float",
                matrix_assignment_name="CapMatrix",
                output_filename="capacitance_matrix.txt",
                regex_pattern="\\bSIG\\s+([0-9.+\\-Ee]+)",
                scaled_output_key="capacitance_f",
                scale=1e-12,
            ),
        ],
        failure_note="Maxwell electrostatic solve failed.",
    )


def _build_busbar_ir_plan(intake: RequirementIntake) -> MaxwellIRPlan:
    spec = intake.simulation_spec or {}
    geometry = spec.get("geometry") if isinstance(spec.get("geometry"), dict) else {}
    excitations = spec.get("excitations") if isinstance(spec.get("excitations"), dict) else {}

    width_mm = _coerce_float(geometry.get("width_mm")) or 10.0
    thickness_mm = _coerce_float(geometry.get("thickness_mm")) or 2.0
    region_padding = _coerce_float(geometry.get("region_padding_mm")) or 20.0
    current_a = _coerce_float(excitations.get("current_a")) or 200.0

    return MaxwellIRPlan(
        design_name="Busbar2D",
        solution_type="Magnetostatic",
        setup_type="Magnetostatic",
        parameters=[
            IRParameterBinding(name="width_mm", source="geometry", field="width_mm", default=width_mm),
            IRParameterBinding(name="thickness_mm", source="geometry", field="thickness_mm", default=thickness_mm),
            IRParameterBinding(name="region_padding", source="geometry", field="region_padding_mm", default=region_padding),
            IRParameterBinding(name="current_a", source="excitations", field="current_a", default=current_a),
        ],
        objects=[
            IRObject(
                name="busbar",
                kind="rectangle",
                material="copper",
                origin_exprs=["-width_mm / 2", "-thickness_mm / 2", "0"],
                sizes_exprs=["width_mm", "thickness_mm"],
            ),
            IRObject(
                name="Region",
                kind="region",
                pad_value_exprs=["region_padding", "region_padding", "region_padding", "region_padding"],
                pad_type="Absolute Offset",
            ),
        ],
        assignments=[
            IRAssignment(name="BusbarCurrent", kind="current", targets=["busbar"], amplitude_expr="current_a"),
            IRAssignment(name="OuterRegion", kind="balloon", targets=["Region"], boundary_name="OuterRegion"),
        ],
        derived_outputs=[
            IRDerivedOutput(output_key="width_mm", expression="width_mm"),
            IRDerivedOutput(output_key="thickness_mm", expression="thickness_mm"),
            IRDerivedOutput(output_key="current_a", expression="current_a"),
            IRDerivedOutput(
                output_key="cross_section_area_mm2",
                expression="max(width_mm * thickness_mm, 1e-9)",
            ),
            IRDerivedOutput(
                output_key="estimated_current_density_a_per_mm2",
                expression="current_a / max(width_mm * thickness_mm, 1e-9)",
            ),
            IRDerivedOutput(
                output_key="reference_surface_field_t",
                expression="(4 * math.pi * 1e-7) * current_a / max(math.pi * thickness_mm * 1e-3, 1e-9)",
            ),
        ],
        postprocess=[
            IRPostprocess(
                kind="field_scalar",
                output_key="max_flux_density_t",
                cast="float",
                quantity="Mag_B",
                scalar_function="Maximum",
                object_name="AllObjects",
            ),
        ],
    )


def _build_solenoid_ir_plan(intake: RequirementIntake) -> MaxwellIRPlan:
    spec = intake.simulation_spec or {}
    geometry = spec.get("geometry") if isinstance(spec.get("geometry"), dict) else {}
    excitations = spec.get("excitations") if isinstance(spec.get("excitations"), dict) else {}

    length_mm = _coerce_float(geometry.get("length_mm")) or 50.0
    radius_mm = _coerce_float(geometry.get("radius_mm")) or 8.0
    coil_thickness_mm = _coerce_float(geometry.get("coil_thickness_mm")) or 2.0
    region_padding = _coerce_float(geometry.get("region_padding_mm")) or 25.0
    current_a = _coerce_float(excitations.get("current_a")) or 1.0
    coil_turns = int(round(_coerce_float(excitations.get("coil_turns")) or 300.0))

    return MaxwellIRPlan(
        design_name="Solenoid2D",
        solution_type="Magnetostatic",
        setup_type="Magnetostatic",
        parameters=[
            IRParameterBinding(name="length_mm", source="geometry", field="length_mm", default=length_mm),
            IRParameterBinding(name="radius_mm", source="geometry", field="radius_mm", default=radius_mm),
            IRParameterBinding(
                name="coil_thickness_mm",
                source="geometry",
                field="coil_thickness_mm",
                default=coil_thickness_mm,
            ),
            IRParameterBinding(
                name="region_padding",
                source="geometry",
                field="region_padding_mm",
                default=region_padding,
            ),
            IRParameterBinding(name="current_a", source="excitations", field="current_a", default=current_a),
            IRParameterBinding(name="coil_turns", source="excitations", field="coil_turns", default=coil_turns, cast="int"),
        ],
        locals=[
            IRLocalValue(name="equivalent_current_a", expression="current_a * coil_turns"),
        ],
        objects=[
            IRObject(
                name="upper_coil",
                kind="rectangle",
                material="copper",
                origin_exprs=["-length_mm / 2", "radius_mm", "0"],
                sizes_exprs=["length_mm", "coil_thickness_mm"],
            ),
            IRObject(
                name="lower_coil",
                kind="rectangle",
                material="copper",
                origin_exprs=["-length_mm / 2", "-radius_mm - coil_thickness_mm", "0"],
                sizes_exprs=["length_mm", "coil_thickness_mm"],
            ),
            IRObject(
                name="Region",
                kind="region",
                pad_value_exprs=["region_padding", "region_padding", "region_padding", "region_padding"],
                pad_type="Absolute Offset",
            ),
        ],
        assignments=[
            IRAssignment(name="UpperCurrent", kind="current", targets=["upper_coil"], amplitude_expr="equivalent_current_a"),
            IRAssignment(name="LowerCurrent", kind="current", targets=["lower_coil"], amplitude_expr="-equivalent_current_a"),
            IRAssignment(name="OuterRegion", kind="balloon", targets=["Region"], boundary_name="OuterRegion"),
        ],
        derived_outputs=[
            IRDerivedOutput(output_key="length_mm", expression="length_mm"),
            IRDerivedOutput(output_key="radius_mm", expression="radius_mm"),
            IRDerivedOutput(output_key="current_a", expression="current_a"),
            IRDerivedOutput(output_key="coil_turns", expression="coil_turns", cast="int"),
            IRDerivedOutput(output_key="equivalent_current_a", expression="equivalent_current_a"),
            IRDerivedOutput(
                output_key="estimated_center_flux_density_t",
                expression="(4 * math.pi * 1e-7) * coil_turns * current_a / max(length_mm * 1e-3, 1e-9)",
            ),
        ],
        postprocess=[
            IRPostprocess(
                kind="field_scalar",
                output_key="max_flux_density_t",
                cast="float",
                quantity="Mag_B",
                scalar_function="Maximum",
                object_name="AllObjects",
            ),
        ],
    )


def _build_coaxial_capacitor_ir_plan(intake: RequirementIntake) -> MaxwellIRPlan:
    spec = intake.simulation_spec or {}
    geometry = spec.get("geometry") if isinstance(spec.get("geometry"), dict) else {}
    excitations = spec.get("excitations") if isinstance(spec.get("excitations"), dict) else {}

    inner_radius_mm = _coerce_float(geometry.get("inner_radius_mm")) or 1.0
    outer_radius_mm = _coerce_float(geometry.get("outer_radius_mm")) or 5.0
    region_padding = _coerce_float(geometry.get("region_padding_mm")) or 8.0
    voltage_v = _coerce_float(excitations.get("voltage_V")) or 100.0

    return MaxwellIRPlan(
        design_name="CoaxialCapacitor2D",
        solution_type="ElectrostaticXY",
        setup_type="Electrostatic",
        parameters=[
            IRParameterBinding(
                name="inner_radius_mm",
                source="geometry",
                field="inner_radius_mm",
                default=inner_radius_mm,
            ),
            IRParameterBinding(
                name="outer_radius_mm",
                source="geometry",
                field="outer_radius_mm",
                default=outer_radius_mm,
            ),
            IRParameterBinding(
                name="region_padding",
                source="geometry",
                field="region_padding_mm",
                default=region_padding,
            ),
            IRParameterBinding(name="voltage_v", source="excitations", field="voltage_V", default=voltage_v),
        ],
        locals=[
            IRLocalValue(name="outer_conductor_thickness_mm", expression="max(1.0, 0.2 * outer_radius_mm)"),
        ],
        objects=[
            IRObject(
                name="inner_conductor",
                kind="circle",
                material="pec",
                origin_exprs=["0", "0", "0"],
                radius_expr="inner_radius_mm",
            ),
            IRObject(
                name="outer_conductor",
                kind="circle",
                material="pec",
                origin_exprs=["0", "0", "0"],
                radius_expr="outer_radius_mm + outer_conductor_thickness_mm",
            ),
            IRObject(
                name="outer_void",
                kind="circle",
                material="vacuum",
                origin_exprs=["0", "0", "0"],
                radius_expr="outer_radius_mm",
            ),
            IRObject(
                name="Region",
                kind="region",
                pad_value_exprs=["region_padding", "region_padding", "region_padding", "region_padding"],
                pad_type="Absolute Offset",
            ),
        ],
        operations=[
            IROperation(kind="subtract", blank_parts=["outer_conductor"], tool_parts=["outer_void"]),
        ],
        assignments=[
            IRAssignment(name="InnerVoltage", kind="voltage", targets=["inner_conductor"], amplitude_expr="voltage_v"),
            IRAssignment(name="OuterGround", kind="voltage", targets=["outer_conductor"], amplitude_expr="0"),
            IRAssignment(
                name="OuterRegion",
                kind="balloon",
                targets=["Region"],
                boundary_name="OuterRegion",
                is_voltage=True,
            ),
            IRAssignment(
                name="CoaxCapMatrix",
                kind="matrix",
                signal_assignments=["InnerVoltage"],
                ground_assignments=["OuterGround"],
            ),
        ],
        derived_outputs=[
            IRDerivedOutput(output_key="inner_radius_mm", expression="inner_radius_mm"),
            IRDerivedOutput(output_key="outer_radius_mm", expression="outer_radius_mm"),
            IRDerivedOutput(output_key="applied_voltage_v", expression="voltage_v"),
            IRDerivedOutput(
                output_key="reference_capacitance_f_per_m",
                expression="2 * math.pi * 8.854187817e-12 / math.log(outer_radius_mm / inner_radius_mm)",
            ),
            IRDerivedOutput(
                output_key="reference_capacitance_pf_per_m",
                expression="(2 * math.pi * 8.854187817e-12 / math.log(outer_radius_mm / inner_radius_mm)) * 1e12",
            ),
            IRDerivedOutput(
                output_key="reference_average_field_v_per_m",
                expression="voltage_v / max((outer_radius_mm - inner_radius_mm) * 1e-3, 1e-12)",
            ),
            IRDerivedOutput(
                output_key="reference_max_field_v_per_m",
                expression="voltage_v / max(inner_radius_mm * 1e-3 * math.log(outer_radius_mm / inner_radius_mm), 1e-12)",
            ),
        ],
        postprocess=[
            IRPostprocess(
                kind="field_scalar",
                output_key="max_electric_field_v_per_m",
                cast="float",
                quantity="Mag_E",
                scalar_function="Maximum",
                object_name="AllObjects",
                solution_expr='"Setup1 : LastAdaptive"',
            ),
            IRPostprocess(
                kind="matrix_export_value",
                output_key="capacitance_pf",
                cast="float",
                matrix_assignment_name="CoaxCapMatrix",
                output_filename="coaxial_capacitance_matrix.txt",
                regex_pattern="\\bInnerVoltage\\s+([0-9.+\\-Ee]+)",
                scaled_output_key="capacitance_f",
                scale=1e-12,
            ),
        ],
        failure_note="Maxwell electrostatic solve failed.",
    )


def _build_generic_2d_ir_plan(
    intake: RequirementIntake,
    primitive_library: PrimitiveLibrary | None = None,
    *,
    return_templates: bool = False,
) -> MaxwellIRPlan | tuple[MaxwellIRPlan, list[PrimitiveTemplate]]:
    spec = intake.simulation_spec or {}
    geometry = _as_mapping(spec.get("geometry"))
    solver = _as_mapping(spec.get("solver"))
    materials_value = spec.get("materials")
    excitations_value = spec.get("excitations")
    boundaries_value = spec.get("boundaries")
    materials = _as_mapping_list(materials_value)
    excitations = _as_mapping_list(excitations_value)
    boundaries = _as_mapping_list(boundaries_value)
    boundary_map = _as_mapping(boundaries_value)
    required_outputs = spec.get("required_outputs")

    if isinstance(materials_value, dict):
        materials_by_object = {
            str(key).strip(): str(value or "").strip()
            for key, value in materials_value.items()
            if str(key).strip()
        }
    else:
        materials_by_object = {}
        for item in materials:
            object_name = str(item.get("object") or item.get("assignment") or item.get("target") or "").strip()
            material_name = str(item.get("material") or item.get("chosen_material") or item.get("value") or "").strip()
            if object_name and material_name:
                materials_by_object[object_name] = material_name
                continue
            applies_to = item.get("applies_to")
            named_material = str(item.get("name") or item.get("material") or "").strip()
            if named_material and isinstance(applies_to, list):
                for target in applies_to:
                    target_name = str(target or "").strip()
                    if target_name:
                        materials_by_object[target_name] = named_material
    geometry_objects = _as_mapping_list(
        geometry.get("objects") or geometry.get("cross_section_objects") or geometry.get("entities")
    )
    if not geometry_objects:
        geometry_objects = _normalize_generic_primitives_to_objects(geometry)
    if not geometry_objects:
        raise RequirementPlanningError("当前通用 2D 任务缺少 geometry.objects，暂时无法合成本地 Maxwell IR。")
    requested_blob = json.dumps(required_outputs or [], ensure_ascii=False).lower()
    solver_blob = json.dumps(solver, ensure_ascii=False).lower()
    excitation_blob = json.dumps(excitations_value, ensure_ascii=False).lower()
    physics_blob = " ".join([requested_blob, solver_blob, excitation_blob])
    is_electrostatic = "electro" in physics_blob or "静电" in physics_blob or "voltage" in excitation_blob or "电压" in excitation_blob or any(
        str(item.get("type") or "").strip().lower() == "voltage" for item in excitations
    )
    is_magnetostatic = "magnet" in physics_blob or "静磁" in physics_blob or "current" in excitation_blob or "电流" in excitation_blob or any(
        str(item.get("type") or "").strip().lower() == "current" for item in excitations
    )
    if is_electrostatic and is_magnetostatic:
        raise RequirementPlanningError("当前通用 2D 任务同时混入静电和静磁激励，暂不支持一次合成。")
    if not is_electrostatic and not is_magnetostatic:
        raise RequirementPlanningError("当前通用 2D 任务无法从结构化结果判断求解物理类型。")

    parameters: list[IRParameterBinding] = []
    locals_list: list[IRLocalValue] = []
    objects: list[IRObject] = []
    operations: list[IROperation] = []
    assignments: list[IRAssignment] = []
    derived_outputs: list[IRDerivedOutput] = []
    postprocess: list[IRPostprocess] = []
    object_name_map: dict[str, str] = {}
    object_area_exprs: dict[str, str] = {}
    extents: list[tuple[float, float, float, float]] = []
    explicit_air_radius_mm: float | None = None
    explicit_air_padding_mm: float | None = None
    used_parameter_names: set[str] = set()
    conductor_raw_names: list[str] = []
    pending_templates: list[PrimitiveTemplate] = []
    pending_template_keys: set[str] = set()

    if boundary_map:
        explicit_air_padding_mm = _coerce_measure_value(
            boundary_map.get("air_pad_mm") or boundary_map.get("region_padding_mm") or boundary_map.get("outer_air_padding_mm")
        )

    def add_parameter(base_name: str, source: str, field: str, default: Any, cast: str = "float") -> str:
        name = _sanitize_identifier(base_name, "param")
        unique = name
        suffix = 2
        while unique in used_parameter_names:
            unique = f"{name}_{suffix}"
            suffix += 1
        used_parameter_names.add(unique)
        parameters.append(
            IRParameterBinding(
                name=unique,
                source=source,
                field=field,
                default=default,
                cast=cast,
            )
        )
        return unique

    def record_extents(x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        extents.append((x_min, x_max, y_min, y_max))

    def register_circle(
        raw_name: str,
        x_mm: float,
        y_mm: float,
        radius_mm: float,
        *,
        material: str,
    ) -> str:
        name = _sanitize_identifier(raw_name, "circle")
        cx_var = add_parameter(f"{name}_cx_mm", "geometry", f"{name}_cx_mm", x_mm)
        cy_var = add_parameter(f"{name}_cy_mm", "geometry", f"{name}_cy_mm", y_mm)
        radius_var = add_parameter(f"{name}_radius_mm", "geometry", f"{name}_radius_mm", radius_mm)
        objects.append(
            IRObject(
                name=name,
                kind="circle",
                material=material,
                origin_exprs=[cx_var, cy_var, "0"],
                radius_expr=radius_var,
            )
        )
        object_name_map[raw_name] = name
        record_extents(x_mm - radius_mm, x_mm + radius_mm, y_mm - radius_mm, y_mm + radius_mm)
        object_area_exprs[name] = f"math.pi * {radius_var} * {radius_var}"
        return name

    def register_rectangle(
        raw_name: str,
        x_mm: float,
        y_mm: float,
        width_mm: float,
        height_mm: float,
        *,
        material: str,
    ) -> str:
        name = _sanitize_identifier(raw_name, "rect")
        cx_var = add_parameter(f"{name}_cx_mm", "geometry", f"{name}_cx_mm", x_mm)
        cy_var = add_parameter(f"{name}_cy_mm", "geometry", f"{name}_cy_mm", y_mm)
        width_var = add_parameter(f"{name}_width_mm", "geometry", f"{name}_width_mm", width_mm)
        height_var = add_parameter(f"{name}_height_mm", "geometry", f"{name}_height_mm", height_mm)
        objects.append(
            IRObject(
                name=name,
                kind="rectangle",
                material=material,
                origin_exprs=[f"{cx_var} - {width_var} / 2", f"{cy_var} - {height_var} / 2", "0"],
                sizes_exprs=[width_var, height_var],
            )
        )
        object_name_map[raw_name] = name
        record_extents(x_mm - width_mm / 2, x_mm + width_mm / 2, y_mm - height_mm / 2, y_mm + height_mm / 2)
        object_area_exprs[name] = f"{width_var} * {height_var}"
        return name

    def register_annulus(
        raw_name: str,
        x_mm: float,
        y_mm: float,
        inner_radius_mm: float,
        outer_radius_mm: float,
        *,
        material: str,
    ) -> str:
        base_name = _sanitize_identifier(raw_name, "annulus")
        cx_var = add_parameter(f"{base_name}_cx_mm", "geometry", f"{base_name}_cx_mm", x_mm)
        cy_var = add_parameter(f"{base_name}_cy_mm", "geometry", f"{base_name}_cy_mm", y_mm)
        inner_var = add_parameter(
            f"{base_name}_inner_radius_mm",
            "geometry",
            f"{base_name}_inner_radius_mm",
            inner_radius_mm,
        )
        outer_var = add_parameter(
            f"{base_name}_outer_radius_mm",
            "geometry",
            f"{base_name}_outer_radius_mm",
            outer_radius_mm,
        )
        outer_name = f"{base_name}_outer"
        inner_name = f"{base_name}_inner_void"
        objects.append(
            IRObject(
                name=outer_name,
                kind="circle",
                material=material,
                origin_exprs=[cx_var, cy_var, "0"],
                radius_expr=outer_var,
            )
        )
        objects.append(
            IRObject(
                name=inner_name,
                kind="circle",
                material="vacuum" if is_magnetostatic else "air",
                origin_exprs=[cx_var, cy_var, "0"],
                radius_expr=inner_var,
            )
        )
        operations.append(IROperation(kind="subtract", blank_parts=[outer_name], tool_parts=[inner_name]))
        object_name_map[raw_name] = outer_name
        record_extents(x_mm - outer_radius_mm, x_mm + outer_radius_mm, y_mm - outer_radius_mm, y_mm + outer_radius_mm)
        object_area_exprs[outer_name] = f"math.pi * ({outer_var} * {outer_var} - {inner_var} * {inner_var})"
        return outer_name

    def register_template_instance(
        raw_object: dict[str, Any],
        template: PrimitiveTemplate,
        *,
        material: str,
    ) -> str:
        flattened = _flatten_generic_object_payload(raw_object)
        instance_name = _sanitize_identifier(raw_object.get("name") or template.primitive_key, template.primitive_key)
        parameter_expr_map: dict[str, str] = {}
        parameter_values: dict[str, float | int | str | bool] = {}

        for parameter in template.parameters:
            raw_value = None
            for alias in [parameter.name, *parameter.aliases]:
                if alias in flattened:
                    raw_value = flattened.get(alias)
                    break
            if raw_value is None:
                raw_value = parameter.default
            if parameter.required and raw_value is None:
                raise RequirementPlanningError(
                    f"通用 2D 复合原语 {template.display_name} 缺少必要参数: {parameter.name}"
                )
            if parameter.cast == "float":
                if raw_value is None:
                    resolved_value = 0.0
                else:
                    numeric = _coerce_measure_value(raw_value)
                    if numeric is None:
                        raise RequirementPlanningError(
                            f"通用 2D 复合原语 {template.display_name} 的参数 {parameter.name} 不是可解析数值。"
                        )
                    resolved_value = float(numeric)
            elif parameter.cast == "int":
                if raw_value is None:
                    resolved_value = 0
                else:
                    numeric = _coerce_measure_value(raw_value)
                    if numeric is None:
                        raise RequirementPlanningError(
                            f"通用 2D 复合原语 {template.display_name} 的参数 {parameter.name} 不是可解析整数。"
                        )
                    resolved_value = int(round(float(numeric)))
            elif parameter.cast == "bool":
                resolved_value = bool(raw_value)
            else:
                resolved_value = str(raw_value or "")
            parameter_values[parameter.name] = resolved_value
            parameter_expr_map[parameter.name] = add_parameter(
                f"{instance_name}_{parameter.name}",
                "geometry",
                f"{instance_name}_{parameter.name}",
                resolved_value,
                cast=parameter.cast,
            )

        role_to_object: dict[str, str] = {}
        for template_object in template.objects:
            object_name = f"{instance_name}_{template_object.role_name}"
            role_to_object[template_object.role_name] = object_name
            resolved_origin = [
                _replace_expression_tokens(item, parameter_expr_map)
                for item in template_object.origin_exprs
            ]
            resolved_sizes = [
                _replace_expression_tokens(item, parameter_expr_map)
                for item in template_object.sizes_exprs
            ]
            resolved_radius = (
                _replace_expression_tokens(template_object.radius_expr, parameter_expr_map)
                if template_object.radius_expr
                else None
            )
            object_material = (
                material
                if template_object.material_mode == "instance"
                else str(template_object.material_value or "vacuum")
            )
            if template_object.kind == "circle":
                objects.append(
                    IRObject(
                        name=object_name,
                        kind="circle",
                        material=object_material,
                        origin_exprs=resolved_origin,
                        radius_expr=resolved_radius,
                    )
                )
                try:
                    center_x = float(eval(template_object.origin_exprs[0], {"__builtins__": {}, "math": math}, parameter_values))
                    center_y = float(eval(template_object.origin_exprs[1], {"__builtins__": {}, "math": math}, parameter_values))
                    radius = float(eval(template_object.radius_expr or "0", {"__builtins__": {}, "math": math}, parameter_values))
                    record_extents(center_x - radius, center_x + radius, center_y - radius, center_y + radius)
                except Exception:
                    pass
            else:
                objects.append(
                    IRObject(
                        name=object_name,
                        kind="rectangle",
                        material=object_material,
                        origin_exprs=resolved_origin,
                        sizes_exprs=resolved_sizes,
                    )
                )
                try:
                    origin_x = float(eval(template_object.origin_exprs[0], {"__builtins__": {}, "math": math}, parameter_values))
                    origin_y = float(eval(template_object.origin_exprs[1], {"__builtins__": {}, "math": math}, parameter_values))
                    width = float(eval(template_object.sizes_exprs[0], {"__builtins__": {}, "math": math}, parameter_values))
                    height = float(eval(template_object.sizes_exprs[1], {"__builtins__": {}, "math": math}, parameter_values))
                    record_extents(origin_x, origin_x + width, origin_y, origin_y + height)
                except Exception:
                    pass

        for operation in template.operations:
            operations.append(
                IROperation(
                    kind=operation.kind,
                    blank_parts=[role_to_object[item] for item in operation.blank_roles],
                    tool_parts=[role_to_object[item] for item in operation.tool_roles],
                    keep_originals=operation.keep_originals,
                )
            )

        result_object_name = role_to_object[template.result_role_name]
        object_name_map[str(raw_object.get("name") or instance_name)] = result_object_name
        if template.result_area_expr:
            object_area_exprs[result_object_name] = _replace_expression_tokens(template.result_area_expr, parameter_expr_map)
        if primitive_library and primitive_library.is_pending(template.primitive_key) and template.primitive_key not in pending_template_keys:
            pending_templates.append(template)
            pending_template_keys.add(template.primitive_key)
        return result_object_name

    for raw_object in geometry_objects:
        raw_name = str(raw_object.get("name") or "").strip() or f"object_{len(objects) + 1}"
        kind = str(raw_object.get("type") or raw_object.get("kind") or raw_object.get("shape") or "").strip().lower()
        material = materials_by_object.get(raw_name)
        material = material or ("pec" if is_electrostatic else "copper")
        lowered_name = raw_name.lower()
        is_region_like = "region" in lowered_name or kind in {"region", "circle_or_rectangle", "air_region"}
        if is_region_like:
            air_radius = _coerce_measure_value(raw_object.get("radius"))
            if air_radius is None:
                air_radius = _coerce_measure_value(raw_object.get("radius_mm"))
            if air_radius is not None:
                explicit_air_radius_mm = max(explicit_air_radius_mm or 0.0, air_radius)
            continue

        if kind == "circle":
            radius_mm = _coerce_measure_value(raw_object.get("radius"))
            if radius_mm is None:
                radius_mm = _coerce_measure_value(raw_object.get("radius_mm"))
            center_x, center_y = _coerce_xy_pair(raw_object.get("center"))
            if center_x is None or center_y is None:
                center_x, center_y = _coerce_xy_pair(raw_object.get("center_mm"))
            if center_x is None or center_y is None:
                center_x, center_y = _coerce_xy_pair(raw_object.get("origin"))
            if radius_mm is None or center_x is None or center_y is None:
                raise RequirementPlanningError(f"通用 2D 圆形对象缺少中心或半径: {raw_name}")
            conductor_raw_names.append(raw_name)
            register_circle(
                raw_name,
                center_x,
                center_y,
                radius_mm,
                material=material,
            )
            continue

        if kind == "rectangle":
            width_mm = _coerce_measure_value(raw_object.get("width"))
            height_mm = _coerce_measure_value(raw_object.get("height"))
            if width_mm is None:
                width_mm = _coerce_measure_value(raw_object.get("width_mm"))
            if height_mm is None:
                height_mm = _coerce_measure_value(raw_object.get("height_mm"))
            if width_mm is None or height_mm is None:
                size = raw_object.get("size") or raw_object.get("sizes")
                if isinstance(size, dict):
                    width_mm = width_mm or _coerce_measure_value(size.get("x") or size.get("width"))
                    height_mm = height_mm or _coerce_measure_value(size.get("y") or size.get("height"))
                elif isinstance(size, (list, tuple)) and len(size) >= 2:
                    width_mm = width_mm or _coerce_measure_value(size[0])
                    height_mm = height_mm or _coerce_measure_value(size[1])
            center_x, center_y = _coerce_xy_pair(raw_object.get("center"))
            if center_x is None or center_y is None:
                center_x, center_y = _coerce_xy_pair(raw_object.get("center_mm"))
            if center_x is None or center_y is None:
                origin_x, origin_y = _coerce_xy_pair(
                    raw_object.get("origin") or raw_object.get("corner") or raw_object.get("lower_left")
                )
                if origin_x is not None and origin_y is not None and width_mm is not None and height_mm is not None:
                    center_x = origin_x + width_mm / 2
                    center_y = origin_y + height_mm / 2
            if None in {center_x, center_y, width_mm, height_mm}:
                raise RequirementPlanningError(f"通用 2D 矩形对象缺少尺寸或位置: {raw_name}")
            conductor_raw_names.append(raw_name)
            register_rectangle(raw_name, center_x, center_y, width_mm, height_mm, material=material)
            continue

        if kind == "annulus":
            inner_radius_mm = _coerce_measure_value(raw_object.get("inner_radius"))
            outer_radius_mm = _coerce_measure_value(raw_object.get("outer_radius"))
            if inner_radius_mm is None:
                inner_radius_mm = _coerce_measure_value(raw_object.get("inner_radius_mm"))
            if outer_radius_mm is None:
                outer_radius_mm = _coerce_measure_value(raw_object.get("outer_radius_mm"))
            center_x, center_y = _coerce_xy_pair(raw_object.get("center"))
            if center_x is None or center_y is None:
                center_x, center_y = _coerce_xy_pair(raw_object.get("center_mm"))
            if center_x is None or center_y is None:
                center_x, center_y = 0.0, 0.0
            if inner_radius_mm is None or outer_radius_mm is None:
                raise RequirementPlanningError(f"通用 2D 环形对象缺少内外半径: {raw_name}")
            conductor_raw_names.append(raw_name)
            register_annulus(raw_name, center_x, center_y, inner_radius_mm, outer_radius_mm, material=material)
            continue

        template = primitive_library.find(kind or raw_name) if primitive_library else None
        if template is not None:
            conductor_raw_names.append(raw_name)
            register_template_instance(raw_object, template, material=material)
            continue

        raise UnknownPrimitiveError(
            message=f"当前通用 2D 对象类型暂不在本地原语库中: {kind or raw_name}",
            primitive_token=kind or raw_name,
            raw_object=raw_object,
        )

    if not objects:
        raise RequirementPlanningError("当前通用 2D 任务没有可用于 Maxwell 的导体几何。")

    if not extents:
        raise RequirementPlanningError("当前通用 2D 任务无法计算建模包围区域。")

    x_min = min(item[0] for item in extents)
    x_max = max(item[1] for item in extents)
    y_min = min(item[2] for item in extents)
    y_max = max(item[3] for item in extents)
    overall_span = max(x_max - x_min, y_max - y_min, 1.0)
    max_extent = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max), 1.0)
    default_padding = max(20.0, 10.0 * overall_span)
    if explicit_air_padding_mm is not None:
        region_padding_mm = max(10.0, explicit_air_padding_mm)
    elif explicit_air_radius_mm is not None:
        region_padding_mm = max(10.0, explicit_air_radius_mm - max_extent)
    else:
        region_padding_mm = default_padding
    region_padding_var = add_parameter("region_padding_mm", "boundaries", "region_padding_mm", region_padding_mm)
    objects.append(
        IRObject(
            name="Region",
            kind="region",
            pad_value_exprs=[region_padding_var, region_padding_var, region_padding_var, region_padding_var],
            pad_type="Absolute Offset",
        )
    )

    voltage_assignments: list[tuple[str, float]] = []
    current_assignments: list[tuple[str, float]] = []
    if not excitations and isinstance(excitations_value, dict):
        if is_electrostatic and len(conductor_raw_names) >= 2:
            positive_v = _coerce_measure_value(excitations_value.get("voltage_pos_v"))
            negative_v = _coerce_measure_value(excitations_value.get("voltage_neg_v"))
            if positive_v is not None:
                excitations.append({"object": conductor_raw_names[0], "type": "voltage", "value": positive_v})
            if negative_v is not None:
                excitations.append({"object": conductor_raw_names[1], "type": "voltage", "value": negative_v})
        elif is_magnetostatic and conductor_raw_names:
            current_a = _coerce_measure_value(excitations_value.get("current_a"))
            if current_a is not None:
                excitations.append({"object": conductor_raw_names[0], "type": "current", "value": current_a})
    for index, excitation in enumerate(excitations, start=1):
        raw_target = str(excitation.get("object") or excitation.get("target") or "").strip()
        target_name = object_name_map.get(raw_target)
        if not target_name:
            continue
        kind = str(excitation.get("type") or excitation.get("excitation_type") or "").strip().lower()
        if kind in {"total_current", "current_total", "dc_current"}:
            kind = "current"
        elif kind in {"electric_potential", "potential"}:
            kind = "voltage"
        value = _coerce_measure_value(excitation.get("value"))
        if value is None:
            value = _coerce_measure_value(excitation.get("current"))
        if value is None:
            value = _coerce_measure_value(excitation.get("voltage"))
        if value is None:
            value = _coerce_measure_value(excitation.get("amplitude"))
        if value is None:
            value = _coerce_measure_value(excitation.get("value_a"))
        if value is None:
            value = _coerce_measure_value(excitation.get("value_v"))
        if value is None:
            value = _coerce_measure_value(excitation.get("value_A"))
        if value is None:
            value = _coerce_measure_value(excitation.get("value_V"))
        if value is None:
            continue
        param_name = add_parameter(
            f"{target_name}_{kind}_{index}",
            "excitations",
            f"{target_name}_{kind}_{index}",
            value,
        )
        assignment_name = _sanitize_identifier(f"{target_name}_{kind}_{index}", f"{kind}_{index}")
        if kind == "voltage":
            assignments.append(
                IRAssignment(name=assignment_name, kind="voltage", targets=[target_name], amplitude_expr=param_name)
            )
            voltage_assignments.append((assignment_name, value))
        elif kind == "current":
            assignments.append(
                IRAssignment(name=assignment_name, kind="current", targets=[target_name], amplitude_expr=param_name)
            )
            current_assignments.append((assignment_name, value))

    if not assignments:
        raise RequirementPlanningError("当前通用 2D 任务没有可映射到 Maxwell 的激励。")

    if is_electrostatic and not voltage_assignments:
        raise RequirementPlanningError("当前通用 2D 静电任务没有可用的电压激励。")
    if is_magnetostatic and not current_assignments:
        raise RequirementPlanningError("当前通用 2D 静磁任务没有可用的电流激励。")

    needs_outer_region = True
    if boundaries:
        boundary_blob = json.dumps(boundaries, ensure_ascii=False).lower()
        needs_outer_region = any(token in boundary_blob for token in ("balloon", "open", "far_field", "outer"))
    if needs_outer_region:
        assignments.append(
            IRAssignment(
                name="OuterRegion",
                kind="balloon",
                targets=["Region"],
                boundary_name="OuterRegion",
                is_voltage=True if is_electrostatic else None,
            )
        )

    if is_electrostatic:
        sorted_voltages = sorted(voltage_assignments, key=lambda item: item[1])
        ground_name = sorted_voltages[0][0]
        signal_name = sorted_voltages[-1][0]
        if signal_name != ground_name:
            assignments.append(
                IRAssignment(
                    name="CapMatrix",
                    kind="matrix",
                    signal_assignments=[signal_name],
                    ground_assignments=[ground_name],
                )
            )
            postprocess.append(
                IRPostprocess(
                    kind="matrix_export_value",
                    output_key="capacitance_per_unit_length_pf_per_m",
                    cast="float",
                    matrix_assignment_name="CapMatrix",
                    output_filename="capacitance_matrix.txt",
                    regex_pattern=rf"\b{re.escape(signal_name)}\s+([0-9.+\-Ee]+)",
                    scaled_output_key="capacitance_per_unit_length_f_per_m",
                    scale=1e-12,
                )
            )
        derived_outputs.append(
            IRDerivedOutput(
                output_key="voltage_difference_v",
                expression=str(abs(sorted_voltages[-1][1] - sorted_voltages[0][1])),
            )
        )
        postprocess.append(
            IRPostprocess(
                kind="field_scalar",
                output_key="max_electric_field_v_per_m",
                cast="float",
                quantity="Mag_E",
                scalar_function="Maximum",
                object_name="AllObjects",
                solution_expr='"Setup1 : LastAdaptive"',
            )
        )
    else:
        postprocess.append(
            IRPostprocess(
                kind="field_scalar",
                output_key="max_flux_density_t",
                cast="float",
                quantity="Mag_B",
                scalar_function="Maximum",
                object_name="AllObjects",
            )
        )
        if len(current_assignments) == 1:
            assignment_name, current_value = current_assignments[0]
            target_assignment = next((item for item in assignments if item.name == assignment_name), None)
            target_object = target_assignment.targets[0] if target_assignment and target_assignment.targets else None
            area_expr = object_area_exprs.get(target_object or "")
            if area_expr:
                derived_outputs.append(IRDerivedOutput(output_key="current_a", expression=str(current_value)))
                derived_outputs.append(
                    IRDerivedOutput(
                        output_key="estimated_current_density_a_per_mm2",
                        expression=f"{current_value} / max({area_expr}, 1e-9)",
                    )
                )

    geometry_kinds = {
        str(item.get("type") or item.get("kind") or item.get("shape") or "").strip().lower() for item in geometry_objects
    }
    design_name = "GenericElectrostatic2D" if is_electrostatic else "GenericMagnetostatic2D"
    if "annulus" in geometry_kinds:
        design_name = "GenericAnnular2D"
    elif "rectangle" in geometry_kinds:
        design_name = "GenericRectangular2D"
    elif geometry_kinds and geometry_kinds.issubset({"circle", "region", "circle_or_rectangle", "air_region"}):
        design_name = "GenericCircular2D"

    plan = MaxwellIRPlan(
        design_name=design_name,
        solution_type="Electrostatic" if is_electrostatic else "Magnetostatic",
        setup_type="Electrostatic" if is_electrostatic else "Magnetostatic",
        parameters=parameters,
        locals=locals_list,
        objects=objects,
        operations=operations,
        assignments=assignments,
        derived_outputs=derived_outputs,
        postprocess=postprocess,
        failure_note="Generic 2D Maxwell solve failed.",
    )
    if return_templates:
        return plan, pending_templates
    return plan


def _build_local_script_from_generic_intake(
    intake: RequirementIntake,
    primitive_library: PrimitiveLibrary | None = None,
) -> GeneratedMaxwellScript:
    plan, learned_templates = _build_generic_2d_ir_plan(
        intake,
        primitive_library=primitive_library,
        return_templates=True,
    )
    script = _render_ir_script_for_intake(
        intake,
        plan,
        summary="根据通用 2D 结构化参数自动合成本地 Maxwell IR，并渲染成可执行脚本。",
        assumptions=["当前脚本不是按任务模板硬编码，而是把通用 2D 几何、激励、边界组合成 Maxwell IR。"],
        warnings=["当前通用本地合成器覆盖 Maxwell 2D 的 circle/rectangle/annulus/region 与静电静磁基础原语。"],
    )
    script.primitive_library_updates = [item.model_dump(mode="json") for item in learned_templates]
    return script


def _can_build_local_generic_2d_ir(
    intake: RequirementIntake,
    primitive_library: PrimitiveLibrary | None = None,
) -> bool:
    try:
        _build_generic_2d_ir_plan(intake, primitive_library=primitive_library)
        return True
    except UnknownPrimitiveError:
        return True
    except RequirementPlanningError:
        return False


def _build_local_script_from_capacitor_intake(intake: RequirementIntake) -> GeneratedMaxwellScript:
    plan = _build_capacitor_ir_plan(intake)
    return _render_ir_script_for_intake(
        intake,
        plan,
        summary="根据二维平行板电容器结构化结果生成本地 Maxwell 2D 静电脚本。",
        assumptions=["当云端脚本生成不可用时，回退到通用 IR 渲染的二维静电电容器脚本。"],
        warnings=["当前 capacitor_2d 首版已经迁移到统一 Maxwell IR 路径。"],
    )
    spec = intake.simulation_spec or {}
    geometry = spec.get("geometry") if isinstance(spec.get("geometry"), dict) else {}
    excitations = spec.get("excitations") if isinstance(spec.get("excitations"), dict) else {}
    boundaries = spec.get("boundaries") if isinstance(spec.get("boundaries"), dict) else {}

    plate_width = _coerce_float(geometry.get("plate_width_mm")) or _coerce_float(
        intake.extracted_parameters.get("板宽_mm")
    ) or 20.0
    plate_spacing = _coerce_float(geometry.get("plate_spacing_mm")) or _coerce_float(
        intake.extracted_parameters.get("板间距_mm")
    ) or 1.0
    air_margin = _coerce_float(geometry.get("air_region_margin_mm")) or _coerce_float(
        boundaries.get("air_region_margin_mm")
    ) or 10.0
    voltage_v = _coerce_float(excitations.get("voltage_V")) or _coerce_float(
        intake.extracted_parameters.get("施加电压_V")
    ) or 100.0

    code = dedent(
        f"""
        import re
        from pathlib import Path
        from ansys.aedt.core import Maxwell2d
        from ansys.aedt.core.modules.boundary.maxwell_boundary import MatrixElectric


        def run_job(job: dict) -> dict:
            simulation_spec = dict(job.get("simulation_spec") or {{}})
            geometry = dict(simulation_spec.get("geometry") or {{}})
            excitations = dict(simulation_spec.get("excitations") or {{}})
            output_dir = Path(str(job["output_dir"]))
            project_file = str(job["project_file"])
            version = job.get("maxwell_version")
            non_graphical = bool(job.get("non_graphical", True))
            student_version = bool(job.get("student_version", False))

            plate_width_mm = float(geometry.get("plate_width_mm", {plate_width}))
            plate_spacing_mm = float(geometry.get("plate_spacing_mm", {plate_spacing}))
            air_margin_mm = float(geometry.get("air_region_margin_mm", {air_margin}))
            voltage_v = float(excitations.get("voltage_V", {voltage_v}))
            plate_thickness_mm = 0.01
            gap_m = plate_spacing_mm / 1000.0
            width_m = plate_width_mm / 1000.0
            epsilon_0 = 8.854187817e-12

            outputs: dict[str, float | str] = {{}}
            with Maxwell2d(
                project=project_file,
                design="Capacitor2D",
                solution_type="ElectrostaticXY",
                version=version,
                non_graphical=non_graphical,
                new_desktop=True,
                close_on_exit=True,
                student_version=student_version,
            ) as app:
                app.modeler.model_units = "mm"
                lower_plate = app.modeler.create_rectangle(
                    origin=[f"{{-plate_width_mm / 2}}mm", "0mm", "0mm"],
                    sizes=[f"{{plate_width_mm}}mm", f"{{plate_thickness_mm}}mm"],
                    name="lower_plate",
                    material="pec",
                )
                upper_plate = app.modeler.create_rectangle(
                    origin=[f"{{-plate_width_mm / 2}}mm", f"{{plate_spacing_mm + plate_thickness_mm}}mm", "0mm"],
                    sizes=[f"{{plate_width_mm}}mm", f"{{plate_thickness_mm}}mm"],
                    name="upper_plate",
                    material="pec",
                )
                region = app.modeler.create_region(
                    pad_value=[
                        f"{{air_margin_mm}}mm",
                        f"{{air_margin_mm}}mm",
                        f"{{air_margin_mm}}mm",
                        f"{{air_margin_mm}}mm",
                    ],
                    pad_type="Absolute Offset",
                    name="Region",
                )
                v_gnd = app.assign_voltage(lower_plate.name, amplitude="0V", name="GND")
                v_sig = app.assign_voltage(upper_plate.name, amplitude=f"{{voltage_v}}V", name="SIG")
                try:
                    app.assign_balloon(region.edges, boundary="OuterRegion", is_voltage=True)
                except Exception:
                    pass
                try:
                    matrix = app.assign_matrix(
                        MatrixElectric(
                            signal_sources=[v_sig.name],
                            ground_sources=[v_gnd.name],
                            matrix_name="CapMatrix",
                        )
                    )
                except Exception:
                    matrix = None
                app.create_setup(name="Setup1", setup_type="Electrostatic")
                app.save_project()
                solve_ok = bool(app.analyze_setup("Setup1"))
                outputs["solve_status"] = "completed" if solve_ok else "failed"
                if not solve_ok:
                    outputs["status"] = "failed"
                    outputs["notes"] = "Maxwell electrostatic solve failed."
                    outputs["project_name"] = app.project_name
                    outputs["design_name"] = app.design_name
                    app.save_project()
                    return outputs
                outputs["project_name"] = app.project_name
                outputs["design_name"] = app.design_name
                outputs["applied_voltage_v"] = voltage_v
                outputs["plate_spacing_mm"] = plate_spacing_mm
                outputs["plate_width_mm"] = plate_width_mm
                outputs["reference_average_field_v_per_m"] = voltage_v / gap_m
                outputs["reference_capacitance_f_for_1m_depth"] = epsilon_0 * width_m / gap_m
                try:
                    outputs["max_electric_field_note"] = str(
                        app.post.get_scalar_field_value(
                            "Mag_E",
                            "Maximum",
                            solution="Setup1 : LastAdaptive",
                            object_name=region.name,
                            object_type="surface",
                        )
                    )
                except Exception as exc:
                    outputs["max_electric_field_note"] = f"Electric field postprocess skipped: {{exc}}"
                if matrix:
                    matrix_path = output_dir / "capacitance_matrix.txt"
                    try:
                        app.export_matrix(matrix_name=matrix.name, output_file=matrix_path, setup="Setup1")
                        outputs["matrix_export_path"] = str(matrix_path)
                        matrix_text = matrix_path.read_text(encoding="utf-8", errors="ignore")
                        match = re.search(r"\\bSIG\\s+([0-9.+\\-Ee]+)", matrix_text)
                        if match:
                            capacitance_pf = float(match.group(1))
                            outputs["capacitance_pf"] = capacitance_pf
                            outputs["capacitance_f"] = capacitance_pf * 1e-12
                    except Exception as exc:
                        outputs["matrix_export_note"] = f"Matrix export skipped: {{exc}}"
                app.save_project()
            return outputs
        """
    ).strip()

    return GeneratedMaxwellScript(
        filename="generated_maxwell_job.py",
        entrypoint="run_job",
        summary="根据二维平行板电容器结构化结果生成本地 Maxwell 2D 静电脚本。",
        code=code,
        assumptions=["当云端脚本生成不可用时，回退到本地二维静电平行板电容器脚本。"],
        warnings=["当前本地回退脚本主要覆盖 capacitor_2d 这类规则几何静电任务。"],
    )


def _build_local_script_from_transformer_intake(intake: RequirementIntake) -> GeneratedMaxwellScript:
    spec = intake.simulation_spec or {}
    geometry = spec.get("geometry") if isinstance(spec.get("geometry"), dict) else {}
    excitations = spec.get("excitations") if isinstance(spec.get("excitations"), dict) else {}

    core_width = _coerce_float(geometry.get("core_width_mm")) or 60.0
    core_height = _coerce_float(geometry.get("core_height_mm")) or 80.0
    core_thickness = _coerce_float(geometry.get("core_thickness_mm")) or 20.0
    primary_coil_width = _coerce_float(geometry.get("primary_coil_width_mm")) or 12.0
    secondary_coil_width = _coerce_float(geometry.get("secondary_coil_width_mm")) or 10.0
    coil_height = _coerce_float(geometry.get("coil_height_mm")) or 48.0
    window_gap = _coerce_float(geometry.get("window_gap_mm")) or 8.0
    region_padding = _coerce_float(geometry.get("region_padding_mm")) or 30.0
    primary_voltage_v = _coerce_float(excitations.get("primary_voltage_v")) or 10000.0
    secondary_target_voltage_v = _coerce_float(excitations.get("secondary_target_voltage_v")) or 220.0
    primary_turns = int(round(_coerce_float(excitations.get("primary_turns")) or 500.0))
    secondary_turns = int(round(_coerce_float(excitations.get("secondary_turns")) or 11.0))
    primary_current_a = _coerce_float(excitations.get("primary_current_a")) or 1.0

    code = dedent(
        f"""
        from ansys.aedt.core import Maxwell2d


        def run_job(job: dict) -> dict:
            simulation_spec = dict(job.get("simulation_spec") or {{}})
            geometry = dict(simulation_spec.get("geometry") or {{}})
            excitations = dict(simulation_spec.get("excitations") or {{}})
            constraints = dict(simulation_spec.get("constraints") or {{}})
            project_file = str(job["project_file"])
            version = job.get("maxwell_version")
            non_graphical = bool(job.get("non_graphical", True))
            student_version = bool(job.get("student_version", False))

            core_width = float(geometry.get("core_width_mm", {core_width}))
            core_height = float(geometry.get("core_height_mm", {core_height}))
            core_thickness = float(geometry.get("core_thickness_mm", {core_thickness}))
            primary_coil_width = float(geometry.get("primary_coil_width_mm", {primary_coil_width}))
            secondary_coil_width = float(geometry.get("secondary_coil_width_mm", {secondary_coil_width}))
            coil_height = float(geometry.get("coil_height_mm", {coil_height}))
            window_gap = float(geometry.get("window_gap_mm", {window_gap}))
            region_padding = float(geometry.get("region_padding_mm", {region_padding}))
            primary_voltage_v = float(excitations.get("primary_voltage_v", {primary_voltage_v}))
            secondary_target_voltage_v = float(excitations.get("secondary_target_voltage_v", {secondary_target_voltage_v}))
            primary_turns = int(excitations.get("primary_turns", {primary_turns}))
            secondary_turns = int(excitations.get("secondary_turns", {secondary_turns}))
            primary_current_a = float(excitations.get("primary_current_a", {primary_current_a}))

            outputs: dict[str, float | str] = {{}}
            turns_ratio = secondary_turns / primary_turns if primary_turns else 0.0
            estimated_secondary_voltage_v = primary_voltage_v * turns_ratio
            with Maxwell2d(
                project=project_file,
                design="Transformer2D",
                solution_type="Magnetostatic",
                version=version,
                non_graphical=non_graphical,
                new_desktop=True,
                close_on_exit=True,
                student_version=student_version,
            ) as app:
                app.modeler.model_units = "mm"
                app.modeler.create_rectangle(
                    origin=["0mm", "0mm", "0mm"],
                    sizes=[f"{{core_width}}mm", f"{{core_height}}mm"],
                    name="core",
                    material="steel_1008",
                )
                app.modeler.create_rectangle(
                    origin=[f"{{-primary_coil_width - window_gap}}mm", f"{{(core_height - coil_height) / 2}}mm", "0mm"],
                    sizes=[f"{{primary_coil_width}}mm", f"{{coil_height}}mm"],
                    name="primary",
                    material="copper",
                )
                app.modeler.create_rectangle(
                    origin=[f"{{core_width + window_gap}}mm", f"{{(core_height - coil_height) / 2}}mm", "0mm"],
                    sizes=[f"{{secondary_coil_width}}mm", f"{{coil_height}}mm"],
                    name="secondary",
                    material="copper",
                )
                region = app.modeler.create_region(
                    pad_value=[f"{{region_padding}}mm", f"{{region_padding}}mm", f"{{region_padding}}mm", f"{{region_padding}}mm"],
                    pad_type="Absolute Offset",
                    name="Region",
                )
                app.assign_current("primary", amplitude=f"{{primary_current_a}}A", name="PrimaryCurrent")
                try:
                    app.assign_current("secondary", amplitude="0A", name="SecondaryCurrent")
                except Exception:
                    pass
                try:
                    app.assign_balloon(region.edges, boundary="OuterRegion")
                except Exception:
                    pass
                app.create_setup(name="Setup1", setup_type="Magnetostatic")
                app.save_project()
                solve_ok = bool(app.analyze_setup("Setup1"))
                outputs["solve_status"] = "completed" if solve_ok else "failed"
                if not solve_ok:
                    outputs["status"] = "failed"
                    outputs["notes"] = "Maxwell solve failed."
                    outputs["project_name"] = app.project_name
                    outputs["design_name"] = app.design_name
                    app.save_project()
                    return outputs
                try:
                    outputs["max_flux_density_t"] = float(app.post.get_scalar_field_value("Mag_B", "Maximum", object_name="AllObjects"))
                except Exception as exc:
                    outputs["max_flux_density_note"] = f"Postprocess skipped: {{exc}}"
                outputs["project_name"] = app.project_name
                outputs["design_name"] = app.design_name
                outputs["primary_voltage_v"] = primary_voltage_v
                outputs["secondary_target_voltage_v"] = secondary_target_voltage_v
                outputs["primary_turns"] = primary_turns
                outputs["secondary_turns"] = secondary_turns
                outputs["turns_ratio"] = turns_ratio
                outputs["estimated_secondary_voltage_v"] = estimated_secondary_voltage_v
                app.save_project()
            return outputs
        """
    ).strip()

    return GeneratedMaxwellScript(
        filename="generated_maxwell_job.py",
        entrypoint="run_job",
        summary="\u6839\u636e\u4e8c\u7ef4\u53d8\u538b\u5668\u6982\u5ff5\u89c4\u683c\u751f\u6210\u672c\u5730 PyAEDT \u811a\u672c\u3002",
        code=code,
        assumptions=["\u4e91\u7aef\u811a\u672c\u4e0d\u53ef\u7528\u65f6\uff0c\u56de\u9000\u5230\u672c\u5730 transformer_2d \u9996\u7248\u811a\u672c\u3002"],
        warnings=["\u5f53\u524d transformer_2d \u9996\u7248\u4ee5\u531d\u6570\u6bd4\u4f30\u7b97\u6b21\u7ea7\u7535\u538b\uff0c\u4e3b\u8981\u7528\u4e8e\u6982\u5ff5\u9a8c\u8bc1\u3002"],
    )


def _build_local_script_from_inductor_intake(intake: RequirementIntake) -> GeneratedMaxwellScript:
    spec = intake.simulation_spec or {}
    geometry = spec.get("geometry") if isinstance(spec.get("geometry"), dict) else {}
    excitations = spec.get("excitations") if isinstance(spec.get("excitations"), dict) else {}

    air_gap = _coerce_float(geometry.get("air_gap_mm")) or 0.5
    core_width = _coerce_float(geometry.get("core_width_mm")) or 30.0
    core_height = _coerce_float(geometry.get("core_height_mm")) or 45.0
    core_thickness = _coerce_float(geometry.get("core_thickness_mm")) or 12.0
    coil_width = _coerce_float(geometry.get("coil_width_mm")) or 16.0
    coil_height = _coerce_float(geometry.get("coil_height_mm")) or 22.0
    region_padding = _coerce_float(geometry.get("region_padding_mm")) or 20.0
    current_a = _coerce_float(excitations.get("current_a")) or 2.0
    coil_turns = int(round(_coerce_float(excitations.get("coil_turns")) or 600.0))

    code = dedent(
        f"""
        import math
        from ansys.aedt.core import Maxwell2d


        def run_job(job: dict) -> dict:
            simulation_spec = dict(job.get("simulation_spec") or {{}})
            geometry = dict(simulation_spec.get("geometry") or {{}})
            excitations = dict(simulation_spec.get("excitations") or {{}})
            constraints = dict(simulation_spec.get("constraints") or {{}})
            project_file = str(job["project_file"])
            version = job.get("maxwell_version")
            non_graphical = bool(job.get("non_graphical", True))
            student_version = bool(job.get("student_version", False))

            air_gap = float(geometry.get("air_gap_mm", {air_gap}))
            core_width = float(geometry.get("core_width_mm", {core_width}))
            core_height = float(geometry.get("core_height_mm", {core_height}))
            core_thickness = float(geometry.get("core_thickness_mm", {core_thickness}))
            coil_width = float(geometry.get("coil_width_mm", {coil_width}))
            coil_height = float(geometry.get("coil_height_mm", {coil_height}))
            region_padding = float(geometry.get("region_padding_mm", {region_padding}))
            current_a = float(excitations.get("current_a", {current_a}))
            coil_turns = int(excitations.get("coil_turns", {coil_turns}))
            target_inductance_h = constraints.get("target_inductance_h")
            try:
                target_inductance_h = float(target_inductance_h) if target_inductance_h is not None else None
            except (TypeError, ValueError):
                target_inductance_h = None

            outputs: dict[str, float | str] = {{}}
            mu0 = 4 * math.pi * 1e-7
            effective_gap_m = max(air_gap, 0.1) * 1e-3
            if target_inductance_h is not None and target_inductance_h > 0 and coil_turns > 0:
                target_area_m2 = target_inductance_h * effective_gap_m / max(mu0 * coil_turns * coil_turns, 1e-18)
                target_area_mm2 = target_area_m2 * 1e6
                core_width = max(0.5, target_area_mm2 / max(core_thickness, 0.1))
                core_height = max(core_height, core_width + 2.0 * core_thickness)
            core_area_m2 = (core_thickness * core_width) * 1e-6
            estimated_inductance_h = mu0 * coil_turns * coil_turns * core_area_m2 / max(effective_gap_m, 1e-6)
            estimated_max_flux_density_t = abs(estimated_inductance_h * current_a / max(coil_turns * core_area_m2, 1e-18))
            with Maxwell2d(
                project=project_file,
                design="Inductor2D",
                solution_type="Magnetostatic",
                version=version,
                non_graphical=non_graphical,
                new_desktop=True,
                close_on_exit=True,
                student_version=student_version,
            ) as app:
                app.modeler.model_units = "mm"
                app.modeler.create_rectangle(
                    origin=["0mm", "0mm", "0mm"],
                    sizes=[f"{{core_width}}mm", f"{{core_height}}mm"],
                    name="core",
                    material="steel_1008",
                )
                app.modeler.create_rectangle(
                    origin=[f"{{core_width}}mm", f"{{(core_height - core_thickness) / 2}}mm", "0mm"],
                    sizes=[f"{{air_gap}}mm", f"{{core_thickness}}mm"],
                    name="gap_block",
                    material="vacuum",
                )
                app.modeler.create_rectangle(
                    origin=[f"{{-coil_width}}mm", f"{{(core_height - coil_height) / 2}}mm", "0mm"],
                    sizes=[f"{{coil_width}}mm", f"{{coil_height}}mm"],
                    name="coil",
                    material="copper",
                )
                region = app.modeler.create_region(
                    pad_value=[f"{{region_padding}}mm", f"{{region_padding}}mm", f"{{region_padding}}mm", f"{{region_padding}}mm"],
                    pad_type="Absolute Offset",
                    name="Region",
                )
                app.assign_current("coil", amplitude=f"{{current_a}}A", name="Current1")
                try:
                    app.assign_balloon(region.edges, boundary="OuterRegion")
                except Exception:
                    pass
                app.create_setup(name="Setup1", setup_type="Magnetostatic")
                app.save_project()
                outputs["status"] = "completed"
                outputs["solve_status"] = "skipped_fast_estimate"
                outputs["execution_mode"] = "maxwell_project_saved_plus_magnetic_circuit_estimate"
                outputs["project_name"] = app.project_name
                outputs["design_name"] = app.design_name
                outputs["air_gap_mm"] = air_gap
                outputs["core_width_mm"] = core_width
                outputs["core_height_mm"] = core_height
                outputs["core_thickness_mm"] = core_thickness
                outputs["current_a"] = current_a
                outputs["coil_turns"] = coil_turns
                outputs["estimated_inductance_h"] = estimated_inductance_h
                outputs["max_flux_density_t"] = estimated_max_flux_density_t
                if target_inductance_h is not None:
                    outputs["target_inductance_h"] = target_inductance_h
                app.save_project()
            return outputs
        """
    ).strip()

    return GeneratedMaxwellScript(
        filename="generated_maxwell_job.py",
        entrypoint="run_job",
        summary="\u6839\u636e\u4e8c\u7ef4\u7535\u611f\u89c4\u683c\u751f\u6210\u672c\u5730 PyAEDT \u811a\u672c\u3002",
        code=code,
        assumptions=["\u4e91\u7aef\u811a\u672c\u4e0d\u53ef\u7528\u65f6\uff0c\u56de\u9000\u5230\u672c\u5730 inductor_2d \u9996\u7248\u811a\u672c\u3002"],
        warnings=["\u5f53\u524d inductor_2d \u9996\u7248\u4f7f\u7528 Maxwell \u78c1\u573a + \u7b80\u5316\u7535\u611f\u4f30\u7b97\u8054\u5408\u8f93\u51fa\u3002"],
    )


def _build_local_script_from_busbar_intake(intake: RequirementIntake) -> GeneratedMaxwellScript:
    plan = _build_busbar_ir_plan(intake)
    return _render_ir_script_for_intake(
        intake,
        plan,
        summary="根据载流母排规格生成本地 PyAEDT 脚本。",
        assumptions=["使用二维矩形载流导体模型验证磁场和截面电流密度。"],
        warnings=["当前 busbar_2d 首版已经迁移到统一 Maxwell IR 路径。"],
    )
    spec = intake.simulation_spec or {}
    geometry = spec.get("geometry") if isinstance(spec.get("geometry"), dict) else {}
    excitations = spec.get("excitations") if isinstance(spec.get("excitations"), dict) else {}

    width_mm = _coerce_float(geometry.get("width_mm")) or 10.0
    thickness_mm = _coerce_float(geometry.get("thickness_mm")) or 2.0
    region_padding = _coerce_float(geometry.get("region_padding_mm")) or 20.0
    current_a = _coerce_float(excitations.get("current_a")) or 200.0

    code = dedent(
        f"""
        import math
        from ansys.aedt.core import Maxwell2d


        def run_job(job: dict) -> dict:
            simulation_spec = dict(job.get("simulation_spec") or {{}})
            geometry = dict(simulation_spec.get("geometry") or {{}})
            excitations = dict(simulation_spec.get("excitations") or {{}})
            project_file = str(job["project_file"])
            version = job.get("maxwell_version")
            non_graphical = bool(job.get("non_graphical", True))
            student_version = bool(job.get("student_version", False))

            width_mm = float(geometry.get("width_mm", {width_mm}))
            thickness_mm = float(geometry.get("thickness_mm", {thickness_mm}))
            region_padding = float(geometry.get("region_padding_mm", {region_padding}))
            current_a = float(excitations.get("current_a", {current_a}))
            cross_section_area_mm2 = max(width_mm * thickness_mm, 1e-9)
            estimated_current_density_a_per_mm2 = current_a / cross_section_area_mm2
            mu0 = 4 * math.pi * 1e-7
            reference_surface_field_t = mu0 * current_a / max(math.pi * thickness_mm * 1e-3, 1e-9)

            outputs: dict[str, float | str] = {{}}
            with Maxwell2d(
                project=project_file,
                design="Busbar2D",
                solution_type="Magnetostatic",
                version=version,
                non_graphical=non_graphical,
                new_desktop=True,
                close_on_exit=True,
                student_version=student_version,
            ) as app:
                app.modeler.model_units = "mm"
                conductor = app.modeler.create_rectangle(
                    origin=[f"{{-width_mm / 2}}mm", f"{{-thickness_mm / 2}}mm", "0mm"],
                    sizes=[f"{{width_mm}}mm", f"{{thickness_mm}}mm"],
                    name="busbar",
                    material="copper",
                )
                region = app.modeler.create_region(
                    pad_value=[f"{{region_padding}}mm", f"{{region_padding}}mm", f"{{region_padding}}mm", f"{{region_padding}}mm"],
                    pad_type="Absolute Offset",
                    name="Region",
                )
                app.assign_current([conductor], amplitude=f"{{current_a}}A", name="BusbarCurrent")
                try:
                    app.assign_balloon(region.edges, boundary="OuterRegion")
                except Exception:
                    pass
                app.create_setup(name="Setup1", setup_type="Magnetostatic")
                app.save_project()
                solve_ok = bool(app.analyze_setup("Setup1"))
                outputs["solve_status"] = "completed" if solve_ok else "failed"
                if not solve_ok:
                    outputs["status"] = "failed"
                    outputs["notes"] = "Maxwell solve failed."
                    outputs["project_name"] = app.project_name
                    outputs["design_name"] = app.design_name
                    app.save_project()
                    return outputs
                try:
                    outputs["max_flux_density_t"] = float(app.post.get_scalar_field_value("Mag_B", "Maximum", object_name="AllObjects"))
                except Exception as exc:
                    outputs["max_flux_density_note"] = f"Postprocess skipped: {{exc}}"
                outputs["project_name"] = app.project_name
                outputs["design_name"] = app.design_name
                outputs["width_mm"] = width_mm
                outputs["thickness_mm"] = thickness_mm
                outputs["current_a"] = current_a
                outputs["cross_section_area_mm2"] = cross_section_area_mm2
                outputs["estimated_current_density_a_per_mm2"] = estimated_current_density_a_per_mm2
                outputs["reference_surface_field_t"] = reference_surface_field_t
                app.save_project()
            return outputs
        """
    ).strip()

    return GeneratedMaxwellScript(
        filename="generated_maxwell_job.py",
        entrypoint="run_job",
        summary="\u6839\u636e\u8f7d\u6d41\u6bcd\u6392\u89c4\u683c\u751f\u6210\u672c\u5730 PyAEDT \u811a\u672c\u3002",
        code=code,
        assumptions=["\u4f7f\u7528\u4e8c\u7ef4\u77e9\u5f62\u8f7d\u6d41\u5bfc\u4f53\u6a21\u578b\u9a8c\u8bc1\u78c1\u573a\u548c\u622a\u9762\u7535\u6d41\u5bc6\u5ea6\u3002"],
        warnings=["\u5f53\u524d busbar_2d \u9996\u7248\u4e3b\u8981\u8986\u76d6\u5355\u6839\u5bfc\u4f53\uff0c\u6682\u672a\u542b\u76f8\u95f4\u8026\u5408\u548c\u70ed\u8bbe\u8ba1\u3002"],
    )


def _build_local_script_from_solenoid_intake(intake: RequirementIntake) -> GeneratedMaxwellScript:
    plan = _build_solenoid_ir_plan(intake)
    return _render_ir_script_for_intake(
        intake,
        plan,
        summary="根据空芯螺线管规格生成本地 PyAEDT 脚本。",
        assumptions=["使用二维等效上下线圈表示空芯螺线管截面。"],
        warnings=["当前 solenoid_2d 首版已经迁移到统一 Maxwell IR 路径。"],
    )
    spec = intake.simulation_spec or {}
    geometry = spec.get("geometry") if isinstance(spec.get("geometry"), dict) else {}
    excitations = spec.get("excitations") if isinstance(spec.get("excitations"), dict) else {}

    length_mm = _coerce_float(geometry.get("length_mm")) or 50.0
    radius_mm = _coerce_float(geometry.get("radius_mm")) or 8.0
    coil_thickness_mm = _coerce_float(geometry.get("coil_thickness_mm")) or 2.0
    region_padding = _coerce_float(geometry.get("region_padding_mm")) or 25.0
    current_a = _coerce_float(excitations.get("current_a")) or 1.0
    coil_turns = int(round(_coerce_float(excitations.get("coil_turns")) or 300.0))

    code = dedent(
        f"""
        import math
        from ansys.aedt.core import Maxwell2d


        def run_job(job: dict) -> dict:
            simulation_spec = dict(job.get("simulation_spec") or {{}})
            geometry = dict(simulation_spec.get("geometry") or {{}})
            excitations = dict(simulation_spec.get("excitations") or {{}})
            project_file = str(job["project_file"])
            version = job.get("maxwell_version")
            non_graphical = bool(job.get("non_graphical", True))
            student_version = bool(job.get("student_version", False))

            length_mm = float(geometry.get("length_mm", {length_mm}))
            radius_mm = float(geometry.get("radius_mm", {radius_mm}))
            coil_thickness_mm = float(geometry.get("coil_thickness_mm", {coil_thickness_mm}))
            region_padding = float(geometry.get("region_padding_mm", {region_padding}))
            current_a = float(excitations.get("current_a", {current_a}))
            coil_turns = int(excitations.get("coil_turns", {coil_turns}))
            equivalent_current_a = current_a * coil_turns
            mu0 = 4 * math.pi * 1e-7
            center_flux_density_t = mu0 * coil_turns * current_a / max(length_mm * 1e-3, 1e-9)

            outputs: dict[str, float | str] = {{}}
            with Maxwell2d(
                project=project_file,
                design="Solenoid2D",
                solution_type="Magnetostatic",
                version=version,
                non_graphical=non_graphical,
                new_desktop=True,
                close_on_exit=True,
                student_version=student_version,
            ) as app:
                app.modeler.model_units = "mm"
                app.modeler.create_rectangle(
                    origin=[f"{{-length_mm / 2}}mm", f"{{radius_mm}}mm", "0mm"],
                    sizes=[f"{{length_mm}}mm", f"{{coil_thickness_mm}}mm"],
                    name="upper_coil",
                    material="copper",
                )
                app.modeler.create_rectangle(
                    origin=[f"{{-length_mm / 2}}mm", f"{{-radius_mm - coil_thickness_mm}}mm", "0mm"],
                    sizes=[f"{{length_mm}}mm", f"{{coil_thickness_mm}}mm"],
                    name="lower_coil",
                    material="copper",
                )
                region = app.modeler.create_region(
                    pad_value=[f"{{region_padding}}mm", f"{{region_padding}}mm", f"{{region_padding}}mm", f"{{region_padding}}mm"],
                    pad_type="Absolute Offset",
                    name="Region",
                )
                app.assign_current("upper_coil", amplitude=f"{{equivalent_current_a}}A", name="UpperCurrent")
                app.assign_current("lower_coil", amplitude=f"{{-equivalent_current_a}}A", name="LowerCurrent")
                try:
                    app.assign_balloon(region.edges, boundary="OuterRegion")
                except Exception:
                    pass
                app.create_setup(name="Setup1", setup_type="Magnetostatic")
                app.save_project()
                solve_ok = bool(app.analyze_setup("Setup1"))
                outputs["solve_status"] = "completed" if solve_ok else "failed"
                if not solve_ok:
                    outputs["status"] = "failed"
                    outputs["notes"] = "Maxwell solve failed."
                    outputs["project_name"] = app.project_name
                    outputs["design_name"] = app.design_name
                    app.save_project()
                    return outputs
                try:
                    outputs["max_flux_density_t"] = float(app.post.get_scalar_field_value("Mag_B", "Maximum", object_name="AllObjects"))
                except Exception as exc:
                    outputs["max_flux_density_note"] = f"Postprocess skipped: {{exc}}"
                outputs["project_name"] = app.project_name
                outputs["design_name"] = app.design_name
                outputs["length_mm"] = length_mm
                outputs["radius_mm"] = radius_mm
                outputs["current_a"] = current_a
                outputs["coil_turns"] = coil_turns
                outputs["equivalent_current_a"] = equivalent_current_a
                outputs["estimated_center_flux_density_t"] = center_flux_density_t
                app.save_project()
            return outputs
        """
    ).strip()

    return GeneratedMaxwellScript(
        filename="generated_maxwell_job.py",
        entrypoint="run_job",
        summary="\u6839\u636e\u7a7a\u82af\u87ba\u7ebf\u7ba1\u89c4\u683c\u751f\u6210\u672c\u5730 PyAEDT \u811a\u672c\u3002",
        code=code,
        assumptions=["\u4f7f\u7528\u4e8c\u7ef4\u7b49\u6548\u4e0a\u4e0b\u7ebf\u5708\u8868\u793a\u7a7a\u82af\u87ba\u7ebf\u7ba1\u622a\u9762\u3002"],
        warnings=["\u8be5\u811a\u672c\u8f93\u51fa Maxwell \u78c1\u573a\u7ed3\u679c\u548c\u4e2d\u5fc3\u78c1\u5bc6\u89e3\u6790\u53c2\u8003\u503c\u3002"],
    )


def _build_local_script_from_coaxial_capacitor_intake(intake: RequirementIntake) -> GeneratedMaxwellScript:
    plan = _build_coaxial_capacitor_ir_plan(intake)
    return _render_ir_script_for_intake(
        intake,
        plan,
        summary="根据同轴电容器规格生成本地 Maxwell 2D 静电脚本。",
        assumptions=["使用二维同轴截面模型提取电容和电场结果。"],
        warnings=["当前 coaxial_capacitor_2d 首版已经迁移到统一 Maxwell IR 路径。"],
    )
    spec = intake.simulation_spec or {}
    geometry = spec.get("geometry") if isinstance(spec.get("geometry"), dict) else {}
    excitations = spec.get("excitations") if isinstance(spec.get("excitations"), dict) else {}

    inner_radius_mm = _coerce_float(geometry.get("inner_radius_mm")) or 1.0
    outer_radius_mm = _coerce_float(geometry.get("outer_radius_mm")) or 5.0
    region_padding = _coerce_float(geometry.get("region_padding_mm")) or 8.0
    voltage_v = _coerce_float(excitations.get("voltage_V")) or 100.0

    code = dedent(
        f"""
        import math
        import re
        from pathlib import Path
        from ansys.aedt.core import Maxwell2d
        from ansys.aedt.core.modules.boundary.maxwell_boundary import MatrixElectric


        def _create_filled_circle(app, name: str, radius_mm: float, material: str):
            return app.modeler.create_circle(
                origin=[0, 0, 0],
                radius=radius_mm,
                name=name,
                material=material,
            )


        def run_job(job: dict) -> dict:
            simulation_spec = dict(job.get("simulation_spec") or {{}})
            geometry = dict(simulation_spec.get("geometry") or {{}})
            excitations = dict(simulation_spec.get("excitations") or {{}})
            output_dir = Path(str(job["output_dir"]))
            project_file = str(job["project_file"])
            version = job.get("maxwell_version")
            non_graphical = bool(job.get("non_graphical", True))
            student_version = bool(job.get("student_version", False))

            inner_radius_mm = float(geometry.get("inner_radius_mm", {inner_radius_mm}))
            outer_radius_mm = float(geometry.get("outer_radius_mm", {outer_radius_mm}))
            region_padding = float(geometry.get("region_padding_mm", {region_padding}))
            voltage_v = float(excitations.get("voltage_V", {voltage_v}))
            outer_conductor_thickness_mm = max(1.0, 0.2 * outer_radius_mm)
            epsilon_0 = 8.854187817e-12
            capacitance_per_m_f = 2 * math.pi * epsilon_0 / math.log(outer_radius_mm / inner_radius_mm)
            reference_electric_field_v_per_m = voltage_v / ((outer_radius_mm - inner_radius_mm) * 1e-3)
            reference_max_field_v_per_m = voltage_v / max(inner_radius_mm * 1e-3 * math.log(outer_radius_mm / inner_radius_mm), 1e-12)

            outputs: dict[str, float | str] = {{}}
            with Maxwell2d(
                project=project_file,
                design="CoaxialCapacitor2D",
                solution_type="ElectrostaticXY",
                version=version,
                non_graphical=non_graphical,
                new_desktop=True,
                close_on_exit=True,
                student_version=student_version,
            ) as app:
                app.modeler.model_units = "mm"
                inner_conductor = _create_filled_circle(app, "inner_conductor", inner_radius_mm, "pec")
                outer_conductor = _create_filled_circle(
                    app,
                    "outer_conductor",
                    outer_radius_mm + outer_conductor_thickness_mm,
                    "pec",
                )
                outer_void = _create_filled_circle(app, "outer_void", outer_radius_mm, "vacuum")
                app.modeler.subtract(
                    blank_list=[outer_conductor.name],
                    tool_list=[outer_void.name],
                    keep_originals=False,
                )
                region = app.modeler.create_region(
                    pad_value=[f"{{region_padding}}mm", f"{{region_padding}}mm", f"{{region_padding}}mm", f"{{region_padding}}mm"],
                    pad_type="Absolute Offset",
                    name="Region",
                )
                v_inner = app.assign_voltage([inner_conductor], amplitude=f"{{voltage_v}}V", name="InnerVoltage")
                v_outer = app.assign_voltage([outer_conductor], amplitude="0V", name="OuterGround")
                try:
                    app.assign_balloon(region.edges, boundary="OuterRegion", is_voltage=True)
                except Exception:
                    pass
                try:
                    matrix = app.assign_matrix(
                        MatrixElectric(
                            signal_sources=[v_inner.name],
                            ground_sources=[v_outer.name],
                            matrix_name="CoaxCapMatrix",
                        )
                    )
                except Exception:
                    matrix = None
                app.create_setup(name="Setup1", setup_type="Electrostatic")
                app.save_project()
                solve_ok = bool(app.analyze_setup("Setup1"))
                outputs["solve_status"] = "completed" if solve_ok else "failed"
                if not solve_ok:
                    outputs["status"] = "failed"
                    outputs["notes"] = "Maxwell electrostatic solve failed."
                    outputs["project_name"] = app.project_name
                    outputs["design_name"] = app.design_name
                    app.save_project()
                    return outputs
                outputs["project_name"] = app.project_name
                outputs["design_name"] = app.design_name
                outputs["inner_radius_mm"] = inner_radius_mm
                outputs["outer_radius_mm"] = outer_radius_mm
                outputs["applied_voltage_v"] = voltage_v
                outputs["reference_capacitance_f_per_m"] = capacitance_per_m_f
                outputs["reference_capacitance_pf_per_m"] = capacitance_per_m_f * 1e12
                outputs["reference_average_field_v_per_m"] = reference_electric_field_v_per_m
                outputs["reference_max_field_v_per_m"] = reference_max_field_v_per_m
                try:
                    outputs["max_electric_field_v_per_m"] = float(
                        app.post.get_scalar_field_value(
                            "Mag_E",
                            "Maximum",
                            solution="Setup1 : LastAdaptive",
                            object_name="AllObjects",
                        )
                    )
                except Exception as exc:
                    outputs["max_electric_field_note"] = f"Electric field postprocess skipped: {{exc}}"
                if matrix:
                    matrix_path = output_dir / "coaxial_capacitance_matrix.txt"
                    try:
                        app.export_matrix(matrix_name=matrix.name, output_file=matrix_path, setup="Setup1")
                        outputs["matrix_export_path"] = str(matrix_path)
                        matrix_text = matrix_path.read_text(encoding="utf-8", errors="ignore")
                        match = re.search(r"\\bInnerVoltage\\s+([0-9.+\\-Ee]+)", matrix_text)
                        if match:
                            capacitance_pf = float(match.group(1))
                            outputs["capacitance_pf"] = capacitance_pf
                            outputs["capacitance_f"] = capacitance_pf * 1e-12
                    except Exception as exc:
                        outputs["matrix_export_note"] = f"Matrix export skipped: {{exc}}"
                app.save_project()
            return outputs
        """
    ).strip()

    return GeneratedMaxwellScript(
        filename="generated_maxwell_job.py",
        entrypoint="run_job",
        summary="\u6839\u636e\u540c\u8f74\u7535\u5bb9\u89c4\u683c\u751f\u6210\u672c\u5730 PyAEDT \u811a\u672c\u3002",
        code=code,
        assumptions=["\u9996\u7248\u4ee5\u4e8c\u7ef4\u622a\u9762\u548c\u771f\u7a7a\u4ecb\u8d28\u5efa\u6a21\uff0c\u5e76\u7ed9\u51fa\u6bcf\u7c73\u957f\u5ea6\u89e3\u6790\u7535\u5bb9\u53c2\u8003\u3002"],
        warnings=["\u5b9e\u9645\u540c\u8f74\u7535\u7f06\u8fd8\u9700\u6269\u5c55\u4ecb\u8d28\u6750\u6599\u3001\u5c4f\u853d\u5c42\u548c\u635f\u8017\u8bc4\u4f30\u3002"],
    )


def _apply_design_patch(
    current_design: ElectromagnetDesign,
    patch: ElectromagnetDesignPatch,
    requirement: str,
) -> ElectromagnetDesign:
    payload = current_design.model_dump(mode="json")
    patch_payload = patch.model_dump(mode="json", exclude_none=True)

    if "summary" in patch_payload:
        payload["summary"] = patch_payload.pop("summary")

    patch_assumptions = patch_payload.pop("assumptions", [])
    patch_warnings = patch_payload.pop("warnings", [])
    for key, value in patch_payload.items():
        payload[key] = value

    payload["assumptions"] = _merge_unique_strings(payload.get("assumptions"), patch_assumptions)
    payload["warnings"] = _merge_unique_strings(payload.get("warnings"), patch_warnings)
    return _validate_design_payload(payload, requirement)


class CodexaLLMClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.codexa_base_url:
            raise ValueError("CODEXA_BASE_URL is not configured.")
        if not settings.codexa_api_key:
            raise ValueError("CODEXA_API_KEY is not configured.")
        self._settings = settings
        self._primitive_library = PrimitiveLibrary(settings.primitive_library_path)
        self._client = OpenAI(
            api_key=settings.codexa_api_key,
            base_url=settings.codexa_base_url,
        )

    def list_models(self) -> list[str]:
        response = self._client.models.list()
        return [item.id for item in response.data]

    def smoke_test(self) -> str:
        response = self._create_response_with_retry(
            model=self._settings.codexa_model,
            input="Reply with only the word OK.",
            reasoning={"effort": "none"},
            store=False,
            timeout=self._settings.codexa_timeout_s,
        )
        return response.output_text.strip()

    def _create_response_with_retry(self, **kwargs):
        max_attempts = int(kwargs.pop("max_attempts", 4))
        last_error: Exception | None = None
        for attempt in range(max(1, max_attempts)):
            try:
                return self._client.responses.create(**kwargs)
            except Exception as exc:
                last_error = exc
                if attempt >= max(1, max_attempts) - 1:
                    break
                time.sleep(2.0 * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unexpected empty response state.")

    def _call_json(
        self,
        instructions: str,
        input_payload: Any,
        schema_model: type,
        effort: str | None = None,
        timeout_s: int | None = None,
        max_attempts: int = 4,
    ) -> dict[str, Any]:
        reasoning_effort = effort or self._settings.codexa_reasoning_effort
        schema: dict[str, Any] = {
            "type": "json_schema",
            "name": f"{schema_model.__name__.lower()}_schema",
            "schema": schema_model.model_json_schema(),
            "strict": True,
        }
        response = self._create_response_with_retry(
            model=self._settings.codexa_model,
            instructions=instructions,
            input=_as_response_input(input_payload),
            text={"format": schema},
            reasoning={"effort": reasoning_effort},
            store=False,
            timeout=timeout_s or self._settings.codexa_timeout_s,
            max_attempts=max_attempts,
        )
        return json.loads(response.output_text)

    def _call_json_fallback(
        self,
        instructions: str,
        input_payload: Any,
        effort: str | None = None,
        timeout_s: int | None = None,
        max_attempts: int = 4,
    ) -> dict[str, Any]:
        reasoning_effort = effort or self._settings.codexa_reasoning_effort
        response = self._create_response_with_retry(
            model=self._settings.codexa_model,
            instructions=instructions,
            input=_as_response_input(input_payload),
            reasoning={"effort": reasoning_effort},
            store=False,
            timeout=timeout_s or self._settings.codexa_timeout_s,
            max_attempts=max_attempts,
        )
        return json.loads(_extract_json_blob(response.output_text))

    @property
    def primitive_library(self) -> PrimitiveLibrary:
        return self._primitive_library

    def learn_primitive_template(
        self,
        requirement: str,
        intake: RequirementIntake,
        primitive_token: str,
        raw_object: dict[str, Any],
        error_details: str | None = None,
        previous_template: PrimitiveTemplate | None = None,
    ) -> PrimitiveTemplateArtifact:
        payload = {
            "requirement": requirement.strip(),
            "intake": intake.model_dump(mode="json"),
            "primitive_token": primitive_token,
            "raw_geometry_object": raw_object,
            "error_details": error_details or "",
        }
        if previous_template is None:
            try:
                result = self._call_json(
                    build_primitive_template_generation_instructions(),
                    payload,
                    PrimitiveTemplateArtifact,
                )
                return _validate_primitive_artifact_payload(result)
            except Exception:
                result = self._call_json_fallback(
                    build_primitive_template_generation_instructions(),
                    payload,
                )
                return _validate_primitive_artifact_payload(result)

        payload["previous_template"] = previous_template.model_dump(mode="json")
        try:
            result = self._call_json(
                build_primitive_template_repair_instructions(),
                payload,
                PrimitiveTemplateArtifact,
            )
            return _validate_primitive_artifact_payload(result)
        except Exception:
            result = self._call_json_fallback(
                build_primitive_template_repair_instructions(),
                payload,
            )
            return _validate_primitive_artifact_payload(result)

    def generate_requirement_intake(self, requirement: str) -> RequirementIntake:
        requirement = requirement.strip()
        try:
            payload = self._call_json(
                build_requirement_structuring_instructions(),
                requirement,
                RequirementIntake,
            )
            intake = _validate_intake_payload(payload, requirement)
            return _rescue_supported_fallback_intake(requirement, intake)
        except Exception:
            try:
                payload = self._call_json_fallback(
                    build_requirement_structuring_instructions(),
                    requirement,
                )
                intake = _validate_intake_payload(payload, requirement)
                return _rescue_supported_fallback_intake(requirement, intake)
            except Exception:
                return _fallback_intake_from_requirement(requirement)

    def refine_requirement_intake(self, requirement: str, intake: RequirementIntake) -> RequirementIntake:
        if intake.supported_now and (
            intake.design is not None
            or bool(intake.simulation_spec.get("execution_ready"))
            or bool(intake.execution_plan.get("execution_ready"))
        ):
            return _rescue_supported_fallback_intake(requirement, intake)
        payload = {
            "requirement": requirement.strip(),
            "current_intake": intake.model_dump(mode="json"),
        }
        try:
            refined = self._call_json(
                build_spec_refinement_instructions(),
                payload,
                RequirementIntake,
            )
            resolved = _validate_intake_payload(refined, requirement)
            return _rescue_supported_fallback_intake(requirement, resolved)
        except Exception:
            try:
                refined = self._call_json_fallback(
                    build_spec_refinement_instructions(),
                    payload,
                )
                resolved = _validate_intake_payload(refined, requirement)
                return _rescue_supported_fallback_intake(requirement, resolved)
            except Exception:
                resolved = _validate_intake_payload(intake.model_dump(mode="json"), requirement)
                return _rescue_supported_fallback_intake(requirement, resolved)

    def generate_ir_artifact(self, requirement: str, intake: RequirementIntake) -> GeneratedIRPlan:
        payload = {
            "requirement": requirement.strip(),
            "intake": intake.model_dump(mode="json"),
        }
        try:
            result = self._call_json_fallback(
                build_ir_generation_instructions(),
                payload,
            )
            return _validate_ir_artifact_payload(result)
        except Exception:
            result = self._call_json(
                build_ir_generation_instructions(),
                payload,
                GeneratedIRPlan,
            )
            return _validate_ir_artifact_payload(result)

    def repair_ir_artifact(
        self,
        requirement: str,
        intake: RequirementIntake,
        previous_artifact: GeneratedIRPlan,
        failure_stage: str,
        error_details: str,
    ) -> GeneratedIRPlan:
        payload = {
            "requirement": requirement.strip(),
            "intake": intake.model_dump(mode="json"),
            "previous_ir_artifact": previous_artifact.model_dump(mode="json"),
            "failure_stage": failure_stage,
            "error_details": error_details,
        }
        try:
            result = self._call_json_fallback(
                build_ir_repair_instructions(),
                payload,
            )
            return _validate_ir_artifact_payload(result)
        except Exception:
            result = self._call_json(
                build_ir_repair_instructions(),
                payload,
                GeneratedIRPlan,
            )
            return _validate_ir_artifact_payload(result)

    def revise_ir_patch_from_feedback(
        self,
        requirement: str,
        intake: RequirementIntake,
        previous_artifact: GeneratedIRPlan,
        outputs: dict[str, float | str] | None,
        evaluation: Any | None,
        feedback_round: int,
    ) -> IRPatch:
        filtered_outputs = {
            str(key): value
            for key, value in (outputs or {}).items()
            if isinstance(value, (str, float, int, bool))
        }
        residuals = compact_residual_payload(analyze_requirement_residuals(outputs or {}, evaluation))
        payload = {
            "requirement": requirement.strip(),
            "current_intake": _compact_intake_for_ir_feedback(intake),
            "current_ir_artifact": previous_artifact.model_dump(mode="json"),
            "previous_outputs": filtered_outputs,
            "previous_evaluation": _compact_feedback_evaluation(evaluation),
            "residuals": residuals,
            "feedback_round": feedback_round,
            "hard_constraints": (
                dict(intake.simulation_spec.get("constraints") or {})
                if isinstance(intake.simulation_spec, dict)
                else {}
            ),
            "allowed_patch_operations": [
                "set_parameter_default",
                "set_local_expression",
                "set_object_material",
                "add_warning",
            ],
        }
        try:
            result = self._call_json_fallback(
                build_ir_patch_feedback_instructions(),
                payload,
                effort="medium",
                timeout_s=max(20, min(self._settings.codexa_timeout_s, 45)),
                max_attempts=1,
            )
            return _validate_ir_patch_payload(result)
        except Exception as fallback_exc:
            try:
                result = self._call_json(
                    build_ir_patch_feedback_instructions(),
                    payload,
                    IRPatch,
                    effort="medium",
                    timeout_s=max(20, min(self._settings.codexa_timeout_s, 45)),
                    max_attempts=1,
                )
                return _validate_ir_patch_payload(result)
            except Exception as strict_exc:
                raise RequirementPlanningError(
                    f"LLM IR patch revision failed. Fallback error: {fallback_exc}; strict-schema error: {strict_exc}"
                ) from strict_exc

    def revise_ir_artifact_from_feedback(
        self,
        requirement: str,
        intake: RequirementIntake,
        outputs: dict[str, float | str] | None,
        evaluation: Any | None,
        feedback_round: int,
    ) -> GeneratedIRPlan:
        previous_artifact = _build_ir_artifact_from_intake(intake)
        try:
            patch = self.revise_ir_patch_from_feedback(
                requirement=requirement,
                intake=intake,
                previous_artifact=previous_artifact,
                outputs=outputs,
                evaluation=evaluation,
                feedback_round=feedback_round,
            )
            patched_plan = apply_ir_patch(previous_artifact.ir_plan, patch)
            summary = patch.summary or f"已根据第 {feedback_round} 轮仿真残差修订 Maxwell IR。"
            return GeneratedIRPlan(
                summary=summary,
                ir_plan=patched_plan,
                assumptions=list(previous_artifact.assumptions),
                warnings=_merge_unique_strings(previous_artifact.warnings, patch.warnings, patch.expected_effects),
            )
        except Exception:
            pass

        filtered_outputs = {
            str(key): value
            for key, value in (outputs or {}).items()
            if isinstance(value, (str, float, int, bool))
        }
        payload = {
            "requirement": requirement.strip(),
            "current_intake": _compact_intake_for_ir_feedback(intake),
            "current_ir_artifact": previous_artifact.model_dump(mode="json"),
            "previous_outputs": filtered_outputs,
            "previous_evaluation": _compact_feedback_evaluation(evaluation),
            "residuals": compact_residual_payload(analyze_requirement_residuals(outputs or {}, evaluation)),
            "feedback_round": feedback_round,
            "hard_constraints": (
                dict(intake.simulation_spec.get("constraints") or {})
                if isinstance(intake.simulation_spec, dict)
                else {}
            ),
        }
        try:
            result = self._call_json_fallback(
                build_ir_feedback_instructions(),
                payload,
                effort="medium",
                timeout_s=max(20, min(self._settings.codexa_timeout_s, 45)),
                max_attempts=1,
            )
            return _validate_ir_artifact_payload(result)
        except Exception as fallback_exc:
            try:
                result = self._call_json(
                    build_ir_feedback_instructions(),
                    payload,
                    GeneratedIRPlan,
                    effort="medium",
                    timeout_s=max(20, min(self._settings.codexa_timeout_s, 45)),
                    max_attempts=1,
                )
                return _validate_ir_artifact_payload(result)
            except Exception as strict_exc:
                raise RequirementPlanningError(
                    f"LLM IR feedback revision failed. Fallback error: {fallback_exc}; strict-schema error: {strict_exc}"
                ) from strict_exc

    @staticmethod
    def _coerce_intake_to_generic_ir_path(requirement: str, intake: RequirementIntake) -> RequirementIntake:
        if intake.design is not None:
            return intake
        if intake.task_family in {"unknown", "generic_maxwell"}:
            return intake
        semantic_hint = infer_builder_hint(intake, requirement=requirement)
        if intake.task_family == "generic_maxwell" and intake_has_generic_object_graph(intake):
            return intake
        if semantic_hint == "generic_maxwell":
            return intake
        if intake.task_family not in {"unknown", "generic_maxwell"} and _task_family_matches_requirement_text(intake.task_family, requirement):
            return intake

        simulation_spec = dict(intake.simulation_spec or {})
        execution_plan = dict(intake.execution_plan or {})
        simulation_spec["task_family"] = "generic_maxwell"
        execution_plan["task_family"] = "generic_maxwell"
        simulation_spec.setdefault("software", "ansys_maxwell")
        execution_plan.setdefault("software", "ansys_maxwell")
        simulation_spec["execution_ready"] = True
        execution_plan["execution_ready"] = True
        solver = simulation_spec.get("solver")
        if not isinstance(solver, dict):
            solver = {}
        solver.setdefault("design_type", execution_plan.get("design_type") or "Maxwell 2D")
        solver.setdefault("solution_type", execution_plan.get("solution_type") or "Magnetostatic")
        simulation_spec["solver"] = solver
        execution_plan.setdefault("design_type", solver.get("design_type") or "Maxwell 2D")
        execution_plan.setdefault("solution_type", solver.get("solution_type") or "Magnetostatic")
        intake.task_family = "generic_maxwell"
        intake.supported_now = True
        intake.support_message = "当前需求不完全匹配已有固定模板，已切换到通用二维 Maxwell IR 执行链路。"
        intake.summary = intake.summary or "已切换到通用二维 Maxwell IR 链路。"
        intake.simulation_spec = simulation_spec
        intake.execution_plan = execution_plan
        intake.warnings = _merge_unique_strings(
            intake.warnings,
            ["当前需求几何不完全匹配已有固定任务族，已切换到通用二维 Maxwell IR 链路。"],
        )
        return intake

    @staticmethod
    def _should_prefer_local_generation(
        requirement: str,
        intake: RequirementIntake,
        primitive_library: PrimitiveLibrary | None = None,
    ) -> bool:
        if intake.design:
            return True
        semantic_hint = infer_builder_hint(intake, requirement=requirement)
        if semantic_hint == "generic_maxwell":
            return _can_build_local_generic_2d_ir(intake, primitive_library=primitive_library)
        if semantic_hint not in {
            "capacitor_2d",
            "coaxial_capacitor_2d",
            "busbar_2d",
            "transformer_2d",
            "inductor_2d",
            "solenoid_2d",
        }:
            return False
        return True

    def generate_script(self, requirement: str, intake: RequirementIntake) -> GeneratedMaxwellScript:
        intake = self._coerce_intake_to_generic_ir_path(requirement, intake)
        primitive_library = getattr(self, "_primitive_library", None)
        if self._should_prefer_local_generation(requirement, intake, primitive_library=primitive_library):
            return self.build_local_fallback_script(intake)
        try:
            if not _extract_ir_payload_from_intake(intake):
                artifact = self.generate_ir_artifact(requirement, intake)
                _attach_ir_artifact_to_intake(intake, artifact)
            return self.build_local_fallback_script(intake)
        except UnknownPrimitiveError:
            raise
        except RequirementPlanningError:
            raise
        except Exception as ir_exc:
            try:
                fallback = self.build_local_fallback_script(intake)
                fallback.warnings.append(f"IR generation failed, returned to local deterministic path: {ir_exc}")
                return fallback
            except UnknownPrimitiveError:
                raise
            except RequirementPlanningError:
                raise RequirementPlanningError("AI IR generation failed and there is no usable local execution path.")

    def repair_script(
        self,
        requirement: str,
        intake: RequirementIntake,
        script: GeneratedMaxwellScript,
        failure_stage: str,
        error_details: str,
    ) -> GeneratedMaxwellScript:
        try:
            ir_payload = _extract_ir_payload_from_intake(intake)
            if ir_payload:
                previous_artifact = GeneratedIRPlan(
                    summary=intake.summary,
                    ir_plan=validate_ir_plan(MaxwellIRPlan.model_validate(ir_payload)),
                    assumptions=list(intake.assumptions),
                    warnings=list(intake.warnings),
                )
            else:
                previous_artifact = self.generate_ir_artifact(requirement, intake)
                _attach_ir_artifact_to_intake(intake, previous_artifact)
            repaired_artifact = self.repair_ir_artifact(
                requirement=requirement,
                intake=intake,
                previous_artifact=previous_artifact,
                failure_stage=failure_stage,
                error_details=error_details,
            )
            _attach_ir_artifact_to_intake(intake, repaired_artifact)
            return self.build_local_fallback_script(intake)
        except UnknownPrimitiveError:
            raise
        except RequirementPlanningError:
            raise
        except Exception as ir_exc:
            try:
                repaired = self.build_local_fallback_script(intake)
                repaired.warnings.append(f"IR repair failed, returned to local deterministic path: {ir_exc}")
                return repaired
            except UnknownPrimitiveError:
                raise
            except RequirementPlanningError:
                raise RequirementPlanningError("AI IR repair failed and there is no usable local execution path.")

    def revise_design_from_feedback(
        self,
        requirement: str,
        intake: RequirementIntake,
        outputs: dict[str, float | str] | None,
        evaluation: Any | None,
    ) -> ElectromagnetDesign:
        if not intake.design:
            raise RequirementPlanningError("Feedback revision currently requires an electromagnet design.")
        filtered_outputs = {
            key: value
            for key, value in (outputs or {}).items()
            if key
            in {
                "force_n",
                "flux_density_t",
                "max_flux_density_t",
                "estimated_coil_resistance_ohm",
                "estimated_current_at_supply_a",
            }
        }
        evaluation_payload = _compact_feedback_evaluation(evaluation)
        compact_checks = list(evaluation_payload.get("checks") or [])
        payload = {
            "requirement": requirement.strip(),
            "current_design": intake.design.model_dump(mode="json"),
            "patch_schema": {
                "summary": "string or null",
                "current_min_a": "number or null",
                "current_a": "number or null",
                "coil_turns": "integer or null",
                "core_width_mm": "number or null",
                "core_height_mm": "number or null",
                "core_thickness_mm": "number or null",
                "coil_width_mm": "number or null",
                "coil_height_mm": "number or null",
                "region_padding_mm": "number or null",
                "assumptions": ["string"],
                "warnings": ["string"],
            },
            "hard_constraints": {
                "supply_voltage_v": intake.design.supply_voltage_v,
                "current_min_a": intake.design.current_min_a,
                "current_limit_a": intake.design.current_limit_a,
                "air_gap_mm": intake.design.air_gap_mm,
            },
            "current_feedback": {
                "estimated_current_at_supply_a": filtered_outputs.get("estimated_current_at_supply_a"),
                "estimated_coil_resistance_ohm": filtered_outputs.get("estimated_coil_resistance_ohm"),
                "force_n": filtered_outputs.get("force_n"),
                "flux_density_t": filtered_outputs.get("flux_density_t")
                or filtered_outputs.get("max_flux_density_t"),
            },
            "unmet_checks": compact_checks,
            "previous_evaluation": evaluation_payload,
        }
        try:
            result = self._call_json_fallback(
                build_design_feedback_instructions(),
                payload,
                timeout_s=max(90, self._settings.codexa_timeout_s),
                max_attempts=2,
            )
            patch = _validate_design_patch_payload(result)
            return _apply_design_patch(intake.design, patch, requirement)
        except Exception as fallback_exc:
            raise RequirementPlanningError(f"LLM feedback revision failed. Fallback error: {fallback_exc}") from fallback_exc

    def revise_intake_from_feedback(
        self,
        requirement: str,
        intake: RequirementIntake,
        outputs: dict[str, float | str] | None,
        evaluation: Any | None,
        feedback_round: int,
    ) -> RequirementIntake:
        if intake.design is not None:
            revised_design = self.revise_design_from_feedback(
                requirement=requirement,
                intake=intake,
                outputs=outputs,
                evaluation=evaluation,
            )
            return self.replace_design_in_intake(
                requirement=requirement,
                intake=intake,
                revised_design=revised_design,
                feedback_round=feedback_round,
                outputs=outputs,
                evaluation=evaluation,
            )

        try:
            revised_artifact = self.revise_ir_artifact_from_feedback(
                requirement=requirement,
                intake=intake,
                outputs=outputs,
                evaluation=evaluation,
                feedback_round=feedback_round,
            )
            payload = intake.model_dump(mode="json")
            payload["supported_now"] = True
            payload["support_message"] = "已根据上一轮仿真反馈自动修订 Maxwell IR，并准备重新执行。"
            payload["summary"] = revised_artifact.summary or payload.get("summary") or "已根据仿真反馈修订 Maxwell IR。"
            revised_intake = _validate_intake_payload(payload, requirement)
            _attach_ir_artifact_to_intake(revised_intake, revised_artifact)
            extracted = dict(revised_intake.extracted_parameters or {})
            extracted["feedback_round"] = feedback_round
            extracted["last_ir_patch_summary"] = revised_artifact.summary
            if evaluation is not None and getattr(evaluation, "overall_status", None):
                extracted["previous_overall_status"] = evaluation.overall_status
            revised_intake.extracted_parameters = _normalize_json_like(extracted)
            return revised_intake
        except RequirementPlanningError:
            pass

        filtered_outputs = {
            str(key): value
            for key, value in (outputs or {}).items()
            if isinstance(value, (str, float, int, bool))
        }
        evaluation_payload = _compact_feedback_evaluation(evaluation)

        payload = {
            "requirement": requirement.strip(),
            "current_intake": {
                "task_family": intake.task_family,
                "supported_now": intake.supported_now,
                "summary": intake.summary,
                "extracted_parameters": intake.extracted_parameters,
                "simulation_spec": intake.simulation_spec,
                "execution_plan": intake.execution_plan,
                "assumptions": intake.assumptions,
                "warnings": intake.warnings,
            },
            "previous_outputs": filtered_outputs,
            "previous_evaluation": evaluation_payload,
            "feedback_round": feedback_round,
        }
        try:
            revised_payload = self._call_json_fallback(
                build_intake_feedback_instructions(),
                payload,
                timeout_s=max(20, min(self._settings.codexa_timeout_s, 45)),
                max_attempts=1,
            )
            revised_intake = _validate_intake_payload(revised_payload, requirement)
            return _rescue_supported_fallback_intake(requirement, revised_intake)
        except Exception as fallback_exc:
            try:
                revised_payload = self._call_json(
                    build_intake_feedback_instructions(),
                    payload,
                    RequirementIntake,
                    timeout_s=max(20, min(self._settings.codexa_timeout_s, 45)),
                    max_attempts=1,
                )
                revised_intake = _validate_intake_payload(revised_payload, requirement)
                return _rescue_supported_fallback_intake(requirement, revised_intake)
            except Exception as strict_exc:
                raise RequirementPlanningError(
                    f"Generic intake feedback revision failed. Fallback error: {fallback_exc}; strict-schema error: {strict_exc}"
                ) from strict_exc

    def replace_design_in_intake(
        self,
        requirement: str,
        intake: RequirementIntake,
        revised_design: ElectromagnetDesign,
        feedback_round: int,
        outputs: dict[str, float | str] | None = None,
        evaluation: Any | None = None,
    ) -> RequirementIntake:
        payload = intake.model_dump(mode="json")
        design_payload = revised_design.model_dump(mode="json")
        payload["task_family"] = "electromagnet_2d"
        payload["supported_now"] = True
        payload["support_message"] = "已根据上一轮仿真反馈自动修正设计，并准备重新执行。"
        payload["summary"] = revised_design.summary or payload.get("summary") or "已根据仿真反馈修正设计。"
        payload["design"] = design_payload
        payload["simulation_spec"] = _build_minimal_spec_from_design(design_payload)
        payload["execution_plan"] = _build_default_execution_plan_from_design(design_payload)

        extracted = dict(payload.get("extracted_parameters") or {})
        extracted.update(
            {
                "feedback_round": feedback_round,
                "last_ir_patch_summary": revised_design.summary or "已根据仿真反馈修正设计参数。",
                "air_gap_mm": revised_design.air_gap_mm,
                "current_a": revised_design.current_a,
                "current_min_a": revised_design.current_min_a,
                "current_limit_a": revised_design.current_limit_a,
                "supply_voltage_v": revised_design.supply_voltage_v,
                "coil_turns": revised_design.coil_turns,
                "coil_width_mm": revised_design.coil_width_mm,
                "coil_height_mm": revised_design.coil_height_mm,
            }
        )
        if outputs:
            estimated_current = _coerce_float(outputs.get("estimated_current_at_supply_a"))
            if estimated_current is not None:
                extracted["previous_estimated_current_at_supply_a"] = estimated_current
        if evaluation is not None and getattr(evaluation, "overall_status", None):
            extracted["previous_overall_status"] = evaluation.overall_status
        payload["extracted_parameters"] = _normalize_json_like(extracted)
        payload["assumptions"] = _merge_unique_strings(intake.assumptions, revised_design.assumptions)
        payload["warnings"] = _merge_unique_strings(intake.warnings, revised_design.warnings)
        return _validate_intake_payload(payload, requirement)

    def generate_design(self, requirement: str) -> ElectromagnetDesign:
        intake = self.refine_requirement_intake(requirement, self.generate_requirement_intake(requirement))
        if not intake.supported_now or not intake.design:
            raise UnsupportedRequirementError(intake.support_message, intake=intake)
        return intake.design

    def generate_design_json(self, requirement: str) -> str:
        intake = self.refine_requirement_intake(requirement, self.generate_requirement_intake(requirement))
        return json.dumps(intake.model_dump(mode="json"), ensure_ascii=False, indent=2)

    def build_local_fallback_script(self, intake: RequirementIntake) -> GeneratedMaxwellScript:
        if _extract_ir_payload_from_intake(intake):
            return _build_local_script_from_ir_payload(intake)
        if intake.design:
            return _build_local_script_from_design(intake.design)
        semantic_hint = infer_builder_hint(intake)
        if semantic_hint == "generic_maxwell":
            return _build_local_script_from_generic_intake(intake, primitive_library=self._primitive_library)
        if semantic_hint == "capacitor_2d":
            return _build_local_script_from_capacitor_intake(intake)
        if semantic_hint == "busbar_2d":
            return _build_local_script_from_busbar_intake(intake)
        if semantic_hint == "transformer_2d":
            return _build_local_script_from_transformer_intake(intake)
        if semantic_hint == "inductor_2d":
            return _build_local_script_from_inductor_intake(intake)
        if semantic_hint == "solenoid_2d":
            return _build_local_script_from_solenoid_intake(intake)
        if semantic_hint == "coaxial_capacitor_2d":
            return _build_local_script_from_coaxial_capacitor_intake(intake)
        raise RequirementPlanningError("当前任务没有可用的本地脚本回退。")
