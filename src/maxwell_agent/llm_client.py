from __future__ import annotations

import json
import re
import time
from textwrap import dedent
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from .config import Settings
from .errors import RequirementPlanningError, UnsupportedRequirementError
from .models import ElectromagnetDesign, ElectromagnetDesignPatch, GeneratedMaxwellScript, RequirementIntake
from .prompting import (
    build_design_feedback_instructions,
    build_intake_feedback_instructions,
    build_requirement_structuring_instructions,
    build_script_generation_instructions,
    build_script_repair_instructions,
    build_spec_refinement_instructions,
)


FLOAT_CORE = r"[0-9]+(?:\.[0-9]+)?"
FLOAT_PATTERN = rf"({FLOAT_CORE})"


def _extract_json_blob(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return text[start : end + 1]


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
        rf"(?:电流|current)\s*{FLOAT_PATTERN}\s*A\s*(?:-|~|～|到|至)\s*{FLOAT_PATTERN}\s*A?",
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


def _looks_like_capacitor_requirement(text: str) -> bool:
    lower = text.lower()
    return any(keyword in text for keyword in ("电容器", "平行板", "平板电容")) or "capacitor" in lower


def _looks_like_transformer_requirement(text: str) -> bool:
    lower = text.lower()
    return any(keyword in text for keyword in ("\u53d8\u538b\u5668", "鍙樺帇鍣?")) or "transformer" in lower


def _looks_like_inductor_requirement(text: str) -> bool:
    lower = text.lower()
    return any(
        keyword in text
        for keyword in (
            "\u7535\u611f",
            "\u7535\u6297\u5668",
            "\u7ebf\u5708\u7535\u611f",
        )
    ) or "inductor" in lower


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


def _fill_simple_constraints_from_requirement(payload: dict[str, Any], requirement: str) -> None:
    current_min, current_max = _extract_current_range(requirement)
    current_limit = _extract_first_float(
        requirement,
        [
            rf"(?:电流|current)[^0-9A-Za-z]{{0,16}}?(?:不超过|不大于|小于等于|上限|最大|max(?:imum)?)\s*{FLOAT_PATTERN}\s*A\b",
            rf"(?:<=|≤)\s*{FLOAT_PATTERN}\s*A\b",
        ],
    )
    exact_current = _extract_first_float(
        requirement,
        [
            rf"(?:电流|current)\s*(?:为|取值为|设为|设置为|=|约|大约|控制在)?\s*{FLOAT_PATTERN}\s*A\b",
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


def _fallback_capacitor_intake(requirement: str) -> RequirementIntake:
    text = requirement.strip()
    lower = text.lower()
    plate_spacing_mm = _extract_plate_spacing_mm(text) or 1.0
    plate_width_mm = _extract_plate_width_mm(text) or 20.0
    voltage_mentions = _extract_voltage_mentions(text)
    voltage_v = voltage_mentions[0] if voltage_mentions else 100.0

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
            "\u6b21\u7ea7\u7535\u538b\u9996\u7248\u6309\u533d\u6570\u6bd4\u4f30\u7b97\uff0c\u7528 Maxwell \u4e8c\u7ef4\u6a21\u578b\u9a8c\u8bc1\u78c1\u5bc6\u6c34\u5e73\u3002",
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
    turns = int(round(_extract_first_float(
        text,
        [
            rf"(?:\u530d\u6570|turns?)\s*(?:=|\u8bbe\u4e3a)?\s*{FLOAT_PATTERN}\b",
        ],
    ) or 600.0))
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


def _fallback_intake_from_requirement(requirement: str) -> RequirementIntake:
    text = requirement.strip()
    lower = text.lower()
    if "变压器" in text or "transformer" in lower:
        return _fallback_transformer_intake(text)
    if any(keyword in text for keyword in ("电磁铁", "线圈", "磁路", "执行器", "衔铁")):
        return _fallback_electromagnet_intake(text)
    return _fallback_unknown_intake(text)


def _fallback_intake_from_requirement(requirement: str) -> RequirementIntake:
    text = requirement.strip()
    lower = text.lower()
    if "变压器" in text or "transformer" in lower:
        return _fallback_transformer_intake(text)
    if _looks_like_capacitor_requirement(text):
        return _fallback_capacitor_intake(text)
    if any(keyword in text for keyword in ("电磁铁", "线圈", "磁路", "执行器", "衔铁")):
        return _fallback_electromagnet_intake(text)
    return _fallback_unknown_intake(text)


def _rescue_supported_fallback_intake(requirement: str, intake: RequirementIntake) -> RequirementIntake:
    text = requirement.strip()
    if intake.task_family == "capacitor_2d":
        if intake.supported_now and (
            bool(intake.simulation_spec.get("execution_ready")) or bool(intake.execution_plan.get("execution_ready"))
        ):
            return intake
        return _fallback_capacitor_intake(text)

    if intake.task_family in {"unknown", "generic_maxwell", "electrostatic_2d"} and _looks_like_capacitor_requirement(text):
        return _fallback_capacitor_intake(text)

    return intake


def _fallback_intake_from_requirement(requirement: str) -> RequirementIntake:
    text = requirement.strip()
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
    if intake.task_family == "capacitor_2d":
        if intake.supported_now and (
            bool(intake.simulation_spec.get("execution_ready")) or bool(intake.execution_plan.get("execution_ready"))
        ):
            return intake
        return _fallback_capacitor_intake(text)

    if intake.task_family == "transformer_2d":
        if intake.supported_now and (
            bool(intake.simulation_spec.get("execution_ready")) or bool(intake.execution_plan.get("execution_ready"))
        ):
            return intake
        return _fallback_transformer_2d_intake(text)

    if intake.task_family == "inductor_2d":
        if intake.supported_now and (
            bool(intake.simulation_spec.get("execution_ready")) or bool(intake.execution_plan.get("execution_ready"))
        ):
            return intake
        return _fallback_inductor_2d_intake(text)

    if intake.task_family in {"unknown", "generic_maxwell", "electrostatic_2d"} and _looks_like_capacitor_requirement(text):
        return _fallback_capacitor_intake(text)
    if intake.task_family in {"unknown", "generic_maxwell"} and _looks_like_transformer_requirement(text):
        return _fallback_transformer_2d_intake(text)
    if intake.task_family in {"unknown", "generic_maxwell"} and _looks_like_inductor_requirement(text):
        return _fallback_inductor_2d_intake(text)

    return intake


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
                app.analyze_setup("Setup1")
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


def _build_local_script_from_capacitor_intake(intake: RequirementIntake) -> GeneratedMaxwellScript:
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
                app.analyze_setup("Setup1")
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
                app.analyze_setup("Setup1")
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

            outputs: dict[str, float | str] = {{}}
            mu0 = 4 * math.pi * 1e-7
            effective_gap_m = max(air_gap, 0.1) * 1e-3
            core_area_m2 = (core_thickness * core_width) * 1e-6
            estimated_inductance_h = mu0 * coil_turns * coil_turns * core_area_m2 / max(effective_gap_m, 1e-6)
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
                app.analyze_setup("Setup1")
                try:
                    outputs["max_flux_density_t"] = float(app.post.get_scalar_field_value("Mag_B", "Maximum", object_name="AllObjects"))
                except Exception as exc:
                    outputs["max_flux_density_note"] = f"Postprocess skipped: {{exc}}"
                outputs["project_name"] = app.project_name
                outputs["design_name"] = app.design_name
                outputs["current_a"] = current_a
                outputs["coil_turns"] = coil_turns
                outputs["estimated_inductance_h"] = estimated_inductance_h
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
        if not settings.codexa_api_key:
            raise ValueError("CODEXA_API_KEY is not configured.")
        self._settings = settings
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
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                return self._client.responses.create(**kwargs)
            except Exception as exc:
                last_error = exc
                if attempt >= 3:
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
            input=json.dumps(input_payload, ensure_ascii=False) if not isinstance(input_payload, str) else input_payload,
            text={"format": schema},
            reasoning={"effort": reasoning_effort},
            store=False,
            timeout=self._settings.codexa_timeout_s,
        )
        return json.loads(response.output_text)

    def _call_json_fallback(self, instructions: str, input_payload: Any, effort: str | None = None) -> dict[str, Any]:
        reasoning_effort = effort or self._settings.codexa_reasoning_effort
        response = self._create_response_with_retry(
            model=self._settings.codexa_model,
            instructions=instructions,
            input=json.dumps(input_payload, ensure_ascii=False) if not isinstance(input_payload, str) else input_payload,
            reasoning={"effort": reasoning_effort},
            store=False,
            timeout=self._settings.codexa_timeout_s,
        )
        return json.loads(_extract_json_blob(response.output_text))

    def generate_requirement_intake(self, requirement: str) -> RequirementIntake:
        requirement = requirement.strip()
        if (
            _looks_like_capacitor_requirement(requirement)
            or _looks_like_transformer_requirement(requirement)
            or _looks_like_inductor_requirement(requirement)
            or _looks_like_electromagnet_requirement(requirement)
        ):
            return _fallback_intake_from_requirement(requirement)
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

    def generate_script(self, requirement: str, intake: RequirementIntake) -> GeneratedMaxwellScript:
        if intake.task_family == "electromagnet_2d" and intake.design:
            return self.build_local_fallback_script(intake)
        if intake.task_family in {"capacitor_2d", "transformer_2d", "inductor_2d"}:
            return self.build_local_fallback_script(intake)
        payload = {
            "requirement": requirement.strip(),
            "intake": intake.model_dump(mode="json"),
        }
        try:
            result = self._call_json(
                build_script_generation_instructions(),
                payload,
                GeneratedMaxwellScript,
            )
            return _validate_script_payload(result)
        except Exception:
            try:
                result = self._call_json_fallback(
                    build_script_generation_instructions(),
                    payload,
                )
                return _validate_script_payload(result)
            except Exception:
                try:
                    return self.build_local_fallback_script(intake)
                except RequirementPlanningError:
                    raise RequirementPlanningError("AI 脚本生成失败，且当前任务没有可用的本地脚本回退。")

    def repair_script(
        self,
        requirement: str,
        intake: RequirementIntake,
        script: GeneratedMaxwellScript,
        failure_stage: str,
        error_details: str,
    ) -> GeneratedMaxwellScript:
        payload = {
            "requirement": requirement.strip(),
            "intake": intake.model_dump(mode="json"),
            "previous_script": script.model_dump(mode="json"),
            "failure_stage": failure_stage,
            "error_details": error_details,
        }
        try:
            result = self._call_json(
                build_script_repair_instructions(),
                payload,
                GeneratedMaxwellScript,
            )
            return _validate_script_payload(result)
        except Exception:
            try:
                result = self._call_json_fallback(
                    build_script_repair_instructions(),
                    payload,
                )
                return _validate_script_payload(result)
            except Exception:
                try:
                    repaired = self.build_local_fallback_script(intake)
                    repaired.warnings.append("AI 脚本修复失败，已回退到本地确定性脚本。")
                    return repaired
                except RequirementPlanningError:
                    raise RequirementPlanningError("AI 脚本修复失败。")


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
        evaluation_payload = (
            evaluation.model_dump(mode="json") if hasattr(evaluation, "model_dump") else evaluation
        ) or {}
        compact_checks: list[dict[str, Any]] = []
        if isinstance(evaluation_payload, dict):
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
            evaluation_payload = {
                "overall_status": evaluation_payload.get("overall_status"),
                "summary": evaluation_payload.get("summary"),
                "checks": compact_checks,
            }
        payload = {
            "requirement": requirement.strip(),
            "current_design": intake.design.model_dump(mode="json"),
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
            )
            patch = _validate_design_patch_payload(result)
            return _apply_design_patch(intake.design, patch, requirement)
        except Exception as fallback_exc:
            try:
                result = self._call_json(
                    build_design_feedback_instructions(),
                    payload,
                    ElectromagnetDesignPatch,
                )
                patch = _validate_design_patch_payload(result)
                return _apply_design_patch(intake.design, patch, requirement)
            except Exception as strict_exc:
                raise RequirementPlanningError(
                    f"LLM feedback revision failed. Fallback error: {fallback_exc}; strict-schema error: {strict_exc}"
                ) from strict_exc

    def revise_intake_from_feedback(
        self,
        requirement: str,
        intake: RequirementIntake,
        outputs: dict[str, float | str] | None,
        evaluation: Any | None,
        feedback_round: int,
    ) -> RequirementIntake:
        if intake.task_family == "electromagnet_2d" and intake.design is not None:
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

        filtered_outputs = {
            str(key): value
            for key, value in (outputs or {}).items()
            if isinstance(value, (str, float, int, bool))
        }
        evaluation_payload = (
            evaluation.model_dump(mode="json") if hasattr(evaluation, "model_dump") else evaluation
        ) or {}
        if isinstance(evaluation_payload, dict):
            compact_checks = []
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
            evaluation_payload = {
                "overall_status": evaluation_payload.get("overall_status"),
                "summary": evaluation_payload.get("summary"),
                "checks": compact_checks,
            }

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
            )
            revised_intake = _validate_intake_payload(revised_payload, requirement)
            return _rescue_supported_fallback_intake(requirement, revised_intake)
        except Exception as fallback_exc:
            try:
                revised_payload = self._call_json(
                    build_intake_feedback_instructions(),
                    payload,
                    RequirementIntake,
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
        if intake.design:
            return _build_local_script_from_design(intake.design)
        if intake.task_family == "capacitor_2d":
            return _build_local_script_from_capacitor_intake(intake)
        if intake.task_family == "transformer_2d":
            return _build_local_script_from_transformer_intake(intake)
        if intake.task_family == "inductor_2d":
            return _build_local_script_from_inductor_intake(intake)
        raise RequirementPlanningError("当前任务没有可用的本地脚本回退。")
