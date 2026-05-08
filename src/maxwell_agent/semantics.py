from __future__ import annotations

import json
import re
from typing import Any

from .models import RequirementIntake


def _normalize_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _required_output_names(required_outputs: Any) -> list[str]:
    names: list[str] = []
    if isinstance(required_outputs, list):
        for item in required_outputs:
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("output_key") or "").strip().lower()
                if name:
                    names.append(name)
            elif isinstance(item, str):
                name = item.strip().lower()
                if name:
                    names.append(name)
    elif isinstance(required_outputs, str):
        name = required_outputs.strip().lower()
        if name:
            names.append(name)
    return names


def infer_physics_type(
    simulation_spec: dict[str, Any] | None,
    execution_plan: dict[str, Any] | None = None,
    requirement: str = "",
) -> str:
    spec = _as_mapping(simulation_spec)
    plan = _as_mapping(execution_plan)
    solver = _as_mapping(spec.get("solver"))
    geometry = _as_mapping(spec.get("geometry"))
    excitations_value = spec.get("excitations")
    excitations = _as_mapping_list(excitations_value)
    required_outputs = spec.get("required_outputs")

    explicit = _normalize_token(
        spec.get("physics_type")
        or plan.get("physics_type")
        or solver.get("physics_type")
        or solver.get("solution_type")
    )
    if "electrostatic" in explicit:
        return "electrostatic_2d"
    if "magnetostatic" in explicit:
        return "magnetostatic_2d"

    requested_blob = json.dumps(required_outputs or [], ensure_ascii=False).lower()
    solver_blob = json.dumps(solver, ensure_ascii=False).lower()
    excitation_blob = json.dumps(excitations_value, ensure_ascii=False).lower()
    geometry_blob = json.dumps(geometry, ensure_ascii=False).lower()
    requirement_blob = requirement.lower()
    combined = " ".join([requested_blob, solver_blob, excitation_blob, geometry_blob, requirement_blob])

    has_voltage = "voltage" in combined or "电压" in combined or any(
        _normalize_token(item.get("type")) == "voltage" for item in excitations
    )
    has_current = "current" in combined or "电流" in combined or any(
        _normalize_token(item.get("type")) == "current" for item in excitations
    )
    has_electric = "electro" in combined or "静电" in combined or "electric_field" in combined or "电场" in combined
    has_magnetic = "magnet" in combined or "静磁" in combined or "flux_density" in combined or "磁密" in combined or "磁场" in combined

    if (has_electric or has_voltage) and not (has_magnetic or has_current):
        return "electrostatic_2d"
    if (has_magnetic or has_current) and not (has_electric or has_voltage):
        return "magnetostatic_2d"
    if has_voltage and has_electric:
        return "electrostatic_2d"
    if has_current and has_magnetic:
        return "magnetostatic_2d"
    return "unknown"


def infer_output_semantics(simulation_spec: dict[str, Any] | None, requirement: str = "") -> dict[str, Any]:
    spec = _as_mapping(simulation_spec)
    names = _required_output_names(spec.get("required_outputs"))
    blob = " ".join(names + [json.dumps(spec.get("required_outputs") or [], ensure_ascii=False).lower(), requirement.lower()])
    return {
        "names": names,
        "wants_capacitance": "capacit" in blob or "电容" in blob,
        "wants_electric_field": "electric_field" in blob or "电场" in blob,
        "wants_flux_density": "flux_density" in blob or "bmax" in blob or "磁密" in blob or "磁场" in blob,
        "wants_current_density": "current_density" in blob or "电流密度" in blob,
        "wants_turns_ratio": "turns_ratio" in blob or "匝比" in blob,
        "wants_voltage": "secondary_voltage" in blob or "output_voltage" in blob or "voltage_result" in blob or "输出电压" in blob,
        "wants_inductance": "inductance" in blob or "电感" in blob,
        "wants_force": "force" in blob or "吸力" in blob or "推力" in blob,
    }


def infer_constraint_semantics(simulation_spec: dict[str, Any] | None, requirement: str = "") -> dict[str, Any]:
    spec = _as_mapping(simulation_spec)
    constraints = _as_mapping(spec.get("constraints"))
    blob = json.dumps(constraints, ensure_ascii=False).lower() + " " + requirement.lower()
    return {
        "has_target_capacitance": "target_capacitance_f" in constraints,
        "has_max_electric_field": "max_electric_field_v_per_m" in constraints,
        "has_max_flux_density": "max_flux_density_t" in constraints,
        "has_max_current_density": "max_current_density_a_per_mm2" in constraints or "j_limit_a_per_mm2" in constraints,
        "has_required_current": "required_current_a" in constraints,
        "has_current_range": "current_min_a" in constraints and "current_limit_a" in constraints,
        "has_current_limit": "current_limit_a" in constraints,
        "has_target_voltage": "output_voltage_v" in constraints or "secondary_target_voltage_v" in constraints,
        "has_target_inductance": "target_inductance_h" in constraints,
        "has_supply_voltage": "supply_voltage_v" in constraints or "供电电压" in blob,
        "has_force_target": "target_force_n" in constraints or "吸力" in blob or "force" in blob,
    }


def intake_has_generic_object_graph(intake: RequirementIntake) -> bool:
    geometry = _as_mapping(_as_mapping(intake.simulation_spec).get("geometry"))
    objects = geometry.get("objects") or geometry.get("cross_section_objects") or geometry.get("entities")
    return isinstance(objects, list) and bool(objects)


def infer_builder_hint(intake: RequirementIntake, requirement: str = "") -> str:
    if intake.design is not None:
        return "electromagnet_design"

    spec = _as_mapping(intake.simulation_spec)
    geometry = _as_mapping(spec.get("geometry"))
    excitations = _as_mapping(spec.get("excitations"))
    outputs = infer_output_semantics(spec, requirement=requirement)
    constraints = infer_constraint_semantics(spec, requirement=requirement)
    physics = infer_physics_type(spec, intake.execution_plan, requirement=requirement)
    geometry_type = _normalize_token(geometry.get("type"))

    if intake_has_generic_object_graph(intake):
        return "generic_maxwell"

    if geometry_type == "parallel_plate_capacitor_2d":
        return "capacitor_2d"
    if geometry_type == "coaxial_capacitor_2d":
        return "coaxial_capacitor_2d"
    if geometry_type == "rectangular_busbar_2d":
        return "busbar_2d"
    if geometry_type == "air_core_solenoid_2d":
        return "solenoid_2d"
    if geometry_type == "core_coil_inductor_2d":
        return "inductor_2d"
    if geometry_type == "core_winding_transformer_2d":
        return "transformer_2d"
    if geometry_type == "core_coil_armature_2d":
        return "electromagnet_2d"

    if physics == "electrostatic_2d":
        if any(key in geometry for key in ("inner_radius_mm", "outer_radius_mm")):
            return "coaxial_capacitor_2d"
        if any(key in geometry for key in ("plate_spacing_mm", "plate_width_mm")):
            return "capacitor_2d"

    if physics == "magnetostatic_2d":
        if constraints["has_target_voltage"] or outputs["wants_turns_ratio"] or outputs["wants_voltage"] or any(
            key in excitations for key in ("primary_voltage_v", "secondary_target_voltage_v", "primary_turns", "secondary_turns")
        ):
            return "transformer_2d"
        if constraints["has_target_inductance"]:
            return "inductor_2d"
        if any(key in geometry for key in ("length_mm", "radius_mm")) and (
            "coil_turns" in excitations or outputs["wants_flux_density"]
        ):
            return "solenoid_2d"
        if any(key in geometry for key in ("width_mm", "thickness_mm")) and (
            constraints["has_required_current"]
            or constraints["has_current_limit"]
            or constraints["has_max_current_density"]
            or outputs["wants_current_density"]
        ):
            return "busbar_2d"
        if any(key in geometry for key in ("air_gap_mm", "coil_width_mm", "coil_height_mm")) and (
            constraints["has_supply_voltage"] or constraints["has_force_target"] or outputs["wants_force"]
        ):
            return "electromagnet_2d"

    return "unknown"


def enrich_intake_semantics(
    simulation_spec: dict[str, Any] | None,
    execution_plan: dict[str, Any] | None,
    requirement: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = dict(_as_mapping(simulation_spec))
    plan = dict(_as_mapping(execution_plan))
    physics = infer_physics_type(spec, plan, requirement=requirement)
    output_semantics = infer_output_semantics(spec, requirement=requirement)
    constraint_semantics = infer_constraint_semantics(spec, requirement=requirement)
    if physics != "unknown":
        spec["physics_type"] = physics
        plan.setdefault("physics_type", physics)
    spec["output_semantics"] = output_semantics
    spec["constraint_semantics"] = constraint_semantics
    return spec, plan
