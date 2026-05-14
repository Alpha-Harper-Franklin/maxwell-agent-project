from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from maxwell_agent.capability_graph import capability_graph_for_intake
from maxwell_agent.config import Settings
from maxwell_agent.experience import ExperienceStore, FailureExperience
from maxwell_agent.llm_client import (
    _attach_ir_artifact_to_intake,
    _build_local_script_from_generic_intake,
    _fallback_busbar_2d_intake,
    _fallback_capacitor_intake,
    _fallback_coaxial_capacitor_2d_intake,
    _fallback_inductor_2d_intake,
    _fallback_intake_from_requirement,
    _fallback_solenoid_2d_intake,
    _fallback_transformer_2d_intake,
)
from maxwell_agent.maxwell_ir import GeneratedIRPlan, IRPatch, MaxwellIRPlan, apply_ir_patch, validate_ir_plan
from maxwell_agent.models import RequirementCheck, RequirementEvaluation, RequirementIntake
from maxwell_agent.residuals import analyze_requirement_residuals, compact_residual_payload
from maxwell_agent.script_validation import static_check_generated_script


BENCHMARK_CASES = [
    {
        "name": "electromagnet_2d",
        "requirement": "做一个24V直流电磁铁，气隙2mm，电流不超过2A，尽量提高吸力，外形不要太大。",
        "builder": _fallback_intake_from_requirement,
        "min_capabilities": ["physics.magnetostatic_2d"],
    },
    {
        "name": "capacitor_2d",
        "requirement": "做一个二维平行板电容器，板间距1mm，板宽20mm，施加100V，电容至少150pF，电场不超过400000V/m。",
        "builder": _fallback_capacitor_intake,
        "min_capabilities": ["physics.electrostatic_2d", "assignment.voltage"],
    },
    {
        "name": "coaxial_capacitor_2d",
        "requirement": "做一个同轴电容器，内半径1mm，外半径5mm，施加100V电压，电容至少30pF，电场不超过70000V/m。",
        "builder": _fallback_coaxial_capacitor_2d_intake,
        "min_capabilities": ["physics.electrostatic_2d"],
    },
    {
        "name": "busbar_2d",
        "requirement": "做一根载流铜排，宽10mm，厚2mm，电流200A，电流密度不超过5A/mm^2，评估最大磁密和电流密度。",
        "builder": _fallback_busbar_2d_intake,
        "min_capabilities": ["physics.magnetostatic_2d", "assignment.current"],
    },
    {
        "name": "solenoid_2d",
        "requirement": "做一个空心螺线管，长度50mm，半径8mm，匝数300，电流1A，求中心磁密。",
        "builder": _fallback_solenoid_2d_intake,
        "min_capabilities": ["physics.magnetostatic_2d", "assignment.current"],
    },
    {
        "name": "inductor_2d",
        "requirement": "做一个二维电感，气隙0.5mm，匝数600，电流2A，目标电感0.3H，输出电感和磁密。",
        "builder": _fallback_inductor_2d_intake,
        "min_capabilities": ["physics.magnetostatic_2d"],
    },
    {
        "name": "transformer_2d",
        "requirement": "做一个10000V到220V工频变压器，输出匝比、次级电压估算和最大磁密。",
        "builder": _fallback_transformer_2d_intake,
        "min_capabilities": ["physics.magnetostatic_2d"],
    },
    {
        "name": "generic_annular_conductor",
        "requirement": "做一个二维同心环导体截面，内半径3mm，外半径6mm，材料为铜，通以100A电流，评估最大磁密。",
        "builder": _fallback_intake_from_requirement,
        "min_capabilities": ["physics.magnetostatic_2d", "geometry.circle", "geometry.subtract", "assignment.current"],
    },
    {
        "name": "generic_round_pair_electrostatic",
        "requirement": "做一个二维双圆导体截面模型，两根圆铜导体半径1mm，中心距6mm，施加100V电压差，评估单位长度电容和最大电场。",
        "builder": _fallback_intake_from_requirement,
        "min_capabilities": ["physics.electrostatic_2d"],
    },
    {
        "name": "generic_dual_strip_electrostatic",
        "requirement": "做一个二维双矩形导体截面模型，两条铜导体各宽4mm厚1mm，中心距10mm，施加100V电压差，评估单位长度电容和最大电场。",
        "builder": _fallback_intake_from_requirement,
        "min_capabilities": ["physics.electrostatic_2d"],
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Maxwell Agent core benchmark.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "artifacts" / f"agent_core_benchmark_{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or output_dir / "summary.json"

    rows = [_run_case(case) for case in BENCHMARK_CASES]
    rows.append(_run_residual_patch_case(output_dir))
    rows.append(_run_experience_case(output_dir))

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "case_count": len(rows),
        "passed_count": sum(1 for item in rows if item["passed"]),
        "failed": [item for item in rows if not item["passed"]],
        "cases": rows,
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(output_path), "passed": summary["passed_count"], "total": len(rows)}, ensure_ascii=False))
    return 0 if summary["passed_count"] == len(rows) else 1


def _run_case(case: dict[str, object]) -> dict[str, object]:
    requirement = str(case["requirement"])
    intake = case["builder"](requirement)
    if not isinstance(intake, RequirementIntake):
        raise TypeError(f"{case['name']} did not produce RequirementIntake.")
    capability_items = capability_graph_for_intake(intake, requirement=requirement)
    capability_keys = {str(item["key"]) for item in capability_items}
    missing = [key for key in case["min_capabilities"] if key not in capability_keys]
    script_ok = _script_static_check_for_intake(intake)
    passed = not missing and script_ok
    return {
        "name": case["name"],
        "passed": passed,
        "task_family": intake.task_family,
        "capability_keys": sorted(capability_keys),
        "missing_capabilities": missing,
        "script_static_check": script_ok,
    }


def _script_static_check_for_intake(intake: RequirementIntake) -> bool:
    try:
        if intake.task_family == "generic_maxwell":
            script = _build_local_script_from_generic_intake(intake)
        else:
            artifact = GeneratedIRPlan(summary="benchmark", ir_plan=_plan_for_static_check(intake))
            _attach_ir_artifact_to_intake(intake, artifact)
            from maxwell_agent.llm_client import _build_local_script_from_ir_payload

            script = _build_local_script_from_ir_payload(intake)
        return static_check_generated_script(script).passed
    except Exception:
        return False


def _plan_for_static_check(intake: RequirementIntake) -> MaxwellIRPlan:
    physics = str((intake.simulation_spec or {}).get("physics_type") or "")
    electrostatic = "electrostatic" in physics
    if electrostatic:
        return MaxwellIRPlan(
            design_name="BenchmarkElectrostatic2D",
            solution_type="ElectrostaticXY",
            setup_type="Electrostatic",
            parameters=[
                {"name": "gap_mm", "source": "geometry", "field": "gap_mm", "default": 5.0},
                {"name": "voltage_v", "source": "excitations", "field": "voltage_v", "default": 100.0},
            ],
            objects=[
                {"name": "left", "kind": "rectangle", "material": "copper", "origin_exprs": ["-gap_mm/2-1", "-5", "0"], "sizes_exprs": ["1", "10"]},
                {"name": "right", "kind": "rectangle", "material": "copper", "origin_exprs": ["gap_mm/2", "-5", "0"], "sizes_exprs": ["1", "10"]},
                {"name": "region", "kind": "region", "pad_value_exprs": ["20", "20", "20", "20"]},
            ],
            assignments=[
                {"name": "signal", "kind": "voltage", "targets": ["left"], "amplitude_expr": "voltage_v"},
                {"name": "ground", "kind": "voltage", "targets": ["right"], "amplitude_expr": "0"},
                {"name": "outer", "kind": "balloon", "targets": ["region"], "boundary_name": "outer"},
            ],
            postprocess=[
                {"kind": "field_scalar", "output_key": "max_electric_field_v_per_m", "quantity": "Mag_E", "scalar_function": "Maximum"},
            ],
        )
    return MaxwellIRPlan(
        design_name="BenchmarkMagnetostatic2D",
        solution_type="Magnetostatic",
        setup_type="Magnetostatic",
        parameters=[
            {"name": "width_mm", "source": "geometry", "field": "width_mm", "default": 10.0},
            {"name": "current_a", "source": "excitations", "field": "current_a", "default": 10.0},
        ],
        objects=[
            {"name": "bar", "kind": "rectangle", "material": "copper", "origin_exprs": ["-width_mm/2", "-1", "0"], "sizes_exprs": ["width_mm", "2"]},
            {"name": "region", "kind": "region", "pad_value_exprs": ["20", "20", "20", "20"]},
        ],
        assignments=[
            {"name": "drive", "kind": "current", "targets": ["bar"], "amplitude_expr": "current_a"},
            {"name": "outer", "kind": "balloon", "targets": ["region"], "boundary_name": "outer"},
        ],
        postprocess=[
            {"kind": "field_scalar", "output_key": "max_flux_density_t", "quantity": "Mag_B", "scalar_function": "Maximum"},
        ],
    )


def _run_residual_patch_case(output_dir: Path) -> dict[str, object]:
    plan = validate_ir_plan(_plan_for_static_check(_fallback_busbar_2d_intake("做一根载流铜排，电流200A。")))
    evaluation = RequirementEvaluation(
        overall_status="failed",
        summary="电流密度超限。",
        checks=[RequirementCheck(name="电流密度上限", status="failed", detail="10 A/mm^2 > 5 A/mm^2")],
    )
    residuals = compact_residual_payload(analyze_requirement_residuals({"estimated_current_density_a_per_mm2": 10.0}, evaluation))
    patch = IRPatch(
        summary="扩大导体宽度以降低电流密度。",
        actions=[{"operation": "set_parameter_default", "target": "width_mm", "value": 20.0, "reason": "电流密度超限一倍，优先扩大截面。"}],
    )
    revised = apply_ir_patch(plan, patch)
    return {
        "name": "residual_to_ir_patch",
        "passed": bool(residuals) and any(item.name == "width_mm" and item.default == 20.0 for item in revised.parameters),
        "residuals": residuals,
    }


def _run_experience_case(output_dir: Path) -> dict[str, object]:
    settings = Settings(_env_file=None, PROJECT_ROOT=output_dir)
    store = ExperienceStore(settings.experience_store_path)
    store.append(
        FailureExperience(
            requirement="测试失败经验沉淀",
            run_directory=str(output_dir),
            stage="constraint_feedback",
            failed_checks=["10 A/mm^2 > 5 A/mm^2"],
            resolved=False,
        )
    )
    return {
        "name": "experience_store",
        "passed": len(store.recent(1)) == 1 and store.recent(1)[0].failed_checks == ["10 A/mm^2 > 5 A/mm^2"],
        "path": str(store.path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
