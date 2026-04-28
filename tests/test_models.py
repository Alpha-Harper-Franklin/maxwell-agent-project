from pathlib import Path

from maxwell_agent.config import Settings
from maxwell_agent.evaluation import build_requirement_evaluation
from maxwell_agent.llm_client import (
    _apply_design_patch,
    _build_local_script_from_capacitor_intake,
    _fallback_capacitor_intake,
    _fallback_intake_from_requirement,
    _normalize_design_payload,
    _normalize_intake_payload,
    _rescue_supported_fallback_intake,
    _validate_design_patch_payload,
)
from maxwell_agent.maxwell_env import _normalize_version_hint
from maxwell_agent.maxwell_executor import MaxwellExecutor
from maxwell_agent.models import ElectromagnetDesign, ElectromagnetDesignPatch, GeneratedMaxwellScript
from maxwell_agent.script_validation import static_check_generated_script


def test_design_variable_expressions() -> None:
    design = ElectromagnetDesign(
        summary="test",
        air_gap_mm=2.5,
        current_a=1.2,
        coil_turns=500,
    )

    expressions = design.variable_expressions()
    assert expressions["air_gap"] == "2.5mm"
    assert expressions["current_amp"] == "1.2A"
    assert expressions["coil_turns"] == "500"


def test_settings_normalize_blank_optional_values() -> None:
    settings = Settings(
        _env_file=None,
        PROJECT_ROOT=r"F:\maxwell_agent_project",
        CODEXA_API_KEY="   ",
        MAXWELL_VERSION="  ",
    )

    assert settings.codexa_api_key is None
    assert settings.maxwell_version is None


def test_settings_default_reasoning_effort_is_high() -> None:
    settings = Settings(
        _env_file=None,
        PROJECT_ROOT=r"F:\maxwell_agent_project",
    )

    assert settings.codexa_model == "gpt-5.4"
    assert settings.codexa_reasoning_effort == "high"


def test_normalize_version_hint_for_student_install() -> None:
    assert _normalize_version_hint("v252") == "2025.2"
    assert _normalize_version_hint("252") == "2025.2"
    assert _normalize_version_hint("2025.2") == "2025.2"


def test_resolve_env_suffix_for_student_install() -> None:
    suffix = MaxwellExecutor._resolve_env_suffix(
        "2025.2",
        Path(r"F:\AnsysEM_Student_2025R2\v252\AnsysEM\ansysedtsv.exe"),
    )
    assert suffix == "252"


def test_resolve_material_name_aliases() -> None:
    assert MaxwellExecutor._resolve_material_name("copper", fallback="copper") == "copper"
    assert MaxwellExecutor._resolve_material_name("soft_iron", fallback="steel_1008") == "soft_iron"


def test_requirement_evaluation_marks_unverified_goal() -> None:
    design = ElectromagnetDesign(
        source_requirement="做一个24V直流电磁铁，电流不超过2A，尽量提高吸力。",
        summary="demo",
        objective="maximize_force",
        current_a=2.0,
        current_limit_a=2.0,
        supply_voltage_v=24.0,
    )

    evaluation = build_requirement_evaluation(
        design,
        outputs={"max_flux_density_t": 0.01},
        run_status="completed",
    )

    assert evaluation.overall_status == "failed"
    assert any(item.status == "passed" for item in evaluation.checks)
    assert any(item.name == "电压/电流联合约束" and item.status == "failed" for item in evaluation.checks)


def test_normalize_design_payload_extracts_current_limit() -> None:
    payload = _normalize_design_payload(
        {"summary": "demo", "current_a": 2.0},
        "做一个24V直流电磁铁，气隙2mm，电流不超过2A，尽量提高吸力。",
    )

    assert payload["current_limit_a"] == 2.0


def test_normalize_design_payload_extracts_exact_current() -> None:
    payload = _normalize_design_payload(
        {"summary": "demo"},
        "做一个U型磁路线圈执行器，气隙1mm，电流1.5A，求吸力。",
    )

    assert payload["current_a"] == 1.5
    assert payload["air_gap_mm"] == 1.0


def test_normalize_design_payload_extracts_supply_voltage_in_chinese_sentence() -> None:
    payload = _normalize_design_payload(
        {"summary": "demo"},
        "做一个24V直流电磁铁，气隙2mm，电流不超过2A，尽量提高吸力。",
    )

    assert payload["supply_voltage_v"] == 24.0


def test_apply_design_patch_updates_selected_fields_only() -> None:
    design = ElectromagnetDesign(
        source_requirement="做一个24V直流电磁铁，气隙2mm，电流不超过2A，尽量提高吸力。",
        summary="demo",
        supply_voltage_v=24.0,
        current_limit_a=2.0,
        current_a=2.0,
        coil_turns=400,
        coil_width_mm=12.0,
        coil_height_mm=20.0,
        core_thickness_mm=10.0,
    )

    patch = ElectromagnetDesignPatch(
        summary="已根据反馈增加匝数。",
        coil_turns=1200,
        current_a=1.9,
        assumptions=["根据上一轮反馈提高等效电阻。"],
        warnings=["本轮只调整线圈参数。"],
    )
    revised = _apply_design_patch(design, patch, design.source_requirement)

    assert revised.coil_turns == 1200
    assert revised.current_a == 1.9
    assert revised.air_gap_mm == design.air_gap_mm
    assert any("等效电阻" in item for item in revised.assumptions)
    assert any("线圈参数" in item for item in revised.warnings)


def test_validate_design_patch_payload_accepts_partial_patch() -> None:
    patch = _validate_design_patch_payload(
        {
            "coil_turns": 1500,
            "current_a": 1.8,
            "assumptions": ["根据上一轮反馈提高匝数。"],
            "warnings": ["保持气隙不变。"],
        }
    )

    assert patch.coil_turns == 1500
    assert patch.current_a == 1.8


def test_normalize_intake_payload_keeps_transformer_structure() -> None:
    payload = _normalize_intake_payload(
        {
            "task_family": "transformer",
            "supported_now": False,
            "support_message": "暂不支持",
            "summary": "已识别为变压器需求。",
            "extracted_parameters": {"input_voltage_v": 10000, "output_voltage_v": 220},
            "simulation_spec": {
                "software": "ansys_maxwell",
                "task_family": "transformer",
                "execution_ready": False,
            },
            "assumptions": ["默认交流工况"],
            "warnings": ["当前暂无执行器"],
            "design": None,
        },
        "做一个10000v到220市电的变压器",
    )

    assert payload["task_family"] == "transformer"
    assert payload["design"] is None
    assert payload["extracted_parameters"]["input_voltage_v"] == 10000
    assert payload["simulation_spec"]["task_family"] == "transformer"


def test_normalize_intake_payload_keeps_generic_task_family_when_script_ready() -> None:
    payload = _normalize_intake_payload(
        {
            "task_family": "Electrostatic 2D",
            "supported_now": False,
            "support_message": "可进入脚本生成。",
            "summary": "已识别为二维静电场任务。",
            "extracted_parameters": {"voltage_v": 100, "gap_mm": 1},
            "simulation_spec": {
                "software": "ansys_maxwell",
                "task_family": "electrostatic_2d",
                "execution_ready": True,
            },
            "execution_plan": {
                "design_type": "Maxwell 2D",
                "solution_type": "Electrostatic",
                "steps": [{"action": "build_geometry"}],
                "execution_ready": True,
            },
            "assumptions": ["按二维静电场处理"],
            "warnings": [],
            "design": None,
        },
        "做一个二维平行板电容器，板间距1mm，电压100V，求电场强度。",
    )

    assert payload["task_family"] == "electrostatic_2d"
    assert payload["supported_now"] is True
    assert payload["design"] is None
    assert payload["execution_plan"]["solution_type"] == "Electrostatic"


def test_fallback_intake_for_transformer_requirement() -> None:
    intake = _fallback_intake_from_requirement("做一个10000v到220市电的变压器")

    assert intake.task_family == "transformer_2d"
    assert intake.supported_now is True
    assert intake.extracted_parameters["input_voltage_v"] == 10000
    assert intake.extracted_parameters["output_voltage_v"] == 220
    assert intake.simulation_spec["task_family"] == "transformer_2d"
    assert intake.simulation_spec["execution_ready"] is True


def test_local_capacitor_fallback_script_is_generated() -> None:
    intake = _fallback_intake_from_requirement("做一个二维平行板电容器，板间距1mm，板宽20mm，施加100V电压，求电场强度和电容。")
    intake.task_family = "capacitor_2d"
    intake.supported_now = True
    intake.simulation_spec = {
        "software": "ansys_maxwell",
        "task_family": "capacitor_2d",
        "geometry": {
            "plate_width_mm": 20,
            "plate_spacing_mm": 1,
            "air_region_margin_mm": 10,
        },
        "excitations": {"voltage_V": 100},
        "execution_ready": True,
    }
    intake.execution_plan = {
        "design_type": "Maxwell 2D",
        "solution_type": "Electrostatic",
        "execution_ready": True,
    }
    intake.design = None

    script = _build_local_script_from_capacitor_intake(intake)

    assert "Maxwell2d" in script.code
    assert 'solution_type="ElectrostaticXY"' in script.code
    assert "MatrixElectric" in script.code
    assert "assign_voltage" in script.code
    assert "assign_matrix" in script.code


def test_fallback_intake_for_capacitor_requirement() -> None:
    intake = _fallback_intake_from_requirement("做一个二维平行板电容器，板间距1mm，板宽20mm，施加100V电压，求电场强度和电容。")

    assert intake.task_family == "capacitor_2d"
    assert intake.supported_now is True
    assert intake.simulation_spec["execution_ready"] is True
    assert intake.execution_plan["execution_ready"] is True


def test_rescue_supported_fallback_intake_upgrades_unknown_capacitor() -> None:
    unknown = _fallback_intake_from_requirement("做一个通用 Maxwell 任务。")
    rescued = _rescue_supported_fallback_intake(
        "做一个二维平行板电容器，板间距1mm，板宽20mm，施加100V电压，求电场强度和电容。",
        unknown,
    )

    assert rescued.task_family == "capacitor_2d"
    assert rescued.supported_now is True


def test_static_check_accepts_simple_maxwell_script() -> None:
    script = GeneratedMaxwellScript(
        code="""
from ansys.aedt.core import Maxwell2d

def run_job(job: dict) -> dict:
    return {"project_name": "demo", "save_project_note": "save_project", "driver": "Maxwell2d"}
""".strip()
    )

    report = static_check_generated_script(script)

    assert report.passed is True


def test_static_check_accepts_simple_maxwell3d_script() -> None:
    script = GeneratedMaxwellScript(
        code="""
from ansys.aedt.core import Maxwell3d

def run_job(job: dict) -> dict:
    return {"project_name": "demo", "save_project_note": "save_project", "driver": "Maxwell3d"}
""".strip()
    )

    report = static_check_generated_script(script)

    assert report.passed is True


def test_static_check_rejects_dangerous_import() -> None:
    script = GeneratedMaxwellScript(
        code="""
import subprocess

def run_job(job: dict) -> dict:
    return {}
""".strip()
    )

    report = static_check_generated_script(script)

    assert report.passed is False
    assert any("不允许导入模块" in item for item in report.errors)
