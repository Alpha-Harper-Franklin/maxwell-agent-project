from pathlib import Path

from maxwell_agent.config import Settings
from maxwell_agent.evaluation import build_requirement_evaluation
from maxwell_agent.llm_client import (
    _apply_ir_parameter_defaults_to_intake,
    _apply_design_patch,
    _attach_ir_artifact_to_intake,
    _build_local_script_from_busbar_intake,
    _build_local_script_from_capacitor_intake,
    _build_local_script_from_coaxial_capacitor_intake,
    _build_local_script_from_generic_intake,
    _build_local_script_from_inductor_intake,
    _build_local_script_from_ir_payload,
    _build_local_script_from_solenoid_intake,
    _enforce_design_current_voltage_constraints,
    _estimate_design_supply_current,
    _fallback_busbar_2d_intake,
    _fallback_capacitor_intake,
    _fallback_coaxial_capacitor_2d_intake,
    _fallback_intake_from_requirement,
    _fallback_solenoid_2d_intake,
    _looks_like_busbar_requirement,
    _normalize_design_payload,
    _normalize_intake_payload,
    _rescue_supported_fallback_intake,
    _validate_ir_artifact_payload,
    _validate_design_patch_payload,
    CodexaLLMClient,
)
from maxwell_agent.maxwell_env import _normalize_version_hint
from maxwell_agent.pyaedt_compat import normalize_aedt_version_for_float, normalize_openai_base_url
from maxwell_agent.primitive_library import PrimitiveLibrary, PrimitiveTemplate
from maxwell_agent.maxwell_ir import GeneratedIRPlan, MaxwellIRPlan
from maxwell_agent.maxwell_executor import MaxwellExecutor
from maxwell_agent.models import ElectromagnetDesign, ElectromagnetDesignPatch, GeneratedMaxwellScript, RequirementIntake
from maxwell_agent.errors import UnknownPrimitiveError
from maxwell_agent.script_validation import static_check_generated_script
from maxwell_agent.agent import MaxwellAgent
from maxwell_agent.primitive_library import PrimitiveTemplateArtifact
from maxwell_agent.semantics import infer_builder_hint


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


def test_settings_project_root_alias_is_honored() -> None:
    settings = Settings(
        _env_file=None,
        PROJECT_ROOT=r"F:\semantic_route_test_root",
    )

    assert str(settings.project_root) == r"F:\semantic_route_test_root"


def test_normalize_version_hint_for_student_install() -> None:
    assert _normalize_version_hint("v252") == "2025.2"
    assert _normalize_version_hint("252") == "2025.2"
    assert _normalize_version_hint("2025.2") == "2025.2"


def test_settings_normalizes_openai_compatible_base_url() -> None:
    settings = Settings(_env_file=None, CODEXA_BASE_URL="https://example.com", CODEXA_API_KEY="x")

    assert settings.codexa_base_url == "https://example.com/v1"
    assert normalize_openai_base_url("https://example.com/v1") == "https://example.com/v1"


def test_student_version_suffix_is_safe_for_numeric_pyaedt_comparison() -> None:
    assert normalize_aedt_version_for_float("2025.2SV") == "2025.2"
    assert normalize_aedt_version_for_float("2025.2") == "2025.2"


def test_electromagnet_feedback_enforces_voltage_current_constraint() -> None:
    design = ElectromagnetDesign(
        source_requirement="做一个24V直流电磁铁，气隙2mm，电流不超过2A，尽量提高吸力，外形不要太大。",
        supply_voltage_v=24.0,
        current_limit_a=2.0,
        current_a=2.0,
        coil_turns=400,
        coil_width_mm=12.0,
        coil_height_mm=20.0,
        core_thickness_mm=10.0,
    )

    revised = _enforce_design_current_voltage_constraints(design)
    estimated_current = _estimate_design_supply_current(revised)

    assert estimated_current is not None
    assert estimated_current <= 2.0 + 1e-9
    assert revised.current_a <= 2.0


def test_normalize_intake_payload_enriches_physics_semantics() -> None:
    payload = _normalize_intake_payload(
        {
            "task_family": "unknown",
            "simulation_spec": {
                "geometry": {"type": "parallel_plate_capacitor_2d", "plate_spacing_mm": 1.0, "plate_width_mm": 20.0},
                "excitations": {"voltage_V": 100.0},
                "required_outputs": ["capacitance", "electric_field"],
            },
            "execution_plan": {"execution_ready": True},
        },
        "做一个二维平行板电容器，施加100V电压，求电容和电场。",
    )

    assert payload["simulation_spec"]["physics_type"] == "electrostatic_2d"
    assert payload["extracted_parameters"]["physics_type"] == "electrostatic_2d"
    assert payload["simulation_spec"]["output_semantics"]["wants_capacitance"] is True


def test_semantic_builder_hint_can_upgrade_unknown_task_family() -> None:
    intake = RequirementIntake(
        task_family="unknown",
        supported_now=True,
        simulation_spec={
            "geometry": {"type": "rectangular_busbar_2d", "width_mm": 10.0, "thickness_mm": 2.0},
            "excitations": {"current_a": 200.0},
            "constraints": {"max_current_density_a_per_mm2": 5.0},
            "required_outputs": ["flux_density", "current_density"],
            "solver": {"solution_type": "Magnetostatic"},
        },
        execution_plan={"execution_ready": True},
    )

    assert infer_builder_hint(intake, "做一根10mm乘2mm的载流导体，通200A电流，评估磁密和电流密度。") == "busbar_2d"
    assert CodexaLLMClient._should_prefer_local_generation("做一根10mm乘2mm的载流导体，通200A电流，评估磁密和电流密度。", intake) is True


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


def test_fallback_intake_for_busbar_requirement() -> None:
    intake = _fallback_intake_from_requirement(
        "\u505a\u4e00\u6839\u8f7d\u6d41\u94dc\u6392\uff0c\u5bbd10mm\uff0c\u539a2mm\uff0c\u7535\u6d41200A\uff0c\u8bc4\u4f30\u6700\u5927\u78c1\u5bc6\u548c\u7535\u6d41\u5bc6\u5ea6\u3002"
    )

    assert intake.task_family == "busbar_2d"
    assert intake.supported_now is True
    assert intake.simulation_spec["execution_ready"] is True
    assert intake.extracted_parameters["width_mm"] == 10.0
    assert intake.extracted_parameters["thickness_mm"] == 2.0
    assert intake.extracted_parameters["current_a"] == 200.0


def test_primitive_library_register_and_find(tmp_path: Path) -> None:
    library = PrimitiveLibrary(tmp_path / "primitive_library.json")
    template = PrimitiveTemplate(
        primitive_key="double_circle",
        display_name="双圆导体",
        aliases=["two_circles", "双圆"],
        parameters=[],
        objects=[
            {
                "role_name": "left_circle",
                "kind": "circle",
                "material_mode": "instance",
                "origin_exprs": ["0", "0", "0"],
                "radius_expr": "1",
            }
        ],
        operations=[],
        result_role_name="left_circle",
        source="learned_from_llm",
    )
    library.register(template, persist=False, mark_persisted=False)

    assert library.find("double_circle") is not None
    assert library.find("双圆") is not None
    assert library.is_pending("double_circle") is True


def test_generic_builder_can_use_learned_rectangular_frame(tmp_path: Path) -> None:
    library = PrimitiveLibrary(tmp_path / "primitive_library.json")
    intake = RequirementIntake(
        task_family="generic_maxwell",
        supported_now=True,
        summary="测试矩形框原语",
        simulation_spec={
            "geometry": {
                "objects": [
                    {
                        "name": "frame1",
                        "kind": "rectangular_frame",
                        "center_mm": [0, 0],
                        "outer_width_mm": 20,
                        "outer_height_mm": 12,
                        "inner_width_mm": 10,
                        "inner_height_mm": 4,
                    }
                ]
            },
            "materials": {"frame1": "copper"},
            "excitations": [{"object": "frame1", "type": "current", "value": 10}],
            "required_outputs": ["max_flux_density", "current_density"],
            "solver": {"type": "magnetostatic"},
        },
        execution_plan={"execution_ready": True},
    )

    script = _build_local_script_from_generic_intake(intake, primitive_library=library)

    assert "frame1_outer_frame" in script.code
    assert "frame1_inner_window" in script.code
    assert "subtract" in script.code
    assert script.primitive_library_updates == []


def test_unknown_primitive_raises_learning_signal(tmp_path: Path) -> None:
    library = PrimitiveLibrary(tmp_path / "primitive_library.json")
    intake = RequirementIntake(
        task_family="generic_maxwell",
        supported_now=True,
        summary="测试未知原语",
        simulation_spec={
            "geometry": {
                "objects": [
                    {
                        "name": "shape1",
                        "kind": "super_frame",
                        "center_mm": [0, 0],
                    }
                ]
            },
            "materials": {"shape1": "copper"},
            "excitations": [{"object": "shape1", "type": "current", "value": 10.0}],
            "required_outputs": ["max_flux_density"],
            "solver": {"type": "magnetostatic"},
        },
        execution_plan={"execution_ready": True},
    )

    try:
        _build_local_script_from_generic_intake(intake, primitive_library=library)
    except UnknownPrimitiveError as exc:
        assert exc.primitive_token == "super_frame"
        assert exc.raw_object["name"] == "shape1"
    else:
        raise AssertionError("Expected UnknownPrimitiveError for unsupported primitive.")


def test_generate_script_bubbles_unknown_primitive_to_learning_path(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, PROJECT_ROOT=str(tmp_path))
    client = object.__new__(CodexaLLMClient)
    client._settings = settings
    client._primitive_library = PrimitiveLibrary(tmp_path / "primitive_library.json")
    client._coerce_intake_to_generic_ir_path = lambda requirement, working_intake: working_intake
    client.build_local_fallback_script = lambda working_intake: _build_local_script_from_generic_intake(
        working_intake,
        primitive_library=client._primitive_library,
    )
    intake = RequirementIntake(
        task_family="generic_maxwell",
        supported_now=True,
        summary="unknown primitive bubble",
        simulation_spec={
            "geometry": {"objects": [{"name": "shape1", "kind": "super_frame", "center_mm": [0, 0]}]},
            "materials": {"shape1": "copper"},
            "excitations": [{"object": "shape1", "type": "current", "value": 10.0}],
            "required_outputs": ["max_flux_density"],
            "solver": {"type": "magnetostatic"},
        },
        execution_plan={"execution_ready": True},
    )

    try:
        client.generate_script("做一个通用框形导体", intake)
    except UnknownPrimitiveError as exc:
        assert exc.primitive_token == "super_frame"
    else:
        raise AssertionError("Expected UnknownPrimitiveError to bubble from generate_script.")


def test_agent_learns_unknown_primitive_and_persists_generic_template(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, PROJECT_ROOT=str(tmp_path))
    agent = object.__new__(MaxwellAgent)
    agent._settings = settings
    agent._llm = object.__new__(CodexaLLMClient)
    agent._llm._settings = settings
    agent._llm._primitive_library = PrimitiveLibrary(tmp_path / "primitive_library.json")
    intake = RequirementIntake(
        task_family="generic_maxwell",
        supported_now=True,
        summary="learn primitive persistence",
        simulation_spec={
            "geometry": {
                "objects": [
                    {
                        "name": "shape1",
                        "kind": "super_frame",
                        "center_mm": [0, 0],
                        "outer_width_mm": 20,
                        "outer_height_mm": 12,
                        "inner_width_mm": 10,
                        "inner_height_mm": 4,
                    }
                ]
            },
            "materials": {"shape1": "copper"},
            "excitations": [{"object": "shape1", "type": "current", "value": 10.0}],
            "required_outputs": ["max_flux_density", "current_density"],
            "solver": {"type": "magnetostatic"},
        },
        execution_plan={"execution_ready": True},
    )

    artifact = PrimitiveTemplateArtifact(
        summary="learned super frame",
        template=PrimitiveTemplate(
            primitive_key="super_frame",
            display_name="通用矩形框截面",
            aliases=["generic_super_frame", "rectangular_hollow_frame"],
            summary="使用外矩形减内矩形生成可复用的二维框形截面。",
            parameters=[
                {"name": "center_x_mm", "aliases": ["center_x_mm", "x_mm"], "cast": "float", "default": 0.0},
                {"name": "center_y_mm", "aliases": ["center_y_mm", "y_mm"], "cast": "float", "default": 0.0},
                {"name": "outer_width_mm", "aliases": ["outer_width_mm"], "cast": "float", "required": True},
                {"name": "outer_height_mm", "aliases": ["outer_height_mm"], "cast": "float", "required": True},
                {"name": "inner_width_mm", "aliases": ["inner_width_mm"], "cast": "float", "required": True},
                {"name": "inner_height_mm", "aliases": ["inner_height_mm"], "cast": "float", "required": True},
            ],
            objects=[
                {
                    "role_name": "outer_frame",
                    "kind": "rectangle",
                    "material_mode": "instance",
                    "origin_exprs": ["center_x_mm - outer_width_mm / 2", "center_y_mm - outer_height_mm / 2", "0"],
                    "sizes_exprs": ["outer_width_mm", "outer_height_mm"],
                },
                {
                    "role_name": "inner_window",
                    "kind": "rectangle",
                    "material_mode": "fixed",
                    "material_value": "vacuum",
                    "origin_exprs": ["center_x_mm - inner_width_mm / 2", "center_y_mm - inner_height_mm / 2", "0"],
                    "sizes_exprs": ["inner_width_mm", "inner_height_mm"],
                },
            ],
            operations=[{"kind": "subtract", "blank_roles": ["outer_frame"], "tool_roles": ["inner_window"]}],
            result_role_name="outer_frame",
            result_area_expr="outer_width_mm * outer_height_mm - inner_width_mm * inner_height_mm",
        ),
        assumptions=[],
        warnings=[],
    )

    agent._llm.learn_primitive_template = lambda **kwargs: artifact

    learned = agent._learn_primitive_with_repair(
        requirement="做一个框形导体截面",
        intake=intake,
        primitive_token="super_frame",
        raw_object={"name": "shape1", "kind": "super_frame"},
        error_details="unknown primitive",
    )

    assert learned.template.primitive_key == "super_frame"
    script = agent._llm.build_local_fallback_script(intake)
    assert "shape1_outer_frame" in script.code
    assert "shape1_inner_window" in script.code

    agent._llm.primitive_library.commit([learned.template])
    library_path = settings.primitive_library_path
    assert library_path.exists() is True
    payload = library_path.read_text(encoding="utf-8")
    assert "super_frame" in payload
    assert '"outer_width_mm"' in payload


def test_fallback_intake_extracts_busbar_constraints() -> None:
    intake = _fallback_busbar_2d_intake(
        "\u505a\u4e00\u6839\u8f7d\u6d41\u94dc\u6392\uff0c\u5bbd10mm\uff0c\u539a2mm\uff0c\u7535\u6d41200A\uff0c\u7535\u6d41\u5bc6\u5ea6\u4e0d\u8d85\u8fc75A/mm^2\uff0c\u78c1\u5bc6\u4e0d\u8d85\u8fc70.02T\u3002"
    )

    assert intake.simulation_spec["constraints"]["required_current_a"] == 200.0
    assert intake.simulation_spec["constraints"]["max_current_density_a_per_mm2"] == 5.0
    assert intake.simulation_spec["constraints"]["max_flux_density_t"] == 0.02
    assert "current_limit_a" not in intake.simulation_spec["constraints"]


def test_fallback_intake_extracts_capacitor_constraints() -> None:
    intake = _fallback_capacitor_intake(
        "\u505a\u4e00\u4e2a\u4e8c\u7ef4\u5e73\u884c\u677f\u7535\u5bb9\u5668\uff0c\u677f\u95f4\u8ddd1mm\uff0c\u677f\u5bbd20mm\uff0c\u65bd\u52a0100V\uff0c\u7535\u5bb9\u81f3\u5c11 1pF\uff0c\u7535\u573a\u4e0d\u8d85\u8fc7 200kV/m\u3002"
    )

    assert intake.simulation_spec["constraints"]["target_capacitance_f"] == 1e-12
    assert intake.simulation_spec["constraints"]["max_electric_field_v_per_m"] == 200000.0


def test_fallback_intake_extracts_coaxial_capacitor_constraints() -> None:
    intake = _fallback_coaxial_capacitor_2d_intake(
        "\u505a\u4e00\u4e2a\u540c\u8f74\u7535\u5bb9\u5668\uff0c\u5185\u534a\u5f841mm\uff0c\u5916\u534a\u5f845mm\uff0c\u65bd\u52a0100V\u7535\u538b\uff0c\u7535\u5bb9\u81f3\u5c1130pF\uff0c\u7535\u573a\u4e0d\u8d85\u8fc770000V/m\u3002"
    )

    assert intake.simulation_spec["constraints"]["target_capacitance_f"] == 30e-12
    assert intake.simulation_spec["constraints"]["max_electric_field_v_per_m"] == 70000.0


def test_busbar_heuristic_does_not_capture_generic_annular_conductor() -> None:
    assert _looks_like_busbar_requirement(
        "\u505a\u4e00\u4e2a\u4e8c\u7ef4\u540c\u5fc3\u73af\u5bfc\u4f53\u622a\u9762\uff0c\u5185\u534a\u5f843mm\uff0c\u5916\u534a\u5f846mm\uff0c\u901a\u4ee5100A\u7535\u6d41\uff0c\u8bc4\u4f30\u6700\u5927\u78c1\u5bc6\u3002"
    ) is False


def test_local_busbar_fallback_script_is_generated_and_static_checked() -> None:
    intake = _fallback_busbar_2d_intake(
        "\u505a\u4e00\u6839\u8f7d\u6d41\u94dc\u6392\uff0c\u5bbd10mm\uff0c\u539a2mm\uff0c\u7535\u6d41200A\uff0c\u8bc4\u4f30\u6700\u5927\u78c1\u5bc6\u548c\u7535\u6d41\u5bc6\u5ea6\u3002"
    )
    script = _build_local_script_from_busbar_intake(intake)

    assert "Busbar2D" in script.code
    assert "estimated_current_density_a_per_mm2" in script.code
    assert "assign_current" in script.code
    assert static_check_generated_script(script).passed is True


def test_fallback_intake_for_solenoid_requirement() -> None:
    intake = _fallback_intake_from_requirement(
        "\u505a\u4e00\u4e2a\u7a7a\u5fc3\u87ba\u7ebf\u7ba1\uff0c\u957f\u5ea650mm\uff0c\u534a\u5f848mm\uff0c\u5305\u6570300\uff0c\u7535\u6d411A\uff0c\u6c42\u4e2d\u5fc3\u78c1\u5bc6\u3002"
    )

    assert intake.task_family == "solenoid_2d"
    assert intake.supported_now is True
    assert intake.simulation_spec["execution_ready"] is True
    assert intake.extracted_parameters["current_a"] == 1.0
    assert intake.extracted_parameters["coil_turns"] == 300


def test_local_solenoid_fallback_script_is_generated_and_static_checked() -> None:
    intake = _fallback_solenoid_2d_intake(
        "\u505a\u4e00\u4e2a\u7a7a\u5fc3\u87ba\u7ebf\u7ba1\uff0c\u957f\u5ea650mm\uff0c\u534a\u5f848mm\uff0c\u5305\u6570300\uff0c\u7535\u6d411A\uff0c\u6c42\u4e2d\u5fc3\u78c1\u5bc6\u3002"
    )
    script = _build_local_script_from_solenoid_intake(intake)

    assert "Maxwell2d" in script.code
    assert "estimated_center_flux_density_t" in script.code
    assert "assign_current" in script.code
    assert static_check_generated_script(script).passed is True


def test_fallback_intake_for_coaxial_capacitor_requirement() -> None:
    intake = _fallback_intake_from_requirement(
        "\u505a\u4e00\u4e2a\u540c\u8f74\u7535\u5bb9\u5668\uff0c\u5185\u534a\u5f841mm\uff0c\u5916\u534a\u5f845mm\uff0c\u65bd\u52a0100V\u7535\u538b\uff0c\u6c42\u7535\u5bb9\u548c\u7535\u573a\u3002"
    )

    assert intake.task_family == "coaxial_capacitor_2d"
    assert intake.supported_now is True
    assert intake.simulation_spec["execution_ready"] is True
    assert intake.extracted_parameters["inner_radius_mm"] == 1.0
    assert intake.extracted_parameters["outer_radius_mm"] == 5.0


def test_local_coaxial_capacitor_script_is_generated_and_static_checked() -> None:
    intake = _fallback_coaxial_capacitor_2d_intake(
        "\u505a\u4e00\u4e2a\u540c\u8f74\u7535\u5bb9\u5668\uff0c\u5185\u534a\u5f841mm\uff0c\u5916\u534a\u5f845mm\uff0c\u65bd\u52a0100V\u7535\u538b\uff0c\u6c42\u7535\u5bb9\u548c\u7535\u573a\u3002"
    )
    script = _build_local_script_from_coaxial_capacitor_intake(intake)

    assert "CoaxialCapacitor2D" in script.code
    assert "reference_capacitance_f_per_m" in script.code
    assert "assign_voltage" in script.code
    assert static_check_generated_script(script).passed is True


def test_generic_ir_payload_script_is_rendered_and_static_checked() -> None:
    intake = RequirementIntake(
        task_family="generic_maxwell",
        supported_now=True,
        summary="使用通用 IR 生成脚本。",
        simulation_spec={
            "ir_plan": {
                "design_name": "GenericIRDemo",
                "solution_type": "Magnetostatic",
                "setup_type": "Magnetostatic",
                "parameters": [
                    {"name": "width_mm", "source": "geometry", "field": "width_mm", "default": 10.0},
                    {"name": "height_mm", "source": "geometry", "field": "height_mm", "default": 4.0},
                    {"name": "current_a", "source": "excitations", "field": "current_a", "default": 12.0},
                ],
                "objects": [
                    {
                        "name": "bar",
                        "kind": "rectangle",
                        "material": "copper",
                        "origin_exprs": ["-width_mm / 2", "-height_mm / 2", "0"],
                        "sizes_exprs": ["width_mm", "height_mm"],
                    },
                    {
                        "name": "Region",
                        "kind": "region",
                        "pad_value_exprs": ["20", "20", "20", "20"],
                        "pad_type": "Absolute Offset",
                    },
                ],
                "assignments": [
                    {"name": "Drive", "kind": "current", "targets": ["bar"], "amplitude_expr": "current_a"},
                    {"name": "OuterRegion", "kind": "balloon", "targets": ["Region"], "boundary_name": "OuterRegion"},
                ],
                "derived_outputs": [
                    {"output_key": "current_a", "expression": "current_a"},
                ],
                "postprocess": [
                    {
                        "kind": "field_scalar",
                        "output_key": "max_flux_density_t",
                        "cast": "float",
                        "quantity": "Mag_B",
                        "scalar_function": "Maximum",
                        "object_name": "AllObjects",
                    }
                ],
            }
        },
    )

    script = _build_local_script_from_ir_payload(intake)

    assert "GenericIRDemo" in script.code
    assert "assign_current" in script.code
    assert static_check_generated_script(script).passed is True


def test_apply_ir_parameter_defaults_handles_list_sections() -> None:
    intake = RequirementIntake(
        task_family="generic_maxwell",
        supported_now=True,
        simulation_spec={
            "geometry": {"shape": "custom"},
            "excitations": [{"object": "cond_1", "type": "voltage"}],
            "boundaries": [{"type": "open"}],
            "constraints": ["电流不超过 2A"],
        },
        execution_plan={},
        extracted_parameters={},
    )
    plan = MaxwellIRPlan.model_validate(
        {
            "design_name": "ListSectionDemo",
            "solution_type": "Electrostatic",
            "setup_type": "Electrostatic",
            "parameters": [
                {"name": "radius_mm", "source": "geometry", "field": "radius_mm", "default": 1.0},
                {"name": "drive_v", "source": "excitations", "field": "drive_v", "default": 100.0},
                {"name": "air_pad_mm", "source": "boundaries", "field": "air_pad_mm", "default": 40.0},
                {"name": "current_limit_a", "source": "constraints", "field": "current_limit_a", "default": 2.0},
            ],
        }
    )

    updated = _apply_ir_parameter_defaults_to_intake(intake, plan)

    assert updated.simulation_spec["geometry"]["radius_mm"] == 1.0
    assert updated.simulation_spec["excitations"]["drive_v"] == 100.0
    assert updated.simulation_spec["boundaries"]["air_pad_mm"] == 40.0
    assert updated.simulation_spec["constraints"]["current_limit_a"] == 2.0
    assert updated.simulation_spec["excitations_notes"] == [{"object": "cond_1", "type": "voltage"}]
    assert updated.simulation_spec["boundaries_notes"] == [{"type": "open"}]
    assert updated.simulation_spec["constraints_notes"] == ["电流不超过 2A"]


def test_validate_ir_artifact_payload_accepts_valid_generic_ir() -> None:
    artifact = _validate_ir_artifact_payload(
        {
            "summary": "通用 IR 方案",
            "ir_plan": {
                "design_name": "ValidatedIRDemo",
                "solution_type": "Magnetostatic",
                "setup_type": "Magnetostatic",
                "parameters": [
                    {"name": "width_mm", "source": "geometry", "field": "width_mm", "default": 10.0},
                ],
                "objects": [
                    {
                        "name": "bar",
                        "kind": "rectangle",
                        "material": "copper",
                        "origin_exprs": ["-width_mm / 2", "-1", "0"],
                        "sizes_exprs": ["width_mm", "2"],
                    }
                ],
            },
            "assumptions": ["按二维截面处理"],
            "warnings": ["首版仅验证磁场"],
        }
    )

    assert artifact.ir_plan.design_name == "ValidatedIRDemo"
    assert artifact.assumptions == ["按二维截面处理"]


def test_validate_ir_artifact_payload_adapts_loose_llm_style_ir() -> None:
    artifact = _validate_ir_artifact_payload(
        {
            "summary": "松格式 IR",
            "ir_plan": {
                "driver": "Maxwell 3D",
                "design_name": "LooseStyleDemo",
                "solution_type": "静磁场",
                "parameters": [
                    {"name": "W", "value": "20", "unit": "mm", "category": "geometry"},
                    {"name": "I", "value": "50", "unit": "A", "category": "excitation"},
                ],
                "locals": [{"name": "HalfW", "expression": "W/2"}],
                "objects": [
                    {
                        "type": "rectangle",
                        "name": "bar",
                        "material": "copper",
                        "corner": ["-HalfW", "-1"],
                        "size": ["W", "2"],
                    },
                    {
                        "type": "region",
                        "name": "air_region",
                        "padding": {"left": "20", "right": "20", "bottom": "20", "top": "20"},
                    },
                ],
                "assignments": [
                    {"type": "current", "name": "drive", "target": "bar", "magnitude": "I"},
                    {"type": "balloon", "name": "outer_region", "target": "air_region"},
                ],
                "derived_outputs": [
                    {"name": "analytic_current", "method": "analytic", "expression": "I"},
                    {"name": "bmax", "method": "field_scalar", "field": "Mag_B", "stat": "max", "target": "AllObjects"},
                ],
            },
        }
    )

    assert artifact.ir_plan.driver == "Maxwell2d"
    assert artifact.ir_plan.solution_type == "Magnetostatic"
    assert artifact.ir_plan.parameters[0].source == "geometry"
    assert artifact.ir_plan.postprocess[0].output_key == "bmax"


def test_generate_script_prefers_llm_ir_path_and_attaches_ir() -> None:
    client = object.__new__(CodexaLLMClient)
    intake = RequirementIntake(
        task_family="generic_maxwell",
        supported_now=True,
        summary="测试 IR 主链。",
        simulation_spec={"geometry": {"width_mm": 8.0}, "excitations": {"current_a": 20.0}},
        execution_plan={"execution_ready": True},
    )

    artifact = GeneratedIRPlan(
        summary="LLM 生成的 IR",
        ir_plan=MaxwellIRPlan(
            design_name="LLMIRDemo",
            solution_type="Magnetostatic",
            setup_type="Magnetostatic",
            parameters=[
                {"name": "width_mm", "source": "geometry", "field": "width_mm", "default": 12.0},
                {"name": "current_a", "source": "excitations", "field": "current_a", "default": 30.0},
            ],
            objects=[
                {
                    "name": "bar",
                    "kind": "rectangle",
                    "material": "copper",
                    "origin_exprs": ["-width_mm / 2", "-1", "0"],
                    "sizes_exprs": ["width_mm", "2"],
                },
                {
                    "name": "Region",
                    "kind": "region",
                    "pad_value_exprs": ["20", "20", "20", "20"],
                },
            ],
            assignments=[
                {"name": "Drive", "kind": "current", "targets": ["bar"], "amplitude_expr": "current_a"},
                {"name": "OuterRegion", "kind": "balloon", "targets": ["Region"], "boundary_name": "OuterRegion"},
            ],
        ),
        assumptions=["由 LLM 直接给出 IR"],
        warnings=[],
    )

    client.generate_ir_artifact = lambda requirement, working_intake: artifact
    client.build_local_fallback_script = lambda working_intake: _build_local_script_from_ir_payload(working_intake)
    client._legacy_generate_script = lambda requirement, working_intake: (_ for _ in ()).throw(RuntimeError("unused"))

    script = CodexaLLMClient.generate_script(client, "做一个通用载流导体模型", intake)

    assert "LLMIRDemo" in script.code
    assert intake.simulation_spec["ir_plan"]["design_name"] == "LLMIRDemo"
    assert intake.simulation_spec["geometry"]["width_mm"] == 12.0
    assert intake.simulation_spec["excitations"]["current_a"] == 30.0
    assert static_check_generated_script(script).passed is True


def test_repair_script_prefers_ir_repair_path() -> None:
    client = object.__new__(CodexaLLMClient)
    intake = RequirementIntake(
        task_family="generic_maxwell",
        supported_now=True,
        summary="测试 IR 修复链。",
        simulation_spec={},
        execution_plan={"execution_ready": True},
    )
    _attach_ir_artifact_to_intake(
        intake,
        GeneratedIRPlan(
            summary="旧 IR",
            ir_plan=MaxwellIRPlan(
                design_name="OldIRDemo",
                solution_type="Magnetostatic",
                setup_type="Magnetostatic",
                objects=[
                    {
                        "name": "bar",
                        "kind": "rectangle",
                        "material": "copper",
                        "origin_exprs": ["-5", "-1", "0"],
                        "sizes_exprs": ["10", "2"],
                    },
                    {
                        "name": "Region",
                        "kind": "region",
                        "pad_value_exprs": ["20", "20", "20", "20"],
                    },
                ],
                assignments=[
                    {"name": "Drive", "kind": "current", "targets": ["bar"], "amplitude_expr": "10"},
                    {"name": "OuterRegion", "kind": "balloon", "targets": ["Region"], "boundary_name": "OuterRegion"},
                ],
            ),
            assumptions=[],
            warnings=[],
        ),
    )

    repaired_artifact = GeneratedIRPlan(
        summary="修复后 IR",
        ir_plan=MaxwellIRPlan(
            design_name="RepairedIRDemo",
            solution_type="Magnetostatic",
            setup_type="Magnetostatic",
            objects=[
                {
                    "name": "bar",
                    "kind": "rectangle",
                    "material": "copper",
                    "origin_exprs": ["-6", "-1.5", "0"],
                    "sizes_exprs": ["12", "3"],
                },
                {
                    "name": "Region",
                    "kind": "region",
                    "pad_value_exprs": ["20", "20", "20", "20"],
                },
            ],
            assignments=[
                {"name": "Drive", "kind": "current", "targets": ["bar"], "amplitude_expr": "12"},
                {"name": "OuterRegion", "kind": "balloon", "targets": ["Region"], "boundary_name": "OuterRegion"},
            ],
        ),
        assumptions=["已根据报错修正 IR"],
        warnings=[],
    )

    client.repair_ir_artifact = lambda **kwargs: repaired_artifact
    client.build_local_fallback_script = lambda working_intake: _build_local_script_from_ir_payload(working_intake)
    client._legacy_repair_script = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("unused"))

    repaired_script = CodexaLLMClient.repair_script(
        client,
        requirement="做一个通用载流导体模型",
        intake=intake,
        script=GeneratedMaxwellScript(code="def run_job(job: dict) -> dict:\n    return {}"),
        failure_stage="runtime",
        error_details="示例报错",
    )

    assert "RepairedIRDemo" in repaired_script.code
    assert intake.simulation_spec["ir_plan"]["design_name"] == "RepairedIRDemo"


def test_revise_intake_from_feedback_prefers_ir_revision_path() -> None:
    client = object.__new__(CodexaLLMClient)
    intake = _fallback_busbar_2d_intake(
        "\u505a\u4e00\u6839\u8f7d\u6d41\u94dc\u6392\uff0c\u5bbd10mm\uff0c\u539a2mm\uff0c\u7535\u6d41200A\uff0c\u7535\u6d41\u5bc6\u5ea6\u4e0d\u8d85\u8fc75A/mm^2\u3002"
    )
    _attach_ir_artifact_to_intake(
        intake,
        GeneratedIRPlan(
            summary="旧 IR",
            ir_plan=MaxwellIRPlan(
                design_name="OldBusbarIR",
                solution_type="Magnetostatic",
                setup_type="Magnetostatic",
                parameters=[
                    {"name": "width_mm", "source": "geometry", "field": "width_mm", "default": 10.0},
                    {"name": "thickness_mm", "source": "geometry", "field": "thickness_mm", "default": 2.0},
                    {"name": "current_a", "source": "excitations", "field": "current_a", "default": 200.0},
                ],
                objects=[
                    {
                        "name": "bar",
                        "kind": "rectangle",
                        "material": "copper",
                        "origin_exprs": ["-width_mm / 2", "-thickness_mm / 2", "0"],
                        "sizes_exprs": ["width_mm", "thickness_mm"],
                    },
                    {
                        "name": "Region",
                        "kind": "region",
                        "pad_value_exprs": ["20", "20", "20", "20"],
                    },
                ],
                assignments=[
                    {"name": "Drive", "kind": "current", "targets": ["bar"], "amplitude_expr": "current_a"},
                    {"name": "OuterRegion", "kind": "balloon", "targets": ["Region"], "boundary_name": "OuterRegion"},
                ],
            ),
        ),
    )

    client.revise_ir_artifact_from_feedback = lambda **kwargs: GeneratedIRPlan(
        summary="\u5df2\u6269\u5927\u622a\u9762\u4ee5\u964d\u4f4e\u7535\u6d41\u5bc6\u5ea6\u3002",
        ir_plan=MaxwellIRPlan(
            design_name="RevisedBusbarIR",
            solution_type="Magnetostatic",
            setup_type="Magnetostatic",
            parameters=[
                {"name": "width_mm", "source": "geometry", "field": "width_mm", "default": 20.0},
                {"name": "thickness_mm", "source": "geometry", "field": "thickness_mm", "default": 2.0},
                {"name": "current_a", "source": "excitations", "field": "current_a", "default": 200.0},
            ],
            objects=[
                {
                    "name": "bar",
                    "kind": "rectangle",
                    "material": "copper",
                    "origin_exprs": ["-width_mm / 2", "-thickness_mm / 2", "0"],
                    "sizes_exprs": ["width_mm", "thickness_mm"],
                },
                {
                    "name": "Region",
                    "kind": "region",
                    "pad_value_exprs": ["20", "20", "20", "20"],
                },
            ],
            assignments=[
                {"name": "Drive", "kind": "current", "targets": ["bar"], "amplitude_expr": "current_a"},
                {"name": "OuterRegion", "kind": "balloon", "targets": ["Region"], "boundary_name": "OuterRegion"},
            ],
        ),
        assumptions=["\u7ef4\u6301 200A \u7535\u6d41\uff0c\u4f18\u5148\u6269\u5927\u622a\u9762\u3002"],
        warnings=[],
    )
    client._call_json_fallback = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should_not_fallback"))

    revised = CodexaLLMClient.revise_intake_from_feedback(
        client,
        requirement="\u505a\u4e00\u6839\u8f7d\u6d41\u94dc\u6392\uff0c\u5bbd10mm\uff0c\u539a2mm\uff0c\u7535\u6d41200A\uff0c\u7535\u6d41\u5bc6\u5ea6\u4e0d\u8d85\u8fc75A/mm^2\u3002",
        intake=intake,
        outputs={"current_a": 200.0, "cross_section_area_mm2": 20.0, "estimated_current_density_a_per_mm2": 10.0},
        evaluation={
            "overall_status": "failed",
            "summary": "\u7535\u6d41\u5bc6\u5ea6\u8d85\u9650\u3002",
            "checks": [
                {"name": "\u7535\u6d41\u5bc6\u5ea6\u4e0a\u9650", "status": "failed", "detail": "10 A/mm^2 > 5 A/mm^2"}
            ],
        },
        feedback_round=1,
    )

    assert revised.summary == "\u5df2\u6269\u5927\u622a\u9762\u4ee5\u964d\u4f4e\u7535\u6d41\u5bc6\u5ea6\u3002"
    assert revised.simulation_spec["ir_plan"]["design_name"] == "RevisedBusbarIR"
    assert revised.simulation_spec["geometry"]["width_mm"] == 20.0
    assert revised.simulation_spec["excitations"]["current_a"] == 200.0
    assert revised.extracted_parameters["feedback_round"] == 1


def test_generate_script_downgrades_mismatched_known_family_to_generic_path() -> None:
    client = object.__new__(CodexaLLMClient)
    intake = RequirementIntake(
        task_family="busbar_2d",
        supported_now=True,
        summary="\u8bef\u5206\u7c7b\u7684\u4efb\u52a1",
        simulation_spec={"geometry": {"inner_radius_mm": 3.0, "outer_radius_mm": 6.0}, "excitations": {"current_a": 100.0}},
        execution_plan={"execution_ready": True, "design_type": "Maxwell 2D", "solution_type": "Magnetostatic"},
    )
    artifact = GeneratedIRPlan(
        summary="\u540c\u5fc3\u73af\u5bfc\u4f53 IR",
        ir_plan=MaxwellIRPlan(
            design_name="AnnularConductor2D",
            solution_type="Magnetostatic",
            setup_type="Magnetostatic",
            parameters=[
                {"name": "inner_radius_mm", "source": "geometry", "field": "inner_radius_mm", "default": 3.0},
                {"name": "outer_radius_mm", "source": "geometry", "field": "outer_radius_mm", "default": 6.0},
                {"name": "current_a", "source": "excitations", "field": "current_a", "default": 100.0},
            ],
            objects=[
                {
                    "name": "outer_ring",
                    "kind": "circle",
                    "material": "copper",
                    "origin_exprs": ["0", "0", "0"],
                    "radius_expr": "outer_radius_mm",
                },
                {
                    "name": "inner_void",
                    "kind": "circle",
                    "material": "vacuum",
                    "origin_exprs": ["0", "0", "0"],
                    "radius_expr": "inner_radius_mm",
                },
                {
                    "name": "Region",
                    "kind": "region",
                    "pad_value_exprs": ["20", "20", "20", "20"],
                },
            ],
            operations=[
                {"kind": "subtract", "blank_parts": ["outer_ring"], "tool_parts": ["inner_void"]},
            ],
            assignments=[
                {"name": "Drive", "kind": "current", "targets": ["outer_ring"], "amplitude_expr": "current_a"},
                {"name": "OuterRegion", "kind": "balloon", "targets": ["Region"], "boundary_name": "OuterRegion"},
            ],
            postprocess=[
                {
                    "kind": "field_scalar",
                    "output_key": "max_flux_density_t",
                    "cast": "float",
                    "quantity": "Mag_B",
                    "scalar_function": "Maximum",
                    "object_name": "AllObjects",
                }
            ],
        ),
        assumptions=[],
        warnings=[],
    )
    client.generate_ir_artifact = lambda requirement, working_intake: artifact
    client.build_local_fallback_script = lambda working_intake: _build_local_script_from_ir_payload(working_intake)

    script = CodexaLLMClient.generate_script(
        client,
        "\u505a\u4e00\u4e2a\u4e8c\u7ef4\u540c\u5fc3\u73af\u5bfc\u4f53\u622a\u9762\uff0c\u5185\u534a\u5f843mm\uff0c\u5916\u534a\u5f846mm\uff0c\u901a\u4ee5100A\u7535\u6d41\uff0c\u8bc4\u4f30\u6700\u5927\u78c1\u5bc6\u3002",
        intake,
    )

    assert intake.task_family == "generic_maxwell"
    assert "AnnularConductor2D" in script.code


def test_build_local_generic_round_pair_script() -> None:
    intake = RequirementIntake.model_validate(
        {
            "task_family": "generic_maxwell",
            "supported_now": True,
            "summary": "二维双圆导体静电截面",
            "simulation_spec": {
                "geometry": {
                    "objects": [
                        {"name": "conductor_1", "type": "circle", "radius": {"value": 1.0}, "center": {"x": -3.0, "y": 0.0}},
                        {"name": "conductor_2", "type": "circle", "radius": {"value": 1.0}, "center": {"x": 3.0, "y": 0.0}},
                        {"name": "air_region", "type": "region", "radius": {"value": 60.0}},
                    ]
                },
                "materials": [
                    {"object": "conductor_1", "material": "copper"},
                    {"object": "conductor_2", "material": "copper"},
                ],
                "excitations": [
                    {"object": "conductor_1", "type": "voltage", "value": {"value": 100.0}},
                    {"object": "conductor_2", "type": "voltage", "value": {"value": 0.0}},
                ],
                "required_outputs": [{"name": "capacitance_per_unit_length"}, {"name": "maximum_electric_field"}],
            },
        }
    )

    script = _build_local_script_from_generic_intake(intake)

    assert "GenericCircular2D" in script.code
    assert "assign_voltage" in script.code
    assert "CapMatrix" in script.code
    assert "capacitance_per_unit_length_f_per_m" in script.code
    assert static_check_generated_script(script).passed is True


def test_build_local_generic_dual_strip_script() -> None:
    intake = RequirementIntake.model_validate(
        {
            "task_family": "generic_maxwell",
            "supported_now": True,
            "summary": "二维双矩形导体静电截面",
            "simulation_spec": {
                "geometry": {
                    "objects": [
                        {"name": "strip_1", "type": "rectangle", "center": {"x": -5.0, "y": 0.0}, "width": {"value": 4.0}, "height": {"value": 1.0}},
                        {"name": "strip_2", "type": "rectangle", "center": {"x": 5.0, "y": 0.0}, "width": {"value": 4.0}, "height": {"value": 1.0}},
                    ]
                },
                "materials": [
                    {"object": "strip_1", "material": "copper"},
                    {"object": "strip_2", "material": "copper"},
                ],
                "excitations": [
                    {"object": "strip_1", "type": "voltage", "value": {"value": 100.0}},
                    {"object": "strip_2", "type": "voltage", "value": {"value": 0.0}},
                ],
                "required_outputs": [{"name": "capacitance_per_unit_length"}, {"name": "maximum_electric_field"}],
            },
        }
    )

    script = _build_local_script_from_generic_intake(intake)

    assert "GenericRectangular2D" in script.code
    assert "create_rectangle" in script.code
    assert "assign_voltage" in script.code
    assert static_check_generated_script(script).passed is True


def test_build_local_generic_annulus_script() -> None:
    intake = RequirementIntake.model_validate(
        {
            "task_family": "generic_maxwell",
            "supported_now": True,
            "summary": "二维同心环载流导体",
            "simulation_spec": {
                "geometry": {
                    "objects": [
                        {"name": "annular_conductor", "type": "annulus", "center": {"x": 0.0, "y": 0.0}, "inner_radius": 3.0, "outer_radius": 6.0}
                    ]
                },
                "materials": [{"object": "annular_conductor", "material": "copper"}],
                "excitations": [{"object": "annular_conductor", "type": "current", "value": 100.0}],
                "required_outputs": [{"name": "Bmax_global"}],
            },
        }
    )

    script = _build_local_script_from_generic_intake(intake)

    assert "GenericAnnular2D" in script.code
    assert "subtract" in script.code
    assert "assign_current" in script.code
    assert static_check_generated_script(script).passed is True


def test_build_local_generic_annulus_script_from_ai_shape_fields() -> None:
    intake = RequirementIntake.model_validate(
        {
            "task_family": "generic_maxwell",
            "supported_now": True,
            "simulation_spec": {
                "geometry": {
                    "objects": [
                        {
                            "name": "annular_conductor",
                            "shape": "annulus",
                            "center": {"x": 0.0, "y": 0.0, "unit": "mm"},
                            "inner_radius": {"value": 3.0, "unit": "mm"},
                            "outer_radius": {"value": 6.0, "unit": "mm"},
                        }
                    ]
                },
                "materials": [{"object": "annular_conductor", "material": "copper"}],
                "excitations": [
                    {
                        "object": "annular_conductor",
                        "type": "total_current",
                        "current": {"value": 100.0, "unit": "A"},
                    }
                ],
                "required_outputs": [{"name": "B_max_global", "unit": "T"}],
            },
        }
    )

    script = _build_local_script_from_generic_intake(intake)

    assert "GenericAnnular2D" in script.code
    assert "subtract" in script.code
    assert "assign_current" in script.code
    assert static_check_generated_script(script).passed is True


def test_build_local_generic_annulus_script_from_ai_primitives_subtract() -> None:
    intake = RequirementIntake.model_validate(
        {
            "task_family": "generic_maxwell",
            "supported_now": True,
            "simulation_spec": {
                "geometry": {
                    "primitives": [
                        {"name": "outer_circle", "type": "circle", "radius": {"value": 6.0, "unit": "mm"}},
                        {"name": "inner_circle", "type": "circle", "radius": {"value": 3.0, "unit": "mm"}},
                        {"name": "annular_conductor", "type": "subtract", "blank": "outer_circle", "tool": "inner_circle"},
                    ]
                },
                "materials": [{"assignment": "annular_conductor", "chosen_material": "copper"}],
                "excitations": [
                    {
                        "target": "annular_conductor",
                        "excitation_type": "total_current",
                        "value": 100.0,
                    }
                ],
                "required_outputs": [{"name": "Bmag_max_global", "unit": "T"}],
            },
        }
    )

    script = _build_local_script_from_generic_intake(intake)

    assert "GenericAnnular2D" in script.code
    assert "subtract" in script.code
    assert "assign_current" in script.code
    assert static_check_generated_script(script).passed is True


def test_build_local_generic_annulus_script_from_ai_operations_subtract() -> None:
    intake = RequirementIntake.model_validate(
        {
            "task_family": "generic_maxwell",
            "supported_now": True,
            "simulation_spec": {
                "geometry": {
                    "primitives": [
                        {"name": "outer_circle", "type": "circle", "center_mm": [0.0, 0.0], "radius_mm": 6.0},
                        {"name": "inner_circle", "type": "circle", "center_mm": [0.0, 0.0], "radius_mm": 3.0},
                    ],
                    "operations": [
                        {"type": "subtract", "blank": "outer_circle", "tool": "inner_circle", "result_name": "annular_conductor"}
                    ],
                },
                "materials": [{"target": "annular_conductor", "material": "copper"}],
                "excitations": [{"target": "annular_conductor", "type": "total_current", "value_A": 100.0}],
                "required_outputs": [{"name": "global_max_magnetic_flux_density_magnitude", "unit": "T"}],
            },
        }
    )

    script = _build_local_script_from_generic_intake(intake)

    assert "GenericAnnular2D" in script.code
    assert "subtract" in script.code
    assert "assign_current" in script.code
    assert static_check_generated_script(script).passed is True


def test_annular_conductor_requirement_uses_local_generic_ir() -> None:
    intake = _fallback_intake_from_requirement(
        "\u505a\u4e00\u4e2a\u4e8c\u7ef4\u540c\u5fc3\u73af\u5bfc\u4f53\u622a\u9762\uff0c\u5185\u534a\u5f843mm\uff0c\u5916\u534a\u5f846mm\uff0c\u901a\u4ee5100A\u7535\u6d41\uff0c\u8bc4\u4f30\u6700\u5927\u78c1\u5bc6\u3002"
    )
    script = _build_local_script_from_generic_intake(intake)

    assert intake.task_family == "generic_maxwell"
    assert "GenericAnnular2D" in script.code
    assert "assign_current" in script.code
    assert static_check_generated_script(script).passed is True


def test_generic_supported_2d_prefers_local_generation() -> None:
    intake = RequirementIntake.model_validate(
        {
            "task_family": "generic_maxwell",
            "supported_now": True,
            "simulation_spec": {
                "geometry": {
                    "objects": [
                        {"name": "conductor_1", "type": "circle", "radius": 1.0, "center": {"x": -3.0, "y": 0.0}},
                        {"name": "conductor_2", "type": "circle", "radius": 1.0, "center": {"x": 3.0, "y": 0.0}},
                    ]
                },
                "excitations": [
                    {"object": "conductor_1", "type": "voltage", "value": 100.0},
                    {"object": "conductor_2", "type": "voltage", "value": 0.0},
                ],
            },
        }
    )

    assert CodexaLLMClient._should_prefer_local_generation("做一个二维双圆导体截面模型", intake) is True


def test_generic_dict_shaped_intake_prefers_local_generation() -> None:
    intake = RequirementIntake.model_validate(
        {
            "task_family": "generic_maxwell",
            "supported_now": True,
            "simulation_spec": {
                "geometry": {
                    "cross_section_objects": [
                        {"name": "conductor_1", "type": "circle", "radius": {"value": 1.0}, "center": {"x": -3.0, "y": 0.0}},
                        {"name": "conductor_2", "type": "circle", "radius": {"value": 1.0}, "center": {"x": 3.0, "y": 0.0}},
                    ],
                },
                "materials": {"conductor_1": "copper", "conductor_2": "copper"},
                "excitations": {"voltage_pos_v": 100.0, "voltage_neg_v": 0.0},
                "boundaries": {"air_pad_mm": 20.0},
            },
        }
    )

    assert CodexaLLMClient._should_prefer_local_generation("做一个二维双圆导体截面模型", intake) is True
    script = _build_local_script_from_generic_intake(intake)
    assert "create_circle" in script.code
    assert "CapMatrix" in script.code


def test_generic_entities_shaped_intake_prefers_local_generation() -> None:
    intake = RequirementIntake.model_validate(
        {
            "task_family": "generic_maxwell",
            "supported_now": True,
            "simulation_spec": {
                "geometry": {
                    "entities": [
                        {"name": "conductor_1", "shape": "circle", "radius": {"value": 1.0}, "center": {"x": -3.0, "y": 0.0}},
                        {"name": "conductor_2", "shape": "circle", "radius": {"value": 1.0}, "center": {"x": 3.0, "y": 0.0}},
                    ],
                },
                "materials": [
                    {"name": "copper", "applies_to": ["conductor_1", "conductor_2"]},
                ],
                "excitations": [
                    {"object": "conductor_1", "type": "voltage", "value": {"value": 100.0}},
                    {"object": "conductor_2", "type": "voltage", "value": {"value": 0.0}},
                ],
            },
        }
    )

    assert CodexaLLMClient._should_prefer_local_generation("做一个二维双圆导体截面模型", intake) is True
    script = _build_local_script_from_generic_intake(intake)
    assert "create_circle" in script.code
    assert "CapMatrix" in script.code


def test_generic_annulus_variant_prefers_local_generation() -> None:
    intake = RequirementIntake.model_validate(
        {
            "task_family": "generic_maxwell",
            "supported_now": True,
            "simulation_spec": {
                "geometry": {
                    "objects": [
                        {
                            "name": "annular_conductor",
                            "shape": "annulus",
                            "center_mm": [0.0, 0.0],
                            "inner_radius_mm": 3.0,
                            "outer_radius_mm": 6.0,
                        }
                    ],
                },
                "materials": [{"object": "annular_conductor", "material": "copper"}],
                "excitations": [{"target": "annular_conductor", "type": "current", "value_A": 100.0}],
                "required_outputs": [{"name": "Bmax_global"}],
            },
        }
    )

    assert CodexaLLMClient._should_prefer_local_generation("做一个二维同心环导体截面", intake) is True
    script = _build_local_script_from_generic_intake(intake)
    assert "GenericAnnular2D" in script.code
    assert "assign_current" in script.code


def test_generic_evaluation_accepts_numeric_electric_field_output() -> None:
    executor = MaxwellExecutor(Settings(_env_file=None, PROJECT_ROOT=r"F:\maxwell_agent_project"))
    intake = RequirementIntake(
        task_family="generic_maxwell",
        supported_now=True,
        simulation_spec={
            "required_outputs": ["capacitance", "electric_field"],
            "constraints": {
                "target_capacitance_f": 10e-12,
                "max_electric_field_v_per_m": 200000.0,
            },
        },
        execution_plan={"execution_ready": True},
    )
    evaluation = executor._build_evaluation(
        intake,
        outputs={
            "capacitance_pf": 12.5,
            "max_electric_field_v_per_m": 150000.0,
        },
        run_status="completed",
    )

    assert evaluation.overall_status == "passed"
    assert any(item.name == "\u7535\u573a\u7ed3\u679c" and item.status == "passed" for item in evaluation.checks)
    assert any(item.name == "\u7535\u573a\u4e0a\u9650" and item.status == "passed" for item in evaluation.checks)


def test_generic_evaluation_accepts_capacitance_per_unit_length_alias() -> None:
    executor = MaxwellExecutor(Settings(_env_file=None, PROJECT_ROOT=r"F:\maxwell_agent_project"))
    intake = RequirementIntake(
        task_family="generic_maxwell",
        supported_now=True,
        simulation_spec={
            "required_outputs": [{"name": "capacitance_per_unit_length", "unit": "F/m"}],
        },
        execution_plan={"execution_ready": True},
    )
    evaluation = executor._build_evaluation(
        intake,
        outputs={"capacitance_per_unit_length_f_per_m": 15.723e-12},
        run_status="completed",
    )

    assert evaluation.overall_status == "passed"
    assert any(item.name == "\u7535\u5bb9\u7ed3\u679c" and item.status == "passed" for item in evaluation.checks)


def test_generic_evaluation_accepts_bmax_aliases() -> None:
    executor = MaxwellExecutor(Settings(_env_file=None, PROJECT_ROOT=r"F:\maxwell_agent_project"))
    intake = RequirementIntake(
        task_family="generic_maxwell",
        supported_now=True,
        simulation_spec={
            "required_outputs": [{"name": "Bmax_global"}],
        },
        execution_plan={"execution_ready": True},
    )
    evaluation = executor._build_evaluation(
        intake,
        outputs={"b_max_global_t": 0.0034},
        run_status="completed",
    )

    assert evaluation.overall_status == "passed"
    assert any(item.name == "\u78c1\u573a\u7ed3\u679c" and item.status == "passed" for item in evaluation.checks)


def test_generic_evaluation_accepts_compact_bmax_aliases() -> None:
    executor = MaxwellExecutor(Settings(_env_file=None, PROJECT_ROOT=r"F:\maxwell_agent_project"))
    intake = RequirementIntake(
        task_family="generic_maxwell",
        supported_now=True,
        simulation_spec={
            "required_outputs": [{"name": "Bmax_global"}],
        },
        execution_plan={"execution_ready": True},
    )
    evaluation = executor._build_evaluation(
        intake,
        outputs={"bmax_global_t": 0.0034},
        run_status="completed",
    )

    assert evaluation.overall_status == "passed"
    assert any(item.name == "\u78c1\u573a\u7ed3\u679c" and item.status == "passed" for item in evaluation.checks)


def test_generic_evaluation_accepts_bmag_max_aliases() -> None:
    executor = MaxwellExecutor(Settings(_env_file=None, PROJECT_ROOT=r"F:\maxwell_agent_project"))
    intake = RequirementIntake(
        task_family="generic_maxwell",
        supported_now=True,
        simulation_spec={
            "required_outputs": [{"name": "Bmag_Max_Global"}],
        },
        execution_plan={"execution_ready": True},
    )
    evaluation = executor._build_evaluation(
        intake,
        outputs={"bmag_max_global_t": 0.0034},
        run_status="completed",
    )

    assert evaluation.overall_status == "passed"
    assert any(item.name == "\u78c1\u573a\u7ed3\u679c" and item.status == "passed" for item in evaluation.checks)


def test_generic_evaluation_accepts_global_max_b_aliases() -> None:
    executor = MaxwellExecutor(Settings(_env_file=None, PROJECT_ROOT=r"F:\maxwell_agent_project"))
    intake = RequirementIntake(
        task_family="generic_maxwell",
        supported_now=True,
        simulation_spec={
            "required_outputs": [{"name": "Bmag_max_global"}],
        },
        execution_plan={"execution_ready": True},
    )
    evaluation = executor._build_evaluation(
        intake,
        outputs={"global_max_b_t": 0.0034},
        run_status="completed",
    )

    assert evaluation.overall_status == "passed"
    assert any(item.name == "\u78c1\u573a\u7ed3\u679c" and item.status == "passed" for item in evaluation.checks)


def test_generic_evaluation_accepts_max_b_magnitude_aliases() -> None:
    executor = MaxwellExecutor(Settings(_env_file=None, PROJECT_ROOT=r"F:\maxwell_agent_project"))
    intake = RequirementIntake(
        task_family="generic_maxwell",
        supported_now=True,
        simulation_spec={
            "required_outputs": [{"name": "global_max_magnetic_flux_density_magnitude"}],
        },
        execution_plan={"execution_ready": True},
    )
    evaluation = executor._build_evaluation(
        intake,
        outputs={"max_b_magnitude_overall": 0.0034},
        run_status="completed",
    )

    assert evaluation.overall_status == "passed"
    assert any(item.name == "\u78c1\u573a\u7ed3\u679c" and item.status == "passed" for item in evaluation.checks)


def test_generic_evaluation_accepts_max_mag_b_aliases() -> None:
    executor = MaxwellExecutor(Settings(_env_file=None, PROJECT_ROOT=r"F:\maxwell_agent_project"))
    intake = RequirementIntake(
        task_family="generic_maxwell",
        supported_now=True,
        simulation_spec={
            "required_outputs": [{"name": "B_max_global"}],
        },
        execution_plan={"execution_ready": True},
    )
    evaluation = executor._build_evaluation(
        intake,
        outputs={"max_mag_b_t": 0.0034},
        run_status="completed",
    )

    assert evaluation.overall_status == "passed"
    assert any(item.name == "\u78c1\u573a\u7ed3\u679c" and item.status == "passed" for item in evaluation.checks)


def test_local_inductor_script_uses_fast_estimate_path() -> None:
    intake = _fallback_intake_from_requirement(
        "\u505a\u4e00\u4e2a\u7535\u611f\u5668\uff0c\u6c14\u96991mm\uff0c\u5305\u6570200\uff0c\u7535\u6d411A\uff0c\u76ee\u6807\u7535\u611f10mH\uff0c\u5e76\u8f93\u51fa\u6700\u5927\u78c1\u5bc6\u3002"
    )
    script = _build_local_script_from_inductor_intake(intake)

    assert "analyze_setup" not in script.code
    assert "skipped_fast_estimate" in script.code
    assert '"estimated_inductance_h"' in script.code
    assert '"max_flux_density_t"' in script.code


def test_inductor_evaluation_passes_target_from_fast_estimate() -> None:
    executor = MaxwellExecutor(Settings(_env_file=None, PROJECT_ROOT=r"F:\maxwell_agent_project"))
    intake = _fallback_intake_from_requirement(
        "\u505a\u4e00\u4e2a\u7535\u611f\u5668\uff0c\u6c14\u96991mm\uff0c\u5305\u6570200\uff0c\u7535\u6d411A\uff0c\u76ee\u6807\u7535\u611f10mH\uff0c\u5e76\u8f93\u51fa\u6700\u5927\u78c1\u5bc6\u3002"
    )
    evaluation = executor._build_evaluation(
        intake,
        outputs={"estimated_inductance_h": 0.01, "max_flux_density_t": 0.5},
        run_status="completed",
    )

    assert evaluation.overall_status == "passed"


def test_evaluation_for_new_generic_task_families() -> None:
    executor = MaxwellExecutor(Settings(_env_file=None, PROJECT_ROOT=r"F:\maxwell_agent_project"))

    busbar_eval = executor._build_evaluation(
        _fallback_busbar_2d_intake(
            "\u505a\u4e00\u6839\u8f7d\u6d41\u94dc\u6392\uff0c\u5bbd10mm\uff0c\u539a2mm\uff0c\u7535\u6d41200A\uff0c\u8bc4\u4f30\u6700\u5927\u78c1\u5bc6\u548c\u7535\u6d41\u5bc6\u5ea6\u3002"
        ),
        outputs={
            "max_flux_density_t": 0.012,
            "estimated_current_density_a_per_mm2": 10.0,
            "current_a": 200.0,
            "cross_section_area_mm2": 20.0,
        },
        run_status="completed",
    )
    solenoid_eval = executor._build_evaluation(
        _fallback_solenoid_2d_intake(
            "\u505a\u4e00\u4e2a\u7a7a\u5fc3\u87ba\u7ebf\u7ba1\uff0c\u957f\u5ea650mm\uff0c\u534a\u5f848mm\uff0c\u5305\u6570300\uff0c\u7535\u6d411A\uff0c\u6c42\u4e2d\u5fc3\u78c1\u5bc6\u3002"
        ),
        outputs={
            "estimated_center_flux_density_t": 0.0075,
            "equivalent_current_a": 300.0,
            "current_a": 1.0,
            "coil_turns": 300,
        },
        run_status="completed",
    )
    coax_eval = executor._build_evaluation(
        _fallback_coaxial_capacitor_2d_intake(
            "\u505a\u4e00\u4e2a\u540c\u8f74\u7535\u5bb9\u5668\uff0c\u5185\u534a\u5f841mm\uff0c\u5916\u534a\u5f845mm\uff0c\u65bd\u52a0100V\u7535\u538b\uff0c\u6c42\u7535\u5bb9\u548c\u7535\u573a\u3002"
        ),
        outputs={
            "capacitance_pf": 34.58,
            "max_electric_field_v_per_m": 58200.0,
        },
        run_status="completed",
    )

    assert busbar_eval.overall_status == "passed"
    assert solenoid_eval.overall_status == "passed"
    assert coax_eval.overall_status == "passed"


def test_busbar_evaluation_fails_when_current_density_limit_is_violated() -> None:
    executor = MaxwellExecutor(Settings(_env_file=None, PROJECT_ROOT=r"F:\maxwell_agent_project"))
    busbar_eval = executor._build_evaluation(
        _fallback_busbar_2d_intake(
            "\u505a\u4e00\u6839\u8f7d\u6d41\u94dc\u6392\uff0c\u5bbd10mm\uff0c\u539a2mm\uff0c\u7535\u6d41200A\uff0c\u7535\u6d41\u5bc6\u5ea6\u4e0d\u8d85\u8fc75A/mm^2\u3002"
        ),
        outputs={
            "max_flux_density_t": 0.012,
            "estimated_current_density_a_per_mm2": 10.0,
            "current_a": 200.0,
            "cross_section_area_mm2": 20.0,
        },
        run_status="completed",
    )

    assert busbar_eval.overall_status == "failed"
    assert any(item.name == "\u7535\u6d41\u5bc6\u5ea6\u4e0a\u9650" and item.status == "failed" for item in busbar_eval.checks)


def test_busbar_evaluation_accepts_llm_style_density_keys() -> None:
    executor = MaxwellExecutor(Settings(_env_file=None, PROJECT_ROOT=r"F:\maxwell_agent_project"))
    intake = _fallback_busbar_2d_intake(
        "\u505a\u4e00\u6839\u8f7d\u6d41\u94dc\u6392\uff0c\u5bbd10mm\uff0c\u539a2mm\uff0c\u7535\u6d41200A\uff0c\u7535\u6d41\u5bc6\u5ea6\u4e0d\u8d85\u8fc75A/mm^2\u3002"
    )
    intake.simulation_spec["constraints"] = {"j_limit_a_per_mm2": 5.0}
    busbar_eval = executor._build_evaluation(
        intake,
        outputs={
            "max_flux_density_t": 0.012,
            "max_current_density_a_per_mm2": 10.0,
            "area_mm2": 20.0,
        },
        run_status="completed",
    )

    assert busbar_eval.overall_status == "failed"
    assert any(item.name == "\u7535\u6d41\u5bc6\u5ea6\u7ed3\u679c" and item.status == "passed" for item in busbar_eval.checks)
    assert any(item.name == "\u7535\u6d41\u5bc6\u5ea6\u4e0a\u9650" and item.status == "failed" for item in busbar_eval.checks)


def test_busbar_evaluation_accepts_avg_density_key() -> None:
    executor = MaxwellExecutor(Settings(_env_file=None, PROJECT_ROOT=r"F:\maxwell_agent_project"))
    intake = _fallback_busbar_2d_intake(
        "\u505a\u4e00\u6839\u8f7d\u6d41\u94dc\u6392\uff0c\u5bbd10mm\uff0c\u539a2mm\uff0c\u7535\u6d41200A\uff0c\u7535\u6d41\u5bc6\u5ea6\u4e0d\u8d85\u8fc75A/mm^2\u3002"
    )
    busbar_eval = executor._build_evaluation(
        intake,
        outputs={
            "max_flux_density_t": 0.012,
            "avg_current_density_a_per_mm2": 10.0,
            "cross_section_area_mm2": 20.0,
            "current_a": 200.0,
        },
        run_status="completed",
    )

    assert any(item.name == "\u7535\u6d41\u5bc6\u5ea6\u7ed3\u679c" and item.status == "passed" for item in busbar_eval.checks)
    assert any(item.name == "\u7535\u6d41\u5bc6\u5ea6\u4e0a\u9650" and item.status == "failed" for item in busbar_eval.checks)


def test_coaxial_capacitor_evaluation_fails_when_constraints_violated() -> None:
    executor = MaxwellExecutor(Settings(_env_file=None, PROJECT_ROOT=r"F:\maxwell_agent_project"))
    intake = _fallback_coaxial_capacitor_2d_intake(
        "\u505a\u4e00\u4e2a\u540c\u8f74\u7535\u5bb9\u5668\uff0c\u5185\u534a\u5f841mm\uff0c\u5916\u534a\u5f845mm\uff0c\u65bd\u52a0100V\u7535\u538b\uff0c\u6c42\u7535\u5bb9\u548c\u7535\u573a\u3002"
    )
    intake.simulation_spec["constraints"] = {
        "target_capacitance_f": 50e-12,
        "max_electric_field_v_per_m": 50000.0,
    }
    evaluation = executor._build_evaluation(
        intake,
        outputs={
            "capacitance_pf": 34.58,
            "max_electric_field_v_per_m": 58200.0,
        },
        run_status="completed",
    )

    assert evaluation.overall_status == "failed"
    assert any(item.name == "\u7535\u5bb9\u76ee\u6807" and item.status == "failed" for item in evaluation.checks)
    assert any(item.name == "\u7535\u573a\u4e0a\u9650" and item.status == "failed" for item in evaluation.checks)


def test_parallel_plate_capacitor_evaluation_uses_gap_average_for_field_limit() -> None:
    executor = MaxwellExecutor(Settings(_env_file=None, PROJECT_ROOT=r"F:\maxwell_agent_project"))
    intake = _fallback_capacitor_intake(
        "\u505a\u4e00\u4e2a\u4e8c\u7ef4\u5e73\u884c\u677f\u7535\u5bb9\u5668\uff0c\u677f\u95f4\u8ddd1mm\uff0c\u677f\u5bbd20mm\uff0c\u65bd\u52a0100V\uff0c\u7535\u5bb9\u81f3\u5c11 50pF\uff0c\u7535\u573a\u4e0d\u8d85\u8fc7 200000V/m\u3002"
    )
    evaluation = executor._build_evaluation(
        intake,
        outputs={
            "capacitance_pf": 198.18,
            "max_electric_field_note": 369654.95,
            "reference_average_field_v_per_m": 100000.0,
        },
        run_status="completed",
    )

    assert evaluation.overall_status == "passed"
    assert any(item.name == "\u7535\u5bb9\u76ee\u6807" and item.status == "passed" for item in evaluation.checks)
    assert any(item.name == "\u7535\u573a\u4e0a\u9650" and item.status == "passed" for item in evaluation.checks)


def test_parallel_plate_capacitor_evaluation_fails_without_gap_average_when_peak_exceeds_limit() -> None:
    executor = MaxwellExecutor(Settings(_env_file=None, PROJECT_ROOT=r"F:\maxwell_agent_project"))
    intake = _fallback_capacitor_intake(
        "\u505a\u4e00\u4e2a\u4e8c\u7ef4\u5e73\u884c\u677f\u7535\u5bb9\u5668\uff0c\u677f\u95f4\u8ddd1mm\uff0c\u677f\u5bbd20mm\uff0c\u65bd\u52a0100V\uff0c\u7535\u5bb9\u81f3\u5c11 50pF\uff0c\u7535\u573a\u4e0d\u8d85\u8fc7 200000V/m\u3002"
    )
    evaluation = executor._build_evaluation(
        intake,
        outputs={
            "capacitance_pf": 198.18,
            "max_electric_field_note": 369654.95,
        },
        run_status="completed",
    )

    assert evaluation.overall_status == "failed"
    assert any(item.name == "\u7535\u573a\u4e0a\u9650" and item.status == "failed" for item in evaluation.checks)


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
