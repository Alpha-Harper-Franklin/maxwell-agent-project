from pathlib import Path

from maxwell_agent.case_report import (
    build_case_delivery_report,
    build_case_insight,
    build_iteration_record,
    persist_case_delivery_report,
)
from maxwell_agent.models import (
    MaxwellEnvironment,
    RequirementCheck,
    RequirementEvaluation,
    RequirementIntake,
    SimulationResult,
)


def _generic_intake() -> RequirementIntake:
    return RequirementIntake(
        task_family="generic_maxwell",
        supported_now=True,
        simulation_spec={
            "physics_type": "magnetostatic_2d",
            "execution_ready": True,
            "geometry": {
                "objects": [
                    {"name": "outer_ring", "primitive": "circle", "material": "copper"},
                    {"name": "inner_hole", "primitive": "circle", "role": "subtract"},
                ]
            },
            "constraints": {"required_current_a": 100, "max_flux_density_t": 1.5},
            "required_outputs": [{"name": "max_flux_density_t"}],
        },
        execution_plan={"physics_type": "magnetostatic_2d"},
    )


def test_case_insight_extracts_geometry_constraints_outputs_and_knowledge(tmp_path: Path) -> None:
    intake = _generic_intake()
    learned = tmp_path / "learned_primitive_annular_conductor.json"
    learned.write_text("{}", encoding="utf-8")

    insight = build_case_insight(
        intake=intake,
        outputs={"max_flux_density_t": 0.82},
        evaluation=RequirementEvaluation(
            overall_status="passed",
            summary="passed",
            checks=[RequirementCheck(name="磁密约束", status="passed", detail="0.82T <= 1.5T")],
        ),
        artifacts=[learned],
        requirement="做一个同心环导体，通以100A电流，输出最大磁密。",
    )

    assert insight.physics_type == "magnetostatic_2d"
    assert any(item["name"] == "outer_ring" for item in insight.geometry_objects)
    assert any(item["name"] == "required_current_a" for item in insight.constraint_items)
    assert any(item["name"] == "max_flux_density_t" for item in insight.output_items)
    assert any("learned_primitive_annular_conductor" in item for item in insight.knowledge_items)
    assert any("满足" in item for item in insight.engineering_explanations)


def test_iteration_and_delivery_report_are_persisted(tmp_path: Path) -> None:
    intake = _generic_intake()
    evaluation = RequirementEvaluation(
        overall_status="failed",
        summary="存在未满足约束",
        checks=[
            RequirementCheck(name="磁密约束", status="failed", detail="1.8T > 1.5T"),
            RequirementCheck(name="电流约束", status="passed", detail="100A 已施加"),
        ],
    )
    result = SimulationResult(
        status="completed",
        message="done",
        run_directory=tmp_path,
        environment=MaxwellEnvironment(installed=True, pyaedt_importable=True),
        intake=intake,
        outputs={"max_flux_density_t": 1.8},
        evaluation=evaluation,
    )
    iteration = build_iteration_record(
        index=1,
        result=result,
        intake=intake,
        requirement="做一个同心环导体，通以100A电流，输出最大磁密。",
        feedback_required=True,
        feedback_reason="1.8T > 1.5T",
    )
    report = build_case_delivery_report(
        requirement="做一个同心环导体，通以100A电流，输出最大磁密。",
        result=result,
        iterations=[iteration],
    )
    paths = persist_case_delivery_report(report, tmp_path)

    assert iteration.feedback_required is True
    assert iteration.failed_checks == ["1.8T > 1.5T"]
    assert len(paths) == 3
    assert (tmp_path / "case_delivery_report.json").exists()
    assert "第 1 轮" in (tmp_path / "case_delivery_report.md").read_text(encoding="utf-8")
    assert "Maxwell Agent 单案例交付报告" in (tmp_path / "case_delivery_report.html").read_text(encoding="utf-8")
