from pathlib import Path

from maxwell_agent.demo import build_demo_bundle, execute_demo
from maxwell_agent.demo_server import _render_page
from maxwell_agent.models import (
    ElectromagnetDesign,
    MaxwellEnvironment,
    RequirementCheck,
    RequirementEvaluation,
    RequirementIntake,
    SimulationResult,
)


def test_build_demo_bundle_formats_status_and_outputs() -> None:
    design = ElectromagnetDesign(summary="demo")
    environment = MaxwellEnvironment(installed=True, pyaedt_importable=True)
    result = SimulationResult(
        status="completed",
        message="done",
        run_directory=Path(r"F:\maxwell_agent_project\workspace\demo"),
        project_file=Path(r"F:\maxwell_agent_project\workspace\demo\demo.aedt"),
        environment=environment,
        design=design,
        outputs={"max_flux_density_t": 0.1234, "project_name": "demo_project"},
        evaluation=RequirementEvaluation(
            overall_status="partial",
            summary="只验证了部分需求。",
            checks=[RequirementCheck(name="电流约束", status="passed", detail="满足 2A 上限。")],
        ),
        artifacts=[Path(r"F:\maxwell_agent_project\workspace\demo\outputs.json")],
    )

    bundle = build_demo_bundle("test requirement", result)

    assert bundle.status == "completed"
    assert bundle.status_label == "已完成"
    assert bundle.design_rows
    assert any(row.value == "0.1234" for row in bundle.output_rows)
    assert bundle.evaluation_summary == "只验证了部分需求。"


def test_execute_demo_forwards_progress_callback(tmp_path: Path) -> None:
    updates: list[tuple[int, str]] = []

    class StubAgent:
        def run(self, requirement: str, progress_callback=None) -> SimulationResult:
            if progress_callback:
                progress_callback(12, "parse")
                progress_callback(100, "done")
            return SimulationResult(
                status="completed",
                message="done",
                run_directory=tmp_path,
                project_file=tmp_path / "demo.aedt",
                environment=MaxwellEnvironment(installed=True, pyaedt_importable=True),
                design=ElectromagnetDesign(summary=requirement),
                outputs={"project_name": "demo_project"},
                artifacts=[tmp_path / "outputs.json"],
            )

    bundle = execute_demo(
        StubAgent(),
        "demo requirement",
        progress_callback=lambda percent, message: updates.append((percent, message)),
    )

    assert updates == [(12, "parse"), (100, "done")]
    assert bundle.summary_text_path == tmp_path / "demo_summary.txt"
    assert bundle.summary_html_path == tmp_path / "demo_summary.html"


def test_render_page_contains_polling_progress_ui() -> None:
    page = _render_page()

    assert "/status" in page
    assert "progress-bar" in page
    assert "setInterval" in page


def test_build_demo_bundle_handles_blocked_scope_without_design() -> None:
    result = SimulationResult(
        status="blocked",
        message="当前系统还没有变压器执行器。",
        run_directory=Path(r"F:\maxwell_agent_project\workspace\blocked"),
        environment=MaxwellEnvironment(installed=True, pyaedt_importable=True),
        design=None,
        intake=RequirementIntake(
            task_family="transformer",
            supported_now=False,
            support_message="当前系统还没有变压器执行器。",
            summary="已识别为变压器需求。",
            extracted_parameters={"input_voltage_v": 10000, "output_voltage_v": 220},
            simulation_spec={"software": "ansys_maxwell", "execution_ready": False},
        ),
        outputs={
            "task_family": "transformer",
            "support_message": "当前系统还没有变压器执行器。",
            "param_input_voltage_v": 10000,
            "spec_execution_ready": "false",
        },
        evaluation=RequirementEvaluation(
            overall_status="failed",
            summary="当前系统还没有变压器执行器。",
            checks=[RequirementCheck(name="任务范围判定", status="failed", detail="变压器任务未进入当前执行链路")],
        ),
        artifacts=[Path(r"F:\maxwell_agent_project\workspace\blocked\requirement.json")],
    )

    bundle = build_demo_bundle("做一个10000v到220市电的变压器", result)

    assert any("结构化参数 / input_voltage_v" == row.label for row in bundle.design_rows)
    assert any("仿真规格 / execution_ready" == row.label for row in bundle.output_rows)
