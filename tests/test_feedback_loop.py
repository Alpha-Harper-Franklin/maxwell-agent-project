from pathlib import Path

from maxwell_agent.agent import MaxwellAgent
from maxwell_agent.models import MaxwellEnvironment, RequirementCheck, RequirementEvaluation, SimulationResult


def _env() -> MaxwellEnvironment:
    return MaxwellEnvironment(installed=True, pyaedt_importable=True)


def test_feedback_iteration_triggers_on_failed_evaluation() -> None:
    result = SimulationResult(
        status="completed",
        message="done",
        run_directory=Path(r"F:\maxwell_agent_project\workspace\dummy"),
        environment=_env(),
        evaluation=RequirementEvaluation(
            overall_status="failed",
            summary="failed",
            checks=[RequirementCheck(name="硬约束", status="failed", detail="not met")],
        ),
    )

    assert MaxwellAgent._needs_feedback_iteration(result) is True


def test_feedback_iteration_does_not_trigger_on_passed_evaluation() -> None:
    result = SimulationResult(
        status="completed",
        message="done",
        run_directory=Path(r"F:\maxwell_agent_project\workspace\dummy"),
        environment=_env(),
        evaluation=RequirementEvaluation(
            overall_status="passed",
            summary="passed",
            checks=[RequirementCheck(name="硬约束", status="passed", detail="ok")],
        ),
    )

    assert MaxwellAgent._needs_feedback_iteration(result) is False
