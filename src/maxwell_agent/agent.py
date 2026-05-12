from __future__ import annotations

from collections.abc import Callable
import json
from datetime import datetime
from pathlib import Path

from .config import Settings
from .errors import RequirementPlanningError, UnknownPrimitiveError
from .llm_client import CodexaLLMClient, _enforce_design_current_voltage_constraints
from .maxwell_env import detect_maxwell_environment
from .maxwell_executor import MaxwellExecutor
from .models import (
    ElectromagnetDesign,
    GeneratedMaxwellScript,
    RequirementCheck,
    RequirementEvaluation,
    RequirementIntake,
    ScriptStaticCheck,
    SimulationResult,
)
from .semantics import infer_builder_hint, intake_has_generic_object_graph
from .script_validation import static_check_generated_script


ProgressCallback = Callable[[int, str], None]


class MaxwellAgent:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = CodexaLLMClient(settings)
        self._executor = MaxwellExecutor(settings)

    def intake(self, requirement: str) -> RequirementIntake:
        initial = self._llm.generate_requirement_intake(requirement)
        return self._llm.refine_requirement_intake(requirement, initial)

    def plan(self, requirement: str) -> ElectromagnetDesign:
        intake = self.intake(requirement)
        if not self._intake_is_executable(intake) or not intake.design:
            raise RequirementPlanningError(intake.support_message, reason_code="unsupported_requirement", intake=intake)
        return intake.design

    def run(
        self,
        requirement: str,
        progress_callback: ProgressCallback | None = None,
    ) -> SimulationResult:
        self._settings.ensure_dirs()
        run_dir = self._create_run_directory()
        all_artifacts: list[Path] = []

        if progress_callback:
            progress_callback(10, "正在调用 AI 进行需求结构化")
        try:
            intake_round1 = self._llm.generate_requirement_intake(requirement)
            all_artifacts.extend(self._persist_intake_artifacts(run_dir, requirement, intake_round1, round_name="round1"))

            if progress_callback:
                progress_callback(18, "正在让 AI 补全缺失信息和工程假设")
            intake = self._llm.refine_requirement_intake(requirement, intake_round1)
            all_artifacts.extend(self._persist_intake_artifacts(run_dir, requirement, intake, round_name="final"))
        except RequirementPlanningError as exc:
            return self._build_blocked_result(
                requirement=requirement,
                run_dir=run_dir,
                error=exc,
                progress_callback=progress_callback,
                extra_artifacts=all_artifacts,
            )

        if not self._intake_is_executable(intake):
            return self._build_blocked_result(
                requirement=requirement,
                run_dir=run_dir,
                error=RequirementPlanningError(
                    intake.support_message or "当前结构化结果还不能直接进入 Maxwell 脚本生成。",
                    reason_code="unsupported_requirement",
                    intake=intake,
                ),
                progress_callback=progress_callback,
                extra_artifacts=all_artifacts,
            )

        current_intake = intake
        final_result: SimulationResult | None = None
        reusable_script: GeneratedMaxwellScript | None = None
        for feedback_round in range(self._settings.design_feedback_max_iters + 1):
            try:
                result, reusable_script = self._execute_for_intake(
                    requirement=requirement,
                    intake=current_intake,
                    run_dir=run_dir,
                    progress_callback=progress_callback,
                    attempt_seed=feedback_round * 10,
                    seed_script=reusable_script,
                )
            except UnknownPrimitiveError as exc:
                if progress_callback:
                    progress_callback(24, f"检测到未知二维原语 {exc.primitive_token}，正在请求 AI 学习并加入本地原语库")
                try:
                    artifact = self._learn_primitive_with_repair(
                        requirement=requirement,
                        intake=current_intake,
                        primitive_token=exc.primitive_token,
                        raw_object=exc.raw_object,
                        error_details=exc.message,
                    )
                    self._llm.primitive_library.register(artifact.template, persist=False, mark_persisted=False)
                    all_artifacts.append(
                        self._persist_json_artifact(
                            run_dir / f"learned_primitive_{artifact.template.primitive_key}.json",
                            artifact.model_dump(mode="json"),
                        )
                    )
                    result, reusable_script = self._execute_for_intake(
                        requirement=requirement,
                        intake=current_intake,
                        run_dir=run_dir,
                        progress_callback=progress_callback,
                        attempt_seed=feedback_round * 10,
                        seed_script=reusable_script,
                    )
                    if result.status == "completed" and reusable_script.primitive_library_updates:
                        committed_templates = []
                        for item in reusable_script.primitive_library_updates:
                            if not isinstance(item, dict):
                                continue
                            primitive_key = str(item.get("primitive_key") or "").strip()
                            template = self._llm.primitive_library.find(primitive_key)
                            if template is not None:
                                committed_templates.append(template)
                        if committed_templates:
                            self._llm.primitive_library.commit(committed_templates)
                except RequirementPlanningError as learn_exc:
                    return self._build_blocked_result(
                        requirement=requirement,
                        run_dir=run_dir,
                        error=RequirementPlanningError(
                            learn_exc.message,
                            reason_code=learn_exc.reason_code or "primitive_learning_failed",
                            intake=current_intake,
                        ),
                        progress_callback=progress_callback,
                        extra_artifacts=self._dedupe_paths(all_artifacts),
                    )
            except RequirementPlanningError as exc:
                return self._build_blocked_result(
                    requirement=requirement,
                    run_dir=run_dir,
                    error=RequirementPlanningError(
                        exc.message,
                        reason_code=exc.reason_code or "execution_preparation_failed",
                        intake=current_intake,
                    ),
                    progress_callback=progress_callback,
                    extra_artifacts=self._dedupe_paths(all_artifacts),
                )
            all_artifacts.extend(result.artifacts)
            result.intake = current_intake
            result.artifacts = self._dedupe_paths(all_artifacts)
            final_result = result

            if not self._needs_feedback_iteration(result):
                return result

            if feedback_round >= self._settings.design_feedback_max_iters:
                return result

            if progress_callback:
                progress_callback(66, f"第 {feedback_round + 1} 轮存在未满足约束，正在把仿真反馈回传给 AI")
            try:
                current_intake = self._revise_intake_with_timeout(
                    requirement=requirement,
                    intake=current_intake,
                    outputs=result.outputs,
                    evaluation=result.evaluation,
                    feedback_round=feedback_round + 1,
                )
            except RequirementPlanningError as exc:
                return self._build_blocked_result(
                    requirement=requirement,
                    run_dir=run_dir,
                    error=RequirementPlanningError(
                        f"LLM 反馈修正失败，本轮未使用任何启发式兜底。{exc.message}",
                        reason_code="feedback_revision_failed",
                        intake=current_intake,
                    ),
                    progress_callback=progress_callback,
                    extra_artifacts=self._dedupe_paths(all_artifacts),
                )
            all_artifacts.extend(
                self._persist_feedback_artifacts(
                    run_dir=run_dir,
                    requirement=requirement,
                    intake=current_intake,
                    revised_design=current_intake.design,
                    feedback_round=feedback_round + 1,
                )
            )

        if final_result is None:
            return self._build_blocked_result(
                requirement=requirement,
                run_dir=run_dir,
                error=RequirementPlanningError("未能产生可执行结果。", reason_code="execution_failed", intake=intake),
                progress_callback=progress_callback,
                extra_artifacts=self._dedupe_paths(all_artifacts),
            )
        final_result.artifacts = self._dedupe_paths(all_artifacts)
        return final_result

    def _revise_intake_with_timeout(
        self,
        requirement: str,
        intake: RequirementIntake,
        outputs: dict[str, float | str] | None,
        evaluation: RequirementEvaluation | None,
        feedback_round: int,
    ) -> RequirementIntake:
        if intake.design is not None:
            revised_design = _enforce_design_current_voltage_constraints(intake.design)
            return self._llm.replace_design_in_intake(
                requirement=requirement,
                intake=intake,
                revised_design=revised_design,
                feedback_round=feedback_round,
                outputs=outputs,
                evaluation=evaluation,
            )
        try:
            return self._llm.revise_intake_from_feedback(
                requirement=requirement,
                intake=intake,
                outputs=outputs,
                evaluation=evaluation,
                feedback_round=feedback_round,
            )
        except RequirementPlanningError:
            if intake.design is not None:
                revised_design = _enforce_design_current_voltage_constraints(intake.design)
                return self._llm.replace_design_in_intake(
                    requirement=requirement,
                    intake=intake,
                    revised_design=revised_design,
                    feedback_round=feedback_round,
                    outputs=outputs,
                    evaluation=evaluation,
                )
            raise

    def _learn_primitive_with_repair(
        self,
        requirement: str,
        intake: RequirementIntake,
        primitive_token: str,
        raw_object: dict[str, object] | None,
        error_details: str,
    ):
        previous_template = None
        last_error = error_details
        max_attempts = max(1, self._settings.script_max_repairs + 1)
        for _ in range(max_attempts):
            artifact = self._llm.learn_primitive_template(
                requirement=requirement,
                intake=intake,
                primitive_token=primitive_token,
                raw_object=raw_object or {},
                error_details=last_error,
                previous_template=previous_template,
            )
            self._llm.primitive_library.register(artifact.template, persist=False, mark_persisted=False)
            try:
                self._llm.build_local_fallback_script(intake)
                return artifact
            except UnknownPrimitiveError as exc:
                if exc.primitive_token != primitive_token:
                    raise
                last_error = exc.message
                previous_template = artifact.template
                continue
            except RequirementPlanningError as exc:
                last_error = exc.message
                previous_template = artifact.template
                continue
        raise RequirementPlanningError(
            f"未知原语 {primitive_token} 经多轮学习后仍无法落到本地可执行原语库。最后错误: {last_error}",
            reason_code="primitive_learning_failed",
            intake=intake,
        )

    def _execute_for_intake(
        self,
        requirement: str,
        intake: RequirementIntake,
        run_dir: Path,
        progress_callback: ProgressCallback | None = None,
        attempt_seed: int = 0,
        seed_script: GeneratedMaxwellScript | None = None,
    ) -> tuple[SimulationResult, GeneratedMaxwellScript]:
        has_ir_plan = bool((intake.simulation_spec or {}).get("ir_plan")) or bool((intake.execution_plan or {}).get("ir_plan"))
        can_reuse_seed = seed_script is not None and not has_ir_plan and intake.design is not None
        if not can_reuse_seed:
            if progress_callback:
                progress_callback(26, "正在调用 AI 生成 PyAEDT 脚本")
            script = self._llm.generate_script(requirement, intake)
        else:
            if progress_callback:
                progress_callback(26, "反馈轮复用上一轮已跑通的 PyAEDT 脚本，只替换设计参数")
            script = seed_script

        static_check, static_artifacts, script = self._repair_until_static_check_passes(
            requirement=requirement,
            intake=intake,
            script=script,
            run_dir=run_dir,
            progress_callback=progress_callback,
            start_index=attempt_seed + 1,
        )
        phase_artifacts: list[Path] = list(static_artifacts)
        if not static_check.passed:
            return self._build_blocked_result(
                requirement=requirement,
                run_dir=run_dir,
                error=RequirementPlanningError(
                    "AI 脚本多轮修复后仍未通过静态检查。",
                    reason_code="script_static_check_failed",
                    intake=intake,
                ),
                progress_callback=progress_callback,
                extra_artifacts=phase_artifacts,
            ), script

        last_result: SimulationResult | None = None
        total_attempts = self._settings.script_max_repairs + 1
        for local_attempt in range(1, total_attempts + 1):
            attempt_index = attempt_seed + local_attempt
            if progress_callback:
                progress_callback(34, f"正在执行第 {local_attempt} 轮 Maxwell 脚本")
            result = self._executor.run(
                requirement=requirement,
                intake=intake,
                script=script,
                run_dir=run_dir,
                attempt_index=attempt_index,
                progress_callback=progress_callback,
            )
            phase_artifacts.extend(result.artifacts)
            result.artifacts = self._dedupe_paths(phase_artifacts)

            if result.status in {"completed", "blocked"}:
                result.intake = intake
                return result, script

            last_result = result
            if not self._runtime_failure_is_repairable(result):
                last_result.intake = intake
                last_result.artifacts = self._dedupe_paths(phase_artifacts)
                return last_result, script
            if local_attempt >= total_attempts:
                break

            error_details = result.outputs.get("runtime_error") if result.outputs else result.message
            if progress_callback:
                progress_callback(40, "脚本执行报错，正在请求 AI 修复")
            script = self._llm.repair_script(
                requirement=requirement,
                intake=intake,
                script=script,
                failure_stage="runtime",
                error_details=str(error_details),
            )
            static_check, repair_artifacts, script = self._repair_until_static_check_passes(
                requirement=requirement,
                intake=intake,
                script=script,
                run_dir=run_dir,
                progress_callback=progress_callback,
                start_index=attempt_index + 1,
            )
            phase_artifacts.extend(repair_artifacts)
            if not static_check.passed:
                return self._build_blocked_result(
                    requirement=requirement,
                    run_dir=run_dir,
                    error=RequirementPlanningError(
                        "运行期修复后的脚本未通过静态检查。",
                        reason_code="script_static_check_failed",
                        intake=intake,
                    ),
                    progress_callback=progress_callback,
                    extra_artifacts=self._dedupe_paths(phase_artifacts),
                ), script

        if last_result is None:
            return self._build_blocked_result(
                requirement=requirement,
                run_dir=run_dir,
                error=RequirementPlanningError("未能产生可执行结果。", reason_code="execution_failed", intake=intake),
                progress_callback=progress_callback,
                extra_artifacts=self._dedupe_paths(phase_artifacts),
            ), script

        if intake.design:
            if progress_callback:
                progress_callback(42, "AI 多轮修复仍失败，切换到本地保底脚本")
            fallback_script = self._llm.build_local_fallback_script(intake)
            fallback_start_index = attempt_seed + self._settings.script_max_repairs + 2
            fallback_check, fallback_artifacts, fallback_script = self._repair_until_static_check_passes(
                requirement=requirement,
                intake=intake,
                script=fallback_script,
                run_dir=run_dir,
                progress_callback=progress_callback,
                start_index=fallback_start_index,
            )
            phase_artifacts.extend(fallback_artifacts)
            if fallback_check.passed:
                fallback_result = self._executor.run(
                    requirement=requirement,
                    intake=intake,
                    script=fallback_script,
                    run_dir=run_dir,
                    attempt_index=fallback_start_index,
                    progress_callback=progress_callback,
                )
                phase_artifacts.extend(fallback_result.artifacts)
                fallback_result.intake = intake
                fallback_result.artifacts = self._dedupe_paths(phase_artifacts)
                return fallback_result, fallback_script

        last_result.intake = intake
        last_result.artifacts = self._dedupe_paths(phase_artifacts)
        return last_result, script

    @staticmethod
    def _intake_is_executable(intake: RequirementIntake) -> bool:
        if intake.supported_now:
            return True
        spec_ready = bool(intake.simulation_spec.get("execution_ready"))
        plan_ready = bool(intake.execution_plan.get("execution_ready"))
        has_steps = isinstance(intake.execution_plan.get("steps"), list) and bool(intake.execution_plan.get("steps"))
        has_solver = bool(intake.execution_plan.get("design_type") or intake.execution_plan.get("solution_type"))
        if spec_ready or plan_ready or (has_steps and has_solver):
            return True
        if intake_has_generic_object_graph(intake) or infer_builder_hint(intake) != "unknown":
            has_any_spec = bool(intake.extracted_parameters) or bool(intake.simulation_spec) or bool(intake.execution_plan)
            return has_any_spec
        return False

    @staticmethod
    def _needs_feedback_iteration(result: SimulationResult) -> bool:
        if result.status != "completed" or result.evaluation is None:
            return False
        return result.evaluation.overall_status == "failed"

    @staticmethod
    def _runtime_failure_is_repairable(result: SimulationResult) -> bool:
        runtime_error = str((result.outputs or {}).get("runtime_error") or result.message or "").lower()
        if "waiting for maxwell to exit" in runtime_error:
            return False
        if "timed out after" in runtime_error and "maxwell" in runtime_error:
            return False
        return True

    def _persist_feedback_artifacts(
        self,
        run_dir: Path,
        requirement: str,
        intake: RequirementIntake,
        revised_design: ElectromagnetDesign | None,
        feedback_round: int,
    ) -> list[Path]:
        round_name = f"feedback_{feedback_round:02d}"
        intake_artifacts = self._persist_intake_artifacts(
            run_dir=run_dir,
            requirement=requirement,
            intake=intake,
            round_name=round_name,
        )
        if revised_design is None:
            return intake_artifacts
        design_path = run_dir / f"feedback_design_{feedback_round:02d}.json"
        design_path.write_text(
            json.dumps(revised_design.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return [design_path, *intake_artifacts]

    def smoke_llm(self) -> str:
        return self._llm.smoke_test()

    def list_models(self) -> list[str]:
        return self._llm.list_models()

    def _repair_until_static_check_passes(
        self,
        requirement: str,
        intake: RequirementIntake,
        script: GeneratedMaxwellScript,
        run_dir: Path,
        progress_callback: ProgressCallback | None = None,
        start_index: int = 1,
    ) -> tuple[ScriptStaticCheck, list[Path], GeneratedMaxwellScript]:
        artifacts: list[Path] = []
        current_script = script
        last_check = ScriptStaticCheck(passed=False)

        for iteration in range(start_index, start_index + self._settings.script_max_repairs + 1):
            last_check = static_check_generated_script(current_script)
            artifacts.extend(self._persist_script_artifacts(run_dir, current_script, last_check, iteration))
            if last_check.passed:
                return last_check, artifacts, current_script
            if iteration >= start_index + self._settings.script_max_repairs:
                break
            if progress_callback:
                progress_callback(30, "脚本静态检查未通过，正在请求 AI 修复")
            current_script = self._llm.repair_script(
                requirement=requirement,
                intake=intake,
                script=current_script,
                failure_stage="static_check",
                error_details=json.dumps(last_check.model_dump(mode="json"), ensure_ascii=False, indent=2),
            )
        return last_check, artifacts, current_script

    def _build_blocked_result(
        self,
        requirement: str,
        run_dir: Path,
        error: RequirementPlanningError,
        progress_callback: ProgressCallback | None = None,
        extra_artifacts: list[Path] | None = None,
    ) -> SimulationResult:
        environment = detect_maxwell_environment()
        intake = error.intake if isinstance(error.intake, RequirementIntake) else None
        evaluation = RequirementEvaluation(
            overall_status="failed",
            summary=error.message,
            checks=[
                RequirementCheck(
                    name="任务范围判定" if error.reason_code == "unsupported_requirement" else "脚本生成/校验",
                    status="failed",
                    detail=error.message,
                )
            ],
        )

        requirement_path = run_dir / "requirement.json"
        environment_path = run_dir / "environment.json"
        evaluation_path = run_dir / "evaluation.json"

        requirement_path.write_text(
            json.dumps(
                {
                    "requirement": requirement,
                    "status": "blocked",
                    "reason_code": error.reason_code,
                    "message": error.message,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        environment_path.write_text(
            json.dumps(environment.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        evaluation_path.write_text(
            json.dumps(evaluation.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        outputs: dict[str, float | str] = {}
        if intake:
            outputs["task_family"] = intake.task_family
            outputs["support_message"] = intake.support_message
            for key, value in self._flatten_mapping(intake.extracted_parameters, prefix="param"):
                outputs[key] = value
            for key, value in self._flatten_mapping(intake.simulation_spec, prefix="spec"):
                outputs[key] = value
            for key, value in self._flatten_mapping(intake.execution_plan, prefix="plan"):
                outputs[key] = value

        if progress_callback:
            progress_callback(100, error.message)

        artifacts = self._dedupe_paths(
            [*(extra_artifacts or []), requirement_path, environment_path, evaluation_path]
        )
        return SimulationResult(
            status="blocked",
            message=error.message,
            run_directory=run_dir,
            environment=environment,
            design=intake.design if intake else None,
            intake=intake,
            outputs=outputs,
            evaluation=evaluation,
            artifacts=artifacts,
        )

    def _persist_intake_artifacts(
        self,
        run_dir: Path,
        requirement: str,
        intake: RequirementIntake,
        round_name: str,
    ) -> list[Path]:
        requirement_path = run_dir / "requirement.json"
        intake_path = run_dir / f"intake_{round_name}.json"
        simulation_spec_path = run_dir / ("simulation_spec.json" if round_name == "final" else f"simulation_spec_{round_name}.json")
        execution_plan_path = run_dir / ("execution_plan.json" if round_name == "final" else f"execution_plan_{round_name}.json")

        if not requirement_path.exists():
            requirement_path.write_text(
                json.dumps({"requirement": requirement}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        intake_path.write_text(
            json.dumps(intake.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        simulation_spec_path.write_text(
            json.dumps(intake.simulation_spec, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        execution_plan_path.write_text(
            json.dumps(intake.execution_plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if round_name == "final":
            final_intake_path = run_dir / "intake.json"
            final_intake_path.write_text(
                json.dumps(intake.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return [requirement_path, intake_path, simulation_spec_path, execution_plan_path, final_intake_path]
        return [requirement_path, intake_path, simulation_spec_path, execution_plan_path]

    def _persist_script_artifacts(
        self,
        run_dir: Path,
        script: GeneratedMaxwellScript,
        check: ScriptStaticCheck,
        iteration: int,
    ) -> list[Path]:
        script_code_path = run_dir / f"script_draft_{iteration:02d}.py"
        script_json_path = run_dir / f"script_draft_{iteration:02d}.json"
        check_path = run_dir / f"script_static_check_{iteration:02d}.json"
        script_code_path.write_text(script.code, encoding="utf-8")
        script_json_path.write_text(
            json.dumps(script.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        check_path.write_text(
            json.dumps(check.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return [script_code_path, script_json_path, check_path]

    @staticmethod
    def _persist_json_artifact(path: Path, payload: object) -> Path:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def _create_run_directory(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_dir = self._settings.workspace_dir / stamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def _flatten_mapping(mapping: dict, prefix: str) -> list[tuple[str, float | str]]:
        rows: list[tuple[str, float | str]] = []

        def walk(value: object, path: list[str]) -> None:
            if value is None:
                return
            if isinstance(value, bool):
                rows.append((f"{prefix}_{'_'.join(path)}", str(value).lower()))
                return
            if isinstance(value, (str, float, int)):
                rows.append((f"{prefix}_{'_'.join(path)}", value))
                return
            if isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, [*path, str(index)])
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    walk(item, [*path, str(key)])

        for key, value in mapping.items():
            walk(value, [str(key)])
        return rows

    @staticmethod
    def _dedupe_paths(paths: list[Path]) -> list[Path]:
        unique: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = str(path.resolve()).lower()
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique
