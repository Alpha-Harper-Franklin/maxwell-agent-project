from __future__ import annotations

from collections.abc import Callable
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import Settings
from .evaluation import build_requirement_evaluation
from .maxwell_env import detect_maxwell_environment
from .models import (
    GeneratedMaxwellScript,
    RequirementCheck,
    RequirementEvaluation,
    RequirementIntake,
    SimulationResult,
)
from .semantics import infer_builder_hint


ProgressCallback = Callable[[int, str], None]


class MaxwellExecutor:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def run(
        self,
        requirement: str,
        intake: RequirementIntake,
        script: GeneratedMaxwellScript,
        run_dir: Path,
        attempt_index: int,
        progress_callback: ProgressCallback | None = None,
    ) -> SimulationResult:
        self._settings.ensure_dirs()
        self._notify_progress(progress_callback, 35, "正在检查本机 Maxwell 环境")
        env = detect_maxwell_environment()

        environment_path = run_dir / "environment.json"
        evaluation_path = run_dir / "evaluation.json"
        outputs_path = run_dir / "outputs.json"
        self._write_json(environment_path, env.model_dump(mode="json"))

        if intake.design:
            self._write_json(run_dir / "design.json", intake.design.model_dump(mode="json"))

        script_path = run_dir / f"generated_script_attempt_{attempt_index:02d}.py"
        script_metadata_path = run_dir / f"generated_script_attempt_{attempt_index:02d}.json"
        job_path = run_dir / f"script_job_attempt_{attempt_index:02d}.json"
        result_path = run_dir / f"script_result_attempt_{attempt_index:02d}.json"
        stdout_path = run_dir / f"script_stdout_attempt_{attempt_index:02d}.log"
        stderr_path = run_dir / f"script_stderr_attempt_{attempt_index:02d}.log"

        script_path.write_text(script.code, encoding="utf-8")
        self._write_json(script_metadata_path, script.model_dump(mode="json"))

        if not env.installed:
            evaluation = self._build_evaluation(intake, outputs={}, run_status="blocked")
            self._write_json(evaluation_path, evaluation.model_dump(mode="json"))
            self._notify_progress(progress_callback, 100, "未检测到 Maxwell，本次执行被阻塞")
            return SimulationResult(
                status="blocked",
                message="Maxwell/AEDT was not found on this machine. Script generation completed, execution is waiting for a local installation.",
                run_directory=run_dir,
                environment=env,
                design=intake.design,
                intake=intake,
                evaluation=evaluation,
                artifacts=[
                    script_metadata_path,
                    script_path,
                    environment_path,
                    evaluation_path,
                ],
            )

        job_payload = self._build_job_payload(requirement, intake, script, run_dir, env, attempt_index)
        self._write_json(job_path, job_payload)

        self._notify_progress(progress_callback, 48, "脚本已生成，正在启动 Maxwell 执行")
        completed = self._run_script_subprocess(
            script_path=script_path,
            job_path=job_path,
            result_path=result_path,
            env_model=env,
        )
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")

        result_payload = self._read_json_if_exists(result_path)
        outputs = {}
        if isinstance(result_payload, dict) and isinstance(result_payload.get("outputs"), dict):
            outputs = {
                str(key): value
                for key, value in result_payload["outputs"].items()
                if isinstance(value, (str, float, int, bool))
            }

        output_status = str(outputs.get("status") or "").strip().lower()
        output_notes = str(outputs.get("notes") or "").strip()
        output_fatal = output_status in {"fatal_error", "failed", "error"}
        runtime_failure = self._detect_runtime_failure(completed, outputs)

        if (
            completed.returncode == 0
            and result_payload.get("status") == "completed"
            and not output_fatal
            and not runtime_failure
        ):
            evaluation = self._build_evaluation(intake, outputs=outputs, run_status="completed")
            self._write_json(outputs_path, outputs)
            self._write_json(evaluation_path, evaluation.model_dump(mode="json"))
            self._notify_progress(progress_callback, 100, "脚本执行完成，结果已生成")
            return SimulationResult(
                status="completed",
                message="Maxwell execution completed.",
                run_directory=run_dir,
                project_file=Path(job_payload["project_file"]),
                environment=env,
                design=intake.design,
                intake=intake,
                outputs=outputs,
                evaluation=evaluation,
                artifacts=[
                    script_metadata_path,
                    script_path,
                    job_path,
                    result_path,
                    stdout_path,
                    stderr_path,
                    outputs_path,
                    evaluation_path,
                    environment_path,
                    Path(job_payload["project_file"]),
                ],
            )

        error_details = output_notes or runtime_failure or self._collect_runtime_error(completed, result_payload)
        evaluation = self._build_evaluation(intake, outputs={}, run_status="failed")
        self._write_json(evaluation_path, evaluation.model_dump(mode="json"))
        self._notify_progress(progress_callback, 100, "脚本执行失败，等待修复")
        return SimulationResult(
            status="failed",
            message=f"Maxwell script execution failed: {error_details}",
            run_directory=run_dir,
            project_file=Path(job_payload["project_file"]) if Path(job_payload["project_file"]).exists() else None,
            environment=env,
            design=intake.design,
            intake=intake,
            outputs={"runtime_error": error_details, **outputs},
            evaluation=evaluation,
            artifacts=[
                script_metadata_path,
                script_path,
                job_path,
                result_path,
                stdout_path,
                stderr_path,
                evaluation_path,
                environment_path,
            ],
        )

    def _build_job_payload(
        self,
        requirement: str,
        intake: RequirementIntake,
        script: GeneratedMaxwellScript,
        run_dir: Path,
        env_model,
        attempt_index: int,
    ) -> dict[str, Any]:
        semantic_hint = infer_builder_hint(intake, requirement=requirement)
        project_name = self._sanitize_project_name(
            str(intake.execution_plan.get("project_name") or semantic_hint or intake.task_family or "generic_maxwell_design")
        )
        project_file = run_dir / f"{project_name}_attempt_{attempt_index:02d}.aedt"
        return {
            "entrypoint": script.entrypoint,
            "requirement": requirement,
            "task_family": intake.task_family,
            "simulation_spec": intake.simulation_spec,
            "execution_plan": intake.execution_plan,
            "extracted_parameters": intake.extracted_parameters,
            "design": intake.design.model_dump(mode="json") if intake.design else None,
            "output_dir": str(run_dir),
            "project_file": str(project_file),
            "maxwell_version": self._settings.maxwell_version or env_model.version_hint,
            "non_graphical": self._settings.maxwell_non_graphical,
            "student_version": bool(env_model.student_version),
        }

    def _run_script_subprocess(
        self,
        script_path: Path,
        job_path: Path,
        result_path: Path,
        env_model,
    ) -> subprocess.CompletedProcess[str]:
        runner_path = self._settings.project_root / "src" / "maxwell_agent" / "script_runner.py"
        command = [
            str(self._settings.runtime_python),
            str(runner_path),
            str(script_path),
            str(job_path),
            str(result_path),
        ]
        process_env = self._build_subprocess_env(env_model)
        baseline_ansys_pids = self._list_ansys_process_ids()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=process_env,
            cwd=str(self._settings.project_root),
        )
        try:
            stdout, stderr = process.communicate(timeout=self._settings.script_execution_timeout_s)
            completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            stdout, stderr = process.communicate()
            self._terminate_process_ids(self._list_ansys_process_ids() - baseline_ansys_pids)
            timeout_message = (
                f"Script execution timed out after {self._settings.script_execution_timeout_s}s while waiting for Maxwell to exit."
            )
            if not result_path.exists():
                self._write_json(
                    result_path,
                    {
                        "status": "failed",
                        "error": timeout_message,
                        "traceback": stderr or "",
                    },
                )
            completed = subprocess.CompletedProcess(
                command,
                124,
                stdout or (exc.stdout or ""),
                stderr or (exc.stderr or timeout_message),
            )
        else:
            self._terminate_process_ids(self._list_ansys_process_ids() - baseline_ansys_pids)
        return completed

    @staticmethod
    def _list_ansys_process_ids() -> set[int]:
        process_names = {"ansysedtsv.exe", "ansysedtng.exe", "ansysedt.exe", "ansyscl.exe"}
        try:
            completed = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except Exception:
            return set()

        rows = csv.reader(completed.stdout.splitlines())
        pids: set[int] = set()
        for row in rows:
            if len(row) < 2:
                continue
            image = row[0].strip().lower()
            if image not in process_names:
                continue
            try:
                pids.add(int(row[1]))
            except ValueError:
                continue
        return pids

    @staticmethod
    def _terminate_process_ids(pids: set[int]) -> None:
        if not pids:
            return
        for pid in sorted(pids):
            try:
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
            except Exception:
                continue
        time.sleep(1.0)

    def _build_evaluation(
        self,
        intake: RequirementIntake,
        outputs: dict[str, float | str] | None,
        run_status: str,
    ) -> RequirementEvaluation:
        if intake.design:
            return build_requirement_evaluation(intake.design, outputs=outputs, run_status=run_status)
        semantic_hint = infer_builder_hint(intake, requirement="")
        if semantic_hint == "capacitor_2d":
            return self._build_capacitor_evaluation(intake, outputs=outputs, run_status=run_status)
        if semantic_hint == "transformer_2d":
            return self._build_transformer_evaluation(intake, outputs=outputs, run_status=run_status)
        if semantic_hint == "inductor_2d":
            return self._build_inductor_evaluation(intake, outputs=outputs, run_status=run_status)
        if semantic_hint == "busbar_2d":
            return self._build_busbar_evaluation(intake, outputs=outputs, run_status=run_status)
        if semantic_hint == "solenoid_2d":
            return self._build_solenoid_evaluation(intake, outputs=outputs, run_status=run_status)
        if semantic_hint == "coaxial_capacitor_2d":
            return self._build_coaxial_capacitor_evaluation(intake, outputs=outputs, run_status=run_status)
        return self._build_generic_task_evaluation(intake, outputs=outputs, run_status=run_status)

    def _build_generic_task_evaluation(
        self,
        intake: RequirementIntake,
        outputs: dict[str, float | str] | None,
        run_status: str,
    ) -> RequirementEvaluation:
        outputs = outputs or {}
        constraints = intake.simulation_spec.get("constraints", {}) if isinstance(intake.simulation_spec, dict) else {}
        if not isinstance(constraints, dict):
            constraints = {}
        required_outputs = intake.simulation_spec.get("required_outputs", []) if isinstance(intake.simulation_spec, dict) else []
        requested_blob = json.dumps(required_outputs, ensure_ascii=False)
        requested_lower = requested_blob.lower()
        required_names_lower: list[str] = []
        if isinstance(required_outputs, list):
            for item in required_outputs:
                if isinstance(item, dict):
                    name = str(item.get("name") or "").strip().lower()
                    if name:
                        required_names_lower.append(name)
                elif isinstance(item, str) and item.strip():
                    required_names_lower.append(item.strip().lower())

        capacitance_pf = self._as_float(outputs.get("capacitance_pf"))
        if capacitance_pf is None:
            capacitance_f = self._as_float(outputs.get("capacitance_f"))
            if capacitance_f is not None:
                capacitance_pf = capacitance_f * 1e12
        if capacitance_pf is None:
            capacitance_pf = self._as_float(outputs.get("capacitance_per_unit_length_pf_per_m"))
        if capacitance_pf is None:
            capacitance_per_length_f = self._as_float(outputs.get("capacitance_per_unit_length_f_per_m"))
            if capacitance_per_length_f is None:
                capacitance_per_length_f = self._as_float(outputs.get("capacitance_per_unit_length_f"))
            if capacitance_per_length_f is None:
                capacitance_per_length_f = self._as_float(outputs.get("mutual_capacitance_c12_f_per_m"))
            if capacitance_per_length_f is None:
                capacitance_per_length_f = self._as_float(outputs.get("self_capacitance_c11_f_per_m"))
            if capacitance_per_length_f is not None:
                capacitance_pf = capacitance_per_length_f * 1e12
        field_v_per_m = self._as_float(outputs.get("max_electric_field_v_per_m"))
        if field_v_per_m is None:
            field_v_per_m = self._as_float(outputs.get("max_electric_field_note"))
        if field_v_per_m is None:
            field_v_per_m = self._as_float(outputs.get("reference_average_field_v_per_m"))
        flux_density = self._as_float(outputs.get("max_flux_density_t"))
        if flux_density is None:
            flux_density = self._as_float(outputs.get("flux_density_t"))
        if flux_density is None:
            flux_density = self._as_float(outputs.get("bmax_global_t"))
        if flux_density is None:
            flux_density = self._as_float(outputs.get("bmax_conductor_t"))
        current_density = self._as_float(outputs.get("estimated_current_density_a_per_mm2"))
        if current_density is None:
            current_density = self._as_float(outputs.get("max_current_density_a_per_mm2"))
        if current_density is None:
            current_density = self._as_float(outputs.get("avg_current_density_a_per_mm2"))
        if current_density is None:
            current_density = self._as_float(outputs.get("nominal_j_a_per_mm2"))
        current_a = self._as_float(outputs.get("current_a"))
        turns_ratio = self._as_float(outputs.get("turns_ratio"))
        voltage_result = self._as_float(outputs.get("estimated_secondary_voltage_v"))
        if voltage_result is None:
            voltage_result = self._as_float(outputs.get("applied_voltage_v"))

        checks = [
            RequirementCheck(
                name="通用脚本执行链路",
                status="passed" if run_status == "completed" else "failed",
                detail=(
                    "当前任务已完成 Maxwell 建模、求解和结果返回。"
                    if run_status == "completed"
                    else "当前任务未完成一次可验证的 Maxwell 执行。"
                ),
            )
        ]

        wants_capacitance = "电容" in requested_blob or "capacit" in requested_lower
        wants_field = "电场" in requested_blob or "electric_field" in requested_lower
        wants_flux = "磁密" in requested_blob or "磁场" in requested_blob or any(
            token in requested_lower for token in ("flux_density", "magnetic_field", "bmax")
        )
        wants_current_density = "电流密度" in requested_blob or "current_density" in requested_lower
        wants_turns_ratio = "匝比" in requested_blob or "turns_ratio" in requested_lower
        wants_voltage = any(
            token in name
            for name in required_names_lower
            for token in ("secondary_voltage", "output_voltage", "voltage_result", "voltage")
        )

        if wants_capacitance:
            checks.append(
                RequirementCheck(
                    name="电容结果",
                    status="passed" if capacitance_pf is not None else ("failed" if run_status == "completed" else "unverified"),
                    detail=(
                        f"已提取 Maxwell 电容结果，约 {capacitance_pf:.6g} pF。"
                        if capacitance_pf is not None
                        else "当前任务要求电容结果，但本轮未提取到可用电容数值。"
                    ),
                )
            )
        if wants_field:
            checks.append(
                RequirementCheck(
                    name="电场结果",
                    status="passed" if field_v_per_m is not None else ("failed" if run_status == "completed" else "unverified"),
                    detail=(
                        f"已提取电场结果，约 {field_v_per_m:.6g} V/m。"
                        if field_v_per_m is not None
                        else "当前任务要求电场结果，但本轮未提取到可用电场数值。"
                    ),
                )
            )
        if wants_flux:
            checks.append(
                RequirementCheck(
                    name="磁场结果",
                    status="passed" if flux_density is not None else ("failed" if run_status == "completed" else "unverified"),
                    detail=(
                        f"已提取磁密结果，约 {flux_density:.6g} T。"
                        if flux_density is not None
                        else "当前任务要求磁场结果，但本轮未提取到可用磁密数值。"
                    ),
                )
            )
        if wants_current_density:
            checks.append(
                RequirementCheck(
                    name="电流密度结果",
                    status="passed" if current_density is not None else ("failed" if run_status == "completed" else "unverified"),
                    detail=(
                        f"已提取电流密度结果，约 {current_density:.6g} A/mm^2。"
                        if current_density is not None
                        else "当前任务要求电流密度结果，但本轮未提取到可用电流密度数值。"
                    ),
                )
            )
        if wants_turns_ratio:
            checks.append(
                RequirementCheck(
                    name="匝比结果",
                    status="passed" if turns_ratio is not None else ("failed" if run_status == "completed" else "unverified"),
                    detail=(
                        f"已提取匝比结果，约 {turns_ratio:.6g}。"
                        if turns_ratio is not None
                        else "当前任务要求匝比结果，但本轮未提取到可用匝比数值。"
                    ),
                )
            )
        if wants_voltage:
            checks.append(
                RequirementCheck(
                    name="电压结果",
                    status="passed" if voltage_result is not None else ("failed" if run_status == "completed" else "unverified"),
                    detail=(
                        f"已提取电压结果，约 {voltage_result:.6g} V。"
                        if voltage_result is not None
                        else "当前任务要求电压结果，但本轮未提取到可用电压数值。"
                    ),
                )
            )

        target_capacitance_f = self._as_float(constraints.get("target_capacitance_f"))
        max_field_limit_v_per_m = self._as_float(constraints.get("max_electric_field_v_per_m"))
        max_flux_limit_t = self._as_float(constraints.get("max_flux_density_t"))
        max_current_density = self._as_float(constraints.get("max_current_density_a_per_mm2"))
        if max_current_density is None:
            max_current_density = self._as_float(constraints.get("j_limit_a_per_mm2"))
        required_current_a = self._as_float(constraints.get("required_current_a"))
        current_min_a = self._as_float(constraints.get("current_min_a"))
        current_limit_a = self._as_float(constraints.get("current_limit_a"))

        if target_capacitance_f is not None:
            target_pf = target_capacitance_f * 1e12
            cap_ok = capacitance_pf is not None and capacitance_pf + 1e-9 >= target_pf
            checks.append(
                RequirementCheck(
                    name="电容目标",
                    status="passed" if cap_ok else ("failed" if run_status == "completed" else "unverified"),
                    detail=(
                        f"电容约 {capacitance_pf:.6g} pF，达到至少 {target_pf:.6g} pF 的要求。"
                        if cap_ok
                        else (
                            f"电容约 {capacitance_pf:.6g} pF，未达到至少 {target_pf:.6g} pF 的要求。"
                            if capacitance_pf is not None
                            else f"用户要求电容至少 {target_pf:.6g} pF，但当前没有可验证数值。"
                        )
                    ),
                )
            )
        if max_field_limit_v_per_m is not None:
            field_ok = field_v_per_m is not None and field_v_per_m <= max_field_limit_v_per_m + 1e-9
            checks.append(
                RequirementCheck(
                    name="电场上限",
                    status="passed" if field_ok else ("failed" if run_status == "completed" else "unverified"),
                    detail=(
                        f"电场约 {field_v_per_m:.6g} V/m，满足不超过 {max_field_limit_v_per_m:.6g} V/m。"
                        if field_ok
                        else (
                            f"电场约 {field_v_per_m:.6g} V/m，已超过 {max_field_limit_v_per_m:.6g} V/m。"
                            if field_v_per_m is not None
                            else f"用户要求电场不超过 {max_field_limit_v_per_m:.6g} V/m，但当前没有可验证数值。"
                        )
                    ),
                )
            )
        if max_flux_limit_t is not None:
            flux_ok = flux_density is not None and flux_density <= max_flux_limit_t + 1e-9
            checks.append(
                RequirementCheck(
                    name="磁密上限",
                    status="passed" if flux_ok else ("failed" if run_status == "completed" else "unverified"),
                    detail=(
                        f"最大磁密约 {flux_density:.6g} T，满足不超过 {max_flux_limit_t:.6g} T。"
                        if flux_ok
                        else (
                            f"最大磁密约 {flux_density:.6g} T，已超过 {max_flux_limit_t:.6g} T。"
                            if flux_density is not None
                            else f"用户要求磁密不超过 {max_flux_limit_t:.6g} T，但当前没有可验证数值。"
                        )
                    ),
                )
            )
        if max_current_density is not None:
            density_ok = current_density is not None and current_density <= max_current_density + 1e-9
            checks.append(
                RequirementCheck(
                    name="电流密度上限",
                    status="passed" if density_ok else ("failed" if run_status == "completed" else "unverified"),
                    detail=(
                        f"电流密度约 {current_density:.6g} A/mm^2，满足不超过 {max_current_density:.6g} A/mm^2。"
                        if density_ok
                        else (
                            f"电流密度约 {current_density:.6g} A/mm^2，已超过 {max_current_density:.6g} A/mm^2。"
                            if current_density is not None
                            else f"用户要求电流密度不超过 {max_current_density:.6g} A/mm^2，但当前没有可验证数值。"
                        )
                    ),
                )
            )
        if required_current_a is not None and current_a is not None:
            current_ok = abs(current_a - required_current_a) <= 1e-6
            checks.append(
                RequirementCheck(
                    name="指定载流电流",
                    status="passed" if current_ok else "failed",
                    detail=(
                        f"用户指定载流电流 {required_current_a:.6g} A，本轮模型按 {current_a:.6g} A 执行。"
                        if current_ok
                        else f"用户指定载流电流 {required_current_a:.6g} A，但本轮模型改成了 {current_a:.6g} A。"
                    ),
                )
            )
        elif current_min_a is not None and current_limit_a is not None and current_a is not None:
            current_ok = current_min_a - 1e-9 <= current_a <= current_limit_a + 1e-9
            checks.append(
                RequirementCheck(
                    name="载流电流范围",
                    status="passed" if current_ok else "failed",
                    detail=(
                        f"用户要求电流 {current_min_a:.6g}-{current_limit_a:.6g} A，本轮模型按 {current_a:.6g} A 执行。"
                        if current_ok
                        else f"用户要求电流 {current_min_a:.6g}-{current_limit_a:.6g} A，但本轮模型按 {current_a:.6g} A 执行。"
                    ),
                )
            )
        elif current_limit_a is not None and current_a is not None:
            current_ok = current_a <= current_limit_a + 1e-9
            checks.append(
                RequirementCheck(
                    name="载流电流上限",
                    status="passed" if current_ok else "failed",
                    detail=(
                        f"用户要求电流不超过 {current_limit_a:.6g} A，本轮模型按 {current_a:.6g} A 执行。"
                        if current_ok
                        else f"用户要求电流不超过 {current_limit_a:.6g} A，但本轮模型按 {current_a:.6g} A 执行。"
                    ),
                )
            )

        if run_status == "completed" and len(checks) == 1:
            has_any_result = any(
                key in outputs
                for key in (
                    "capacitance_f",
                    "capacitance_pf",
                    "max_electric_field_v_per_m",
                    "max_electric_field_note",
                    "max_flux_density_t",
                    "flux_density_t",
                    "estimated_current_density_a_per_mm2",
                    "max_current_density_a_per_mm2",
                    "turns_ratio",
                    "estimated_secondary_voltage_v",
                )
            )
            checks.append(
                RequirementCheck(
                    name="关键结果提取",
                    status="passed" if has_any_result else "unverified",
                    detail=(
                        "当前任务已提取到至少一项可展示的关键结果。"
                        if has_any_result
                        else "当前任务虽然完成执行，但还没有提取到明确的关键结果数值。"
                    ),
                )
            )
        return self._finalize_generic_evaluation(run_status, checks)

    def _build_transformer_evaluation(
        self,
        intake: RequirementIntake,
        outputs: dict[str, float | str] | None,
        run_status: str,
    ) -> RequirementEvaluation:
        outputs = outputs or {}
        constraints = intake.simulation_spec.get("constraints", {}) if isinstance(intake.simulation_spec, dict) else {}
        target_output_v = self._as_float(constraints.get("output_voltage_v"))
        estimated_output_v = self._as_float(outputs.get("estimated_secondary_voltage_v"))
        turns_ratio = self._as_float(outputs.get("turns_ratio"))
        flux_density = self._as_float(outputs.get("max_flux_density_t"))
        checks = [
            RequirementCheck(
                name="通用脚本执行链路",
                status="passed" if run_status == "completed" else "failed",
                detail="当前 transformer_2d 任务已完成 Maxwell 执行。" if run_status == "completed" else "当前 transformer_2d 任务未完成 Maxwell 执行。",
            )
        ]
        if target_output_v is not None:
            checks.append(
                RequirementCheck(
                    name="次级电压目标",
                    status="passed" if estimated_output_v is not None and abs(estimated_output_v - target_output_v) <= max(1.0, abs(target_output_v) * 0.1) else "failed",
                    detail=(
                        f"估算次级电压约 {estimated_output_v:.4g} V，目标为 {target_output_v:.4g} V。"
                        if estimated_output_v is not None
                        else f"当前任务目标为 {target_output_v:.4g} V，但尚未提取到次级电压估算结果。"
                    ),
                )
            )
        if turns_ratio is not None:
            checks.append(RequirementCheck(name="匝比结果", status="passed", detail=f"当前已提取匝比结果，约为 {turns_ratio:.6g}。"))
        if flux_density is not None:
            checks.append(RequirementCheck(name="磁密结果", status="passed", detail=f"当前已提取最大磁密，约为 {flux_density:.6g} T。"))
        return self._finalize_generic_evaluation(run_status, checks)

    def _build_capacitor_evaluation(
        self,
        intake: RequirementIntake,
        outputs: dict[str, float | str] | None,
        run_status: str,
    ) -> RequirementEvaluation:
        outputs = outputs or {}
        constraints = intake.simulation_spec.get("constraints", {}) if isinstance(intake.simulation_spec, dict) else {}
        capacitance_pf = self._as_float(outputs.get("capacitance_pf"))
        if capacitance_pf is None:
            capacitance_f = self._as_float(outputs.get("capacitance_f"))
            if capacitance_f is not None:
                capacitance_pf = capacitance_f * 1e12
        field_v_per_m = self._as_float(outputs.get("max_electric_field_v_per_m"))
        if field_v_per_m is None:
            field_v_per_m = self._as_float(outputs.get("max_electric_field_note"))
        reference_field_v_per_m = self._as_float(outputs.get("reference_average_field_v_per_m"))
        target_capacitance_f = self._as_float(constraints.get("target_capacitance_f"))
        max_field_limit_v_per_m = self._as_float(constraints.get("max_electric_field_v_per_m"))

        checks = [
            RequirementCheck(
                name="通用脚本执行链路",
                status="passed" if run_status == "completed" else "failed",
                detail=(
                    "当前 capacitor_2d 任务已完成 Maxwell 建模、求解和结果返回。"
                    if run_status == "completed"
                    else "当前 capacitor_2d 任务未完成 Maxwell 执行。"
                ),
            )
        ]
        if capacitance_pf is not None:
            checks.append(
                RequirementCheck(
                    name="电容结果",
                    status="passed",
                    detail=f"已提取 Maxwell 电容结果，约 {capacitance_pf:.6g} pF。",
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="电容结果",
                    status="failed" if run_status == "completed" else "unverified",
                    detail="当前未提取到可用的电容数值。",
                )
            )
        if target_capacitance_f is not None:
            target_capacitance_pf = target_capacitance_f * 1e12
            if capacitance_pf is None:
                checks.append(
                    RequirementCheck(
                        name="电容目标",
                        status="failed" if run_status == "completed" else "unverified",
                        detail=f"用户要求电容至少 {target_capacitance_pf:.6g} pF，但当前没有可验证的 Maxwell 电容数值。",
                    )
                )
            else:
                cap_ok = capacitance_pf + 1e-9 >= target_capacitance_pf
                checks.append(
                    RequirementCheck(
                        name="电容目标",
                        status="passed" if cap_ok else "failed",
                        detail=(
                            f"电容约 {capacitance_pf:.6g} pF，达到至少 {target_capacitance_pf:.6g} pF 的要求。"
                            if cap_ok
                            else f"电容约 {capacitance_pf:.6g} pF，未达到至少 {target_capacitance_pf:.6g} pF 的要求。"
                        ),
                    )
                )
        if field_v_per_m is not None and field_v_per_m > 0:
            checks.append(
                RequirementCheck(
                    name="电场结果",
                    status="passed",
                    detail=f"已提取 Maxwell 电场结果，约 {field_v_per_m:.6g} V/m。",
                )
            )
        elif reference_field_v_per_m is not None:
            checks.append(
                RequirementCheck(
                    name="电场结果",
                    status="unverified",
                    detail=f"当前只有参考电场估算值，约 {reference_field_v_per_m:.6g} V/m，尚未提取到 Maxwell 电场后处理结果。",
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="电场结果",
                    status="failed" if run_status == "completed" else "unverified",
                    detail="当前未提取到可用的电场数值。",
                )
            )
        if max_field_limit_v_per_m is not None:
            candidate_field = field_v_per_m if field_v_per_m is not None else reference_field_v_per_m
            if candidate_field is None:
                checks.append(
                    RequirementCheck(
                        name="电场上限",
                        status="failed" if run_status == "completed" else "unverified",
                        detail=f"用户要求电场不超过 {max_field_limit_v_per_m:.6g} V/m，但当前没有可验证数值。",
                    )
                )
            else:
                field_ok = candidate_field <= max_field_limit_v_per_m + 1e-9
                checks.append(
                    RequirementCheck(
                        name="电场上限",
                        status="passed" if field_ok else "failed",
                        detail=(
                            f"电场约 {candidate_field:.6g} V/m，满足不超过 {max_field_limit_v_per_m:.6g} V/m。"
                            if field_ok
                            else f"电场约 {candidate_field:.6g} V/m，已超过 {max_field_limit_v_per_m:.6g} V/m。"
                        ),
                    )
                )
        return self._finalize_generic_evaluation(run_status, checks)

    def _build_inductor_evaluation(
        self,
        intake: RequirementIntake,
        outputs: dict[str, float | str] | None,
        run_status: str,
    ) -> RequirementEvaluation:
        outputs = outputs or {}
        constraints = intake.simulation_spec.get("constraints", {}) if isinstance(intake.simulation_spec, dict) else {}
        target_inductance_h = self._as_float(constraints.get("target_inductance_h"))
        estimated_inductance_h = self._as_float(outputs.get("estimated_inductance_h"))
        flux_density = self._as_float(outputs.get("max_flux_density_t"))
        checks = [
            RequirementCheck(
                name="通用脚本执行链路",
                status="passed" if run_status == "completed" else "failed",
                detail="当前 inductor_2d 任务已完成 Maxwell 执行。" if run_status == "completed" else "当前 inductor_2d 任务未完成 Maxwell 执行。",
            )
        ]
        if target_inductance_h is not None:
            checks.append(
                RequirementCheck(
                    name="电感目标",
                    status="passed" if estimated_inductance_h is not None and abs(estimated_inductance_h - target_inductance_h) <= max(1e-6, abs(target_inductance_h) * 0.2) else "failed",
                    detail=(
                        f"估算电感约 {estimated_inductance_h:.6g} H，目标为 {target_inductance_h:.6g} H。"
                        if estimated_inductance_h is not None
                        else f"当前任务目标为 {target_inductance_h:.6g} H，但尚未提取到电感结果。"
                    ),
                )
            )
        elif estimated_inductance_h is not None:
            checks.append(RequirementCheck(name="电感结果", status="passed", detail=f"当前已提取电感结果，约为 {estimated_inductance_h:.6g} H。"))
        if flux_density is not None:
            checks.append(RequirementCheck(name="磁密结果", status="passed", detail=f"当前已提取最大磁密，约为 {flux_density:.6g} T。"))
        return self._finalize_generic_evaluation(run_status, checks)

    def _build_busbar_evaluation(
        self,
        intake: RequirementIntake,
        outputs: dict[str, float | str] | None,
        run_status: str,
    ) -> RequirementEvaluation:
        outputs = outputs or {}
        constraints = intake.simulation_spec.get("constraints", {}) if isinstance(intake.simulation_spec, dict) else {}
        flux_density = self._as_float(outputs.get("max_flux_density_t"))
        current_density = self._as_float(outputs.get("estimated_current_density_a_per_mm2"))
        if current_density is None:
            current_density = self._as_float(outputs.get("max_current_density_a_per_mm2"))
        if current_density is None:
            current_density = self._as_float(outputs.get("nominal_j_a_per_mm2"))
        if current_density is None:
            current_density = self._as_float(outputs.get("avg_current_density_a_per_mm2"))
        current_a = self._as_float(outputs.get("current_a"))
        area_mm2 = self._as_float(outputs.get("cross_section_area_mm2"))
        if area_mm2 is None:
            area_mm2 = self._as_float(outputs.get("area_mm2"))
        required_current_a = self._as_float(constraints.get("required_current_a"))
        current_min_a = self._as_float(constraints.get("current_min_a"))
        current_limit_a = self._as_float(constraints.get("current_limit_a"))
        max_current_density = self._as_float(constraints.get("max_current_density_a_per_mm2"))
        if max_current_density is None:
            max_current_density = self._as_float(constraints.get("j_limit_a_per_mm2"))
        max_flux_density = self._as_float(constraints.get("max_flux_density_t"))

        checks = [
            RequirementCheck(
                name="\u901a\u7528\u811a\u672c\u6267\u884c\u94fe\u8def",
                status="passed" if run_status == "completed" else "failed",
                detail=(
                    "\u5f53\u524d busbar_2d \u4efb\u52a1\u5df2\u5b8c\u6210 Maxwell \u5efa\u6a21\u3001\u6c42\u89e3\u548c\u7ed3\u679c\u8fd4\u56de\u3002"
                    if run_status == "completed"
                    else "\u5f53\u524d busbar_2d \u4efb\u52a1\u672a\u5b8c\u6210 Maxwell \u6267\u884c\u3002"
                ),
            )
        ]
        if flux_density is not None:
            checks.append(
                RequirementCheck(
                    name="\u78c1\u573a\u7ed3\u679c",
                    status="passed",
                    detail=f"\u5df2\u63d0\u53d6 Maxwell \u540e\u5904\u7406\u6700\u5927\u78c1\u5bc6\uff0c\u7ea6 {flux_density:.6g} T\u3002",
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="\u78c1\u573a\u7ed3\u679c",
                    status="failed" if run_status == "completed" else "unverified",
                    detail="\u5f53\u524d\u672a\u83b7\u53d6\u5230\u53ef\u7528\u7684\u6bcd\u6392\u78c1\u573a\u7ed3\u679c\u3002",
                )
            )
        if current_density is not None:
            detail = f"\u5df2\u6839\u636e\u622a\u9762\u79ef\u4f30\u7b97\u7535\u6d41\u5bc6\u5ea6\uff0c\u7ea6 {current_density:.6g} A/mm^2\u3002"
            if current_a is not None and area_mm2 is not None:
                detail = f"\u7535\u6d41 {current_a:.6g} A\uff0c\u622a\u9762 {area_mm2:.6g} mm^2\uff0c\u4f30\u7b97\u7535\u6d41\u5bc6\u5ea6 {current_density:.6g} A/mm^2\u3002"
            checks.append(
                RequirementCheck(
                    name="\u7535\u6d41\u5bc6\u5ea6\u7ed3\u679c",
                    status="passed",
                    detail=detail,
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="\u7535\u6d41\u5bc6\u5ea6\u7ed3\u679c",
                    status="failed" if run_status == "completed" else "unverified",
                    detail="\u5f53\u524d\u672a\u83b7\u53d6\u5230\u53ef\u7528\u7684\u7535\u6d41\u5bc6\u5ea6\u7ed3\u679c\u3002",
                )
            )

        if required_current_a is not None and current_a is not None:
            current_ok = abs(current_a - required_current_a) <= 1e-6
            checks.append(
                RequirementCheck(
                    name="\u6307\u5b9a\u8f7d\u6d41\u7535\u6d41",
                    status="passed" if current_ok else "failed",
                    detail=(
                        f"\u7528\u6237\u6307\u5b9a\u8f7d\u6d41\u7535\u6d41 {required_current_a:.6g} A\uff0c\u672c\u8f6e\u6a21\u578b\u6309 {current_a:.6g} A \u6267\u884c\u3002"
                        if current_ok
                        else f"\u7528\u6237\u6307\u5b9a\u8f7d\u6d41\u7535\u6d41 {required_current_a:.6g} A\uff0c\u4f46\u672c\u8f6e\u6a21\u578b\u6539\u6210\u4e86 {current_a:.6g} A\u3002"
                    ),
                )
            )
        elif current_min_a is not None and current_limit_a is not None and current_a is not None:
            current_ok = current_min_a - 1e-9 <= current_a <= current_limit_a + 1e-9
            checks.append(
                RequirementCheck(
                    name="\u8f7d\u6d41\u7535\u6d41\u8303\u56f4",
                    status="passed" if current_ok else "failed",
                    detail=(
                        f"\u7528\u6237\u8981\u6c42\u7535\u6d41 {current_min_a:.6g}-{current_limit_a:.6g} A\uff0c\u672c\u8f6e\u6a21\u578b\u6309 {current_a:.6g} A \u6267\u884c\u3002"
                        if current_ok
                        else f"\u7528\u6237\u8981\u6c42\u7535\u6d41 {current_min_a:.6g}-{current_limit_a:.6g} A\uff0c\u4f46\u672c\u8f6e\u6a21\u578b\u6309 {current_a:.6g} A \u6267\u884c\u3002"
                    ),
                )
            )
        elif current_limit_a is not None and current_a is not None:
            current_ok = current_a <= current_limit_a + 1e-9
            checks.append(
                RequirementCheck(
                    name="\u8f7d\u6d41\u7535\u6d41\u4e0a\u9650",
                    status="passed" if current_ok else "failed",
                    detail=(
                        f"\u7528\u6237\u8981\u6c42\u7535\u6d41\u4e0d\u8d85\u8fc7 {current_limit_a:.6g} A\uff0c\u672c\u8f6e\u6a21\u578b\u6309 {current_a:.6g} A \u6267\u884c\u3002"
                        if current_ok
                        else f"\u7528\u6237\u8981\u6c42\u7535\u6d41\u4e0d\u8d85\u8fc7 {current_limit_a:.6g} A\uff0c\u4f46\u672c\u8f6e\u6a21\u578b\u6309 {current_a:.6g} A \u6267\u884c\u3002"
                    ),
                )
            )

        if max_current_density is not None:
            if current_density is None:
                checks.append(
                    RequirementCheck(
                        name="\u7535\u6d41\u5bc6\u5ea6\u4e0a\u9650",
                        status="failed" if run_status == "completed" else "unverified",
                        detail=f"\u7528\u6237\u8981\u6c42\u7535\u6d41\u5bc6\u5ea6\u4e0d\u8d85\u8fc7 {max_current_density:.6g} A/mm^2\uff0c\u4f46\u5f53\u524d\u6ca1\u6709\u53ef\u9a8c\u8bc1\u6570\u503c\u3002",
                    )
                )
            else:
                density_ok = current_density <= max_current_density + 1e-9
                checks.append(
                    RequirementCheck(
                        name="\u7535\u6d41\u5bc6\u5ea6\u4e0a\u9650",
                        status="passed" if density_ok else "failed",
                        detail=(
                            f"\u7535\u6d41\u5bc6\u5ea6\u7ea6 {current_density:.6g} A/mm^2\uff0c\u6ee1\u8db3\u4e0d\u8d85\u8fc7 {max_current_density:.6g} A/mm^2\u3002"
                            if density_ok
                            else f"\u7535\u6d41\u5bc6\u5ea6\u7ea6 {current_density:.6g} A/mm^2\uff0c\u5df2\u8d85\u8fc7 {max_current_density:.6g} A/mm^2\u3002"
                        ),
                    )
                )

        if max_flux_density is not None:
            if flux_density is None:
                checks.append(
                    RequirementCheck(
                        name="\u78c1\u5bc6\u4e0a\u9650",
                        status="failed" if run_status == "completed" else "unverified",
                        detail=f"\u7528\u6237\u8981\u6c42\u78c1\u5bc6\u4e0d\u8d85\u8fc7 {max_flux_density:.6g} T\uff0c\u4f46\u5f53\u524d\u6ca1\u6709\u53ef\u9a8c\u8bc1\u6570\u503c\u3002",
                    )
                )
            else:
                flux_ok = flux_density <= max_flux_density + 1e-9
                checks.append(
                    RequirementCheck(
                        name="\u78c1\u5bc6\u4e0a\u9650",
                        status="passed" if flux_ok else "failed",
                        detail=(
                            f"\u6700\u5927\u78c1\u5bc6\u7ea6 {flux_density:.6g} T\uff0c\u6ee1\u8db3\u4e0d\u8d85\u8fc7 {max_flux_density:.6g} T\u3002"
                            if flux_ok
                            else f"\u6700\u5927\u78c1\u5bc6\u7ea6 {flux_density:.6g} T\uff0c\u5df2\u8d85\u8fc7 {max_flux_density:.6g} T\u3002"
                        ),
                    )
                )
        return self._finalize_generic_evaluation(run_status, checks)

    def _build_solenoid_evaluation(
        self,
        intake: RequirementIntake,
        outputs: dict[str, float | str] | None,
        run_status: str,
    ) -> RequirementEvaluation:
        outputs = outputs or {}
        center_flux_density = self._as_float(outputs.get("estimated_center_flux_density_t"))
        max_flux_density = self._as_float(outputs.get("max_flux_density_t"))
        equivalent_current = self._as_float(outputs.get("equivalent_current_a"))
        current_a = self._as_float(outputs.get("current_a"))
        coil_turns = self._as_float(outputs.get("coil_turns"))

        checks = [
            RequirementCheck(
                name="\u901a\u7528\u811a\u672c\u6267\u884c\u94fe\u8def",
                status="passed" if run_status == "completed" else "failed",
                detail=(
                    "\u5f53\u524d solenoid_2d \u4efb\u52a1\u5df2\u5b8c\u6210 Maxwell \u5efa\u6a21\u3001\u6c42\u89e3\u548c\u7ed3\u679c\u8fd4\u56de\u3002"
                    if run_status == "completed"
                    else "\u5f53\u524d solenoid_2d \u4efb\u52a1\u672a\u5b8c\u6210 Maxwell \u6267\u884c\u3002"
                ),
            )
        ]
        if center_flux_density is not None or max_flux_density is not None:
            detail_parts = []
            if center_flux_density is not None:
                detail_parts.append(f"\u4e2d\u5fc3\u78c1\u5bc6\u53c2\u8003\u503c\u7ea6 {center_flux_density:.6g} T")
            if max_flux_density is not None:
                detail_parts.append(f"Maxwell \u540e\u5904\u7406\u6700\u5927\u78c1\u5bc6\u7ea6 {max_flux_density:.6g} T")
            checks.append(
                RequirementCheck(
                    name="\u78c1\u573a\u7ed3\u679c",
                    status="passed",
                    detail="\uff1b".join(detail_parts) + "\u3002",
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="\u78c1\u573a\u7ed3\u679c",
                    status="failed" if run_status == "completed" else "unverified",
                    detail="\u4efb\u52a1\u8981\u6c42\u8f93\u51fa\u87ba\u7ebf\u7ba1\u78c1\u573a\uff0c\u4f46\u672c\u8f6e\u672a\u83b7\u53d6\u5230\u53ef\u7528\u78c1\u573a\u6570\u503c\u3002",
                )
            )
        if equivalent_current is not None:
            checks.append(
                RequirementCheck(
                    name="\u7b49\u6548\u5b89\u531d\u6570",
                    status="passed",
                    detail=f"\u7535\u6d41 {current_a or 0.0:.6g} A\uff0c\u531d\u6570 {coil_turns or 0.0:.6g}\uff0c\u7b49\u6548\u5b89\u531d\u6570 {equivalent_current:.6g} A-turn\u3002",
                )
            )
        return self._finalize_generic_evaluation(run_status, checks)

    def _build_coaxial_capacitor_evaluation(
        self,
        intake: RequirementIntake,
        outputs: dict[str, float | str] | None,
        run_status: str,
    ) -> RequirementEvaluation:
        outputs = outputs or {}
        constraints = intake.simulation_spec.get("constraints", {}) if isinstance(intake.simulation_spec, dict) else {}
        capacitance_pf = self._as_float(outputs.get("capacitance_pf"))
        reference_capacitance_pf = self._as_float(outputs.get("reference_capacitance_pf_per_m"))
        field_v_per_m = self._as_float(outputs.get("max_electric_field_v_per_m"))
        if field_v_per_m is None:
            field_v_per_m = self._as_float(outputs.get("max_electric_field_note"))
        reference_field_v_per_m = self._as_float(outputs.get("reference_average_field_v_per_m"))
        target_capacitance_f = self._as_float(constraints.get("target_capacitance_f"))
        max_field_limit_v_per_m = self._as_float(constraints.get("max_electric_field_v_per_m"))

        checks = [
            RequirementCheck(
                name="\u901a\u7528\u811a\u672c\u6267\u884c\u94fe\u8def",
                status="passed" if run_status == "completed" else "failed",
                detail=(
                    "\u5f53\u524d coaxial_capacitor_2d \u4efb\u52a1\u5df2\u5b8c\u6210 Maxwell \u5efa\u6a21\u3001\u6c42\u89e3\u548c\u7ed3\u679c\u8fd4\u56de\u3002"
                    if run_status == "completed"
                    else "\u5f53\u524d coaxial_capacitor_2d \u4efb\u52a1\u672a\u5b8c\u6210 Maxwell \u6267\u884c\u3002"
                ),
            )
        ]
        if capacitance_pf is not None:
            checks.append(
                RequirementCheck(
                    name="\u7535\u5bb9\u7ed3\u679c",
                    status="passed",
                    detail=f"\u5df2\u4ece Maxwell \u77e9\u9635\u7ed3\u679c\u63d0\u53d6\u7535\u5bb9\uff0c\u7ea6 {capacitance_pf:.6g} pF\u3002",
                )
            )
        elif reference_capacitance_pf is not None:
            checks.append(
                RequirementCheck(
                    name="\u7535\u5bb9\u7ed3\u679c",
                    status="unverified",
                    detail=f"\u5f53\u524d\u53ea\u6709\u89e3\u6790\u53c2\u8003\u7535\u5bb9\uff0c\u7ea6 {reference_capacitance_pf:.6g} pF/m\uff0c\u5c1a\u672a\u63d0\u53d6\u5230 Maxwell \u77e9\u9635\u6570\u503c\u3002",
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="\u7535\u5bb9\u7ed3\u679c",
                    status="failed" if run_status == "completed" else "unverified",
                    detail="\u4efb\u52a1\u8981\u6c42\u7535\u5bb9\u7ed3\u679c\uff0c\u4f46\u672c\u8f6e\u672a\u83b7\u53d6\u5230\u53ef\u7528\u7535\u5bb9\u6570\u503c\u3002",
                )
            )
        if target_capacitance_f is not None:
            target_capacitance_pf = target_capacitance_f * 1e12
            if capacitance_pf is None:
                checks.append(
                    RequirementCheck(
                        name="\u7535\u5bb9\u76ee\u6807",
                        status="failed" if run_status == "completed" else "unverified",
                        detail=f"\u7528\u6237\u8981\u6c42\u7535\u5bb9\u81f3\u5c11 {target_capacitance_pf:.6g} pF\uff0c\u4f46\u5f53\u524d\u6ca1\u6709\u53ef\u9a8c\u8bc1\u7684 Maxwell \u7535\u5bb9\u6570\u503c\u3002",
                    )
                )
            else:
                cap_ok = capacitance_pf + 1e-9 >= target_capacitance_pf
                checks.append(
                    RequirementCheck(
                        name="\u7535\u5bb9\u76ee\u6807",
                        status="passed" if cap_ok else "failed",
                        detail=(
                            f"\u7535\u5bb9\u7ea6 {capacitance_pf:.6g} pF\uff0c\u8fbe\u5230\u81f3\u5c11 {target_capacitance_pf:.6g} pF \u7684\u8981\u6c42\u3002"
                            if cap_ok
                            else f"\u7535\u5bb9\u7ea6 {capacitance_pf:.6g} pF\uff0c\u672a\u8fbe\u5230\u81f3\u5c11 {target_capacitance_pf:.6g} pF \u7684\u8981\u6c42\u3002"
                        ),
                    )
                )
        if field_v_per_m is not None and field_v_per_m > 0:
            checks.append(
                RequirementCheck(
                    name="\u7535\u573a\u7ed3\u679c",
                    status="passed",
                    detail=f"\u5df2\u63d0\u53d6 Maxwell \u540e\u5904\u7406\u7535\u573a\u503c\uff0c\u7ea6 {field_v_per_m:.6g} V/m\u3002",
                )
            )
        elif reference_field_v_per_m is not None:
            checks.append(
                RequirementCheck(
                    name="\u7535\u573a\u7ed3\u679c",
                    status="unverified",
                    detail=f"\u5f53\u524d\u53ea\u6709\u51e0\u4f55/\u7535\u538b\u53c2\u8003\u7535\u573a\uff0c\u7ea6 {reference_field_v_per_m:.6g} V/m\uff0c\u5c1a\u672a\u63d0\u53d6\u5230\u53ef\u9760\u7684 Maxwell \u7535\u573a\u540e\u5904\u7406\u6570\u503c\u3002",
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="\u7535\u573a\u7ed3\u679c",
                    status="failed" if run_status == "completed" else "unverified",
                    detail="\u4efb\u52a1\u8981\u6c42\u7535\u573a\u7ed3\u679c\uff0c\u4f46\u672c\u8f6e\u672a\u83b7\u53d6\u5230\u53ef\u7528\u7535\u573a\u6570\u503c\u3002",
                )
            )
        if max_field_limit_v_per_m is not None:
            candidate_field = field_v_per_m if field_v_per_m is not None else reference_field_v_per_m
            if candidate_field is None:
                checks.append(
                    RequirementCheck(
                        name="\u7535\u573a\u4e0a\u9650",
                        status="failed" if run_status == "completed" else "unverified",
                        detail=f"\u7528\u6237\u8981\u6c42\u7535\u573a\u4e0d\u8d85\u8fc7 {max_field_limit_v_per_m:.6g} V/m\uff0c\u4f46\u5f53\u524d\u6ca1\u6709\u53ef\u9a8c\u8bc1\u6570\u503c\u3002",
                    )
                )
            else:
                field_ok = candidate_field <= max_field_limit_v_per_m + 1e-9
                checks.append(
                    RequirementCheck(
                        name="\u7535\u573a\u4e0a\u9650",
                        status="passed" if field_ok else "failed",
                        detail=(
                            f"\u7535\u573a\u7ea6 {candidate_field:.6g} V/m\uff0c\u6ee1\u8db3\u4e0d\u8d85\u8fc7 {max_field_limit_v_per_m:.6g} V/m\u3002"
                            if field_ok
                            else f"\u7535\u573a\u7ea6 {candidate_field:.6g} V/m\uff0c\u5df2\u8d85\u8fc7 {max_field_limit_v_per_m:.6g} V/m\u3002"
                        ),
                    )
                )
        return self._finalize_generic_evaluation(run_status, checks)

    @staticmethod
    def _finalize_generic_evaluation(run_status: str, checks: list[RequirementCheck]) -> RequirementEvaluation:
        if run_status != "completed":
            return RequirementEvaluation(overall_status="failed", summary="当前没有完成一次可验证的 Maxwell 执行。", checks=checks)
        statuses = {item.status for item in checks}
        if "failed" in statuses:
            return RequirementEvaluation(overall_status="failed", summary="当前任务已完成执行，但仍有关键约束或结果项未满足。", checks=checks)
        if "unverified" in statuses:
            return RequirementEvaluation(overall_status="partial", summary="当前任务已完成执行，但仍有部分结果项待进一步验证。", checks=checks)
        return RequirementEvaluation(overall_status="passed", summary="当前任务已完成执行，且当前已验证的需求项均满足。", checks=checks)

    @staticmethod
    def _as_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _build_subprocess_env(self, env_model) -> dict[str, str]:
        process_env = dict(os.environ)
        process_env["PYTHONUTF8"] = "1"
        process_env["PYTHONIOENCODING"] = "utf-8"
        if env_model.executable:
            executable = Path(env_model.executable)
            version_suffix = self._resolve_env_suffix(env_model.version_hint, executable)
            em_root = executable.parent
            install_root = em_root.parent
            if version_suffix:
                process_env.setdefault(f"ANSYSEM_ROOT{version_suffix}", str(em_root))
                process_env.setdefault(f"AWP_ROOT{version_suffix}", str(install_root))
                if env_model.student_version:
                    process_env.setdefault(f"ANSYSEMSV_ROOT{version_suffix}", str(em_root))
        return process_env

    @staticmethod
    def _resolve_env_suffix(version_hint: str | None, executable: Path) -> str | None:
        if version_hint:
            match = re.fullmatch(r"(\d{4})\.(\d)", version_hint)
            if match:
                return f"{match.group(1)[2:]}{match.group(2)}"

        for part in executable.parts:
            match = re.fullmatch(r"[vV](\d{3})", part)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _resolve_material_name(name: str, fallback: str) -> str:
        material = (name or "").strip().lower()
        aliases = {
            "copper": "copper",
            "soft_iron": "soft_iron",
            "low_carbon_steel": "steel_1008",
            "silicon_steel": "steel_1008",
            "steel_1008": "steel_1008",
            "vacuum": "vacuum",
        }
        return aliases.get(material, name or fallback)

    @staticmethod
    def _read_json_if_exists(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _detect_runtime_failure(
        completed: subprocess.CompletedProcess[str],
        outputs: dict[str, float | str],
    ) -> str | None:
        failure_markers = {
            "error in solving setup": "PyAEDT reported that the Maxwell solve failed.",
            "solutions are empty": "AEDT did not produce usable solution data.",
            "maxwell solve failed": "The generated script reported a Maxwell solve failure.",
        }
        for item in (
            completed.stderr,
            completed.stdout,
            outputs.get("notes"),
            outputs.get("matrix_export_note"),
            outputs.get("max_electric_field_note"),
            outputs.get("max_flux_density_note"),
        ):
            text = str(item or "").strip()
            if not text:
                continue
            lowered = text.lower()
            for marker, message in failure_markers.items():
                if marker in lowered:
                    return f"{message} Details: {text}"
        return None

    @staticmethod
    def _collect_runtime_error(completed: subprocess.CompletedProcess[str], result_payload: dict[str, Any]) -> str:
        candidates = [
            result_payload.get("traceback"),
            result_payload.get("error"),
            completed.stderr,
            completed.stdout,
            f"Process exited with code {completed.returncode}",
        ]
        for item in candidates:
            text = str(item or "").strip()
            if text:
                return text
        return "Unknown runtime error."

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _notify_progress(
        callback: ProgressCallback | None,
        percent: int,
        message: str,
    ) -> None:
        if callback:
            callback(percent, message)

    @staticmethod
    def _sanitize_project_name(name: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
        return clean or "generic_maxwell_design"
