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

        if completed.returncode == 0 and result_payload.get("status") == "completed" and not output_fatal:
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

        error_details = output_notes or self._collect_runtime_error(completed, result_payload)
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
        project_name = self._sanitize_project_name(
            str(intake.execution_plan.get("project_name") or intake.task_family or "generic_maxwell_design")
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
        if intake.task_family == "transformer_2d":
            return self._build_transformer_evaluation(intake, outputs=outputs, run_status=run_status)
        if intake.task_family == "inductor_2d":
            return self._build_inductor_evaluation(intake, outputs=outputs, run_status=run_status)
        outputs = outputs or {}
        checks: list[RequirementCheck] = []

        if run_status == "completed":
            checks.append(
                RequirementCheck(
                    name="通用脚本执行链路",
                    status="passed",
                    detail="当前任务已完成 Maxwell 建模、求解和结果导出。",
                )
            )
        else:
            checks.append(
                RequirementCheck(
                    name="通用脚本执行链路",
                    status="failed",
                    detail="当前任务未完成一次可验证的 Maxwell 执行。",
                )
            )

        requested_blob = json.dumps(intake.simulation_spec.get("required_outputs", []), ensure_ascii=False)
        requested_lower = requested_blob.lower()

        def _has_number(value: Any) -> bool:
            try:
                float(value)
                return True
            except (TypeError, ValueError):
                return False

        has_capacitance = _has_number(outputs.get("capacitance_f")) or _has_number(outputs.get("capacitance_pf"))
        has_field = _has_number(outputs.get("max_electric_field_note"))
        has_any_result = any(
            key in outputs
            for key in (
                "capacitance_f",
                "capacitance_pf",
                "max_electric_field_note",
                "max_flux_density_t",
                "force_n",
                "estimated_current_at_supply_a",
            )
        )

        wants_capacitance = "电容" in requested_blob or "capacit" in requested_lower
        wants_field = "电场" in requested_blob or "electric_field" in requested_lower

        if wants_capacitance:
            checks.append(
                RequirementCheck(
                    name="电容结果",
                    status="passed" if has_capacitance else ("failed" if run_status == "completed" else "unverified"),
                    detail=(
                        f"已提取电容结果，当前约为 {float(outputs.get('capacitance_pf')):.6g} pF。"
                        if has_capacitance and outputs.get("capacitance_pf") is not None
                        else "当前任务要求电容结果，但本轮未提取到可用电容数值。"
                    ),
                )
            )

        if wants_field:
            checks.append(
                RequirementCheck(
                    name="电场结果",
                    status="passed" if has_field else ("failed" if run_status == "completed" else "unverified"),
                    detail=(
                        f"已提取最大电场强度，当前约为 {float(outputs.get('max_electric_field_note')):.6g}。"
                        if has_field
                        else "当前任务要求电场结果，但本轮未提取到可用电场数值。"
                    ),
                )
            )

        if run_status == "completed" and not wants_capacitance and not wants_field:
            checks.append(
                RequirementCheck(
                    name="关键结果提取",
                    status="passed" if has_any_result else "unverified",
                    detail=(
                        "当前任务已提取到至少一项可展示的关键结果。"
                        if has_any_result
                        else "当前任务虽已完成执行，但还没有提取到明确的关键结果数值。"
                    ),
                )
            )

        if run_status != "completed":
            summary = "当前没有完成一次可验证的 Maxwell 执行。"
            status = "failed"
        elif all(item.status == "passed" for item in checks):
            summary = "当前任务已完成执行，且已验证的关键结果项均已给出。"
            status = "passed"
        elif any(item.status == "failed" for item in checks):
            summary = "当前任务已完成执行，但仍有关键结果项未成功提取。"
            status = "failed"
        elif any(item.status == "unverified" for item in checks):
            summary = "当前任务已完成执行，但部分结果项仍待进一步验证。"
            status = "partial"
        else:
            summary = "当前任务已完成执行。"
            status = "passed"

        return RequirementEvaluation(overall_status=status, summary=summary, checks=checks)

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
