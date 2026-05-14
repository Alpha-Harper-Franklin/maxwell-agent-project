from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


Objective = Literal["maximize_force", "balance_force_and_size", "maximize_inductance"]
SizePreference = Literal["compact", "balanced", "performance"]
OutputMetric = Literal["flux_density", "force", "inductance", "current_density"]
RunStatus = Literal["completed", "blocked", "failed"]
RequirementCheckStatus = Literal["passed", "failed", "unverified"]
RequirementOverallStatus = Literal["passed", "failed", "partial", "unverified"]
TaskFamily = str


class ElectromagnetDesign(BaseModel):
    problem_type: Literal["electromagnet_2d"] = "electromagnet_2d"
    source_requirement: str = ""
    summary: str = Field(
        default="Generated electromagnet_2d design from natural-language requirement.",
        description="Short natural-language summary of the interpreted design goal.",
    )
    objective: Objective = "maximize_force"
    size_preference: SizePreference = "balanced"
    supply_voltage_v: float | None = Field(default=None, ge=0.0, le=1000.0)
    current_min_a: float | None = Field(default=None, gt=0.0, le=100.0)
    current_limit_a: float | None = Field(default=None, gt=0.0, le=100.0)
    target_force_n: float | None = Field(default=None, ge=0.0, le=100000.0)
    air_gap_mm: float = Field(default=2.0, ge=0.1, le=20.0)
    current_a: float = Field(default=1.0, gt=0.0, le=100.0)
    coil_turns: int = Field(default=400, ge=10, le=10000)
    core_width_mm: float = Field(default=20.0, ge=5.0, le=300.0)
    core_height_mm: float = Field(default=40.0, ge=5.0, le=500.0)
    core_thickness_mm: float = Field(default=10.0, ge=2.0, le=100.0)
    coil_width_mm: float = Field(default=12.0, ge=1.0, le=100.0)
    coil_height_mm: float = Field(default=20.0, ge=1.0, le=300.0)
    region_padding_mm: float = Field(default=20.0, ge=5.0, le=200.0)
    core_material: str = "steel_1008"
    coil_material: str = "copper"
    required_outputs: list[OutputMetric] = Field(
        default_factory=lambda: ["flux_density", "force", "inductance"]
    )
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def variable_expressions(self) -> dict[str, str]:
        return {
            "air_gap": f"{self.air_gap_mm}mm",
            "core_width": f"{self.core_width_mm}mm",
            "core_height": f"{self.core_height_mm}mm",
            "core_thickness": f"{self.core_thickness_mm}mm",
            "coil_width": f"{self.coil_width_mm}mm",
            "coil_height": f"{self.coil_height_mm}mm",
            "region_padding": f"{self.region_padding_mm}mm",
            "coil_turns": str(self.coil_turns),
            "current_amp": f"{self.current_a}A",
        }


class ElectromagnetDesignPatch(BaseModel):
    summary: str | None = None
    current_min_a: float | None = Field(default=None, gt=0.0, le=100.0)
    current_a: float | None = Field(default=None, gt=0.0, le=100.0)
    coil_turns: int | None = Field(default=None, ge=10, le=10000)
    core_width_mm: float | None = Field(default=None, ge=5.0, le=300.0)
    core_height_mm: float | None = Field(default=None, ge=5.0, le=500.0)
    core_thickness_mm: float | None = Field(default=None, ge=2.0, le=100.0)
    coil_width_mm: float | None = Field(default=None, ge=1.0, le=100.0)
    coil_height_mm: float | None = Field(default=None, ge=1.0, le=300.0)
    region_padding_mm: float | None = Field(default=None, ge=5.0, le=200.0)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RequirementIntake(BaseModel):
    task_family: TaskFamily = "unknown"
    supported_now: bool = False
    support_message: str = "当前任务暂未落到可执行模板。"
    summary: str = Field(
        default="已完成需求结构化。",
        description="Short natural-language summary of the interpreted requirement.",
    )
    extracted_parameters: dict[str, Any] = Field(default_factory=dict)
    simulation_spec: dict[str, Any] = Field(default_factory=dict)
    execution_plan: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    design: ElectromagnetDesign | None = None


class GeneratedMaxwellScript(BaseModel):
    filename: str = "generated_maxwell_job.py"
    entrypoint: str = "run_job"
    summary: str = Field(
        default="AI generated a PyAEDT script for the current requirement.",
        description="Short summary of what the generated script is expected to do.",
    )
    code: str = ""
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    primitive_library_updates: list[dict[str, Any]] = Field(default_factory=list)


class ScriptStaticCheck(BaseModel):
    passed: bool = False
    required_entrypoint: str = "run_job"
    imported_modules: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MaxwellEnvironment(BaseModel):
    installed: bool
    executable: str | None = None
    version_hint: str | None = None
    student_version: bool = False
    pyaedt_importable: bool = False
    candidates: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RequirementCheck(BaseModel):
    name: str
    status: RequirementCheckStatus
    detail: str


class RequirementEvaluation(BaseModel):
    overall_status: RequirementOverallStatus
    summary: str
    checks: list[RequirementCheck] = Field(default_factory=list)


class CaseInsight(BaseModel):
    physics_type: str = "unknown"
    helper_label: str = "unknown"
    builder_hint: str = "unknown"
    capability_items: list[dict[str, Any]] = Field(default_factory=list)
    geometry_objects: list[dict[str, str]] = Field(default_factory=list)
    constraint_items: list[dict[str, str]] = Field(default_factory=list)
    output_items: list[dict[str, str]] = Field(default_factory=list)
    residual_items: list[dict[str, Any]] = Field(default_factory=list)
    knowledge_items: list[str] = Field(default_factory=list)
    engineering_explanations: list[str] = Field(default_factory=list)


class IterationRecord(BaseModel):
    index: int
    stage: str = "execute"
    status: RunStatus | str = "failed"
    evaluation_status: RequirementOverallStatus | str = "unverified"
    message: str = ""
    feedback_required: bool = False
    feedback_reason: str = ""
    design_snapshot: dict[str, Any] = Field(default_factory=dict)
    output_snapshot: dict[str, Any] = Field(default_factory=dict)
    failed_checks: list[str] = Field(default_factory=list)
    passed_checks: list[str] = Field(default_factory=list)
    unverified_checks: list[str] = Field(default_factory=list)
    residual_items: list[dict[str, Any]] = Field(default_factory=list)
    ir_patch_summary: str = ""
    insight: CaseInsight = Field(default_factory=CaseInsight)


class CaseDeliveryReport(BaseModel):
    requirement: str
    generated_at: str
    run_directory: Path
    project_file: Path | None = None
    final_status: RunStatus | str
    final_evaluation_status: RequirementOverallStatus | str = "unverified"
    final_summary: str = ""
    final_outputs: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    insight: CaseInsight = Field(default_factory=CaseInsight)
    iterations: list[IterationRecord] = Field(default_factory=list)
    artifact_paths: list[Path] = Field(default_factory=list)


class SimulationResult(BaseModel):
    status: RunStatus
    message: str
    run_directory: Path
    project_file: Path | None = None
    environment: MaxwellEnvironment
    design: ElectromagnetDesign | None = None
    intake: RequirementIntake | None = None
    outputs: dict[str, float | str] = Field(default_factory=dict)
    evaluation: RequirementEvaluation | None = None
    artifacts: list[Path] = Field(default_factory=list)
    iterations: list[IterationRecord] = Field(default_factory=list)
    insight: CaseInsight | None = None
    delivery_report: CaseDeliveryReport | None = None
