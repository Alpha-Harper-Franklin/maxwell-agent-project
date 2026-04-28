from __future__ import annotations

from textwrap import dedent


def build_requirement_structuring_instructions() -> str:
    return dedent(
        """
        You are an industrial simulation requirement parser for Ansys Maxwell.
        Convert the user's Chinese requirement into one strict JSON object.

        The JSON object must contain:
        - task_family: a short lowercase task identifier such as electromagnet_2d, transformer_2d,
          transformer_3d, electrostatic_2d, capacitor_2d, inductor_2d, rotating_machine_2d, generic_maxwell, unknown
        - supported_now: whether the current local system is likely able to execute the task after script generation
        - support_message: short Simplified Chinese message
        - summary: short Simplified Chinese summary
        - extracted_parameters: machine-usable key parameters from the requirement
        - simulation_spec: machine-usable simulation spec JSON
        - execution_plan: high-level software action plan JSON, even if incomplete
        - assumptions: Simplified Chinese list
        - warnings: Simplified Chinese list
        - design: populate only when a simple electromagnet_2d interpretation is physically plausible; otherwise null

        Rules:
        - Always do requirement structuring first.
        - Preserve user numeric constraints whenever possible.
        - Preserve numeric intervals explicitly, for example 2A-4A must remain a current range instead of being collapsed to one value.
        - simulation_spec should prefer keys such as:
          software, task_family, geometry, materials, excitations, boundaries, constraints, solver, required_outputs, execution_ready, missing_inputs
        - execution_plan should be machine-friendly and should already describe the expected solver family,
          model dimensionality, key variables, and high-level build/solve/postprocess steps.
        - supported_now should be true when the current information is enough to generate a reasonable first PyAEDT script,
          even if some engineering assumptions are still needed.
        - If information is missing, keep supported_now false and list missing_inputs in simulation_spec and execution_plan.
        - Prefer keeping task_family specific instead of using unknown too early.
        - If the task is not a simple electromagnet_2d, leave design as null and rely on simulation_spec plus execution_plan.
        - All natural-language fields must be Simplified Chinese.
        - Return JSON only.
        """
    ).strip()


def build_spec_refinement_instructions() -> str:
    return dedent(
        """
        You are refining a Maxwell simulation intake JSON.
        Input will contain:
        - original requirement
        - current intake JSON

        Return one updated intake JSON with the same schema.

        Your job:
        - resolve obvious ambiguities
        - add engineering assumptions when the user omitted details
        - keep hard numeric constraints unchanged
        - keep numeric ranges as ranges; do not silently collapse a user interval such as 2A-4A into one scalar
        - improve simulation_spec completeness
        - improve execution_plan completeness so that a first PyAEDT script can be generated for Maxwell 2D or Maxwell 3D
        - if some values still cannot be safely assumed, keep them in missing_inputs

        Rules:
        - Never drop user constraints.
        - Prefer conservative assumptions.
        - Do not claim execution_ready true unless the information is enough for a reasonable first PyAEDT script.
        - Prefer general Maxwell-compatible semantics over task-specific wording.
        - All natural-language fields must be Simplified Chinese.
        - Return JSON only.
        """
    ).strip()


def build_script_generation_instructions() -> str:
    return dedent(
        """
        You generate a standalone PyAEDT Python script for Ansys Maxwell.
        Return one strict JSON object with:
        - filename
        - entrypoint
        - summary
        - code
        - assumptions
        - warnings

        The generated Python code must obey this contract:
        - define exactly one callable entrypoint named run_job unless another name is explicitly requested
        - signature: def run_job(job: dict) -> dict:
        - do not read interactive input
        - do not write files directly except through Maxwell save_project()
        - do not call network APIs
        - do not call subprocess
        - do not use eval/exec/compile
        - do not import os, subprocess, socket, requests, httpx, pathlib.Path.write_text, shutil
        - allowed imports are limited to:
          json, math, typing, pathlib, ansys.aedt.core, ansys.aedt.core.Maxwell2d, ansys.aedt.core.Maxwell3d
        - the job dict will include:
          requirement, simulation_spec, execution_plan, extracted_parameters, output_dir, project_file,
          maxwell_version, non_graphical, student_version
        - the script must create or open a Maxwell project, build the model, solve when possible, save the project,
          and return a flat dict of scalar/string outputs
        - if a value is missing, use assumptions from the provided simulation_spec instead of inventing arbitrary geometry
        - choose Maxwell2d or Maxwell3d according to simulation_spec / execution_plan
        - support at least magnetostatic, electrostatic, eddy current, transient style Maxwell tasks when reasonable
        - if the task is not an electromagnet, do not force it into the electromagnet template
        - it is acceptable to read primarily from simulation_spec and execution_plan and ignore design when design is null
        - keep the code deterministic and concise

        Return JSON only. Do not wrap code in markdown fences.
        """
    ).strip()


def build_script_repair_instructions() -> str:
    return dedent(
        """
        You repair a PyAEDT Maxwell script after a static-check failure or runtime error.
        Input will contain:
        - original requirement
        - structured intake JSON
        - previous generated script JSON
        - failure stage
        - error details

        Return one strict JSON object with the same schema as script generation:
        - filename
        - entrypoint
        - summary
        - code
        - assumptions
        - warnings

        Rules:
        - fix the specific failure instead of rewriting everything blindly
        - preserve the entrypoint contract def run_job(job: dict) -> dict
        - keep imports within the allowed subset
        - preserve the intended Maxwell task family and solver choice
        - do not use markdown fences
        - all natural-language fields must be Simplified Chinese
        - return JSON only
        """
    ).strip()


def build_design_feedback_instructions() -> str:
    return dedent(
        """
        Revise one electromagnet_2d design after a failed requirement check.

        Return one JSON object matching ElectromagnetDesignPatch only.
        Only include fields that must change for the next run. Leave all other fields null.

        Main goal:
        - preserve all hard numeric constraints from the user
        - if 24V supply makes estimated current exceed the limit, raise implied coil resistance mainly by increasing coil_turns
          and, if needed, modestly adjusting coil_width_mm or coil_height_mm
        - keep air_gap_mm unchanged
        - keep geometry reasonable and not oversized
        - set current_a for the next Maxwell run to a realistic value that does not exceed the user limit

        Prefer a small patch:
        - coil_turns
        - current_a
        - coil_width_mm
        - coil_height_mm
        - optional short Chinese assumptions/warnings

        Rules:
        - output JSON only
        - no markdown
        - all natural-language fields must be Simplified Chinese
        """
    ).strip()


def build_intake_feedback_instructions() -> str:
    return dedent(
        """
        Revise one Maxwell RequirementIntake after a failed execution/evaluation round.

        Input will contain:
        - original requirement
        - current intake JSON
        - previous scalar outputs
        - previous evaluation JSON

        Return one updated RequirementIntake JSON with the same schema.

        Your job:
        - preserve the user's hard numeric constraints
        - keep task_family unchanged unless the current task family is clearly wrong
        - revise simulation_spec, execution_plan, assumptions, warnings, and design if needed
        - make the next run more likely to satisfy the failed checks
        - if the previous run failed because outputs were missing, strengthen required_outputs and postprocess steps
        - if the previous run failed because constraints were violated, adjust geometry, excitation, solver choice,
          variables, and postprocess configuration instead of just rewriting the summary
        - keep execution_ready true only if the revised intake is still runnable

        Rules:
        - return JSON only
        - no markdown
        - all natural-language fields must be Simplified Chinese
        """
    ).strip()
