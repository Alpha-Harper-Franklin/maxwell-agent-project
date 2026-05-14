from __future__ import annotations

from textwrap import dedent


def build_requirement_structuring_instructions() -> str:
    return dedent(
        """
        You are an industrial simulation requirement parser for Ansys Maxwell.
        Convert the user's Chinese requirement into one strict JSON object.

        The JSON object must contain:
        - task_family: an optional short lowercase helper label such as electromagnet_2d, transformer_2d,
          electrostatic_2d, capacitor_2d, coaxial_capacitor_2d, inductor_2d, solenoid_2d, busbar_2d,
          generic_maxwell, unknown; this is only a semantic hint, not the only execution key
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
        - Determine physics_type first, then constraint semantics, then output semantics, and only then decide whether a helper task_family label is useful.
        - If the geometry is not one of the known named families but can still be represented by 2D rectangles, circles, and subtract operations, prefer task_family=generic_maxwell instead of forcing it into a wrong known family.
        - Examples that should often remain generic_maxwell: two separated round conductors, annular current-carrying conductors, two-strip electrostatic cross-sections, custom frame/window cross-sections.
        - simulation_spec should prefer keys such as:
          software, physics_type, task_family, geometry, materials, excitations, boundaries, constraints, solver, required_outputs, execution_ready, missing_inputs
        - execution_plan should be machine-friendly and should already describe the expected solver family,
          model dimensionality, key variables, and high-level build/solve/postprocess steps.
        - supported_now should be true when the current information is enough to generate a reasonable first PyAEDT script,
          even if some engineering assumptions are still needed.
        - If information is missing, keep supported_now false and list missing_inputs in simulation_spec and execution_plan.
        - Prefer correct physics_type plus correct constraints/outputs over forcing a specific task_family too early.
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
        - Keep physics_type, hard constraints, and required_outputs explicit even if task_family remains unknown.
        - If the requirement can be approximated with 2D rectangles/circles/subtract, it is acceptable to keep task_family=generic_maxwell and mark execution_ready true.
        - All natural-language fields must be Simplified Chinese.
        - Return JSON only.
        """
    ).strip()


def build_ir_generation_instructions() -> str:
    return dedent(
        """
        You generate a machine-usable Maxwell IR plan instead of Python code.
        Return one strict JSON object with:
        - summary
        - ir_plan
        - assumptions
        - warnings

        The ir_plan object must follow this structure:
        - driver: Maxwell2d or Maxwell3d
        - design_name
        - solution_type
        - model_units
        - setup_name
        - setup_type
        - parameters: list of parameter bindings from geometry/excitations/boundaries/constraints/design
        - locals: list of derived local expressions
        - objects: use only rectangle, circle, region
        - operations: use only subtract
        - assignments: use only current, voltage, balloon, matrix
        - derived_outputs
        - postprocess: use only field_scalar, matrix_export_value
        - failure_note

        Rules:
        - Preserve the user's hard numeric constraints whenever possible.
        - Prefer a runnable first-pass Maxwell model over overcomplicated geometry.
        - When the requirement is broad or partially missing data, make conservative engineering assumptions and state them.
        - The current local renderer supports only a Maxwell 2D executable IR subset. Always output driver=Maxwell2d and a 2D approximation that can actually run now.
        - Prefer Maxwell 2D when a 2D cross-section can reasonably validate the core idea. If the real device is 3D, keep the unmodeled 3D effects in warnings instead of switching to Maxwell3d.
        - Do not output Python code.
        - Only use the supported IR primitives listed above.
        - Use the exact schema field names required by the local renderer. Do not invent alternative keys such as type, unit, category, corner, size, magnitude, target, method, stat unless they also exist in the exact schema.
        - If the user's wording implies a geometry not directly supported, approximate it using combinations of rectangles/circles and subtract operations.
        - For arbitrary two-conductor electrostatic cross-sections, prefer two conductor objects + voltage assignments + one matrix assignment + field_scalar/matrix_export_value postprocess.
        - For annular or hollow conductors, prefer outer shape minus inner shape, then assign current or voltage to the resulting conductor.
        - For multi-conductor magnetic cross-sections, prefer one object per conductor and use separate current assignments instead of collapsing them into a single bar.
        - All internal identifiers inside ir_plan should use short ASCII snake_case names so they can be rendered directly into Python variables.
        - The output must be execution-oriented: it should be renderable into a PyAEDT script without any manual editing.
        - All natural-language fields must be Simplified Chinese.
        - Return JSON only.

        Minimal style example:
        {
          "summary": "二维磁静态铜排模型",
          "ir_plan": {
            "driver": "Maxwell2d",
            "design_name": "busbar_demo",
            "solution_type": "Magnetostatic",
            "setup_name": "Setup1",
            "setup_type": "Magnetostatic",
            "parameters": [
              {"name": "width_mm", "source": "geometry", "field": "width_mm", "default": 10, "cast": "float"},
              {"name": "current_a", "source": "excitations", "field": "current_a", "default": 200, "cast": "float"}
            ],
            "locals": [],
            "objects": [
              {"name": "bar", "kind": "rectangle", "material": "copper", "origin_exprs": ["-width_mm/2", "-1", "0"], "sizes_exprs": ["width_mm", "2"]},
              {"name": "region", "kind": "region", "pad_value_exprs": ["20", "20", "20", "20"], "pad_type": "Absolute Offset"}
            ],
            "operations": [],
            "assignments": [
              {"name": "drive", "kind": "current", "targets": ["bar"], "amplitude_expr": "current_a"},
              {"name": "outer_region", "kind": "balloon", "targets": ["region"], "boundary_name": "outer_region"}
            ],
            "derived_outputs": [
              {"output_key": "current_a", "expression": "current_a", "cast": "float", "phase": "before_solve"}
            ],
            "postprocess": [
              {"kind": "field_scalar", "output_key": "max_flux_density_t", "cast": "float", "quantity": "Mag_B", "scalar_function": "Maximum", "object_name": "AllObjects"}
            ],
            "failure_note": "Maxwell solve failed."
          },
          "assumptions": ["按二维截面处理"],
          "warnings": []
        }
        """
    ).strip()


def build_ir_repair_instructions() -> str:
    return dedent(
        """
        You repair a Maxwell IR plan after static validation failure or runtime execution failure.
        Input will contain:
        - original requirement
        - current intake JSON
        - previous IR artifact JSON
        - failure stage
        - error details

        Return one strict JSON object with:
        - summary
        - ir_plan
        - assumptions
        - warnings

        Repair rules:
        - Fix the specific invalid geometry / bad reference / unsupported primitive / bad postprocess / solver mismatch that caused the failure.
        - Preserve the user's hard numeric constraints.
        - Keep the plan within the supported IR primitive set:
          rectangle, circle, region, subtract, current, voltage, balloon, matrix, field_scalar, matrix_export_value.
        - Keep driver=Maxwell2d and stay within the current 2D executable subset.
        - Use the exact schema field names required by the local renderer; do not replace them with alternative names.
        - Keep internal identifiers in ASCII snake_case.
        - Prefer changing the IR structure instead of describing the fix in prose.
        - Keep the plan directly executable after local rendering.
        - All natural-language fields must be Simplified Chinese.
        - Return JSON only.
        """
    ).strip()


def build_ir_feedback_instructions() -> str:
    return dedent(
        """
        You revise a Maxwell IR plan after a completed simulation run failed some user constraints.
        Input will contain:
        - original requirement
        - current intake JSON
        - current IR artifact JSON
        - previous scalar outputs
        - previous evaluation JSON
        - feedback_round

        Return one strict JSON object with:
        - summary
        - ir_plan
        - assumptions
        - warnings

        Your job:
        - preserve the user's hard numeric constraints
        - use the failed and unverified checks plus previous outputs to revise geometry, excitations, outputs, and postprocess
        - prefer modifying parameter defaults, object expressions, and assignments in the IR itself instead of rewriting prose
        - keep the plan within the currently supported IR subset:
          rectangle, circle, region, subtract, current, voltage, balloon, matrix, field_scalar, matrix_export_value
        - keep driver=Maxwell2d and stay within the current 2D executable subset
        - use exact local renderer field names; do not invent alternative key names
        - keep internal identifiers in ASCII snake_case
        - keep the returned IR directly executable after local rendering
        - when a constraint can be solved algebraically from previous outputs, compute the needed parameter change directly

        Typical revision directions:
        - if current density is too high, enlarge conductor cross-section or reduce excitation only if that does not violate the user's required current
        - if magnetic flux density exceeds a user limit, enlarge geometry, increase spacing, or reduce excitation only when allowed by the requirement
        - if capacitance is too low, increase effective electrode overlap or reduce dielectric gap while preserving fixed voltages
        - if electric field exceeds a user limit, increase insulation spacing or reduce local field concentration while preserving hard user constraints when possible
        - if required outputs are missing, strengthen postprocess items instead of only changing summary text

        All natural-language fields must be Simplified Chinese.
        Return JSON only.
        """
    ).strip()


def build_ir_patch_feedback_instructions() -> str:
    return dedent(
        """
        You revise a Maxwell IR plan by returning a small, checkable patch.
        Input will contain:
        - original requirement
        - current intake JSON
        - current IR artifact JSON
        - previous scalar outputs
        - previous evaluation JSON
        - residuals extracted from failed constraints
        - feedback_round
        - allowed_patch_operations

        Return one strict JSON object with:
        - summary: short Simplified Chinese explanation of the intended repair
        - actions: a list of patch actions
        - expected_effects: Simplified Chinese list
        - warnings: Simplified Chinese list

        Patch action schema:
        - operation: one of set_parameter_default, set_local_expression, set_object_material, add_warning
        - target: for set_parameter_default use an existing ir_plan.parameters[].name; for set_local_expression use
          an existing ir_plan.locals[].name; for set_object_material use an existing ir_plan.objects[].name
        - value: the new value or expression
        - reason: short Simplified Chinese reason

        Rules:
        - Preserve the user's hard numeric constraints.
        - Do not return a full IR plan in this mode.
        - Only modify existing IR parameters, locals, or object material. Do not invent target names.
        - Use residuals first. If a failed constraint has actual, target, relation, compute the direction and size of
          the change from that residual instead of making a vague suggestion.
        - If current density is too high, increase existing conductor width/thickness parameters when available; reduce
          current only when the user allows it.
        - If magnetic flux density is too high, increase existing gap/spacing/geometry-size parameters when available;
          reduce current only when allowed.
        - If capacitance is too low, increase existing plate width/overlap/radius parameters or decrease existing gap
          parameters while preserving voltage constraints.
        - If electric field is too high, increase existing spacing/air-gap parameters or reduce voltage only when allowed.
        - If no safe patch exists with current targets, return actions=[] and explain the missing adjustable parameter
          in warnings.
        - All natural-language fields must be Simplified Chinese.
        - Return JSON only.
        """
    ).strip()


def build_primitive_template_generation_instructions() -> str:
    return dedent(
        """
        You generate a reusable local primitive template for one previously unsupported Maxwell 2D geometry primitive.
        Return one strict JSON object with:
        - summary
        - template
        - assumptions
        - warnings

        The template object must follow this structure:
        - primitive_key
        - display_name
        - aliases
        - summary
        - parameters
        - objects
        - operations
        - result_role_name
        - result_area_expr
        - assumptions
        - warnings
        - source

        Primitive template rules:
        - The template must be reusable for one class of composite 2D geometry, not a single fixed-size instance.
        - Do not hardcode the current case dimensions into object expressions. Put reusable dimensions into parameters.
        - parameters must use short ASCII snake_case names.
        - Each parameter should include aliases that help the local loader read common object fields.
        - objects may use only circle or rectangle.
        - operations may use only subtract.
        - result_role_name must point to the role that remains after subtract and will receive material/excitation.
        - result_area_expr should be provided whenever the remaining conductor area can be written algebraically.
        - material handling must be generic:
          use material_mode=instance for the main conductor/body,
          and material_mode=fixed only for helper voids.
        - This template is for geometry decomposition only. Do not include solver settings, excitations, or postprocess.
        - If the requested primitive cannot be represented by circle/rectangle/subtract, still return the closest conservative 2D approximation that can run now, and explain the approximation in warnings.
        - All natural-language fields must be Simplified Chinese.
        - Return JSON only.
        """
    ).strip()


def build_primitive_template_repair_instructions() -> str:
    return dedent(
        """
        You repair a reusable local primitive template for a Maxwell 2D composite geometry.
        Input will contain:
        - original requirement
        - current intake JSON
        - raw geometry object JSON
        - previous primitive template JSON
        - error details

        Return one strict JSON object with:
        - summary
        - template
        - assumptions
        - warnings

        Repair rules:
        - Keep the template reusable for a whole primitive class, not only for the failing example.
        - Fix the specific structural issue: missing parameter, bad role reference, unsupported object kind, missing area expression, or non-runnable decomposition.
        - Stay within the supported primitive subset:
          circle, rectangle, subtract.
        - Keep parameter names in ASCII snake_case.
        - Preserve the intended physical meaning of the primitive.
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
        Use exactly the patch_schema keys supplied in the input.

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
        - include assumptions and warnings as arrays, even when empty
        - do not omit required schema keys; set unused scalar fields to null
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
        - when useful, revise or add simulation_spec.ir_plan / execution_plan.ir_plan so the next round can render directly into a Maxwell script
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
