from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .models import GeneratedMaxwellScript


IRParameterSource = Literal["geometry", "excitations", "boundaries", "constraints", "design"]
IRParameterCast = Literal["float", "int", "str", "bool"]
IRLocalCast = Literal["float", "int", "str", "bool"]
IRObjectKind = Literal["rectangle", "circle", "region"]
IROperationKind = Literal["subtract"]
IRAssignmentKind = Literal["current", "voltage", "balloon", "matrix"]
IRDerivedOutputPhase = Literal["before_solve", "after_solve"]
IRDerivedOutputCast = Literal["float", "int", "str"]
IRPostprocessKind = Literal["field_scalar", "matrix_export_value"]
IRPostprocessCast = Literal["float", "str"]


class IRParameterBinding(BaseModel):
    name: str
    source: IRParameterSource
    field: str
    default: float | int | str | bool
    cast: IRParameterCast = "float"


class IRLocalValue(BaseModel):
    name: str
    expression: str
    cast: IRLocalCast = "float"


class IRObject(BaseModel):
    name: str
    kind: IRObjectKind
    material: str | None = None
    origin_exprs: list[str] = Field(default_factory=list)
    sizes_exprs: list[str] = Field(default_factory=list)
    radius_expr: str | None = None
    pad_value_exprs: list[str] = Field(default_factory=list)
    pad_type: str | None = None


class IROperation(BaseModel):
    kind: IROperationKind
    blank_parts: list[str]
    tool_parts: list[str]
    keep_originals: bool = False


class IRAssignment(BaseModel):
    name: str
    kind: IRAssignmentKind
    targets: list[str] = Field(default_factory=list)
    amplitude_expr: str | None = None
    boundary_name: str | None = None
    is_voltage: bool | None = None
    signal_assignments: list[str] = Field(default_factory=list)
    ground_assignments: list[str] = Field(default_factory=list)


class IRDerivedOutput(BaseModel):
    output_key: str
    expression: str
    cast: IRDerivedOutputCast = "float"
    phase: IRDerivedOutputPhase = "before_solve"


class IRPostprocess(BaseModel):
    kind: IRPostprocessKind
    output_key: str
    cast: IRPostprocessCast = "float"
    setup_name: str | None = None
    quantity: str | None = None
    scalar_function: str | None = None
    object_name: str | None = None
    object_type: str | None = None
    solution_expr: str | None = None
    error_output_key: str | None = None
    matrix_assignment_name: str | None = None
    output_filename: str | None = None
    regex_pattern: str | None = None
    scaled_output_key: str | None = None
    scale: float | None = None


class MaxwellIRPlan(BaseModel):
    driver: Literal["Maxwell2d", "Maxwell3d"] = "Maxwell2d"
    design_name: str
    solution_type: str
    model_units: str = "mm"
    setup_name: str = "Setup1"
    setup_type: str
    parameters: list[IRParameterBinding] = Field(default_factory=list)
    locals: list[IRLocalValue] = Field(default_factory=list)
    objects: list[IRObject] = Field(default_factory=list)
    operations: list[IROperation] = Field(default_factory=list)
    assignments: list[IRAssignment] = Field(default_factory=list)
    derived_outputs: list[IRDerivedOutput] = Field(default_factory=list)
    postprocess: list[IRPostprocess] = Field(default_factory=list)
    failure_note: str = "Maxwell solve failed."


class GeneratedIRPlan(BaseModel):
    summary: str = "AI generated a Maxwell IR plan for the current requirement."
    ir_plan: MaxwellIRPlan
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def validate_ir_plan(plan: MaxwellIRPlan) -> MaxwellIRPlan:
    parameter_names = {item.name for item in plan.parameters}
    local_names = {item.name for item in plan.locals}
    object_names = [item.name for item in plan.objects]
    assignment_names = {item.name for item in plan.assignments}

    if len(parameter_names) != len(plan.parameters):
        raise ValueError("Duplicate IR parameter names are not allowed.")
    if len(local_names) != len(plan.locals):
        raise ValueError("Duplicate IR local names are not allowed.")
    if len(object_names) != len(set(object_names)):
        raise ValueError("Duplicate IR object names are not allowed.")
    if len(assignment_names) != len(plan.assignments):
        raise ValueError("Duplicate IR assignment names are not allowed.")

    known_objects = set(object_names)
    for operation in plan.operations:
        missing = [name for name in [*operation.blank_parts, *operation.tool_parts] if name not in known_objects]
        if missing:
            raise ValueError(f"IR operation references unknown objects: {missing}")

    for assignment in plan.assignments:
        if assignment.kind in {"current", "voltage", "balloon"}:
            missing = [name for name in assignment.targets if name not in known_objects]
            if missing:
                raise ValueError(f"IR assignment {assignment.name} references unknown objects: {missing}")
        if assignment.kind == "matrix":
            referenced = [*assignment.signal_assignments, *assignment.ground_assignments]
            missing = [name for name in referenced if name not in assignment_names]
            if missing:
                raise ValueError(f"IR matrix assignment {assignment.name} references unknown assignments: {missing}")

    for item in plan.postprocess:
        if item.kind == "matrix_export_value":
            matrix_name = item.matrix_assignment_name or ""
            if matrix_name and matrix_name not in assignment_names:
                raise ValueError(f"IR postprocess references unknown matrix assignment: {matrix_name}")

    return plan


def render_script_from_ir(
    plan: MaxwellIRPlan,
    summary: str,
    assumptions: list[str] | None = None,
    warnings: list[str] | None = None,
) -> GeneratedMaxwellScript:
    plan = validate_ir_plan(plan)
    assumptions = assumptions or []
    warnings = warnings or []
    uses_matrix = any(item.kind == "matrix" for item in plan.assignments)
    uses_matrix_export = any(item.kind == "matrix_export_value" for item in plan.postprocess)
    object_names = {item.name for item in plan.objects}

    imports = ["import math"]
    if uses_matrix_export:
        imports.extend(["import re", "from pathlib import Path"])
    imports.append(f"from ansys.aedt.core import {plan.driver}")
    if uses_matrix:
        imports.append("from ansys.aedt.core.modules.boundary.maxwell_boundary import MatrixElectric")

    lines: list[str] = [*imports, "", "", "def run_job(job: dict) -> dict:"]
    lines.extend(
        [
            '    def _as_mapping(value):',
            '        return dict(value) if isinstance(value, dict) else {}',
            '    simulation_spec = _as_mapping(job.get("simulation_spec"))',
            '    geometry = _as_mapping(simulation_spec.get("geometry"))',
            '    excitations = _as_mapping(simulation_spec.get("excitations"))',
            '    boundaries = _as_mapping(simulation_spec.get("boundaries"))',
            '    constraints = _as_mapping(simulation_spec.get("constraints"))',
            '    design = _as_mapping(job.get("design"))',
            '    project_file = str(job["project_file"])',
            '    version = job.get("maxwell_version")',
            '    non_graphical = bool(job.get("non_graphical", True))',
            '    student_version = bool(job.get("student_version", False))',
        ]
    )
    if uses_matrix_export:
        lines.append('    output_dir = Path(str(job["output_dir"]))')
    lines.append("")

    for binding in plan.parameters:
        lines.append(f"    {binding.name} = {_render_binding(binding)}")
    if plan.parameters:
        lines.append("")
    for item in plan.locals:
        lines.append(f"    {item.name} = {_render_cast_expression(item.expression, item.cast)}")
    if plan.locals:
        lines.append("")

    lines.append("    outputs: dict[str, float | str] = {}")
    for item in plan.derived_outputs:
        if item.phase == "before_solve":
            lines.append(f'    outputs["{item.output_key}"] = {_render_cast_expression(item.expression, item.cast)}')
    lines.extend(
        [
            f"    with {plan.driver}(",
            "        project=project_file,",
            f'        design="{plan.design_name}",',
            f'        solution_type="{plan.solution_type}",',
            "        version=version,",
            "        non_graphical=non_graphical,",
            "        new_desktop=True,",
            "        close_on_exit=True,",
            "        student_version=student_version,",
            "    ) as app:",
            f'        app.modeler.model_units = "{plan.model_units}"',
            "        objects = {}",
            "        assignments = {}",
        ]
    )

    for obj in plan.objects:
        lines.extend(_render_object_block(obj))
    for op in plan.operations:
        lines.extend(_render_operation_block(op))
    for assignment in plan.assignments:
        lines.extend(_render_assignment_block(assignment))

    lines.extend(
        [
            f'        app.create_setup(name="{plan.setup_name}", setup_type="{plan.setup_type}")',
            "        app.save_project()",
            f'        solve_ok = bool(app.analyze_setup("{plan.setup_name}"))',
            '        outputs["solve_status"] = "completed" if solve_ok else "failed"',
            "        if not solve_ok:",
            '            outputs["status"] = "failed"',
            f'            outputs["notes"] = "{plan.failure_note}"',
            '            outputs["project_name"] = app.project_name',
            '            outputs["design_name"] = app.design_name',
            "            app.save_project()",
            "            return outputs",
            '        outputs["project_name"] = app.project_name',
            '        outputs["design_name"] = app.design_name',
        ]
    )

    for item in plan.derived_outputs:
        if item.phase == "after_solve":
            lines.append(f'        outputs["{item.output_key}"] = {_render_cast_expression(item.expression, item.cast)}')
    for item in plan.postprocess:
        lines.extend(_render_postprocess_block(item, object_names))

    lines.extend(["        app.save_project()", "    return outputs"])
    code = "\n".join(lines).strip()
    return GeneratedMaxwellScript(
        filename="generated_maxwell_job.py",
        entrypoint="run_job",
        summary=summary,
        code=code,
        assumptions=assumptions,
        warnings=warnings,
    )


def _render_binding(binding: IRParameterBinding) -> str:
    source_map = {
        "geometry": "geometry",
        "excitations": "excitations",
        "boundaries": "boundaries",
        "constraints": "constraints",
        "design": "design",
    }
    source_expr = f'{source_map[binding.source]}.get("{binding.field}", {binding.default!r})'
    if binding.cast == "float":
        return f"float({source_expr})"
    if binding.cast == "int":
        return f"int({source_expr})"
    if binding.cast == "bool":
        return f"bool({source_expr})"
    return f"str({source_expr})"


def _render_object_block(obj: IRObject) -> list[str]:
    if obj.kind == "rectangle":
        return [
            f'        objects["{obj.name}"] = app.modeler.create_rectangle(',
            f"            origin={_render_unit_expr_list(obj.origin_exprs, 'mm')},",
            f"            sizes={_render_unit_expr_list(obj.sizes_exprs, 'mm')},",
            f'            name="{obj.name}",',
            f'            material="{obj.material or "vacuum"}",',
            "        )",
        ]
    if obj.kind == "circle":
        return [
            f'        objects["{obj.name}"] = app.modeler.create_circle(',
            f"            origin={_render_unit_expr_list(obj.origin_exprs, 'mm')},",
            f"            radius={_render_unit_expr(obj.radius_expr or '0', 'mm')},",
            f'            name="{obj.name}",',
            f'            material="{obj.material or "vacuum"}",',
            "        )",
        ]
    if obj.kind == "region":
        return [
            f'        objects["{obj.name}"] = app.modeler.create_region(',
            f"            pad_value={_render_unit_expr_list(obj.pad_value_exprs, 'mm')},",
            f'            pad_type="{obj.pad_type or "Absolute Offset"}",',
            f'            name="{obj.name}",',
            "        )",
        ]
    raise ValueError(f"Unsupported IR object kind: {obj.kind}")


def _render_operation_block(op: IROperation) -> list[str]:
    if op.kind == "subtract":
        blank_parts = ", ".join(f'objects["{name}"].name' for name in op.blank_parts)
        tool_parts = ", ".join(f'objects["{name}"].name' for name in op.tool_parts)
        return [
            "        app.modeler.subtract(",
            f"            blank_list=[{blank_parts}],",
            f"            tool_list=[{tool_parts}],",
            f"            keep_originals={str(op.keep_originals)},",
            "        )",
        ]
    raise ValueError(f"Unsupported IR operation kind: {op.kind}")


def _render_assignment_block(assignment: IRAssignment) -> list[str]:
    if assignment.kind == "current":
        targets = ", ".join(f'objects["{name}"]' for name in assignment.targets)
        return [
            f'        assignments["{assignment.name}"] = app.assign_current(',
            f"            [{targets}],",
            f"            amplitude={_render_unit_expr(assignment.amplitude_expr or '0', 'A')},",
            f'            name="{assignment.name}",',
            "        )",
        ]
    if assignment.kind == "voltage":
        targets = ", ".join(f'objects["{name}"]' for name in assignment.targets)
        return [
            f'        assignments["{assignment.name}"] = app.assign_voltage(',
            f"            [{targets}],",
            f"            amplitude={_render_unit_expr(assignment.amplitude_expr or '0', 'V')},",
            f'            name="{assignment.name}",',
            "        )",
        ]
    if assignment.kind == "balloon":
        target_name = assignment.targets[0]
        args = [
            f'objects["{target_name}"].edges',
            f'boundary="{assignment.boundary_name or assignment.name}"',
        ]
        if assignment.is_voltage is not None:
            args.append(f"is_voltage={str(assignment.is_voltage)}")
        args_text = ", ".join(args)
        return [
            "        try:",
            f"            assignments[{assignment.name!r}] = app.assign_balloon({args_text})",
            "        except Exception:",
            "            pass",
        ]
    if assignment.kind == "matrix":
        signal_sources = ", ".join(f'assignments["{name}"].name' for name in assignment.signal_assignments)
        ground_sources = ", ".join(f'assignments["{name}"].name' for name in assignment.ground_assignments)
        return [
            "        try:",
            f'            assignments["{assignment.name}"] = app.assign_matrix(',
            "                MatrixElectric(",
            f"                    signal_sources=[{signal_sources}],",
            f"                    ground_sources=[{ground_sources}],",
            f'                    matrix_name="{assignment.name}",',
            "                )",
            "            )",
            "        except Exception:",
            f'            assignments["{assignment.name}"] = None',
        ]
    raise ValueError(f"Unsupported IR assignment kind: {assignment.kind}")


def _render_postprocess_block(item: IRPostprocess, object_names: set[str]) -> list[str]:
    error_output_key = item.error_output_key or f"{item.output_key}_note"
    if item.kind == "field_scalar":
        quantity = item.quantity or "Mag_B"
        scalar_function = item.scalar_function or "Maximum"
        object_name_expr = '"AllObjects"'
        if item.object_name:
            if item.object_name in object_names:
                object_name_expr = f'objects["{item.object_name}"].name'
            else:
                object_name_expr = repr(item.object_name)
        call_args = [
            repr(quantity),
            repr(scalar_function),
            f"object_name={object_name_expr}",
        ]
        if item.solution_expr:
            call_args.append(f"solution={item.solution_expr}")
        if item.object_type:
            call_args.append(f'object_type="{item.object_type}"')
        call_text = ", ".join(call_args)
        return [
            "        try:",
            f'            outputs["{item.output_key}"] = {_render_postprocess_cast(item.cast)}(',
            f"                app.post.get_scalar_field_value({call_text})",
            "            )",
            "        except Exception as exc:",
            f'            outputs["{error_output_key}"] = f"Postprocess skipped: {{exc}}"',
        ]
    if item.kind == "matrix_export_value":
        matrix_name = item.matrix_assignment_name or ""
        return [
            f'        if assignments.get("{matrix_name}"):',
            f'            matrix_path = output_dir / "{item.output_filename or "matrix.txt"}"',
            "            try:",
            f'                app.export_matrix(matrix_name=assignments["{matrix_name}"].name, output_file=matrix_path, setup="{item.setup_name or "Setup1"}")',
            '                outputs["matrix_export_path"] = str(matrix_path)',
            '                matrix_text = matrix_path.read_text(encoding="utf-8", errors="ignore")',
            f"                match = re.search({item.regex_pattern!r}, matrix_text)",
            "                if match:",
            f'                    outputs["{item.output_key}"] = float(match.group(1))',
            *(
                [
                    f'                    outputs["{item.scaled_output_key}"] = outputs["{item.output_key}"] * {item.scale!r}'
                ]
                if item.scaled_output_key and item.scale is not None
                else []
            ),
            "            except Exception as exc:",
            f'                outputs["{error_output_key}"] = f"Matrix export skipped: {{exc}}"',
        ]
    raise ValueError(f"Unsupported IR postprocess kind: {item.kind}")


def _render_postprocess_cast(cast: IRPostprocessCast) -> str:
    if cast == "float":
        return "float"
    return "str"


def _render_cast_expression(expression: str, cast: IRDerivedOutputCast) -> str:
    if cast == "float":
        return f"float({expression})"
    if cast == "int":
        return f"int({expression})"
    return f"str({expression})"


def _render_unit_expr(expression: str, unit: str) -> str:
    return f'f"{{{expression}}}{unit}"'


def _render_unit_expr_list(expressions: list[str], unit: str) -> str:
    return "[" + ", ".join(_render_unit_expr(item, unit) for item in expressions) + "]"
