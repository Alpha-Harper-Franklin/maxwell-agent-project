from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


PrimitiveParameterCast = Literal["float", "int", "str", "bool"]
PrimitiveObjectKind = Literal["rectangle", "circle"]
PrimitiveMaterialMode = Literal["instance", "fixed"]
PrimitiveOperationKind = Literal["subtract"]
PrimitiveSource = Literal["built_in", "learned_from_llm", "learned_from_execution"]


class PrimitiveTemplateParameter(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    cast: PrimitiveParameterCast = "float"
    default: float | int | str | bool | None = None
    required: bool = False
    description: str = ""


class PrimitiveTemplateObject(BaseModel):
    role_name: str
    kind: PrimitiveObjectKind
    material_mode: PrimitiveMaterialMode = "instance"
    material_value: str | None = None
    origin_exprs: list[str] = Field(default_factory=list)
    sizes_exprs: list[str] = Field(default_factory=list)
    radius_expr: str | None = None


class PrimitiveTemplateOperation(BaseModel):
    kind: PrimitiveOperationKind = "subtract"
    blank_roles: list[str] = Field(default_factory=list)
    tool_roles: list[str] = Field(default_factory=list)
    keep_originals: bool = False


class PrimitiveTemplate(BaseModel):
    primitive_key: str
    display_name: str
    aliases: list[str] = Field(default_factory=list)
    summary: str = ""
    parameters: list[PrimitiveTemplateParameter] = Field(default_factory=list)
    objects: list[PrimitiveTemplateObject] = Field(default_factory=list)
    operations: list[PrimitiveTemplateOperation] = Field(default_factory=list)
    result_role_name: str
    result_area_expr: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source: PrimitiveSource = "learned_from_llm"


class PrimitiveTemplateArtifact(BaseModel):
    summary: str
    template: PrimitiveTemplate
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PrimitiveLibrarySnapshot(BaseModel):
    version: int = 1
    primitives: list[PrimitiveTemplate] = Field(default_factory=list)


def validate_primitive_template(template: PrimitiveTemplate) -> PrimitiveTemplate:
    parameter_names = [item.name for item in template.parameters]
    if len(parameter_names) != len(set(parameter_names)):
        raise ValueError("Duplicate primitive parameter names are not allowed.")

    role_names = [item.role_name for item in template.objects]
    if len(role_names) != len(set(role_names)):
        raise ValueError("Duplicate primitive object role names are not allowed.")

    known_roles = set(role_names)
    if template.result_role_name not in known_roles:
        raise ValueError("Primitive result_role_name must reference a declared object role.")

    for operation in template.operations:
        missing = [name for name in [*operation.blank_roles, *operation.tool_roles] if name not in known_roles]
        if missing:
            raise ValueError(f"Primitive operation references unknown roles: {missing}")

    return template


def _normalize_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _builtin_templates() -> list[PrimitiveTemplate]:
    return [
        PrimitiveTemplate(
            primitive_key="annulus",
            display_name="同心圆环截面",
            aliases=["ring", "hollow_circle", "concentric_ring", "同心圆", "圆环", "环形导体", "annular_conductor"],
            summary="使用外圆减内圆生成环形截面。",
            parameters=[
                PrimitiveTemplateParameter(name="center_x_mm", aliases=["center_x_mm", "cx_mm", "x_mm"], cast="float", default=0.0),
                PrimitiveTemplateParameter(name="center_y_mm", aliases=["center_y_mm", "cy_mm", "y_mm"], cast="float", default=0.0),
                PrimitiveTemplateParameter(name="inner_radius_mm", aliases=["inner_radius_mm", "inner_radius"], cast="float", required=True),
                PrimitiveTemplateParameter(name="outer_radius_mm", aliases=["outer_radius_mm", "outer_radius"], cast="float", required=True),
            ],
            objects=[
                PrimitiveTemplateObject(
                    role_name="outer_body",
                    kind="circle",
                    material_mode="instance",
                    origin_exprs=["center_x_mm", "center_y_mm", "0"],
                    radius_expr="outer_radius_mm",
                ),
                PrimitiveTemplateObject(
                    role_name="inner_void",
                    kind="circle",
                    material_mode="fixed",
                    material_value="vacuum",
                    origin_exprs=["center_x_mm", "center_y_mm", "0"],
                    radius_expr="inner_radius_mm",
                ),
            ],
            operations=[
                PrimitiveTemplateOperation(
                    kind="subtract",
                    blank_roles=["outer_body"],
                    tool_roles=["inner_void"],
                )
            ],
            result_role_name="outer_body",
            result_area_expr="math.pi * (outer_radius_mm * outer_radius_mm - inner_radius_mm * inner_radius_mm)",
            assumptions=["按二维同心圆环截面处理。"],
            source="built_in",
        ),
        PrimitiveTemplate(
            primitive_key="rectangular_frame",
            display_name="矩形框截面",
            aliases=["frame", "hollow_rectangle", "rect_frame", "矩形框", "空心矩形", "窗口框"],
            summary="使用外矩形减内矩形生成矩形框截面。",
            parameters=[
                PrimitiveTemplateParameter(name="center_x_mm", aliases=["center_x_mm", "cx_mm", "x_mm"], cast="float", default=0.0),
                PrimitiveTemplateParameter(name="center_y_mm", aliases=["center_y_mm", "cy_mm", "y_mm"], cast="float", default=0.0),
                PrimitiveTemplateParameter(name="outer_width_mm", aliases=["outer_width_mm", "outer_width"], cast="float", required=True),
                PrimitiveTemplateParameter(name="outer_height_mm", aliases=["outer_height_mm", "outer_height"], cast="float", required=True),
                PrimitiveTemplateParameter(name="inner_width_mm", aliases=["inner_width_mm", "inner_width"], cast="float", required=True),
                PrimitiveTemplateParameter(name="inner_height_mm", aliases=["inner_height_mm", "inner_height"], cast="float", required=True),
            ],
            objects=[
                PrimitiveTemplateObject(
                    role_name="outer_frame",
                    kind="rectangle",
                    material_mode="instance",
                    origin_exprs=["center_x_mm - outer_width_mm / 2", "center_y_mm - outer_height_mm / 2", "0"],
                    sizes_exprs=["outer_width_mm", "outer_height_mm"],
                ),
                PrimitiveTemplateObject(
                    role_name="inner_window",
                    kind="rectangle",
                    material_mode="fixed",
                    material_value="vacuum",
                    origin_exprs=["center_x_mm - inner_width_mm / 2", "center_y_mm - inner_height_mm / 2", "0"],
                    sizes_exprs=["inner_width_mm", "inner_height_mm"],
                ),
            ],
            operations=[
                PrimitiveTemplateOperation(
                    kind="subtract",
                    blank_roles=["outer_frame"],
                    tool_roles=["inner_window"],
                )
            ],
            result_role_name="outer_frame",
            result_area_expr="outer_width_mm * outer_height_mm - inner_width_mm * inner_height_mm",
            assumptions=["按二维矩形框截面处理。"],
            source="built_in",
        ),
    ]


class PrimitiveLibrary:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._templates: dict[str, PrimitiveTemplate] = {}
        self._alias_index: dict[str, str] = {}
        self._persisted_keys: set[str] = set()
        self._pending_keys: set[str] = set()
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> None:
        snapshot = PrimitiveLibrarySnapshot(primitives=_builtin_templates())
        if self._path.exists():
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8"))
                snapshot = PrimitiveLibrarySnapshot.model_validate(payload)
            except Exception:
                snapshot = PrimitiveLibrarySnapshot(primitives=_builtin_templates())
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)

        for template in _builtin_templates():
            self.register(template, persist=False, mark_persisted=False)
        for template in snapshot.primitives:
            self.register(template, persist=False, mark_persisted=True)

        if not self._path.exists():
            self.save()

    def register(
        self,
        template: PrimitiveTemplate,
        *,
        persist: bool = False,
        mark_persisted: bool = False,
    ) -> PrimitiveTemplate:
        template = validate_primitive_template(template)
        key = _normalize_token(template.primitive_key)
        template.primitive_key = key
        self._templates[key] = template
        self._alias_index[key] = key
        for alias in [template.display_name, *template.aliases]:
            normalized = _normalize_token(alias)
            if normalized:
                self._alias_index[normalized] = key
        if mark_persisted:
            self._persisted_keys.add(key)
            self._pending_keys.discard(key)
        else:
            if key not in self._persisted_keys:
                self._pending_keys.add(key)
        if persist:
            self.save()
        return template

    def find(self, token: str) -> PrimitiveTemplate | None:
        normalized = _normalize_token(token)
        key = self._alias_index.get(normalized)
        if not key:
            return None
        return self._templates.get(key)

    def is_pending(self, primitive_key: str) -> bool:
        return _normalize_token(primitive_key) in self._pending_keys

    def commit(self, templates: list[PrimitiveTemplate]) -> None:
        updated = False
        for template in templates:
            key = _normalize_token(template.primitive_key)
            if key not in self._templates:
                self.register(template, persist=False, mark_persisted=False)
            if key in self._pending_keys or key not in self._persisted_keys:
                self._persisted_keys.add(key)
                self._pending_keys.discard(key)
                updated = True
        if updated:
            self.save()

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        persisted_keys = sorted(self._persisted_keys | {item.primitive_key for item in _builtin_templates()})
        primitives = [
            self._templates[key]
            for key in persisted_keys
            if key in self._templates
        ]
        snapshot = PrimitiveLibrarySnapshot(primitives=primitives)
        self._path.write_text(
            json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._persisted_keys = {item.primitive_key for item in primitives}
        self._pending_keys.difference_update(self._persisted_keys)

    def list_templates(self) -> list[PrimitiveTemplate]:
        return list(self._templates.values())
