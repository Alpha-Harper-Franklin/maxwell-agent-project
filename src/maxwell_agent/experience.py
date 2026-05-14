from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class FailureExperience(BaseModel):
    experience_id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    requirement: str
    run_directory: str
    stage: str
    physics_type: str = "unknown"
    builder_hint: str = "unknown"
    failed_checks: list[str] = Field(default_factory=list)
    residual_items: list[dict[str, Any]] = Field(default_factory=list)
    capability_items: list[dict[str, Any]] = Field(default_factory=list)
    outputs: dict[str, Any] = Field(default_factory=dict)
    repair_summary: str = ""
    resolved: bool = False


class ExperienceStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def append(self, experience: FailureExperience) -> None:
        records = self.load()
        records.append(experience)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps([item.model_dump(mode="json") for item in records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> list[FailureExperience]:
        if not self._path.exists():
            return []
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        records: list[FailureExperience] = []
        for item in payload:
            if isinstance(item, dict):
                try:
                    records.append(FailureExperience.model_validate(item))
                except Exception:
                    continue
        return records

    def recent(self, limit: int = 20) -> list[FailureExperience]:
        records = self.load()
        return records[-limit:]
