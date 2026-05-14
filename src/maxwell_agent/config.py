from __future__ import annotations

from pathlib import Path
import sys

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .pyaedt_compat import normalize_openai_base_url


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    project_root: Path = Field(default=DEFAULT_PROJECT_ROOT, validation_alias=AliasChoices("PROJECT_ROOT", "project_root"))
    codexa_base_url: str | None = Field(default=None, validation_alias=AliasChoices("CODEXA_BASE_URL", "codexa_base_url"))
    codexa_api_key: str | None = Field(default=None, validation_alias=AliasChoices("CODEXA_API_KEY", "codexa_api_key"))
    codexa_model: str = Field(default="gpt-5.4", validation_alias=AliasChoices("CODEXA_MODEL", "codexa_model"))
    codexa_reasoning_effort: str = Field(
        default="high",
        validation_alias=AliasChoices("CODEXA_REASONING_EFFORT", "codexa_reasoning_effort"),
    )
    codexa_timeout_s: int = Field(default=180, validation_alias=AliasChoices("CODEXA_TIMEOUT_S", "codexa_timeout_s"))
    script_execution_timeout_s: int = Field(
        default=240,
        validation_alias=AliasChoices("SCRIPT_EXECUTION_TIMEOUT_S", "script_execution_timeout_s"),
    )
    maxwell_version: str | None = Field(default=None, validation_alias=AliasChoices("MAXWELL_VERSION", "maxwell_version"))
    maxwell_non_graphical: bool = Field(
        default=True,
        validation_alias=AliasChoices("MAXWELL_NON_GRAPHICAL", "maxwell_non_graphical"),
    )
    script_max_repairs: int = Field(default=2, validation_alias=AliasChoices("SCRIPT_MAX_REPAIRS", "script_max_repairs"))
    design_feedback_max_iters: int = Field(
        default=2,
        validation_alias=AliasChoices("DESIGN_FEEDBACK_MAX_ITERS", "design_feedback_max_iters"),
    )

    @field_validator("project_root", mode="before")
    @classmethod
    def _normalize_path(cls, value: str | Path) -> Path:
        if isinstance(value, Path):
            return value
        return Path(value)

    @field_validator("codexa_base_url", "codexa_api_key", "maxwell_version", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @field_validator("codexa_base_url", mode="after")
    @classmethod
    def _normalize_base_url(cls, value: str | None) -> str | None:
        return normalize_openai_base_url(value)

    @field_validator("codexa_reasoning_effort", mode="before")
    @classmethod
    def _normalize_reasoning_effort(cls, value: str | None) -> str:
        if not value:
            return "high"
        normalized = str(value).strip().lower()
        if normalized not in {"none", "low", "medium", "high"}:
            return "high"
        return normalized

    @property
    def workspace_dir(self) -> Path:
        return self.project_root / "workspace"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def knowledge_dir(self) -> Path:
        return self.project_root / "knowledge"

    @property
    def primitive_library_path(self) -> Path:
        root_level = self.project_root / "primitive_library.json"
        if root_level.exists():
            return root_level
        return self.knowledge_dir / "primitive_library.json"

    @property
    def experience_store_path(self) -> Path:
        return self.knowledge_dir / "failure_experience.json"

    @property
    def runtime_python(self) -> Path:
        venv_python = self.project_root / ".venv" / "Scripts" / "python.exe"
        if venv_python.exists():
            return venv_python
        return Path(sys.executable)

    def ensure_dirs(self) -> None:
        for path in (self.project_root, self.workspace_dir, self.artifacts_dir, self.logs_dir, self.knowledge_dir):
            path.mkdir(parents=True, exist_ok=True)
