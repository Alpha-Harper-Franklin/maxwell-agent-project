from __future__ import annotations

import os
import shutil
import re
from pathlib import Path

from .models import MaxwellEnvironment


def _normalize_version_hint(raw_version: str | None) -> str | None:
    if not raw_version:
        return None
    match = re.fullmatch(r"[vV]?(\d{3})", raw_version.strip())
    if not match:
        return raw_version
    digits = match.group(1)
    major = 2000 + int(digits[:2])
    minor = int(digits[2])
    return f"{major}.{minor}"


def _collect_candidates() -> list[Path]:
    candidates: list[Path] = []

    for exe_name in ("ansysedt.exe", "ansysedtng.exe", "ansysedtsv.exe"):
        direct_hit = shutil.which(exe_name)
        if direct_hit:
            candidates.append(Path(direct_hit))

    for env_key, value in os.environ.items():
        if env_key.startswith("AWP_ROOT") and value:
            root = Path(value)
            for probe in (
                root / "Win64" / "ansysedt.exe",
                root / "Win64" / "ansysedtng.exe",
                root / "Win64" / "ansysedtsv.exe",
                root / "AnsysEM" / "ansysedt.exe",
                root / "AnsysEM" / "ansysedtng.exe",
                root / "AnsysEM" / "ansysedtsv.exe",
            ):
                if probe.exists():
                    candidates.append(probe)

    common_patterns = [
        Path(r"C:\Program Files\AnsysEM"),
        Path(r"F:\AnsysEM"),
        Path(r"F:\Downloads\AnsysEM"),
        Path(r"F:\AnsysEM_Student_2025R2"),
    ]
    for base in common_patterns:
        if not base.exists():
            continue
        for pattern in (
            r"*\Win64\ansysedt.exe",
            r"*\Win64\ansysedtng.exe",
            r"*\Win64\ansysedtsv.exe",
            r"*\AnsysEM\ansysedt.exe",
            r"*\AnsysEM\ansysedtng.exe",
            r"*\AnsysEM\ansysedtsv.exe",
        ):
            for item in sorted(base.glob(pattern)):
                candidates.append(item)

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def detect_maxwell_environment() -> MaxwellEnvironment:
    try:
        import ansys.aedt.core  # noqa: F401

        pyaedt_importable = True
    except Exception:
        pyaedt_importable = False

    candidates = _collect_candidates()
    installed = bool(candidates)
    notes: list[str] = []
    version_hint: str | None = None
    student_version = False

    if installed:
        raw_hint = candidates[0].parts[-3] if len(candidates[0].parts) >= 3 else None
        version_hint = _normalize_version_hint(raw_hint)
        first_candidate = str(candidates[0]).lower()
        student_version = "student" in first_candidate or "ansysedtsv.exe" in first_candidate
    else:
        notes.append(
            "No AEDT executable was found in PATH, AWP_ROOT*, or common install folders."
        )

    if not pyaedt_importable:
        notes.append("PyAEDT import failed in the current Python environment.")

    return MaxwellEnvironment(
        installed=installed,
        executable=str(candidates[0]) if installed else None,
        version_hint=version_hint,
        student_version=student_version,
        pyaedt_importable=pyaedt_importable,
        candidates=[str(path) for path in candidates],
        notes=notes,
    )
