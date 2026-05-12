from __future__ import annotations

import re
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def normalize_openai_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().rstrip("/")
    if not text:
        return None
    if re.search(r"/v\d+(?:/.*)?$", text):
        return text
    return f"{text}/v1"


def normalize_aedt_version_for_float(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    match = re.match(r"^(\d{4}\.\d)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return text or None


def patch_pyaedt_student_version_float_bug() -> None:
    try:
        import ansys.aedt.core.desktop as desktop_module
    except Exception:
        return

    desktop_class = getattr(desktop_module, "Desktop", None)
    if desktop_class is None or getattr(desktop_class, "_maxwell_agent_student_patch", False):
        return

    original_check = desktop_class.check_starting_mode

    def patched_check_starting_mode(self):
        original_version = getattr(self, "aedt_version_id", None)
        normalized = normalize_aedt_version_for_float(original_version)
        if normalized and normalized != original_version:
            self.aedt_version_id = normalized
            try:
                return original_check(self)
            finally:
                self.aedt_version_id = original_version
        return original_check(self)

    desktop_class.check_starting_mode = patched_check_starting_mode
    desktop_class._maxwell_agent_student_patch = True


def terminate_process_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def terminate_process_ids(pids: set[int]) -> None:
    if not pids:
        return
    for pid in sorted(pids):
        terminate_process_tree(pid)
    time.sleep(1.0)


class FileLock:
    def __init__(self, path: str | Path, poll_interval_s: float = 0.5) -> None:
        self.path = Path(path)
        self.poll_interval_s = poll_interval_s
        self._handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                self._handle = self.path.open("x", encoding="utf-8")
                self._handle.write(str(os.getpid()))
                self._handle.flush()
                return self
            except FileExistsError:
                if self._lock_is_stale():
                    self._remove_stale_lock()
                    continue
                time.sleep(self.poll_interval_s)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        try:
            self.path.unlink()
        except OSError:
            pass

    def _lock_is_stale(self) -> bool:
        try:
            pid_text = self.path.read_text(encoding="utf-8").strip()
            pid = int(pid_text)
        except Exception:
            return True
        return not _process_exists(pid)

    def _remove_stale_lock(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass


def _process_exists(pid: int) -> bool:
    completed = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return str(pid) in completed.stdout


def scrub_aedt_locks(project_file: str | Path) -> None:
    path = Path(project_file)
    lock_candidates = [
        Path(str(path) + ".lock"),
        path.with_suffix(path.suffix + ".lock"),
        path.with_suffix(".aedt.lock"),
    ]
    for candidate in lock_candidates:
        try:
            if candidate.exists():
                candidate.unlink()
        except OSError:
            continue


def prepare_generated_script_runtime(job: dict[str, Any] | None) -> None:
    patch_pyaedt_student_version_float_bug()
    if isinstance(job, dict) and job.get("project_file"):
        scrub_aedt_locks(str(job["project_file"]))
