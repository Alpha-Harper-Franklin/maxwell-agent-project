from __future__ import annotations

import json
from pathlib import Path

from maxwell_agent import script_runner


def test_script_runner_prepares_runtime_before_calling_generated_script(tmp_path: Path, monkeypatch) -> None:
    script_path = tmp_path / "generated.py"
    job_path = tmp_path / "job.json"
    result_path = tmp_path / "result.json"
    marker_path = tmp_path / "prepared.txt"

    script_path.write_text(
        """
def run_job(job):
    return {"prepared_marker_exists": str(job["marker_path"])}
""".strip(),
        encoding="utf-8",
    )
    job_path.write_text(
        json.dumps(
            {
                "entrypoint": "run_job",
                "project_file": str(tmp_path / "demo.aedt"),
                "marker_path": str(marker_path.exists()),
            }
        ),
        encoding="utf-8",
    )

    def fake_prepare(job):
        marker_path.write_text(job["project_file"], encoding="utf-8")
        job["marker_path"] = str(marker_path.exists())

    monkeypatch.setattr(script_runner, "prepare_generated_script_runtime", fake_prepare)

    assert script_runner.main([str(script_path), str(job_path), str(result_path)]) == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["outputs"]["prepared_marker_exists"] == "True"
