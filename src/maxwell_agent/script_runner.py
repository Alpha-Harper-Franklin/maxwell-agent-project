from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

try:
    from .pyaedt_compat import prepare_generated_script_runtime
except ImportError:  # pragma: no cover - used when this file is executed by path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from maxwell_agent.pyaedt_compat import prepare_generated_script_runtime


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("generated_maxwell_job", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load generated script: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if len(args) != 3:
        print("Usage: script_runner.py <script_path> <job_json> <result_json>", file=sys.stderr)
        return 2

    script_path = Path(args[0])
    job_path = Path(args[1])
    result_path = Path(args[2])

    try:
        module = _load_module(script_path)
        job = json.loads(job_path.read_text(encoding="utf-8"))
        prepare_generated_script_runtime(job)
        entrypoint = str(job.get("entrypoint") or "run_job")
        if not hasattr(module, entrypoint):
            raise AttributeError(f"Generated script does not define {entrypoint}().")
        func = getattr(module, entrypoint)
        outputs = func(job)
        if not isinstance(outputs, dict):
            raise TypeError(f"{entrypoint}() must return a dict, got {type(outputs).__name__}.")
        output_status = str(outputs.get("status") or "completed").strip().lower()
        payload = {"status": output_status or "completed", "outputs": outputs}
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if output_status in {"failed", "error", "fatal_error"}:
            return 1
        return 0
    except Exception as exc:  # pragma: no cover
        payload = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(payload["traceback"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
