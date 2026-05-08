from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from maxwell_agent.agent import MaxwellAgent
from maxwell_agent.config import Settings


CHILD_ENV_FLAG = "MAXWELL_GENERIC_2D_CHILD"


CASES = [
    {
        "name": "generic_round_pair_electrostatic",
        "requirement": "\u505a\u4e00\u4e2a\u4e8c\u7ef4\u53cc\u5706\u5bfc\u4f53\u622a\u9762\u6a21\u578b\uff0c\u4e24\u6839\u5706\u94dc\u5bfc\u4f53\u534a\u5f841mm\uff0c\u4e2d\u5fc3\u8ddd6mm\uff0c\u65bd\u52a0100V\u7535\u538b\u5dee\uff0c\u8bc4\u4f30\u5355\u4f4d\u957f\u5ea6\u7535\u5bb9\u548c\u6700\u5927\u7535\u573a\u3002",
    },
    {
        "name": "generic_annular_conductor_magnetostatic",
        "requirement": "\u505a\u4e00\u4e2a\u4e8c\u7ef4\u540c\u5fc3\u73af\u5bfc\u4f53\u622a\u9762\uff0c\u5185\u534a\u5f843mm\uff0c\u5916\u534a\u5f846mm\uff0c\u6750\u6599\u4e3a\u94dc\uff0c\u901a\u4ee5100A\u7535\u6d41\uff0c\u8bc4\u4f30\u6700\u5927\u78c1\u5bc6\u3002",
    },
    {
        "name": "generic_dual_strip_electrostatic",
        "requirement": "\u505a\u4e00\u4e2a\u4e8c\u7ef4\u53cc\u77e9\u5f62\u5bfc\u4f53\u622a\u9762\u6a21\u578b\uff0c\u4e24\u6761\u94dc\u5bfc\u4f53\u5404\u5bbd4mm\u539a1mm\uff0c\u4e2d\u5fc3\u8ddd10mm\uff0c\u65bd\u52a0100V\u7535\u538b\u5dee\uff0c\u8bc4\u4f30\u5355\u4f4d\u957f\u5ea6\u7535\u5bb9\u548c\u6700\u5927\u7535\u573a\u3002",
    },
]


def _run_cases_inline(cases: list[dict[str, str]], summary_path: Path) -> list[dict[str, object]]:
    settings = Settings(_env_file=PROJECT_ROOT / ".env", project_root=PROJECT_ROOT)
    settings.design_feedback_max_iters = 2
    settings.codexa_timeout_s = max(settings.codexa_timeout_s, 120)
    agent = MaxwellAgent(settings)

    summary: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        requirement = str(case["requirement"])
        name = str(case["name"])
        print(f"[{index}/{len(cases)}] running {name}", flush=True)
        result = agent.run(requirement)
        entry = {
            "name": name,
            "requirement": requirement,
            "status": result.status,
            "message": result.message,
            "run_directory": str(result.run_directory),
            "task_family": result.intake.task_family if result.intake else None,
            "evaluation_status": result.evaluation.overall_status if result.evaluation else None,
            "evaluation_summary": result.evaluation.summary if result.evaluation else None,
            "outputs": result.outputs,
        }
        summary.append(entry)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "name": name,
                    "task_family": entry["task_family"],
                    "status": result.status,
                    "evaluation_status": entry["evaluation_status"],
                    "run_directory": entry["run_directory"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return summary


def _run_cases_isolated(cases: list[dict[str, str]], summary_dir: Path, summary_path: Path) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        name = str(case["name"])
        print(f"[{index}/{len(cases)}] running {name}", flush=True)
        child_env = os.environ.copy()
        child_env[CHILD_ENV_FLAG] = "1"
        child_env["PYTHONPATH"] = str(SRC_ROOT)
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), name],
            cwd=str(PROJECT_ROOT),
            env=child_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        (summary_dir / f"{name}_stdout.log").write_text(child.stdout or "", encoding="utf-8")
        (summary_dir / f"{name}_stderr.log").write_text(child.stderr or "", encoding="utf-8")
        child_summary_path: Path | None = None
        for line in (child.stdout or "").splitlines():
            if line.startswith("summary saved to "):
                child_summary_path = Path(line.removeprefix("summary saved to ").strip())
                break
        if child_summary_path is None or not child_summary_path.exists():
            raise RuntimeError(f"child summary missing for {name}")
        child_summary = json.loads(child_summary_path.read_text(encoding="utf-8"))
        if not isinstance(child_summary, list) or not child_summary:
            raise RuntimeError(f"child summary malformed for {name}")
        summary.append(child_summary[0])
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "name": summary[-1]["name"],
                    "task_family": summary[-1]["task_family"],
                    "status": summary[-1]["status"],
                    "evaluation_status": summary[-1]["evaluation_status"],
                    "run_directory": summary[-1]["run_directory"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return summary


def main() -> int:
    selected = {item.strip() for item in sys.argv[1:] if item.strip()}
    cases = [case for case in CASES if not selected or case["name"] in selected]
    if not cases:
        print(json.dumps({"error": "no matching cases"}, ensure_ascii=False))
        return 2

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_dir = PROJECT_ROOT / "artifacts" / f"generic_2d_regression_{timestamp}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "summary.json"
    if os.environ.get(CHILD_ENV_FLAG) == "1" or len(cases) == 1:
        summary = _run_cases_inline(cases, summary_path)
    else:
        summary = _run_cases_isolated(cases, summary_dir, summary_path)

    print(f"summary saved to {summary_path}", flush=True)
    failed = [
        item
        for item in summary
        if item["status"] != "completed" or item["evaluation_status"] != "passed"
    ]
    if failed:
        print(json.dumps({"failed_cases": [item["name"] for item in failed]}, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
