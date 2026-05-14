from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from maxwell_agent.agent import MaxwellAgent
from maxwell_agent.config import Settings


CASES = [
    {
        "name": "electromagnet_2d",
        "requirement": "\u505a\u4e00\u4e2a24V\u76f4\u6d41\u7535\u78c1\u94c1\uff0c\u6c14\u96992mm\uff0c\u7535\u6d41\u4e0d\u8d85\u8fc72A\uff0c\u5c3d\u91cf\u63d0\u9ad8\u5438\u529b\uff0c\u5916\u5f62\u4e0d\u8981\u592a\u5927\u3002",
    },
    {
        "name": "capacitor_2d",
        "requirement": "\u505a\u4e00\u4e2a\u4e8c\u7ef4\u5e73\u884c\u677f\u7535\u5bb9\u5668\uff0c\u677f\u95f4\u8ddd1mm\uff0c\u677f\u5bbd20mm\uff0c\u65bd\u52a0100V\uff0c\u7535\u5bb9\u81f3\u5c1150pF\uff0c\u7535\u573a\u4e0d\u8d85\u8fc7400000V/m\u3002",
    },
    {
        "name": "coaxial_capacitor_2d",
        "requirement": "\u505a\u4e00\u4e2a\u540c\u8f74\u7535\u5bb9\u5668\uff0c\u5185\u534a\u5f841mm\uff0c\u5916\u534a\u5f845mm\uff0c\u65bd\u52a0100V\u7535\u538b\uff0c\u7535\u5bb9\u81f3\u5c1130pF\uff0c\u7535\u573a\u4e0d\u8d85\u8fc770000V/m\u3002",
    },
    {
        "name": "busbar_2d",
        "requirement": "\u505a\u4e00\u6839\u8f7d\u6d41\u94dc\u6392\uff0c\u5bbd10mm\uff0c\u539a2mm\uff0c\u7535\u6d41200A\uff0c\u7535\u6d41\u5bc6\u5ea6\u4e0d\u8d85\u8fc75A/mm^2\uff0c\u8bc4\u4f30\u6700\u5927\u78c1\u5bc6\u548c\u7535\u6d41\u5bc6\u5ea6\u3002",
    },
    {
        "name": "solenoid_2d",
        "requirement": "\u505a\u4e00\u4e2a\u7a7a\u5fc3\u87ba\u7ebf\u7ba1\uff0c\u957f\u5ea650mm\uff0c\u534a\u5f848mm\uff0c\u5305\u6570300\uff0c\u7535\u6d411A\uff0c\u6c42\u4e2d\u5fc3\u78c1\u5bc6\u3002",
    },
    {
        "name": "inductor_2d",
        "requirement": "\u505a\u4e00\u4e2a\u4e8c\u7ef4\u7535\u611f\uff0c\u6c14\u96990.5mm\uff0c\u533d\u6570600\uff0c\u7535\u6d412A\uff0c\u76ee\u6807\u7535\u611f0.3H\uff0c\u8f93\u51fa\u7535\u611f\u548c\u78c1\u5bc6\u3002",
    },
    {
        "name": "transformer_2d",
        "requirement": "\u505a\u4e00\u4e2a10000V\u5230220V\u5de5\u9891\u53d8\u538b\u5668\uff0c\u8f93\u51fa\u531d\u6bd4\u3001\u6b21\u7ea7\u7535\u538b\u4f30\u7b97\u548c\u6700\u5927\u78c1\u5bc6\u3002",
    },
    {
        "name": "compact_relay_actuator_2d",
        "requirement": "\u505a\u4e00\u4e2a\u7d27\u51d1\u76f4\u6d41\u7ee7\u7535\u5668\u7535\u78c1\u6267\u884c\u5668\u4e8c\u7ef4\u622a\u9762\uff0c\u4f9b\u753524V\uff0c\u6c14\u96991.5mm\uff0c\u7535\u6d41\u4e0d\u8d85\u8fc72A\uff0c\u7ebf\u5708\u5916\u5f62\u4e0d\u8d85\u8fc730mm\u4e5825mm\uff0c\u6700\u5927\u78c1\u5bc6\u4e0d\u8d85\u8fc71.8T\uff0c\u5c3d\u91cf\u63d0\u9ad8\u6c14\u9699\u5438\u529b\uff0c\u8f93\u51fa\u7535\u6d41\u3001\u5438\u529b\u3001\u6700\u5927\u78c1\u5bc6\u548c\u6bcf\u4e2a\u7ea6\u675f\u662f\u5426\u6ee1\u8db3\u3002",
    },
]


def main() -> int:
    selected = {item.strip() for item in sys.argv[1:] if item.strip()}
    cases = [case for case in CASES if not selected or case["name"] in selected]
    if not cases:
        print(json.dumps({"error": "no matching cases"}, ensure_ascii=False))
        return 2

    timestamp = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    summary_dir = PROJECT_ROOT / "artifacts" / f"regression_2d_{timestamp}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "summary.json"

    settings = Settings(_env_file=PROJECT_ROOT / ".env", project_root=PROJECT_ROOT)
    settings.design_feedback_max_iters = 3
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
                    "status": result.status,
                    "evaluation_status": entry["evaluation_status"],
                    "run_directory": entry["run_directory"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

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
