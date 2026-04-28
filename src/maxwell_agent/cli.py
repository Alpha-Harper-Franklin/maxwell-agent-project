from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.json import JSON

from .agent import MaxwellAgent
from .config import Settings
from .demo import execute_demo
from .demo_server import serve_demo
from .maxwell_env import detect_maxwell_environment


console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Maxwell 2D industrial agent scaffold.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("probe-env", help="Detect local Maxwell/AEDT availability.")
    sub.add_parser("smoke-llm", help="Call the configured Codexa model once.")
    sub.add_parser("list-models", help="List available cloud models from Codexa.")

    plan = sub.add_parser("plan", help="Convert a requirement into structured simulation data.")
    plan.add_argument("requirement", help="Natural-language requirement.")

    run = sub.add_parser("run", help="Plan the design and try to execute Maxwell.")
    run.add_argument("requirement", help="Natural-language requirement.")

    demo = sub.add_parser("demo", help="Run a user-friendly CLI demo.")
    demo.add_argument("requirement", nargs="?", help="Natural-language requirement.")

    serve = sub.add_parser("serve", help="Start a local demo page.")
    serve.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    serve.add_argument("--port", type=int, default=8765, help="Port to bind.")

    return parser


def _print_json(payload: dict) -> None:
    console.print(JSON.from_data(payload))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings()

    if args.command == "probe-env":
        _print_json(detect_maxwell_environment().model_dump())
        return

    agent = MaxwellAgent(settings)

    if args.command == "smoke-llm":
        console.print(agent.smoke_llm())
        return

    if args.command == "list-models":
        _print_json({"models": agent.list_models()})
        return

    if args.command == "plan":
        intake = agent.intake(args.requirement)
        _print_json(intake.model_dump(mode="json"))
        return

    if args.command == "run":
        result = agent.run(args.requirement)
        payload = result.model_dump(mode="json")
        payload["run_directory"] = str(Path(payload["run_directory"]))
        if payload.get("project_file"):
            payload["project_file"] = str(Path(payload["project_file"]))
        payload["artifacts"] = [str(Path(item)) for item in payload["artifacts"]]
        console.print(JSON.from_data(payload))
        return

    if args.command == "demo":
        requirement = args.requirement.strip() if args.requirement else ""
        if not requirement:
            console.print("[bold]请输入需求：[/bold]", end="")
            requirement = input().strip()
        if not requirement:
            parser.error("A non-empty requirement is required for demo mode.")
        bundle = execute_demo(agent, requirement)
        console.print(bundle.to_text_report(), markup=False)
        if bundle.summary_text_path:
            console.print(f"\n摘要文件: {bundle.summary_text_path}", markup=False)
        if bundle.summary_html_path:
            console.print(f"HTML 摘要: {bundle.summary_html_path}", markup=False)
        return

    if args.command == "serve":
        console.print(f"Launching demo page at http://{args.host}:{args.port}/")
        serve_demo(agent=agent, host=args.host, port=args.port)
        return

    parser.error("Unknown command.")


if __name__ == "__main__":
    main()
