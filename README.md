# Maxwell Agent Project

This is a Windows-first `Ansys Maxwell 2D` agent. It accepts an engineering requirement, calls an LLM for semantic parsing, drives local Maxwell execution, extracts results, checks constraints, and can revise the design when a constraint is not satisfied.

The repository is kept clean. It contains runnable source code, tests, verified case scripts, a Windows setup script, and two user-facing launchers. It does not contain private API keys, downloaded installers, local run artifacts, internal reports, or obsolete implementation branches.

## Quick Start

After cloning the repository, ordinary users only need these two entry files:

- `run_cli_agent.bat`: command-line agent. Double-click it, enter a requirement, and wait for the result.
- `run_web_agent.bat`: web-page agent. Double-click it, then use the browser page at `http://127.0.0.1:8765/`.

On first run, either launcher will call `scripts/setup_windows.ps1`. The setup script creates `.venv`, installs dependencies, and asks for:

- OpenAI-compatible API base URL
- API key
- model name

The API information is written only to local `.env`. It is not included in GitHub.

## Requirements

- Windows
- Python `3.12+`
- local `Ansys Electronics Desktop / Maxwell`
- an OpenAI-compatible LLM API endpoint and key

## What It Can Run

Verified named Maxwell 2D families:

- `electromagnet_2d`
- `capacitor_2d`
- `coaxial_capacitor_2d`
- `busbar_2d`
- `solenoid_2d`
- `inductor_2d`
- `transformer_2d`

Verified generic 2D path:

- unknown 2D tasks that can be decomposed into supported primitives such as circles, rectangles, annular cuts, rectangular frames, regions, and subtract relations

## How It Works

The execution chain is:

`requirement -> AI semantic parse -> local semantic validation -> primitive/object graph -> local Maxwell execution -> output extraction -> constraint evaluation -> feedback revision -> primitive learning if needed`

The LLM handles semantic interpretation and feedback revision. The local executor handles deterministic Maxwell operations, result extraction, and hard constraint checks.

## Developer Commands

These commands are mainly for debugging and verification:

```powershell
python -m maxwell_agent.cli probe-env
python -m maxwell_agent.cli smoke-llm
python -m maxwell_agent.cli demo "Design a 24V DC electromagnet with a 2mm air gap, current no higher than 2A, and maximize force within a compact size."
python scripts/run_2d_regression.py
python scripts/run_generic_2d_regression.py
pytest tests/test_models.py -q
pytest tests/test_demo.py -q
```

## Repository Layout

- `src/maxwell_agent/`: runnable implementation
- `tests/`: automated verification
- `scripts/`: setup script and verified case runners
- `run_cli_agent.bat`: simple command-line launcher
- `run_web_agent.bat`: simple web-page launcher

## License

MIT
