# Maxwell Agent Project

`Maxwell Agent Project` is a Windows-first simulation agent that turns Chinese natural-language requirements into runnable `Ansys Maxwell` jobs.

The current public version focuses on a reproducible end-to-end loop:

1. Parse a requirement into structured simulation data.
2. Generate or repair a `PyAEDT` script.
3. Launch local `Ansys Maxwell`.
4. Save a real `.aedt` project.
5. Extract results and evaluate whether the requirement is satisfied.
6. When a design fails hard constraints, feed the failure back to the LLM and run another round.

## What It Can Do

The current codebase has verified execution paths for:

- `electromagnet_2d`
- `capacitor_2d`
- `transformer_2d`
- `inductor_2d`

This does **not** mean every Maxwell problem is already supported. The repository provides a general agent skeleton plus several working task families.

## Architecture

The runtime pipeline is:

`natural language -> requirement intake -> simulation_spec -> execution_plan -> generated PyAEDT script -> Maxwell -> .aedt -> outputs -> evaluation -> feedback iteration`

Main modules:

- `src/maxwell_agent/agent.py`: orchestration and feedback loop
- `src/maxwell_agent/llm_client.py`: LLM calls plus deterministic fallbacks
- `src/maxwell_agent/maxwell_executor.py`: script execution, Maxwell process control, artifact capture
- `src/maxwell_agent/evaluation.py`: requirement validation for executed jobs
- `src/maxwell_agent/script_validation.py`: static validation for generated scripts

## Requirements

This project expects a real local Maxwell installation. It does **not** bundle Ansys software.

Required:

- Windows
- Python `3.12+`
- `Ansys Electronics Desktop / Maxwell` installed locally
- A Codexa-compatible API endpoint and key

Recommended:

- `Ansys Electronics Desktop Student 2025 R2` or a compatible commercial install
- A dedicated Python virtual environment

## Quick Start

### 1. Clone the repository

```powershell
git clone <your-repo-url>
cd maxwell_agent_project
```

### 2. Create and activate a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

### 3. Create `.env`

Copy `.env.example` to `.env` and fill in your own values.

```powershell
Copy-Item .env.example .env
```

Minimum required variables:

- `CODEXA_BASE_URL`
- `CODEXA_API_KEY`
- `CODEXA_MODEL`
- `MAXWELL_VERSION` (optional if auto-detection is enough)

### 4. Verify the environment

```powershell
python -m maxwell_agent.cli probe-env
python -m maxwell_agent.cli smoke-llm
```

### 5. Run a CLI demo

```powershell
python -m maxwell_agent.cli demo "Design a 24V DC electromagnet with a 2 mm air gap, current not exceeding 2 A, and maximize force."
```

### 6. Start the local web demo

```powershell
python -m maxwell_agent.cli serve --host 127.0.0.1 --port 8765
```

Then open:

- `http://127.0.0.1:8765/`

## Repository Layout

- `src/maxwell_agent/`: source code
- `tests/`: regression tests
- `scripts/`: local launch and helper scripts
- `workspace/`: runtime artifacts for each run, including generated scripts and `.aedt` files

The following directories are intentionally excluded from version control because they contain machine-specific or private content:

- `workspace/`
- `artifacts/`
- `logs/`
- `downloads/`
- `temp_extract/`
- internal `docs/` report folders

## Example Commands

Plan only:

```powershell
python -m maxwell_agent.cli plan "Design a 2D parallel-plate capacitor with 1 mm spacing, 20 mm plate width, 100 V excitation, and report electric field and capacitance."
```

Run:

```powershell
python -m maxwell_agent.cli run "Design a 2D parallel-plate capacitor with 1 mm spacing, 20 mm plate width, 100 V excitation, and report electric field and capacitance."
```

List remote models:

```powershell
python -m maxwell_agent.cli list-models
```

## Notes on Public Release

- The public repo does not include API keys.
- The public repo does not include Ansys installers.
- The public repo does not include internal reports or copied project briefs.
- Generated run artifacts remain local in `workspace/`.

## License

MIT
