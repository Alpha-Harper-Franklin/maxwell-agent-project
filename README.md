# Maxwell Agent Project

`Maxwell Agent Project` is a Windows-first `Ansys Maxwell 2D` agent. It converts Chinese engineering requirements into local Maxwell execution, result extraction, constraint evaluation, feedback-driven revision, and primitive learning.

This repository is intentionally kept clean. It contains only:

- runnable source code
- automated tests
- verified successful case scripts

It does not include installers, downloaded packages, local run artifacts, private configuration, internal reports, or obsolete implementation branches.

## Current Capability

The current open-source version focuses on `Maxwell 2D` tasks that can be represented through supported primitives and solved through a semantic-first execution chain:

`Chinese requirement -> AI semantic parse -> local semantic validation -> primitive/object graph -> local Maxwell execution -> output extraction -> constraint evaluation -> feedback revision -> primitive learning if needed`

Verified named 2D families:

- `electromagnet_2d`
- `capacitor_2d`
- `coaxial_capacitor_2d`
- `busbar_2d`
- `solenoid_2d`
- `inductor_2d`
- `transformer_2d`

Verified generic 2D path:

- unknown 2D tasks that can be decomposed into supported primitives such as circles, rectangles, annular cuts, rectangular frames, regions, and subtract relations

## Repository Layout

- `src/maxwell_agent/`: latest runnable implementation
- `tests/`: automated verification
- `scripts/`: verified successful case runners

## Requirements

- Windows
- Python `3.12+`
- local `Ansys Electronics Desktop / Maxwell`
- a compatible LLM API endpoint and key

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
Copy-Item .env.example .env
```

Fill `.env` with your own values:

- `CODEXA_BASE_URL`
- `CODEXA_API_KEY`
- `CODEXA_MODEL`
- `MAXWELL_VERSION` if you want to force a version

## Basic Checks

```powershell
python -m maxwell_agent.cli probe-env
python -m maxwell_agent.cli smoke-llm
```

## Run

```powershell
python -m maxwell_agent.cli demo "做一个24V直流电磁铁，气隙2mm，电流不超过2A，尽量提高吸力。"
```

## Verified Successful Cases

Run verified named 2D cases:

```powershell
python scripts/run_2d_regression.py
```

Run verified generic 2D cases:

```powershell
python scripts/run_generic_2d_regression.py
```

## Tests

```powershell
pytest tests/test_models.py -q
pytest tests/test_demo.py -q
```

## License

MIT
