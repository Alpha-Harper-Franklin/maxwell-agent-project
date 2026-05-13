param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot

Write-Host "Maxwell Agent Windows setup"
Write-Host "Project root: $ProjectRoot"

function Resolve-PythonCommand {
    $candidates = @(
        @("python"),
        @("py", "-3.12"),
        @("py", "-3.13")
    )
    foreach ($candidate in $candidates) {
        $exe = $candidate[0]
        $args = @($candidate | Select-Object -Skip 1)
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
            continue
        }
        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $versionText = & $exe @args -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousErrorActionPreference
        if ($exitCode -eq 0 -and $versionText) {
            $parts = $versionText.Trim().Split(".")
            $major = [int]$parts[0]
            $minor = [int]$parts[1]
            if ($major -eq 3 -and $minor -ge 12) {
                return @{Exe = $exe; Args = $args}
            }
        }
    }
    throw "Python 3.12+ was not found. Install Python 3.12+ or make it available as 'python'."
}

$pythonCommand = Resolve-PythonCommand

if (-not (Test-Path ".venv")) {
    & $pythonCommand.Exe @($pythonCommand.Args + @("-m", "venv", ".venv"))
}

$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment was not created correctly."
}

& $python -m pip install --upgrade pip
& $python -m pip install -e .

$envPath = Join-Path $ProjectRoot ".env"
if (-not (Test-Path $envPath)) {
    $baseUrl = Read-Host "Enter your OpenAI-compatible API base URL, for example https://example.com/v1"
    $apiKey = Read-Host "Enter your API key"
    $model = Read-Host "Enter model name [gpt-5.4]"
    if (-not $model.Trim()) {
        $model = "gpt-5.4"
    }

    $lines = @(
        "PROJECT_ROOT=$ProjectRoot",
        "CODEXA_BASE_URL=$($baseUrl.Trim())",
        "CODEXA_API_KEY=$($apiKey.Trim())",
        "CODEXA_MODEL=$($model.Trim())",
        "CODEXA_REASONING_EFFORT=high",
        "CODEXA_TIMEOUT_S=180",
        "MAXWELL_VERSION=",
        "MAXWELL_NON_GRAPHICAL=true",
        "SCRIPT_EXECUTION_TIMEOUT_S=240",
        "SCRIPT_MAX_REPAIRS=2",
        "DESIGN_FEEDBACK_MAX_ITERS=2"
    )
    Set-Content -LiteralPath $envPath -Value $lines -Encoding UTF8
    Write-Host ".env created."
} else {
    Write-Host ".env already exists. Existing API configuration was not overwritten."
}

Write-Host ""
Write-Host "Setup complete. Try:"
Write-Host ".\.venv\Scripts\python.exe -m maxwell_agent.cli probe-env"
Write-Host ".\.venv\Scripts\python.exe -m maxwell_agent.cli smoke-llm"
