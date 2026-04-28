param(
    [string]$ProjectRoot = "",
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"

if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Path $PSScriptRoot -Parent
}

$ProjectRoot = [string]$ProjectRoot
$ProjectRoot = $ProjectRoot.Trim().Trim('"')
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path

$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$envFile = Join-Path $ProjectRoot ".env"
$logDir = Join-Path $ProjectRoot "logs"
$stdoutLog = Join-Path $logDir "web_demo_stdout.log"
$stderrLog = Join-Path $logDir "web_demo_stderr.log"

if (-not (Test-Path $python)) {
    Write-Error "Missing Python environment: $python"
}

if (-not (Test-Path $envFile)) {
    Write-Error "Missing config file: $envFile"
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($existing) {
    try {
        Stop-Process -Id $existing.OwningProcess -Force -ErrorAction Stop
        Start-Sleep -Seconds 1
    } catch {
        Write-Warning ("Failed to stop existing process on port {0}: {1}" -f $Port, $_.Exception.Message)
    }
}

Remove-Item $stdoutLog,$stderrLog -Force -ErrorAction SilentlyContinue

$proc = Start-Process `
    -FilePath $python `
    -ArgumentList "-m","maxwell_agent.cli","serve","--host","127.0.0.1","--port",$Port `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 3
        if ($resp.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch {
    }

    if ($proc.HasExited) {
        break
    }
}

if (-not $ready) {
    $stderr = if (Test-Path $stderrLog) { Get-Content $stderrLog -Raw -ErrorAction SilentlyContinue } else { "" }
    $stdout = if (Test-Path $stdoutLog) { Get-Content $stdoutLog -Raw -ErrorAction SilentlyContinue } else { "" }
    $message = @(
        "Web demo failed to start on http://127.0.0.1:$Port/"
        "Process exited: $($proc.HasExited)"
        "Stdout:"
        $stdout
        "Stderr:"
        $stderr
    ) -join [Environment]::NewLine
    Write-Error $message
}

Start-Process "http://127.0.0.1:$Port/"
Write-Output "Web demo is running at http://127.0.0.1:$Port/"
