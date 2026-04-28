param(
    [string]$ZipPath = "F:\maxwell_agent_project\downloads\ELECTRONICSSTUDENT_2025R2_WINX64.zip",
    [string]$ExtractDir = "F:\maxwell_agent_project\temp_extract\ansys_electronics_student_2025R2",
    [string]$InstallDir = "F:\AnsysEM_Student_2025R2",
    [string]$LogPath = "F:\maxwell_agent_project\logs\ansys_extract_install.log",
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $LogPath -Value "[$timestamp] $Message"
}

function Wait-ForStableFile {
    param(
        [string]$Path,
        [int]$StableChecks = 3,
        [int]$IntervalSeconds = 60
    )

    $lastLength = -1
    $stable = 0
    while ($true) {
        if (-not (Test-Path -LiteralPath $Path)) {
            $stable = 0
            $lastLength = -1
            Start-Sleep -Seconds $IntervalSeconds
            continue
        }

        $currentLength = (Get-Item -LiteralPath $Path).Length
        if ($currentLength -eq $lastLength) {
            $stable += 1
            if ($stable -ge $StableChecks) {
                return
            }
        } else {
            $stable = 0
            $lastLength = $currentLength
            Write-Log ("Observed zip size={0:N0} bytes" -f $currentLength)
        }

        Start-Sleep -Seconds $IntervalSeconds
    }
}

New-Item -ItemType Directory -Path (Split-Path -Parent $LogPath) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $ZipPath) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $ExtractDir) -Force | Out-Null
New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null

Write-Log "Extract/install watcher started."
Write-Log "ZipPath=$ZipPath"
Write-Log "ExtractDir=$ExtractDir"
Write-Log "InstallDir=$InstallDir"

Wait-ForStableFile -Path $ZipPath -StableChecks 3 -IntervalSeconds $PollSeconds
Write-Log "Zip is present and stable. Starting extraction."

if (Test-Path -LiteralPath $ExtractDir) {
    Remove-Item -LiteralPath $ExtractDir -Recurse -Force
}
New-Item -ItemType Directory -Path $ExtractDir -Force | Out-Null

$sevenZip = "C:\Program Files\7-Zip\7z.exe"
if (Test-Path -LiteralPath $sevenZip) {
    & $sevenZip x "-o$ExtractDir" "-y" $ZipPath | Out-Null
    Write-Log "Extraction completed with 7-Zip."
} else {
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $ExtractDir -Force
    Write-Log "Extraction completed with Expand-Archive."
}

$setup = Get-ChildItem -LiteralPath $ExtractDir -Recurse -Force -File -Filter setup.exe -ErrorAction SilentlyContinue |
    Sort-Object FullName |
    Select-Object -First 1

if (-not $setup) {
    Write-Log "setup.exe was not found after extraction."
    throw "setup.exe was not found in extracted media."
}

Write-Log "Launching silent installer: $($setup.FullName)"
Write-Log "Official Ansys silent install syntax uses setup.exe -silent -install_dir <path>."

$arguments = @(
    "-silent",
    "-install_dir",
    $InstallDir
)

$process = Start-Process -FilePath $setup.FullName -ArgumentList $arguments -Wait -PassThru
Write-Log "Installer exited with code $($process.ExitCode)."
