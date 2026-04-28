param(
    [string]$RootDir = "F:\codex-new-relay\unified-tmp-v3",
    [string]$TargetName = "ELECTRONICSSTUDENT_2025R2_WINX64.zip",
    [string]$FinalPath = "F:\maxwell_agent_project\downloads\ELECTRONICSSTUDENT_2025R2_WINX64.zip",
    [string]$LogPath = "F:\maxwell_agent_project\logs\ansys_download_watch.log",
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $LogPath -Value "[$timestamp] $Message"
}

function Get-DownloadCandidate {
    param(
        [string]$SearchRoot,
        [string]$ExpectedName
    )
    if (-not (Test-Path -LiteralPath $SearchRoot)) {
        return $null
    }

    $candidates = Get-ChildItem -LiteralPath $SearchRoot -Recurse -Force -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -eq $ExpectedName -or
            $_.Name -eq "$ExpectedName.crdownload" -or
            ($_.Name -like "*$ExpectedName*" -and $_.DirectoryName -like "*playwright-artifacts*") -or
            ($_.Extension -eq ".crdownload" -and $_.DirectoryName -like "*playwright-artifacts*")
        } |
        Sort-Object LastWriteTime -Descending

    return $candidates | Select-Object -First 1
}

New-Item -ItemType Directory -Path (Split-Path -Parent $LogPath) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $FinalPath) -Force | Out-Null

Write-Log "Watcher started."
Write-Log "RootDir=$RootDir"
Write-Log "TargetName=$TargetName"
Write-Log "FinalPath=$FinalPath"

$lastSize = -1
$stableCount = 0

while ($true) {
    $finalExists = Test-Path -LiteralPath $FinalPath
    $candidate = Get-DownloadCandidate -SearchRoot $RootDir -ExpectedName $TargetName

    if ($finalExists) {
        Write-Log "Final file already exists. Exiting watcher."
        break
    }

    $currentSize = $null
    if ($candidate) {
        $currentSize = $candidate.Length
    }

    if ($currentSize -ne $null) {
        if ($currentSize -eq $lastSize) {
            $stableCount += 1
        } else {
            $stableCount = 0
            $lastSize = $currentSize
            Write-Log ("Progress candidate={0} size={1:N0} bytes" -f $candidate.FullName, $currentSize)
        }
    }

    if ($candidate -and $candidate.Extension -ne ".crdownload") {
        try {
            Copy-Item -LiteralPath $candidate.FullName -Destination $FinalPath -Force
            Write-Log "Completed candidate was copied to final zip."
            break
        } catch {
            Write-Log "Copy attempt failed: $($_.Exception.Message)"
        }
    }

    if ($candidate -and $candidate.Extension -eq ".crdownload" -and $stableCount -ge 20) {
        Write-Log "Candidate size was stable for a long period; leaving watcher running."
        $stableCount = 0
    }

    Start-Sleep -Seconds $PollSeconds
}

Write-Log "Watcher exited."
