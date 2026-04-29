$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$tex = "maxwell_agent_academic_report_brief_20260429.tex"
& "C:\texlive\2024\bin\windows\xelatex.exe" -interaction=nonstopmode -halt-on-error $tex | Out-Host
& "C:\texlive\2024\bin\windows\xelatex.exe" -interaction=nonstopmode -halt-on-error $tex | Out-Host
