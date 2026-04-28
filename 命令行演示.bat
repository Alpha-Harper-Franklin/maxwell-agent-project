@echo off
setlocal

cd /d "%~dp0"
title Maxwell Agent CLI Demo

if not exist ".venv\Scripts\python.exe" (
  echo Missing Python environment: %~dp0.venv\Scripts\python.exe
  pause
  exit /b 1
)

if not exist ".env" (
  echo Missing config file: %~dp0.env
  pause
  exit /b 1
)

set "REQUIREMENT="
set /p REQUIREMENT=Enter requirement: 
if "%REQUIREMENT%"=="" (
  echo Requirement cannot be empty.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m maxwell_agent.cli demo "%REQUIREMENT%"
echo.
pause

endlocal
