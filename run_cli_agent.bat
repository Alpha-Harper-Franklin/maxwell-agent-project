@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  powershell -ExecutionPolicy Bypass -File ".\scripts\setup_windows.ps1"
  if errorlevel 1 (
    echo.
    echo Setup failed. Press any key to exit.
    pause >nul
    exit /b 1
  )
)

echo.
echo Maxwell Agent CLI
echo Enter a requirement and press Enter.
echo Example: Design a 24V DC electromagnet with a 2mm air gap and current no higher than 2A.
echo.
set /p REQUIREMENT=Requirement: 
if "%REQUIREMENT%"=="" (
  echo Requirement is empty. Press any key to exit.
  pause >nul
  exit /b 1
)

".venv\Scripts\python.exe" -m maxwell_agent.cli demo "%REQUIREMENT%"
echo.
echo Finished. Press any key to exit.
pause >nul
