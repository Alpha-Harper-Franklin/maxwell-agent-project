@echo off
setlocal
cd /d "%~dp0"

echo.
echo Initializing Maxwell Agent CLI...
if not exist ".venv\Scripts\python.exe" (
  powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\setup_windows.ps1"
  if errorlevel 1 (
    echo.
    echo Setup failed. Press any key to exit.
    pause >nul
    exit /b 1
  )
)

echo.
echo Maxwell Agent CLI
echo Type a Chinese or English requirement, then press Enter.
echo Example: make a 24V DC electromagnet, air gap 2mm, current no higher than 2A, maximize force.
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
