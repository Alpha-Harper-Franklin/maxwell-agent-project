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

set HOST=127.0.0.1
set PORT=8765
set URL=http://%HOST%:%PORT%/

echo.
echo Starting Maxwell Agent web page at %URL%
echo Keep this window open while using the page.
echo.
start "" "%URL%"
".venv\Scripts\python.exe" -m maxwell_agent.cli serve --host %HOST% --port %PORT%

echo.
echo Server stopped. Press any key to exit.
pause >nul
