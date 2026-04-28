@echo off
setlocal

cd /d "%~dp0"
title Maxwell Agent Demo Launcher

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

echo ==========================================
echo Maxwell Agent Demo Launcher
echo ==========================================
echo 1. Web Demo
echo 2. CLI Demo
echo 3. Exit
echo.

set "MODE="
set /p MODE=Select 1/2/3: 

if "%MODE%"=="1" goto PAGE
if "%MODE%"=="2" goto CLI
if "%MODE%"=="3" goto END

echo Invalid option.
pause
goto END

:PAGE
echo Starting web demo...
powershell -NoProfile -ExecutionPolicy Bypass -File "%cd%\scripts\start_web_demo.ps1"
if errorlevel 1 (
  echo Web demo failed to start.
  echo See logs:
  echo   %cd%\logs\web_demo_stdout.log
  echo   %cd%\logs\web_demo_stderr.log
  pause
  goto END
)
start "" "http://127.0.0.1:8765/"
goto END

:CLI
echo.
set "REQUIREMENT="
set /p REQUIREMENT=Enter requirement: 
if "%REQUIREMENT%"=="" (
  echo Requirement cannot be empty.
  pause
  goto END
)
call ".venv\Scripts\activate.bat"
".venv\Scripts\python.exe" -m maxwell_agent.cli demo "%REQUIREMENT%"
echo.
pause
goto END

:END
endlocal
