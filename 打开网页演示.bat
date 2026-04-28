@echo off
setlocal

cd /d "%~dp0"
title Maxwell Agent Web Demo

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

echo Starting web demo...
powershell -NoProfile -ExecutionPolicy Bypass -File "%cd%\scripts\start_web_demo.ps1"
if errorlevel 1 (
  echo Web demo failed to start.
  echo See logs:
  echo   %cd%\logs\web_demo_stdout.log
  echo   %cd%\logs\web_demo_stderr.log
  pause
  exit /b 1
)

endlocal
