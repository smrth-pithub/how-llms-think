@echo off
REM Times both language models (close the demo app first).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First time on this laptop? Run setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m tools.speedtest
pause
