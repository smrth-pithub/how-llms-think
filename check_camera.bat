@echo off
REM Lists which camera numbers work.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First time on this laptop? Run setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m tools.check_camera
pause
