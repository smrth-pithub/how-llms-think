@echo off
REM Starts the demo app and opens it in the browser.
REM   start.bat               (webcam 0)
REM   start.bat --camera 1    (if the external webcam is camera 1: run check_camera.bat to find out)
REM   start.bat --pose lite   (faster pose model if the camera feed is choppy)
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First time on this laptop? Run setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m app.server %*
pause
