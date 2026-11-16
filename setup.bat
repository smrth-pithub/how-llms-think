@echo off
REM One-time setup on a new laptop:
REM   1. builds the Python environment (.venv)
REM   2. downloads the big files that aren't in the GitHub repo (models + llama.cpp, about 3.3 GB)
REM Needs: Windows, Python 3.11-3.14 installed (python.org), and internet for this step only.
REM Safe to run again: it skips whatever is already there. After that, everything runs offline.
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import mediapipe, cv2, fastapi, uvicorn, httpx" >nul 2>&1
  if not errorlevel 1 (
    echo Python environment: already set up.
    goto :assets
  )
  echo The .venv folder came from another laptop and doesn't work here. Rebuilding it...
  rmdir /s /q ".venv"
)

set "PY="
for %%V in (3.14 3.13 3.12 3.11) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 14) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo.
  echo Python 3.11 to 3.14 was not found on this laptop.
  echo Install Python 3.14 from https://www.python.org/downloads/windows/
  echo then run setup.bat again.
  pause
  exit /b 1
)

echo Using: %PY%
%PY% -m venv .venv || goto :fail
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
".venv\Scripts\python.exe" -c "import mediapipe, cv2, fastapi, torch; print('Python environment: OK')" || goto :fail

:assets
echo.
".venv\Scripts\python.exe" -m tools.download_assets || goto :fail
echo.
echo Setup OK. Next: check_camera.bat, then start.bat
pause
exit /b 0

:fail
echo.
echo Setup failed (see the messages above). Check the internet connection and run setup.bat again.
pause
exit /b 1
