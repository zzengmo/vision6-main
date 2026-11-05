@echo off
setlocal
cd /d "%~dp0"
set "STEELVISION_PY=.venv\Scripts\python.exe"
if not exist "%STEELVISION_PY%" set "STEELVISION_PY=..\.venv\Scripts\python.exe"
if not exist "%STEELVISION_PY%" (
    echo Python environment missing. Run Setup_Test1_Web.cmd first.
    pause
    exit /b 1
)
"%STEELVISION_PY%" -c "import json,urllib.request; h=json.load(urllib.request.urlopen('http://127.0.0.1:8502/api/health',timeout=1)); assert h['app']=='steelvision-react' and h['version']=='video-home-v2'" >nul 2>&1
if not errorlevel 1 (
    start "" "http://127.0.0.1:8502"
    exit /b 0
)
"%STEELVISION_PY%" -c "import json,urllib.request; h=json.load(urllib.request.urlopen('http://127.0.0.1:8502/api/health',timeout=1)); assert h['app']=='steelvision-react'" >nul 2>&1
if not errorlevel 1 (
    echo An older SteelVision server is running. Close its window with Ctrl+C, then run again.
    pause
    exit /b 1
)
"%STEELVISION_PY%" server.py --port 8502 --open-browser
if errorlevel 1 pause
