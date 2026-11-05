@echo off
setlocal
cd /d "%~dp0"
set "TEST1_PY=.venv\Scripts\python.exe"
if not exist "%TEST1_PY%" set "TEST1_PY=..\.venv\Scripts\python.exe"
if not exist "%TEST1_PY%" (
    echo Python environment missing. Run Setup_Test1_Web.cmd first.
    pause
    exit /b 1
)
"%TEST1_PY%" -c "import urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8503/_stcore/health',timeout=1); assert r.read().strip()==b'ok'" >nul 2>&1
if not errorlevel 1 (
    start "" "http://127.0.0.1:8503"
    exit /b 0
)
"%TEST1_PY%" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8503 --browser.gatherUsageStats false
if errorlevel 1 pause
