@echo off
setlocal
cd /d "%~dp0"
set "DEEP_PY=.venv-model\Scripts\python.exe"
if not exist "%DEEP_PY%" set "DEEP_PY=..\.venv-yolo\Scripts\python.exe"
if not exist "%DEEP_PY%" (
    echo Model Python missing. Check settings.json or run Setup_Test1_Model.cmd.
    pause
    exit /b 1
)
"%DEEP_PY%" deep_worker.py audit
if errorlevel 1 pause
exit /b %errorlevel%
