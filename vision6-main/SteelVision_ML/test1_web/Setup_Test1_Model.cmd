@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-model\Scripts\python.exe" (
    python -m venv .venv-model
    if errorlevel 1 exit /b 1
)
set "TEST1_INDEX=https://download.pytorch.org/whl/cu128"
if /i "%~1"=="cpu" set "TEST1_INDEX=https://download.pytorch.org/whl/cpu"
".venv-model\Scripts\python.exe" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url "%TEST1_INDEX%"
if errorlevel 1 exit /b 1
".venv-model\Scripts\python.exe" -m pip install -r requirements-model.txt
exit /b %errorlevel%
