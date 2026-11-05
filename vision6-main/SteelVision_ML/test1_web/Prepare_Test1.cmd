@echo off
setlocal
cd /d "%~dp0"
set "TEST1_UI=.venv\Scripts\python.exe"
if not exist "%TEST1_UI%" set "TEST1_UI=..\.venv\Scripts\python.exe"
set "TEST1_MODEL=.venv-model\Scripts\python.exe"
if not exist "%TEST1_MODEL%" set "TEST1_MODEL=..\.venv-yolo\Scripts\python.exe"
if not exist "%TEST1_UI%" (
    echo Run Setup_Test1_Web.cmd first.
    exit /b 1
)
"%TEST1_UI%" run_test1.py prepare-rf
if errorlevel 1 exit /b 1
if not exist "%TEST1_MODEL%" (
    echo U-Net environment missing. Run Setup_Test1_Model.cmd or configure settings.json.
    exit /b 1
)
"%TEST1_MODEL%" run_test1.py train-unet
exit /b %errorlevel%
