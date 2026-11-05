@echo off
setlocal
set "STEELVISION_APP=%~dp0vision6-main\SteelVision_ML\test1_web"
if not exist "%STEELVISION_APP%\Start_Figma_Web.cmd" (
    echo SteelVision application folder is missing.
    pause
    exit /b 1
)
call "%STEELVISION_APP%\Start_Figma_Web.cmd"
exit /b %errorlevel%
