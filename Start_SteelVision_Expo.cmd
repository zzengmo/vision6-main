@echo off
setlocal
cd /d "%~dp0mobile"
where node >nul 2>&1
if errorlevel 1 (
    echo Node.js is required. Install a current Node.js LTS version.
    pause
    exit /b 1
)
node -e "try{const p=require('./package.json');const v=require('expo/package.json').version;process.exit(v.split('.')[0]===p.dependencies.expo.match(/\d+/)[0]?0:1)}catch{process.exit(1)}" >nul 2>&1
if errorlevel 1 (
    echo Installing matching Expo app packages...
    call npm.cmd ci --no-audit --no-fund
    if errorlevel 1 (
        pause
        exit /b 1
    )
)
node scripts/start.mjs %*
if errorlevel 1 pause
