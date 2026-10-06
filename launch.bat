@echo off
TITLE SolarBurst Platform Launcher
echo ===============================================================
echo   SolarBurst — ISRO Chandrayaan-2 XSM Solar Burst Platform
echo ===============================================================
echo.

REM Verify Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

REM Verify Node.js
node --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js not found. Please install Node.js 18+ and add it to PATH.
    pause
    exit /b 1
)

echo [1/3] Ensuring package is installed in editable mode...
pip install -e . --no-deps >nul 2>&1

echo [2/3] Checking Node backend dependencies...
cd backend
if not exist node_modules (
    echo Installing backend packages...
    npm install
)

echo [3/3] Starting SolarBurst Application on http://127.0.0.1:5000...
start http://127.0.0.1:5000
node server.js

pause
