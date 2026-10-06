#!/usr/bin/env bash
set -e

echo "==============================================================="
echo "  SolarBurst — ISRO Chandrayaan-2 XSM Solar Burst Platform"
echo "==============================================================="

# Verify dependencies
command -v python3 >/dev/null 2>&1 || { echo >&2 "[ERROR] python3 is required."; exit 1; }
command -v node >/dev/null 2>&1 || { echo >&2 "[ERROR] node is required."; exit 1; }

echo "[1/3] Ensuring solarburst package is installed..."
pip install -e . --no-deps >/dev/null 2>&1 || pip install -r requirements.txt

echo "[2/3] Checking Node.js backend dependencies..."
cd backend
if [ ! -d "node_modules" ]; then
    npm install
fi

echo "[3/3] Launching SolarBurst Platform on http://127.0.0.1:5000..."
node server.js
