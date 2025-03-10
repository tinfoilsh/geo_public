#!/usr/bin/env bash
set -e

# Cleanup function to kill background processes
cleanup() {
    echo -e "\033[91m[run_host] 💥 Cleaning up processes...\033[0m"
    if [ ! -z "$HOST_PID" ]; then
        kill $HOST_PID 2>/dev/null || true
    fi
    exit 0
}

# Set up trap for script interruption
trap cleanup EXIT SIGINT SIGTERM

# Check if .venv exists; create it if not
if [ ! -d ".venv" ]; then
    echo -e "\033[94m[run_host] 🌐 Creating virtual environment...\033[0m"
    python3 -m venv .venv
fi

echo -e "\033[94m[run_host] 🔑 Activating virtual environment...\033[0m"
source .venv/bin/activate

echo -e "\033[94m[run_host] 📦 Installing dependencies...\033[0m"
pip install -r requirements.txt

echo -e "\033[93m[run_host] 🚀 Starting host server on port 5001...\033[0m"
python3 wsgi.py

deactivate