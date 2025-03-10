#!/usr/bin/env bash
set -e

# Parse command line arguments
SKIP_INSTALL=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-install) SKIP_INSTALL=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Color function definitions
color() {
    local color_name=$1
    local text=$2
    case $color_name in
        "red")     echo -e "\033[91m$text\033[0m" ;;
        "green")   echo -e "\033[92m$text\033[0m" ;;
        "yellow")  echo -e "\033[93m$text\033[0m" ;;
        "blue")    echo -e "\033[94m$text\033[0m" ;;
        "magenta") echo -e "\033[95m$text\033[0m" ;;
        "cyan")    echo -e "\033[96m$text\033[0m" ;;
        "white")   echo -e "\033[97m$text\033[0m" ;;
        "black")   echo -e "\033[30m$text\033[0m" ;;
        "gray")    echo -e "\033[90m$text\033[0m" ;;
        "purple")  echo -e "\033[35m$text\033[0m" ;;
        "orange")  echo -e "\033[38;5;208m$text\033[0m" ;;
        "pink")    echo -e "\033[38;5;205m$text\033[0m" ;;
        "brown")   echo -e "\033[38;5;130m$text\033[0m" ;;
        "lime")    echo -e "\033[38;5;118m$text\033[0m" ;;
        "teal")    echo -e "\033[38;5;31m$text\033[0m" ;;
        "maroon")  echo -e "\033[38;5;88m$text\033[0m" ;;
        "navy")    echo -e "\033[38;5;17m$text\033[0m" ;;
        "olive")   echo -e "\033[38;5;142m$text\033[0m" ;;
        "coral")   echo -e "\033[38;5;209m$text\033[0m" ;;
        "gold")    echo -e "\033[38;5;220m$text\033[0m" ;;
        "violet")  echo -e "\033[38;5;165m$text\033[0m" ;;
        "indigo")  echo -e "\033[38;5;54m$text\033[0m" ;;
        "crimson") echo -e "\033[38;5;160m$text\033[0m" ;;
        *)         echo "$text" ;;
    esac
}

# Cleanup function to kill background processes
cleanup() {
    echo "$(color white "[run_demo] 💥 Cleaning up processes...")"
    
    # Kill all running processes and their children
    for pid in $HOST_PID $LANDMARK_A_PID $LANDMARK_B_PID $LANDMARK_C_PID; do
        if [ ! -z "$pid" ] && ps -p $pid > /dev/null 2>&1; then
            echo "$(color white "[run_demo] Stopping process $pid and its children...")"
            pkill -P $pid 2>/dev/null || true
            kill -9 $pid 2>/dev/null || true
        fi
    done
    
    echo "$(color white "[run_demo] Checking for remaining Python processes...")"
    pkill -f "python3 host/host.py" 2>/dev/null || true
    pkill -f "python3 landmark/landmark.py" 2>/dev/null || true
    
    # Release ports explicitly
    for port in 5000 5001 5002 5003; do
        fuser -k $port/tcp 2>/dev/null || true
    done
    
    exit 0
}

# Set up trap for script interruption (Ctrl+C) and normal exit
trap cleanup EXIT SIGINT SIGTERM

# Check if .venv exists; create it if not
if [ ! -d ".venv" ]; then
    echo "$(color white "[run_demo] 🌐 .venv not found, creating virtual environment...")"
    python3 -m venv .venv
else
    echo "$(color white "[run_demo] 🚀 Using existing virtual environment .venv")"
fi

echo "$(color white "[run_demo] 🔑 Activating virtual environment...")"
source .venv/bin/activate

if [ "$SKIP_INSTALL" = false ]; then
    echo "$(color white "[run_demo] 📦 Installing Python dependencies from requirements.txt...")"
    pip install -r requirements.txt || {
        echo "Error installing Python dependencies."
        deactivate
        exit 1
    }

    echo "$(color white "[run_demo] Installing and building frontend...")"
    pushd frontend
    bun install
    bun run build
    popd

else
    echo "$(color white "[run_demo] 📦 Skipping dependency installation...")"
fi

echo "$(color white "[run_demo] 📁 Creating required directories and files...")"
touch verifier.log
chmod 666 verifier.log

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "$(color red "[run_demo] ❌ Error: Port $port is already in use")"
        exit 1
    fi
}

# Function to start a process with a colored prefix
start_process() {
    local color_name=$1
    local name=$2
    local command=$3
    local pid_var=$4
    local port=$5

    if [ ! -z "$port" ]; then
        check_port $port
    fi

    echo "$(color white "[run_demo] 🌀 Starting ${name}...")"
    
    if [[ $command == LANDMARK_* ]]; then
        local env_vars=""
        local actual_command=""
        IFS=' ' read -ra ADDR <<< "$command"
        for i in "${ADDR[@]}"; do
            if [[ $i == *"="* ]]; then
                env_vars="$env_vars $i"
            else
                actual_command="$actual_command $i"
            fi
        done
        eval "env $env_vars $actual_command 2>&1 | while IFS= read -r line; do echo -e \"\$(color $color_name \"[$name]\") \$line\"; done &"
    else
        eval "$command 2>&1 | while IFS= read -r line; do echo -e \"\$(color $color_name \"[$name]\") \$line\"; done &"
    fi
    eval "${pid_var}=\$!"
}

# Start host server
start_process "yellow" "Host server (Flask)" "python3 host/host.py" "HOST_PID" "5000"

# Give the host server a moment to start
sleep 2

# Start landmarks with distinct colors
# start_process "navy" "Landmark A (port 5001)" "LANDMARK_SERVER_MODE=api LANDMARK_HOSTNAME=127.0.0.1:5001 LANDMARK_PORT=5001 python3 landmark/landmark.py" "LANDMARK_A_PID" "5001"
# start_process "maroon" "Landmark B (port 5002)" "LANDMARK_SERVER_MODE=api LANDMARK_HOSTNAME=127.0.0.1:5002 LANDMARK_PORT=5002 python3 landmark/landmark.py" "LANDMARK_B_PID" "5002"
# start_process "olive" "Landmark C (port 5003)" "LANDMARK_SERVER_MODE=api LANDMARK_HOSTNAME=127.0.0.1:5003 LANDMARK_PORT=5003 python3 landmark/landmark.py" "LANDMARK_C_PID" "5003"
for landmark in $(jq -c '.landmarks[]' landmark/landmark_registry.json); do
    hostname=$(echo $landmark | jq -r '.hostname')
    port=$(echo $hostname | cut -d':' -f2)
    
    if [[ $hostname == $CURRENT_HOSTNAME* ]]; then
        start_process "navy" "Landmark A (port $port)" "LANDMARK_SERVER_MODE=api LANDMARK_HOSTNAME=$hostname LANDMARK_PORT=$port python3 landmark/landmark.py" "LANDMARK_A_PID" "$port"
    fi
done


# Give services a moment to start
sleep 2

echo "$(color white "\n\n[run_demo] ✨ All services are running:")"
echo

echo

# Keep script running until user stops it
wait        