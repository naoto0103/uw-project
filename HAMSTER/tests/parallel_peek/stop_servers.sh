#!/bin/bash
#
# Stop all PEEK servers started by start_servers.sh
#
# Usage:
#   ./stop_servers.sh
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PID_FILE="$SCRIPT_DIR/server_pids.txt"

echo "============================================================"
echo "Stopping PEEK (VILA-3B) Servers"
echo "============================================================"

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found at $PID_FILE"
    echo "Trying to find and kill PEEK server processes..."

    # Try to find and kill any vila_server.py processes on ports 8010-8013
    pkill -f "vila_server.py --host 0.0.0.0 --port 801" 2>/dev/null && echo "Killed vila_server.py processes" || echo "No vila_server.py processes found"
    exit 0
fi

# Read PIDs and kill processes
KILLED=0
while read -r LINE; do
    PID=$(echo "$LINE" | cut -d' ' -f1)
    GPU_ID=$(echo "$LINE" | cut -d' ' -f2)
    PORT=$(echo "$LINE" | cut -d' ' -f3)

    if [ -n "$PID" ]; then
        if kill -0 $PID 2>/dev/null; then
            echo "Stopping server on GPU $GPU_ID (port $PORT, PID: $PID)..."
            kill $PID 2>/dev/null
            KILLED=$((KILLED + 1))
        else
            echo "Server on GPU $GPU_ID (port $PORT, PID: $PID) already stopped"
        fi
    fi
done < "$PID_FILE"

# Wait a moment for processes to terminate
sleep 2

# Force kill any remaining processes
while read -r LINE; do
    PID=$(echo "$LINE" | cut -d' ' -f1)
    if [ -n "$PID" ] && kill -0 $PID 2>/dev/null; then
        echo "Force killing PID $PID..."
        kill -9 $PID 2>/dev/null
    fi
done < "$PID_FILE"

# Clean up PID file
rm -f "$PID_FILE"

echo ""
echo "Stopped $KILLED server(s)"
echo "============================================================"
