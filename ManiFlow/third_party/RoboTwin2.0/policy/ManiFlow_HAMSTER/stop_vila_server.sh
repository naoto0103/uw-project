#!/bin/bash
#
# Stop the VILA server started by start_vila_server.sh
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PID_FILE="$SCRIPT_DIR/server_pid.txt"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 $PID 2>/dev/null; then
        echo "Stopping VILA server (PID: $PID)..."
        kill $PID
        sleep 2

        # Force kill if still running
        if kill -0 $PID 2>/dev/null; then
            echo "Force killing..."
            kill -9 $PID
        fi

        echo "Server stopped."
    else
        echo "Server not running (PID: $PID)"
    fi
    rm -f "$PID_FILE"
else
    echo "No PID file found. Server may not be running."
    echo ""
    echo "To manually stop, find and kill the process:"
    echo "  ps aux | grep server.py"
fi
