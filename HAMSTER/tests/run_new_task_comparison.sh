#!/bin/bash

set -e

echo "============================================================"
echo "New Task Comparison: VILA vs Qwen3"
echo "Task: Pick up the apple and put it behind the hammer"
echo "============================================================"

# Check if servers are running
echo ""
echo "Checking servers..."

if ! curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "ERROR: VILA server is not running on port 8000"
    echo "Please start VILA server first:"
    echo "  cd /home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER"
    echo "  ./setup_server.sh"
    exit 1
fi
echo "✓ VILA server (port 8000) is running"

if ! curl -s http://127.0.0.1:8001/health > /dev/null 2>&1; then
    echo "ERROR: Qwen3 server is not running on port 8001"
    echo "Please start Qwen3 server first:"
    echo "  cd /home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER"
    echo "  ./setup_qwen3_server.sh"
    exit 1
fi
echo "✓ Qwen3 server (port 8001) is running"

echo ""
echo "============================================================"
echo "Step 1: Generate path with VILA"
echo "============================================================"
cd /home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER
python3 tests/test_new_task_vila.py

echo ""
echo "============================================================"
echo "Step 2: Generate path with Qwen3"
echo "============================================================"
python3 tests/test_new_task_qwen3.py

echo ""
echo "============================================================"
echo "Step 3: Create comparison visualization"
echo "============================================================"
python3 tests/visualize_new_task_comparison.py

echo ""
echo "============================================================"
echo "DONE!"
echo "============================================================"
echo "Check the results:"
echo "  - results/vila_new_task_path.pkl"
echo "  - results/qwen3_new_task_path.pkl"
echo "  - results/visualizations/new_task_comparison.png"
