#!/bin/bash
# Wait for GPU VRAM to drop below 5000 MiB

THRESHOLD=5000
CHECK_INTERVAL=10  # seconds

echo "Monitoring GPU VRAM usage..."
echo "Threshold: ${THRESHOLD} MiB"
echo "Check interval: ${CHECK_INTERVAL} seconds"
echo ""

while true; do
    VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[${TIMESTAMP}] VRAM usage: ${VRAM_USED} MiB"

    if [ "$VRAM_USED" -lt "$THRESHOLD" ]; then
        echo ""
        echo "VRAM usage dropped below ${THRESHOLD} MiB!"
        echo "Ready to start Stage 2"
        exit 0
    fi

    sleep $CHECK_INTERVAL
done
