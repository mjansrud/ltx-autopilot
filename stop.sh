#!/bin/bash
# Stop the autopilot pipeline without killing other python processes (like ComfyUI)
if [ -f workspace/current/.pid ]; then
    PID=$(cat workspace/current/.pid)
    taskkill //F //PID $PID //T 2>/dev/null
    rm -f workspace/current/.pid
    echo "Killed pipeline PID $PID"
else
    echo "No pipeline PID found. Killing node (lustpress) only."
fi
# Always kill lustpress
taskkill //F //IM "node.exe" //T 2>/dev/null
echo "Done"
