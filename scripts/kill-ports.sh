#!/bin/bash

# Script to kill processes on specific ports before starting dev servers

echo "Checking and killing processes on port 3009..."
# Kill processes on port 3009 (if any)
PORT_3009=$(lsof -t -i:3009)
if [ ! -z "$PORT_3009" ]; then
    echo "Killing processes on port 3009 (PIDs: $PORT_3009)"
    kill $PORT_3009
    sleep 1  # Give time for the process to terminate
else
    echo "No processes found on port 3009"
fi

echo "Checking and killing processes on port 8880..."
# Kill processes on port 8880 (if any)
PORT_8880=$(lsof -t -i:8880)
if [ ! -z "$PORT_8880" ]; then
    echo "Killing processes on port 8880 (PIDs: $PORT_8880)"
    kill $PORT_8880
    sleep 1  # Give time for the process to terminate
else
    echo "No processes found on port 8880"
fi

# Kill processes on port 3008 (UI server) that should be stopped before starting
PORT_3008=$(lsof -t -i:3008)
if [ ! -z "$PORT_3008" ]; then
    echo "Killing processes on port 3008 (PIDs: $PORT_3008)"
    kill $PORT_3008
    sleep 1  # Give time for the process to terminate
else
    echo "No processes found on port 3008"
fi

echo "Port cleanup completed"