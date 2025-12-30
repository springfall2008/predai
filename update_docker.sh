#!/bin/bash

# Find the running predai container
CONTAINER_ID=$(docker ps -q -f ancestor=predai:latest)

if [ -z "$CONTAINER_ID" ]; then
    echo "No running predai container found. Start it first with ./run_docker.sh"
    exit 1
fi

echo "Found container: $CONTAINER_ID"
echo "Copying predai.py to container..."

# Copy the updated predai.py into the container
docker cp predai/rootfs/predai.py "$CONTAINER_ID":/predai.py

echo "File copied successfully!"
echo "You may need to restart the container for changes to take effect:"
echo "  docker restart $CONTAINER_ID"
