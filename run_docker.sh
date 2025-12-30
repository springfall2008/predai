#!/bin/bash
# Run the PredAI Docker container locally for testing

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOTFS_DIR="$SCRIPT_DIR/predai/rootfs"

# Create a temporary config directory with test configuration
TEMP_CONFIG=$(mktemp -d)
echo "Using temporary config directory: $TEMP_CONFIG"

# Copy predai.py and test config to temp directory
cp "$ROOTFS_DIR/predai.py" "$TEMP_CONFIG/"
cp "$ROOTFS_DIR/predai.yaml" "$TEMP_CONFIG/" 2>/dev/null || echo "sensors: []" > "$TEMP_CONFIG/predai.yaml"

echo ""
echo "Running Docker container with /config mapped to $TEMP_CONFIG"
echo "Press Ctrl+C to stop"
echo ""

# Run the container with config mapped
docker run --rm -it \
  -v "$TEMP_CONFIG:/config" \
  -e SUPERVISOR_TOKEN="test-token" \
  predai:latest

# Cleanup
rm -rf "$TEMP_CONFIG"
