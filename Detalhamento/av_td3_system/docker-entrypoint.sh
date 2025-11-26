#!/bin/bash
set -e

# Source ROS 2 Humble environment (Ubuntu 22.04)
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo "✅ Sourced ROS 2 Humble environment"
else
    echo "⚠️  ROS 2 Humble not found - ROS features will be disabled"
fi

# Start CARLA server in background (headless mode)
echo "Starting CARLA server..."
/home/carla/carla/CarlaUE4.sh \
    -RenderOffScreen \
    -nosound \
    -quality-level=Low \
    -world-port=2000 \
    -timeout=10s &

CARLA_PID=$!
echo "CARLA server started with PID: $CARLA_PID"

# Wait for CARLA to initialize
echo "Waiting for CARLA to initialize (15 seconds)..."
sleep 15

# Test CARLA connection
python3 -c "import carla; client = carla.Client('localhost', 2000); client.set_timeout(10.0); print('CARLA Version:', client.get_server_version())" || {
    echo "ERROR: CARLA server not responding"
    kill $CARLA_PID
    exit 1
}

echo "CARLA server ready!"

# Execute the provided command
echo "Executing command: $@"
exec "$@"

# Cleanup function
cleanup() {
    echo "Shutting down CARLA server..."
    kill $CARLA_PID || true
}
trap cleanup EXIT
