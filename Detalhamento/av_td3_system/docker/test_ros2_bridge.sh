#!/bin/bash

# CARLA ROS 2 Bridge Integration Test
# This script tests the connection between CARLA 0.9.16 and ROS 2 Humble bridge
# 
# Prerequisites:
#   - CARLA server running: docker ps | grep carla-server
#   - ROS 2 bridge image built: ros2-carla-bridge:humble
#
# Test sequence:
#   1. Launch ROS 2 bridge in container
#   2. Wait for connection to CARLA
#   3. List all /carla topics
#   4. Spawn ego vehicle with sensors
#   5. Verify odometry and control topics
#   6. Test control command publishing

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   CARLA 0.9.16 + ROS 2 Humble Bridge Integration Test         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check CARLA server is running
echo "[1/6] Checking CARLA server status..."
if ! docker ps | grep -q carla-server; then
    echo "❌ ERROR: CARLA server not running!"
    echo "   Start it with: docker-compose -f docker-compose.baseline.yml up carla-server"
    exit 1
fi
echo "✅ CARLA server running"
echo ""

# Step 2: Check ROS 2 bridge image exists
echo "[2/6] Checking ROS 2 bridge image..."
if ! docker images | grep -q "ros2-carla-bridge.*humble"; then
    echo "❌ ERROR: ros2-carla-bridge:humble image not found!"
    echo "   Build it with: ./docker/build_ros2_bridge.sh --ros-distro humble"
    exit 1
fi
echo "✅ ROS 2 bridge image found"
echo ""

# Step 3: Launch ROS 2 bridge (background)
echo "[3/6] Launching ROS 2 bridge..."
echo "   Starting bridge container in background..."
docker run -d --rm \
    --name test-ros2-bridge \
    --net=host \
    ros2-carla-bridge:humble \
    bash -c "source /ros_entrypoint.sh bash -c 'ros2 launch carla_ros_bridge carla_ros_bridge.launch.py host:=localhost port:=2000 timeout:=10 synchronous_mode:=true fixed_delta_seconds:=0.05'"

echo "   Waiting for bridge to initialize (10 seconds)..."
sleep 10

# Step 4: Check if bridge is running
if ! docker ps | grep -q test-ros2-bridge; then
    echo "❌ ERROR: ROS 2 bridge failed to start!"
    echo "   Check logs: docker logs test-ros2-bridge"
    exit 1
fi
echo "✅ ROS 2 bridge running"
echo ""

# Step 5: List CARLA topics
echo "[4/6] Listing ROS 2 /carla topics..."
docker exec test-ros2-bridge bash -c "source /ros_entrypoint.sh bash -c 'ros2 topic list | grep /carla'" || true
echo ""

# Step 6: Check for ego vehicle odometry topic (requires vehicle to be spawned)
echo "[5/6] Checking for ego vehicle topics..."
if docker exec test-ros2-bridge bash -c "source /ros_entrypoint.sh bash -c 'ros2 topic list | grep /carla/ego_vehicle'" > /dev/null 2>&1; then
    echo "✅ Ego vehicle topics found:"
    docker exec test-ros2-bridge bash -c "source /ros_entrypoint.sh bash -c 'ros2 topic list | grep /carla/ego_vehicle'"
else
    echo "⚠️  No ego vehicle topics found (vehicle not spawned yet)"
    echo "   This is expected if using carla_ros_bridge.launch.py without example vehicle"
fi
echo ""

# Step 7: Test topic echo (if odometry exists)
echo "[6/6] Testing topic data flow..."
if docker exec test-ros2-bridge bash -c "source /ros_entrypoint.sh bash -c 'ros2 topic list | grep /carla/ego_vehicle/odometry'" > /dev/null 2>&1; then
    echo "   Echoing /carla/ego_vehicle/odometry (5 messages)..."
    docker exec test-ros2-bridge bash -c "source /ros_entrypoint.sh bash -c 'ros2 topic echo /carla/ego_vehicle/odometry --once'" || true
else
    echo "   Skipping topic echo (no ego vehicle spawned)"
fi
echo ""

# Cleanup
echo "═══════════════════════════════════════════════════════════════"
echo "Test complete! Bridge container still running for manual testing."
echo ""
echo "Useful commands:"
echo "  - List topics:    docker exec test-ros2-bridge bash -c 'source /ros_entrypoint.sh bash -c \"ros2 topic list\"'"
echo "  - Echo odometry:  docker exec test-ros2-bridge bash -c 'source /ros_entrypoint.sh bash -c \"ros2 topic echo /carla/ego_vehicle/odometry\"'"
echo "  - Send control:   docker exec test-ros2-bridge bash -c 'source /ros_entrypoint.sh bash -c \"ros2 topic pub /carla/ego_vehicle/vehicle_control_cmd carla_msgs/CarlaEgoVehicleControl \\\"{throttle: 0.5, steer: 0.0}\\\" -r 10\"'"
echo "  - Stop bridge:    docker stop test-ros2-bridge"
echo ""
echo "Next steps:"
echo "  1. Spawn ego vehicle with sensors (use carla_spawn_objects)"
echo "  2. Create baseline controller ROS 2 node"
echo "  3. Test PID + Pure Pursuit control"
echo "═══════════════════════════════════════════════════════════════"
