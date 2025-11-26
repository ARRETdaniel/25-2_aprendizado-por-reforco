#!/bin/bash
# ROS 2 Bridge Testing - Complete Test Sequence (FIXED)
# Date: November 25, 2025
# Status: Ready to run after Docker rebuild

set -e

PROJECT_ROOT="/media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system"

echo "======================================================================"
echo "ROS 2 BRIDGE TESTING - WITH CARLA_TWIST_TO_CONTROL NODE"
echo "======================================================================"
echo ""
echo "CRITICAL CHANGES:"
echo "  ✅ Fixed carla_env.py: close() → destroy()"
echo "  ✅ Fixed docker-entrypoint.sh: Foxy → Humble"
echo "  ✅ Added diagnostic logging to ROSBridgeInterface"
echo "  ✅ MUST launch carla_twist_to_control node in ROS Bridge!"
echo ""
echo "======================================================================"
echo ""

# Step 1: Cleanup
echo "[1/7] Cleaning up existing containers..."
docker stop carla-server ros2-bridge 2>/dev/null || true
docker rm carla-server ros2-bridge 2>/dev/null || true
echo "✅ Cleanup complete"
echo ""

# Step 2: Start CARLA Server
echo "[2/7] Starting CARLA server..."
docker run -d --name carla-server \
  --runtime=nvidia \
  --net=host \
  --env=NVIDIA_VISIBLE_DEVICES=all \
  --env=NVIDIA_DRIVER_CAPABILITIES=all \
  carlasim/carla:0.9.16 \
  bash CarlaUE4.sh -RenderOffScreen -nosound

echo "⏳ Waiting 45 seconds for CARLA to initialize..."
sleep 1

# Verify CARLA is running
if docker ps | grep -q carla-server; then
    echo "✅ CARLA server running"
else
    echo "❌ CARLA server failed to start"
    exit 1
fi
echo ""

# Step 3: Start ROS Bridge WITH carla_twist_to_control
echo "[3/7] Starting ROS 2 Bridge with BOTH nodes..."
echo "   - carla_ros_bridge (spawns vehicle, manages CARLA)"
echo "   - carla_twist_to_control (converts Twist → VehicleControl) ← CRITICAL!"
echo ""

docker run -d --name ros2-bridge \
  --network host \
  --env ROS_DOMAIN_ID=0 \
  ros2-carla-bridge:humble-v4 \
  bash -c "
    source /opt/ros/humble/setup.bash && \
    source /opt/carla-ros-bridge/install/setup.bash && \
    echo '🚀 Launching carla_ros_bridge...' && \
    ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py timeout:=10 & \
    sleep 5 && \
    echo '🚀 Launching carla_twist_to_control...' && \
    ros2 launch carla_twist_to_control carla_twist_to_control.launch.py role_name:=ego_vehicle & \
    wait
  "

echo "⏳ Waiting 15 seconds for ROS bridge to initialize..."
sleep 15

# Verify ROS bridge is running
if docker ps | grep -q ros2-bridge; then
    echo "✅ ROS bridge container running"
else
    echo "❌ ROS bridge failed to start"
    docker logs ros2-bridge
    exit 1
fi
echo ""

# Step 4: Verify ROS Bridge Logs
echo "[4/7] Verifying BOTH nodes are running..."
echo ""
echo "Checking ROS bridge logs for node launches:"
docker logs ros2-bridge 2>&1 | grep -E "(carla_ros_bridge|carla_twist_to_control)" | tail -20
echo ""

if docker logs ros2-bridge 2>&1 | grep -q "carla_twist_to_control"; then
    echo "✅ carla_twist_to_control node detected in logs"
else
    echo "⚠️  WARNING: carla_twist_to_control node NOT detected!"
    echo "   Vehicle may not move without this converter node!"
fi
echo ""

# Step 5: Verify ROS Topics
echo "[5/7] Verifying ROS 2 topics..."
echo ""

docker run --rm \
  --network host \
  -e ROS_DOMAIN_ID=0 \
  av-td3-system:ubuntu22.04-test \
  bash -c "source /opt/ros/humble/setup.bash && \
           echo 'Available topics:' && \
           ros2 topic list | grep ego_vehicle"

echo ""
echo "CRITICAL CHECK: /carla/ego_vehicle/vehicle_control_cmd topic MUST exist!"
echo "                 This proves carla_twist_to_control is running."
echo ""

if docker run --rm --network host -e ROS_DOMAIN_ID=0 av-td3-system:ubuntu22.04-test \
   bash -c "source /opt/ros/humble/setup.bash && ros2 topic list" | grep -q "vehicle_control_cmd"; then
    echo "✅ /carla/ego_vehicle/vehicle_control_cmd exists - converter is running!"
else
    echo "❌ /carla/ego_vehicle/vehicle_control_cmd NOT found!"
    echo "   The carla_twist_to_control node is NOT running properly."
    echo "   Vehicle WILL NOT MOVE!"
    exit 1
fi
echo ""

# Step 6: Run Evaluation with ROS Bridge
echo "[6/7] Running baseline evaluation with ROS Bridge..."
echo ""
echo "Expected log output:"
echo "  [ROS BRIDGE] Published Twist #1: linear.x=X.XX m/s, angular.z=X.XXX rad"
echo "  DEBUG Step 0: Speed: XX.XX km/h ← VEHICLE SHOULD BE MOVING!"
echo ""
echo "Starting evaluation..."
echo ""

cd "$PROJECT_ROOT"

docker run --rm \
  --network host \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -e ROS_DOMAIN_ID=0 \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/workspace \
  -w /workspace \
  --privileged \
  av-td3-system:ubuntu22.04-test \
  bash -c "source /opt/ros/humble/setup.bash && \
           python3 scripts/evaluate_baseline.py \
             --scenario 0 \
             --num-episodes 1 \
             --use-ros-bridge \
             --debug" \
  2>&1 | tee docs/day-25/migration/test_ros2_bridge_FIXED_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "[7/7] Test complete!"
echo ""

# Check if test succeeded
if grep -q "User requested quit" docs/day-25/migration/test_ros2_bridge_FIXED_*.log 2>/dev/null; then
    echo "⚠️  Test was manually quit - check if vehicle was moving"
else
    echo "✅ Test completed (check log for results)"
fi

echo ""
echo "======================================================================"
echo "VERIFICATION CHECKLIST:"
echo "======================================================================"
echo ""
echo "Did you see these in the log?"
echo "  [ ] [ROS BRIDGE] Initialized native rclpy Twist publisher"
echo "  [ ] [ROS BRIDGE] Published Twist #1: ..."
echo "  [ ] DEBUG Step 0: Speed: XX.XX km/h (>0 means vehicle moved!)"
echo "  [ ] No 'Error closing ROS interface' warning"
echo ""
echo "======================================================================"
echo "Next Steps:"
echo "======================================================================"
echo ""
echo "If vehicle moved:"
echo "  ✅ ROS Bridge integration working!"
echo "  📊 Compare latency vs direct CARLA API (Phase 4)"
echo "  📝 Update paper methodology"
echo ""
echo "If vehicle did NOT move:"
echo "  ❌ Check ROS bridge logs: docker logs ros2-bridge"
echo "  ❌ Verify topics: ros2 topic list | grep ego_vehicle"
echo "  ❌ Monitor Twist: ros2 topic echo /carla/ego_vehicle/twist"
echo "  ❌ Monitor Control: ros2 topic echo /carla/ego_vehicle/vehicle_control_cmd"
echo ""
echo "======================================================================"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up containers..."
    docker stop carla-server ros2-bridge 2>/dev/null || true
    docker rm carla-server ros2-bridge 2>/dev/null || true
    echo "✅ Cleanup complete"
}

trap cleanup EXIT
