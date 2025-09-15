#!/bin/bash

# CARLA ROS 2 Bridge Setup and Test Script
# This script configures the environment and tests the CARLA ROS 2 bridge

echo "🤖 CARLA ROS 2 Bridge Setup and Test"
echo "===================================="

# Set proper Python environment
export PATH="/usr/bin:/bin:$PATH"
export CARLA_ROOT=/home/danielterra/carla-source
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.16-cp38-cp38-linux_x86_64.whl:$CARLA_ROOT/PythonAPI/carla

# Navigate to ROS workspace
cd ~/carla-ros-bridge

# Source ROS environment
echo "📦 Setting up ROS 2 environment..."
source /opt/ros/foxy/setup.bash
source ./install/setup.bash

# Check ROS packages
echo "🔍 Checking available CARLA packages..."
CARLA_PACKAGES=$(ros2 pkg list | grep carla | wc -l)
echo "   Found $CARLA_PACKAGES CARLA ROS packages"

# Test CARLA connection
echo "🔗 Testing CARLA server connection..."
python3 -c "
import carla
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    print(f'   ✅ Connected to CARLA: {world.get_map().name}')
except Exception as e:
    print(f'   ❌ CARLA connection failed: {e}')
    exit(1)
"

echo ""
echo "🚀 Ready to launch ROS bridge!"
echo "   Use one of these commands:"
echo "   Basic bridge:"
echo "     ros2 launch carla_ros_bridge carla_ros_bridge.launch.py"
echo ""
echo "   Bridge with ego vehicle:"
echo "     ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py"
echo ""
echo "   Manual control:"
echo "     ros2 run carla_manual_control carla_manual_control"
echo ""
echo "   List topics:"
echo "     ros2 topic list"
