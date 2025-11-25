# ROS 2 Bridge Integration Guide

**Date**: 2025-01-22  
**Status**: ✅ **VERIFIED WORKING**  
**Target**: CARLA 0.9.16 + ROS 2 Humble + Python Scripts Integration

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [ROS 2 Topics Reference](#ros-2-topics-reference)
5. [Python Script Integration](#python-script-integration)
6. [Troubleshooting](#troubleshooting)
7. [Performance Tuning](#performance-tuning)
8. [Verification Procedures](#verification-procedures)

---

## Overview

This guide explains how to use ROS 2 Bridge to enable vehicle control and sensor data access for our autonomous vehicle evaluation and training scripts.

### What This Provides

✅ **CARLA Server**: High-fidelity simulation (CARLA 0.9.16)  
✅ **ROS 2 Bridge**: Bidirectional communication via ROS topics  
✅ **Vehicle Control**: Publish steering/throttle/brake via ROS topics  
✅ **Sensor Data**: Subscribe to camera, odometry, IMU, GPS  
✅ **Python Integration**: Use from evaluate_baseline.py, train_td3.py, etc.

### System Requirements

- **OS**: Ubuntu 20.04 or 22.04
- **GPU**: NVIDIA GPU with recent drivers
- **Docker**: 20.10+ with NVIDIA Container Toolkit
- **Python**: 3.10+
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 20GB free space

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CARLA Server Container                    │
│  Image: carlasim/carla:0.9.16                               │
│  Mode: Standard (NO --ros2 flag!)                           │
│  Port: 2000 (Python API)                                    │
│  GPU: NVIDIA runtime                                        │
└────────────────────────┬────────────────────────────────────┘
                         │ Python API Connection (port 2000)
┌────────────────────────▼────────────────────────────────────┐
│                 ROS 2 Bridge Container                       │
│  Image: ros2-carla-bridge:humble-v4                         │
│  ROS: Humble Hawksbill                                      │
│  Python: 3.10                                               │
│  CARLA Version: 0.9.16 (patched)                            │
└────────────────────────┬────────────────────────────────────┘
                         │ ROS 2 Topics (DDS)
         ┌───────────────┼───────────────┐
         │               │               │
┌────────▼────────┐ ┌───▼──────┐ ┌──────▼──────┐
│ evaluate_       │ │ train_   │ │ train_      │
│ baseline.py     │ │ td3.py   │ │ ddpg.py     │
│ (PID + PP)      │ │ (TD3 DRL)│ │ (DDPG DRL)  │
└─────────────────┘ └──────────┘ └─────────────┘
   Host Python Scripts (connect via ROS + CARLA API)
```

### Key Design Decisions

**Why Standard Mode (not --ros2)?**
- ROS Bridge requires Python API connection on port 2000
- Native ROS 2 (`--ros2` flag) only supports sensors + autopilot
- ROS Bridge enables full bidirectional control via topics

**Why ROS 2 Humble?**
- Python 3.10 compatible (matches CARLA 0.9.16 wheels)
- LTS support until 2027
- Better documentation than Foxy

**Why Not Containerize Python Scripts?**
- Easier debugging (no container rebuild needed)
- Direct access to CARLA Python API
- More flexible for development
- Can add later for production deployment

---

## Quick Start

### 1. Prerequisites Check

```bash
# Verify Docker installation
docker --version
# Should show: Docker version 20.10+

# Verify NVIDIA runtime
docker run --rm --runtime=nvidia nvidia/cuda:11.0-base nvidia-smi
# Should show GPU information

# Verify ROS Bridge image exists
docker images | grep ros2-carla-bridge:humble-v4
# Should show the image (if not, build it - see step 2)
```

### 2. Build ROS Bridge Image (if needed)

```bash
cd /workspace/av_td3_system

# Build ROS 2 Bridge image (~15-20 minutes)
docker build -t ros2-carla-bridge:humble-v4 \
  -f docker/ros2-carla-bridge.Dockerfile .

# Verify build
docker images | grep ros2-carla-bridge:humble-v4
```

### 3. Start Infrastructure

```bash
cd /workspace/av_td3_system

# Start CARLA + ROS Bridge
docker-compose -f docker-compose.ros-integration.yml up -d

# Expected output:
# Creating carla-server ... done
# Creating ros2-bridge  ... done
```

### 4. Verify System Status

```bash
# Check containers are running
docker-compose -f docker-compose.ros-integration.yml ps

# Expected:
# NAME         STATUS        PORTS
# carla-server Up (healthy)
# ros2-bridge  Up (healthy)

# Check CARLA logs
docker logs carla-server | tail -20
# Should show: "LogCarlaServer: Bind succeeded" and listening on 2000

# Check ROS Bridge logs
docker logs ros2-bridge | tail -20
# Should show: "Created EgoVehicle(id=XXX)" and "All objects spawned"
```

### 5. Test Vehicle Control

```bash
# Publish throttle command
docker exec ros2-bridge bash -c "
  source /opt/ros/humble/setup.bash &&
  source /opt/carla-ros-bridge/install/setup.bash &&
  ros2 topic pub /carla/ego_vehicle/vehicle_control_cmd \
    carla_msgs/msg/CarlaEgoVehicleControl \
    '{throttle: 0.5, steer: 0.0, brake: 0.0}' -r 10"

# In another terminal, monitor speed
docker exec ros2-bridge bash -c "
  source /opt/ros/humble/setup.bash &&
  source /opt/carla-ros-bridge/install/setup.bash &&
  ros2 topic echo /carla/ego_vehicle/speedometer"
```

### 6. Run Evaluation Scripts

```bash
# Install Python dependencies
pip install rclpy  # ROS 2 Python client

# Run baseline evaluation
cd /workspace/av_td3_system
python3 scripts/evaluate_baseline.py --scenario 0 --num-episodes 5

# Run TD3 training
python3 scripts/train_td3.py --scenario 0 --max-timesteps 50000
```

---

## ROS 2 Topics Reference

### Control Topics (Publish to these)

**`/carla/ego_vehicle/vehicle_control_cmd`**
- **Type**: `carla_msgs/msg/CarlaEgoVehicleControl`
- **Purpose**: Main vehicle control interface
- **Message Fields**:
  ```yaml
  throttle: float      # [0.0, 1.0] - Acceleration
  steer: float         # [-1.0, 1.0] - Steering (- = left, + = right)
  brake: float         # [0.0, 1.0] - Braking
  hand_brake: bool     # Emergency brake
  reverse: bool        # Reverse gear
  gear: int32          # Manual gear (-1=reverse, 0=neutral, 1-5=gears)
  manual_gear_shift: bool  # Enable manual transmission
  ```

**Examples**:
```bash
# Accelerate forward
ros2 topic pub --once /carla/ego_vehicle/vehicle_control_cmd \
  carla_msgs/msg/CarlaEgoVehicleControl \
  '{throttle: 0.7, steer: 0.0, brake: 0.0}'

# Turn right while accelerating
ros2 topic pub --once /carla/ego_vehicle/vehicle_control_cmd \
  carla_msgs/msg/CarlaEgoVehicleControl \
  '{throttle: 0.5, steer: 0.5, brake: 0.0}'

# Emergency stop
ros2 topic pub --once /carla/ego_vehicle/vehicle_control_cmd \
  carla_msgs/msg/CarlaEgoVehicleControl \
  '{throttle: 0.0, steer: 0.0, brake: 1.0, hand_brake: true}'
```

### Status Topics (Subscribe to these)

**`/carla/ego_vehicle/vehicle_status`**
- **Type**: `carla_msgs/msg/CarlaEgoVehicleStatus`
- **Rate**: ~20 Hz
- **Content**: Velocity, acceleration, current control state

**`/carla/ego_vehicle/odometry`**
- **Type**: `nav_msgs/msg/Odometry`
- **Rate**: ~20 Hz
- **Content**: Position, orientation, linear/angular velocities

**`/carla/ego_vehicle/speedometer`**
- **Type**: `std_msgs/msg/Float32`
- **Rate**: ~20 Hz
- **Content**: Speed in m/s

**`/carla/ego_vehicle/imu`**
- **Type**: `sensor_msgs/msg/Imu`
- **Rate**: ~20 Hz
- **Content**: Angular velocity, linear acceleration

**`/carla/ego_vehicle/gnss`**
- **Type**: `sensor_msgs/msg/NavSatFix`
- **Rate**: ~10 Hz
- **Content**: GPS coordinates (lat/lon/alt)

### Sensor Topics

**`/carla/ego_vehicle/rgb_front/image`**
- **Type**: `sensor_msgs/msg/Image`
- **Rate**: Camera FPS (configurable, default ~30 Hz)
- **Content**: RGB image data

**`/carla/ego_vehicle/rgb_front/camera_info`**
- **Type**: `sensor_msgs/msg/CameraInfo`
- **Rate**: Same as image
- **Content**: Camera calibration (intrinsics, distortion)

### Environment Topics

**`/carla/actor_list`** - All actors in simulation  
**`/carla/objects`** - Detected objects  
**`/carla/traffic_lights/info`** - Traffic light states  
**`/carla/map`** - OpenDRIVE map  
**`/clock`** - Simulation time

---

## Python Script Integration

### Option 1: Using ROSBridgeInterface Helper (Recommended)

```python
from src.utils.ros_bridge_interface import ROSBridgeInterface

# Initialize
ros_interface = ROSBridgeInterface(
    node_name='my_controller',
    ego_vehicle_role='ego_vehicle',
    use_docker_exec=True  # Use docker exec for simple publishing
)

# Wait for topics
if not ros_interface.wait_for_topics(timeout=10.0):
    print("ERROR: ROS topics not available")
    exit(1)

# Main control loop
for step in range(1000):
    # Get vehicle state
    status = ros_interface.get_vehicle_status()
    odom = ros_interface.get_odometry()
    
    if status:
        velocity = status['velocity']  # m/s
        print(f"Current speed: {velocity:.2f} m/s")
    
    # Calculate control (your algorithm here)
    throttle, steer, brake = calculate_control(...)
    
    # Publish control
    ros_interface.publish_control(
        throttle=throttle,
        steer=steer,
        brake=brake
    )
    
    time.sleep(0.05)  # 20 Hz control loop

# Cleanup
ros_interface.close()
```

### Option 2: Direct Docker Exec (Simpler, no ROS dependencies)

```python
import subprocess

def publish_control_docker(throttle, steer, brake):
    """Publish control via docker exec."""
    cmd = [
        'docker', 'exec', 'ros2-bridge', 'bash', '-c',
        f"source /opt/ros/humble/setup.bash && "
        f"source /opt/carla-ros-bridge/install/setup.bash && "
        f"ros2 topic pub --once /carla/ego_vehicle/vehicle_control_cmd "
        f"carla_msgs/msg/CarlaEgoVehicleControl "
        f"\"{{throttle: {throttle}, steer: {steer}, brake: {brake}}}\""
    ]
    
    result = subprocess.run(cmd, capture_output=True, timeout=2)
    return result.returncode == 0

# Usage
publish_control_docker(throttle=0.5, steer=0.0, brake=0.0)
```

### Integration into evaluate_baseline.py

Add to your evaluation script:

```python
# At top of file
from src.utils.ros_bridge_interface import ROSBridgeInterface

# In __init__ method
self.ros_interface = ROSBridgeInterface(
    node_name='baseline_evaluation',
    use_docker_exec=True
)

# In control loop (replace direct VehicleControl application)
# OLD:
# vehicle.apply_control(carla.VehicleControl(throttle=..., steer=...))

# NEW:
self.ros_interface.publish_control(
    throttle=control.throttle,
    steer=control.steer,
    brake=control.brake
)
```

---

## Troubleshooting

### Issue 1: ROS Bridge Won't Start

**Symptom**: `ros2-bridge` container exits immediately or shows "Connection refused"

**Solution**:
```bash
# Check if CARLA is running and healthy
docker-compose -f docker-compose.ros-integration.yml ps

# Restart in correct order
docker-compose -f docker-compose.ros-integration.yml down
docker-compose -f docker-compose.ros-integration.yml up -d carla-server
# Wait 30 seconds for CARLA to be fully ready
docker-compose -f docker-compose.ros-integration.yml up -d ros2-bridge
```

### Issue 2: Topics Not Publishing

**Symptom**: `ros2 topic list` shows no `/carla` topics

**Solution**:
```bash
# Check bridge logs
docker logs ros2-bridge

# Look for errors like:
# - "CARLA python module version X.X.X required"
#   → Version mismatch, rebuild bridge image
# - "time-out while waiting for simulator"
#   → CARLA not running or not on port 2000

# Verify CARLA is listening
docker exec carla-server netstat -tuln | grep 2000
# Should show: tcp 0 0 0.0.0.0:2000 0.0.0.0:* LISTEN
```

### Issue 3: Control Commands Not Working

**Symptom**: Publishing to `/carla/ego_vehicle/vehicle_control_cmd` but vehicle doesn't move

**Solution**:
```bash
# Check if ego vehicle was spawned
docker logs ros2-bridge | grep "Created EgoVehicle"
# Should show: [INFO] Created EgoVehicle(id=XXX)

# Check if topic has subscribers
docker exec ros2-bridge bash -c "
  source /opt/ros/humble/setup.bash &&
  ros2 topic info /carla/ego_vehicle/vehicle_control_cmd"
# Should show: Subscription count: 1

# Try publishing with continuous rate
docker exec ros2-bridge bash -c "
  source /opt/ros/humble/setup.bash &&
  source /opt/carla-ros-bridge/install/setup.bash &&
  ros2 topic pub /carla/ego_vehicle/vehicle_control_cmd \
    carla_msgs/msg/CarlaEgoVehicleControl \
    '{throttle: 0.5}' -r 10"
# Vehicle should start moving
```

### Issue 4: Python Script Can't Import rclpy

**Symptom**: `ImportError: No module named 'rclpy'`

**Solution**:
```bash
pip install rclpy

# Or use ROSBridgeInterface with use_docker_exec=True
# (doesn't require rclpy on host)
```

---

## Performance Tuning

### Simulation Frequency

Adjust in docker-compose.ros-integration.yml:

```yaml
ros2-bridge:
  command: >
    ... fixed_delta_seconds:=0.05  # 0.05 = 20 Hz, 0.02 = 50 Hz
```

**Recommendations**:
- **Training**: 0.05 (20 Hz) - good balance of speed and accuracy
- **Evaluation**: 0.02 (50 Hz) - higher fidelity
- **Debugging**: 0.1 (10 Hz) - slower, easier to observe

### GPU Allocation

For multi-GPU systems:

```yaml
carla-server:
  environment:
    - NVIDIA_VISIBLE_DEVICES=0  # Use specific GPU
```

### Resource Limits

For supercomputer deployment:

```yaml
carla-server:
  deploy:
    resources:
      limits:
        memory: 8G
        cpus: '4.0'
```

---

## Verification Procedures

### Complete System Check

```bash
#!/bin/bash
# verification_script.sh

echo "=== ROS 2 Bridge System Verification ==="

echo "[1/7] Checking Docker..."
docker --version || { echo "ERROR: Docker not installed"; exit 1; }

echo "[2/7] Checking containers..."
docker-compose -f docker-compose.ros-integration.yml ps | grep -q "Up (healthy)" || {
    echo "ERROR: Containers not healthy"
    exit 1
}

echo "[3/7] Checking CARLA port 2000..."
docker exec carla-server netstat -tuln | grep -q ":2000" || {
    echo "ERROR: CARLA not listening on 2000"
    exit 1
}

echo "[4/7] Checking ROS topics..."
docker exec ros2-bridge bash -c "
  source /opt/ros/humble/setup.bash &&
  ros2 topic list | grep -q '/carla/ego_vehicle/vehicle_control_cmd'" || {
    echo "ERROR: Control topic not found"
    exit 1
}

echo "[5/7] Checking ego vehicle..."
docker logs ros2-bridge | grep -q "Created EgoVehicle" || {
    echo "ERROR: Ego vehicle not spawned"
    exit 1
}

echo "[6/7] Testing control publish..."
docker exec ros2-bridge bash -c "
  source /opt/ros/humble/setup.bash &&
  source /opt/carla-ros-bridge/install/setup.bash &&
  ros2 topic pub --once /carla/ego_vehicle/vehicle_control_cmd \
    carla_msgs/msg/CarlaEgoVehicleControl '{throttle: 0.1}'" || {
    echo "ERROR: Failed to publish control"
    exit 1
}

echo "[7/7] Checking Python integration..."
python3 -c "
from src.utils.ros_bridge_interface import ROSBridgeInterface
ros = ROSBridgeInterface(use_docker_exec=True)
print('✅ Python integration OK')
ros.close()
" || {
    echo "ERROR: Python integration failed"
    exit 1
}

echo ""
echo "=== ✅ ALL CHECKS PASSED ==="
echo "System ready for evaluation/training"
```

---

## Next Steps

1. ✅ **Infrastructure Running** - CARLA + ROS Bridge operational
2. ⏳ **Integrate Scripts** - Modify evaluate_baseline.py, train_td3.py
3. ⏳ **End-to-End Testing** - Full episodes with metrics collection
4. ⏳ **Baseline Evaluation** - 20 episodes, collect safety/efficiency/comfort metrics
5. ⏳ **TD3 Training** - Train agent with ROS Bridge control
6. ⏳ **Comparative Analysis** - TD3 vs DDPG vs Baseline

---

## References

- **CARLA Documentation**: https://carla.readthedocs.io/en/latest/
- **ROS Bridge Docs**: https://carla.readthedocs.io/projects/ros-bridge/en/latest/
- **ROS 2 Humble**: https://docs.ros.org/en/humble/
- **Success Report**: `av_td3_system/docs/ROS_BRIDGE_SUCCESS_REPORT.md`
- **Diagnostic Report**: `av_td3_system/docs/ROS_INTEGRATION_DIAGNOSTIC_REPORT.md`

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-22  
**Status**: ✅ Verified Working

