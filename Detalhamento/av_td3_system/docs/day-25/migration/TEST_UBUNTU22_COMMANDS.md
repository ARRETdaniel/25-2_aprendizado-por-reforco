# Ubuntu 22.04 Migration Testing Commands

**Date**: January 2025
**Image**: av-td3-system:ubuntu22.04-test
**Purpose**: Step-by-step testing of native ROS 2 Humble support

---

## Phase 3: Start CARLA Server

```bash
# Clean up any existing containers
docker stop carla-server 2>/dev/null || true
docker rm carla-server 2>/dev/null || true

# Start fresh CARLA server
docker run -d --name carla-server \
  --runtime=nvidia \
  --net=host \
  --env=NVIDIA_VISIBLE_DEVICES=all \
  --env=NVIDIA_DRIVER_CAPABILITIES=all \
  carlasim/carla:0.9.16 \
  bash CarlaUE4.sh -RenderOffScreen -nosound

# Wait for CARLA to initialize
echo "⏳ Waiting 45 seconds for CARLA server to initialize..."
sleep 45

# Verify CARLA is running
docker ps | grep carla-server
docker logs carla-server --tail 20
```

---

## Phase 4: Test Baseline WITHOUT ROS Bridge (Direct CARLA API)

This tests basic CARLA connectivity using the Python API directly.

```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

docker run --rm \
  --network host \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -v $(pwd):/workspace \
  -w /workspace \
  av-td3-system:ubuntu22.04-test \
  bash -c "source /opt/ros/humble/setup.bash && \
           python3 scripts/evaluate_baseline.py \
             --scenario 0 \
             --num-episodes 1 \
             --baseline-config config/baseline_config.yaml \
             --debug" \
  2>&1 | tee docs/day-25/migration/test_baseline_direct_api.log
```

**Expected**: Baseline evaluation completes successfully using direct CARLA API (no ROS).

---

## Phase 5-7: Test Baseline WITH ROS Bridge (Native rclpy)

### Step 1: Start ROS 2 Bridge Container

```bash
# Clean up existing ROS bridge
docker stop ros2-bridge 2>/dev/null || true
docker rm ros2-bridge 2>/dev/null || true

# Start ROS 2 Bridge
docker run -d --name ros2-bridge \
  --network host \
  --env ROS_DOMAIN_ID=0 \
  ros2-carla-bridge:humble-v4 \
  bash -c "
    source /opt/ros/humble/setup.bash && \
    source /opt/carla-ros-bridge/install/setup.bash && \
    ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py timeout:=10 & \
    sleep 5 && \
    ros2 launch carla_twist_to_control carla_twist_to_control.launch.py role_name:=ego_vehicle & \
    wait
  "

# Wait for ROS bridge to initialize
echo "⏳ Waiting 10 seconds for ROS bridge to initialize..."
sleep 10

# Verify ROS bridge is running
docker ps | grep ros2-bridge
docker logs ros2-bridge --tail 30
```

### Step 2: Run Baseline with Native rclpy

```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

docker run --rm \
  --network host \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -e ROS_DOMAIN_ID=0 \
  -v $(pwd):/workspace \
  -w /workspace \
  av-td3-system:ubuntu22.04-test \
  bash -c "source /opt/ros/humble/setup.bash && \
           python3 scripts/evaluate_baseline.py \
             --scenario 0 \
             --num-episodes 1 \
             --baseline-config config/baseline_config.yaml \
             --use-ros-bridge \
             --debug" \
  2>&1 | tee docs/day-25/migration/test_baseline_native_rclpy.log
```

**Expected**:
- Baseline uses native rclpy to publish control commands
- Latency <10ms (vs 3150ms docker-exec)
- Evaluation completes successfully

---

## Verification Commands

### Check All Containers Running

```bash
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
```

Expected output:
```
NAMES          IMAGE                           STATUS
ros2-bridge    ros2-carla-bridge:humble-v4     Up X seconds
carla-server   carlasim/carla:0.9.16           Up X seconds
```

### Check ROS 2 Topics (from Ubuntu 22.04 container)

```bash
docker run --rm \
  --network host \
  av-td3-system:ubuntu22.04-test \
  bash -c "source /opt/ros/humble/setup.bash && ros2 topic list"
```

Expected topics:
```
/carla/ego_vehicle/vehicle_control_cmd
/carla/ego_vehicle/vehicle_status
/clock
/tf
```

### Monitor ROS Topic Messages

```bash
docker run --rm \
  --network host \
  av-td3-system:ubuntu22.04-test \
  bash -c "source /opt/ros/humble/setup.bash && \
           ros2 topic echo /carla/ego_vehicle/vehicle_control_cmd --once"
```

---

## Cleanup Commands

```bash
# Stop all containers
docker stop ros2-bridge carla-server 2>/dev/null || true

# Remove containers
docker rm ros2-bridge carla-server 2>/dev/null || true

# Verify cleanup
docker ps -a | grep -E '(carla-server|ros2-bridge)'
```

---

## Full Test Sequence (All-in-One)

```bash
#!/bin/bash
set -e

cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

echo "======================================================================"
echo "Ubuntu 22.04 Migration Testing - Full Sequence"
echo "======================================================================"

# Cleanup
echo ""
echo "[1/6] Cleaning up existing containers..."
docker stop carla-server ros2-bridge 2>/dev/null || true
docker rm carla-server ros2-bridge 2>/dev/null || true

# Start CARLA
echo ""
echo "[2/6] Starting CARLA server..."
docker run -d --name carla-server \
  --runtime=nvidia --net=host \
  --env=NVIDIA_VISIBLE_DEVICES=all \
  --env=NVIDIA_DRIVER_CAPABILITIES=all \
  carlasim/carla:0.9.16 \
  bash CarlaUE4.sh -RenderOffScreen -nosound

echo "⏳ Waiting 45 seconds for CARLA..."
sleep 45

# Test Direct API (no ROS)
echo ""
echo "[3/6] Testing baseline WITHOUT ROS bridge (direct CARLA API)..."
docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -v $(pwd):/workspace -w /workspace \
  av-td3-system:ubuntu22.04-test \
  bash -c "source /opt/ros/humble/setup.bash && \
           python3 scripts/evaluate_baseline.py --scenario 0 --num-episodes 1 --debug" \
  2>&1 | tee docs/day-25/migration/test_direct_api.log

echo "✅ Direct API test complete!"

# Start ROS Bridge
echo ""
echo "[4/6] Starting ROS 2 Bridge..."
docker run -d --name ros2-bridge --network host --env ROS_DOMAIN_ID=0 \
  ros2-carla-bridge:humble-v4 \
  bash -c "source /opt/ros/humble/setup.bash && \
           source /opt/carla-ros-bridge/install/setup.bash && \
           ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py timeout:=10 & \
           sleep 5 && \
           ros2 launch carla_twist_to_control carla_twist_to_control.launch.py role_name:=ego_vehicle & \
           wait"

echo "⏳ Waiting 10 seconds for ROS bridge..."
sleep 10

# Test Native rclpy
echo ""
echo "[5/6] Testing baseline WITH ROS bridge (native rclpy)..."
docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -e ROS_DOMAIN_ID=0 \
  -v $(pwd):/workspace -w /workspace \
  av-td3-system:ubuntu22.04-test \
  bash -c "source /opt/ros/humble/setup.bash && \
           python3 scripts/evaluate_baseline.py --scenario 0 --num-episodes 1 --use-ros-bridge --debug" \
  2>&1 | tee docs/day-25/migration/test_native_rclpy.log

echo "✅ Native rclpy test complete!"

# Summary
echo ""
echo "[6/6] Test Summary"
echo "======================================================================"
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
echo ""
echo "📊 Check logs:"
echo "  - Direct API:    docs/day-25/migration/test_direct_api.log"
echo "  - Native rclpy:  docs/day-25/migration/test_native_rclpy.log"
echo ""
echo "✅ All tests complete!"
```

---

## Expected Results

### Success Criteria

- ✅ CARLA server starts and runs without errors
- ✅ Direct CARLA API test completes 1 episode successfully
- ✅ ROS 2 Bridge container starts without errors
- ✅ Native rclpy test completes 1 episode successfully
- ✅ ROS topics are visible and active
- ✅ Performance improvement documented (compare latency)

### Performance Metrics

| Metric | Old (docker-exec) | New (native rclpy) | Improvement |
|--------|-------------------|---------------------|-------------|
| ROS Latency | 3150ms | <10ms | ~630x faster |
| Container Setup | Complex | Simple | Cleaner |
| Python Version | 3.10 (conda) | 3.10 (system) | Native |

---

**Status**: Ready to execute Phase 3-7 testing
