# Docker Setup for Phase 2: Baseline Controller System

This directory contains Docker configurations for running the CARLA 0.9.16 + ROS 2 Foxy + Baseline Controller system in containers.

## Architecture

The system uses a **multi-container architecture** with 3 main services:

```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│ CARLA Server    │◄───│ ROS 2 Bridge     │◄───│ Baseline           │
│ (carlasim/      │    │ (ros2-carla-     │    │ Controller         │
│  carla:0.9.16)  │    │  bridge:foxy)    │    │ (baseline-         │
│ Port: 2000      │    │ Translates CARLA │    │  controller:foxy)  │
│ GPU: NVIDIA     │    │ to ROS 2 topics  │    │ PID + Pure Pursuit │
└─────────────────┘    └──────────────────┘    └────────────────────┘
       │                        │                        │
       └────────────────────────┴────────────────────────┘
                    Host Network (--net=host)
```

## Files

### Core Docker Files

* **`ros2-carla-bridge.Dockerfile`** - Multi-stage build for ROS 2 + CARLA bridge
  - Stage 1: Extract CARLA Python API from `carlasim/carla:0.9.16`
  - Stage 2: Build ROS 2 workspace with bridge
  - Output: `ros2-carla-bridge:foxy` image

* **`baseline-controller.Dockerfile`** - Baseline controller (PID + Pure Pursuit)
  - Extends `ros2-carla-bridge:foxy`
  - Adds controller code from `src/baselines/` and `src/ros_nodes/`
  - Output: `baseline-controller:foxy` image

### Orchestration

* **`docker-compose.baseline.yml`** - Multi-container orchestration
  - Service 1: `carla-server` (CARLA simulation)
  - Service 2: `ros2-bridge` (CARLA ↔ ROS 2 translation)
  - Service 3: `baseline-controller` (waypoint following)

### Helper Scripts

* **`build_ros2_bridge.sh`** - Build ROS 2 bridge image
* **`build_baseline.sh`** - Build baseline controller image (to be created)
* **`test_bridge.sh`** - Test bridge connection (to be created)

## Prerequisites

### 1. Install Docker

```bash
# Ubuntu 20.04
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo usermod -aG docker $USER
newgrp docker  # Or logout and login again
```

### 2. Install NVIDIA Container Toolkit (for GPU support)

```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --runtime=nvidia nvidia/cuda:11.0-base nvidia-smi
```

### 3. Verify CARLA Image

```bash
# Pull CARLA 0.9.16
docker pull carlasim/carla:0.9.16

# Test CARLA server (should listen on port 2000)
docker run --rm --runtime=nvidia --net=host \
    carlasim/carla:0.9.16 bash CarlaUE4.sh -RenderOffScreen -nosound &

# Check port 2000
netstat -tuln | grep 2000
# Should show: tcp 0 0 0.0.0.0:2000 0.0.0.0:* LISTEN

# Stop CARLA
docker stop $(docker ps -q --filter ancestor=carlasim/carla:0.9.16)
```

## Quick Start

### Step 1: Build Images

```bash
cd /workspace/av_td3_system

# Build ROS 2 + CARLA bridge image (~15 minutes)
./docker/build_ros2_bridge.sh

# Build baseline controller image (~5 minutes, after bridge image is ready)
# ./docker/build_baseline.sh  # To be implemented
```

### Step 2: Prepare Configuration Files

```bash
# Ensure waypoints.txt exists
ls -lh config/waypoints.txt

# Create baseline parameters (if not exists)
# File: config/baseline_params.yaml
```

### Step 3: Launch System

```bash
# Launch all 3 containers
docker-compose -f docker-compose.baseline.yml up

# Or in detached mode (background)
docker-compose -f docker-compose.baseline.yml up -d

# View logs
docker-compose -f docker-compose.baseline.yml logs -f
```

### Step 4: Verify System

```bash
# Check containers are running
docker-compose -f docker-compose.baseline.yml ps

# Expected output:
# NAME                STATUS              PORTS
# carla-server        Up (healthy)        
# ros2-bridge         Up (healthy)
# baseline-controller Up

# Check ROS 2 topics
docker-compose -f docker-compose.baseline.yml exec ros2-bridge ros2 topic list

# Expected topics:
# /carla/ego_vehicle/odometry
# /carla/ego_vehicle/vehicle_control_cmd
# /carla/ego_vehicle/vehicle_status
# /clock

# Check topic frequency (should be ~20 Hz)
docker-compose -f docker-compose.baseline.yml exec ros2-bridge \
    ros2 topic hz /carla/ego_vehicle/odometry
```

## Development Workflow

### Interactive Container Access

```bash
# Access ROS 2 bridge container
docker-compose -f docker-compose.baseline.yml exec ros2-bridge bash

# Inside container, test commands:
source /opt/ros/foxy/setup.bash
ros2 topic list
ros2 topic echo /carla/ego_vehicle/odometry --once
```

### Rebuild After Code Changes

```bash
# Rebuild baseline controller image
docker build -t baseline-controller:foxy -f docker/baseline-controller.Dockerfile .

# Restart only baseline controller service
docker-compose -f docker-compose.baseline.yml up -d --force-recreate baseline-controller
```

### Manual Vehicle Control (Testing)

```bash
# Publish control command manually
docker-compose -f docker-compose.baseline.yml exec ros2-bridge ros2 topic pub --once \
    /carla/ego_vehicle/vehicle_control_cmd carla_msgs/msg/CarlaEgoVehicleControl \
    "{throttle: 0.5, steer: 0.0, brake: 0.0, hand_brake: false, reverse: false, manual_gear_shift: false, gear: 1}"

# Vehicle should accelerate forward
```

## Troubleshooting

### Issue 1: CARLA server not starting

**Symptom**: `carla-server` container exits immediately

**Solution**:
```bash
# Check NVIDIA runtime
docker run --rm --runtime=nvidia nvidia/cuda:11.0-base nvidia-smi

# Check GPU availability
nvidia-smi

# Check CARLA logs
docker-compose -f docker-compose.baseline.yml logs carla-server
```

### Issue 2: Bridge cannot connect to CARLA

**Symptom**: `ros2-bridge` logs show "Connection refused to localhost:2000"

**Solution**:
```bash
# Verify CARLA is listening
docker-compose -f docker-compose.baseline.yml exec carla-server netstat -tuln | grep 2000

# Check health status
docker-compose -f docker-compose.baseline.yml ps
# carla-server should show "Up (healthy)"

# Restart bridge after CARLA is healthy
docker-compose -f docker-compose.baseline.yml restart ros2-bridge
```

### Issue 3: No ROS 2 topics appearing

**Symptom**: `ros2 topic list` shows empty or no `/carla` topics

**Solution**:
```bash
# Check bridge logs for errors
docker-compose -f docker-compose.baseline.yml logs -f ros2-bridge

# Verify ROS_DOMAIN_ID is consistent
docker-compose -f docker-compose.baseline.yml exec ros2-bridge env | grep ROS_DOMAIN_ID
# Should show: ROS_DOMAIN_ID=0

# Test bridge manually
docker-compose -f docker-compose.baseline.yml exec ros2-bridge bash
source /opt/ros/foxy/setup.bash
source /opt/carla-ros-bridge/install/setup.bash
ros2 launch carla_ros_bridge carla_ros_bridge.launch.py
```

### Issue 4: Permission denied on volume mounts

**Symptom**: Container cannot write to `./data/logs` or `./data/results`

**Solution**:
```bash
# Create directories with correct permissions
mkdir -p data/logs data/results
chmod -R 777 data/

# Or run as current user (add to docker-compose.baseline.yml):
# user: "${UID}:${GID}"
```

## Performance Tuning

### For Supercomputer Deployment

Edit `docker-compose.baseline.yml`:

```yaml
services:
  carla-server:
    environment:
      - NVIDIA_VISIBLE_DEVICES=0  # Specific GPU
    deploy:
      resources:
        limits:
          memory: 8G  # Limit RAM usage
          cpus: '4.0'  # Limit CPU cores
```

### For Multi-GPU Systems

```bash
# Run multiple experiments in parallel
NVIDIA_VISIBLE_DEVICES=0 docker-compose -f docker-compose.baseline.yml up &
NVIDIA_VISIBLE_DEVICES=1 docker-compose -f docker-compose.baseline.yml up &
```

## Next Steps

After Phase 2 is complete:

1. **Phase 3**: Integrate DRL agent (TD3) with ROS 2 interface
2. **Phase 4**: Comparative evaluation (TD3 vs DDPG vs Baseline)
3. **Phase 5**: Deployment to supercomputer for large-scale training

## References

* CARLA Documentation: https://carla.readthedocs.io/en/latest/
* CARLA ROS Bridge: https://github.com/carla-simulator/ros-bridge
* ROS 2 Foxy Documentation: https://docs.ros.org/en/foxy/
* Docker Best Practices: https://docs.docker.com/develop/dev-best-practices/

## Support

For issues specific to this setup:
1. Check logs: `docker-compose -f docker-compose.baseline.yml logs`
2. Check documentation: `docs/day-22/baseline/PHASE_2_DOCKER_ARCHITECTURE.md`
3. Check existing issues in GitHub repository

---

**Last Updated**: November 22, 2025  
**Phase**: Phase 2 - Baseline Controller System  
**Status**: ✅ Docker configuration complete, ready for testing
