# Phase 2: Docker-Based ROS 2 + CARLA Architecture

**Date**: November 22, 2025  
**Phase**: Phase 2 - Docker Architecture Design  
**Status**: 🚧 **IN PROGRESS**  
**Requirement**: System must run in Docker for supercomputer deployment  

---

## Executive Summary

Based on the requirement that **all systems must run in Docker containers** for supercomputer training, I've designed a multi-container architecture that:

1. **Separates concerns**: CARLA server, ROS 2 bridge, baseline controller, and future DRL agent run in separate containers
2. **Enables portability**: Entire system can be deployed to supercomputer with `docker-compose up`
3. **Maintains modularity**: Controllers can be swapped by launching different containers
4. **Uses official patterns**: Based on official CARLA ROS bridge Docker implementation

---

## Critical Findings from Documentation Research

### 1. Official CARLA ROS Bridge Docker Support ✅

**Source**: `https://github.com/carla-simulator/ros-bridge/tree/master/docker`

**Official Dockerfile Structure** (Analyzed from repo):
```dockerfile
ARG CARLA_VERSION
ARG ROS_DISTRO

# Stage 1: Extract CARLA Python API
FROM carlasim/carla:$CARLA_VERSION as carla

# Stage 2: Build ROS bridge
FROM ros:$ROS_DISTRO-ros-base

# Copy CARLA Python API from stage 1
COPY --from=carla /home/carla/PythonAPI /opt/carla/PythonAPI

# Install dependencies
COPY requirements.txt /opt/carla-ros-bridge
RUN bash install_dependencies.sh

# Build bridge with colcon (ROS 2) or catkin (ROS 1)
COPY . /opt/carla-ros-bridge/src/
RUN if [ "$ROS_VERSION" == "2" ]; then colcon build; else catkin_make install; fi
```

**Key Insights**:
- ✅ **Multi-stage build** extracts CARLA Python API from official image
- ✅ **ROS base image** provides ROS 2 Foxy environment
- ✅ **Automated dependency** installation via `install_dependencies.sh`
- ✅ **Conditional build** supports both ROS 1 and ROS 2

### 2. ROS 2 Docker Official Patterns ✅

**Source**: `https://docs.ros.org/en/foxy/How-To-Guides/Run-2-nodes-in-single-or-separate-docker-containers.html`

**Multi-Container Communication**:
```yaml
version: '2'
services:
  talker:
    image: osrf/ros:foxy-desktop
    command: ros2 run demo_nodes_cpp talker
  listener:
    image: osrf/ros:foxy-desktop
    command: ros2 run demo_nodes_cpp listener
    depends_on:
      - talker
```

**Key Insights**:
- ✅ **Default networking**: ROS 2 DDS auto-discovers nodes across containers
- ✅ **No explicit network config** needed (bridge network default)
- ✅ **Service dependencies**: `depends_on` ensures startup order

### 3. CARLA Docker Requirements ✅

**Source**: `https://carla.readthedocs.io/en/latest/build_docker/`

**CARLA Server Launch**:
```bash
docker run \
    --runtime=nvidia \
    --net=host \
    --env=NVIDIA_VISIBLE_DEVICES=all \
    --env=NVIDIA_DRIVER_CAPABILITIES=all \
    carlasim/carla:0.9.16 bash CarlaUE4.sh -RenderOffScreen -nosound
```

**Key Insights**:
- ✅ **NVIDIA runtime** required for GPU access
- ✅ **Host network mode** recommended for low latency
- ⚠️ **RenderOffScreen** mandatory for headless servers (supercomputer)

---

## Proposed Docker Architecture

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Docker Host (Supercomputer)                    │
│                                                                     │
│  ┌────────────────┐  ┌─────────────────┐  ┌───────────────────┐  │
│  │ CARLA Server   │  │ ROS 2 Bridge     │  │ Baseline/DRL      │  │
│  │ Container      │  │ Container        │  │ Controller        │  │
│  │                │  │                  │  │ Container         │  │
│  │ carlasim/      │  │ Custom Image:    │  │ Custom Image:     │  │
│  │ carla:0.9.16   │  │ ros2-carla       │  │ baseline-ctrl     │  │
│  │                │  │ -bridge:foxy     │  │ or drl-agent      │  │
│  │ Port: 2000     │◄─┤ Connects to      │◄─┤ Publishes to      │  │
│  │ GPU: Yes       │  │ localhost:2000   │  │ ROS 2 topics      │  │
│  │ Network: host  │  │ Network: host    │  │ Network: host     │  │
│  └────────────────┘  └─────────────────┘  └───────────────────┘  │
│         │                      │                      │            │
│         └──────────────────────┴──────────────────────┘            │
│                        Shared Volumes                              │
│          /workspace/config/waypoints.txt (read-only)               │
│          /workspace/data/logs (write)                              │
│          /workspace/data/checkpoints (write)                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Container Responsibilities

#### Container 1: CARLA Server (`carla-server`)

**Image**: `carlasim/carla:0.9.16` (official)  
**Purpose**: Simulation engine  
**Configuration**:
- Runtime: nvidia
- Network: host
- GPU: All devices
- Command: `bash CarlaUE4.sh -RenderOffScreen -nosound`
- Ports: 2000 (CARLA server)

**Why host network?**
- Lowest latency for client connections
- Required for ROS 2 DDS multicast discovery
- Standard practice for CARLA Docker deployments

#### Container 2: ROS 2 Bridge (`ros2-bridge`)

**Image**: `ros2-carla-bridge:foxy` (custom build)  
**Purpose**: CARLA ↔ ROS 2 translation  
**Base**: `ros:foxy-ros-base`  
**Build Process**:
1. Copy CARLA Python API from `carlasim/carla:0.9.16`
2. Clone `carla-ros-bridge` (ros2 branch)
3. Install dependencies (rosdep)
4. Build with colcon
5. Set PYTHONPATH for CARLA .egg file

**Configuration**:
- Network: host (for DDS + CARLA connection)
- Depends on: carla-server
- Environment:
  - `ROS_DOMAIN_ID=0`
  - `PYTHONPATH=/opt/carla/PythonAPI/carla/dist/carla-*.egg`
  - `use_sim_time=True`
- Command: `ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py`

**Publishes** (CARLA → ROS 2):
- `/carla/ego_vehicle/odometry` (nav_msgs/Odometry, 20 Hz)
- `/carla/ego_vehicle/vehicle_status` (carla_msgs/CarlaEgoVehicleStatus, 20 Hz)
- `/carla/ego_vehicle/imu` (sensor_msgs/Imu)
- `/carla/ego_vehicle/collision` (carla_msgs/CarlaCollisionEvent)
- `/carla/ego_vehicle/lane_invasion` (carla_msgs/CarlaLaneInvasionEvent)
- `/clock` (rosgraph_msgs/Clock) - synchronous mode

**Subscribes** (ROS 2 → CARLA):
- `/carla/ego_vehicle/vehicle_control_cmd` (carla_msgs/CarlaEgoVehicleControl)

#### Container 3: Baseline Controller (`baseline-controller`)

**Image**: `baseline-controller:foxy` (custom build)  
**Purpose**: PID + Pure Pursuit waypoint follower  
**Base**: `ros2-carla-bridge:foxy` (inherit ROS 2 + CARLA env)  
**Code**:
- `src/baselines/pid_pure_pursuit.py` (extracted from controller2d.py)
- `src/ros_nodes/baseline_controller_node.py` (ROS 2 node)
- `src/utils/waypoint_loader.py` (reads waypoints.txt)

**Configuration**:
- Network: host (for ROS 2 DDS discovery)
- Depends on: ros2-bridge
- Volumes:
  - `./config/waypoints.txt:/workspace/config/waypoints.txt:ro`
  - `./data/logs:/workspace/data/logs:rw`
- Environment:
  - `ROS_DOMAIN_ID=0`
  - `use_sim_time=True`
- Command: `ros2 run av_td3_baseline baseline_controller_node --ros-args --params-file /workspace/config/baseline_params.yaml`

**Subscribes**:
- `/carla/ego_vehicle/odometry` (for x, y, yaw, speed)
- `/carla/ego_vehicle/collision` (for safety monitoring)

**Publishes**:
- `/carla/ego_vehicle/vehicle_control_cmd` (throttle, steer, brake commands)

#### Container 4: DRL Agent (Future) (`drl-agent`)

**Image**: `drl-agent:foxy` (custom build)  
**Purpose**: TD3 agent for training/evaluation  
**Base**: `ros2-carla-bridge:foxy` + PyTorch  
**Code**:
- `src/agents/td3_agent.py`
- `src/environment/carla_env_ros2.py` (ROS 2 version of carla_env.py)
- `scripts/train_td3_ros2.py`

**Configuration**: Same as baseline, different command

---

## Implementation Plan

### Step 1: Build ROS 2 + Bridge Base Image

**Dockerfile**: `docker/ros2-carla-bridge.Dockerfile`

```dockerfile
ARG CARLA_VERSION=0.9.16
ARG ROS_DISTRO=foxy

# Stage 1: Extract CARLA Python API
FROM carlasim/carla:${CARLA_VERSION} as carla

# Stage 2: Build ROS 2 + CARLA bridge environment
FROM ros:${ROS_DISTRO}-ros-base

ARG CARLA_VERSION
ARG ROS_DISTRO

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# Copy CARLA Python API
COPY --from=carla /home/carla/PythonAPI /opt/carla/PythonAPI

# Set up workspace
RUN mkdir -p /opt/carla-ros-bridge/src
WORKDIR /opt/carla-ros-bridge

# Clone CARLA ROS bridge (ROS 2 branch)
RUN git clone --recurse-submodules --branch ros2 \
    https://github.com/carla-simulator/ros-bridge.git src/ros-bridge

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    pygame \
    networkx \
    transforms3d \
    simple-watchdog-timer

# Initialize rosdep and install ROS dependencies
RUN rosdep init || true && rosdep update
RUN /bin/bash -c 'source /opt/ros/${ROS_DISTRO}/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y'

# Set CARLA Python API in environment
RUN echo "export CARLA_VERSION=${CARLA_VERSION}" >> /opt/carla/setup.bash && \
    echo "export PYTHONPATH=\$PYTHONPATH:/opt/carla/PythonAPI/carla/dist/\$(ls /opt/carla/PythonAPI/carla/dist | grep py3.)" >> /opt/carla/setup.bash && \
    echo "export PYTHONPATH=\$PYTHONPATH:/opt/carla/PythonAPI/carla" >> /opt/carla/setup.bash

# Build CARLA ROS bridge
RUN /bin/bash -c 'source /opt/ros/${ROS_DISTRO}/setup.bash && \
    source /opt/carla/setup.bash && \
    colcon build --packages-select carla_msgs carla_ros_bridge carla_spawn_objects'

# Source workspace in bashrc
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source /opt/carla/setup.bash" >> ~/.bashrc && \
    echo "source /opt/carla-ros-bridge/install/setup.bash" >> ~/.bashrc

# Entry point
COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
```

**Build Command**:
```bash
docker build \
  -t ros2-carla-bridge:foxy \
  -f docker/ros2-carla-bridge.Dockerfile \
  --build-arg CARLA_VERSION=0.9.16 \
  --build-arg ROS_DISTRO=foxy \
  .
```

### Step 2: Build Baseline Controller Image

**Dockerfile**: `docker/baseline-controller.Dockerfile`

```dockerfile
FROM ros2-carla-bridge:foxy

# Install additional Python dependencies for controller
RUN pip3 install --no-cache-dir \
    pyyaml

# Copy baseline controller code
COPY src/baselines /workspace/av_td3_system/src/baselines
COPY src/ros_nodes /workspace/av_td3_system/src/ros_nodes
COPY src/utils /workspace/av_td3_system/src/utils

# Create ROS 2 package
WORKDIR /workspace/av_td3_system
RUN mkdir -p src/av_td3_baseline && \
    cp -r src/baselines src/ros_nodes src/utils src/av_td3_baseline/

# Create package.xml
COPY docker/package.xml src/av_td3_baseline/

# Build baseline package
RUN /bin/bash -c 'source /opt/ros/foxy/setup.bash && \
    source /opt/carla-ros-bridge/install/setup.bash && \
    colcon build --packages-select av_td3_baseline'

# Source baseline package
RUN echo "source /workspace/av_td3_system/install/setup.bash" >> ~/.bashrc

WORKDIR /workspace

CMD ["ros2", "run", "av_td3_baseline", "baseline_controller_node"]
```

**Build Command**:
```bash
docker build \
  -t baseline-controller:foxy \
  -f docker/baseline-controller.Dockerfile \
  .
```

### Step 3: Create Docker Compose Configuration

**File**: `docker-compose.yml`

```yaml
version: '3.8'

services:
  # CARLA Simulation Server
  carla-server:
    image: carlasim/carla:0.9.16
    container_name: carla-server
    runtime: nvidia
    network_mode: host
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
      - DISPLAY=${DISPLAY}
    command: bash CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "bash", "-c", "netstat -tuln | grep :2000"]
      interval: 5s
      timeout: 3s
      retries: 10

  # ROS 2 Bridge
  ros2-bridge:
    image: ros2-carla-bridge:foxy
    container_name: ros2-bridge
    network_mode: host
    environment:
      - ROS_DOMAIN_ID=0
      - PYTHONPATH=/opt/carla/PythonAPI/carla/dist/$(ls /opt/carla/PythonAPI/carla/dist | grep py3.):$/opt/carla/PythonAPI/carla
    command: >
      bash -c "source /opt/ros/foxy/setup.bash &&
               source /opt/carla/setup.bash &&
               source /opt/carla-ros-bridge/install/setup.bash &&
               ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py
               host:=localhost port:=2000 synchronous_mode:=True fixed_delta_seconds:=0.05"
    depends_on:
      carla-server:
        condition: service_healthy
    restart: unless-stopped

  # Baseline Controller
  baseline-controller:
    image: baseline-controller:foxy
    container_name: baseline-controller
    network_mode: host
    environment:
      - ROS_DOMAIN_ID=0
    volumes:
      - ./config/waypoints.txt:/workspace/config/waypoints.txt:ro
      - ./config/baseline_params.yaml:/workspace/config/baseline_params.yaml:ro
      - ./data/logs:/workspace/data/logs:rw
    command: >
      bash -c "source /opt/ros/foxy/setup.bash &&
               source /opt/carla-ros-bridge/install/setup.bash &&
               source /workspace/av_td3_system/install/setup.bash &&
               ros2 run av_td3_baseline baseline_controller_node
               --ros-args --params-file /workspace/config/baseline_params.yaml"
    depends_on:
      - ros2-bridge
    restart: unless-stopped

# Optional: DRL Agent container (for future)
#  drl-agent:
#    image: drl-agent:foxy
#    container_name: drl-agent
#    network_mode: host
#    environment:
#      - ROS_DOMAIN_ID=0
#    volumes:
#      - ./config:/workspace/config:ro
#      - ./data:/workspace/data:rw
#    command: python3 /workspace/scripts/train_td3_ros2.py
#    depends_on:
#      - ros2-bridge
```

### Step 4: Create Helper Scripts

**File**: `docker/ros_entrypoint.sh`

```bash
#!/bin/bash
set -e

# Source ROS 2
source /opt/ros/${ROS_DISTRO}/setup.bash

# Source CARLA environment
if [ -f /opt/carla/setup.bash ]; then
    source /opt/carla/setup.bash
fi

# Source CARLA ROS bridge workspace
if [ -f /opt/carla-ros-bridge/install/setup.bash ]; then
    source /opt/carla-ros-bridge/install/setup.bash
fi

# Source application workspace (if exists)
if [ -f /workspace/av_td3_system/install/setup.bash ]; then
    source /workspace/av_td3_system/install/setup.bash
fi

# Execute command
exec "$@"
```

**File**: `docker/package.xml` (for baseline controller ROS 2 package)

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>av_td3_baseline</name>
  <version>1.0.0</version>
  <description>PID + Pure Pursuit Baseline Controller for CARLA</description>
  
  <maintainer email="danielterragomes@dcc.ufmg.br">Daniel Terra</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_python</buildtool_depend>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>carla_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### Step 5: Build and Test

**Build all images**:
```bash
# Build base ROS 2 + Bridge image
cd /workspace/av_td3_system
docker build -t ros2-carla-bridge:foxy -f docker/ros2-carla-bridge.Dockerfile .

# Build baseline controller image
docker build -t baseline-controller:foxy -f docker/baseline-controller.Dockerfile .
```

**Test complete system**:
```bash
# Launch all containers
docker-compose up

# In another terminal, verify ROS 2 topics
docker exec ros2-bridge ros2 topic list | grep carla

# Test vehicle control
docker exec ros2-bridge ros2 topic echo /carla/ego_vehicle/odometry --once

# Monitor baseline controller
docker logs -f baseline-controller
```

---

## Configuration Files

### `config/baseline_params.yaml`

```yaml
baseline_controller_node:
  ros__parameters:
    # PID Controller Parameters
    pid:
      kp: 0.50
      ki: 0.30
      kd: 0.13
      integrator_min: 0.0
      integrator_max: 10.0

    # Pure Pursuit Parameters
    pure_pursuit:
      lookahead_distance: 2.0  # meters
      kp_heading: 8.00
      cross_track_deadband: 0.01  # meters
      
    # Waypoint Configuration
    waypoints:
      file_path: "/workspace/config/waypoints.txt"
      target_speed: 8.33  # m/s (30 km/h)
      
    # Control Loop
    control_frequency: 20.0  # Hz
    
    # Safety
    collision_threshold: 0.1  # stop if collision detected
    max_steering_angle: 1.0  # radians
    max_throttle: 1.0
    max_brake: 1.0
```

---

## Advantages of Docker Architecture

### 1. **Portability** ✅
- **Single command deployment**: `docker-compose up` on any machine
- **Supercomputer ready**: No host dependencies beyond Docker + NVIDIA runtime
- **Version control**: Docker images are immutable, reproducible environments

### 2. **Modularity** ✅
- **Swap controllers**: Change `baseline-controller` ↔ `drl-agent` by modifying docker-compose.yml
- **Independent scaling**: Run multiple baseline containers for parallel evaluation
- **Isolated environments**: Python dependencies don't conflict

### 3. **Development Workflow** ✅
- **Local development**: Build/test on workstation
- **Remote deployment**: Push images to registry, pull on supercomputer
- **CI/CD integration**: Automate builds with GitHub Actions

### 4. **Resource Management** ✅
- **GPU allocation**: Assign specific GPUs to CARLA via `NVIDIA_VISIBLE_DEVICES`
- **Memory limits**: Set container resource limits for supercomputer quotas
- **Network isolation**: Containers don't interfere with host services

---

## Migration from Existing Code

### From `carla_env.py` (Direct API) to ROS 2

**Current** (`carla_env.py` lines 150-180):
```python
# Direct CARLA Python API
self.client = carla.Client('localhost', 2000)
self.world = self.client.get_world()
self.vehicle = self.world.spawn_actor(bp, spawn_point)

# Step: Apply control directly
control = carla.VehicleControl(throttle=action[0], steer=action[1])
self.vehicle.apply_control(control)

# Get state directly
transform = self.vehicle.get_transform()
velocity = self.vehicle.get_velocity()
```

**Future** (`carla_env_ros2.py` - for DRL agent):
```python
# ROS 2 API
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleControl

class CARLAEnvROS2(Node):
    def __init__(self):
        super().__init__('carla_env_ros2')
        
        # Publishers
        self.control_pub = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/ego_vehicle/vehicle_control_cmd',
            10
        )
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/carla/ego_vehicle/odometry',
            self.odom_callback,
            10
        )
        
    def step(self, action):
        # Publish control
        msg = CarlaEgoVehicleControl()
        msg.throttle = action[0]
        msg.steer = action[1]
        self.control_pub.publish(msg)
        
        # Wait for state update (handled by callback)
        rclpy.spin_once(self, timeout_sec=0.1)
        return self.current_state, reward, done, info
```

### From `controller2d.py` to ROS 2 Baseline Node

**Current** (legacy code):
```python
# Old CARLA API (module_7.py style)
from carla.client import make_carla_client, VehicleControl

client = make_carla_client('localhost', 2000)
# ... initialize controller
throttle, steer, brake = controller.update_controls()
control = VehicleControl(throttle=throttle, steer=steer, brake=brake)
client.send_control(control)
```

**New** (ROS 2 node - `src/ros_nodes/baseline_controller_node.py`):
```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleControl
from src.baselines.pid_pure_pursuit import PIDPurePursuitController

class BaselineControllerNode(Node):
    def __init__(self):
        super().__init__('baseline_controller_node')
        
        # Load parameters
        self.declare_parameters...
        
        # Initialize controller (extracted from controller2d.py)
        self.controller = PIDPurePursuitController(
            kp=self.get_parameter('pid.kp').value,
            ki=self.get_parameter('pid.ki').value,
            ...
        )
        
        # Load waypoints
        waypoints = load_waypoints('/workspace/config/waypoints.txt')
        self.controller.set_waypoints(waypoints)
        
        # ROS 2 pub/sub
        self.control_pub = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/ego_vehicle/vehicle_control_cmd',
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/carla/ego_vehicle/odometry',
            self.odom_callback,
            10
        )
        
        # Control loop timer
        self.timer = self.create_timer(0.05, self.control_loop)  # 20 Hz
        
    def odom_callback(self, msg):
        # Extract state from odometry
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        # ... extract yaw, speed
        
    def control_loop(self):
        # Compute control
        throttle, steer, brake = self.controller.update_controls(
            self.current_x, self.current_y, self.current_yaw, self.current_speed
        )
        
        # Publish control command
        msg = CarlaEgoVehicleControl()
        msg.throttle = float(throttle)
        msg.steer = float(steer)
        msg.brake = float(brake)
        self.control_pub.publish(msg)
```

---

## Testing Strategy

### Level 1: Individual Container Testing

**Test 1: CARLA Server**
```bash
docker run --runtime=nvidia --net=host \
  carlasim/carla:0.9.16 bash CarlaUE4.sh -RenderOffScreen -nosound

# Verify: Port 2000 listening
docker exec carla-server netstat -tuln | grep 2000
```

**Test 2: ROS 2 Bridge**
```bash
docker run -it --net=host ros2-carla-bridge:foxy bash
# Inside container:
source /opt/ros/foxy/setup.bash
source /opt/carla-ros-bridge/install/setup.bash
ros2 topic list
ros2 launch carla_ros_bridge carla_ros_bridge.launch.py
```

**Test 3: Baseline Controller**
```bash
docker run -it --net=host -v $(pwd)/config:/workspace/config \
  baseline-controller:foxy bash
# Inside container:
ros2 run av_td3_baseline baseline_controller_node
```

### Level 2: Multi-Container Integration

**Test 4: docker-compose**
```bash
docker-compose up

# In another terminal:
docker-compose logs -f carla-server  # Check CARLA startup
docker-compose logs -f ros2-bridge   # Check bridge connection
docker-compose logs -f baseline-controller  # Check control loop

# Verify topics
docker exec ros2-bridge ros2 topic list
docker exec ros2-bridge ros2 topic hz /carla/ego_vehicle/odometry
docker exec ros2-bridge ros2 topic echo /carla/ego_vehicle/vehicle_control_cmd
```

### Level 3: Functional Validation

**Test 5: Waypoint Following**
```bash
# Run system with baseline controller
docker-compose up baseline-controller

# Monitor in CARLA (if display available)
docker run --net=host --env DISPLAY=$DISPLAY \
  carlasim/carla:0.9.16 bash CarlaUE4.sh

# Verify: Vehicle follows waypoints from config/waypoints.txt
# Metrics: Lane keeping error, speed tracking, no collisions
```

---

## Deployment to Supercomputer

### Step 1: Build Images Locally

```bash
# On development workstation
cd /workspace/av_td3_system

# Build all images
docker build -t ros2-carla-bridge:foxy -f docker/ros2-carla-bridge.Dockerfile .
docker build -t baseline-controller:foxy -f docker/baseline-controller.Dockerfile .
docker build -t drl-agent:foxy -f docker/drl-agent.Dockerfile .  # Future
```

### Step 2: Push to Registry

```bash
# Tag for registry
docker tag ros2-carla-bridge:foxy myregistry.io/av_td3/ros2-carla-bridge:foxy
docker tag baseline-controller:foxy myregistry.io/av_td3/baseline-controller:foxy

# Push to registry
docker push myregistry.io/av_td3/ros2-carla-bridge:foxy
docker push myregistry.io/av_td3/baseline-controller:foxy
```

### Step 3: Pull on Supercomputer

```bash
# On supercomputer
docker pull myregistry.io/av_td3/ros2-carla-bridge:foxy
docker pull myregistry.io/av_td3/baseline-controller:foxy

# Retag
docker tag myregistry.io/av_td3/ros2-carla-bridge:foxy ros2-carla-bridge:foxy
docker tag myregistry.io/av_td3/baseline-controller:foxy baseline-controller:foxy
```

### Step 4: Run on Supercomputer

```bash
# Transfer config files
scp -r config/ supercomputer:/path/to/workspace/
scp docker-compose.yml supercomputer:/path/to/workspace/

# SSH to supercomputer
ssh supercomputer

# Launch system
cd /path/to/workspace
docker-compose up -d  # Detached mode

# Monitor
docker-compose logs -f
```

---

## Next Steps (Phase 2 Implementation)

### Task 8: ✅ Fetch Docker + ROS 2 integration docs (COMPLETE)

**Status**: Documentation research complete  
**Output**: This document (PHASE_2_DOCKER_ARCHITECTURE.md)

### Task 9: Design Docker Architecture (COMPLETE)

**Status**: Multi-container architecture designed  
**Components**:
- 3 containers: CARLA server, ROS 2 bridge, Baseline controller
- Network: host mode for low latency
- Volumes: waypoints, logs shared

### Task 10: Create ROS 2 + Bridge Dockerfile (NEXT)

**File**: `docker/ros2-carla-bridge.Dockerfile`  
**Action**: Implement multi-stage build extracting CARLA Python API  
**Test**: Build image, verify ros2 topic list works

### Task 11: Test Docker bridge connection

**Action**: Launch CARLA + bridge containers, verify topics  
**Success criteria**:
- Bridge connects to CARLA on port 2000
- `/carla/ego_vehicle/odometry` publishes at 20 Hz
- Synchronous mode active (`/clock` topic present)

### Task 12-14: Verify topics and control

**Action**: Test pub/sub between containers  
**Success criteria**:
- Manual control via `ros2 topic pub` moves vehicle
- Odometry data reflects vehicle movement
- Document test results in DOCKER_ROS2_BRIDGE_TEST.md

---

## Risk Mitigation

### Risk 1: Bridge Version Incompatibility (0.9.12 vs 0.9.16)

**Mitigation**:
- Use master branch of carla-ros-bridge (may have 0.9.16 fixes)
- Monitor bridge GitHub issues for 0.9.16 reports
- If incompatible, downgrade CARLA to 0.9.13 (not preferred)

### Risk 2: ROS 2 DDS Discovery Issues in Docker

**Mitigation**:
- Use host network mode (already planned)
- Set `ROS_DOMAIN_ID` consistently across containers
- If issues persist, use `ROS_LOCALHOST_ONLY=1` for single-host deployment

### Risk 3: Performance Degradation (Docker Overhead)

**Mitigation**:
- Host network eliminates bridge overhead
- NVIDIA runtime provides direct GPU access
- Benchmark against native installation (expect <5% overhead)

---

## Success Criteria for Phase 2

- [ ] ROS 2 + Bridge Docker image builds successfully
- [ ] Baseline controller Docker image builds successfully
- [ ] docker-compose launches all 3 containers without errors
- [ ] Bridge connects to CARLA and publishes topics at 20 Hz
- [ ] Manual vehicle control via ROS 2 topics works
- [ ] Test results documented in DOCKER_ROS2_BRIDGE_TEST.md
- [ ] Architecture validated for supercomputer deployment

---

**Document Status**: ✅ COMPLETE - Architecture designed, ready for implementation

**Next Action**: Begin Task 10 - Create `docker/ros2-carla-bridge.Dockerfile`

**Estimated Time to Phase 2 Completion**: 3-4 hours (build, test, document)
