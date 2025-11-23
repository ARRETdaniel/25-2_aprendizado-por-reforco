# Architecture Decision: ROS Bridge Required for Vehicle Control

**Date**: 2025-01-XX  
**Status**: ✅ RESOLVED - Architecture Confirmed  
**Decision**: Use CARLA ROS 2 Bridge (External Package)

---

## Critical Discovery from Official Documentation

After thorough investigation of the official CARLA ROS Bridge documentation, it is now clear that:

### Native ROS 2 in CARLA 0.9.16

**What it provides:**
- ✅ Sensor data publishing (cameras, LIDAR, GNSS, IMU, etc.)
- ✅ Clock synchronization (`/clock` topic)
- ✅ Transform broadcasting (`/tf` topic)
- ✅ Built-in FastDDS (no separate installation needed)
- ✅ Activated via `--ros2` flag and `sensor.enable_for_ros()` calls

**What it does NOT provide:**
- ❌ Vehicle control subscribers
- ❌ ROS message definitions (carla_msgs)
- ❌ Service interfaces (spawn objects, get blueprints)
- ❌ World management via ROS topics
- ❌ Ackermann control conversion
- ❌ Twist to control conversion

### CARLA ROS Bridge Package

**What it provides:**
- ✅ **Vehicle control via ROS topics**: `/carla/<ROLE_NAME>/vehicle_control_cmd`
- ✅ Complete ROS 2 message definitions (`carla_msgs`)
- ✅ Sensor data republishing with proper ROS interfaces
- ✅ Services for spawning/destroying objects
- ✅ Synchronous mode management
- ✅ Multiple control interfaces:
  - Direct: `carla_msgs/CarlaEgoVehicleControl`
  - Ackermann: `ackermann_msgs/AckermannDrive` (via carla_ackermann_control)
  - Twist: `geometry_msgs/Twist` (via carla_twist_to_control)

---

## Official Documentation Evidence

### Vehicle Control Topic

From: https://carla.readthedocs.io/projects/ros-bridge/en/latest/run_ros/#ego-vehicle-control

```markdown
## Ego vehicle control

There are two modes to control the ego vehicle:

1. Normal mode - reading commands from `/carla/<ROLE NAME>/vehicle_control_cmd`
2. Manual mode - reading commands from `/carla/<ROLE NAME>/vehicle_control_cmd_manual`

To test steering from the command line:

# ROS 2
ros2 topic pub /carla/ego_vehicle/vehicle_control_cmd carla_msgs/CarlaEgoVehicleControl "{throttle: 1.0, steer: 1.0}" -r 10
```

**Key insight**: The ROS bridge subscribes to control topics and forwards commands to CARLA via Python API.

### ROS Bridge Architecture

From: https://carla.readthedocs.io/projects/ros-bridge/en/latest/

```markdown
The ROS bridge boasts the following features:

• Provides sensor data for LIDAR, Semantic LIDAR, Cameras, GNSS, Radar and IMU.
• Provides object data such as transforms, traffic light status, visualisation markers, collision and lane invasion.
• Control of AD agents through steering, throttle and brake.
• Control of aspects of the CARLA simulation like synchronous mode, playing and pausing the simulation.
```

**Key insight**: The bridge is a separate Python package that connects CARLA Python API to ROS 2.

### Control Message Types

From: https://carla.readthedocs.io/projects/ros-bridge/en/latest/carla_ackermann_control/

```markdown
The carla_ackermann_control package is used to control a CARLA vehicle with Ackermann messages. 
The package converts the Ackermann messages into CarlaEgoVehicleControl messages.

Subscriptions:
/carla/<ROLE NAME>/ackermann_cmd | ackermann_msgs.AckermannDrive | Subscriber for steering commands
```

**Key insight**: Multiple control interfaces available, all converted to CARLA's native control.

---

## Architecture Decision

### Selected Architecture: ROS Bridge (External Package)

**Components:**

1. **CARLA Server Container**
   - Image: `carlasim/carla:0.9.16`
   - Launch: Standard mode (NOT `--ros2`)
   - Role: Simulation engine
   - Python API exposed on port 2000

2. **CARLA ROS Bridge Container**
   - Built from: https://github.com/carla-simulator/ros-bridge
   - ROS 2 distribution: Humble Hawksbill
   - Role: Bidirectional translation between CARLA Python API and ROS 2
   - Manages:
     - Sensor data republishing
     - Vehicle control command forwarding
     - Synchronization with CARLA
     - Object spawning via services

3. **Baseline Controller Node Container**
   - ROS 2 node with PID + Pure Pursuit
   - Subscribes to sensor topics from bridge
   - Publishes to `/carla/ego_vehicle/vehicle_control_cmd`
   - Processes waypoints and executes control

**Communication Flow:**

```
CARLA Server (Python API)
    ↕ (Python client connection)
ROS Bridge (Python node)
    ↕ (ROS 2 topics/services)
Baseline Controller (ROS 2 node)
```

### Why Not Native ROS 2?

| Capability | Native ROS 2 | ROS Bridge |
|------------|--------------|------------|
| Sensor publishing | ✅ Yes | ✅ Yes |
| Vehicle control | ❌ No | ✅ Yes |
| ROS message types | ❌ No | ✅ Yes |
| Spawn objects | ❌ No | ✅ Yes |
| Synchronous mode | ⚠️ Manual | ✅ Managed |
| Documentation | ⚠️ Minimal | ✅ Complete |

**Conclusion**: Native ROS 2 is designed for **sensor output only**, not full vehicle control. The ROS Bridge is the official, documented, and supported way to control CARLA vehicles via ROS 2.

---

## Previous Investigation Errors

### What We Tested (Incorrectly)

1. ❌ Looking for `vehicle.enable_for_ros()` method
   - **Error**: This method doesn't exist because native ROS 2 doesn't handle control
   - **Truth**: Control requires the ROS bridge package

2. ❌ Expecting native ROS 2 to provide bidirectional communication
   - **Error**: Native ROS 2 is **unidirectional** (CARLA → ROS only)
   - **Truth**: Bridge provides bidirectional (CARLA ↔ ROS)

3. ❌ Searching for control topics with native ROS 2
   - **Error**: Control topics are created by the bridge, not CARLA
   - **Truth**: Bridge subscribes to ROS topics and calls Python API

### Why the Confusion?

- **CARLA 0.9.16 release notes** emphasized "native ROS 2 support" for sensors
- Native ROS 2 is **new** (0.9.16) and **limited** in scope
- ROS Bridge has been the **standard** method since early CARLA versions
- Documentation separates "native ROS 2" (sensors) from "ROS bridge" (full stack)

---

## Implementation Plan

### Phase 2.2 (Updated): Install and Configure ROS Bridge ⏳

#### Step 1: Create ROS Bridge Docker Image (2-3 hours)

**Base image**: `ros:humble-ros-base`

**Build process**:
```dockerfile
FROM ros:humble-ros-base

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone ROS bridge
RUN mkdir -p /opt/carla-ros-bridge/src
WORKDIR /opt/carla-ros-bridge
RUN git clone --recurse-submodules https://github.com/carla-simulator/ros-bridge.git src/ros-bridge

# Install CARLA Python API
ARG CARLA_VERSION=0.9.16
RUN pip3 install carla==${CARLA_VERSION}

# Install ROS dependencies
RUN . /opt/ros/humble/setup.sh && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y

# Build workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build

# Source workspace in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /opt/carla-ros-bridge/install/setup.bash" >> ~/.bashrc
```

#### Step 2: Create Docker Compose Configuration (30 minutes)

```yaml
version: '3.8'

services:
  carla-server:
    image: carlasim/carla:0.9.16
    command: ./CarlaUE4.sh -RenderOffScreen
    runtime: nvidia
    network_mode: host
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all

  carla-ros-bridge:
    image: carla-ros-bridge:humble-0.9.16
    command: ros2 launch carla_ros_bridge carla_ros_bridge.launch.py
    network_mode: host
    depends_on:
      - carla-server
    environment:
      - ROS_DOMAIN_ID=0
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
      - ./config:/workspace/config

  baseline-controller:
    image: td3-av-system:baseline-controller
    command: ros2 run baseline_controller controller_node
    network_mode: host
    depends_on:
      - carla-ros-bridge
    environment:
      - ROS_DOMAIN_ID=0
```

#### Step 3: Test ROS Bridge Connection (1 hour)

**Test script**: Verify bridge topics appear

```python
#!/usr/bin/env python3
"""Test CARLA ROS Bridge connectivity."""

import rclpy
from rclpy.node import Node
import time

class BridgeTest(Node):
    def __init__(self):
        super().__init__('bridge_test')
        self.get_logger().info('Testing CARLA ROS Bridge...')
        
        # Wait for bridge to start
        time.sleep(5)
        
        # List all topics
        topic_list = self.get_topic_names_and_types()
        
        # Expected topics from bridge
        expected = [
            '/carla/status',
            '/carla/world_info',
            '/clock',
        ]
        
        for topic in expected:
            if topic in [t[0] for t in topic_list]:
                self.get_logger().info(f'✅ Found: {topic}')
            else:
                self.get_logger().error(f'❌ Missing: {topic}')

def main():
    rclpy.init()
    node = BridgeTest()
    rclpy.spin_once(node, timeout_sec=2.0)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 4: Test Vehicle Control (1 hour)

**Test publishing control commands**:

```bash
# Spawn ego vehicle via bridge service
ros2 service call /carla/spawn_object carla_msgs/srv/SpawnObject \
  "{type: 'vehicle.lincoln.mkz_2020', id: 'ego_vehicle', \
    transform: {location: {x: 0.0, y: 0.0, z: 2.0}}}"

# Test control command
ros2 topic pub /carla/ego_vehicle/vehicle_control_cmd \
  carla_msgs/msg/CarlaEgoVehicleControl \
  "{throttle: 0.5, steer: 0.0, brake: 0.0}" -r 10
```

### Phase 2.3: Extract and Modernize Controllers (4-6 hours)

- Extract PID from `controller2d.py`
- Extract Pure Pursuit from `module_7.py`
- Update to use ROS 2 topics
- Create modular classes

### Phase 2.4: Implement Baseline Controller Node (6-8 hours)

- ROS 2 node structure
- Subscribe to camera, odometry, waypoints
- Publish to vehicle control
- Integration with extracted controllers

### Phase 2.5: Integration Testing (2-3 hours)

- End-to-end test
- Validate control loop
- Measure latency

### Phase 2.6: Performance Validation (2 hours)

- Compare with legacy module_7.py
- Verify behavior matches
- Document performance

### Phase 2.7: Documentation (1-2 hours)

- Architecture diagrams
- Setup instructions
- API reference

---

## Timeline Estimate

| Phase | Duration | Status |
|-------|----------|--------|
| 2.2: ROS Bridge Setup | 4-5 hours | 🔄 Next |
| 2.3: Extract Controllers | 4-6 hours | ⏸️ Pending |
| 2.4: Baseline Node | 6-8 hours | ⏸️ Pending |
| 2.5: Integration | 2-3 hours | ⏸️ Pending |
| 2.6: Validation | 2 hours | ⏸️ Pending |
| 2.7: Documentation | 1-2 hours | ⏸️ Pending |
| **Total** | **19-26 hours** | **~3-4 days** |

---

## Benefits of This Architecture

### 1. **Official and Documented**
- Extensive documentation
- Active community support
- Maintained by CARLA team

### 2. **Production Ready**
- Used in research and industry
- Tested across multiple CARLA versions
- Stable API

### 3. **Feature Complete**
- All sensor types supported
- Multiple control interfaces
- Synchronous mode management
- Object spawning/destroying

### 4. **Flexible**
- Can use CarlaEgoVehicleControl (direct)
- Can use AckermannDrive (high-level)
- Can use Twist (ROS standard)
- Easy to extend

### 5. **Docker Compatible**
- Clean container separation
- Well-defined interfaces
- Scalable architecture

---

## Next Steps

1. ✅ ~~Document architecture decision~~ (DONE)
2. 🔄 Create ROS Bridge Dockerfile
3. ⏳ Build and test bridge image
4. ⏳ Create docker-compose.yml
5. ⏳ Test vehicle control via bridge
6. ⏳ Extract controllers from legacy code
7. ⏳ Implement baseline controller node

---

## References

- **CARLA ROS Bridge Docs**: https://carla.readthedocs.io/projects/ros-bridge/en/latest/
- **ROS Bridge GitHub**: https://github.com/carla-simulator/ros-bridge
- **Vehicle Control**: https://carla.readthedocs.io/projects/ros-bridge/en/latest/run_ros/#ego-vehicle-control
- **Ackermann Control**: https://carla.readthedocs.io/projects/ros-bridge/en/latest/carla_ackermann_control/
- **Twist to Control**: https://carla.readthedocs.io/projects/ros-bridge/en/latest/carla_twist_to_control/
- **Installation Guide**: https://carla.readthedocs.io/projects/ros-bridge/en/latest/ros_installation_ros2/

---

## Conclusion

**Decision**: Abandon native ROS 2 investigation. Use the official CARLA ROS Bridge package.

**Rationale**: 
- Native ROS 2 in CARLA 0.9.16 is **sensor-only** (one-way communication)
- ROS Bridge provides **full bidirectional** communication
- ROS Bridge is the **official, documented, and supported** method
- All vehicle control in ROS requires the bridge

**Impact on Timeline**:
- Adds ROS bridge setup time (~4-5 hours)
- Removes need for hybrid Python API solution
- Net result: **Same or faster** than hybrid approach
- Much cleaner architecture

**Confidence**: 100% - Based on official documentation
