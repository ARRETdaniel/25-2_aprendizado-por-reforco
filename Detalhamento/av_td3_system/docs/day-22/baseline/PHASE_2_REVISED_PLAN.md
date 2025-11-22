# Phase 2 Implementation Plan - REVISED

**Date**: 2025-01-XX  
**Status**: 🔄 UPDATED after ROS 2 Native Investigation  
**Strategy**: Dual-track approach (Bridge now, Native later if needed)

---

## Investigation Summary

After thorough investigation triggered by user's valid challenge, we discovered:

✅ **CARLA 0.9.16 DOES have native ROS 2 support** (`--ros2` flag)  
❌ **BUT it's NOT compiled into Docker images** (carlasim/carla:0.9.16)  
📋 See full findings in: `ROS2_NATIVE_INVESTIGATION_FINDINGS.md`

---

## Revised Architecture Decision

### **Track 1: External Bridge (IMMEDIATE)** ⭐ PRIMARY

**Why**:
- ✅ Works with official Docker images (no custom builds)
- ✅ Well-documented and tested by community
- ✅ Lower risk, faster implementation
- ✅ Already designed (just needs path fix)
- ⚠️ Slightly higher latency (acceptable for baseline PID+Pure Pursuit)

**Timeline**: Week 1-2

**Architecture**:
```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  CARLA Server       │     │  ROS 2 Bridge        │     │ Baseline Controller │
│  (carlasim:0.9.16) │◄───►│  (custom Foxy)       │◄───►│  (PID + Pure Pursuit│
│                     │     │                      │     │   ROS 2 node)       │
│  - Town01           │     │  - Python API client │     │                     │
│  - Ego vehicle      │     │  - Topic publishers  │     │  - Subscribe: odom  │
│  - Sensors          │     │  - Control subscriber│     │  - Publish: control │
│  - NPCs             │     │                      │     │  - PID (long.)      │
│                     │     │                      │     │  - Pure Pursuit     │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
     Port 2000                   Host network                 Host network
```

---

### **Track 2: Native ROS 2 (FUTURE)** 🚀 OPTIONAL

**When to activate**:
- IF bridge latency impacts control performance
- IF we need to match TD3's 20Hz+ control loop
- IF paper reviewers question "native ROS 2" claims

**Why deferred**:
- ⏱️ Requires building CARLA from source (4+ hours)
- 💾 Large build environment (130GB disk space)
- 🔧 More complex Docker setup
- ❓ Less documented than bridge approach

**Architecture** (if implemented):
```
┌─────────────────────┐     ┌─────────────────────┐
│  CARLA Server       │     │ Baseline Controller │
│  (custom w/ ROS 2)  │◄───►│  (ROS 2 node)       │
│                     │     │                     │
│  - Town01           │     │  - Subscribe: odom  │
│  - Ego vehicle      │     │  - Publish: control │
│  - Sensors          │     │  - PID + Pure Pur.  │
│  - Built-in FastDDS │     │                     │
│  - Launch: --ros2   │     │                     │
└─────────────────────┘     └─────────────────────┘
   Native DDS comms          DDS participant
```

**Decision Point**: After baseline evaluation (Week 3)

---

## Phase 2 Roadmap - Track 1 (External Bridge)

### **Week 1: Docker Infrastructure** ✅ 80% Complete

#### Task 1.1: Fix ROS 2 Bridge Dockerfile ⚠️ BLOCKER
**Current Issue**: Wrong CARLA Python API path
```dockerfile
# WRONG (current):
COPY --from=carla /home/carla/PythonAPI /opt/carla/PythonAPI

# CORRECT (needed):
COPY --from=carla /workspace/PythonAPI /opt/carla/PythonAPI
```

**Action**:
- [ ] Update `docker/ros2-carla-bridge.Dockerfile` line ~20
- [ ] Test build: `bash docker/build_ros2_bridge.sh`
- [ ] Verify CARLA package import in container

**Acceptance Criteria**:
- Image builds successfully without errors
- Container can `import carla` without ImportError
- Bridge process can connect to CARLA on port 2000

---

#### Task 1.2: Test Bridge Connectivity
**Prerequisites**: Task 1.1 complete

**Commands**:
```bash
# Terminal 1: Start CARLA server
docker-compose -f docker-compose.baseline.yml up carla-server

# Terminal 2: Start bridge
docker-compose -f docker-compose.baseline.yml up ros2-bridge

# Terminal 3: Verify topics
docker exec ros2-bridge-container bash -c "source /opt/ros/foxy/setup.bash && ros2 topic list"
```

**Expected Topics**:
```
/carla/ego_vehicle/odometry
/carla/ego_vehicle/vehicle_control_cmd
/carla/ego_vehicle/camera/rgb/front/image_color
/tf
/clock
```

**Acceptance Criteria**:
- Bridge connects to CARLA without timeout
- Topics are published at expected rates
- No error messages in bridge logs
- Vehicle spawns successfully in Town01

---

### **Week 1-2: Baseline Controller Implementation**

#### Task 2.1: Extract PID+Pure Pursuit Code
**Source Files**:
- `related_works/.../controller2d.py` - PID implementation
- `related_works/.../module_7.py` - Pure Pursuit + old CARLA API

**Target**: `src/baselines/pid_pure_pursuit.py`

**Modernization Steps**:

1. **Extract PID Controller**:
```python
# From controller2d.py
class PIDController:
    """
    Longitudinal velocity controller using PID.
    
    Parameters (from controller2d.py):
        kp: 0.50 (proportional gain)
        ki: 0.30 (integral gain)
        kd: 0.13 (derivative gain)
    """
    def __init__(self, kp=0.50, ki=0.30, kd=0.13):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
    
    def update(self, target_velocity, current_velocity, dt):
        """
        Calculate throttle/brake from velocity error.
        
        Returns:
            float: throttle [0,1] if positive, brake [0,1] if negative
        """
        error = target_velocity - current_velocity
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        
        return output
```

2. **Extract Pure Pursuit**:
```python
# From module_7.py, modernized
class PurePursuitController:
    """
    Lateral control using Pure Pursuit algorithm.
    
    Parameters:
        lookahead_distance: 2.0m (from controller2d.py)
    """
    def __init__(self, lookahead_distance=2.0, wheelbase=2.89):
        self.lookahead = lookahead_distance
        self.wheelbase = wheelbase  # Lincoln MKZ wheelbase
    
    def find_target_waypoint(self, waypoints, current_position):
        """
        Find waypoint at lookahead distance ahead.
        
        Args:
            waypoints: List of carla.Location
            current_position: carla.Location
        
        Returns:
            carla.Location: Target waypoint
        """
        # Find closest waypoint
        min_dist = float('inf')
        closest_idx = 0
        
        for i, wp in enumerate(waypoints):
            dist = current_position.distance(wp)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Find waypoint at lookahead distance
        cumulative_dist = 0.0
        for i in range(closest_idx, len(waypoints) - 1):
            segment_dist = waypoints[i].distance(waypoints[i+1])
            cumulative_dist += segment_dist
            
            if cumulative_dist >= self.lookahead:
                return waypoints[i+1]
        
        return waypoints[-1]  # Return last waypoint if path is short
    
    def calculate_steering(self, target_waypoint, current_transform):
        """
        Calculate steering angle using Pure Pursuit.
        
        Args:
            target_waypoint: carla.Location
            current_transform: carla.Transform (vehicle pose)
        
        Returns:
            float: steering angle [-1, 1]
        """
        # Transform waypoint to vehicle frame
        forward_vec = current_transform.get_forward_vector()
        right_vec = current_transform.get_right_vector()
        
        target_vec = target_waypoint - current_transform.location
        
        # Lateral error (distance to right)
        lateral_error = target_vec.x * right_vec.x + target_vec.y * right_vec.y
        
        # Pure Pursuit formula
        curvature = 2.0 * lateral_error / (self.lookahead ** 2)
        steering = math.atan(curvature * self.wheelbase)
        
        # Normalize to [-1, 1]
        max_steer = math.radians(70)  # CARLA default max steer
        steering = np.clip(steering / max_steer, -1.0, 1.0)
        
        return steering
```

3. **Load Waypoints**:
```python
# From FinalProject/waypoints.txt, modernize to CARLA 0.9.16 API
def load_waypoints_town01(filepath="waypoints.txt"):
    """
    Load waypoints from text file.
    
    Format (from FinalProject/waypoints.txt):
        x y z
        example: -39.5 13.2 0.2
    
    Returns:
        List[carla.Location]
    """
    import carla
    
    waypoints = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                x, y, z = map(float, line.split())
                waypoints.append(carla.Location(x=x, y=y, z=z))
    
    return waypoints
```

**Acceptance Criteria**:
- All functions have type hints
- Docstrings explain parameters and return values
- No dependencies on old CARLA API (cutils, etc.)
- Unit tests pass (simple waypoint following test)

---

#### Task 2.2: Create ROS 2 Node Wrapper
**Target**: `src/ros_nodes/baseline_controller_node.py`

**Implementation**:
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleControl
from geometry_msgs.msg import Twist
import numpy as np

from baselines.pid_pure_pursuit import PIDController, PurePursuitController, load_waypoints_town01

class BaselineControllerNode(Node):
    """
    ROS 2 node for PID+Pure Pursuit baseline controller.
    
    Subscribes:
        /carla/ego_vehicle/odometry (nav_msgs/Odometry)
    
    Publishes:
        /carla/ego_vehicle/vehicle_control_cmd (carla_msgs/CarlaEgoVehicleControl)
    """
    
    def __init__(self):
        super().__init__('baseline_controller')
        
        # Parameters (ROS 2 params for tunability)
        self.declare_parameter('target_velocity', 30.0)  # km/h
        self.declare_parameter('pid_kp', 0.50)
        self.declare_parameter('pid_ki', 0.30)
        self.declare_parameter('pid_kd', 0.13)
        self.declare_parameter('lookahead_distance', 2.0)
        self.declare_parameter('waypoints_file', '/waypoints/waypoints.txt')
        self.declare_parameter('control_frequency', 20.0)  # Hz
        
        # Load parameters
        target_vel = self.get_parameter('target_velocity').value
        self.target_velocity = target_vel / 3.6  # Convert km/h to m/s
        
        # Controllers
        self.pid = PIDController(
            kp=self.get_parameter('pid_kp').value,
            ki=self.get_parameter('pid_ki').value,
            kd=self.get_parameter('pid_kd').value
        )
        
        self.pure_pursuit = PurePursuitController(
            lookahead_distance=self.get_parameter('lookahead_distance').value
        )
        
        # Load waypoints
        waypoints_file = self.get_parameter('waypoints_file').value
        self.waypoints = load_waypoints_town01(waypoints_file)
        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints')
        
        # ROS 2 communication
        self.odom_sub = self.create_subscription(
            Odometry,
            '/carla/ego_vehicle/odometry',
            self.odometry_callback,
            10
        )
        
        self.control_pub = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/ego_vehicle/vehicle_control_cmd',
            10
        )
        
        # State
        self.current_velocity = 0.0
        self.current_transform = None
        self.last_update_time = self.get_clock().now()
        
        # Control timer
        control_freq = self.get_parameter('control_frequency').value
        control_period = 1.0 / control_freq
        self.control_timer = self.create_timer(control_period, self.control_loop)
        
        self.get_logger().info('Baseline controller initialized')
    
    def odometry_callback(self, msg):
        """Process odometry data from CARLA."""
        # Extract velocity (linear.x in vehicle frame)
        self.current_velocity = msg.twist.twist.linear.x  # m/s
        
        # Extract pose (for Pure Pursuit)
        pose = msg.pose.pose
        self.current_transform = carla.Transform(
            location=carla.Location(
                x=pose.position.x,
                y=pose.position.y,
                z=pose.position.z
            ),
            rotation=carla.Rotation(
                pitch=0,  # Can extract from quaternion if needed
                yaw=self.quaternion_to_yaw(pose.orientation),
                roll=0
            )
        )
    
    def control_loop(self):
        """Main control loop (called at control_frequency Hz)."""
        if self.current_transform is None:
            self.get_logger().warn('No odometry received yet', throttle_duration_sec=1.0)
            return
        
        # Calculate dt
        current_time = self.get_clock().now()
        dt = (current_time - self.last_update_time).nanoseconds / 1e9
        self.last_update_time = current_time
        
        # Longitudinal control (PID)
        throttle_brake = self.pid.update(
            self.target_velocity,
            self.current_velocity,
            dt
        )
        
        # Separate throttle and brake
        if throttle_brake >= 0:
            throttle = min(throttle_brake, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(-throttle_brake, 1.0)
        
        # Lateral control (Pure Pursuit)
        target_waypoint = self.pure_pursuit.find_target_waypoint(
            self.waypoints,
            self.current_transform.location
        )
        
        steer = self.pure_pursuit.calculate_steering(
            target_waypoint,
            self.current_transform
        )
        
        # Publish control command
        control_msg = CarlaEgoVehicleControl()
        control_msg.throttle = float(throttle)
        control_msg.steer = float(steer)
        control_msg.brake = float(brake)
        control_msg.hand_brake = False
        control_msg.reverse = False
        control_msg.manual_gear_shift = False
        control_msg.gear = 1
        
        self.control_pub.publish(control_msg)
        
        # Debug logging
        self.get_logger().debug(
            f'vel={self.current_velocity:.1f} m/s, '
            f'throttle={throttle:.2f}, brake={brake:.2f}, steer={steer:.2f}'
        )
    
    @staticmethod
    def quaternion_to_yaw(q):
        """Convert quaternion to yaw angle (degrees)."""
        import math
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return math.degrees(yaw)


def main(args=None):
    rclpy.init(args=args)
    node = BaselineControllerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

**Acceptance Criteria**:
- Node subscribes to odometry topic
- Node publishes control commands
- ROS 2 parameters are declared and used
- Logging provides debug information
- Node can be launched via `ros2 run` or launch file

---

#### Task 2.3: Create Baseline Controller Dockerfile
**Target**: `docker/baseline-controller.Dockerfile`

**Implementation**:
```dockerfile
FROM ros:foxy-ros-base

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
RUN mkdir -p /workspace/src
WORKDIR /workspace

# Copy code
COPY src/baselines /workspace/src/baselines
COPY src/ros_nodes /workspace/src/ros_nodes

# Copy waypoints
COPY FinalProject/waypoints.txt /waypoints/waypoints.txt

# Install Python packages
RUN pip3 install --no-cache-dir \
    numpy \
    scipy

# Source ROS 2
RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc

# Entrypoint
COPY docker/entrypoint_baseline.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "run", "baseline_controller", "baseline_controller_node"]
```

**Entrypoint** (`docker/entrypoint_baseline.sh`):
```bash
#!/bin/bash
set -e

# Source ROS 2
source /opt/ros/foxy/setup.bash

# Wait for CARLA bridge to be ready
echo "Waiting for ROS 2 bridge..."
timeout 30 bash -c 'until ros2 topic list | grep -q "/carla"; do sleep 1; done' || echo "Warning: Bridge topics not found"

# Execute command
exec "$@"
```

---

#### Task 2.4: Update docker-compose.baseline.yml
**Add baseline controller service**:

```yaml
version: '3.8'

services:
  carla-server:
    # ... (existing config)
  
  ros2-bridge:
    # ... (existing config)
  
  baseline-controller:
    build:
      context: .
      dockerfile: docker/baseline-controller.Dockerfile
    image: baseline-controller:foxy
    container_name: baseline-controller
    network_mode: host
    depends_on:
      ros2-bridge:
        condition: service_healthy
    volumes:
      - ./FinalProject/waypoints.txt:/waypoints/waypoints.txt:ro
      - ./logs:/workspace/logs
    environment:
      - ROS_DOMAIN_ID=0
      - PYTHONUNBUFFERED=1
    command: >
      ros2 run baseline_controller baseline_controller_node
      --ros-args
      -p target_velocity:=30.0
      -p control_frequency:=20.0
      --log-level baseline_controller:=info
```

---

### **Week 2: Testing & Evaluation**

#### Task 3.1: Integration Test
**Goal**: Verify complete 3-container stack works

**Test Procedure**:
```bash
# 1. Start all services
docker-compose -f docker-compose.baseline.yml up

# 2. Monitor topics (new terminal)
docker exec -it baseline-controller bash -c "
  source /opt/ros/foxy/setup.bash && \
  ros2 topic hz /carla/ego_vehicle/odometry
"

# Expected: ~20 Hz

# 3. Monitor control commands
docker exec -it baseline-controller bash -c "
  source /opt/ros/foxy/setup.bash && \
  ros2 topic echo /carla/ego_vehicle/vehicle_control_cmd
"

# Expected: Throttle/steer/brake values updating

# 4. Visualize in RViz (optional)
# Run RViz on host machine connected to same ROS_DOMAIN_ID
```

**Success Criteria**:
- ✅ All 3 containers start without errors
- ✅ Vehicle spawns in Town01
- ✅ Topics publish at expected rates (≥20 Hz)
- ✅ Vehicle follows waypoints (visual check)
- ✅ No crashes or exceptions in logs

---

#### Task 3.2: Performance Metrics
**Measure**:

1. **Control Loop Frequency**:
   ```bash
   ros2 topic hz /carla/ego_vehicle/vehicle_control_cmd
   # Target: ≥20 Hz
   ```

2. **End-to-End Latency**:
   ```bash
   # Measure timestamp difference between:
   # - Sensor data published by bridge
   # - Control command received by CARLA
   # Target: <50 ms
   ```

3. **CPU Usage**:
   ```bash
   docker stats --no-stream
   # Monitor CPU% for each container
   ```

4. **Waypoint Following Accuracy**:
   - Record lateral deviation from path
   - Target: Mean <0.5m, Max <1.0m

**Document results** in `docs/day-22/baseline/BASELINE_PERFORMANCE.md`

---

#### Task 3.3: Comparison vs Classical (module_7.py)
**Goal**: Validate that ROS 2 wrapper doesn't degrade performance

**Metrics**:
| Metric | module_7.py (old) | ROS 2 Baseline | Δ |
|--------|-------------------|----------------|---|
| Waypoint following accuracy | ? | ? | ? |
| Average speed | ? | ? | ? |
| Control latency | ? | ? | ? |
| Collisions/km | ? | ? | ? |

**Action**: Run both implementations on same route, compare metrics

---

## Week 3: Evaluation & Decision Point

### Baseline Evaluation Complete ✅

**Deliverables**:
- [ ] PID+Pure Pursuit controller extracted and modernized
- [ ] ROS 2 node implemented and tested
- [ ] Docker 3-container stack running
- [ ] Performance metrics documented
- [ ] Comparison with classical baseline

### **Decision Point: Native ROS 2?**

**Trigger evaluation if**:
- Bridge latency >50ms
- Control loop <20Hz
- CPU overhead >10% for bridge

**If performance is acceptable**:
- ✅ Keep external bridge architecture
- ✅ Proceed to Phase 3 (DRL integration)
- 📝 Document as "validated architecture"

**If performance is NOT acceptable**:
- 🔄 Activate Track 2 (Native ROS 2)
- 🏗️ Build CARLA from source with --ros2
- 📦 Create custom Docker image
- 🧪 Re-evaluate performance

---

## Risk Mitigation

### Risk 1: Bridge Latency Too High
**Probability**: Low (bridge is proven solution)  
**Impact**: Medium (may need native ROS 2)  
**Mitigation**:
- Have Track 2 (native ROS 2) as backup plan
- Monitor latency from Day 1
- Early warning if approaching 50ms threshold

### Risk 2: Docker Build Issues
**Probability**: Medium (complex multi-stage build)  
**Impact**: Low (can debug iteratively)  
**Mitigation**:
- Test Dockerfile changes in isolation
- Use Docker layer caching
- Document each build step with verification

### Risk 3: Waypoint Loading/Format Issues
**Probability**: Low (waypoints.txt is simple format)  
**Impact**: Low (easy to fix)  
**Mitigation**:
- Add validation when loading waypoints
- Unit test waypoint parser
- Visualize loaded waypoints in RViz

### Risk 4: CARLA Python API Version Mismatch
**Probability**: Medium (Docker image has specific version)  
**Impact**: High (bridge won't work)  
**Mitigation**:
- ✅ Already discovered: Use `/workspace/PythonAPI`
- Build bridge with exact CARLA 0.9.16 .egg file
- Test import in container before full build

---

## Success Criteria for Phase 2

### Minimum Viable Product (MVP)

**Week 2 Completion**:
- ✅ Vehicle drives autonomously in Town01
- ✅ Follows predefined waypoints
- ✅ Control frequency ≥15 Hz (acceptable for baseline)
- ✅ No crashes during 5-minute run
- ✅ All containers healthy and communicating

### Optimal Product

**Week 3 Completion**:
- ✅ Control frequency ≥20 Hz
- ✅ End-to-end latency <50 ms
- ✅ Waypoint following error <0.5m mean
- ✅ Performance matches classical baseline
- ✅ Documented and reproducible

### Stretch Goals (If Time Permits)

**Week 3+**:
- ⭐ Native ROS 2 implementation tested
- ⭐ Performance comparison (bridge vs native)
- ⭐ RViz visualization working
- ⭐ Multiple scenarios (different traffic densities)

---

## Next Immediate Actions

### **TODAY**:
1. ✅ Create this revised plan document
2. ⏭️ Fix Dockerfile CARLA path (Task 1.1)
3. ⏭️ Rebuild ros2-carla-bridge image
4. ⏭️ Test bridge connectivity (Task 1.2)

### **This Week**:
1. Extract PID+Pure Pursuit code (Task 2.1)
2. Implement ROS 2 node (Task 2.2)
3. Create baseline Dockerfile (Task 2.3)
4. Update docker-compose.yml (Task 2.4)

### **Next Week**:
1. Integration testing (Task 3.1)
2. Performance measurements (Task 3.2)
3. Comparison with classical (Task 3.3)
4. **Decision: Native ROS 2 or proceed to Phase 3**

---

## Appendix: Key Lessons Learned

### From Native ROS 2 Investigation:

1. **Always verify documentation claims**: Release notes claimed feature, but Docker images didn't include it
2. **Source code is truth**: Found implementation in GitHub when docs were unclear
3. **Testing must be exhaustive**: We tested for ROS installation, not for --ros2 flag
4. **Community bridges exist for a reason**: External bridge is well-supported despite latency trade-off

### From User Feedback:

1. **Challenge assumptions**: User correctly questioned our Phase 1 conclusion
2. **Deep investigation pays off**: Found source code, examples, build instructions
3. **Document thoroughly**: This investigation will save time for future work
4. **Backup plans are essential**: Having dual-track approach reduces risk

---

**Document Status**: 🟢 APPROVED for implementation  
**Owner**: Development team  
**Review Date**: After Week 2 (Performance evaluation)
