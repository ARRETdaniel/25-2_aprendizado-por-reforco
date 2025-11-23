# Baseline Controller Implementation Plan
## PID + Pure Pursuit for ROS 2 Humble + CARLA 0.9.16

**Date**: November 22, 2025
**Status**: 🚀 READY TO IMPLEMENT
**Prerequisites**: ✅ ROS 2 Bridge Docker image built and verified

---

## 1. Executive Summary

Based on comprehensive analysis of the legacy controller (`controller2d.py`, `module_7.py`) and official CARLA ROS bridge documentation, this plan details the implementation of a modular PID + Pure Pursuit baseline controller for autonomous vehicle control in CARLA via ROS 2.

**Key Decisions**:
- ✅ Use **external ROS bridge** (not native ROS 2) - officially supported
- ✅ Implement as **ROS 2 Humble** node (Python 3.10+)
- ✅ **PID for longitudinal** control (speed tracking)
- ✅ **Pure Pursuit for lateral** control (path following)
- ✅ **Modular architecture** - same interface can be reused by TD3 agent

---

## 2. Controller Analysis

### 2.1 PID Controller (Longitudinal Control)

**Source**: `controller2d.py` lines 114-184

**Purpose**: Track desired speed from waypoints

**Algorithm**:
```python
v_error = v_desired - v_current
v_error_integral += v_error * dt
v_error_derivative = (v_error - v_error_prev) / dt

throttle = kp * v_error + ki * v_error_integral + kd * v_error_derivative
```

**Parameters** (from legacy code):
```python
kp = 0.50
ki = 0.30
kd = 0.13
integrator_min = 0.0
integrator_max = 10.0
```

**Inputs**:
- Current speed: `v_current` (from `/carla/ego_vehicle/vehicle_status`)
- Desired speed: `v_desired` (from waypoint closest to vehicle)
- Delta time: `dt` (from timestamps)

**Outputs**:
- Throttle command: `[0.0, 1.0]`
- Brake command: `[0.0, 1.0]` (if throttle < 0)

### 2.2 Pure Pursuit Controller (Lateral Control)

**Source**: `controller2d.py` lines 186-237

**Purpose**: Follow waypoint path by computing steering angle

**Algorithm**:
1. Find lookahead point on path (distance = 2.0m from vehicle)
2. Compute crosstrack error (distance from path)
3. Compute heading error (angle difference to path)
4. Compute steering command:
   ```python
   steer = heading_error + arctan(kp_heading * crosstrack_sign * crosstrack_error / (v + k_speed_crosstrack))
   ```

**Parameters** (from legacy code):
```python
lookahead_distance = 2.0  # meters
kp_heading = 8.00
k_speed_crosstrack = 0.00
cross_track_deadband = 0.01  # to reduce oscillations
```

**Inputs**:
- Current pose: `(x, y, yaw)` (from `/carla/ego_vehicle/odometry`)
- Waypoints: List of `[(x, y, speed), ...]`
- Current speed: `v` (for speed-dependent steering)

**Outputs**:
- Steering command: `[-1.0, 1.0]` (normalized from radians)

---

## 3. ROS 2 Integration Architecture

### 3.1 Topic Interfaces

#### Subscriptions:
| Topic | Message Type | Frequency | Usage |
|-------|-------------|-----------|--------|
| `/carla/ego_vehicle/odometry` | `nav_msgs/Odometry` | 20 Hz | Vehicle pose (x, y, yaw) |
| `/carla/ego_vehicle/vehicle_status` | `carla_msgs/CarlaEgoVehicleStatus` | 20 Hz | Current speed, acceleration |

#### Publications:
| Topic | Message Type | Frequency | Usage |
|-------|-------------|-----------|--------|
| `/carla/ego_vehicle/vehicle_control_cmd` | `carla_msgs/CarlaEgoVehicleControl` | 20 Hz | Throttle, steer, brake commands |

### 3.2 Parameters (ROS 2 YAML config)

```yaml
baseline_controller:
  ros__parameters:
    # Control loop
    control_frequency: 20.0  # Hz
    
    # Waypoints
    waypoint_file: "/workspace/config/waypoints/town01.txt"
    
    # PID parameters (longitudinal)
    pid:
      kp: 0.50
      ki: 0.30
      kd: 0.13
      integrator_min: 0.0
      integrator_max: 10.0
    
    # Pure Pursuit parameters (lateral)
    pure_pursuit:
      lookahead_distance: 2.0  # meters
      kp_heading: 8.00
      k_speed_crosstrack: 0.00
      cross_track_deadband: 0.01  # meters
      
    # Safety limits
    max_throttle: 0.75
    max_brake: 1.0
    max_steer: 1.0
```

---

## 4. File Structure

```
av_td3_system/
├── src/
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── pid_controller.py          # NEW: PID implementation
│   │   ├── pure_pursuit_controller.py # NEW: Pure Pursuit implementation
│   │   └── utils.py                    # NEW: Helper functions (waypoint loading, etc.)
│   ├── ros_nodes/
│   │   ├── __init__.py
│   │   └── baseline_controller_node.py # NEW: ROS 2 node
│   └── common/
│       ├── __init__.py
│       └── waypoint_loader.py         # NEW: Load waypoints from file
├── config/
│   ├── waypoints/
│   │   └── town01.txt                 # Waypoint file (from legacy)
│   └── baseline_controller.yaml       # ROS 2 parameters
├── launch/
│   └── baseline_controller.launch.py  # ROS 2 launch file
└── docker/
    └── baseline-controller.Dockerfile # Docker image for baseline
```

---

## 5. Implementation Steps

### Step 1: Extract and Modernize Controllers (2-3 hours)

**File**: `src/baselines/pid_controller.py`

```python
"""
PID Controller for longitudinal (speed) control.
Based on controller2d.py from TCC project.
Updated for CARLA 0.9.16 and Python 3.10+.
"""

from typing import Optional
import numpy as np


class PIDController:
    """
    Proportional-Integral-Derivative controller for speed tracking.
    
    Args:
        kp: Proportional gain
        ki: Integral gain  
        kd: Derivative gain
        integrator_min: Minimum integrator value (anti-windup)
        integrator_max: Maximum integrator value (anti-windup)
    """
    
    def __init__(
        self,
        kp: float = 0.50,
        ki: float = 0.30,
        kd: float = 0.13,
        integrator_min: float = 0.0,
        integrator_max: float = 10.0
    ):
        # ... implementation
```

**File**: `src/baselines/pure_pursuit_controller.py`

```python
"""
Pure Pursuit Controller for lateral (steering) control.
Based on controller2d.py from TCC project.
Updated for CARLA 0.9.16 and Python 3.10+.
"""

from typing import List, Tuple
import numpy as np


class PurePursuitController:
    """
    Pure Pursuit algorithm for path following.
    
    Args:
        lookahead_distance: Distance ahead to look for target point (meters)
        kp_heading: Proportional gain for heading error
        k_speed_crosstrack: Speed-dependent crosstrack adjustment
        cross_track_deadband: Deadband to reduce oscillations (meters)
    """
    
    def __init__(
        self,
        lookahead_distance: float = 2.0,
        kp_heading: float = 8.00,
        k_speed_crosstrack: float = 0.00,
        cross_track_deadband: float = 0.01
    ):
        # ... implementation
```

**File**: `src/common/waypoint_loader.py`

```python
"""
Waypoint loading utilities.
Reads waypoint files from TCC format: x, y, speed (one per line).
"""

from typing import List, Tuple
import numpy as np


def load_waypoints(filepath: str) -> List[Tuple[float, float, float]]:
    """
    Load waypoints from file.
    
    Format (space-separated):
        x1 y1 speed1
        x2 y2 speed2
        ...
    
    Args:
        filepath: Path to waypoint file
        
    Returns:
        List of (x, y, speed) tuples
    """
    waypoints = []
    with open(filepath, 'r') as f:
        for line in f:
            # ... implementation
```

### Step 2: Create ROS 2 Node (3-4 hours)

**File**: `src/ros_nodes/baseline_controller_node.py`

```python
"""
ROS 2 Node for Baseline Controller (PID + Pure Pursuit).
Subscribes to CARLA topics and publishes vehicle control commands.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleStatus, CarlaEgoVehicleControl

from baselines.pid_controller import PIDController
from baselines.pure_pursuit_controller import PurePursuitController
from common.waypoint_loader import load_waypoints


class BaselineControllerNode(Node):
    """ROS 2 node for baseline vehicle controller."""
    
    def __init__(self):
        super().__init__('baseline_controller')
        
        # Declare parameters
        self.declare_parameters()
        
        # Initialize controllers
        self.pid = PIDController(...)
        self.pure_pursuit = PurePursuitController(...)
        
        # Load waypoints
        waypoint_file = self.get_parameter('waypoint_file').value
        self.waypoints = load_waypoints(waypoint_file)
        
        # Create subscriptions
        self.odom_sub = self.create_subscription(...)
        self.status_sub = self.create_subscription(...)
        
        # Create publisher
        self.control_pub = self.create_publisher(...)
        
        # Control loop timer (20 Hz)
        control_freq = self.get_parameter('control_frequency').value
        self.timer = self.create_timer(1.0 / control_freq, self.control_loop)
        
    def control_loop(self):
        """Main control loop called at control_frequency Hz."""
        # ... implementation
```

### Step 3: Create Configuration Files (1 hour)

**File**: `config/baseline_controller.yaml`

**File**: `launch/baseline_controller.launch.py`

### Step 4: Create Docker Image (2 hours)

**File**: `docker/baseline-controller.Dockerfile`

```dockerfile
FROM ros:humble-ros-base

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Install ROS 2 packages
RUN apt-get update && apt-get install -y \
    ros-humble-nav-msgs \
    ros-humble-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace
COPY src/ /workspace/src/
COPY config/ /workspace/config/
COPY launch/ /workspace/launch/

# Build ROS 2 package (if using ament_python)
# For now, we'll use direct Python execution

# Entrypoint
COPY docker/baseline_entrypoint.sh /
RUN chmod +x /baseline_entrypoint.sh
ENTRYPOINT ["/baseline_entrypoint.sh"]
CMD ["bash"]
```

### Step 5: Update docker-compose.yml (1 hour)

Add baseline controller service to `docker-compose.baseline.yml`:

```yaml
  baseline-controller:
    image: baseline-controller:humble
    container_name: baseline-controller
    network_mode: host
    environment:
      - ROS_DOMAIN_ID=0
    depends_on:
      ros2-bridge:
        condition: service_healthy
    volumes:
      - ./config:/workspace/config:ro
      - ./results:/workspace/results:rw
    command: >
      bash -c "source /opt/ros/humble/setup.bash &&
               ros2 run baseline_controller baseline_controller_node
               --ros-args --params-file /workspace/config/baseline_controller.yaml"
```

---

## 6. Testing Strategy

### 6.1 Unit Tests
- Test PID controller with known inputs/outputs
- Test Pure Pursuit with simple waypoint paths
- Test waypoint loader with sample files

### 6.2 Integration Tests
1. Launch CARLA server
2. Launch ROS 2 bridge with ego vehicle
3. Launch baseline controller
4. Verify vehicle follows waypoints
5. Record metrics:
   - Lap completion time
   - Average speed
   - Maximum lateral error
   - Maximum longitudinal error
   - Control smoothness (jerk)

### 6.3 Comparison with Legacy
- Run legacy `module_7.py` on same waypoints
- Compare performance metrics
- Document any differences

---

## 7. Success Criteria

✅ **Phase 2.1**: ROS 2 Bridge verified (COMPLETED)
- [x] Docker image built successfully
- [x] CARLA Python API imports correctly
- [x] ROS 2 packages available
- [x] Launch files discovered

✅ **Phase 2.2**: Full stack tested
- [ ] CARLA server + ROS 2 bridge communication verified
- [ ] Topics publishing at expected rates
- [ ] Control commands accepted by CARLA

✅ **Phase 2.3**: Controllers modernized
- [ ] PID controller implemented with type hints
- [ ] Pure Pursuit controller implemented with type hints
- [ ] Code passes mypy type checking
- [ ] Comprehensive docstrings added

✅ **Phase 2.4**: ROS 2 node created
- [ ] Node subscribes to odometry and status
- [ ] Node publishes control commands
- [ ] Parameters loaded from YAML
- [ ] 20 Hz control loop verified

✅ **Phase 2.5**: Docker image built
- [ ] Dockerfile created
- [ ] Image builds successfully
- [ ] Image < 1GB size

✅ **Phase 2.6**: Integration test passed
- [ ] Vehicle completes full lap
- [ ] No crashes or collisions
- [ ] Performance comparable to legacy

✅ **Phase 2.7**: Documentation complete
- [ ] Performance metrics documented
- [ ] Comparison report created
- [ ] Decision on native ROS 2 investigation

---

## 8. Timeline Estimate

| Phase | Description | Estimated Time | Status |
|-------|-------------|----------------|--------|
| 2.1 | Verify ROS 2 Bridge | 1 hour | ✅ DONE |
| 2.2 | Test full stack | 2 hours | 🔄 NEXT |
| 2.3 | Modernize controllers | 3 hours | ⏳ Pending |
| 2.4 | Create ROS 2 node | 4 hours | ⏳ Pending |
| 2.5 | Docker image | 2 hours | ⏳ Pending |
| 2.6 | Integration test | 3 hours | ⏳ Pending |
| 2.7 | Documentation | 2 hours | ⏳ Pending |
| **TOTAL** | | **17 hours** (~2-3 days) | |

---

## 9. Next Immediate Action

**NOW**: Execute Phase 2.2 - Test Full ROS 2 Bridge Stack

```bash
# Terminal 1: Start CARLA server
docker run --rm --gpus all --net=host carlasim/carla:0.9.16 \
  bash CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000

# Terminal 2: Start ROS 2 bridge with example ego vehicle
docker run --rm --net=host ros2-carla-bridge:humble-v4 \
  bash -c "source /ros_entrypoint.sh bash -c 'ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py host:=localhost port:=2000 synchronous_mode:=true fixed_delta_seconds:=0.05'"

# Terminal 3: Monitor topics
docker run --rm --net=host ros2-carla-bridge:humble-v4 \
  bash -c "source /ros_entrypoint.sh bash -c 'ros2 topic list'"

# Terminal 4: Echo odometry
docker run --rm --net=host ros2-carla-bridge:humble-v4 \
  bash -c "source /ros_entrypoint.sh bash -c 'ros2 topic echo /carla/ego_vehicle/odometry --once'"
```

---

## 10. References

- CARLA ROS Bridge Docs: https://carla.readthedocs.io/projects/ros-bridge/
- Legacy controller: `related_works/exemple-carlaClient-openCV-YOLO-town01-waypoints/controller2d.py`
- ROS 2 Humble Docs: https://docs.ros.org/en/humble/
- Pure Pursuit Algorithm: https://www.mathworks.com/help/robotics/ug/pure-pursuit-controller.html
- PID Control: https://en.wikipedia.org/wiki/PID_controller
