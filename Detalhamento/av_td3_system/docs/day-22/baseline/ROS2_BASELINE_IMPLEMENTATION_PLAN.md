# ROS 2 Baseline Controller Implementation Plan
## CARLA 0.9.16 Native Integration Analysis

**Document Version:** 1.0  
**Date:** November 22, 2025  
**Author:** GitHub Copilot Agent  
**Project:** End-to-End Visual Autonomous Navigation with TD3

---

## Executive Summary

This document provides a systematic analysis and implementation plan for integrating a PID + Pure Pursuit baseline controller with CARLA 0.9.16's **native ROS 2 support**. The key insight from CARLA 0.9.16 release notes is:

> **"CARLA 0.9.16 ships with native ROS2 integration, opening the door for: Out-of-the-box compatibility with modern robotic stacks, DDS-based message passing and time synchronization, Integration examples included. You can now connect CARLA directly to ROS2 Foxy, Galactic, Humble and more — with sensor streams and ego control - all without the latency of a bridge tool."**

This **eliminates the need for the external ROS bridge** and provides a simpler, lower-latency architecture.

---

## Table of Contents

1. [Critical Architecture Decision](#1-critical-architecture-decision)
2. [CARLA 0.9.16 Native ROS 2 Features](#2-carla-0916-native-ros-2-features)
3. [Current System Analysis](#3-current-system-analysis)
4. [Legacy Controller Analysis](#4-legacy-controller-analysis)
5. [Proposed Architecture](#5-proposed-architecture)
6. [Implementation Phases](#6-implementation-phases)
7. [Technical Specifications](#7-technical-specifications)
8. [Migration Path for DRL Agent](#8-migration-path-for-drl-agent)
9. [Risk Assessment](#9-risk-assessment)
10. [Next Steps](#10-next-steps)

---

## 1. Critical Architecture Decision

### 1.1 Three Possible Approaches

| Approach | Description | Pros | Cons | Recommendation |
|----------|-------------|------|------|----------------|
| **A: Keep Direct Python API** | Continue using `carla_env.py` with Python API | ✅ Working system<br>✅ No migration needed<br>✅ Full API access | ❌ Not modular<br>❌ Can't swap controllers<br>❌ Not using ROS 2 | ❌ **Not Recommended** |
| **B: External ROS Bridge** | Use `carla-ros-bridge` package | ✅ Established patterns<br>✅ Good documentation | ❌ Additional latency<br>❌ Extra dependency<br>❌ Not using native feature | ❌ **Not Recommended** |
| **C: Native ROS 2 Integration** | Use CARLA 0.9.16 native support | ✅ **Zero bridge latency**<br>✅ **Native DDS**<br>✅ **Out-of-the-box**<br>✅ Modular architecture | ⚠️ Documentation may be sparse<br>⚠️ Need to learn API | ✅ **RECOMMENDED** |

### 1.2 Decision Rationale

**We will use Approach C: Native ROS 2 Integration** for the following reasons:

1. **Performance**: Eliminates bridge latency (critical for 20 Hz control loop)
2. **Simplicity**: No external dependencies beyond CARLA itself
3. **Modularity**: Clean separation of simulation, control, and agent logic
4. **Future-proof**: Aligns with CARLA's architectural direction
5. **Paper contribution**: Demonstrates modern autonomous vehicle stack architecture

### 1.3 Architecture Philosophy

The implementation will follow a **hybrid approach**:

- **Baseline Controller (PID+Pure Pursuit)**: Pure ROS 2 nodes communicating via native CARLA topics
- **DRL Agent (TD3)**: Initially keep direct Python API, then migrate incrementally to ROS 2 topics for fair comparison

This allows us to:
- Deliver baseline results quickly using proven ROS 2 patterns
- Maintain working DRL system during migration
- Validate ROS 2 integration before risking DRL disruption
- Compare performance metrics fairly (both systems using same infrastructure)

---

## 2. CARLA 0.9.16 Native ROS 2 Features

### 2.1 Release Highlights

From official release notes (https://carla.org/2025/09/16/release-0.9.16/):

```
🧭 Native ROS2 Support

CARLA 0.9.16 ships with native ROS2 integration, opening the door for:
• Out-of-the-box compatibility with modern robotic stacks
• DDS-based message passing and time synchronization
• Integration examples included

You can now connect CARLA directly to ROS2 Foxy, Galactic, Humble and more —
with sensor streams and ego control - all without the latency of a bridge tool.

Autonomy teams, rejoice.
```

### 2.2 Expected Native Topics (Based on Bridge Conventions)

**Note**: Official documentation for native ROS 2 API is pending. Based on release notes and standard practices, we expect:

#### Control Topics (Publishers from ROS 2 → CARLA)
```
/carla/ego_vehicle/vehicle_control_cmd
  Type: carla_msgs/CarlaEgoVehicleControl
  Fields: {throttle, steer, brake, hand_brake, reverse, gear}
```

#### State Topics (Subscribers from CARLA → ROS 2)
```
/carla/ego_vehicle/vehicle_status
  Type: carla_msgs/CarlaEgoVehicleStatus
  Fields: {velocity, acceleration, orientation, control}

/carla/ego_vehicle/odometry
  Type: nav_msgs/Odometry
  Fields: {pose, twist, covariance}

/carla/ego_vehicle/imu
  Type: sensor_msgs/Imu
  Fields: {orientation, angular_velocity, linear_acceleration}
```

#### Sensor Topics
```
/carla/ego_vehicle/camera/rgb/image_raw
  Type: sensor_msgs/Image
  Fields: {header, height, width, encoding, data}

/carla/ego_vehicle/collision
  Type: carla_msgs/CarlaCollisionEvent
  
/carla/ego_vehicle/lane_invasion
  Type: carla_msgs/CarlaLaneInvasionEvent
```

### 2.3 Synchronous Mode with Native ROS 2

Expected synchronization mechanism (to be verified):

```python
# CARLA server publishes clock
/clock
  Type: rosgraph_msgs/Clock
  
# Clients subscribe to clock for time synchronization
# DDS ensures deterministic message ordering
```

### 2.4 Documentation Sources

**Primary**: 
- Release notes: https://carla.org/2025/09/16/release-0.9.16/
- **TODO**: Check for native ROS 2 examples in CARLA 0.9.16 installation
- **TODO**: Search CARLA GitHub repository for ROS 2 integration code

**Fallback** (if native docs incomplete):
- External bridge patterns: https://carla.readthedocs.io/projects/ros-bridge/en/latest/
- Adapt bridge message types to native implementation

---

## 3. Current System Analysis

### 3.1 DRL System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    train_td3.py                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              TD3 Agent (agents/td3_agent.py)        │   │
│  │  • Actor Network (CNN → MLP)                        │   │
│  │  • Twin Critics                                      │   │
│  │  • Replay Buffer                                     │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │ select_action() / train()             │
│                     ▼                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │      Gymnasium Environment (environment/carla_env.py)│   │
│  │  • reset() → initial state                          │   │
│  │  • step(action) → next_state, reward, done          │   │
│  │  • State construction: [CNN features, v, d, φ]      │   │
│  └──────────────────┬──────────────────────────────────┘   │
│                     │ CARLA Python API                      │
│                     ▼                                        │
└─────────────────────┼──────────────────────────────────────┘
                      │
        ┌─────────────▼──────────────┐
        │   CARLA 0.9.16 Server      │
        │   • Town01 map             │
        │   • Synchronous mode       │
        │   • Physics @ 20 Hz        │
        │   • Sensors: Camera, IMU   │
        └────────────────────────────┘
```

**Key Characteristics**:
- **Direct coupling**: `carla_env.py` instantiates CARLA client within Gymnasium interface
- **State construction**: Frame stacking, CNN extraction, concatenation all in `carla_env.py`
- **Monolithic**: Single process handles simulation, perception, and learning
- **Not modular**: Cannot easily swap DRL agent for classical controller

### 3.2 Configuration Files

#### `config/carla_config.yaml` (Relevant Sections)

```yaml
world:
  map: 'Town01'
  synchronous_mode: true
  fixed_delta_seconds: 0.05  # 20 Hz control loop
  
route:
  use_dynamic_generation: true
  waypoints_file: '/workspace/config/waypoints.txt'
  sampling_resolution: 2.0
  lookahead_distance: 50.0
  
ego_vehicle:
  blueprint: 'vehicle.tesla.model3'
  spawn_point_index: 1
  target_speed: 30.0  # km/h (8.33 m/s)
  
sensors:
  camera:
    enabled: true
    width: 800
    height: 600
    fov: 90
    x: 2.0
    y: 0.0
    z: 1.4
```

**Critical Parameters**:
- `fixed_delta_seconds: 0.05` → 20 Hz update rate (must match ROS 2 node rates)
- `waypoints_file` → Shared between DRL and baseline (same route for fair comparison)
- `target_speed: 30.0 km/h` → Reference speed for PID controller

---

## 4. Legacy Controller Analysis

### 4.1 Code Location

```
/related_works/exemple-carlaClient-openCV-YOLO-town01-waypoints/
├── controller2d.py         # PID + Pure Pursuit implementation
├── module_7.py             # Main integration with old CARLA API
├── cutils.py               # Variable storage helper
└── waypoints.txt           # Route definition (similar format to our config/waypoints.txt)
```

### 4.2 Controller Architecture

#### 4.2.1 PID Longitudinal Controller

**File**: `controller2d.py` lines 114-127

```python
# PID Gains
self.vars.create_var('kp', 0.50)
self.vars.create_var('ki', 0.30)
self.vars.create_var('kd', 0.13)
self.vars.create_var('integrator_min', 0.0)
self.vars.create_var('integrator_max', 10.0)

# PID computation
self.vars.v_error = v_desired - v
self.vars.v_error_integral += self.vars.v_error * dt
v_error_rate_of_change = (self.vars.v_error - self.vars.v_error_prev) / dt

# Anti-windup
self.vars.v_error_integral = np.clip(self.vars.v_error_integral, 
                                      self.vars.integrator_min, 
                                      self.vars.integrator_max)

# Control output
throttle_output = kp * v_error + ki * v_error_integral + kd * v_error_rate_of_change
```

**Key Features**:
- **P gain (0.50)**: Proportional response to speed error
- **I gain (0.30)**: Eliminates steady-state error, handles inclines
- **D gain (0.13)**: Damping to reduce oscillations
- **Anti-windup**: Clamps integrator to [0.0, 10.0] to prevent saturation

#### 4.2.2 Pure Pursuit Lateral Controller

**File**: `controller2d.py` lines 128-178

```python
# Pure Pursuit gains
self.vars.create_var('kp_heading', 8.00)
self.vars.create_var('k_speed_crosstrack', 0.00)
self.vars.create_var('cross_track_deadband', 0.01)

# Lookahead distance
self._lookahead_distance = 2.0  # meters

# Find lookahead waypoint
ce_idx = self.get_lookahead_index(self._lookahead_distance)
crosstrack_vector = np.array([
    waypoints[ce_idx][0] - x - lookahead_distance * np.cos(yaw),
    waypoints[ce_idx][1] - y - lookahead_distance * np.sin(yaw)
])
crosstrack_error = np.linalg.norm(crosstrack_vector)

# Deadband to reduce oscillations
if crosstrack_error < self.vars.cross_track_deadband:
    crosstrack_error = 0.0

# Compute crosstrack sign
crosstrack_heading = np.arctan2(crosstrack_vector[1], crosstrack_vector[0])
crosstrack_heading_error = normalize_angle(crosstrack_heading - yaw)
crosstrack_sign = np.sign(crosstrack_heading_error)

# Compute heading error relative to path
trajectory_heading = np.arctan2(waypoints[ce_idx+1][1] - waypoints[ce_idx][1],
                                 waypoints[ce_idx+1][0] - waypoints[ce_idx][0])
heading_error = normalize_angle(trajectory_heading - yaw)

# Stanley-like steering law
steer_output = heading_error + np.arctan(kp_heading * crosstrack_sign * crosstrack_error / 
                                          (v + k_speed_crosstrack))
```

**Key Features**:
- **Lookahead distance (2.0m)**: Fixed distance ahead on path
- **Heading error**: Alignment with path tangent
- **Crosstrack error**: Lateral deviation from path
- **Stanley component**: `arctan(kp * error / speed)` for crosstrack correction
- **Deadband (0.01m)**: Prevents chattering near path

#### 4.2.3 Waypoint Structure

**Format** (from waypoints.txt):
```
# x, y, v_desired
132.7, 195.4, 5.6  # Position in meters, speed in m/s
135.1, 195.4, 5.6
...
```

**Compatibility Check**:
- ✅ Our `config/waypoints.txt` uses same format
- ✅ `carla_config.yaml` specifies `waypoints_file: '/workspace/config/waypoints.txt'`
- ✅ Can reuse existing waypoint loading logic

### 4.3 Integration with Old CARLA API

**File**: `module_7.py` (excerpts)

```python
# Old CARLA API (pre-0.9.x)
from carla.client import make_carla_client, VehicleControl
from carla.settings import CarlaSettings

# Connection
with make_carla_client('localhost', 2000) as client:
    settings = make_carla_settings(args)
    client.load_settings(settings)
    
    # Control loop
    control = VehicleControl()
    throttle, steer, brake = controller.get_commands()
    control.throttle = throttle
    control.steer = steer
    control.brake = brake
    client.send_control(control)
    
    # State retrieval
    measurements, sensor_data = client.read_data()
    player_measurements = measurements.player_measurements
    x = player_measurements.transform.location.x
    y = player_measurements.transform.location.y
    yaw = math.radians(player_measurements.transform.rotation.yaw)
    speed = player_measurements.forward_speed
```

**Modernization Required**:
1. **API migration**: `carla.client.Client` (new) vs `make_carla_client` (old)
2. **Synchronous mode**: Explicit `world.tick()` vs implicit in old API
3. **Sensor handling**: Callback-based vs polling
4. **Transform access**: `actor.get_transform()` vs measurements

---

## 5. Proposed Architecture

### 5.1 System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         ROS 2 Workspace                               │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              baseline_controller_node.py                     │   │
│  │  ┌────────────────────┐  ┌──────────────────────────────┐   │   │
│  │  │  PID Controller    │  │  Pure Pursuit Controller     │   │   │
│  │  │  • Speed tracking  │  │  • Waypoint following        │   │   │
│  │  │  • Anti-windup     │  │  • Crosstrack correction     │   │   │
│  │  └────────┬───────────┘  └──────────┬───────────────────┘   │   │
│  │           │                          │                        │   │
│  │           └──────────┬───────────────┘                        │   │
│  │                      ▼                                        │   │
│  │          CarlaEgoVehicleControl msg                           │   │
│  │          {throttle, steer, brake}                             │   │
│  └──────────────────────┼─────────────────────────────────────┘   │
│                         │ publish                                   │
│                         ▼                                           │
│         /carla/ego_vehicle/vehicle_control_cmd (ROS 2 topic)        │
│                         │                                           │
│                         │ DDS (native)                              │
│                         ▼                                           │
└─────────────────────────┼─────────────────────────────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │   CARLA 0.9.16 Server               │
        │   (Native ROS 2 Integration)        │
        │                                     │
        │   ┌─────────────────────────────┐  │
        │   │  DDS Publisher/Subscriber   │  │
        │   │  • Control topic listener   │  │
        │   │  • State topic publisher    │  │
        │   │  • Sensor topic publisher   │  │
        │   │  • Clock publisher          │  │
        │   └─────────────────────────────┘  │
        │                                     │
        │   ┌─────────────────────────────┐  │
        │   │  Ego Vehicle Actor          │  │
        │   │  • Apply control commands   │  │
        │   │  • Update physics @ 20 Hz   │  │
        │   └─────────────────────────────┘  │
        └─────────────────┬───────────────────┘
                          │ publish
                          ▼
         /carla/ego_vehicle/vehicle_status (ROS 2 topic)
         /carla/ego_vehicle/odometry
         /carla/ego_vehicle/imu
                          │
                          │ subscribe
                          ▼
        ┌──────────────────────────────────────┐
        │   baseline_controller_node.py        │
        │   • Update vehicle state             │
        │   • Find closest/lookahead waypoint  │
        │   • Compute control                  │
        └──────────────────────────────────────┘
```

### 5.2 ROS 2 Node Design

#### 5.2.1 `baseline_controller_node.py`

**Location**: `av_td3_system/src/ros_nodes/baseline_controller_node.py`

**Responsibilities**:
1. Subscribe to CARLA native topics (vehicle state, odometry)
2. Load waypoints from `config/waypoints.txt`
3. Compute PID + Pure Pursuit control
4. Publish control commands to CARLA native topic
5. Log metrics (speed error, crosstrack error, steering, throttle)

**ROS 2 Interfaces**:

```python
import rclpy
from rclpy.node import Node
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

class BaselineControllerNode(Node):
    def __init__(self):
        super().__init__('baseline_controller_node')
        
        # Publishers
        self.control_pub = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/ego_vehicle/vehicle_control_cmd',
            10
        )
        
        # Subscribers
        self.status_sub = self.create_subscription(
            CarlaEgoVehicleStatus,
            '/carla/ego_vehicle/vehicle_status',
            self.status_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/carla/ego_vehicle/odometry',
            self.odometry_callback,
            10
        )
        
        # Controller components
        from baselines.pid_pure_pursuit import PIDController, PurePursuitController
        self.pid_controller = PIDController(kp=0.50, ki=0.30, kd=0.13)
        self.lateral_controller = PurePursuitController(
            lookahead_distance=2.0,
            kp_heading=8.00
        )
        
        # Waypoints
        self.waypoints = self.load_waypoints('/workspace/config/waypoints.txt')
        
        # Control loop timer (20 Hz to match CARLA)
        self.timer = self.create_timer(0.05, self.control_loop)
```

**Control Loop**:

```python
def control_loop(self):
    """Main control loop - runs at 20 Hz"""
    
    # Check if state is initialized
    if not hasattr(self, 'current_state'):
        return
    
    # Find closest and lookahead waypoints
    closest_idx = self.find_closest_waypoint()
    lookahead_idx = self.get_lookahead_index(self.lookahead_distance)
    
    # Longitudinal control (PID)
    v_desired = self.waypoints[closest_idx][2]
    throttle, brake = self.pid_controller.update(
        v_desired=v_desired,
        v_current=self.current_state['speed'],
        dt=0.05
    )
    
    # Lateral control (Pure Pursuit)
    steer = self.lateral_controller.update(
        x=self.current_state['x'],
        y=self.current_state['y'],
        yaw=self.current_state['yaw'],
        speed=self.current_state['speed'],
        waypoints=self.waypoints,
        lookahead_idx=lookahead_idx
    )
    
    # Publish control command
    control_msg = CarlaEgoVehicleControl()
    control_msg.throttle = float(np.clip(throttle, 0.0, 1.0))
    control_msg.steer = float(np.clip(steer, -1.0, 1.0))
    control_msg.brake = float(np.clip(brake, 0.0, 1.0))
    control_msg.hand_brake = False
    control_msg.reverse = False
    control_msg.gear = 1
    
    self.control_pub.publish(control_msg)
    
    # Log metrics
    self.log_metrics(throttle, steer, brake, v_desired)
```

### 5.3 Baseline Controller Module

#### 5.3.1 `src/baselines/pid_pure_pursuit.py`

**Extract and modernize** from `controller2d.py`:

```python
"""
PID + Pure Pursuit Baseline Controller
Modernized from legacy CARLA implementation for CARLA 0.9.16
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class PIDGains:
    """PID controller gains"""
    kp: float = 0.50
    ki: float = 0.30
    kd: float = 0.13
    integrator_min: float = 0.0
    integrator_max: float = 10.0

class PIDController:
    """Longitudinal speed controller using PID"""
    
    def __init__(self, kp=0.50, ki=0.30, kd=0.13):
        self.gains = PIDGains(kp=kp, ki=ki, kd=kd)
        self.v_error_integral = 0.0
        self.v_error_prev = 0.0
        
    def update(self, v_desired: float, v_current: float, dt: float) -> Tuple[float, float]:
        """
        Compute throttle/brake from speed error
        
        Args:
            v_desired: Target speed (m/s)
            v_current: Current speed (m/s)
            dt: Time step (s)
            
        Returns:
            (throttle, brake): Control outputs [0, 1]
        """
        # Compute error
        v_error = v_desired - v_current
        
        # Integral with anti-windup
        self.v_error_integral += v_error * dt
        self.v_error_integral = np.clip(
            self.v_error_integral,
            self.gains.integrator_min,
            self.gains.integrator_max
        )
        
        # Derivative
        v_error_rate = (v_error - self.v_error_prev) / dt if dt > 0 else 0.0
        
        # PID output
        control = (self.gains.kp * v_error + 
                   self.gains.ki * self.v_error_integral + 
                   self.gains.kd * v_error_rate)
        
        # Split into throttle/brake
        if control >= 0:
            throttle = control
            brake = 0.0
        else:
            throttle = 0.0
            brake = -control
        
        # Store for next iteration
        self.v_error_prev = v_error
        
        return throttle, brake
    
    def reset(self):
        """Reset controller state"""
        self.v_error_integral = 0.0
        self.v_error_prev = 0.0


@dataclass
class PurePursuitGains:
    """Pure Pursuit controller gains"""
    lookahead_distance: float = 2.0  # meters
    kp_heading: float = 8.00
    k_speed_crosstrack: float = 0.00
    cross_track_deadband: float = 0.01  # meters


class PurePursuitController:
    """Lateral controller using Pure Pursuit algorithm"""
    
    def __init__(self, lookahead_distance=2.0, kp_heading=8.00):
        self.gains = PurePursuitGains(
            lookahead_distance=lookahead_distance,
            kp_heading=kp_heading
        )
        
    def update(self, x: float, y: float, yaw: float, speed: float,
               waypoints: List[Tuple[float, float, float]], 
               lookahead_idx: int) -> float:
        """
        Compute steering angle using Pure Pursuit
        
        Args:
            x, y: Current position (m)
            yaw: Current heading (rad)
            speed: Current speed (m/s)
            waypoints: List of (x, y, v) tuples
            lookahead_idx: Index of lookahead waypoint
            
        Returns:
            steer: Steering angle in radians (will be converted to [-1, 1])
        """
        # Lookahead point
        lookahead_x = waypoints[lookahead_idx][0]
        lookahead_y = waypoints[lookahead_idx][1]
        
        # Crosstrack error vector
        crosstrack_vector = np.array([
            lookahead_x - x - self.gains.lookahead_distance * np.cos(yaw),
            lookahead_y - y - self.gains.lookahead_distance * np.sin(yaw)
        ])
        crosstrack_error = np.linalg.norm(crosstrack_vector)
        
        # Apply deadband
        if crosstrack_error < self.gains.cross_track_deadband:
            crosstrack_error = 0.0
        
        # Crosstrack heading error
        crosstrack_heading = np.arctan2(crosstrack_vector[1], crosstrack_vector[0])
        crosstrack_heading_error = self._normalize_angle(crosstrack_heading - yaw)
        crosstrack_sign = np.sign(crosstrack_heading_error)
        
        # Path heading error
        if lookahead_idx < len(waypoints) - 1:
            wp_curr = waypoints[lookahead_idx]
            wp_next = waypoints[lookahead_idx + 1]
            trajectory_heading = np.arctan2(wp_next[1] - wp_curr[1],
                                             wp_next[0] - wp_curr[0])
        else:
            # Loop back to start
            trajectory_heading = np.arctan2(waypoints[0][1] - waypoints[-1][1],
                                             waypoints[0][0] - waypoints[-1][0])
        
        heading_error = self._normalize_angle(trajectory_heading - yaw)
        
        # Stanley-like steering law
        steer_output = heading_error + np.arctan(
            self.gains.kp_heading * crosstrack_sign * crosstrack_error / 
            (speed + self.gains.k_speed_crosstrack)
        )
        
        return steer_output
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi
```

### 5.4 Waypoint Management

#### 5.4.1 Waypoint Loader

**Location**: `src/utils/waypoint_loader.py`

```python
"""Waypoint loading and management utilities"""

import numpy as np
from typing import List, Tuple
from pathlib import Path

def load_waypoints(filepath: str) -> List[Tuple[float, float, float]]:
    """
    Load waypoints from text file
    
    Format: x, y, v_desired (one per line, comma-separated)
    
    Args:
        filepath: Path to waypoints file
        
    Returns:
        List of (x, y, v) tuples
    """
    waypoints = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse x, y, v
            parts = line.split(',')
            if len(parts) != 3:
                continue
            
            try:
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                v = float(parts[2].strip())
                waypoints.append((x, y, v))
            except ValueError:
                continue
    
    return waypoints


def find_closest_waypoint_index(x: float, y: float, 
                                  waypoints: List[Tuple[float, float, float]]) -> int:
    """
    Find index of closest waypoint to current position
    
    Args:
        x, y: Current position
        waypoints: List of waypoints
        
    Returns:
        Index of closest waypoint
    """
    min_dist = float('inf')
    min_idx = 0
    
    for i, (wx, wy, _) in enumerate(waypoints):
        dist = np.sqrt((wx - x)**2 + (wy - y)**2)
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    
    return min_idx


def get_lookahead_waypoint_index(x: float, y: float, 
                                   waypoints: List[Tuple[float, float, float]],
                                   lookahead_distance: float) -> int:
    """
    Find waypoint at lookahead distance along path
    
    Args:
        x, y: Current position
        waypoints: List of waypoints
        lookahead_distance: Distance to look ahead (meters)
        
    Returns:
        Index of lookahead waypoint
    """
    # Find closest waypoint
    closest_idx = find_closest_waypoint_index(x, y, waypoints)
    
    # Accumulate distance along path
    total_dist = 0.0
    lookahead_idx = closest_idx
    
    for i in range(closest_idx + 1, len(waypoints)):
        segment_dist = np.sqrt(
            (waypoints[i][0] - waypoints[i-1][0])**2 +
            (waypoints[i][1] - waypoints[i-1][1])**2
        )
        total_dist += segment_dist
        
        if total_dist >= lookahead_distance:
            lookahead_idx = i
            break
        
        lookahead_idx = i
    
    return lookahead_idx
```

---

## 6. Implementation Phases

### Phase 1: Research & Setup (1-2 days)

**Objectives**:
- [ ] Locate CARLA 0.9.16 native ROS 2 examples
- [ ] Identify exact topic names and message types
- [ ] Test basic pub/sub with CARLA server
- [ ] Verify synchronous mode behavior

**Tasks**:
1. **Search CARLA installation**:
   ```bash
   # Check for ROS 2 examples
   find /opt/carla-simulator-0.9.16 -name "*ros2*" -o -name "*ROS2*"
   
   # Search documentation
   grep -r "ros2" /opt/carla-simulator-0.9.16/Docs/
   ```

2. **Test minimal ROS 2 connection**:
   ```python
   # test_carla_ros2.py
   import rclpy
   from rclpy.node import Node
   
   class MinimalSubscriber(Node):
       def __init__(self):
           super().__init__('carla_test')
           # Try to find available topics
           topic_list = self.get_topic_names_and_types()
           self.get_logger().info(f'Available topics: {topic_list}')
   
   rclpy.init()
   node = MinimalSubscriber()
   rclpy.spin_once(node)
   node.destroy_node()
   rclpy.shutdown()
   ```

3. **Document findings**:
   - Create `ROS2_CARLA_NATIVE_API.md` with topic names, message types, launch procedures

### Phase 2: Baseline Controller Implementation (2-3 days)

**Objectives**:
- [ ] Extract PID + Pure Pursuit to standalone module
- [ ] Create ROS 2 node for baseline controller
- [ ] Test in simulation with waypoint following

**Tasks**:

1. **Create baseline module**:
   ```bash
   # File structure
   av_td3_system/src/baselines/
   ├── __init__.py
   ├── pid_pure_pursuit.py      # Controller logic (NEW)
   └── idm_mobil.py             # OLD - can remove
   ```

2. **Create ROS 2 node**:
   ```bash
   av_td3_system/src/ros_nodes/
   ├── __init__.py
   └── baseline_controller_node.py  # Main ROS 2 node (NEW)
   ```

3. **Create launch file**:
   ```bash
   av_td3_system/launch/
   └── baseline_controller.launch.py  # ROS 2 launch file (NEW)
   ```

4. **Test baseline controller**:
   ```bash
   # Terminal 1: Start CARLA with ROS 2 support
   ./CarlaUE4.sh -ROS2
   
   # Terminal 2: Launch baseline controller
   ros2 launch av_td3_system baseline_controller.launch.py
   
   # Terminal 3: Monitor topics
   ros2 topic echo /carla/ego_vehicle/vehicle_status
   ros2 topic echo /carla/ego_vehicle/vehicle_control_cmd
   ```

### Phase 3: Evaluation Infrastructure (2-3 days)

**Objectives**:
- [ ] Create evaluation script for baseline
- [ ] Run baseline on test scenarios (20, 50, 100 NPCs)
- [ ] Collect metrics for paper comparison

**Tasks**:

1. **Create evaluation script**:
   ```bash
   av_td3_system/scripts/
   └── evaluate_baseline.py  # Matches structure of evaluate_td3.py
   ```

2. **Run evaluation**:
   ```bash
   # Low traffic
   python scripts/evaluate_baseline.py --npc-count 20 --episodes 20
   
   # Medium traffic
   python scripts/evaluate_baseline.py --npc-count 50 --episodes 20
   
   # High traffic
   python scripts/evaluate_baseline.py --npc-count 100 --episodes 20
   ```

3. **Generate comparison report**:
   ```bash
   python scripts/compare_baselines.py --td3-results results/td3/ --baseline-results results/baseline/
   ```

### Phase 4: DRL Agent ROS 2 Migration (3-5 days, OPTIONAL)

**Objectives**:
- [ ] Migrate TD3 agent to use same ROS 2 topics
- [ ] Ensure fair comparison (same infrastructure)
- [ ] Validate performance parity with direct API

**Approach**:
- Keep TD3 training with direct Python API initially
- Create ROS 2 wrapper for **evaluation only**
- Compare: Direct API TD3 vs ROS 2 TD3 vs Baseline

---

## 7. Technical Specifications

### 7.1 Control Loop Timing

| Parameter | Value | Source |
|-----------|-------|--------|
| CARLA simulation rate | 20 Hz (0.05s) | `carla_config.yaml: fixed_delta_seconds` |
| ROS 2 node rate | 20 Hz (0.05s) | Match CARLA for synchronous operation |
| PID update rate | 20 Hz | Same as control loop |
| Pure Pursuit update rate | 20 Hz | Same as control loop |

### 7.2 Controller Parameters

#### PID Gains (from legacy `controller2d.py`)
```yaml
kp: 0.50
ki: 0.30
kd: 0.13
integrator_min: 0.0
integrator_max: 10.0
```

#### Pure Pursuit Gains
```yaml
lookahead_distance: 2.0  # meters
kp_heading: 8.00
k_speed_crosstrack: 0.00
cross_track_deadband: 0.01  # meters
```

### 7.3 State Variables

| Variable | Source Topic | Message Field | Units |
|----------|--------------|---------------|-------|
| x, y | `/carla/ego_vehicle/odometry` | `pose.pose.position.x/y` | meters |
| yaw | `/carla/ego_vehicle/odometry` | `pose.pose.orientation` (quaternion→euler) | radians |
| speed | `/carla/ego_vehicle/vehicle_status` | `velocity` | m/s |
| acceleration | `/carla/ego_vehicle/imu` | `linear_acceleration.x` | m/s² |

### 7.4 Control Outputs

| Variable | Target Topic | Message Field | Range |
|----------|--------------|---------------|-------|
| throttle | `/carla/ego_vehicle/vehicle_control_cmd` | `throttle` | [0.0, 1.0] |
| steer | `/carla/ego_vehicle/vehicle_control_cmd` | `steer` | [-1.0, 1.0] |
| brake | `/carla/ego_vehicle/vehicle_control_cmd` | `brake` | [0.0, 1.0] |

---

## 8. Migration Path for DRL Agent

### 8.1 Current DRL Architecture

```python
# environment/carla_env.py (current)
class CarlaEnv(gymnasium.Env):
    def __init__(self):
        # Direct CARLA client
        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()
        self.ego_vehicle = self.spawn_ego_vehicle()
        
    def step(self, action):
        # Direct control
        control = carla.VehicleControl(
            throttle=action[0],
            steer=action[1],
            brake=action[2]
        )
        self.ego_vehicle.apply_control(control)
        
        # Direct state retrieval
        transform = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()
        # ...
        
        return state, reward, done, info
```

### 8.2 Proposed ROS 2 Wrapper (Optional Future Work)

```python
# environment/carla_env_ros2.py (future)
class CarlaEnvROS2(gymnasium.Env):
    def __init__(self):
        # ROS 2 node interface
        self.ros_node = CarlaInterface()
        
    def step(self, action):
        # Publish control via ROS 2
        self.ros_node.publish_control(action)
        
        # Wait for state update (synchronous)
        state = self.ros_node.get_state()
        
        # Compute reward (same logic)
        reward = self.compute_reward(state)
        
        return state, reward, done, info
```

**Benefits**:
- Same infrastructure as baseline (fair comparison)
- Easier to swap simulators (CARLA → other ROS 2-compatible sim)
- Modular sensor processing (can add new sensors via ROS topics)

**Risks**:
- ROS 2 overhead may affect training performance
- Need to ensure state synchronization is reliable
- More complex debugging (distributed system)

**Recommendation**: Implement baseline with ROS 2 first, then evaluate if DRL migration is necessary for the paper.

---

## 9. Risk Assessment

### 9.1 Critical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Native ROS 2 API undocumented** | HIGH | HIGH | Fallback: Use external ROS bridge with documented API |
| **Topic names differ from bridge** | MEDIUM | MEDIUM | Systematic topic discovery with `ros2 topic list` |
| **Synchronous mode incompatible** | LOW | HIGH | Test early; document CARLA launch flags for ROS 2 sync |
| **Message types unavailable** | MEDIUM | MEDIUM | Check for `carla_msgs` package; may need to build from source |

### 9.2 Technical Challenges

| Challenge | Difficulty | Strategy |
|-----------|------------|----------|
| Finding native ROS 2 examples | MEDIUM | Search CARLA repo, Discord, GitHub issues |
| Converting quaternion to Euler | LOW | Use `tf_transformations` or manual conversion |
| Tuning controller for CARLA physics | MEDIUM | Start with legacy gains, iterate if needed |
| Handling ROS 2 / Python 3 compatibility | LOW | Use ROS 2 Foxy (Python 3.8+) |

### 9.3 Contingency Plan

**If native ROS 2 integration is incomplete**:

1. **Plan A**: Use external `carla-ros-bridge` with ROS 2 support
   - More latency, but well-documented
   - Proven architecture from previous work
   
2. **Plan B**: Keep DRL with direct API, baseline with Python wrapper
   - Create Python wrapper around `controller2d.py`
   - Call directly from evaluation script (no ROS 2)
   - Less modular, but faster to implement

3. **Plan C**: Implement both controllers with direct Python API
   - Create `BaselineAgent` class matching `TD3Agent` interface
   - No ROS 2 at all (simplest, but not using modern stack)

**Decision Point**: After Phase 1 research (1-2 days), decide which path to take.

---

## 10. Next Steps

### 10.1 Immediate Actions (Today)

1. **Locate CARLA 0.9.16 installation**:
   ```bash
   # Find CARLA installation
   which CarlaUE4.sh
   
   # Check version
   cat /opt/carla-simulator-0.9.16/VERSION
   
   # Search for ROS 2 documentation
   find /opt/carla-simulator-0.9.16 -name "*.md" | xargs grep -l "ROS2"
   ```

2. **Test CARLA with ROS 2 flag**:
   ```bash
   # Try launching CARLA with ROS 2 support
   ./CarlaUE4.sh -ROS2
   
   # In another terminal, check for ROS 2 topics
   source /opt/ros/foxy/setup.bash
   ros2 topic list
   ```

3. **Read existing codebase**:
   ```bash
   # Check current environment implementation
   cat av_td3_system/src/environment/carla_env.py | less
   
   # Check TD3 agent
   cat av_td3_system/src/agents/td3_agent.py | less
   
   # Review configuration
   cat av_td3_system/config/carla_config.yaml
   ```

### 10.2 Phase 1 Deliverables (1-2 days)

- [ ] `ROS2_CARLA_NATIVE_API.md`: Documentation of native ROS 2 topics/messages
- [ ] `test_carla_ros2.py`: Minimal test script for ROS 2 connection
- [ ] Decision: Native ROS 2 vs External Bridge vs Direct API

### 10.3 Phase 2 Deliverables (2-3 days)

- [ ] `src/baselines/pid_pure_pursuit.py`: Extracted controller module
- [ ] `src/ros_nodes/baseline_controller_node.py`: ROS 2 node implementation
- [ ] `launch/baseline_controller.launch.py`: Launch file
- [ ] `src/utils/waypoint_loader.py`: Waypoint utilities

### 10.4 Phase 3 Deliverables (2-3 days)

- [ ] `scripts/evaluate_baseline.py`: Evaluation script
- [ ] `results/baseline/`: Results for 20, 50, 100 NPC scenarios
- [ ] `BASELINE_RESULTS.md`: Metrics comparison with TD3

### 10.5 Paper Updates (1 day)

- [ ] Update Section IV.B with PID + Pure Pursuit description
- [ ] Add ROS 2 architecture diagram to Section III
- [ ] Update experimental results with baseline comparison

---

## Conclusion

This implementation plan provides a **systematic, research-driven approach** to integrating the PID + Pure Pursuit baseline controller with CARLA 0.9.16's native ROS 2 support. The plan prioritizes:

1. **Verification first**: Research native ROS 2 API before committing to architecture
2. **Modularity**: Clean separation via ROS 2 topics for future extensibility
3. **Scientific rigor**: Fair comparison between baseline and DRL using same infrastructure
4. **Risk management**: Multiple contingency plans if native support is incomplete

**Estimated Timeline**: 5-9 days total (assuming native ROS 2 API is usable)

**Next Action**: Begin Phase 1 research to locate CARLA 0.9.16 native ROS 2 documentation and test basic connectivity.

---

## References

1. CARLA 0.9.16 Release Notes: https://carla.org/2025/09/16/release-0.9.16/
2. CARLA ROS Bridge Documentation: https://carla.readthedocs.io/projects/ros-bridge/en/latest/
3. ROS 2 Foxy Documentation: https://docs.ros.org/en/foxy/
4. Legacy Controller Implementation: `related_works/exemple-carlaClient-openCV-YOLO-town01-waypoints/controller2d.py`
5. Current DRL System: `av_td3_system/src/environment/carla_env.py`

---

**Document Status**: ✅ Complete - Ready for Review and Implementation

