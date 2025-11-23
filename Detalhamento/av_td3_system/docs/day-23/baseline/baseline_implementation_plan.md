# Baseline Controller Implementation Plan

**Date**: 2025-01-20  
**Goal**: Implement PID + Pure Pursuit baseline controller for CARLA 0.9.16 without ROS Bridge dependency

---

## 1. Executive Summary

### 1.1 Context
After 4 unsuccessful iterations attempting to integrate ROS Bridge for baseline controller implementation, we are **pivoting to a direct Python API approach**. This decision is based on:

- ✅ **Working Pattern Exists**: `train_td3.py` successfully uses direct CARLA 0.9.16 Python API
- ✅ **Reusable Infrastructure**: `CARLANavigationEnv` wrapper provides complete CARLA integration
- ✅ **Proven Approach**: TD3 system demonstrates stable connection and control
- ⚠️ **ROS Blocker**: Control messages not reaching vehicle after extensive debugging

### 1.2 Objectives
1. **Implement PID + Pure Pursuit controllers** adapted from TCC reference code (CARLA 0.8.x → 0.9.x)
2. **Create baseline evaluation script** following `train_td3.py` pattern
3. **Ensure fair comparison** with TD3 agent (same waypoints, map, evaluation metrics)
4. **Docker integration** for consistent execution environment

### 1.3 Success Criteria
- ✅ Baseline controller successfully follows waypoints in Town01
- ✅ Same evaluation metrics as TD3 (success rate, collisions/km, TTC, speed, jerk, lateral acceleration)
- ✅ Executable via Docker (matching TD3 execution pattern)
- ✅ Results comparable with TD3 for scientific paper

---

## 2. Architecture Overview

### 2.1 System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    CARLA 0.9.16 Simulator                       │
│                    (Docker Container)                           │
└────────────────────┬────────────────────────────────────────────┘
                     │ Direct Python API
                     │ (carla.Client, carla.VehicleControl)
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              CARLANavigationEnv (Existing)                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ - CARLA connection (_connect_to_carla)                    │  │
│  │ - Synchronous mode setup                                  │  │
│  │ - Waypoint management (legacy + dynamic)                  │  │
│  │ - Sensor suite integration                                │  │
│  │ - Gymnasium interface (reset, step, observation)          │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │ obs, reward, done, info
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Baseline Controller (New)                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ PID Controller (Longitudinal)                             │  │
│  │  • Input: current_speed, target_speed                     │  │
│  │  • Output: throttle [0,1], brake [0,1]                    │  │
│  │  • Params: kp=0.50, ki=0.30, kd=0.13                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Pure Pursuit (Lateral)                                    │  │
│  │  • Input: vehicle_pose, waypoints                         │  │
│  │  • Output: steering [-1,1]                                │  │
│  │  • Params: lookahead=2.0m, kp_heading=8.00                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  compute_control(obs, waypoints) → VehicleControl              │
└────────────────────┬────────────────────────────────────────────┘
                     │ carla.VehicleControl(throttle, steer, brake)
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              Evaluation Script (New)                            │
│  - Initialize environment                                       │
│  - Load waypoints                                               │
│  - Run episodes (reset → control → step → metrics)             │
│  - Collect metrics (success, collisions, speed, comfort)        │
│  - Save results                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

**Evaluation Loop**:
```python
# 1. Environment Reset
obs, info = env.reset()
# obs = {'image': (4,84,84), 'vector': [v, d, φ, wx, wy, wz]}

# 2. Extract State
velocity = obs['vector'][0]  # Current speed
waypoints = info['waypoints']  # Local waypoints

# 3. Compute Control (Baseline Controller)
vehicle_transform = env.vehicle.get_transform()
vehicle_velocity = env.vehicle.get_velocity()
speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

# PID: target_speed → throttle/brake
throttle, brake = pid_controller.update(speed, target_speed, dt)

# Pure Pursuit: pose + waypoints → steering
steer = pure_pursuit.update(vehicle_transform, waypoints)

# 4. Create Control Command
control = carla.VehicleControl(
    throttle=throttle,
    steer=steer,
    brake=brake
)

# 5. Apply Control (via environment)
obs, reward, done, truncated, info = env.step(control)

# 6. Collect Metrics
metrics.update(info)
```

---

## 3. API Migration (0.8.x → 0.9.x)

### 3.1 API Compatibility Analysis

Based on CARLA 0.9.16 Python API documentation:

**✅ COMPATIBLE (No Changes Needed)**:

| Component | 0.8.x API | 0.9.x API | Status |
|-----------|-----------|-----------|--------|
| Vehicle Control | `carla.VehicleControl(throttle, steer, brake)` | `carla.VehicleControl(throttle, steer, brake)` | ✅ Same |
| Apply Control | `vehicle.apply_control(control)` | `vehicle.apply_control(control)` | ✅ Same |
| Get Velocity | `vehicle.get_velocity()` → `Vector3D` | `vehicle.get_velocity()` → `Vector3D` | ✅ Same |
| Get Transform | `vehicle.get_transform()` → `Transform` | `vehicle.get_transform()` → `Transform` | ✅ Same |
| Speed Limit | `vehicle.get_speed_limit()` → `float` | `vehicle.get_speed_limit()` → `float` | ✅ Same |

**❌ DEPRECATED (Need Replacement)**:

| Old API (0.8.x) | New API (0.9.x) | Impact |
|-----------------|-----------------|--------|
| `make_carla_client(host, port)` | `carla.Client(host, port)` | Client initialization |
| `CarlaSettings()` | `world.get_settings()` / `world.apply_settings()` | World configuration |
| Legacy client methods | Direct World/Actor API | Environment setup |

**✅ MITIGATION**: The `CARLANavigationEnv` already uses 0.9.x API, so no migration needed for client connection and world setup.

### 3.2 Controller Code Migration

**Reference Code** (controller2d.py - CARLA 0.8.x):
```python
# PID Controller (UNIVERSAL - No API dependency)
v_error = v_desired - v_current
v_error_integral += v_error * dt
v_error_derivative = (v_error - v_error_prev) / dt
throttle = kp * v_error + ki * v_error_integral + kd * v_error_derivative
throttle = np.clip(throttle, 0.0, 1.0)
```

**Target Code** (baseline_controller.py - CARLA 0.9.x):
```python
# SAME MATH - Just get state from carla_env instead of legacy client

# State retrieval (from CARLANavigationEnv, not controller)
velocity = vehicle.get_velocity()  # ✅ Same API
speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

# PID computation (✅ IDENTICAL - no API dependency)
v_error = target_speed - speed
v_error_integral += v_error * dt
v_error_derivative = (v_error - v_error_prev) / dt
throttle = kp * v_error + ki * v_error_integral + kd * v_error_derivative
throttle = np.clip(throttle, 0.0, 1.0)
```

**Key Insight**: The controller **math is universal** - only the state input source changes (from legacy client → `carla_env.vehicle`).

---

## 4. Implementation Details

### 4.1 File Structure

```
av_td3_system/
├── src/
│   ├── baselines/                    ← NEW DIRECTORY
│   │   ├── __init__.py
│   │   ├── pid_controller.py         ← PID implementation
│   │   ├── pure_pursuit_controller.py ← Pure Pursuit implementation
│   │   └── baseline_controller.py    ← Combined controller
│   ├── environment/                  ← EXISTING (Reuse)
│   │   └── carla_env.py              ✅ No changes needed
│   ├── agents/                       ← EXISTING (TD3 agent)
│   ├── networks/                     ← EXISTING (CNN)
│   └── utils/                        ← EXISTING
├── scripts/
│   ├── train_td3.py                  ← EXISTING (Reference pattern)
│   └── evaluate_baseline.py          ← NEW (Baseline evaluation)
├── config/
│   ├── waypoints.txt                 ✅ Same waypoints as TD3
│   ├── carla_config.yaml             ✅ Reuse
│   ├── td3_config.yaml               ✅ Reuse
│   └── baseline_config.yaml          ← NEW (Controller parameters)
├── docker/                           ✅ Reuse
├── docker-compose.yml                ✅ Reuse
└── docs/
    └── baseline_implementation_plan.md ← THIS FILE
```

### 4.2 Component Specifications

#### 4.2.1 PID Controller (`src/baselines/pid_controller.py`)

**Purpose**: Longitudinal speed control

**Inputs**:
- `current_speed` (float): Vehicle speed in m/s
- `target_speed` (float): Desired speed in m/s
- `dt` (float): Time step in seconds

**Outputs**:
- `throttle` (float): Throttle command [0, 1]
- `brake` (float): Brake command [0, 1]

**Parameters** (from controller2d.py):
- `kp = 0.50`: Proportional gain
- `ki = 0.30`: Integral gain
- `kd = 0.13`: Derivative gain
- `integrator_min = 0.0`: Integral term lower bound
- `integrator_max = 10.0`: Integral term upper bound

**Implementation**:
```python
class PIDController:
    def __init__(self, kp=0.50, ki=0.30, kd=0.13, integrator_min=0.0, integrator_max=10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integrator_min = integrator_min
        self.integrator_max = integrator_max
        
        # State variables
        self.v_error_integral = 0.0
        self.v_error_prev = 0.0
        self.t_prev = 0.0
    
    def update(self, current_speed, target_speed, dt):
        """
        Compute throttle/brake based on speed error.
        
        Args:
            current_speed (float): Current vehicle speed (m/s)
            target_speed (float): Desired vehicle speed (m/s)
            dt (float): Time step (seconds)
        
        Returns:
            tuple: (throttle [0,1], brake [0,1])
        """
        # Error calculation
        v_error = target_speed - current_speed
        
        # Integral term (with anti-windup)
        self.v_error_integral += v_error * dt
        self.v_error_integral = np.clip(
            self.v_error_integral, 
            self.integrator_min, 
            self.integrator_max
        )
        
        # Derivative term
        if dt > 0:
            v_error_derivative = (v_error - self.v_error_prev) / dt
        else:
            v_error_derivative = 0.0
        
        # PID output
        control_output = (
            self.kp * v_error +
            self.ki * self.v_error_integral +
            self.kd * v_error_derivative
        )
        
        # Split into throttle/brake
        if control_output >= 0:
            throttle = np.clip(control_output, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-control_output, 0.0, 1.0)
        
        # Update state
        self.v_error_prev = v_error
        
        return throttle, brake
    
    def reset(self):
        """Reset controller state."""
        self.v_error_integral = 0.0
        self.v_error_prev = 0.0
```

#### 4.2.2 Pure Pursuit Controller (`src/baselines/pure_pursuit_controller.py`)

**Purpose**: Lateral path following

**Inputs**:
- `vehicle_transform` (carla.Transform): Current vehicle pose (location, rotation)
- `waypoints` (list): List of (x, y, z) waypoint coordinates
- `current_speed` (float): Vehicle speed (for speed-adaptive steering)

**Outputs**:
- `steer` (float): Steering command [-1, 1] (radians internally)

**Parameters** (from controller2d.py):
- `lookahead_distance = 2.0`: Lookahead distance in meters
- `kp_heading = 8.00`: Heading error gain
- `k_speed_crosstrack = 0.00`: Speed-dependent crosstrack correction
- `cross_track_deadband = 0.01`: Deadband to reduce oscillations

**Implementation**:
```python
class PurePursuitController:
    def __init__(self, lookahead_distance=2.0, kp_heading=8.00, 
                 k_speed_crosstrack=0.00, cross_track_deadband=0.01):
        self.lookahead_distance = lookahead_distance
        self.kp_heading = kp_heading
        self.k_speed_crosstrack = k_speed_crosstrack
        self.cross_track_deadband = cross_track_deadband
        
        # Conversion factor (from controller2d.py)
        self.conv_rad_to_steer = 180.0 / 70.0 / np.pi
        self.pi = np.pi
        self.two_pi = 2.0 * np.pi
    
    def get_lookahead_index(self, current_x, current_y, waypoints):
        """Find index of waypoint at lookahead distance."""
        # Find closest waypoint
        min_idx = 0
        min_dist = float("inf")
        for i, wp in enumerate(waypoints):
            dist = np.linalg.norm([wp[0] - current_x, wp[1] - current_y])
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        # Advance to lookahead distance
        total_dist = min_dist
        lookahead_idx = min_idx
        for i in range(min_idx + 1, len(waypoints)):
            if total_dist >= self.lookahead_distance:
                break
            total_dist += np.linalg.norm([
                waypoints[i][0] - waypoints[i-1][0],
                waypoints[i][1] - waypoints[i-1][1]
            ])
            lookahead_idx = i
        
        return lookahead_idx
    
    def update(self, vehicle_transform, waypoints, current_speed):
        """
        Compute steering based on pure pursuit + heading error.
        
        Args:
            vehicle_transform (carla.Transform): Current vehicle pose
            waypoints (list): List of (x, y, z) waypoints
            current_speed (float): Current vehicle speed (m/s)
        
        Returns:
            float: Steering command [-1, 1]
        """
        # Extract vehicle state
        x = vehicle_transform.location.x
        y = vehicle_transform.location.y
        yaw = np.radians(vehicle_transform.rotation.yaw)  # Convert to radians
        
        # Find lookahead waypoint
        ce_idx = self.get_lookahead_index(x, y, waypoints)
        
        # Crosstrack error (lateral deviation)
        crosstrack_vector = np.array([
            waypoints[ce_idx][0] - x - self.lookahead_distance * np.cos(yaw),
            waypoints[ce_idx][1] - y - self.lookahead_distance * np.sin(yaw)
        ])
        crosstrack_error = np.linalg.norm(crosstrack_vector)
        
        # Apply deadband
        if crosstrack_error < self.cross_track_deadband:
            crosstrack_error = 0.0
        
        # Crosstrack heading (direction to lookahead point)
        crosstrack_heading = np.arctan2(crosstrack_vector[1], crosstrack_vector[0])
        crosstrack_heading_error = crosstrack_heading - yaw
        crosstrack_heading_error = (crosstrack_heading_error + self.pi) % self.two_pi - self.pi
        crosstrack_sign = np.sign(crosstrack_heading_error)
        
        # Trajectory heading (path direction)
        if ce_idx < len(waypoints) - 1:
            vect_wp0_to_wp1 = np.array([
                waypoints[ce_idx+1][0] - waypoints[ce_idx][0],
                waypoints[ce_idx+1][1] - waypoints[ce_idx][1]
            ])
        else:
            # Loop back to start
            vect_wp0_to_wp1 = np.array([
                waypoints[0][0] - waypoints[-1][0],
                waypoints[0][1] - waypoints[-1][1]
            ])
        
        trajectory_heading = np.arctan2(vect_wp0_to_wp1[1], vect_wp0_to_wp1[0])
        heading_error = trajectory_heading - yaw
        heading_error = (heading_error + self.pi) % self.two_pi - self.pi
        
        # Stanley controller formula (heading error + crosstrack correction)
        steer_rad = heading_error + np.arctan(
            self.kp_heading * crosstrack_sign * crosstrack_error / 
            (current_speed + self.k_speed_crosstrack)
        )
        
        # Convert radians to [-1, 1] range
        steer_normalized = self.conv_rad_to_steer * steer_rad
        steer_normalized = np.clip(steer_normalized, -1.0, 1.0)
        
        return steer_normalized
```

#### 4.2.3 Combined Baseline Controller (`src/baselines/baseline_controller.py`)

**Purpose**: Integrate PID and Pure Pursuit controllers

```python
class BaselineController:
    """
    Combined PID + Pure Pursuit controller for CARLA 0.9.16.
    Adapted from TCC controller2d.py (CARLA 0.8.x).
    """
    
    def __init__(self, 
                 # PID parameters
                 kp=0.50, ki=0.30, kd=0.13,
                 # Pure Pursuit parameters  
                 lookahead_distance=2.0, kp_heading=8.00):
        
        self.pid_controller = PIDController(kp, ki, kd)
        self.pure_pursuit = PurePursuitController(lookahead_distance, kp_heading)
    
    def compute_control(self, vehicle, waypoints, target_speed, dt):
        """
        Compute VehicleControl based on current state and waypoints.
        
        Args:
            vehicle (carla.Actor): CARLA vehicle actor
            waypoints (list): List of (x, y, z) waypoints
            target_speed (float): Desired speed (m/s)
            dt (float): Time step (seconds)
        
        Returns:
            carla.VehicleControl: Control command to apply
        """
        # Get vehicle state (CARLA 0.9.x API)
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        transform = vehicle.get_transform()
        
        # Longitudinal control (PID)
        throttle, brake = self.pid_controller.update(speed, target_speed, dt)
        
        # Lateral control (Pure Pursuit)
        steer = self.pure_pursuit.update(transform, waypoints, speed)
        
        # Create CARLA control command
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=False,
            reverse=False
        )
        
        return control
    
    def reset(self):
        """Reset controller state (for new episode)."""
        self.pid_controller.reset()
```

### 4.3 Evaluation Script (`scripts/evaluate_baseline.py`)

**Purpose**: Execute baseline controller in CARLA and collect metrics

**Pattern**: Follow `train_td3.py` structure

```python
#!/usr/bin/env python3
"""
Baseline Controller Evaluation Script for CARLA 0.9.16

Evaluates PID + Pure Pursuit baseline controller following the same pattern as TD3 training.
NO ROS Bridge dependency - uses direct CARLA Python API via CARLANavigationEnv.

Author: [Your Name]
Date: 2025-01-20
"""

import os
import sys
import time
import math
import yaml
import numpy as np
import carla

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.carla_env import CARLANavigationEnv
from src.baselines.baseline_controller import BaselineController

def load_waypoints(waypoint_file):
    """Load waypoints from file (x, y, z format)."""
    waypoints = []
    with open(waypoint_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    waypoints.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return waypoints

def run_evaluation(config_paths, num_episodes=20, render=False):
    """
    Run baseline controller evaluation.
    
    Args:
        config_paths (dict): Paths to configuration files
        num_episodes (int): Number of evaluation episodes
        render (bool): Whether to enable rendering
    
    Returns:
        dict: Evaluation metrics
    """
    # Load configurations
    with open(config_paths['carla'], 'r') as f:
        carla_config = yaml.safe_load(f)
    
    with open(config_paths['baseline'], 'r') as f:
        baseline_config = yaml.safe_load(f)
    
    # Initialize environment (SAME as TD3)
    env = CARLANavigationEnv(
        carla_config_path=config_paths['carla'],
        td3_config_path=config_paths['td3'],  # Reuse for observation params
        training_config_path=None,  # No training
        host='localhost',
        port=2000,
        headless=not render
    )
    
    # Initialize baseline controller
    controller = BaselineController(
        kp=baseline_config.get('kp', 0.50),
        ki=baseline_config.get('ki', 0.30),
        kd=baseline_config.get('kd', 0.13),
        lookahead_distance=baseline_config.get('lookahead_distance', 2.0),
        kp_heading=baseline_config.get('kp_heading', 8.00)
    )
    
    # Load waypoints (SAME as TD3)
    waypoints = load_waypoints(config_paths['waypoints'])
    target_speed = baseline_config.get('target_speed', 30.0 / 3.6)  # km/h → m/s
    
    # Metrics storage
    metrics = {
        'success_count': 0,
        'collision_count': 0,
        'route_completion': [],
        'episode_rewards': [],
        'avg_speed': [],
        'comfort_metrics': {
            'longitudinal_jerk': [],
            'lateral_acceleration': []
        }
    }
    
    # Evaluation loop
    print(f"\n{'='*60}")
    print(f"Starting Baseline Evaluation: {num_episodes} episodes")
    print(f"{'='*60}\n")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        controller.reset()
        
        episode_reward = 0
        episode_speed = []
        done = False
        truncated = False
        step = 0
        
        print(f"Episode {episode + 1}/{num_episodes} started...")
        
        while not (done or truncated):
            # Compute control (Baseline Controller)
            control = controller.compute_control(
                vehicle=env.vehicle,
                waypoints=waypoints,
                target_speed=target_speed,
                dt=env.fixed_delta_seconds
            )
            
            # Step environment (SAME as TD3)
            obs, reward, done, truncated, info = env.step(control)
            
            episode_reward += reward
            episode_speed.append(info.get('speed', 0.0))
            step += 1
            
            # Check termination
            if done:
                if info.get('collision', False):
                    metrics['collision_count'] += 1
                    print(f"  ❌ Episode {episode + 1} ended: COLLISION at step {step}")
                elif info.get('success', False):
                    metrics['success_count'] += 1
                    print(f"  ✅ Episode {episode + 1} ended: SUCCESS at step {step}")
                else:
                    print(f"  ⚠️  Episode {episode + 1} ended: OTHER REASON at step {step}")
            
            if truncated:
                print(f"  🕒 Episode {episode + 1} ended: TIMEOUT at step {step}")
        
        # Collect episode metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['route_completion'].append(info.get('route_completion', 0.0))
        metrics['avg_speed'].append(np.mean(episode_speed) if episode_speed else 0.0)
        
        if 'comfort_metrics' in info:
            metrics['comfort_metrics']['longitudinal_jerk'].append(
                info['comfort_metrics'].get('avg_longitudinal_jerk', 0.0)
            )
            metrics['comfort_metrics']['lateral_acceleration'].append(
                info['comfort_metrics'].get('avg_lateral_accel', 0.0)
            )
        
        print(f"  Reward: {episode_reward:.2f} | Speed: {np.mean(episode_speed)*3.6:.1f} km/h | Completion: {info.get('route_completion', 0)*100:.1f}%\n")
    
    # Clean up
    env.close()
    
    # Compute summary statistics
    summary = {
        'success_rate': (metrics['success_count'] / num_episodes) * 100,
        'collision_rate': (metrics['collision_count'] / num_episodes) * 100,
        'avg_reward': np.mean(metrics['episode_rewards']),
        'avg_route_completion': np.mean(metrics['route_completion']) * 100,
        'avg_speed_kmh': np.mean(metrics['avg_speed']) * 3.6,
        'avg_longitudinal_jerk': np.mean(metrics['comfort_metrics']['longitudinal_jerk']),
        'avg_lateral_accel': np.mean(metrics['comfort_metrics']['lateral_acceleration'])
    }
    
    print(f"\n{'='*60}")
    print(f"Evaluation Summary ({num_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Success Rate:         {summary['success_rate']:.1f}%")
    print(f"Collision Rate:       {summary['collision_rate']:.1f}%")
    print(f"Avg Route Completion: {summary['avg_route_completion']:.1f}%")
    print(f"Avg Speed:            {summary['avg_speed_kmh']:.1f} km/h")
    print(f"Avg Reward:           {summary['avg_reward']:.2f}")
    print(f"Avg Longitudinal Jerk: {summary['avg_longitudinal_jerk']:.3f} m/s³")
    print(f"Avg Lateral Accel:     {summary['avg_lateral_accel']:.3f} m/s²")
    print(f"{'='*60}\n")
    
    return summary, metrics

def main():
    """Main entry point."""
    # Configuration paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_paths = {
        'carla': os.path.join(project_root, 'config', 'carla_config.yaml'),
        'td3': os.path.join(project_root, 'config', 'td3_config.yaml'),
        'baseline': os.path.join(project_root, 'config', 'baseline_config.yaml'),
        'waypoints': os.path.join(project_root, 'config', 'waypoints.txt')
    }
    
    # Run evaluation
    summary, metrics = run_evaluation(
        config_paths=config_paths,
        num_episodes=20,
        render=False
    )
    
    # Save results
    results_dir = os.path.join(project_root, 'results', 'baseline')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'baseline_evaluation_{timestamp}.yaml')
    
    with open(results_file, 'w') as f:
        yaml.dump({'summary': summary, 'raw_metrics': metrics}, f)
    
    print(f"Results saved to: {results_file}")

if __name__ == '__main__':
    main()
```

### 4.4 Configuration File (`config/baseline_config.yaml`)

```yaml
# Baseline Controller Configuration for CARLA 0.9.16
# Adapted from TCC controller2d.py parameters

# PID Controller Parameters (Longitudinal Control)
kp: 0.50                # Proportional gain
ki: 0.30                # Integral gain
kd: 0.13                # Derivative gain
integrator_min: 0.0     # Integral term lower bound
integrator_max: 10.0    # Integral term upper bound

# Pure Pursuit Parameters (Lateral Control)
lookahead_distance: 2.0     # Lookahead distance (meters)
kp_heading: 8.00            # Heading error gain
k_speed_crosstrack: 0.00    # Speed-dependent crosstrack correction
cross_track_deadband: 0.01  # Deadband to reduce oscillations

# Evaluation Parameters
target_speed: 30.0          # Target speed (km/h)
num_episodes: 20            # Number of evaluation episodes
max_steps_per_episode: 2000 # Maximum steps before timeout

# Logging
log_level: INFO
save_metrics: true
results_dir: results/baseline
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

**Test 1: PID Controller**
```python
def test_pid_zero_error():
    """Test PID with zero error (should output zero throttle)."""
    pid = PIDController()
    throttle, brake = pid.update(current_speed=10.0, target_speed=10.0, dt=0.05)
    assert throttle < 0.1  # Small tolerance
    assert brake == 0.0

def test_pid_positive_error():
    """Test PID with positive error (should accelerate)."""
    pid = PIDController()
    throttle, brake = pid.update(current_speed=10.0, target_speed=20.0, dt=0.05)
    assert throttle > 0.0
    assert brake == 0.0

def test_pid_negative_error():
    """Test PID with negative error (should brake)."""
    pid = PIDController()
    throttle, brake = pid.update(current_speed=20.0, target_speed=10.0, dt=0.05)
    assert throttle == 0.0
    assert brake > 0.0
```

**Test 2: Pure Pursuit**
```python
def test_pure_pursuit_straight_path():
    """Test Pure Pursuit on straight path (should output near-zero steering)."""
    pp = PurePursuitController()
    
    # Straight waypoints
    waypoints = [[i, 0, 0] for i in range(10)]
    
    # Vehicle heading straight
    transform = carla.Transform(
        carla.Location(0, 0, 0),
        carla.Rotation(0, 0, 0)
    )
    
    steer = pp.update(transform, waypoints, speed=10.0)
    assert abs(steer) < 0.1  # Near straight
```

### 5.2 Integration Tests

**Test 3: Environment Integration**
```bash
# Start CARLA server
docker-compose up carla-server

# Run single episode test
python scripts/evaluate_baseline.py --test-mode --num-episodes 1
```

**Expected**:
- ✅ Environment connects successfully
- ✅ Vehicle spawns in Town01
- ✅ Controller computes valid controls (throttle, steer, brake in valid ranges)
- ✅ Vehicle moves along waypoints
- ✅ Episode terminates (success, collision, or timeout)

### 5.3 Performance Validation

**Test 4: Waypoint Following Accuracy**
```python
def test_waypoint_following():
    """Validate that baseline follows waypoints within acceptable tolerance."""
    # Run 5 episodes
    # Measure crosstrack error at each step
    # Assert: mean crosstrack error < 1.0 meter
    pass
```

**Test 5: Comparison with TD3**
```bash
# Run baseline evaluation
python scripts/evaluate_baseline.py --num-episodes 20

# Compare with TD3 results
python scripts/compare_agents.py --baseline results/baseline/latest.yaml \
                                  --td3 results/td3/latest.yaml
```

**Expected Metrics**:
- Success Rate: ~70-90% (baseline should be stable but less adaptive than TD3)
- Collision Rate: <20%
- Route Completion: >80%
- Average Speed: ~25-30 km/h

---

## 6. Docker Integration

### 6.1 Docker Execution Pattern

**Same as TD3** (no ROS Bridge needed):

```bash
# Start CARLA server
docker-compose up carla-server

# Run baseline evaluation (from host or container)
docker-compose exec agent python scripts/evaluate_baseline.py
```

### 6.2 Dockerfile (No Changes Needed)

The existing `Dockerfile` for the agent container already has all necessary dependencies:
- ✅ Python 3.8+
- ✅ CARLA Python API 0.9.16
- ✅ NumPy, PyYAML
- ✅ PyTorch (for TD3, not needed for baseline but doesn't hurt)

---

## 7. Validation Checklist

Before marking implementation complete, verify:

**✅ Functionality**:
- [ ] PID controller computes throttle/brake correctly
- [ ] Pure Pursuit computes steering correctly
- [ ] Baseline controller integrates both successfully
- [ ] Environment interface works (step, reset, observation, reward)
- [ ] Waypoints loaded and used correctly

**✅ Fair Comparison**:
- [ ] Same waypoints as TD3 (config/waypoints.txt)
- [ ] Same map (Town01)
- [ ] Same evaluation metrics (success, collisions, speed, jerk, lateral accel)
- [ ] Same episode termination conditions

**✅ Code Quality**:
- [ ] Type hints added
- [ ] Docstrings for all functions
- [ ] Comments explaining controller logic
- [ ] Unit tests passing
- [ ] Integration test successful

**✅ Docker**:
- [ ] Runs in Docker container
- [ ] No manual setup required
- [ ] Results saved to `results/baseline/`

**✅ Documentation**:
- [ ] Implementation plan complete (this document)
- [ ] Code comments clear
- [ ] README updated with baseline instructions

---

## 8. Timeline Estimate

| Task | Estimated Time | Priority |
|------|----------------|----------|
| **Phase 1: Controller Implementation** |  |  |
| Implement PID controller | 2 hours | HIGH |
| Implement Pure Pursuit controller | 3 hours | HIGH |
| Implement combined baseline controller | 1 hour | HIGH |
| Unit tests for controllers | 2 hours | MEDIUM |
| **Phase 2: Evaluation Script** |  |  |
| Create evaluate_baseline.py | 3 hours | HIGH |
| Add metrics collection | 2 hours | HIGH |
| Test environment integration | 2 hours | HIGH |
| **Phase 3: Configuration & Testing** |  |  |
| Create baseline_config.yaml | 1 hour | HIGH |
| Docker integration testing | 2 hours | HIGH |
| Full evaluation run (20 episodes) | 1 hour | HIGH |
| **Phase 4: Validation** |  |  |
| Compare with TD3 results | 2 hours | MEDIUM |
| Debug and fix issues | 4 hours | MEDIUM |
| Documentation update | 2 hours | LOW |
| **Total** | **~27 hours** | **~3-4 working days** |

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Controller instability (oscillations) | Medium | High | Tune PID/PP gains; add filtering |
| Environment integration issues | Low | Medium | Reuse proven CARLANavigationEnv |
| Docker execution problems | Low | Low | Test early; documented troubleshooting |
| Performance worse than expected | Medium | Medium | Benchmark against TCC; adjust parameters |
| API compatibility issues | Low | Low | Already validated 0.9.x API |

---

## 10. Success Metrics

**Minimum Viable Product (MVP)**:
- ✅ Baseline controller runs in CARLA 0.9.16
- ✅ Completes at least 50% of episodes successfully
- ✅ Metrics comparable to TD3 (within same order of magnitude)
- ✅ Executable via Docker

**Desired Performance**:
- ✅ Success rate >70%
- ✅ Collision rate <20%
- ✅ Route completion >80%
- ✅ Average speed 25-30 km/h

**Stretch Goals**:
- ✅ Success rate >85%
- ✅ Automated comparison report with TD3
- ✅ Ablation study (PID-only vs Pure Pursuit-only vs combined)

---

## 11. Next Steps

### Immediate Actions:
1. **Create baseline directory structure** (`src/baselines/`)
2. **Implement PID controller** (`pid_controller.py`)
3. **Implement Pure Pursuit controller** (`pure_pursuit_controller.py`)
4. **Implement combined controller** (`baseline_controller.py`)
5. **Create evaluation script** (`evaluate_baseline.py`)
6. **Create config file** (`baseline_config.yaml`)
7. **Test in Docker**

### Dependencies:
- ✅ CARLA 0.9.16 server running
- ✅ CARLANavigationEnv functional (already working)
- ✅ Waypoints file available (config/waypoints.txt)
- ✅ Docker infrastructure ready

### Blockers:
- ❌ **NONE** - All dependencies satisfied, ready to proceed

---

## 12. References

### CARLA Documentation:
- [CARLA 0.9.16 Python API](https://carla.readthedocs.io/en/0.9.16/python_api/)
- [VehicleControl](https://carla.readthedocs.io/en/0.9.16/python_api/#carla.VehicleControl)
- [Client](https://carla.readthedocs.io/en/0.9.16/python_api/#carla.Client)

### Reference Implementations:
- `av_td3_system/scripts/train_td3.py` (Working TD3 pattern)
- `av_td3_system/src/environment/carla_env.py` (CARLA wrapper)
- `related_works/.../controller2d.py` (PID + Pure Pursuit reference - CARLA 0.8.x)
- `related_works/.../module_7.py` (Legacy client integration)

### Configuration:
- `av_td3_system/config/waypoints.txt` (Town01 route)
- `av_td3_system/config/carla_config.yaml` (CARLA settings)
- `av_td3_system/docker-compose.yml` (Docker orchestration)

---

**END OF IMPLEMENTATION PLAN**

*This plan provides a complete roadmap for implementing the PID + Pure Pursuit baseline controller for CARLA 0.9.16 without ROS Bridge dependency, following proven patterns from the existing TD3 system.*
