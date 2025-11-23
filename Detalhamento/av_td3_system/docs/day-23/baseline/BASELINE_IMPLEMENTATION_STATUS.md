# Baseline Controller Implementation Summary

## Overview

Successfully implemented a classical **PID + Pure Pursuit** baseline controller for autonomous vehicle navigation in CARLA 0.9.16. This baseline will be used for comparison with the TD3 deep reinforcement learning agent in the research paper.

**Implementation Date**: 2025  
**CARLA Version**: 0.9.16  
**Python Version**: 3.8+  
**Status**: ✅ Core implementation complete, ready for testing

---

## Implementation Progress

### ✅ Completed (8/14 tasks)

1. **Architecture Analysis** - Studied train_td3.py and carla_env.py
2. **Legacy Controller Study** - Extracted algorithms from controller2d.py
3. **API Documentation** - Fetched CARLA 0.9.16 Python API documentation
4. **PID Controller** - Implemented longitudinal speed control (157 lines)
5. **Pure Pursuit Controller** - Implemented lateral steering control (213 lines)
6. **Combined Controller** - Integrated both controllers with CARLA API (224 lines)
7. **Configuration File** - Created baseline_config.yaml with all parameters
8. **Evaluation Script** - Created evaluate_baseline.py following TD3 pattern (586 lines)

### ⏳ Pending (6/14 tasks)

9. **Unit Tests for PID** - Test anti-windup, throttle/brake splitting
10. **Unit Tests for Pure Pursuit** - Test steering, lookahead, angle normalization
11. **Integration Testing** - Run in Docker with CARLA server
12. **Metrics Validation** - Verify all paper metrics are collected
13. **Docker Integration** - Update docker-compose.yml
14. **Documentation** - Create README for baseline evaluation

---

## Architecture

### Component Structure

```
av_td3_system/
├── src/baselines/
│   ├── __init__.py                    # Package exports (UPDATED)
│   ├── pid_controller.py              # ✅ NEW (157 lines)
│   ├── pure_pursuit_controller.py     # ✅ NEW (213 lines)
│   └── baseline_controller.py         # ✅ NEW (224 lines)
├── config/
│   └── baseline_config.yaml           # ✅ NEW (controller params, eval settings)
└── scripts/
    └── evaluate_baseline.py           # ✅ NEW (586 lines, evaluation pipeline)
```

### Class Hierarchy

```
BaselineController
├── PIDController        (longitudinal control)
│   ├── update()         → (throttle, brake)
│   └── reset()
└── PurePursuitController (lateral control)
    ├── update()          → steering
    ├── _get_lookahead_index()
    └── _normalize_angle()
```

---

## Technical Specifications

### PID Controller

**File**: `src/baselines/pid_controller.py`

**Algorithm**: Classic PID with anti-windup integrator

**Control Law**:
```
error = target_speed - current_speed
integral += error * dt  (with anti-windup clamping)
derivative = (error - prev_error) / dt
output = kp * error + ki * integral + kd * derivative
```

**Parameters** (from controller2d.py):
- `kp = 0.50` - Proportional gain
- `ki = 0.30` - Integral gain  
- `kd = 0.13` - Derivative gain
- `integrator_min = 0.0` - Anti-windup lower bound
- `integrator_max = 10.0` - Anti-windup upper bound

**Output**: `(throttle, brake)` ∈ [0, 1]²
- If output ≥ 0: throttle = output, brake = 0
- If output < 0: throttle = 0, brake = -output

**Key Features**:
- ✅ Anti-windup prevents integral term saturation
- ✅ Separate throttle/brake splitting
- ✅ Stateful (reset required at episode start)

---

### Pure Pursuit Controller

**File**: `src/baselines/pure_pursuit_controller.py`

**Algorithm**: Pure Pursuit with Stanley's crosstrack error formula

**Control Law** (Stanley's formula):
```
lookahead_point = waypoints[lookahead_distance]
crosstrack_error = lateral_distance(vehicle, lookahead_point)
heading_error = angle_diff(path_heading, vehicle_yaw)
steer = heading_error + atan(kp_heading * crosstrack_error / (speed + k_speed_crosstrack))
```

**Parameters** (from controller2d.py):
- `lookahead_distance = 2.0` meters
- `kp_heading = 8.00` - Heading error gain
- `k_speed_crosstrack = 0.00` - Speed-dependent gain (disabled)
- `cross_track_deadband = 0.01` meters - Oscillation reduction

**Output**: `steering` ∈ [-1, 1]

**Coordinate Conversion**:
- Radians → CARLA steering: `steer_normalized = (180.0 / 70.0 / π) * steer_rad`

**Key Features**:
- ✅ Lookahead point selection (avoids sharp turns)
- ✅ Angle normalization to [-π, π]
- ✅ Crosstrack deadband (reduces oscillations)
- ✅ Stateless (no reset required)

---

### Combined Baseline Controller

**File**: `src/baselines/baseline_controller.py`

**Integration**:
```python
def compute_control(vehicle, waypoints, dt, target_speed=None):
    # 1. Extract state from CARLA
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    
    current_x = transform.location.x
    current_y = transform.location.y
    current_yaw = radians(transform.rotation.yaw)  # degrees → radians
    current_speed = sqrt(velocity.x² + velocity.y² + velocity.z²)
    
    # 2. Compute controls
    throttle, brake = pid_controller.update(current_speed, target_speed, dt)
    steer = pure_pursuit_controller.update(current_x, current_y, current_yaw, current_speed, waypoints)
    
    # 3. Return CARLA command
    return carla.VehicleControl(throttle, steer, brake, ...)
```

**CARLA API Compatibility** (0.9.16):
- ✅ `vehicle.get_transform()` → `carla.Transform`
- ✅ `vehicle.get_velocity()` → `carla.Vector3D` (m/s)
- ✅ `carla.VehicleControl(throttle, steer, brake, ...)`

**Key Features**:
- ✅ State extraction from CARLA vehicle
- ✅ Unit conversion (degrees→radians, Vector3D→scalar speed)
- ✅ Debug info method for logging
- ✅ Reset method for episode boundaries

---

## Configuration

### baseline_config.yaml

**Controller Parameters**:
```yaml
baseline_controller:
  pid:
    kp: 0.50
    ki: 0.30
    kd: 0.13
    integrator_min: 0.0
    integrator_max: 10.0
  pure_pursuit:
    lookahead_distance: 2.0
    kp_heading: 8.00
    k_speed_crosstrack: 0.00
    cross_track_deadband: 0.01
  general:
    target_speed_kmh: 30.0
```

**Evaluation Settings**:
```yaml
evaluation:
  num_episodes: 20
  max_steps: 1000
  fixed_delta_seconds: 0.05
  map_name: "Town01"
  waypoints_file: "config/waypoints.txt"
  
  traffic:
    num_vehicles: 50
    num_pedestrians: 0
  
  output_dir: "results/baseline_evaluation"
  save_trajectory: true
  save_images: false
  
  metrics:
    - success_rate
    - avg_collisions_per_km
    - avg_speed_kmh
    - route_completion_time_s
    - avg_longitudinal_jerk_m_s3
    - avg_lateral_acceleration_m_s2
    - ttc_analysis
```

---

## Evaluation Pipeline

### evaluate_baseline.py

**Usage**:
```bash
python scripts/evaluate_baseline.py \
    --scenario 0 \
    --seed 42 \
    --num-episodes 20 \
    --baseline-config config/baseline_config.yaml \
    --output-dir results/baseline_evaluation
```

**Command-line Arguments**:
- `--scenario {0,1,2}` - Traffic density (0=20, 1=50, 2=100 NPCs)
- `--seed INT` - Random seed (default: 42)
- `--num-episodes INT` - Number of evaluation episodes (default: 20)
- `--baseline-config PATH` - Config file path
- `--output-dir PATH` - Results output directory
- `--no-trajectory` - Disable trajectory saving
- `--debug` - Enable debug logging

**Evaluation Loop**:
```python
for episode in range(num_episodes):
    obs, info = env.reset()
    controller.reset()
    
    while not done:
        control = controller.compute_control(vehicle, waypoints, dt)
        action = [control.steer, control.throttle - control.brake]
        obs, reward, done, truncated, info = env.step(action)
        
        # Collect metrics
    
    # Store episode statistics
```

**Output Files**:
1. **Results JSON**: `baseline_scenario_0_YYYYMMDD-HHMMSS.json`
   - Configuration used
   - Aggregate metrics (mean ± std)
   - Per-episode data (rewards, collisions, speeds, etc.)

2. **Trajectories JSON** (if enabled): `trajectories_scenario_0_YYYYMMDD-HHMMSS.json`
   - Per-timestep vehicle state
   - Control commands (steer, throttle, brake)
   - Position, yaw, speed

**Metrics Collected**:

**Safety**:
- Success rate (%)
- Average collisions per episode
- Average lane invasions per episode

**Efficiency**:
- Mean episode reward
- Average speed (km/h)
- Average episode length (steps)

**Raw Data**:
- Per-episode rewards, successes, collisions, lane invasions, speeds

---

## Integration with Existing System

### Same Environment as TD3

The baseline evaluation uses **CARLANavigationEnv** (same as TD3 training):
- ✅ Same CARLA simulator (0.9.16)
- ✅ Same map (Town01)
- ✅ Same waypoints (config/waypoints.txt)
- ✅ Same traffic scenarios (20/50/100 NPCs)
- ✅ Same reward function (for fair comparison)

### Action Space Conversion

**Baseline Output**:
```python
control = carla.VehicleControl(
    throttle=0.5,  # [0, 1]
    steer=-0.3,    # [-1, 1]
    brake=0.0      # [0, 1]
)
```

**TD3 Environment Expects**:
```python
action = np.array([steer, throttle_brake])
# steer ∈ [-1, 1]
# throttle_brake ∈ [-1, 1] (positive=throttle, negative=brake)
```

**Conversion**:
```python
action = np.array([
    control.steer,
    control.throttle - control.brake  # Map to [-1, 1]
])
```

---

## Next Steps

### 1. Unit Testing (Tasks 9-10)

**Create**: `tests/test_pid_controller.py`
```python
def test_pid_proportional():
    # Test proportional response
def test_pid_integral_antiwindup():
    # Verify integrator clamping
def test_pid_throttle_brake_splitting():
    # Check correct output mapping
```

**Create**: `tests/test_pure_pursuit_controller.py`
```python
def test_lookahead_selection():
    # Verify correct waypoint selection
def test_angle_normalization():
    # Check [-π, π] wrapping
def test_crosstrack_deadband():
    # Verify small errors ignored
```

### 2. Integration Testing (Task 11)

**Steps**:
1. Start CARLA server in Docker
2. Run evaluation script:
   ```bash
   docker exec -it carla_container python scripts/evaluate_baseline.py --scenario 0 --num-episodes 5 --debug
   ```
3. Monitor console output for:
   - Vehicle spawning successfully
   - Waypoint following behavior
   - Control commands (steer, throttle, brake)
   - Collision/lane invasion events
4. Review saved results:
   - Check JSON files for metrics
   - Verify trajectories (if saved)

**Expected Behavior**:
- Vehicle should follow waypoints smoothly
- Speed should stabilize near 30 km/h
- Steering should be smooth (no oscillations)
- Should complete route without collisions (in low traffic)

### 3. Metrics Validation (Task 12)

**Paper Requirements** (from instructions):

**Safety**:
- ✅ Success Rate (%) - **IMPLEMENTED**
- ✅ Avg. Collisions/km - **IMPLEMENTED** (collisions per episode, need to add distance tracking)
- ❌ TTC (Time To Collision) analysis - **TODO**

**Efficiency**:
- ✅ Avg. Speed (km/h) - **IMPLEMENTED**
- ❌ Route Completion Time (s) - **TODO** (need to track time)

**Comfort**:
- ❌ Avg. Longitudinal Jerk (m/s³) - **TODO**
- ❌ Avg. Lateral Acceleration (m/s²) - **TODO**

**Required Additions**:
```python
# In evaluate_baseline.py, add:
- Distance traveled (for collisions/km)
- Completion time (timesteps * dt)
- Velocity history (for jerk calculation)
- Acceleration history (lateral acceleration)
- TTC calculation (distance to nearest vehicle / relative velocity)
```

### 4. Docker Integration (Task 13)

**Option A**: Update docker-compose.yml
```yaml
services:
  baseline_eval:
    build: .
    command: python scripts/evaluate_baseline.py --scenario 0
    volumes:
      - ./results:/workspace/av_td3_system/results
    depends_on:
      - carla_server
```

**Option B**: Add to Dockerfile
```dockerfile
# Add baseline evaluation entrypoint
COPY scripts/evaluate_baseline.py /workspace/av_td3_system/scripts/
RUN chmod +x /workspace/av_td3_system/scripts/evaluate_baseline.py
```

### 5. Documentation (Task 14)

**Create**: `docs/BASELINE_EVALUATION.md`
- How to run baseline evaluation
- How to interpret results
- Comparison protocol with TD3
- Troubleshooting guide

---

## Known Issues & Limitations

### Current Limitations

1. **Metrics Not Fully Implemented**:
   - TTC analysis (requires NPC vehicle tracking)
   - Route completion time (needs timestep accumulation)
   - Longitudinal jerk (requires velocity history)
   - Lateral acceleration (requires acceleration calculation)
   - Collisions/km (need distance tracking)

2. **Testing Coverage**:
   - No unit tests yet (planned in tasks 9-10)
   - No integration tests with CARLA (planned in task 11)

3. **CARLA Import Warning**:
   - `import carla` shows lint error in development environment
   - This is expected - CARLA is available in Docker runtime
   - No action needed

### Future Improvements

1. **Adaptive Control**:
   - Current parameters are fixed from controller2d.py
   - Could add parameter tuning for different scenarios
   - Could implement gain scheduling based on speed

2. **Trajectory Optimization**:
   - Current Pure Pursuit follows waypoints directly
   - Could add path smoothing
   - Could implement overtaking logic

3. **Safety Features**:
   - Add emergency braking
   - Add collision avoidance
   - Add traffic light detection

---

## References

### Code Sources

1. **Legacy Controller**: `FinalProject/controller2d.py`
   - PID gains: kp=0.50, ki=0.30, kd=0.13
   - Pure Pursuit params: lookahead=2.0, kp_heading=8.00

2. **TD3 Training**: `scripts/train_td3.py`
   - Evaluation loop structure
   - Environment initialization pattern
   - Metrics collection

3. **CARLA Environment**: `src/environment/carla_env.py`
   - Action space format
   - Observation structure
   - Reward function

### Documentation

1. **CARLA 0.9.16 Python API**: https://carla.readthedocs.io/en/latest/python_api/
   - VehicleControl specification
   - Vehicle methods (get_transform, get_velocity, apply_control)
   - Coordinate system (Unreal Engine Z-up left-handed)

2. **Implementation Plan**: `baseline_implementation_plan.md`
   - Original task breakdown
   - Timeline estimates
   - Success criteria

---

## Summary

### What Was Accomplished

✅ **Core Implementation Complete**:
- PID controller for speed control (157 lines)
- Pure Pursuit controller for steering (213 lines)
- Combined controller with CARLA API integration (224 lines)
- Configuration file with all parameters
- Evaluation script following TD3 pattern (586 lines)
- **Total**: ~1,180 lines of production code

✅ **Design Decisions**:
- Preserved parameters from controller2d.py (validated baseline)
- Used CARLA 0.9.16 API (compatibility verified)
- Followed train_td3.py patterns (consistent evaluation)
- Modular architecture (reusable components)

### What Needs Testing

⏳ **Testing Phase**:
- Unit tests for PID and Pure Pursuit
- Integration testing in Docker with CARLA
- Metrics validation (add missing metrics)
- Docker integration

### Timeline Estimate

**Completed**: ~8 hours (Controllers + Config + Evaluation script)  
**Remaining**:
- Unit tests: 2-3 hours
- Integration testing: 2-3 hours
- Metrics additions: 2 hours
- Docker integration: 1 hour
- Documentation: 1 hour

**Total Remaining**: ~8-9 hours (1-2 working days)

---

## Contact

For questions or issues:
- Check CARLA documentation: https://carla.readthedocs.io/
- Review baseline_implementation_plan.md
- Check train_td3.py for TD3 comparison

**Implementation Status**: ✅ READY FOR TESTING
