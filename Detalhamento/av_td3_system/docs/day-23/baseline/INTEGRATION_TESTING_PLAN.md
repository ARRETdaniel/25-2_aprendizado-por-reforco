# Baseline Controller Integration Testing Plan

**Date**: November 23, 2025  
**Status**: ⏳ **READY TO START**

---

## Overview

With unit tests complete (41/41 passing), we now proceed to integration testing with the CARLA simulator to validate real-world waypoint following behavior.

---

## Prerequisites

### Completed ✅

- ✅ PID Controller implementation (`src/baselines/pid_controller.py`)
- ✅ Pure Pursuit Controller implementation (`src/baselines/pure_pursuit_controller.py`)
- ✅ Combined Baseline Controller (`src/baselines/baseline_controller.py`)
- ✅ Configuration file (`config/baseline_config.yaml`)
- ✅ Evaluation script (`scripts/evaluate_baseline.py`)
- ✅ Waypoints file (`config/waypoints.txt` - same as TD3)
- ✅ Unit tests (18 PID + 23 Pure Pursuit = 41 total, all passing)

### Required for Integration Testing

- Docker with CARLA 0.9.16 image
- GPU access for rendering (if needed)
- Network access for CARLA server

---

## Testing Strategy

### Phase 1: Basic Connectivity (30 min)

**Goal**: Verify CARLA server connection and vehicle spawning

**Steps**:

1. **Start CARLA server**:
```bash
docker run --rm -it --gpus all --net=host \
    carlasim/carla:0.9.16 \
    /bin/bash CarlaUE4.sh -RenderOffScreen -nosound
```

2. **Run minimal test** (1 episode):
```bash
python scripts/evaluate_baseline.py \
    --scenario 0 \
    --num-episodes 1 \
    --baseline-config config/baseline_config.yaml \
    --debug
```

3. **Expected Output**:
```
[INFO] Connecting to CARLA server at localhost:2000...
[INFO] Connected! Server version: 0.9.16
[INFO] Loading waypoints from config/waypoints.txt...
[INFO] Loaded 1400 waypoints
[INFO] Starting evaluation scenario 0 (20 NPCs)...
[INFO] Episode 1/1: Spawning vehicle...
[INFO] Vehicle spawned at (x, y, z)
[INFO] Running episode...
```

**Success Criteria**:
- No connection errors
- Vehicle spawns successfully
- Script runs without crashes
- Episode completes (success or failure)

**Potential Issues**:
- CARLA server not responding → Check port 2000
- Vehicle spawn failure → Check spawn point availability
- Script crashes → Check traceback for errors

---

### Phase 2: Control Verification (1 hour)

**Goal**: Verify controllers produce reasonable control commands

**Steps**:

1. **Add debug logging** to evaluate_baseline.py:
```python
# In evaluation loop:
if step % 20 == 0:  # Log every second (20 steps @ 20Hz)
    print(f"[DEBUG] Step {step}:")
    print(f"  Current speed: {current_speed:.2f} m/s")
    print(f"  Target speed: {target_speed:.2f} m/s")
    print(f"  Steering: {steering:.3f}")
    print(f"  Throttle: {throttle:.3f}")
    print(f"  Brake: {brake:.3f}")
    print(f"  Distance to waypoint: {distance_to_next_wp:.2f} m")
```

2. **Run test with logging**:
```bash
python scripts/evaluate_baseline.py \
    --scenario 0 \
    --num-episodes 1 \
    --baseline-config config/baseline_config.yaml \
    --debug \
    2>&1 | tee logs/integration_test_debug.log
```

3. **Analyze control values**:
- Steering in [-1, 1]? ✓
- Throttle in [0, 1]? ✓
- Brake in [0, 1]? ✓
- Throttle and brake mutually exclusive? ✓
- Speed tracking target? ✓

**Success Criteria**:
- Control commands within valid ranges
- Speed converges toward target (e.g., 30 km/h)
- Steering responds to waypoint direction
- No NaN or Inf in control outputs

**Potential Issues**:
- Speed not converging → Tune PID gains
- Excessive steering oscillation → Reduce heading gain
- Vehicle stuck/not moving → Check throttle computation

---

### Phase 3: Waypoint Following (2 hours)

**Goal**: Verify vehicle follows waypoints accurately

**Steps**:

1. **Enable trajectory saving**:
```python
# In evaluate_baseline.py config
save_trajectory: true
trajectory_dir: 'results/baseline_trajectories'
```

2. **Run short route**:
```bash
python scripts/evaluate_baseline.py \
    --scenario 0 \
    --num-episodes 3 \
    --baseline-config config/baseline_config.yaml \
    --save-trajectories
```

3. **Visualize trajectory**:
```python
import matplotlib.pyplot as plt
import numpy as np

# Load trajectory
trajectory = np.loadtxt('results/baseline_trajectories/episode_0.txt')
waypoints = np.loadtxt('config/waypoints.txt', delimiter=',')

# Plot
plt.figure(figsize=(12, 8))
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Vehicle trajectory', linewidth=2)
plt.plot(waypoints[:, 0], waypoints[:, 1], 'r--', label='Waypoints', linewidth=1)
plt.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, label='Start')
plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, label='End')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.title('Baseline Controller Trajectory')
plt.grid(True)
plt.axis('equal')
plt.savefig('results/baseline_trajectory_plot.png', dpi=150)
plt.show()
```

**Success Criteria**:
- Trajectory follows waypoints closely
- No excessive weaving or oscillation
- Vehicle stays on road
- Smooth steering behavior

**Metrics to Check**:
- Average crosstrack error < 1.0 m
- Maximum crosstrack error < 3.0 m
- Heading error < 15°
- Completion rate > 80%

**Potential Issues**:
- High crosstrack error → Increase heading gain or reduce lookahead
- Oscillation → Reduce heading gain or add damping
- Cutting corners → Increase lookahead distance
- Missing waypoints → Check lookahead selection logic

---

### Phase 4: NPC Interaction (2 hours)

**Goal**: Test baseline controller with dynamic obstacles (NPCs)

**Steps**:

1. **Run all scenarios**:
```bash
# Scenario 0: 20 NPCs
python scripts/evaluate_baseline.py --scenario 0 --num-episodes 5

# Scenario 1: 50 NPCs  
python scripts/evaluate_baseline.py --scenario 1 --num-episodes 5

# Scenario 2: 100 NPCs
python scripts/evaluate_baseline.py --scenario 2 --num-episodes 5
```

2. **Monitor safety metrics**:
```python
# Expected metrics from evaluation
{
    "success_rate": 0.60,  # 60% completion (baseline expected)
    "avg_collisions": 0.8,  # Per episode
    "avg_lane_invasions": 2.5,
    "avg_speed": 25.3,  # km/h (target 30)
    "avg_episode_length": 450  # steps
}
```

3. **Compare across scenarios**:
- Success rate decreases with NPC count? ✓ (expected)
- Collision rate increases? ✓ (expected)
- Speed maintained? ✓ (should stay near 30 km/h)

**Success Criteria**:
- No script crashes with NPCs
- Reasonable completion rate (>50% for scenario 0)
- Collision detection working
- Lane invasion detection working

**Known Limitations** (baseline controller):
- No obstacle avoidance (will collide with blocking NPCs)
- No traffic light/stop sign handling
- Simple speed tracking (no adaptive cruise)

**Potential Issues**:
- Very low success rate → Check waypoint quality
- High collision rate → Expected (no obstacle avoidance)
- Script crashes with NPCs → Check NPC spawn logic

---

### Phase 5: Metrics Validation (1 hour)

**Goal**: Ensure all metrics match paper requirements

**Current Metrics** (from evaluate_baseline.py):
- ✅ Success rate (%)
- ✅ Average collisions per episode
- ✅ Average lane invasions
- ✅ Average speed (km/h)
- ✅ Episode length (steps)

**Missing Metrics** (from paper):
- ❌ Collisions per kilometer
- ❌ Route completion time (seconds)
- ❌ Longitudinal jerk (m/s³)
- ❌ Lateral acceleration (m/s²)
- ❌ TTC (Time To Collision) analysis

**Implementation**:

```python
class MetricsTracker:
    def __init__(self):
        self.distance_traveled = 0.0
        self.velocity_history = []
        self.acceleration_history = []
        self.lateral_accel_history = []
        self.start_time = None
        self.completion_time = None
    
    def update(self, dt, vehicle):
        # Distance tracking
        velocity = vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        self.distance_traveled += speed * dt
        self.velocity_history.append(speed)
        
        # Time tracking
        if self.start_time is None:
            self.start_time = time.time()
    
    def compute_final_metrics(self, collision_count):
        # Collisions per km
        distance_km = self.distance_traveled / 1000.0
        collisions_per_km = collision_count / distance_km if distance_km > 0 else 0.0
        
        # Route completion time
        if self.completion_time is None:
            self.completion_time = time.time()
        completion_time_s = self.completion_time - self.start_time
        
        # Longitudinal jerk
        velocities = np.array(self.velocity_history)
        accelerations = np.diff(velocities) / dt
        jerks = np.diff(accelerations) / dt
        avg_jerk = np.mean(np.abs(jerks)) if len(jerks) > 0 else 0.0
        
        # Lateral acceleration (requires position history)
        # TODO: Calculate from trajectory curvature
        
        return {
            'collisions_per_km': collisions_per_km,
            'completion_time_s': completion_time_s,
            'avg_jerk': avg_jerk,
            'distance_traveled_km': distance_km
        }
```

**Testing**:
```bash
# Run with updated metrics
python scripts/evaluate_baseline.py \
    --scenario 0 \
    --num-episodes 5 \
    --baseline-config config/baseline_config.yaml
```

**Validation**:
- Check JSON output contains all required metrics
- Verify values are reasonable (no NaN, Inf, negative distances)
- Compare with paper expected ranges

---

## Expected Results

### Baseline Performance Targets (from Paper)

**Safety**:
- Success Rate: 50-70% (scenario 0: 20 NPCs)
- Collisions/km: 0.5-1.5 (higher than TD3 expected)
- TTC: N/A (no predictive capability)

**Efficiency**:
- Average Speed: 25-30 km/h (target 30, but safety reduces this)
- Completion Time: Slower than TD3 (more cautious)

**Comfort**:
- Jerk: Higher than TD3 (abrupt PID corrections)
- Lateral Acceleration: Higher (reactive steering)

### Comparison Philosophy

The baseline is **expected to perform worse** than TD3 in:
- Safety (no obstacle prediction)
- Efficiency (reactive not proactive)
- Comfort (not optimized for smoothness)

This demonstrates the **value of learning-based approaches** (TD3).

---

## Debugging Checklist

If integration test fails, check:

### Connection Issues
- [ ] CARLA server running? (`docker ps`)
- [ ] Port 2000 accessible? (`telnet localhost 2000`)
- [ ] Firewall blocking? (check iptables)

### Spawn Issues
- [ ] Map loaded? (check CARLA logs)
- [ ] Spawn point available? (try different spawn index)
- [ ] Vehicle blueprint exists? (check actor_filter='vehicle.tesla.model3')

### Control Issues
- [ ] Waypoints loaded? (print len(waypoints))
- [ ] Control values valid? (print steering, throttle, brake)
- [ ] VehicleControl applied? (check CARLA API call)
- [ ] Timestep correct? (dt = 0.05 matches CARLA fixed_delta_seconds)

### Metrics Issues
- [ ] Sensors attached? (collision, lane_invasion)
- [ ] Callbacks registered? (weak_ref pattern)
- [ ] Episode termination working? (check done flag)
- [ ] File writes successful? (check output directory permissions)

---

## Success Criteria Summary

**Phase 1 (Connectivity)**: ✅ If script runs without connection errors  
**Phase 2 (Control)**: ✅ If control values are valid and reasonable  
**Phase 3 (Waypoints)**: ✅ If trajectory follows waypoints with <1m avg error  
**Phase 4 (NPCs)**: ✅ If success rate >50% in scenario 0  
**Phase 5 (Metrics)**: ✅ If all paper metrics collected and valid

---

## Timeline

- **Phase 1**: 30 minutes (basic connectivity)
- **Phase 2**: 1 hour (control verification)
- **Phase 3**: 2 hours (waypoint following)
- **Phase 4**: 2 hours (NPC interaction)
- **Phase 5**: 1 hour (metrics validation)

**Total Estimated Time**: 6.5 hours (1 working day)

---

## Next Steps After Integration Testing

1. **Metrics Addition** (Task 12)
   - Implement missing metrics (jerk, lateral accel, TTC)
   - Validate against paper requirements

2. **Docker Integration** (Task 13)
   - Update docker-compose.yml
   - Test full pipeline in container

3. **Documentation** (Task 14)
   - Create baseline evaluation README
   - Document results format for paper

4. **Paper Comparison**
   - Run baseline evaluation (20 episodes × 3 scenarios = 60 episodes)
   - Run TD3 evaluation (same protocol)
   - Generate comparison plots
   - Write results section

---

## Notes

- Keep all test logs in `logs/integration_test_*.log`
- Save all trajectories for later analysis
- Document any parameter tuning decisions
- Record all failures and their resolutions
- Take screenshots of interesting behaviors
- Keep metrics JSON files for paper comparison

**Status**: Ready to begin Phase 1 (Basic Connectivity) ⏳
