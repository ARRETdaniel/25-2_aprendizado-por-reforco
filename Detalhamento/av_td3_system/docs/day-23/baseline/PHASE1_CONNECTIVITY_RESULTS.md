# Phase 1: Basic Connectivity - RESULTS

**Date**: November 23, 2025
**Status**: ✅ **COMPLETE - ALL SUCCESS CRITERIA MET**

---

## Test Configuration

**CARLA Server**:
```bash
docker run -d --name carla-server --runtime=nvidia --net=host \
    --env=NVIDIA_VISIBLE_DEVICES=all \
    --env=NVIDIA_DRIVER_CAPABILITIES=all \
    carlasim/carla:0.9.16 bash CarlaUE4.sh -RenderOffScreen -nosound
```

**Baseline Evaluation** (matching TD3 pattern):
```bash
cd /path/to/av_td3_system && \
docker run --rm --network host --runtime nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH=/workspace \
    -v $(pwd):/workspace \
    -w /workspace \
    td3-av-system:v2.0-python310 \
    python3 scripts/evaluate_baseline.py \
        --scenario 0 \
        --num-episodes 1 \
        --baseline-config config/baseline_config.yaml \
        --debug
```

---

## Test Results

### ✅ Success Criteria Met

1. **CARLA Connection**: ✅ SUCCESS
   - Connected to CARLA server at localhost:2000
   - No connection errors
   - Server version: 0.9.16

2. **Vehicle Spawning**: ✅ SUCCESS
   - Vehicle spawned successfully
   - No spawn errors
   - Vehicle actor created

3. **Script Execution**: ✅ SUCCESS
   - No crashes
   - Episode completed normally
   - Clean shutdown

4. **Episode Completion**: ✅ SUCCESS
   - Episode ran for 37 steps
   - Terminated gracefully
   - No runtime errors

### Test Output

```
======================================================================
BASELINE CONTROLLER EVALUATION - PID + PURE PURSUIT
======================================================================

[CONFIG] Loading configurations...
[CONFIG] CARLA config: config/carla_config.yaml
[CONFIG] Baseline config: config/baseline_config.yaml
[CONFIG] Scenario: 0 (0=20, 1=50, 2=100 NPCs)
[CONFIG] Set CARLA_SCENARIO_INDEX=0
[CONFIG] Expected NPC count: 20
[CONFIG] Traffic Manager port: 8000

[ENVIRONMENT] Initializing CARLA environment...
[ENVIRONMENT] Map: Town01
[ENVIRONMENT] Max episode steps: 1000
[WAYPOINTS] Loaded 86 waypoints from /workspace/config/waypoints.txt

[CONTROLLER] Initializing PID + Pure Pursuit baseline...
[CONTROLLER] PID gains: kp=0.5, ki=0.3, kd=0.13
[CONTROLLER] Pure Pursuit: lookahead=2.0m, kp_heading=8.0
[CONTROLLER] Target speed: 30.0 km/h
[CONTROLLER] Simulation timestep: 0.05s

[INIT] Baseline evaluation pipeline ready!
[INIT] Number of episodes: 1
[INIT] Seed: 42
[INIT] Debug mode: True
======================================================================

[EVAL] Starting evaluation...

[EVAL] Episode 1/1
[EVAL] Episode 1 complete:
       Reward: 71.29
       Success: False
       Collisions: 0
       Lane Invasions: 1
       Length: 37 steps
       Avg Speed: 17.07 km/h

======================================================================
EVALUATION SUMMARY
======================================================================

Safety Metrics:
  Success Rate: 0.0%
  Avg Collisions: 0.00 ± 0.00
  Avg Lane Invasions: 1.00 ± 0.00

Efficiency Metrics:
  Mean Reward: 71.29 ± 0.00
  Avg Speed: 17.07 ± 0.00 km/h
  Avg Episode Length: 37.0 ± 0.0 steps
```

---

## Initial Performance Observations

**Speed Tracking**:
- Target: 30.0 km/h
- Achieved: 17.07 km/h (56.9% of target)
- **Issue**: Vehicle may not be accelerating enough (needs Phase 2 control verification)

**Safety**:
- Collisions: 0 ✅
- Lane Invasions: 1 ⚠️
- Episode terminated early (37 steps vs 1000 max)

**Waypoint Following**:
- Loaded 86 waypoints successfully
- Controller executed without errors
- Episode ended prematurely (possibly due to lane invasion or other termination condition)

---

## Warnings Observed

1. **GlobalRoutePlanner not available** - EXPECTED
   - Using legacy waypoint manager instead
   - Does not affect baseline controller operation

2. **Lane Invasion Warning**:
   ```
   WARNING:src.environment.sensors:Lane invasion detected
   WARNING:src.environment.reward_functions:[LANE_KEEPING] Lane invasion detected - applying maximum penalty (-1.0)
   WARNING:src.environment.reward_functions:[SAFETY-OFFROAD] penalty=-10.0
   WARNING:src.environment.reward_functions:[SAFETY-LANE_INVASION] penalty=-10.0
   ```
   - **Action Required**: Tune controller parameters to reduce lane invasions
   - Likely cause: Steering control needs adjustment

---

## Files Generated

**Trajectories**:
```
results/baseline_evaluation/trajectories/trajectories_scenario_0_20251123-131226.json
```

**Evaluation Results**:
```
results/baseline_evaluation/baseline_scenario_0_20251123-131226.json
```

---

## Issues Fixed During Phase 1

1. **Import Error**: `ModuleNotFoundError: No module named 'src'`
   - **Fix**: Changed sys.path from hardcoded `/workspace/av_td3_system` to dynamic `os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))`

2. **CARLA Import Error**: `ModuleNotFoundError: No module named 'carla'`
   - **Fix**: Updated `src/baselines/__init__.py` to remove IDMMOBILBaseline import (requires CARLA)
   - **Reason**: Baseline controller only needs PID + Pure Pursuit

3. **Debug Info KeyError**: `KeyError: 'current_speed_ms'`
   - **Fix**: Updated evaluate_baseline.py to use correct keys:
     - `debug_info['speed_m_s']` instead of `debug_info['current_speed_ms']`
     - `debug_info['position']['x']` instead of `debug_info['position_x']`
     - `debug_info['rotation']['yaw']` instead of `debug_info['rotation_yaw']`

4. **Volume Mount Path**: FileNotFoundError for waypoints.txt
   - **Fix**: Changed volume mount from `-v $(pwd)/av_td3_system:/workspace/av_td3_system` to `-v $(pwd):/workspace`
   - **Matches**: TD3 training pattern from RUN-COMMAND.md

---

## Next Steps: Phase 2 - Control Verification

**Goal**: Verify controllers produce reasonable control commands

**Actions**:
1. Add detailed debug logging (steering, throttle, brake, speed every second)
2. Run test with logging enabled
3. Analyze:
   - Control command bounds (steering [-1,1], throttle/brake [0,1])
   - Speed tracking convergence
   - Steering response to waypoints
   - No NaN/Inf in outputs

**Expected Time**: 1 hour

---

## Conclusion

**Phase 1 Status**: ✅ **COMPLETE**

All success criteria met:
- CARLA connection working ✅
- Vehicle spawning successful ✅
- Script executes without crashes ✅
- Episode completes (though terminated early) ✅

**Identified Issues**:
- ⚠️ Low speed (17 km/h vs 30 km/h target) - needs control tuning
- ⚠️ Lane invasion - needs steering tuning
- ⚠️ Episode terminated early - investigate termination conditions

**Ready to proceed to Phase 2**: Control Verification 🚀
