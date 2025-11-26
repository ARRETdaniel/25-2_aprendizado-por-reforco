# Phase 2.2: Minimal ROS Bridge Test - Version 4 Results

## Test Configuration (v4)

**Date**: 2025-01-23
**Test Version**: v4 (Manual Gear Control)
**Container**: ros2-bridge-minimal (ros2-carla-bridge:humble-v4)
**CARLA**: 0.9.16 (async mode, 20 FPS)

### Changes from v3
- **GEAR FIX ATTEMPT**: Set `manual_gear_shift=True` in control messages
- **Justification**: CARLA documentation suggests `gear` field is only respected when `manual_gear_shift=True`
- **Implementation**:
  - `brake_release` phase: `manual_gear_shift=True`, `gear=1`
  - `testing` phase: `manual_gear_shift=True`, `gear=1`

---

## Test Results

### Test Execution
```
[INFO] [1763860116.747182194] [minimal_bridge_controller]: ✅ Odometry received! Initial position: x=92.11, y=-30.82, z=0.00
[INFO] [1763860116.771865090] [minimal_bridge_controller]: 🔓 Releasing initial brake...
[INFO] [1763860117.821114298] [minimal_bridge_controller]: 🏁 Starting test: throttle=0.5, gear=1 (manual) for 5 seconds
[INFO] [1763860119.871578425] [minimal_bridge_controller]:   t=2.0s: distance = 0.00m
[INFO] [1763860121.821739518] [minimal_bridge_controller]:   t=4.0s: distance = 0.00m
[INFO] [1763860122.871194439] [minimal_bridge_controller]:   t=5.0s: distance = 0.00m
[INFO] [1763860122.871648693] [minimal_bridge_controller]: 🛑 Test complete, applying brake
```

### Final Metrics
- **Initial Position**: x=92.11, y=-30.82, z=0.00
- **Final Position**: x=92.11, y=-30.82, z=0.00
- **Distance Moved**: **0.00 meters** ❌
- **Test Duration**: 5 seconds
- **Throttle Applied**: 0.5
- **Gear**: 1 (manual mode)
- **Result**: **FAILED** ❌

---

## Analysis

### What Was Fixed
✅ **Spawn Point**: Vehicle at ground level (z=0.00) - working since v2
✅ **Brake Release**: Brake=0.0 published for 1s before throttle - implemented in v3
✅ **Manual Gear Control**: `manual_gear_shift=True` with `gear=1` - implemented in v4

### What STILL Doesn't Work
❌ **Vehicle Movement**: Despite all fixes, vehicle remains completely stationary

### Critical Finding
**The gear hypothesis was INCORRECT**. Even with manual gear control explicitly enabled and gear set to 1, the vehicle does NOT respond to throttle commands.

---

## Root Cause Investigation

Given that THREE successive fixes have failed:
1. v2: Fixed spawn (underground → ground level) ✅ → Still no movement
2. v3: Added brake release phase ✅ → Still no movement
3. v4: Enabled manual gear control ✅ → Still no movement

**This suggests the issue is NOT with**:
- Spawn position (already fixed)
- Brake engagement (already released)
- Gear configuration (manual control enabled)
- Message format (CarlaEgoVehicleControl structure correct)
- Manual override (already disabled)
- Simulation running (frame count increasing)

**Possible remaining causes**:
1. **Physics Simulation**: Vehicle physics might be disabled/suspended
2. **ActorControl Issues**: The spawned actor might not have physics enabled
3. **Control Topic Mapping**: Messages might not be reaching the correct vehicle
4. **CARLA Bridge Translation**: ROS → CARLA control conversion might be failing
5. **Vehicle Blueprint**: The spawned vehicle might have physics constraints
6. **Collision/Constraints**: Vehicle might be stuck against invisible geometry

---

## Next Steps

### 1. Verify Physics Simulation
- Check if vehicle has `simulate_physics` enabled
- Verify vehicle is not kinematic-only
- Inspect vehicle's physics control settings

### 2. Validate Control Flow
- Monitor ROS Bridge logs for control message processing
- Check CARLA logs for applied vehicle controls
- Verify vehicle ID matches control topic

### 3. Test Alternative Approaches
- Try applying direct CARLA Python API control (bypass ROS)
- Test with different vehicle blueprints
- Verify spawned vehicle has drivable characteristics

### 4. Debug Information Needed
```bash
# Check vehicle physics status
ros2 topic echo /carla/ego_vehicle/vehicle_status

# Monitor control command reception
ros2 topic echo /carla/ego_vehicle/vehicle_control_cmd

# Inspect ROS Bridge logs
docker logs ros2-bridge-minimal | grep -i "control\|physics\|gear"
```

---

## Test Progress

| Test Version | Spawn | Brake | Gear | Movement | Result |
|--------------|-------|-------|------|----------|--------|
| v1 | ❌ Underground | ❌ Engaged | ❌ Auto | ❌ 0.00m | FAIL |
| v2 | ✅ Ground | ❌ Engaged | ❌ Auto | ❌ 0.00m | FAIL |
| v3 | ✅ Ground | ✅ Released | ❌ Auto | ❌ 0.00m | FAIL |
| v4 | ✅ Ground | ✅ Released | ✅ Manual | ❌ 0.00m | **FAIL** |

---

## Phase 2.2 Status

**Progress**: 95% → Blocked
**Blockers**:
- Root cause of vehicle immobility still unknown after 3 fix attempts
- Need deeper investigation into CARLA vehicle physics/control

**Impact on Downstream Phases**:
- Phase 2.3 (PID+Pure Pursuit extraction) remains blocked
- Cannot proceed without functional vehicle control via ROS Bridge

---

## References

- Test script: `av_td3_system/tests/test_minimal_ros_bridge.py`
- Previous results: `PHASE_2_2_MINIMAL_BRIDGE_TEST_RESULTS.md`
- CARLA VehicleControl docs: https://carla.readthedocs.io/en/latest/python_api/#carlavehiclecontrol
- Manual gear shift property:
  ```python
  manual_gear_shift (bool): Determines whether the vehicle will be
  controlled by changing gears manually. Default is False.
  gear (int): States which gear is the vehicle running on.
  ```
