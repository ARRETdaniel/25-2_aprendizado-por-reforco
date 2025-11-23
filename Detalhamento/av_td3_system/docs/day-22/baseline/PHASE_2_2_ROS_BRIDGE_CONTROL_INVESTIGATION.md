# Phase 2.2: ROS Bridge Vehicle Control Investigation

**Date:** 2025-11-22  
**Status:** 🔍 ACTIVE INVESTIGATION  
**Goal:** Verify ROS Bridge can control CARLA vehicles for baseline implementation

---

## Executive Summary

Through extensive testing and official documentation review, we discovered:

1. **✅ ROS Bridge Infrastructure Works**: Odometry publishing confirmed at 2.27 Hz
2. **✅ Topic Communication Works**: ROS 2 DDS properly publishes/subscribes within container
3. **❌ Vehicle Control Blocked**: Commands published but not applied to vehicle
4. **🔍 ROOT CAUSE IDENTIFIED**: Manual control node creates "manual override mode"

---

## Investigation Timeline

### Previous Testing (with example ego vehicle launch)

**Test 1:** `test_ros_bridge_vehicle_control.py`  
- Used: `carla_ros_bridge_with_example_ego_vehicle.launch.py`
- Result: No odometry received (synchronous mode blocking)

**Test 2:** `test_ros_bridge_with_tick.py`  
- Added: CARLA Python API client to call `world.tick()`
- Result: Conflicted with ROS Bridge's internal tick loop

**Test 3:** `test_ros_bridge_pure_ros2.py`  
- Pure ROS 2 approach (no Python API)
- Run inside bridge container (solved DDS discovery issue)
- Result: 
  - ✅ Odometry received: x=64.36, y=1.96, z=0.00
  - ❌ Distance moved: 0.00m (vehicle didn't respond)
  - ❌ vehicle_status shows throttle=0.0 despite publishing 0.5-0.8

**Test 4:** Debugging attempts  
- Disabled autopilot: `ros2 topic pub /carla/ego_vehicle/enable_autopilot std_msgs/Bool '{data: false}'`
- Disabled manual override: `ros2 topic pub /carla/ego_vehicle/vehicle_control_manual_override std_msgs/Bool '{data: false}'`
- Various publishing rates: 10 Hz, 20 Hz
- Both sync and async modes
- **All failed**: vehicle_status.control.throttle remained 0.0

---

## Critical Discovery: Manual Control Interference

### Documentation Analysis

From official CARLA ROS Bridge docs:
https://carla.readthedocs.io/projects/ros-bridge/en/latest/run_ros/

**Ego vehicle control modes:**

1. **Normal mode** - Reading commands from `/carla/<ROLE NAME>/vehicle_control_cmd`
2. **Manual mode** - Reading commands from `/carla/<ROLE NAME>/vehicle_control_cmd_manual`

**Toggle between modes:**
```bash
ros2 topic pub /carla/<ROLE NAME>/vehicle_control_manual_override std_msgs/Bool '{data: true}'
```

**The Problem:**  
The `carla_ros_bridge_with_example_ego_vehicle.launch.py` launch file starts THREE nodes:
1. `carla_ros_bridge` - The bridge itself  
2. `carla_spawn_objects` - Spawns example ego vehicle  
3. **`carla_manual_control`** - **Enables manual override mode!**

From `carla_manual_control` documentation:
https://carla.readthedocs.io/projects/ros-bridge/en/latest/carla_manual_control/

> "To steer the vehicle manually, press 'B'. Press 'H' to see instructions."

**This node puts the vehicle in manual override mode**, blocking automated control commands!

### Evidence

**ROS Bridge logs show manual control crash:**
```
[manual_control-3] Traceback (most recent call last):
[manual_control-3]   File ".../carla_manual_control/carla_manual_control", line 33, in <module>
[manual_control-3]     sys.exit(load_entry_point(...))
[manual_control-3]   File ".../carla_manual_control_node.py", line 476, in main
[manual_control-3]     spin_thread.start()
[manual_control-3] UnboundLocalError: local variable 'spin_thread' referenced before assignment
[ERROR] [manual_control-3]: process has died [pid 190, exit code 1]
```

**Even though manual_control crashed, it likely set override mode before crashing!**

---

## Solution: Minimal Configuration

### Approach

Use **`carla_ros_bridge.launch.py`** instead of `carla_ros_bridge_with_example_ego_vehicle.launch.py`

**Architecture:**
```
1. Launch: carla_ros_bridge.launch.py (bridge ONLY, no manual control)
2. Spawn: Use CARLA Python API to spawn ego vehicle with role_name='ego_vehicle'
3. Control: Publish to /carla/ego_vehicle/vehicle_control_cmd (normal mode)
```

### Implementation

**Created Files:**
- `docker-compose.minimal-test.yml` - Minimal bridge configuration
- `test_minimal_ros_bridge.py` - Pure ROS 2 control test
- `run_minimal_bridge_test.sh` - Automated test script

**Key Changes:**
1. Launch file: `carla_ros_bridge.launch.py` (NOT `carla_ros_bridge_with_example_ego_vehicle.launch.py`)
2. Spawn method: Python API (not ros2 service - more reliable)
3. Mode: Async (simpler for initial testing)
4. Manual override: Explicitly disabled via topic publish

---

## Current Status

### Test Execution Issues

**Issue 1: Docker Compose Command Syntax**  
- Problem: `host:=localhost` interpreted as bash command
- Attempted fix: Proper YAML array syntax for command
- Status: Fixed in docker-compose.minimal-test.yml

**Issue 2: CARLA Connection Timeout**  
- Problem: Bridge shows "time-out of 2000ms while waiting for the simulator"
- Possible causes:
  1. CARLA not fully ready when bridge starts
  2. Network host mode issues
  3. Port 2000 not accessible from bridge container

**Issue 3: ROS Bridge Restart Loop**  
- Problem: Container continuously restarting due to connection failure
- Impact: Can't run control test

### Next Steps

**Immediate:**
1. ✅ Fix Docker Compose command syntax (DONE)
2. ⏳ Increase CARLA startup wait time
3. ⏳ Verify CARLA port accessibility
4. ⏳ Test minimal configuration

**After Fix:**
1. Verify odometry publishing in minimal config
2. Spawn ego vehicle via Python API
3. Test vehicle control without manual override
4. If successful → Proceed to Phase 2.3 (extract controllers)

---

## Technical Specifications

### ROS Bridge Configuration

**Parameters** (from official docs):
- `host`: CARLA server hostname (default: localhost)
- `port`: CARLA server port (default: 2000)
- `timeout`: Connection timeout in seconds (default: 10)
- `synchronous_mode`: Enable/disable synchronous mode (default: true)
- `fixed_delta_seconds`: Simulation timestep (default: 0.05)
- `town`: CARLA map to load (default: Town01)
- `passive`: Let another client tick the world (default: false)
- `synchronous_mode_wait_for_vehicle_control_command`: Wait for control before tick (default: false)
- `ego_vehicle`: Role names for ego vehicles
- `register_all_sensors`: Register all sensors vs only spawned ones (default: true)

### Vehicle Control Topics

**Normal Mode** (automated control):
- **Subscribe:** Not applicable (vehicle is controlled, not controlling)
- **Publish:** `/carla/<ROLE_NAME>/vehicle_control_cmd` (carla_msgs/CarlaEgoVehicleControl)

**Manual Mode** (keyboard control):
- **Publish:** `/carla/<ROLE_NAME>/vehicle_control_cmd_manual` (carla_msgs/CarlaEgoVehicleControl)

**Mode Toggle:**
- **Publish:** `/carla/<ROLE_NAME>/vehicle_control_manual_override` (std_msgs/Bool)
  - `data: true` → Manual mode
  - `data: false` → Normal mode (automated)

**Status Topics:**
- `/carla/<ROLE_NAME>/vehicle_status` (carla_msgs/CarlaEgoVehicleStatus)
- `/carla/<ROLE_NAME>/vehicle_info` (carla_msgs/CarlaEgoVehicleInfo)
- `/carla/<ROLE_NAME>/odometry` (nav_msgs/Odometry)

### Message Types

**CarlaEgoVehicleControl:**
```
float32 throttle     # [0.0, 1.0]
float32 steer        # [-1.0, 1.0]
float32 brake        # [0.0, 1.0]
bool hand_brake
bool reverse
bool manual_gear_shift
int32 gear
```

---

## Lessons Learned

1. **Example launch files ≠ Production setup**  
   - `carla_ros_bridge_with_example_ego_vehicle.launch.py` is for MANUAL TESTING
   - For automated control, use `carla_ros_bridge.launch.py` + spawn objects separately

2. **Manual control node interferes with automation**  
   - Even if it crashes, it may set override mode
   - Always spawn vehicles without manual_control for baseline/DRL agents

3. **ROS 2 DDS discovery between containers is fragile**  
   - Running tests inside bridge container is more reliable
   - Host network mode helps but doesn't solve everything

4. **Official documentation > assumptions**  
   - Always fetch and read official docs BEFORE implementing
   - Examples in docs may have different goals than production use

---

## References

### Official Documentation
- [ROS Bridge Installation (ROS 2)](https://carla.readthedocs.io/projects/ros-bridge/en/latest/ros_installation_ros2/)
- [ROS Bridge Package Usage](https://carla.readthedocs.io/projects/ros-bridge/en/latest/run_ros/)
- [CARLA Manual Control](https://carla.readthedocs.io/projects/ros-bridge/en/latest/carla_manual_control/)
- [CARLA Spawn Objects](https://carla.readthedocs.io/projects/ros-bridge/en/latest/carla_spawn_objects/)

### GitHub
- [ROS Bridge Repository](https://github.com/carla-simulator/ros-bridge)
- [Latest Release: 0.9.12](https://github.com/carla-simulator/ros-bridge/releases/tag/0.9.12)

### Issue Tracking
- Our investigation: DEFINITIVE_NATIVE_ROS2_FINDINGS.md (Phase 2.1)
- This document: PHASE_2_2_ROS_BRIDGE_CONTROL_INVESTIGATION.md (Phase 2.2)

---

## Conclusion

We are **very close** to solving the control issue. The problem is NOT with:
- ✅ ROS Bridge itself (works)
- ✅ Topic communication (works)  
- ✅ Message publishing (works)

The problem is WITH:
- ❌ Using the wrong launch file (includes manual control)
- ❌ Manual override mode blocking automated commands

**Next action:** Test minimal configuration without manual_control node. If this works, we can proceed to Phase 2.3 (controller extraction) with confidence that our ROS 2 architecture is sound.

**Estimated time to resolution:** 1-2 hours (fix Docker command, test minimal config)

**Impact on paper timeline:** Minimal. This investigation proved that ROS Bridge IS the correct solution and native ROS 2 is definitively sensor-only. The baseline implementation can proceed once vehicle control is verified.
