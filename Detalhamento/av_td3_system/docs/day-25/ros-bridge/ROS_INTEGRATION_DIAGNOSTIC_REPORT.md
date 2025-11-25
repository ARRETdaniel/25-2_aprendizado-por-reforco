# ROS 2 Integration Diagnostic Report

**Date**: 2025-01-22
**Status**: 🔍 DIAGNOSIS IN PROGRESS
**Objective**: Diagnose and fix ROS Bridge vehicle control failure

---

## Executive Summary

### Current Situation

✅ **Working**:
- CARLA 0.9.16 Docker server running
- Baseline evaluation implementation complete (all metrics)
- ROS 2 Humble + CARLA Python compatibility resolved

⚠️ **Blocked**:
- ROS Bridge cannot control CARLA vehicle
- Baseline controller cannot integrate with ROS 2 ecosystem

### Root Cause Hypothesis

Based on official documentation review, the most likely root cause is:

**❌ CARLA server running with `--ros2` flag (native ROS 2 mode) instead of standard mode required by ROS Bridge**

Evidence:
- `commands.md` shows: `docker run ... CarlaUE4.sh --ros2 -RenderOffScreen`
- Official docs state: "Start CARLA server: `./CarlaUE4.sh` (NO --ros2 flag for bridge!)"
- Native ROS 2 and ROS Bridge are **mutually exclusive** systems

---

## Understanding the Two ROS 2 Systems

### System 1: Native ROS 2 (Built into CARLA 0.9.16)

**What it is:**
- Built-in FastDDS integration in CARLA binaries
- Enabled via `--ros2` flag when launching CARLA
- No separate installation needed
- Direct C++/Python integration with DDS middleware

**How to use:**
```bash
# Start CARLA with native ROS 2
docker run --rm --runtime=nvidia --net=host \
  carlasim/carla:0.9.16 bash CarlaUE4.sh --ros2 -RenderOffScreen

# In Python code
sensor.enable_for_ros()  # Enables ROS 2 publisher for sensor
```

**Capabilities (from release notes):**
> "Connect CARLA directly to ROS2... with sensor streams **and ego control**"

**What We Know:**
- ✅ Sensors work (confirmed in previous tests)
- ❓ Vehicle control capability **unconfirmed** (needs testing)
- 📝 Minimal official documentation available
- 🔍 Example exists: `/workspace/PythonAPI/examples/ros2/ros2_native.py`

**When to use:**
- If you want lowest latency (no bridge middleware)
- If native control capability is confirmed
- If minimal dependencies preferred

---

### System 2: ROS Bridge (External Package)

**What it is:**
- Separate GitHub repository: https://github.com/carla-simulator/ros-bridge
- Python package that connects via CARLA Python API (port 2000)
- Must be installed separately
- Translates between CARLA and ROS 2 messages

**How to use:**
```bash
# Start CARLA in STANDARD mode (NO --ros2 flag!)
docker run --rm --runtime=nvidia --net=host \
  carlasim/carla:0.9.16 bash CarlaUE4.sh -RenderOffScreen

# Export CARLA Python API
export CARLA_ROOT=/opt/carla-simulator
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.16-py3.10-linux-x86_64.egg

# Launch bridge
ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py
```

**Capabilities:**
- ✅ Complete ROS 2 message definitions (`carla_msgs`)
- ✅ Vehicle control via `/carla/ego_vehicle/vehicle_control_cmd`
- ✅ Multiple control interfaces (direct, Ackermann, Twist)
- ✅ Services for spawning/destroying objects
- ✅ Fully documented and supported

**What We Know:**
- ✅ Comprehensive documentation available
- ✅ Well-tested across CARLA versions
- ⚠️ Official support for 0.9.13 only (we have 0.9.16)
- ✅ Workaround applied: Patched CARLA_VERSION file
- ✅ Python 3.10 compatibility resolved (ROS 2 Humble)

**When to use:**
- Need complete ROS 2 ecosystem integration
- Want standard ROS message types
- Need documented, tested solution
- Require services (spawn, destroy, etc.)

---

## Critical Incompatibility

**⚠️ THESE TWO SYSTEMS CANNOT RUN SIMULTANEOUSLY**

| Aspect | Native ROS 2 | ROS Bridge |
|--------|--------------|------------|
| **CARLA Launch Flag** | `--ros2` | *No flag* (standard mode) |
| **Connection Method** | Built-in DDS | Python API (port 2000) |
| **Topic Naming** | `/carla/<ros_name>/<sensor_type>` | `/carla/<role>/<topic>` |
| **Control Method** | `vehicle.enable_for_ros()` (?) | Topic `/carla/ego_vehicle/vehicle_control_cmd` |
| **Installation** | Built-in | External package required |
| **Documentation** | Minimal | Comprehensive |

**You MUST choose one or the other based on testing results.**

---

## Current Setup Analysis

### From `commands.md`

**Current CARLA Server Command:**
```bash
docker run -d --name carla-server --runtime=nvidia --net=host \
  --env=NVIDIA_VISIBLE_DEVICES=all \
  --env=NVIDIA_DRIVER_CAPABILITIES=all \
  carlasim/carla:0.9.16 bash CarlaUE4.sh --ros2 -RenderOffScreen -nosound
```

**Problem Identified:**
```
--ros2 flag present ← This enables NATIVE ROS 2 mode
                     ← ROS Bridge CANNOT connect to this!
```

### ROS Bridge v4 Build Status

**From BRIDGE_VERSION_COMPATIBILITY.md:**
- Build objective: Patch CARLA_VERSION from "0.9.13" to "0.9.16"
- Status: Build was in progress (estimated 15-20 minutes)
- Method: Direct file modification in Dockerfile
- Expected image: `ros2-carla-bridge:humble-v4`

**Unknown:**
- ❓ Did build complete successfully?
- ❓ Was image created?
- ❓ Is it ready for testing?

**Verification needed:**
```bash
docker images | grep ros2-carla-bridge:humble-v4
```

---

## Decision Path

### Test 1: Native ROS 2 Vehicle Control

**Objective:** Determine if native ROS 2 (--ros2 flag) supports vehicle control

**Method:**
1. Keep CARLA server with `--ros2` flag
2. Examine `/workspace/PythonAPI/examples/ros2/ros2_native.py` inside CARLA container
3. Test if `vehicle.enable_for_ros()` method exists
4. Search for control topic naming convention
5. Attempt to publish control commands

**Commands:**
```bash
# Access CARLA container
docker exec -it carla-server bash

# Find native ROS 2 example
find /workspace -name "*ros2_native.py"
cat /workspace/PythonAPI/examples/ros2/ros2_native.py

# Check for vehicle control references
grep -r "enable_for_ros" /workspace/PythonAPI/
grep -r "vehicle.*control.*ros" /workspace/PythonAPI/
```

**Expected Outcomes:**

**Scenario A: Native Control Works ✅**
- Find `vehicle.enable_for_ros()` method in example
- Identify control topic naming convention
- Can publish control commands successfully

**Action:** Proceed with pure native ROS 2 architecture
- Simpler (2 containers: CARLA + Baseline)
- Lower latency
- Less dependencies

---

**Scenario B: Native Control Doesn't Work ❌**
- No `vehicle.enable_for_ros()` method found
- Example only shows sensor publishing
- Cannot control vehicle via topics

**Action:** Switch to ROS Bridge architecture
- Fix CARLA server mode (remove `--ros2` flag)
- Verify ROS Bridge v4 build
- Test bridge connection

---

### Test 2: ROS Bridge Connection (If Native Fails)

**Prerequisites:**
- CARLA server running in **STANDARD MODE** (NO `--ros2` flag)
- ROS Bridge v4 image built successfully
- PYTHONPATH configured correctly

**Updated CARLA Command:**
```bash
# CORRECTED (for ROS Bridge)
docker run -d --name carla-server --runtime=nvidia --net=host \
  --env=NVIDIA_VISIBLE_DEVICES=all \
  --env=NVIDIA_DRIVER_CAPABILITIES=all \
  carlasim/carla:0.9.16 bash CarlaUE4.sh -RenderOffScreen -nosound
  # ↑ NO --ros2 flag!
```

**Launch ROS Bridge:**
```bash
docker run --rm --net=host \
  -e CARLA_ROOT=/opt/carla-simulator \
  -e PYTHONPATH=/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.16-py3.10-linux-x86_64.egg \
  ros2-carla-bridge:humble-v4 \
  ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py \
  host:=localhost port:=2000
```

**Verification Steps:**
```bash
# 1. Check bridge started successfully
docker logs <bridge_container_id>
# Should show: "ROS Bridge successfully connected to CARLA"

# 2. Verify CARLA Python import
docker exec <bridge_container_id> python3 -c 'import carla; print(carla.__version__)'
# Should show: 0.9.16

# 3. Check ROS 2 topics
docker exec <bridge_container_id> ros2 topic list | grep carla
# Should show:
#   /carla/ego_vehicle/vehicle_control_cmd
#   /carla/ego_vehicle/vehicle_status
#   /carla/ego_vehicle/odometry

# 4. Test control command
docker exec <bridge_container_id> ros2 topic pub --once \
  /carla/ego_vehicle/vehicle_control_cmd \
  carla_msgs/msg/CarlaEgoVehicleControl \
  "{throttle: 0.5, steer: 0.0}"
# Vehicle should move in CARLA
```

---

## Recommended Action Plan

### Phase 1: Test Native ROS 2 (30 minutes)

**Priority: HIGH** - Simplest solution if it works

1. ✅ Keep current CARLA setup (with `--ros2` flag)
2. 🔍 Access CARLA container
3. 📖 Read `/workspace/PythonAPI/examples/ros2/ros2_native.py`
4. 🧪 Test vehicle control capability
5. 📝 Document findings

**Success Criteria:**
- Vehicle control via native ROS 2 confirmed working
- Topic naming convention identified
- Can publish control commands

**If successful:** Create native ROS 2 architecture
**If unsuccessful:** Proceed to Phase 2

#### ROS 2 native seem to be working but only for sensor and autopilet
---

### Phase 2: Fix ROS Bridge Setup (1 hour)

**Prerequisites:**
- Native ROS 2 control confirmed NOT working
- ROS Bridge v4 build verified

**Steps:**

1. **Stop Current CARLA Server:**
   ```bash
   docker stop carla-server
   docker rm carla-server
   ```

2. **Start CARLA in Standard Mode:**
   ```bash
   docker run -d --name carla-server --runtime=nvidia --net=host \
     carlasim/carla:0.9.16 bash CarlaUE4.sh -RenderOffScreen -nosound
   # NO --ros2 flag!
   ```

3. **Verify ROS Bridge Image:**
   ```bash
   docker images | grep ros2-carla-bridge:humble-v4
   # Should show image with ~4GB size
   ```

4. **Test Bridge Connection:**
   - Launch bridge container
   - Verify CARLA Python import
   - Check ROS topics
   - Test vehicle control

5. **Document Working Setup:**
   - Create docker-compose.baseline-integration.yml
   - Write setup guide
   - Create troubleshooting section

**Success Criteria:**
- Bridge connects to CARLA successfully
- ROS topics published
- Vehicle control commands work
- No version errors

---

### Phase 3: Create Docker Compose (30 minutes)

**Based on successful architecture from Phase 1 or 2**

**For Native ROS 2:**
```yaml
version: '3.8'
services:
  carla-server:
    image: carlasim/carla:0.9.16
    command: bash CarlaUE4.sh --ros2 -RenderOffScreen
    runtime: nvidia
    network_mode: host

  baseline-controller:
    build: ./docker/baseline-controller.Dockerfile
    depends_on:
      - carla-server
    network_mode: host
```

**For ROS Bridge:**
```yaml
version: '3.8'
services:
  carla-server:
    image: carlasim/carla:0.9.16
    command: bash CarlaUE4.sh -RenderOffScreen  # NO --ros2!
    runtime: nvidia
    network_mode: host

  ros-bridge:
    image: ros2-carla-bridge:humble-v4
    depends_on:
      - carla-server
    environment:
      - CARLA_ROOT=/opt/carla-simulator
      - PYTHONPATH=/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.16-py3.10-linux-x86_64.egg
    network_mode: host
    command: ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py

  baseline-controller:
    build: ./docker/baseline-controller.Dockerfile
    depends_on:
      - ros-bridge
    network_mode: host
```

---

## Known Issues & Workarounds

### Issue 1: Version Compatibility (0.9.16 vs 0.9.13)

**Status:** Patched in ROS Bridge v4 build
**Workaround:** CARLA_VERSION file updated to "0.9.16"
**Risk:** Potential API incompatibilities (needs testing)

**Mitigation:**
- Comprehensive testing of all bridge features
- Monitor for runtime errors
- Fallback to native Python API if critical issues

### Issue 2: Python Compatibility (3.8 vs 3.10)

**Status:** ✅ Resolved
**Solution:** Upgraded to ROS 2 Humble (Python 3.10)
**Verification:** CARLA wheel imports successfully

### Issue 3: Server Mode Confusion

**Status:** ⚠️ Current blocker
**Problem:** Running `--ros2` mode with ROS Bridge attempt
**Solution:** Choose one system:
- Native ROS 2 → Keep `--ros2` flag
- ROS Bridge → Remove `--ros2` flag

---

## Success Metrics

### Baseline Controller Integration

**Required Functionality:**
1. ✅ CARLA simulation running
2. ✅ ROS 2 topics publishing (sensors, odometry)
3. ✅ Can subscribe to sensor data
4. ✅ Can publish control commands
5. ✅ Vehicle responds to commands
6. ✅ Stable operation for 20+ episodes
7. ✅ Metrics collection working

**Performance Targets:**
- Topic frequency: ~20 Hz
- Control latency: < 100ms
- Zero crashes during operation
- Reproducible results across runs

---

## Next Actions

1. **IMMEDIATE**: Test native ROS 2 vehicle control capability
2. **IF NEEDED**: Fix CARLA server mode for ROS Bridge
3. **VERIFY**: ROS Bridge v4 build status
4. **TEST**: End-to-end vehicle control
5. **DOCUMENT**: Working architecture and setup guide
6. **INTEGRATE**: Baseline controller with ROS 2

---

## References

### Official Documentation
- [CARLA 0.9.16 Release](https://carla.org/2025/09/16/release-0.9.16/)
- [ROS Bridge Installation](https://carla.readthedocs.io/projects/ros-bridge/en/latest/ros_installation_ros2/)
- [ROS Bridge Running](https://carla.readthedocs.io/projects/ros-bridge/en/latest/run_ros/)
- [CARLA Spawn Objects](https://carla.readthedocs.io/projects/ros-bridge/en/latest/carla_spawn_objects/)

### Project Documentation
- `BRIDGE_VERSION_COMPATIBILITY.md` - Version mismatch analysis
- `PYTHON_COMPATIBILITY_ISSUE.md` - Python 3.10 solution
- `CORRECTED_INVESTIGATION.md` - Native vs Bridge clarification
- `README.md` - Docker setup instructions

---

**Document Status**: 📋 Diagnostic Phase Complete
**Next Step**: Execute Test Phase 1 (Native ROS 2)
**Expected Time**: 30-90 minutes to resolution
