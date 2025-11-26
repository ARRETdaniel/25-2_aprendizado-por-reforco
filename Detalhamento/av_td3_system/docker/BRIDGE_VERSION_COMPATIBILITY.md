# CARLA ROS Bridge Version Compatibility Analysis

Python Version Check:

CARLA 0.9.16: Provides wheels for Python 3.10, 3.11, 3.12 (as per official release)
ROS 2 Foxy: Requires Python 3.8 (Ubuntu 20.04 default)
ROS 2 Humble: Requires Python 3.10 (Ubuntu 22.04 default)
Our current setup:

Base image: CARLA 0.9.16 (Ubuntu 20.04 Focal)
Python: 3.10 (installed via Miniforge)
ROS 2: Trying to install Foxy (requires Python 3.8) THIS WON'T WORK!


CARLA 0.9.16: Requires Python 3.10/3.11/3.12 (no 3.8 wheels)
ROS 2 Foxy: Requires Python 3.8 (Ubuntu 20.04 default)
ROS 2 Humble: Requires Python 3.10 + Ubuntu 22.04
Root Cause: Our container uses Python 3.10 (for CARLA), but base image is Ubuntu 20.04 (which only supports ROS 2 Foxy/Python 3.8).


## Problem Summary

**Issue**: The official CARLA ROS Bridge repository does NOT support CARLA 0.9.16.
**Impact**: Bridge fails to start with fatal error: "CARLA python module version 0.9.13 required. Found: 0.9.16"

## Root Cause Analysis

### Timeline Investigation

| CARLA Version | Release Date | Bridge Release | Status |
|---------------|--------------|----------------|--------|
| 0.9.16 | September 2025 | None | **NOT SUPPORTED** |
| 0.9.15 | N/A | None | **NOT SUPPORTED** |
| 0.9.14 | N/A | None | **NOT SUPPORTED** |
| **0.9.13** | **2022** | **0.9.12 (July 22, 2022)** | **Last Official Release** |
| 0.9.12 | 2021 | 0.9.11 | Supported |
| 0.9.11 | 2021 | 0.9.10.1 | Supported |

**Key Finding**: The ros-bridge repository has been **inactive for 2.5+ years**. Last commit was July 22, 2022, updating support to CARLA 0.9.13. **No releases exist for CARLA 0.9.14, 0.9.15, or 0.9.16**.

### Documentation Analysis

**Official CARLA Documentation** (carla.readthedocs.io/projects/ros-bridge/):
- States: "CARLA 0.9.11 or later"
- Recommends: "Match the ROS bridge version to the CARLA version when possible"
- **Does NOT mention 0.9.16 support**

**GitHub Repository** (github.com/carla-simulator/ros-bridge):
- Latest release: 0.9.12 (July 22, 2022)
- README: "This version requires CARLA 0.9.13"
- **No tags for 0.9.14, 0.9.15, or 0.9.16**
- Master branch: No commits since July 2022
- **No community issues/PRs for 0.9.16**

### Technical Details

**Version Check Mechanism**:
```python
# File: carla_ros_bridge/src/carla_ros_bridge/bridge.py (line ~410)

# Read expected version from file
with open(os.path.join(os.path.dirname(__file__), "CARLA_VERSION")) as f:
    CARLA_VERSION = f.read()[:-1]  # Returns "0.9.13"

# Check installed CARLA version
dist = pkg_resources.get_distribution("carla")
if LooseVersion(dist.version) != LooseVersion(CarlaRosBridge.CARLA_VERSION):
    carla_bridge.logfatal("CARLA python module version {} required. Found: {}".format(
        CarlaRosBridge.CARLA_VERSION, dist.version))
    sys.exit(1)  # FATAL EXIT
```

**CARLA_VERSION File Content**:
```bash
$ cat carla_ros_bridge/src/carla_ros_bridge/CARLA_VERSION
0.9.13
```

**Error Message**:
```
[FATAL] [1763829495.451446635] [carla_ros_bridge]:
CARLA python module version 0.9.13 required. Found: 0.9.16

Traceback:
  File ".../carla_ros_bridge/bridge.py", line 419, in main
    sys.exit(1)
  ...
  AttributeError: 'CarlaRosBridge' object has no attribute 'shutdown'
```

## Solution Options Analysis

### Option 1: Downgrade to CARLA 0.9.13 ❌ **NOT VIABLE**

**Pros:**
- Guaranteed bridge compatibility
- Matches bridge's expected version

**Cons:**
- **Paper specifies CARLA 0.9.16** (cannot change)
- Lose 0.9.16 features and improvements
- Python compatibility issues (0.9.13 may not have Python 3.10 wheels)

**Verdict**: **REJECTED** - Paper requirements mandate 0.9.16

### Option 2: Search for Community Fork 🟡 **INVESTIGATED**

**Search Results:**
- GitHub search: No active forks with 0.9.16 support
- No community issues discussing 0.9.16
- No pull requests targeting 0.9.16

**Verdict**: **NO VIABLE FORKS FOUND**

### Option 3: Use Native Python API Instead of ROS Bridge 🟡 **ALTERNATIVE**

**Pros:**
- Direct CARLA API control (no middleware)
- Guaranteed 0.9.16 compatibility
- Simpler architecture

**Cons:**
- **Violates paper architecture** (requires ROS 2)
- More implementation work
- Lose ROS ecosystem benefits

**Verdict**: **FALLBACK OPTION** (only if bridge patch fails)

### Option 4: Patch Version Check ✅ **SELECTED SOLUTION**

**Approach**: Update `CARLA_VERSION` file from "0.9.13" to "0.9.16"

**Implementation**:
```dockerfile
# In Dockerfile after cloning bridge repository
RUN cd src/ros-bridge/carla_ros_bridge/src/carla_ros_bridge && \
  echo "Patching CARLA version requirement from 0.9.13 to 0.9.16" && \
  echo "0.9.16" > CARLA_VERSION && \
  cat CARLA_VERSION  # Verify: outputs "0.9.16"
```

**Pros:**
- ✅ Simple, minimal change (1 file, 1 line)
- ✅ No code logic modification
- ✅ Preserves ROS 2 architecture
- ✅ Fast to implement and test
- ✅ Meets paper requirements

**Cons:**
- ⚠️ **Untested configuration** (no official validation)
- ⚠️ **Potential API incompatibilities** between 0.9.13 and 0.9.16
- ⚠️ **No upstream support** if bugs occur

**Risk Mitigation**:
1. **Comprehensive Testing**:
   - Test all bridge features (sensors, control, topics)
   - Verify topic publication rates
   - Test synchronous mode
   - Stress test with NPCs
2. **API Compatibility Check**:
   - Review CARLA 0.9.16 changelog for breaking changes
   - Test sensor APIs (camera, LIDAR, IMU)
   - Verify vehicle control commands
3. **Fallback Plan**:
   - If critical issues found → Option 3 (Native Python API)
   - Document all issues for paper discussion

**Verdict**: **APPROVED FOR IMPLEMENTATION**

## Implementation Log

### Build v4 (Current)

**Date**: 2025-01-XX
**Objective**: Apply version patch and rebuild bridge

**Changes**:
1. Created `CARLA_VERSION` update step in Dockerfile
2. Removed failed patch file approach
3. Build command:
   ```bash
   DOCKER_BUILDKIT=1 docker build \
     -t "ros2-carla-bridge:humble-v4" \
     -f "av_td3_system/docker/ros2-carla-bridge.Dockerfile" \
     --build-arg ROS_DISTRO="humble" \
     --build-arg CARLA_VERSION="0.9.16" \
     --no-cache \
     --progress=plain . \
     2>&1 | tee av_td3_system/docker/build_log_humble_v4_clean.log
   ```

**Status**: ⏳ BUILD IN PROGRESS (estimated 15-20 minutes)

**Expected Outcome**:
- ✅ Bridge compiles without version errors
- ✅ Image builds successfully (~4GB)
- ✅ CARLA Python module imports
- ⏳ Runtime verification pending

### Next Steps

1. **Wait for build completion**
2. **Verify build success**:
   ```bash
   docker images | grep ros2-carla-bridge:humble-v4
   ```
3. **Test bridge startup**:
   ```bash
   # Ensure CARLA server running
   docker ps | grep carla

   # Launch bridge
   docker run --rm --net=host ros2-carla-bridge:humble-v4 \
     ros2 launch carla_ros_bridge carla_ros_bridge.launch.py \
     host:=localhost port:=2000
   ```
4. **Verify topics published**:
   ```bash
   docker exec <container_id> ros2 topic list | grep carla
   docker exec <container_id> ros2 topic echo /carla/status --once
   ```
5. **Test sensor data**:
   ```bash
   ros2 topic echo /carla/ego_vehicle/camera/rgb/front/image --once
   ros2 topic echo /carla/ego_vehicle/odometry --once
   ```
6. **Test control commands**:
   ```bash
   ros2 topic pub /carla/ego_vehicle/vehicle_control_cmd \
     carla_msgs/msg/CarlaEgoVehicleControl \
     '{throttle: 0.5, steer: 0.0, brake: 0.0}'
   ```
7. **Document any issues**

## API Compatibility Concerns

### Potential Breaking Changes (0.9.13 → 0.9.16)

**Need to verify**:
- Sensor API changes (camera, LIDAR formats)
- Vehicle control command structure
- World/Map API modifications
- Traffic manager updates
- Synchronous mode behavior

**Mitigation**:
- Read CARLA 0.9.14, 0.9.15, 0.9.16 changelogs
- Cross-reference bridge code with new APIs
- Patch bridge if necessary

### Known CARLA 0.9.16 Features

From release notes (https://carla.org/2025/09/16/release-0.9.16/):
- ✅ ROS 2 support: "Foxy, Galactic, Humble and more"
- ✅ Python 3.10, 3.11, 3.12 wheels
- New sensor capabilities
- Performance improvements

**Good News**: Official confirmation of ROS 2 Humble support suggests bridge compatibility likely.

## Conclusion

**Current Status**: **WORKAROUND IN PROGRESS**

**Solution**: Patching `CARLA_VERSION` file to allow 0.9.16

**Confidence Level**: **MEDIUM-HIGH**
- ✅ Simple, low-risk approach
- ✅ Official ROS 2 Humble support confirmed
- ⚠️ Untested configuration (no official 0.9.16 bridge)
- ⚠️ Requires comprehensive testing

**Success Criteria**:
1. Bridge starts without fatal errors
2. All ROS 2 topics publish correctly
3. Control commands work bidirectionally
4. No crashes during extended operation

**Fallback**: If critical issues arise, implement native Python API integration without ROS 2 bridge (Option 3).

---

**Document Version**: 1.0
**Last Updated**: 2025-01-XX
**Author**: AI Assistant + User
**Status**: Active Development
