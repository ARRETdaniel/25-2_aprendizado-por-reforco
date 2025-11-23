# Phase 2.2 Status Report: Native ROS 2 Investigation

**Date:** 2025-01-XX  
**Status:** 🔄 IN PROGRESS - Partial verification complete  
**Blocker:** ROS 2 topics not publishing despite `ros_name` attribute existing

---

## Summary of Findings

### ✅ CONFIRMED: Native ROS 2 Code Exists in Docker
1. **`--ros2` flag accepted** - CARLA container starts without errors
2. **`ros_name` attribute EXISTS** - Both vehicles and sensors have this blueprint attribute  
3. **Test vehicle spawned successfully** - Script completed without errors
4. **No error messages** - CARLA didn't complain about ROS 2 configuration

### ❌ ISSUE: Topics Not Publishing

**Expected topics:**
```
/carla/test_vehicle/vehicle_control_cmd
/carla/test_vehicle/front_camera/image  
/carla/clock
```

**Actual topics** (from `ros2 topic list`):
```
/clock (standard ROS 2 clock)
/parameter_events (standard ROS 2)
/rosout (standard ROS 2 logging)
```

**Conclusion:** The CARLA-specific topics are NOT being published, despite:
- Server running with `--ros2` flag  
- Vehicle spawned with `ros_name='test_vehicle'`
- Camera attached with `ros_name='front_camera'`
- No errors in CARLA logs

---

## Possible Explanations

### Hypothesis 1: DDS Domain Mismatch ⚠️ LIKELY

**Theory:** CARLA's FastDDS might be using a different DDS domain than ROS 2's default (domain 0).

**Evidence:**
- CARLA uses embedded FastDDS with custom configuration
- Previous investigation mentioned "DDS Domain 0 (default)" but this wasn't verified
- ROS 2's domain can be changed via `ROS_DOMAIN_ID` environment variable

**Test needed:**
```bash
# Try different DDS domains
for domain in {0..100}; do
    echo "Testing domain $domain"
    ROS_DOMAIN_ID=$domain ros2 topic list | grep carla && break
done
```

### Hypothesis 2: Missing FastDDS Configuration 🤔 POSSIBLE

**Theory:** CARLA's native ROS 2 might require additional configuration files or environment variables.

**Evidence:**
- No DDS discovery ports found listening (checked port range 7400-7500)
- CARLA logs don't mention ROS 2 initialization
- No FastDDS-specific environment variables set

**Test needed:**
```bash
# Check if FastDDS discovery is active
docker exec carla-server netstat -tulpn | grep -E "740[0-9]"
docker exec carla-server lsof -i | grep CARLA
```

### Hypothesis 3: Library Not Actually Compiled In ⚠️ CONCERNING

**Theory:** The `ros_name` attribute code exists in Python API, but the C++ DDS publishers were not compiled into the Docker image binaries.

**Evidence:**
- Previous investigation found `ros_name` examples in `/workspace/PythonAPI/examples/ros2/`
- But couldn't find `libcarla-ros2*.so` libraries
- `ros_name` attribute might be defined in blueprint but not functional

**Test needed:**
```bash
# Search for ROS 2-related shared libraries
docker exec carla-server find / -name "*fastdds*" -o -name "*Fast-DDS*" 2>/dev/null
docker exec carla-server ldd /home/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping | grep -i fast
```

### Hypothesis 4: Requires Explicit Enablement 🎯 MOST LIKELY

**Theory:** Spawning actors with `ros_name` isn't enough - native ROS 2 might require explicit initialization via Python API.

**Evidence:**
- The example file `ros2_native.py` might contain additional setup code we haven't seen yet
- CARLA's API might require calling a specific method to activate ROS 2 publishers
- The `--ros2` flag might only enable the capability, not activate it automatically

**Test needed:**
```bash
# Extract and analyze the official example
docker exec carla-server cat /home/carla/PythonAPI/examples/ros2/ros2_native.py
```

---

## Recommended Next Steps

### Priority 1: Analyze Official Example (IMMEDIATE)

Extract and study the `ros2_native.py` example that ships with CARLA:

```bash
# Copy example from container
docker cp carla-server:/home/carla/PythonAPI/examples/ros2/ ./carla_ros2_examples/

# Study the code to understand:
# 1. How ROS 2 is initialized
# 2. What additional API calls are needed
# 3. What configuration is required
```

### Priority 2: Check for FastDDS Binaries (CRITICAL)

Verify if FastDDS is actually compiled into the CARLA binaries:

```bash
# Search for FastDDS in binary
docker exec carla-server strings /home/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping | grep -i "fastdds\|fast.dds\|dds" | head -20

# Check linked libraries
docker exec carla-server ldd /home/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping | grep -i fast
```

**If FastDDS NOT found:** Native ROS 2 is NOT in Docker → Must use external bridge  
**If FastDDS found:** Configuration or initialization issue → Solvable

### Priority 3: Test DDS Discovery (DIAGNOSTIC)

Check if DDS discovery is working at all:

```bash
# Install FastDDS discovery tool in separate container
docker run --rm --net=host \
  -it ubuntu:20.04 bash -c \
  "apt-get update && apt-get install -y fastdds-tools && \
   fastdds discovery"
```

### Priority 4: Decision Point (BLOCKING)

Based on findings from steps 1-3, choose path:

**Path A: Native ROS 2 Works (after fixing configuration)**
→ Proceed with unified container approach  
→ Extract PID+Pure Pursuit controllers  
→ Create ROS 2 baseline node  
→ Test control loop

**Path B: Native ROS 2 Doesn't Work in Docker**
→ Revert to external bridge approach  
→ Fix path issues in ros2-carla-bridge Dockerfile  
→ Use 3-container architecture  
→ Proceed with baseline implementation

---

## Technical Details for Investigation

### CARLA Container Details
```
Container: carla-server
Image: carlasim/carla:0.9.16
Status: Running
Command: bash CarlaUE4.sh --ros2 -RenderOffScreen -nosound
Network: host
Runtime: nvidia
```

### Test Vehicle Details
```
Vehicle ID: 26
Blueprint: vehicle.lincoln.mkz_2020
ros_name: test_vehicle (attribute successfully set)
Spawn Status: Success
```

### Camera Sensor Details
```
Camera ID: 27
Blueprint: sensor.camera.rgb
ros_name: front_camera (attribute successfully set)
Resolution: 800x600
Attachment: Success (attached to vehicle ID 26)
```

### ROS 2 Environment
```
Distribution: Humble
Container: ros:humble-ros-core (used for topic listing)
Network: host
DDS Domain: 0 (default, not explicitly set)
```

---

## Questions to Answer

1. **Is FastDDS actually compiled into carlasim/carla:0.9.16?**
   - Check binary strings for FastDDS references
   - Check linked libraries
   - Search for .so files

2. **What does the official ros2_native.py example do?**
   - Extract and analyze the example code
   - Look for initialization calls
   - Check for configuration requirements

3. **Is DDS discovery working?**
   - Are discovery ports open?
   - Can we see DDS participants?
   - Is the domain ID correct?

4. **Do we need additional Python API calls?**
   - Is spawning with `ros_name` sufficient?
   - Do we need to explicitly start publishers?
   - Is there a ROS 2 manager class to instantiate?

---

## Timeline Estimate

**If Native ROS 2 works:** 2-3 hours to fix configuration + 4-6 hours for baseline implementation = **1 day total**

**If we must use external bridge:** 3-4 hours to fix bridge Dockerfile + 6-8 hours for baseline = **2 days total**

**Current blocker:** Need 30-60 minutes for diagnostic investigation before choosing path.

---

## User Decision Point

**Question for user:** Should we:

**Option A:** Spend 1-2 hours investigating FastDDS/native ROS 2 configuration to see if we can make it work?
- **Pro:** If it works, best performance and simplest architecture
- **Con:** Might be a dead end if libraries aren't actually in Docker
- **Risk:** Could waste time if it's not fixable

**Option B:** Switch immediately to external bridge approach (known to work)?
- **Pro:** Proven solution, well-documented, can start implementation now
- **Con:** Slightly higher latency, one more container
- **Risk:** Lower risk, guaranteed to work

**Recommendation:** **Option A** - Spend 1 hour on investigation first. The `ros_name` attribute existing is a strong signal that native ROS 2 is there, we just need to figure out the correct configuration. If investigation fails after 1 hour, switch to Option B.

---

## Next Action

**Execute diagnostic investigation:**

1. Extract official ROS 2 example (5 min)
2. Analyze example code (10 min)
3. Check for FastDDS in binaries (10 min)
4. Test DDS discovery (15 min)
5. Make decision: native ROS 2 or external bridge (5 min)

**Total time investment:** 45 minutes before committing to approach.

---

**Status:** Awaiting user input on investigation vs. immediate bridge implementation.
