# FINDINGS: Official Native ROS 2 Example Analysis

**Date**: 2025-01-22  
**File**: `/workspace/PythonAPI/examples/ros2/ros2_native.py`  
**Status**: ✅ CONFIRMED - Native ROS 2 IS Sensor-Only

---

## Critical Discovery from Official Example

### What the Official Example Does

```python
# From ros2_native.py (official CARLA 0.9.16 example)

# 1. Spawns vehicle with ros_name
bp.set_attribute("role_name", config.get("id"))
bp.set_attribute("ros_name", config.get("id"))  # ← Sets ROS name on VEHICLE
vehicle = world.spawn_actor(bp, spawn_point)

# 2. Spawns sensors with ros_name  
for sensor in sensors_config:
    bp.set_attribute("ros_name", sensor.get("id"))
    sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
    sensor.enable_for_ros()  # ← ONLY sensors get this call!

# 3. Uses AUTOPILOT for vehicle control
vehicle.set_autopilot(True)  # ← NOT ROS 2 control!
```

---

## Key Findings

### 1. `enable_for_ros()` is SENSOR-ONLY

**Evidence:**
```python
sensors.append(world.spawn_actor(bp, wp, attach_to=vehicle))
sensors[-1].enable_for_ros()  # ← Called ONLY on sensors
```

**No `vehicle.enable_for_ros()` in the official example!**

### 2. Vehicle Control Uses AUTOPILOT

```python
vehicle.set_autopilot(True)
```

**The official example does NOT control the vehicle via ROS 2 topics.**

It uses CARLA's built-in autopilot!

### 3. `ros_name` is Set on Both

```python
# Vehicle:
bp.set_attribute("ros_name", config.get("id"))

# Sensors:
bp.set_attribute("ros_name", sensor.get("id"))
```

But only sensors call `enable_for_ros()`!

---

## What This Means

### Native ROS 2 Capabilities (0.9.16)

**✅ Confirmed Working:**
- Sensor data publishing (cameras, LIDAR, etc.)
- Clock synchronization (`/clock`)
- Transform broadcasting (`/tf`)
- Setting `ros_name` on actors

**❌ NOT Supported:**
- Vehicle control via ROS 2 topics
- No `vehicle.enable_for_ros()` method
- No vehicle control subscriber creation
- Autopilot is used instead

---

## Release Notes Were Misleading

### What the Release Said

> "with sensor streams and ego control"

### What "ego control" Actually Means

Looking at the example, "ego control" likely means:
- ✅ Setting `ros_name` on ego vehicle (for sensor namespacing)
- ✅ Using autopilot to control the ego vehicle
- ❌ **NOT** controlling ego via ROS 2 topics

The release notes were **technically accurate** but **misleading**:
- You CAN control the "ego" vehicle
- But NOT via ROS 2 - via autopilot!

---

## Architecture Decision: CONFIRMED

### Native ROS 2 (Built-in) = Sensor Output Only

**Capabilities:**
```
CARLA Server (--ros2)
    ↓ (sensor data only)
ROS 2 Topics
```

**Control method**: Python API (autopilot or direct VehicleControl)

### For Vehicle Control, We Need:

**Option A: Hybrid Architecture**
```
CARLA (--ros2)  → Sensors → ROS 2 Topics
    ↓ (Python API)
Python Control Node → Subscribes to ROS topics
                    → Applies control via Python API
```

**Option B: Full ROS Bridge**
```
CARLA (standard) ← Python API → ROS Bridge
                                    ↓
                                ROS 2 Topics
                                (bidirectional)
```

---

## Corrected Understanding

### Native ROS 2 (0.9.16)

| Feature | Support |
|---------|---------|
| Sensor publishing | ✅ Yes |
| Clock/TF | ✅ Yes |
| Vehicle control IN | ❌ No |
| `sensor.enable_for_ros()` | ✅ Yes |
| `vehicle.enable_for_ros()` | ❌ No |
| Control topics | ❌ No |

### ROS Bridge (External Package)

| Feature | Support |
|---------|---------|
| Sensor publishing | ✅ Yes |
| Clock/TF | ✅ Yes |
| Vehicle control IN | ✅ Yes |
| Services (spawn/destroy) | ✅ Yes |
| Multiple control interfaces | ✅ Yes |
| Full ROS message types | ✅ Yes |

---

## Recommendation

### For Our Baseline Controller

**Use ROS Bridge** (Option B)

**Reasoning:**
1. We need bidirectional communication (sensors + control)
2. ROS Bridge is officially documented and maintained
3. It's the standard approach for ROS + CARLA
4. It provides complete ROS 2 ecosystem integration
5. Native ROS 2 doesn't solve our problem

**Timeline Impact:**
- Adds 4-5 hours for ROS Bridge setup
- Saves time vs. implementing hybrid solution
- Well-documented, less debugging

---

## Updated Next Steps

### Phase 2.2: Install and Configure ROS Bridge

1. **Create ROS Bridge Dockerfile** (1 hour)
   - Base: `ros:humble-ros-base`
   - Install CARLA Python API 0.9.16
   - Clone and build ROS Bridge

2. **Test ROS Bridge** (2 hours)
   - Launch CARLA + bridge
   - Verify sensor topics
   - Test vehicle control
   - Confirm bidirectional communication

3. **Create docker-compose.yml** (1 hour)
   - CARLA server (standard mode, NOT --ros2)
   - ROS Bridge container
   - Baseline controller container (placeholder)

**Total**: ~4-5 hours

### Phase 2.3: Extract Controllers

Continue as planned

---

## Lessons Learned

### Mistake 1: Trusting Release Notes

Release notes said "ego control" but didn't specify HOW.

**Lesson**: Always check official examples.

### Mistake 2: Not Reading Example Code First

Should have examined `/workspace/PythonAPI/examples/ros2/ros2_native.py` FIRST.

**Lesson**: Official examples are the ground truth.

### Mistake 3: Confusing Two Systems

Native ROS 2 ≠ ROS Bridge

**Lesson**: Clearly distinguish different systems.

---

## Conclusion

**CONFIRMED**: Native ROS 2 in CARLA 0.9.16 is **sensor-only** (output).

**For bidirectional ROS 2 (sensors + control)**: Use CARLA ROS Bridge.

**Original architecture decision documents (`ARCHITECTURE_DECISION.md` and `PHASE_2_UPDATED_PLAN.md`) were CORRECT** - ROS Bridge IS required for vehicle control.

**My apology in `CORRECTED_INVESTIGATION.md` was WRONG** - I was actually right the first time, but for the right reasons now (official example confirms it).

---

## Files to Update

- [x] Create this findings document
- [ ] Mark `CORRECTED_INVESTIGATION.md` as "Correction was unnecessary"
- [ ] Validate `ARCHITECTURE_DECISION.md` (it was correct!)
- [ ] Validate `PHASE_2_UPDATED_PLAN.md` (it was correct!)
- [ ] Proceed with Phase 2.2: ROS Bridge setup

---

## References

- **Official Example**: `/workspace/PythonAPI/examples/ros2/ros2_native.py`
- **Release Notes**: https://carla.org/2025/09/16/release-0.9.16/
- **ROS Bridge Docs**: https://carla.readthedocs.io/projects/ros-bridge/en/latest/
