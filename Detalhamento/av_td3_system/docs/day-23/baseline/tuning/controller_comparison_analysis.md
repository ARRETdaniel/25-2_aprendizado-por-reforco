# Controller Implementation Comparison Analysis

**Date**: 2025-01-23  
**Issue**: Baseline controller exhibits zigzag behavior  
**Root Cause**: Using Stanley controller instead of Pure Pursuit

---

## Summary

The current baseline implementation uses the **Stanley controller** from Course 4 Final Project, while the working GitHub implementation uses **Pure Pursuit** from Course 1 Final Project. These are fundamentally different lateral control algorithms.

---

## Algorithm Comparison

### Pure Pursuit (Course 1 - WORKING)

**Geometric Path Tracking Algorithm**

```python
# Key characteristics:
# 1. Speed-adaptive lookahead distance
lookahead_distance = max(min_ld, Kp_ld * v)

# 2. Find "carrot" waypoint at lookahead distance
x_rear = x - L * cos(yaw) / 2  # Rear axle position
y_rear = y - L * sin(yaw) / 2

for wp in waypoints:
    dist = sqrt((wp[0] - x_rear)**2 + (wp[1] - y_rear)**2)
    if dist > lookahead_distance:
        carrot = wp  # "Carrot on a stick"
        break

# 3. Bicycle model steering formula
alpha = atan2(carrot[1] - y_rear, carrot[0] - x_rear) - yaw
steer = atan2(2 * L * sin(alpha), lookahead_distance)
```

**Parameters**:
- `Kp_ld = 0.8` - Lookahead distance gain (speed-adaptive)
- `min_ld = 10` meters - Minimum lookahead distance
- `L = 3` meters - Vehicle wheelbase

**Behavior**:
- Smooth tracking with speed-dependent preview
- Larger lookahead at higher speeds → smoother turns
- Smaller lookahead at lower speeds → tighter tracking

---

### Stanley (Course 4 - CURRENT BASELINE)

**Crosstrack + Heading Error Algorithm**

```python
# Key characteristics:
# 1. Fixed lookahead distance
lookahead_distance = 2.0  # Fixed!

# 2. Crosstrack error calculation
crosstrack_vector = np.array([
    waypoints[lookahead_idx][0] - x - lookahead_distance * cos(yaw),
    waypoints[lookahead_idx][1] - y - lookahead_distance * sin(yaw)
])
crosstrack_error = np.linalg.norm(crosstrack_vector)

# 3. Stanley formula
heading_error = trajectory_heading - current_yaw
steer = heading_error + atan(kp_heading * crosstrack_error / (speed + k))
```

**Parameters**:
- `kp_heading = 8.00` - Heading error gain
- `lookahead_distance = 2.0` meters - Fixed lookahead
- `k_speed_crosstrack = 0.00` - Speed dependency (disabled)

**Behavior**:
- More aggressive corrections (higher gain)
- No speed adaptation → same response at all speeds
- Can cause oscillations/zigzag at higher speeds

---

## Why Stanley Causes Zigzag

1. **Fixed Lookahead**: At higher speeds (>5 m/s), a 2m lookahead is too short
   - Vehicle overshoots corrections
   - Creates oscillatory behavior

2. **High Heading Gain**: `kp_heading = 8.00` causes aggressive steering
   - Small heading errors → large steering commands
   - Amplifies oscillations

3. **No Speed Adaptation**: Same response at 1 m/s and 10 m/s
   - Pure Pursuit scales lookahead: 10-18m at typical speeds
   - Stanley uses fixed 2m → inadequate preview

---

## Code Comparison

### **Pure Pursuit (GitHub - WORKING)**

**File**: `Course1FinalProject/controller2d.py` (lines 192-228)

```python
# LATERAL CONTROLLER - PURE PURSUIT
Kp_ld = 0.8      # Lookahead distance gain
min_ld = 10      # Minimum lookahead [m]
L = 3            # Wheelbase [m]

# Rear axle position
x_rear = x - L * cos(yaw) / 2
y_rear = y - L * sin(yaw) / 2

# Speed-adaptive lookahead
lookahead_distance = max(min_ld, Kp_ld * v)

# Find carrot waypoint
for wp in waypoints:
    dist = sqrt((wp[0] - x_rear)**2 + (wp[1] - y_rear)**2)
    if dist > lookahead_distance:
        carrot = wp
        break
else:
    carrot = waypoints[0]

# Pure Pursuit steering
alpha = atan2(carrot[1] - y_rear, carrot[0] - x_rear) - yaw
steer_output = atan2(2 * L * sin(alpha), lookahead_distance)
```

**Speed Behavior**:
- At 5 m/s: lookahead = max(10, 0.8 × 5) = **10 meters**
- At 10 m/s: lookahead = max(10, 0.8 × 10) = **10 meters**
- At 20 m/s: lookahead = max(10, 0.8 × 20) = **16 meters**

---

### **Stanley (Current Baseline - ZIGZAG)**

**File**: `src/baselines/pure_pursuit_controller.py` (lines 76-211)

```python
# LATERAL CONTROLLER - STANLEY (misnamed as Pure Pursuit!)
lookahead_distance = 2.0  # FIXED!
kp_heading = 8.00         # High gain

# Find lookahead index
lookahead_idx = self._get_lookahead_index(current_x, current_y, waypoints)

# Crosstrack error
crosstrack_vector = np.array([
    waypoints[lookahead_idx][0] - current_x - lookahead_distance * np.cos(current_yaw),
    waypoints[lookahead_idx][1] - current_y - lookahead_distance * np.sin(current_yaw)
])
crosstrack_error = np.linalg.norm(crosstrack_vector)

# Trajectory heading
trajectory_heading = np.arctan2(vect_wp0_to_wp1[1], vect_wp0_to_wp1[0])

# Stanley formula
heading_error = self._normalize_angle(trajectory_heading - current_yaw)
steer_rad = heading_error + np.arctan(
    kp_heading * crosstrack_sign * crosstrack_error / (speed + k_speed_crosstrack)
)
```

**Speed Behavior**:
- At 5 m/s: lookahead = **2 meters** (too short!)
- At 10 m/s: lookahead = **2 meters** (way too short!)
- At 20 m/s: lookahead = **2 meters** (catastrophically short!)

---

## Naming Confusion

**Critical Issue**: The current baseline file is named `pure_pursuit_controller.py` but implements Stanley!

```python
# File: src/baselines/pure_pursuit_controller.py
class PurePursuitController:  # ← Misleading name!
    """
    Pure Pursuit controller for lateral (steering) control.  # ← Wrong description!
    
    The implementation uses Stanley's formula:  # ← Actually Stanley!
        steer = heading_error + atan(k * crosstrack_error / speed)
    """
```

This caused the initial confusion - we thought we were using Pure Pursuit!

---

## Implementation Source Files

### Working Code (GitHub)
- **Repo**: ARRETdaniel/Self-Driving_Cars_Specialization
- **Path**: `CarlaSimulator/PythonClient/Course1FinalProject/controller2d.py`
- **Algorithm**: Pure Pursuit (geometric)
- **Lines**: 192-228 (lateral controller)
- **Status**: ✅ Working smoothly in module_7.py

### Current Baseline (Local)
- **Path**: `src/baselines/pure_pursuit_controller.py`
- **Algorithm**: Stanley (crosstrack + heading)
- **Based On**: Course4FinalProject/controller2d.py
- **Status**: ❌ Zigzag behavior

### Attached Reference (Local)
- **Path**: `related_works/.../controller2d.py`
- **Algorithm**: Stanley (Course 4)
- **Lines**: 99-214
- **Note**: Same as Course4FinalProject

---

## Recommended Fix

**Replace Stanley with Pure Pursuit**

1. **Create new implementation** based on Course1FinalProject/controller2d.py
2. **Use speed-adaptive lookahead**: `lookahead_distance = max(10, 0.8 * speed)`
3. **Implement bicycle model**: `steer = atan2(2 * L * sin(alpha), lookahead_distance)`
4. **Use rear axle position** for tracking point
5. **Find carrot waypoint** geometrically (not by index accumulation)

---

## Expected Results After Fix

### Before (Stanley)
- **Lateral Deviation**: 0.865m mean
- **Heading Error**: 9.74° mean
- **Behavior**: Zigzag oscillations
- **Speed Sensitivity**: High (no adaptation)

### After (Pure Pursuit - Expected)
- **Lateral Deviation**: ~0.4-0.6m mean (50% reduction)
- **Heading Error**: ~4-6° mean (40% reduction)
- **Behavior**: Smooth tracking
- **Speed Sensitivity**: Low (adaptive lookahead)

---

## References

1. **Pure Pursuit Paper**: R. Craig Coulter, "Implementation of the Pure Pursuit Path Tracking Algorithm", CMU-RI-TR-92-01
2. **Stanley Paper**: Thrun et al., "Stanley: The Robot that Won the DARPA Grand Challenge", 2006
3. **Working Code**: [Course1FinalProject/controller2d.py](https://github.com/ARRETdaniel/Self-Driving_Cars_Specialization/blob/main/CarlaSimulator/PythonClient/Course1FinalProject/controller2d.py)

---

## Conclusion

The zigzag behavior is NOT a tuning problem - it's a fundamental algorithm mismatch. The baseline uses Stanley (aggressive, fixed lookahead) when it should use Pure Pursuit (smooth, adaptive lookahead).

**Action Required**: Implement true Pure Pursuit based on Course1FinalProject code.
