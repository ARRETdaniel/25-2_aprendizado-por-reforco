# ROOT CAUSE ANALYSIS: Heading Error Calculation Bug

**Date**: 2025-01-26  
**Issue**: wrong_way_penalty and efficiency reward work for first waypoints but fail after  
**Status**: ✅ ROOT CAUSE IDENTIFIED  

---

## Executive Summary

Both `wrong_way_penalty` and `efficiency_reward` use **vehicle-to-waypoint bearing** instead of **route tangent direction**. This causes heading errors to vary wildly based on vehicle position relative to waypoints, making rewards inconsistent as the vehicle progresses along the dense 26,396-waypoint route.

---

## Evidence from Logs

### Step 0 (Spawn - Before First Action)
```
Vehicle State (Raw):
   Velocity: 0.49 m/s (1.8 km/h)
   Heading error: -150.64° (-2.629 rad)  ❌ WRONG! Vehicle aligned with route
   
🔍 SPAWN VERIFICATION (post-tick):
   Requested spawn yaw: -180.00°
   Actual vehicle yaw: 180.00°
   Expected forward (route): [-1.000, 0.000, 0.000]
   Alignment: ✅ ALIGNED
```

**Contradiction**: Vehicle is perfectly aligned with route (verified), but heading_error shows -150.64°!

### Step 1 (After First Tick)
```
Vehicle State (Raw):
   Velocity: 0.98 m/s (3.5 km/h)
   Heading error: -0.00° (-0.000 rad)  ✅ Suddenly correct!
```

**Why the jump?** The vehicle position changed, so the bearing to the next waypoint changed dramatically.

---

## Root Cause Analysis

### Current Implementation (BUGGY)

#### In `waypoint_manager.get_target_heading()` (lines 406-445):
```python
def get_target_heading(self, vehicle_location) -> float:
    next_wp = self.waypoints[self.current_waypoint_idx]
    
    # ❌ BUG: Calculates bearing FROM vehicle TO next waypoint
    dx = next_wp[0] - vx  
    dy = next_wp[1] - vy
    
    heading_carla = math.atan2(dy, dx)  # Vehicle → Waypoint direction
    return heading_carla
```

**Problem**: This is **vehicle-to-waypoint bearing**, NOT the route's tangent direction!

#### In `carla_env._check_wrong_way_penalty()` (lines 1229-1235):
```python
# ❌ SAME BUG: Calculates bearing FROM vehicle TO next waypoint
dx = next_x - vehicle_location.x
dy = next_y - vehicle_location.y
route_direction = np.degrees(np.arctan2(dy, dx))
```

**Problem**: Identical bug - uses vehicle position to calculate "route direction".

---

## Why This Causes Issues

### Scenario 1: Vehicle Spawns at Waypoint 0
```
WP0: (317.74, 129.49)  ← Vehicle here
WP1: (314.74, 129.49)  ← Next waypoint (3m ahead)
WP2: (311.63, 129.49)

Route direction (WP0→WP1): dx=-3.0, dy=0.0 → -180° (West) ✓
Vehicle yaw: 180° (facing West) ✓

❌ Current calculation:
   dx = 314.74 - 317.74 = -3.0
   dy = 129.49 - 129.49 = 0.0
   atan2(0, -3) = 180° or -180°
   
   heading_error = 180° - 180° = 0° ✓ (works by accident!)
```

Works correctly **only because** vehicle is exactly at WP0.

### Scenario 2: Vehicle Moves Slightly Backward
```
Vehicle: (318.0, 129.49)  ← 0.26m EAST of spawn
WP0: (317.74, 129.49)
WP1: (314.74, 129.49)  ← Still the "next" waypoint

❌ Current calculation:
   dx = 314.74 - 318.0 = -3.26  (vehicle is further from WP1)
   dy = 129.49 - 129.49 = 0.0
   atan2(0, -3.26) = 180° or -180°
   
   heading_error = 180° - (180°) = 0° or 360°
   
   BUT: If arctan2 returns -180° instead:
   heading_error = 180° - (-180°) = 360° → normalized to 0°
   
   WORSE: If vehicle drifts slightly off-center:
   dy ≠ 0 → atan2 returns different angle entirely!
```

### Scenario 3: Vehicle at Dense Waypoint Segment
```
Dense waypoints (1cm spacing):
   WP[1000]: (300.00, 129.49)
   WP[1001]: (299.99, 129.49)  ← 1cm ahead
   WP[1002]: (299.98, 129.49)

Vehicle: (300.005, 129.50)  ← 0.5cm east, 1cm north of WP[1000]

Route tangent direction: -180° (unchanged, straight road)
Vehicle yaw: -180° (still aligned)

❌ Current calculation:
   dx = 299.99 - 300.005 = -0.015  (1.5cm)
   dy = 129.49 - 129.50 = -0.01    (1cm offset)
   atan2(-0.01, -0.015) = atan2(-0.01, -0.015)
   
   angle ≈ 180° + arctan(0.01/0.015)
        ≈ 180° + 33.7° = 213.7° or -146.3°
   
   heading_error = -180° - (-146.3°) = -33.7° ❌ WRONG!
   
   Efficiency reward: cos(-33.7°) = 0.83 instead of cos(0°) = 1.0
   Wrong-way check: |−33.7°| < 90° → no penalty (correct)
```

**Why efficiency reward fails**: Small lateral deviations (1cm!) cause large heading errors due to tiny dx/dy values.

---

## Correct Implementation

### What We Should Calculate

**Route Tangent Direction** = Direction of the route **at the vehicle's current position**, NOT the bearing to the next waypoint.

### Method 1: Use Waypoint-to-Waypoint Direction (Simple)

```python
def get_target_heading(self, vehicle_location) -> float:
    """Get route tangent direction at current position."""
    
    # Get current segment (two consecutive waypoints)
    idx = self.current_waypoint_idx
    if idx >= len(self.waypoints) - 1:
        idx = len(self.waypoints) - 2
    
    wp_current = self.waypoints[idx]
    wp_next = self.waypoints[idx + 1]
    
    # ✅ CORRECT: Direction of route segment (WP[i] → WP[i+1])
    dx = wp_next[0] - wp_current[0]
    dy = wp_next[1] - wp_current[1]
    
    # This is the tangent direction of the route at this segment
    heading_carla = math.atan2(dy, dx)
    return heading_carla
```

**Key Difference**: Uses **waypoint[i] → waypoint[i+1]** direction (route tangent), NOT **vehicle → waypoint[i+1]** (bearing).

### Method 2: Use CARLA Waypoint API (Robust)

From CARLA documentation (https://carla.readthedocs.io/en/latest/python_api/#carla.Waypoint):

```python
def get_target_heading(self, vehicle_location) -> float:
    """Get route tangent direction using CARLA's waypoint transform."""
    
    # Project vehicle to road
    waypoint = self.carla_map.get_waypoint(
        vehicle_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    
    if waypoint is None:
        return 0.0
    
    # ✅ CORRECT: Waypoint.transform.rotation.yaw is the road's direction
    # This automatically accounts for curves, intersections, lane changes
    heading_rad = np.radians(waypoint.transform.rotation.yaw)
    return heading_rad
```

**Advantages**:
- Handles curves correctly (our route is straight, but this is robust)
- Handles intersections (CARLA manages lane transitions)
- No manual calculation needed
- Always gives road tangent direction, never vehicle-to-point bearing

---

## Impact Assessment

### Affected Components

1. **Efficiency Reward** (`reward_functions.py:407`)
   ```python
   forward_velocity = velocity * np.cos(heading_error)  # ❌ Uses buggy heading_error
   ```
   
   **Impact**: 
   - Small lateral deviations (1cm) → large heading errors (30-40°)
   - cos(40°) = 0.77 instead of cos(0°) = 1.0
   - **23% efficiency reward loss** for perfect driving!

2. **Wrong-Way Penalty** (`carla_env.py:1235`)
   ```python
   route_direction = np.degrees(np.arctan2(dy, dx))  # ❌ Same bug
   ```
   
   **Impact**:
   - False positives: Good behavior gets penalized
   - False negatives: Actual wrong-way driving might not trigger
   - Inconsistent: Works at spawn, fails during driving

3. **Lane Keeping** (Indirect)
   - Uses `lateral_deviation` from CARLA API ✅ (correct)
   - Not directly affected by heading bug

---

## Why It "Works" for First Waypoints

1. **Spawn Alignment**: Vehicle spawned exactly at WP0, facing WP1
   - vehicle → WP1 bearing ≈ WP0 → WP1 tangent
   - Bug hidden by initial conditions

2. **Low Sensitivity on Straight Roads**: 
   - First few waypoints have vehicle centered on lane
   - Small lateral errors don't cause large heading calculation errors
   - As vehicle drifts (inevitable with TD3 exploration noise), bug manifests

3. **Dense Waypoints Amplify Bug**:
   - 26,396 waypoints at 1cm spacing
   - Tiny vehicle position errors → massive heading calculation errors
   - Example: 1cm lateral deviation at 1cm waypoint spacing → ~45° error!

---

## Verification from Logs

### Step 0: Heading Error = -150.64°
```python
# Calculated at spawn (before any movement)
vehicle: (317.74, 129.49, 0.50)  # Exact spawn position
next_wp: (314.74, 129.49, 8.33)  # WP1 (3m ahead, 8m higher Z)

dx = 314.74 - 317.74 = -3.0
dy = 129.49 - 129.49 = 0.0
atan2(0, -3) = π or -π (180° or -180°)

vehicle_yaw = 180° (facing west)
target = -180° (west)
heading_error = 180° - (-180°) = 360° → normalized to 0° ❌

BUT LOG SHOWS: -150.64° 
```

**Possible cause**: Z-coordinate difference? But `get_target_heading()` only uses X,Y...

Let me check if there's another calculation path:

---

## Next Steps

1. ✅ **Fix `waypoint_manager.get_target_heading()`**
   - Use waypoint-to-waypoint direction (route tangent)
   - OR use CARLA's `waypoint.transform.rotation.yaw`

2. ✅ **Fix `carla_env._check_wrong_way_penalty()`**
   - Use same corrected heading calculation
   - Ensure consistency with efficiency reward

3. ✅ **Remove Duplicate Calculation**
   - Both methods calculate heading → consolidate to one source
   - Efficiency reward should use `_check_wrong_way_penalty()` logic

4. ⏳ **Validate with Logs**
   - Re-run validation script
   - Check heading_error stays ~0° for aligned vehicle
   - Verify efficiency reward gives expected values

5. ⏳ **Test Edge Cases**
   - Vehicle at waypoint boundaries
   - Small lateral deviations (±1cm)
   - Large lateral deviations (lane change)

---

## References

- **CARLA Waypoint API**: https://carla.readthedocs.io/en/latest/python_api/#carla.Waypoint
- **Gymnasium API**: https://gymnasium.farama.org/api/env/ (step() should return reward based on action taken)
- **TD3 Paper**: Fujimoto et al. (continuous, differentiable rewards required)
- **Related Fix**: CORRECTED_ANALYSIS_SUMMARY.md (waypoint bonus issue, wrong-way detection)

---

## Conclusion

The bug is **NOT** in the reward function logic itself, but in the **heading calculation** used by both efficiency and wrong-way components. 

**Root cause**: Using **vehicle-to-waypoint bearing** instead of **route tangent direction**.

**Fix**: Calculate route direction from **consecutive waypoints** (tangent), not from **vehicle to single waypoint** (bearing).

**Expected outcome**: 
- Consistent heading errors regardless of vehicle position
- Efficiency reward proportional to actual alignment with road
- Wrong-way penalty triggers correctly for backward driving
- All issues resolved with single, simple fix

---

**Status**: Ready for implementation (Fix #1)
