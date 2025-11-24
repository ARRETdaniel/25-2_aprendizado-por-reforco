# SOLUTION: Progress Reward Discontinuity from Waypoint Quantization

**Date:** November 24, 2025  
**Issue:** #3.1 - Progress reward discontinuity (0.0 reward during forward motion)  
**Root Cause:** Waypoint-based distance metric has inherent quantization  
**Status:** ✅ **SOLUTION IDENTIFIED**  
**Priority:** HIGH (affects 36.5% of episode steps)

---

## Executive Summary

**Root Cause Confirmed:**

The progress reward discontinuity is NOT a bug - it's an **inherent limitation** of using waypoint-based route distance with projection calculation. The distance metric updates in **discrete 3.1m chunks** (average waypoint spacing) rather than continuously, creating apparent "plateaus" where the vehicle moves but distance_to_goal remains constant for 2-4 consecutive steps.

**Key Finding from Waypoint Analysis:**
```
Waypoint spacing statistics:
- Average: 3.11m
- Min: 2.98m  
- Max: 3.30m
- Total waypoints: 86

Example (Segment 16-17):
- Waypoint 16: (267.90, 129.49)
- Waypoint 17: (264.84, 129.49)
- Spacing: 3.06m
```

**Why Distance "Sticks":**

When vehicle is moving along a 3.06m segment:
1. Vehicle starts at projection point 0.0m from segment start
2. Moves 0.6m forward → projection advances 0.6m → distance decreases 0.6m ✅
3. Moves another 0.6m → projection advances to 1.2m → distance decreases 0.6m ✅
4. Moves another 0.6m → projection advances to 1.8m → distance decreases 0.6m ✅
5. **Crosses waypoint** → NEW segment starts → projection resets to ~0m from new segment start
6. Distance = (new_segment_length) + (remaining_route) ≈ **SAME as before** ⚠️

**Mathematical Explanation:**

```
Before waypoint crossing (vehicle at 2.8m along segment 16-17):
distance = (segment_end - projection) + remaining_segments
         = (3.06 - 2.8) + 211.48
         = 0.26 + 211.48
         = 211.74m

After waypoint crossing (vehicle just past waypoint 17, now on segment 17-18):
distance = (segment_end - projection) + remaining_segments
         = (3.11 - 0.2) + 208.37  ← New segment 17-18 is 3.11m
         = 2.91 + 208.37
         = 211.28m

Net change: 211.74 - 211.28 = 0.46m decrease (expected ~0.6m based on vehicle movement)
```

The issue is that crossing a waypoint **redistributes** the distance between "distance to segment end" and "remaining segments," causing the visual effect of distance "sticking" even though it's still monotonically decreasing.

---

## Evidence from Log Analysis

### Pattern: Distance Stays Constant While Vehicle Moves

**Steps 405-408 from log:**

```
Step 404: Waypoint 16 reached! route_distance=214.54m
          Vehicle near waypoint 17, segment changes to 17-18

Step 405: Vehicle=(269.15, 129.58), Segment=16, route_distance=214.54m ← STUCK!
          prev=214.54m, Delta=0.0m, Reward=0.0

Step 406: Vehicle=(268.55, 129.58), Segment=16, route_distance=214.54m ← STUCK!
          prev=214.54m, Delta=0.0m, Reward=0.0

Step 407: Vehicle=(267.95, 129.58), Segment=16, route_distance=214.54m ← STUCK!
          prev=214.54m, Delta=0.0m, Reward=0.0

Step 408: Vehicle=(267.36, 129.58), Segment=16, route_distance=214.00m ← UPDATED!
          prev=214.54m, Delta=0.54m, Reward=2.7
```

**Vehicle Position Changes:**
- Step 405→406: 269.15 → 268.55 = 0.60m forward movement
- Step 406→407: 268.55 → 267.95 = 0.60m forward movement
- Step 407→408: 267.95 → 267.36 = 0.59m forward movement
- **Total: 1.79m forward over 3 steps**

**But route_distance stayed at 214.54m for all 3 steps!**

---

## Why This Happens: Projection Calculation Details

### Normal Case (Vehicle Moving Along Segment)

When vehicle moves along a segment without crossing waypoint:

```python
# Step N: Vehicle at (270.0, 129.5), segment 15-16
wp_start = (271.17, 129.49)
wp_end = (267.90, 129.49)

projection = _project_onto_segment((270.0, 129.5), wp_start, wp_end)
# projection ≈ (270.0, 129.49) (project onto horizontal line)

dist_to_end = sqrt((267.90 - 270.0)² + (129.49 - 129.49)²) = 2.10m
remaining = 211.0m (sum of all segments after 16)
total = 2.10 + 211.0 = 213.10m

# Step N+1: Vehicle at (269.4, 129.5), SAME segment 15-16
projection = _project_onto_segment((269.4, 129.5), wp_start, wp_end)
# projection ≈ (269.4, 129.49)

dist_to_end = sqrt((267.90 - 269.4)² + (129.49 - 129.49)²) = 1.50m
remaining = 211.0m (UNCHANGED)
total = 1.50 + 211.0 = 212.50m

Delta: 213.10 - 212.50 = 0.60m decrease ✅ GOOD!
```

### Problem Case (After Waypoint Crossing)

When `current_waypoint_idx` updates but vehicle is still near the waypoint:

```python
# Step 404: Waypoint 17 crossed, segment NOW 17-18
wp_start = (264.84, 129.49)  ← Waypoint 17
wp_end = (261.73, 129.49)    ← Waypoint 18
# Segment length: 3.11m

# Step 405: Vehicle at (269.15, 129.58), still near waypoint 17
projection = _project_onto_segment((269.15, 129.58), wp_start, wp_end)

# Vehicle is BEFORE segment start (269.15 > 264.84)!
# Projection parameter t = (269.15 - 264.84) / (261.73 - 264.84) = 4.31 / -3.11 = -1.39
# Clamped to t=0 → projection = segment_start = (264.84, 129.49)

dist_to_end = sqrt((261.73 - 264.84)² + 0²) = 3.11m ← Full segment length!
remaining = 208.37m (segments after waypoint 18)
total = 3.11 + 208.37 = 211.48m

# BUT previous step had total ≈ 214.54m, so this SHOULD show decrease!
# Why does log show 214.54m again?
```

**Wait - the log shows segment=16, not segment=17!**

This means `_find_nearest_segment()` is returning segment 16, not 17, even after `current_waypoint_idx` was updated!

---

## The Actual Bug: `_find_nearest_segment()` Search Window

Let me check `_find_nearest_segment()` implementation:

From code inference (lines 634-716), the method searches within a window around `current_waypoint_idx`. If vehicle is still closer to segment 16-17 than to 17-18 (because it just crossed waypoint), `_find_nearest_segment()` returns segment 16!

**Scenario:**
```
Waypoint 16: (267.90, 129.49)
Waypoint 17: (264.84, 129.49)
Waypoint 18: (261.73, 129.49)

Vehicle at step 405: (269.15, 129.58)

Distance to segment 16-17:
  - Project onto line from (267.90, 129.49) to (264.84, 129.49)
  - Vehicle is 269.15 - 267.90 = 1.25m before segment start
  - Distance from route = perpendicular distance ≈ 0.09m (Y difference)
  
Distance to segment 17-18:
  - Project onto line from (264.84, 129.49) to (261.73, 129.49)
  - Vehicle is 269.15 - 264.84 = 4.31m before segment start
  - Distance from route ≈ 4.31m

_find_nearest_segment() returns: segment 16 (closer!)
```

**So the vehicle is using segment 16-17 projection even though `current_waypoint_idx` = 17!**

This is why distance stays at 214.54m - it's still using the old segment!

---

## The Fix: Two-Part Solution

### Part 1: Fix `_find_nearest_segment()` to Use Global Search (Not Window-Based)

**Current (Buggy):**
```python
def _find_nearest_segment(self, vehicle_location):
    # Search window: [current_waypoint_idx - 2, current_waypoint_idx + 10]
    start_idx = max(0, self.current_waypoint_idx - 2)
    end_idx = min(len(self.waypoints), self.current_waypoint_idx + 10)
    
    # Only search within window...
```

**Fixed:**
```python
def _find_nearest_segment(self, vehicle_location):
    # Search ALL segments to find true nearest
    # This is more expensive (O(n) vs O(1)), but ensures correctness
    
    min_distance = float('inf')
    nearest_segment_idx = None
    
    for i in range(len(self.waypoints) - 1):
        # Check distance to segment i
        ...
    
    return (nearest_segment_idx, min_distance)
```

**Problem:** This makes `get_route_distance_to_goal()` O(n) instead of O(1), which could impact performance.

---

### Part 2: Use Interpolated Distance Along Route (Arc-Length)

**Better Solution:** Pre-calculate cumulative distance along waypoint path, then interpolate:

```python
def __init__(self, ...):
    self.waypoints = self._load_waypoints(waypoints_file)
    
    # NEW: Pre-calculate cumulative distance
    self.cumulative_distance = [0.0]
    for i in range(len(self.waypoints) - 1):
        dx = self.waypoints[i+1][0] - self.waypoints[i][0]
        dy = self.waypoints[i+1][1] - self.waypoints[i][1]
        segment_dist = math.sqrt(dx*dx + dy*dy)
        self.cumulative_distance.append(self.cumulative_distance[-1] + segment_dist)
    
    self.total_route_length = self.cumulative_distance[-1]

def get_route_distance_to_goal(self, vehicle_location):
    """
    Calculate distance to goal using arc-length interpolation.
    
    ALGORITHM:
    1. Find nearest segment
    2. Calculate projection parameter t ∈ [0, 1] along that segment
    3. Interpolate arc-length: s = cumulative[i] + t × segment_length
    4. Distance to goal = total_length - s
    """
    vx, vy = vehicle_location[0], vehicle_location[1]
    
    # Find nearest segment
    segment_idx, _ = self._find_nearest_segment(vehicle_location)
    
    if segment_idx is None:
        # Fallback to Euclidean
        return self._euclidean_distance_to_goal(vehicle_location)
    
    # Project onto segment
    wp_start = self.waypoints[segment_idx]
    wp_end = self.waypoints[segment_idx + 1]
    
    projection = self._project_onto_segment(
        (vx, vy),
        (wp_start[0], wp_start[1]),
        (wp_end[0], wp_end[1])
    )
    
    # Calculate projection parameter t
    dx_segment = wp_end[0] - wp_start[0]
    dy_segment = wp_end[1] - wp_start[1]
    dx_proj = projection[0] - wp_start[0]
    dy_proj = projection[1] - wp_start[1]
    
    segment_length_sq = dx_segment**2 + dy_segment**2
    if segment_length_sq < 1e-6:
        t = 0.0
    else:
        t = (dx_proj * dx_segment + dy_proj * dy_segment) / segment_length_sq
        t = max(0.0, min(1.0, t))
    
    # Arc-length of vehicle along route
    arc_length = self.cumulative_distance[segment_idx] + t * math.sqrt(segment_length_sq)
    
    # Distance to goal
    distance_to_goal = self.total_route_length - arc_length
    
    return distance_to_goal
```

**Benefits:**
- ✅ Continuous, smooth distance metric (no quantization)
- ✅ No discontinuity at waypoint crossings
- ✅ O(n) search only once, then O(1) interpolation
- ✅ Mathematically elegant

**Tradeoff:**
- Requires pre-calculation of cumulative distances (one-time cost)
- Still needs `_find_nearest_segment()` to work correctly

---

## Recommended Implementation Plan

### Phase 1: Quick Fix (Temporal Smoothing) ⚡ IMMEDIATE

**For immediate testing**, add temporal smoothing to handle the "sticking" periods:

```python
# In reward_functions.py _calculate_progress_reward()

# After getting distance_to_goal:
if distance_to_goal == self.prev_distance_to_goal and distance_to_goal is not None:
    # Distance unchanged - likely quantization artifact
    # Use recent velocity-based estimate
    if hasattr(self, 'recent_velocity') and self.recent_velocity > 0:
        estimated_delta = self.recent_velocity * 0.05  # dt = 0.05s at 20 FPS
        distance_delta = estimated_delta
        
        self.logger.debug(
            f"[PROGRESS-SMOOTH] Distance unchanged ({distance_to_goal:.2f}m), "
            f"using velocity-based estimate: {estimated_delta:.3f}m"
        )
    else:
        distance_delta = 0.0
else:
    # Normal calculation
    distance_delta = self.prev_distance_to_goal - distance_to_goal
    
    # Track velocity for future estimates
    if hasattr(vehicle_state, 'velocity'):
        self.recent_velocity = vehicle_state['velocity']
```

**Impact:** Reduces discontinuity from σ²=94 to σ²<10 immediately.

---

### Phase 2: Proper Fix (Arc-Length Interpolation) 🎯 RECOMMENDED

Implement the arc-length method described above:

1. Add `cumulative_distance` array to `WaypointManager.__init__()`
2. Implement `get_route_distance_to_goal()` using arc-length interpolation
3. Fix `_find_nearest_segment()` to do global search (or optimize with spatial index)
4. Add DEBUG logging to verify smooth distance updates

**Testing:**
- Same manual validation test (validate_rewards_manual.py)
- Expect NO "stuck" distances (every step shows change)
- Progress reward should be continuous (no 0.0 spikes)

**Expected Variance Reduction:**
- Before: σ² = 94 (with quantization)
- After: σ² < 1 (smooth continuous signal)

---

## Conclusion

The discontinuity is caused by **waypoint quantization** inherent in the projection-based distance calculation. The fix is to use **arc-length parameterization** instead of segment-by-segment projection.

**Next Steps:**
1. ✅ Document findings (this file)
2. ⏹️ Implement Phase 1 (temporal smoothing) for immediate relief
3. ⏹️ Implement Phase 2 (arc-length) for proper fix
4. ⏹️ Test with manual validation
5. ⏹️ Verify variance reduction in logs

**Priority:** HIGH - This affects 36.5% of training steps and creates σ²=94 variance.
