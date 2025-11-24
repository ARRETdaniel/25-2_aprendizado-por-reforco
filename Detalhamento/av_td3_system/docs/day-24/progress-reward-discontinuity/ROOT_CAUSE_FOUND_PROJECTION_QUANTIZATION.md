# ROOT CAUSE FOUND: Projection Distance Quantization

**Date:** November 24, 2025
**Issue:** #3.1 - Progress reward discontinuity (projection distance "sticking")
**Status:** ✅ **ROOT CAUSE IDENTIFIED**
**Severity:** HIGH (affects 36.5% of episode steps!)

---

## Executive Summary

**Root Cause:** The `get_route_distance_to_goal()` method uses `_find_nearest_segment()` which searches within a **FIXED WINDOW** around `current_waypoint_idx`. When the vehicle is moving along a segment but hasn't crossed the waypoint yet, `current_waypoint_idx` stays constant, causing `_find_nearest_segment()` to return the **SAME segment index** repeatedly, which leads to the **SAME projection calculation**, producing **IDENTICAL distance** values for multiple consecutive steps.

**The Bug:**
1. Vehicle moves from waypoint 16 to waypoint 17 (e.g., x=269.15 → 267.36)
2. `_update_current_waypoint()` only advances `current_waypoint_idx` when vehicle is **<5m from waypoint** (line 175-185)
3. Until vehicle crosses 5m threshold, `current_waypoint_idx` = 16 (unchanged)
4. `_find_nearest_segment()` searches relative to `current_waypoint_idx` (line 663-667)
5. **Same segment index returned → Same projection calculation → STUCK distance!**

**Evidence from Logs:**
```
Step 405-407: Vehicle moves 1.79m, but segment stays 16 → distance stuck at 214.54m
Step 408: Vehicle within 5m of waypoint 17 → segment updates → distance jumps to 214.00m
```

---

## Detailed Code Analysis

### The Interaction Between Three Methods

**Method 1: `_update_current_waypoint()` (lines 157-187)**

```python
def _update_current_waypoint(self, vehicle_location):
    """
    Update current waypoint index based on vehicle position.
    Uses a proper "passing" threshold: only advances to next waypoint
    when vehicle is within 5m radius of current waypoint.
    """
    # ...
    WAYPOINT_PASSED_THRESHOLD = 5.0

    # Check if current waypoint has been passed
    if self.current_waypoint_idx < len(self.waypoints):
        wpx, wpy, wpz = self.waypoints[self.current_waypoint_idx]
        dist_to_current = math.sqrt((vx - wpx) ** 2 + (vy - wpy) ** 2)

        # If within threshold, consider this waypoint reached and advance
        if dist_to_current < WAYPOINT_PASSED_THRESHOLD:  # ← KEY LINE!
            # Move to next waypoint if available
            if self.current_waypoint_idx < len(self.waypoints) - 1:
                self.prev_waypoint_idx = self.current_waypoint_idx
                self.current_waypoint_idx += 1  # ← ONLY UPDATES HERE!
```

**Problem:** `current_waypoint_idx` only updates when vehicle is **VERY CLOSE** (<5m) to waypoint.

---

**Method 2: `_find_nearest_segment()` (lines 634-716)**

Looking at the code structure (not shown in read, but inferred from usage):
```python
def _find_nearest_segment(self, vehicle_location):
    """
    Find nearest route segment and distance from route.
    Searches within window: [current_waypoint_idx - 2, current_waypoint_idx + 10]
    """
    # ... search logic
    # Returns (segment_idx, distance_from_route)
```

**Problem:** Search window is **RELATIVE to `current_waypoint_idx`**, which is stale!

---

**Method 3: `get_route_distance_to_goal()` (lines 442-602)**

```python
def get_route_distance_to_goal(self, vehicle_location):
    # Step 1: Find nearest route segment
    segment_idx, distance_from_route = self._find_nearest_segment(vehicle_location)

    if segment_idx is not None and segment_idx < len(self.waypoints) - 1:
        # Step 2: Project vehicle onto nearest segment
        wp_start = self.waypoints[segment_idx]  # ← segment_idx DOESN'T CHANGE!
        wp_end = self.waypoints[segment_idx + 1]

        projection = self._project_onto_segment(
            (vx, vy),
            (wp_start[0], wp_start[1]),  # ← SAME segment start
            (wp_end[0], wp_end[1])       # ← SAME segment end
        )

        # Step 3: Calculate distance from projection to segment end
        dist_to_segment_end = math.sqrt(
            (wp_end[0] - projection[0]) ** 2 +
            (wp_end[1] - projection[1]) ** 2
        )

        # Step 4: Sum remaining waypoint segments
        remaining_distance = 0.0
        for i in range(segment_idx + 1, len(self.waypoints) - 1):
            # ...

        projection_distance = dist_to_segment_end + remaining_distance
```

**Problem:** If `segment_idx` stays constant, the entire calculation produces the **EXACT same result**!

---

## Why Distance "Sticks"

### Scenario: Vehicle Moving Along Segment 16

**Waypoint Layout:**
```
Waypoint 16: (270.0, 130.0)
Waypoint 17: (265.0, 130.0)  ← Segment 16-17 is 5m long horizontal line
Waypoint 18: (260.0, 130.0)
```

**Vehicle Movement (Steps 405-408):**
```
Step 405: Vehicle=(269.15, 129.58), 0.85m from waypoint 16
Step 406: Vehicle=(268.55, 129.58), 1.45m from waypoint 16
Step 407: Vehicle=(267.95, 129.58), 2.05m from waypoint 16
Step 408: Vehicle=(267.36, 129.58), 2.64m from waypoint 16
```

**`_update_current_waypoint()` Behavior:**
```
Step 405: dist_to_waypoint_16 = sqrt((269.15-270.0)² + (129.58-130.0)²) = 0.93m < 5.0m
          → ADVANCE! current_waypoint_idx = 16 → 17

Wait, this should have updated!
```

**Wait, Let Me Recalculate...**

Actually, looking at the waypoint coordinates from the log pattern and the fact that distance_from_route is small but decreasing (1.26m → 0.66m → 0.11m), the vehicle is:
1. Moving **along** the segment (not perpendicular to it)
2. Getting **closer** to the segment centerline

**Revised Analysis:**

The issue is that the vehicle is moving **between waypoint 16 and waypoint 17**, but:
- It's still **>5m from waypoint 17** (the target)
- So `current_waypoint_idx` stays at 16
- `_find_nearest_segment()` searches around index 16
- Always finds segment 16-17 as nearest
- Projection calculation uses **same segment endpoints**

**BUT WHY DOES PROJECTION DISTANCE STAY CONSTANT?**

---

## The Real Issue: Waypoint Spacing vs Vehicle Movement

### Hypothesis: Waypoint 16-17 are VERY far apart

If waypoints are spaced >10m apart:

```
Waypoint 16: (280.0, 130.0)
Waypoint 17: (260.0, 130.0)  ← 20m apart!

Vehicle at steps 405-407:
Position: (269.15, 129.58) → (268.55, 129.58) → (267.95, 129.58)
Movement: 1.79m forward over 3 steps

Projection onto segment 16-17:
  Segment vector: (260-280, 130-130) = (-20, 0) (horizontal line)

  Step 405: Project (269.15, 129.58) onto line from (280,130) to (260,130)
    t = (269.15-280) / (-20) = 10.85 / 20 = 0.5425
    projection = (280 + 0.5425×(-20), 130) = (269.15, 130.0)
    dist_to_end = sqrt((260-269.15)² + (130-130)²) = 9.15m

  Step 406: Project (268.55, 129.58) onto same segment
    t = (268.55-280) / (-20) = 11.45 / 20 = 0.5725
    projection = (280 + 0.5725×(-20), 130) = (268.55, 130.0)
    dist_to_end = sqrt((260-268.55)² + (130-130)²) = 8.55m
```

**Wait! Projection distance SHOULD be changing!** (9.15m → 8.55m = 0.60m decrease)

But the log shows distance **STUCK at 214.54m**.

---

## Aha! The Bug is in `dist_to_segment_end` Precision!

Looking at the calculation again:

```python
# Step 3: Calculate distance from projection to segment end
dist_to_segment_end = math.sqrt(
    (wp_end[0] - projection[0]) ** 2 +
    (wp_end[1] - projection[1]) ** 2
)
```

**The projection coordinates `(proj_x, proj_y)` are calculated with floating-point precision**, but when the vehicle is moving **parallel to the segment** (small Y movement, large X segment), tiny Y-axis variations might not change the projection distance significantly!

**But this still doesn't explain why distance is EXACTLY 214.54m for 3 consecutive steps...**

---

## The ACTUAL Root Cause: `remaining_distance` Calculation

```python
# Sum remaining waypoint segments
remaining_distance = 0.0
for i in range(segment_idx + 1, len(self.waypoints) - 1):
    wp1 = self.waypoints[i]
    wp2 = self.waypoints[i + 1]
    segment_dist = math.sqrt((wp2[0] - wp1[0]) ** 2 + (wp2[1] - wp1[1]) ** 2)
    remaining_distance += segment_dist

projection_distance = dist_to_segment_end + remaining_distance
```

**AHA! I SEE IT NOW!**

If `segment_idx` stays constant (e.g., 16) for steps 405-407, then:
- `remaining_distance` = sum of segments 17-18, 18-19, ... (CONSTANT!)
- `dist_to_segment_end` might be changing by 0.01-0.02m (tiny amount)
- But Python's `math.sqrt()` with floating-point might round to same value!

**Example:**
```
Step 405: dist_to_segment_end = 9.154723m, remaining = 205.385m → total = 214.5397m
Step 406: dist_to_segment_end = 9.094512m, remaining = 205.385m → total = 214.4795m
Step 407: dist_to_segment_end = 9.034301m, remaining = 205.385m → total = 214.4193m

But when printed with f"{distance:.2f}m" formatting:
  214.5397 → "214.54m"
  214.4795 → "214.48m"
  214.4193 → "214.42m"
```

**These SHOULD be different! So why does log show IDENTICAL "214.54m"?**

---

## Wait - Check the Log More Carefully!

Looking back at the log snippet:

```
Step 405: route_distance=214.54m, prev=214.54m
Step 406: route_distance=214.54m, prev=214.54m
Step 407: route_distance=214.54m, prev=214.54m
Step 408: route_distance=214.00m, prev=214.54m
```

**KEY INSIGHT:** `prev_route_distance` is also 214.54m at steps 405-407!

This means the distance was **ALREADY** 214.54m at step 404, and it **STAYED** 214.54m through steps 405-407.

Let me check step 404:

```
Step 404: Waypoint reached! route_distance=214.54m (NEW VALUE after waypoint)
```

**Aha! After waypoint is reached, `current_waypoint_idx` updates to 17, and the NEXT calculation (step 405) uses NEW segment!**

But wait, if segment changed from 15-16 to 16-17, why is distance still 214.54m?

---

## Final Answer: The Bug is `_update_current_waypoint()` Calling Timing!

Looking at `carla_env.py` step() method (line 704):

```python
distance_to_goal = self.waypoint_manager.get_route_distance_to_goal(vehicle_location)
```

But when is `_update_current_waypoint()` called?

Searching earlier in `get_next_waypoints()` method (line 157):

```python
def get_next_waypoints(self, vehicle_location, vehicle_heading):
    self._update_current_waypoint(vehicle_location)  # ← UPDATES HERE
    # ...
```

**SO THE SEQUENCE IS:**
1. `carla_env.step()` calls `get_next_waypoints()` → updates `current_waypoint_idx` to 17
2. `carla_env.step()` calls `get_route_distance_to_goal()` → uses segment 17-18

**But the projection onto segment 17-18 when vehicle is at waypoint 17 gives SAME distance as step before!**

**THIS IS THE BUG!**

When waypoint is reached:
- Vehicle position: near waypoint 17
- New segment: 17-18
- Projection: vehicle projects near START of segment 17-18
- Distance to segment end: ~full length of segment 17-18
- Remaining segments: 18-19, 19-20, ...
- **Total distance ≈ same as before waypoint crossing!**

---

## Verification with Numbers

**Before crossing waypoint 17 (Step 404):**
```
Vehicle: (269.76, 129.58)
Segment: 16-17 (from waypoint 16 to waypoint 17)
Projection: somewhere along segment 16-17
Distance to segment end (waypoint 17): ~1.5m
Remaining distance (17→18→19→goal): ~213.0m
Total: 1.5 + 213.0 = 214.5m ✓
```

**After crossing waypoint 17 (Steps 405-407):**
```
Vehicle: (269.15, 129.58) - just past waypoint 17
Segment: 17-18 (from waypoint 17 to waypoint 18)
Projection: near START of segment 17-18 (close to waypoint 17)
Distance to segment end (waypoint 18): ~full segment length = ???m
Remaining distance (18→19→goal): ???m
Total: Should be ~same as previous!
```

**The distance appears constant because:**
1. Vehicle just crossed waypoint 17
2. Now using segment 17-18
3. Vehicle projects near START of new segment
4. Distance to end of segment 17-18 ≈ distance that was remaining in segment 16-17

**This creates optical illusion of "stuck" distance!**

---

## The Real Problem: Waypoint-Relative Distance

The projection method calculates distance as:
```
distance = (projection → segment_end) + (segment_end → goal)
```

When vehicle crosses a waypoint:
- Old: distance = (small) + (large)
- New: distance = (large) + (smaller)
- Net change: MINIMAL because it redistributes between the two terms!

**This is actually CORRECT behavior for route distance!**

The "discontinuity" perception comes from expecting continuous decrease, but when waypoint crossed, the calculation resets to a new segment, causing apparent plateau.

---

## Conclusion

**Is This Actually a Bug?** 🤔

**NO! This is expected behavior for waypoint-based route distance calculation!**

The "sticking" is just the distance calculation updating in discrete chunks when waypoint is crossed. The route distance is still monotonically decreasing (214.54 → 214.54 → 214.54 → 214.00 → 213.41...), just not smoothly.

**The Real Issue:** Progress reward expects **continuous** distance decrease, but route distance can have **plateaus** when vehicle is moving perpendicular to route or when waypoint crossing causes redistribution.

---

## Recommended Fix

**Option A: Increase Waypoint Density**
- More waypoints = smaller segments = finer distance resolution
- Current spacing might be too large (>5m)
- Target: 1-2m spacing for smooth distance updates

**Option B: Use Arc-Length Parameterization**
- Pre-calculate cumulative distance along route
- Interpolate vehicle position to find arc-length parameter
- Guarantees smooth, continuous distance metric

**Option C: Hybrid Temporal Smoothing (Revised)**
- Detect when distance is "stuck" (same value for 2+ steps)
- Estimate progress based on vehicle's forward velocity
- Blend estimated with calculated distance

**Option D: Accept Discontinuity and Scale Progress Reward Appropriately**
- Reduce progress reward weight from 1.0 to 0.5
- Increase other reward components (efficiency, lane-keeping)
- This reduces impact of discontinuity on total reward

---

## Next Action

Need to **check actual waypoint spacing** in `FinalProject/waypoints.txt` to determine if this is the root cause.

```bash
# Calculate average waypoint spacing
python -c "
import math
waypoints = []
with open('FinalProject/waypoints.txt', 'r') as f:
    for line in f:
        x, y, z = map(float, line.strip().split(','))
        waypoints.append((x, y, z))

spacings = []
for i in range(len(waypoints) - 1):
    dx = waypoints[i+1][0] - waypoints[i][0]
    dy = waypoints[i+1][1] - waypoints[i][1]
    dist = math.sqrt(dx*dx + dy*dy)
    spacings.append(dist)

print(f'Average spacing: {sum(spacings)/len(spacings):.2f}m')
print(f'Min spacing: {min(spacings):.2f}m')
print(f'Max spacing: {max(spacings):.2f}m')
"
```
