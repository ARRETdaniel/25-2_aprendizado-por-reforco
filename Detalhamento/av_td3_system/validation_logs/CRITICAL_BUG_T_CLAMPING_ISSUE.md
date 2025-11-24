# CRITICAL BUG: t-Parameter Clamping Causes Segment Sticking

**Date**: 2025-11-24
**Severity**: 🔴 **CRITICAL** - Arc-length projection still has discontinuities
**Status**: 🔍 **ROOT CAUSE IDENTIFIED**

---

## Executive Summary

**Problem**: The arc-length projection fix is STILL causing progress reward discontinuities because vehicles that pass a segment endpoint (t > 1.0) get STUCK on that segment with t clamped at 1.0, causing distance to remain constant.

**Impact**: Zero progress rewards despite forward movement, same as before!

**Root Cause**: The nearest segment search uses **perpendicular distance**, which can select a segment the vehicle has ALREADY PASSED when t would be > 1.0 (vehicle is beyond segment endpoint).

---

## Evidence from Logs

From `docs/day-24/progress.log` (Steps 49-51):

```
Step 49:
  Vehicle=(316.00, 129.49)
  SegmentIdx=102, t=1.0000 ← CLAMPED!
  PerpendicularDist=0.713m
  ArcLength=263.35m
  Delta=0.000m, Reward=0.00

Step 50:
  Vehicle=(315.78, 129.49) ← MOVED 0.22m FORWARD!
  SegmentIdx=102, t=1.0000 ← STILL CLAMPED!
  PerpendicularDist=0.932m ← INCREASING!
  ArcLength=263.35m ← UNCHANGED!
  Delta=0.000m, Reward=0.00 ❌

Step 51:
  Vehicle=(315.55, 129.49) ← MOVED 0.23m FORWARD!
  SegmentIdx=102, t=1.0000 ← STILL CLAMPED!
  PerpendicularDist=1.163m ← INCREASING!
  ArcLength=263.35m ← UNCHANGED!
  Delta=0.000m, Reward=0.00 ❌
```

**Analysis**:
- Vehicle moving forward: X decreasing (316.00 → 315.78 → 315.55)
- t stuck at 1.0000 (clamped at segment endpoint)
- Perpendicular distance INCREASING (vehicle moving away from segment 102 endpoint)
- Arc-length CONSTANT because `(1 - t) * segment_length = (1 - 1.0) * 0.01 = 0.00m`

**The vehicle has PASSED segment 102 but the search still selects it!**

---

## Why This Happens

### The Flawed Search Algorithm

**Current implementation** (waypoint_manager.py lines 633-668):

```python
for i in range(search_start, search_end):
    # Project vehicle onto segment i
    t = ((vx - wp_a[0]) * seg_x + (vy - wp_a[1]) * seg_y) / seg_length_sq

    # Clamp t to [0, 1]
    t = max(0.0, min(1.0, t))  # ← PROBLEM: Hides when vehicle past segment!

    # Calculate closest point on segment
    closest_x = wp_a[0] + t * seg_x
    closest_y = wp_a[1] + t * seg_y

    # Distance from vehicle to closest point
    dist = sqrt((vx - closest_x)² + (vy - closest_y)²)

    if dist < min_dist:
        min_dist = dist
        nearest_segment_idx = i  # ← Selects based on perpendicular distance!
```

**Scenario**: Vehicle at (315.78, 129.49), segments are 1cm apart

```
Segment 102: (316.01, 129.49) → (316.00, 129.49)  [1cm west]
Segment 103: (316.00, 129.49) → (315.99, 129.49)  [1cm west]
Segment 104: (315.99, 129.49) → (315.98, 129.49)  [1cm west]
...
Vehicle:     (315.78, 129.49)  ← Vehicle is PAST all these segments!
```

**For segment 102**:
```
wp_a = (316.01, 129.49)
wp_b = (316.00, 129.49)
seg_vector = (-0.01, 0.0)

t_unclamped = ((315.78 - 316.01) * -0.01 + 0) / 0.01²
            = (-0.23 * -0.01) / 0.0001
            = 0.0023 / 0.0001
            = 23.0  ← Vehicle is 23 segments PAST endpoint!

t_clamped = 1.0

projection = (316.01 + 1.0 * -0.01, 129.49) = (316.00, 129.49)
dist = sqrt((315.78 - 316.00)² + 0²) = 0.22m
```

**For segment 125** (where vehicle actually is):
```
wp_a = (315.79, 129.49)
wp_b = (315.78, 129.49)
seg_vector = (-0.01, 0.0)

t_unclamped = ((315.78 - 315.79) * -0.01 + 0) / 0.01²
            = (-0.01 * -0.01) / 0.0001
            = 0.0001 / 0.0001
            = 1.0  ← At segment endpoint

projection = (315.79 + 1.0 * -0.01, 129.49) = (315.78, 129.49)
dist = sqrt(0² + 0²) = 0.00m  ← PERFECT MATCH!
```

**BUT**: If segment 125 is outside the search window (`current_waypoint_idx + 100`), it won't be checked!

---

## The Real Problem: Search Window

From the code:

```python
search_start = max(0, self.current_waypoint_idx - 10)
search_end = min(len(self.dense_waypoints) - 1, self.current_waypoint_idx + 100)
```

**If `current_waypoint_idx` is outdated**, the search window might not include the segment where the vehicle actually is!

**Example**:
```
current_waypoint_idx = 0 (original waypoint, not dense waypoint index!)
search_start = 0
search_end = min(26395, 0 + 100) = 100

Vehicle is at dense waypoint 125 (beyond search window!)
→ Search only checks segments 0-100
→ Finds segment 100 as "nearest" (even though vehicle is 25 segments past it!)
→ t = 1.0000 (clamped)
→ Distance stuck!
```

---

## The Solution

**Option 1: Expand Search Window (BAND-AID)**
```python
search_start = 0  # Search ALL segments
search_end = len(self.dense_waypoints) - 1
```
**Pros**: Simple fix
**Cons**: O(N) search every step (N=26,396), VERY SLOW!

**Option 2: Use Unclamped t for Selection (RECOMMENDED)**

Only clamp `t` AFTER finding the nearest segment:

```python
for i in range(search_start, search_end):
    # Calculate UNCLAMPED t
    t_unclamped = ((vx - wp_a[0]) * seg_x + (vy - wp_a[1]) * seg_y) / seg_length_sq

    # For perpendicular distance, use CLAMPED t
    t_clamped = max(0.0, min(1.0, t_unclamped))
    closest_x = wp_a[0] + t_clamped * seg_x
    closest_y = wp_a[1] + t_clamped * seg_y
    dist = sqrt((vx - closest_x)² + (vy - closest_y)²)

    # BUT: Only select segments where vehicle is BETWEEN start and end!
    # Check if 0.0 <= t_unclamped <= 1.0
    if 0.0 <= t_unclamped <= 1.0:
        # Vehicle is ON this segment
        if dist < min_dist:
            min_dist = dist
            nearest_segment_idx = i
    # If t_unclamped < 0.0: vehicle is BEFORE segment (skip)
    # If t_unclamped > 1.0: vehicle is PAST segment (skip)
```

**Benefit**: Only selects segments the vehicle is ACTUALLY on (not past)!

**Option 3: Progressive Search from current_waypoint_idx (BEST)**

Start from `current_waypoint_idx` and search forward until finding a segment with `t <= 1.0`:

```python
# Start from current waypoint and search forward
for i in range(self.current_waypoint_idx, min(len(self.dense_waypoints) - 1, self.current_waypoint_idx + 200)):
    wp_a = self.dense_waypoints[i]
    wp_b = self.dense_waypoints[i + 1]

    # Calculate t
    seg_x = wp_b[0] - wp_a[0]
    seg_y = wp_b[1] - wp_a[1]
    seg_length_sq = seg_x ** 2 + seg_y ** 2

    if seg_length_sq < 1e-6:
        continue

    t = ((vx - wp_a[0]) * seg_x + (vy - wp_a[1]) * seg_y) / seg_length_sq

    if t < 0.0:
        # Vehicle is before this segment, stop searching
        nearest_segment_idx = max(0, i - 1)
        break
    elif 0.0 <= t <= 1.0:
        # Vehicle is ON this segment!
        nearest_segment_idx = i
        break
    # else t > 1.0: vehicle is past this segment, continue to next

# Update current_waypoint_idx to stay close to vehicle
self.current_waypoint_idx = nearest_segment_idx
```

**Benefits**:
- ✅ O(1) average case (only checks ~1-3 segments)
- ✅ Automatically finds correct segment
- ✅ Updates current_waypoint_idx to keep search window relevant
- ✅ No possibility of selecting past segments

---

## Recommended Fix

Implement **Option 3** (Progressive Search):

1. Start from `current_waypoint_idx`
2. Search forward until finding segment with `0 <= t <= 1`
3. Update `current_waypoint_idx` to track vehicle position
4. This ensures search window always includes vehicle

---

## Impact

**Current State (BROKEN)**:
- ❌ Progress reward = 0.00 when t clamped at 1.0
- ❌ Distance stuck despite forward movement
- ❌ Same discontinuity as before!

**After Fix (CORRECT)**:
- ✅ Always finds segment vehicle is currently on
- ✅ t varies smoothly 0→1 within each segment
- ✅ Distance decreases continuously
- ✅ No stuck segments!

---

**Status**: 🔴 **CRITICAL** - Must fix before training can succeed!

**Reference**: This is why the Nov 21 projection approach had issues - search window limitations!
