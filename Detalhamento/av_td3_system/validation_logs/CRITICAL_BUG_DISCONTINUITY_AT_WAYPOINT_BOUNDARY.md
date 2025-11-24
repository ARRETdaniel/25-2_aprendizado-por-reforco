# CRITICAL BUG: Distance Discontinuity at Dense Waypoint Boundaries

**Date**: 2025-01-24
**Severity**: 🔴 **CRITICAL** - Penalizes forward movement incorrectly
**Status**: 🔍 **IDENTIFIED** - Requires immediate fix

---

## Executive Summary

**Problem**: When vehicle crosses a dense waypoint boundary, the distance-to-goal can INCREASE even though the vehicle is moving forward, resulting in negative progress rewards for good behavior.

**Impact**: Agent receives negative rewards for forward movement, corrupting the learning signal.

**Root Cause**: Using "nearest waypoint + distance to waypoint" creates geometric discontinuities when switching between waypoints. The `VehicleToWP` distance jumps when the nearest waypoint changes.

---

## Problem Analysis

### Evidence from Logs

**Step 46 → Step 47** (from `docs/day-24/progress.log`):

```
Step 46:
  Vehicle: (316.66, 129.49)
  NearestIdx: 101
  VehicleToWP: 0.069m
  TotalDist: 263.44m
  Delta: +0.026m (forward), Reward: +0.13 ✅

Step 47:
  Vehicle: (316.49, 129.49) ← MOVED FORWARD 0.17m!
  NearestIdx: 102 ← SWITCHED TO NEXT WAYPOINT
  VehicleToWP: 0.233m ← JUMPED FROM 0.069m!
  TotalDist: 263.59m
  Delta: -0.154m (backward), Reward: -0.77 ❌ WRONG!
```

**Vehicle moved 0.17m FORWARD, but got NEGATIVE reward!**

### Geometric Explanation

```
Dense Waypoints (1cm spacing):

Step 46:
GOAL ← ... ← WP[102] ← ── 0.94cm ── ← WP[101] ← ── 0.069m ── ← VEHICLE
                                                  (nearest=101)
Distance = 0.069m + (WP[101]→GOAL) = 263.44m


Step 47: (vehicle moved 0.17m forward)
GOAL ← ... ← WP[103] ← ── 1.0cm ── ← WP[102] ← ── 0.233m ── ← VEHICLE
                                                  (nearest=102, switched!)
Distance = 0.233m + (WP[102]→GOAL) = 263.59m ← INCREASED! ❌
```

**The Problem**:
- Vehicle moved from near WP[101] to past WP[102]
- WP[101]→WP[102] distance = ~0.01m (1cm spacing)
- Vehicle is now 0.233m FROM WP[102] (it overshot forward)
- But WP[102] is ONE waypoint closer to goal
- **Net effect**: Distance INCREASED by 0.15m despite forward movement!

---

## Why This Happens

When using **nearest waypoint + distance to nearest**:

1. Vehicle approaches WP[101] → distance decreases ✅
2. Vehicle reaches WP[101] → VehicleToWP ≈ 0.0m ✅
3. **Vehicle passes WP[101] and continues forward**:
   - Still nearest to WP[101] while distance < 0.5cm
   - VehicleToWP increases (moving away from WP[101])
   - Distance formula: `0.004m + chain_from_101` = INCREASING ❌
4. **Vehicle reaches 0.5cm from WP[101]**:
   - Now nearest to WP[102] (switches!)
   - VehicleToWP jumps to distance from WP[102]
   - Distance formula: `0.233m + chain_from_102`
   - **Discontinuous jump**: Distance increases by waypoint spacing!

---

## Correct Solution: Arc-Length Projection

Instead of "nearest waypoint + distance", we need **arc-length projection onto path**:

```python
def get_route_distance_to_goal(self, vehicle_location):
    """Calculate distance using projection onto dense waypoint path."""

    # Find nearest segment (between two consecutive waypoints)
    nearest_segment_idx = find_nearest_segment(vehicle, dense_waypoints)

    # Project vehicle onto segment to find closest point ON THE PATH
    wp_a = dense_waypoints[nearest_segment_idx]
    wp_b = dense_waypoints[nearest_segment_idx + 1]

    # Projection gives us position along segment (t ∈ [0, 1])
    t = project_point_onto_segment(vehicle, wp_a, wp_b)

    # Distance from vehicle to projection point (perpendicular distance)
    # This is the "cross-track error" - how far off the path
    lateral_dist = perpendicular_distance(vehicle, wp_a, wp_b, t)

    # Arc-length distance ALONG the path from projection to goal
    arc_length_to_goal = (
        (1 - t) * segment_length(wp_a, wp_b)  # Remaining on current segment
        + sum_segments(nearest_segment_idx + 1, end)  # All remaining segments
    )

    # Total distance = lateral + along-path
    # OR: Just use along-path distance (ignore lateral for progress)
    distance_to_goal = arc_length_to_goal

    return distance_to_goal
```

### Why This Works

**Same scenario with projection**:

```
Step 46:
GOAL ← WP[102] ← WP[101] ← VEHICLE
         ^         ^         └─ projects to point P1 on segment[101→102]
         1.0cm     0.0cm

Distance = (segment[101→102] - distance_to_P1) + sum(segments[102→end])
         = (0.01m - 0.002m) + 263.43m
         = 263.44m ✅

Step 47: (moved 0.17m forward, now past WP[102])
GOAL ← WP[103] ← WP[102] ← ← ← ← ← VEHICLE
                   ^                └─ projects to P2 on segment[102→103]

Distance = (segment[102→103] - distance_to_P2) + sum(segments[103→end])
         = (0.01m - 0.007m) + 263.28m
         = 263.29m ✅ DECREASED BY 0.15m!
```

**No discontinuity** because projection smoothly transitions across waypoint boundaries!

---

## Impact Assessment

### Current State (BROKEN)
- ❌ Negative rewards for forward movement at waypoint boundaries
- ❌ Agent penalized for good behavior
- ❌ Learning signal corrupted
- ❌ Cannot train successfully

### After Fix (CORRECT)
- ✅ Distance monotonically decreases during forward movement
- ✅ Smooth transitions at waypoint boundaries
- ✅ Continuous positive rewards for progress
- ✅ Correct learning signal

---

## Implementation Plan

**Option 1: Arc-Length Projection (RECOMMENDED)**
- Use existing `_project_onto_segment()` method (currently deprecated)
- Find nearest segment (not nearest waypoint)
- Project vehicle onto segment
- Calculate arc-length from projection to goal
- **Pros**: Smooth, continuous, handles curves correctly
- **Cons**: Slightly more complex (but we already have the code!)

**Option 2: Filtered Distance**
- Keep current method but add smoothing/filtering
- **Pros**: Simple
- **Cons**: Doesn't fix underlying geometric issue

**Recommended**: Use Option 1 (arc-length projection) with dense waypoints

---

## Lessons Learned

1. **Geometric discontinuities matter**: "Nearest point" methods can have jumps
2. **Test at boundaries**: Always test edge cases (waypoint crossings)
3. **Project onto path, not points**: Paths are continuous, points are discrete
4. **Dense waypoints aren't enough**: Still need proper projection for continuity

---

## Next Steps

1. ✅ **Document bug** (this file)
2. ⏹️ **Implement arc-length projection** with dense waypoints
3. ⏹️ **Test at waypoint boundaries** - verify no discontinuities
4. ⏹️ **Validate continuous rewards** - check logs for smooth deltas
5. ⏹️ **Proceed to training** - only after fix validated

---

**Status**: 🔴 **BLOCKING TRAINING** - Fix required immediately

**Previous Bug**: Missing vehicle-to-waypoint distance (FIXED)
**Current Bug**: Discontinuity at waypoint boundaries (THIS BUG)

**Reference**: This is exactly the problem Phase 5 (arc-length projection) was solving!
