# Simple Fix Proposal: Progress Reward Discontinuity

**Date:** November 24, 2025
**Issue:** Reward discontinuity persists after smooth blending implementation
**Status:** PROPOSED SOLUTION

---

## Problem Analysis

**User Reports:**
1. ✅ No debug logs appear from `[ROUTE_DISTANCE_BLEND]` during manual validation
2. ✅ Discontinuity still exists (10.0 → 0.0 → 10.0 oscillation)

**Root Cause (Updated):**

After reviewing the code and CARLA documentation, the actual issue is:

### Issue 1: Logging Not Visible
```python
# waypoint_manager.py uses DEBUG level
self.logger.debug("[ROUTE_DISTANCE_BLEND] ...")

# But validate_rewards_manual.py doesn't configure logging level!
# Default level is WARNING, so DEBUG logs are suppressed
```

### Issue 2: Simple Math Error in Distance Calculation

Looking at the projection method in `get_route_distance_to_goal()`:

```python
# Current implementation calculates TWO distances:
# 1. projection_distance (from projection point to goal)
# 2. euclidean_distance (from vehicle to goal)

# Then blends them:
final_distance = (1 - blend_factor) * projection_distance + blend_factor * euclidean_distance
```

**The Problem:**
When vehicle is at coordinates (x1, y1) and we project onto segment to get (x2, y2):
- `projection_distance` starts from (x2, y2) to goal
- `euclidean_distance` starts from (x1, y1) to goal
- These are **different starting points**!

This creates discontinuity because:
- When on-route: Uses projection_distance (starts from road)
- When off-route: Uses euclidean_distance (starts from actual position)
- **Jump occurs because starting points differ!**

---

## Simple Fix: Use Single Starting Point

Instead of blending two distances with different origins, **always measure from vehicle position**, just change the PATH taken:

```python
def get_route_distance_to_goal(self, vehicle_location):
    """Calculate distance to goal using smooth path interpolation."""

    # Vehicle position
    if hasattr(vehicle_location, 'x'):
        vx, vy = vehicle_location.x, vehicle_location.y
    else:
        vx, vy = vehicle_location[0], vehicle_location[1]

    # Find nearest segment
    segment_idx, distance_from_route = self._find_nearest_segment(vehicle_location)

    # Goal position
    goal_x, goal_y, _ = self.waypoints[-1]

    # METHOD 1: On-route - follow waypoint path
    if distance_from_route <= 5.0 and segment_idx is not None:
        # Distance from vehicle to nearest point on route
        wp_start = self.waypoints[segment_idx]
        wp_end = self.waypoints[segment_idx + 1]
        projection = self._project_onto_segment((vx, vy), (wp_start[0], wp_start[1]), (wp_end[0], wp_end[1]))

        dist_to_route = math.sqrt((projection[0] - vx)**2 + (projection[1] - vy)**2)

        # Distance along route from projection to goal
        dist_projection_to_segment_end = math.sqrt(
            (wp_end[0] - projection[0])**2 + (wp_end[1] - projection[1])**2
        )

        # Remaining segments
        remaining = 0.0
        for i in range(segment_idx + 1, len(self.waypoints) - 1):
            wp1 = self.waypoints[i]
            wp2 = self.waypoints[i + 1]
            remaining += math.sqrt((wp2[0] - wp1[0])**2 + (wp2[1] - wp1[1])**2)

        route_distance = dist_to_route + dist_projection_to_segment_end + remaining
        return route_distance

    # METHOD 2: Off-route - direct euclidean
    else:
        euclidean_distance = math.sqrt((goal_x - vx)**2 + (goal_y - vy)**2)
        return euclidean_distance
```

**Why This Works:**
- ✅ Both methods start from **vehicle position** (vx, vy)
- ✅ No blending needed - clean switch
- ✅ No discontinuity because same origin
- ✅ On-route: `dist_to_route + path_along_route`
- ✅ Off-route: `straight_line_distance`

---

## Even Simpler Fix: Just Fix the Logger

If the blending algorithm is actually correct but logs aren't showing:

```python
# In validate_rewards_manual.py - ADD THIS AT THE TOP
import logging

# Set root logger to DEBUG
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

**Or add command-line argument:**
```bash
python scripts/validate_rewards_manual.py \
    --config config/baseline_config.yaml \
    --log-level DEBUG  # ← Add this option
```

Then implement in script:
```python
parser.add_argument('--log-level', type=str, default='INFO',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                    help='Logging level')

# Apply it
logging.basicConfig(level=getattr(logging, args.log_level))
```

---

## Recommended Action

**Option A (Safest):** Fix logger first to see what's actually happening
1. Add logging configuration to `validate_rewards_manual.py`
2. Re-run test with DEBUG logs visible
3. Analyze if smooth blending is actually executing

**Option B (If blending confirmed broken):** Implement simpler distance calculation
1. Replace smooth blending with single-origin distance
2. Test again
3. Compare results

---

## Expected Results After Fix

### With DEBUG Logging Enabled:

```bash
$ python scripts/validate_rewards_manual.py --log-level DEBUG

[ROUTE_DISTANCE_BLEND] ON-ROUTE: dist_from_route=1.20m, using 100% projection=45.30m
[ROUTE_DISTANCE_BLEND] ON-ROUTE: dist_from_route=1.35m, using 100% projection=44.80m
[ROUTE_DISTANCE_BLEND] TRANSITION: dist_from_route=7.50m, blend=0.17, projection=44.20m, euclidean=42.10m, final=43.85m
[ROUTE_DISTANCE_BLEND] FAR OFF-ROUTE: dist_from_route=22.00m, using 100% Euclidean=38.50m
```

### Success Criteria:

- ✅ Logs appear in terminal
- ✅ No sudden 10→0 jumps in reward
- ✅ Smooth transitions visible in logs
- ✅ `blend_factor` changes gradually (0.0 → 1.0)
- ✅ Progress reward stays continuous

---

## Implementation

Choose one:

### Quick Fix (5 minutes):
Add logging to validation script - see if blending is working

### Proper Fix (30 minutes):
If blending isn't working, implement single-origin distance calculation

Let me know which approach you prefer!
