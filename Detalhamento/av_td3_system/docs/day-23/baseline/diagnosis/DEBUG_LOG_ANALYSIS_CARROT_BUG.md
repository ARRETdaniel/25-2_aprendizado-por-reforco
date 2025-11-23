# Debug Log Analysis - Carrot Waypoint Bug Discovery

**Date**: 2025-11-23  
**Issue**: Left drift at step ~130  
**Status**: 🎯 **ROOT CAUSE IDENTIFIED!**

---

## TL;DR

The **carrot waypoint is getting stuck at the first waypoint** (317.74, 129.49) after step ~60, causing the vehicle to try steering **backward** (alpha = -180°) which manifests as leftward drift!

**Fix**: Change fallback from `waypoints[0]` to `waypoints[-1]` (last waypoint) for linear paths.

---

## Debug Log Evidence

### Normal Operation (Steps 10-50)

```
[PP-DEBUG Step  10] Carrot=(302.40, 129.49) | Alpha=+0.00° ✅ Forward
[PP-DEBUG Step  20] Carrot=(302.40, 129.49) | Alpha=+0.00° ✅ Forward
[PP-DEBUG Step  30] Carrot=(299.30, 129.49) | Alpha=+0.00° ✅ Forward
[PP-DEBUG Step  40] Carrot=(293.05, 129.49) | Alpha=+0.00° ✅ Forward
[PP-DEBUG Step  50] Carrot=(286.80, 129.49) | Alpha=+0.00° ✅ Forward
```

**Carrot is correctly advancing** as vehicle moves forward.

---

### **THE BUG** (Steps 60+)

```
[PP-DEBUG Step  60] Carrot=(317.74, 129.49) | Alpha=-180.00° ❌ BACKWARD!
[PP-DEBUG Step  70] Carrot=(317.74, 129.49) | Alpha=-180.00° ❌ BACKWARD!
[PP-DEBUG Step  80] Carrot=(317.74, 129.49) | Alpha=-180.00° ❌ BACKWARD!
...
[PP-DEBUG Step 270] Carrot=(317.74, 129.49) | Alpha=-176.58° ❌ BACKWARD!
[PP-DEBUG Step 280] Carrot=(317.74, 129.49) | Alpha=-174.79° ❌ BACKWARD!
[PP-DEBUG Step 290] Carrot=(317.74, 129.49) | Alpha=-172.07° ❌ BACKWARD!
```

**Carrot waypoint stuck at 317.74** - the **FIRST** waypoint in the list!

---

## Root Cause Analysis

### Waypoint List Structure

```python
# waypoints.txt (simplified)
317.74, 129.49, 8.333  # [0] - START (easternmost)
314.74, 129.49, 8.333  # [1]
311.63, 129.49, 8.333  # [2]
...
101.57, 129.49, 8.333  # [69] - Last straight waypoint
98.59,  129.22, 2.5    # [70] - Curve starts
...
92.34,  86.73,  2.5    # [85] - END (southernmost)
```

**Path direction**: East → West (X decreases from 317.74 to 92.34)

---

### The Bug in `_find_carrot_waypoint()`

```python
def _find_carrot_waypoint(...):
    carrot_x, carrot_y = waypoints[0][0], waypoints[0][1]  # ❌ BUG: Default to FIRST
    
    for waypoint in waypoints:
        dist = np.sqrt((waypoint[0] - x_rear)**2 + (waypoint[1] - y_rear)**2)
        
        if dist >= lookahead_distance:
            carrot_x = waypoint[0]
            carrot_y = waypoint[1]
            break
    # If no waypoint found, carrot remains waypoints[0] ❌
    
    return carrot_x, carrot_y
```

**Problem**: When vehicle is far along the path (X < 286.80), **all remaining waypoints** are closer than 15m lookahead. The loop completes **without finding a waypoint**, so carrot defaults to `waypoints[0]` = **(317.74, 129.49)** which is now **BEHIND** the vehicle!

---

### Why Alpha Becomes Negative

**Step 60 State**:
- Vehicle position: X = 297.41 (heading **west**)
- Rear axle: X = 298.91
- Carrot: X = 317.74 (to the **east**, behind vehicle!)
- Yaw: ≈ 180° (west)

**Alpha Calculation**:
```python
alpha = atan2(carrot_y - y_rear, carrot_x - x_rear) - yaw
      = atan2(129.49 - 129.49, 317.74 - 298.91) - 180°
      = atan2(0, 18.83) - 180°
      = 0° - 180°
      = -180°  # Pointing BACKWARD!
```

**Alpha = -180°** means "turn around and go backward!"

---

### Why This Causes Left Drift

As vehicle continues westward:
- Distance to carrot (317.74) increases
- Angle to carrot rotates counterclockwise
- Alpha changes: -180° → -179° → -178° → ... → -172°

**Steering Response**:
```python
steer_rad = atan2(2 * L * sin(alpha), L_d)

# At alpha = -180°:
sin(-180°) = 0 → steer ≈ 0 (straight)

# At alpha = -172° (step 290):
sin(-172°) = -0.139 → steer < 0 (NEGATIVE = steer LEFT!)
```

**Negative alpha** with increasing magnitude → **increasing left steering** → **left drift**!

---

## Progression Timeline

| Step | Vehicle X | Carrot X | Alpha | Steer | Y-Drift | Explanation |
|------|-----------|----------|-------|-------|---------|-------------|
| 50 | 302.68 | 286.80 | 0° | 0.0000 | 0.0mm | Correct ✅ |
| 60 | 297.41 | **317.74** | -180° | -0.0000 | 0.0mm | **BUG STARTS** ❌ |
| 100 | 278.61 | 317.74 | -180° | -0.0000 | 0.3mm | Minimal drift |
| 130 | 265.75 | 317.74 | -179.99° | -0.0001 | 2.2mm | Drift begins |
| 150 | 257.34 | 317.74 | -179.98° | -0.0001 | 5.3mm | Drift accelerating |
| 180 | 244.81 | 317.74 | -179.92° | -0.0005 | 18.1mm | Visible drift |
| 270 | 207.30 | 317.74 | -176.58° | -0.0195 | **704mm** | **OFFROAD!** |
| 290 | 199.02 | 317.74 | -172.07° | -0.0451 | **1614mm** | Disaster |

**Pattern**: As vehicle moves away from stuck carrot, alpha magnitude increases → steering increases → drift compounds exponentially!

---

## Why It Worked in GitHub Code

Looking back at the GitHub code analysis, the same bug exists there too! So why did it work?

**Hypothesis**: The GitHub code might have had:
1. **Shorter route** - ended before bug manifested
2. **Different waypoint spacing** - always had waypoints within lookahead
3. **Circular path** - where wrapping to first waypoint is correct
4. **Different lookahead** - 10m vs 15m means bug appears 50% sooner

Our 15m lookahead + long straight path (220m) + waypoint spacing (~3m) = **inevitable bug after ~60 steps**!

---

## The Fix

### Option 1: Use Last Waypoint (RECOMMENDED) ✅

For **linear paths**, use the last waypoint when no waypoint is beyond lookahead:

```python
def _find_carrot_waypoint(...):
    carrot_x, carrot_y = waypoints[-1][0], waypoints[-1][1]  # ✅ Use LAST waypoint
    
    for waypoint in waypoints:
        dist = np.sqrt((waypoint[0] - x_rear)**2 + (waypoint[1] - y_rear)**2)
        
        if dist >= lookahead_distance:
            carrot_x = waypoint[0]
            carrot_y = waypoint[1]
            break
    # If no waypoint found, carrot is waypoints[-1] ✅
    
    return carrot_x, carrot_y
```

**Why this works**:
- Last waypoint is (101.57, 129.49) - still **ahead** of vehicle until X < 101.57
- Alpha stays near 0° (forward)
- No backward steering
- Natural goal-seeking behavior

---

### Option 2: Closest Waypoint Fallback

Use the closest waypoint instead of first:

```python
def _find_carrot_waypoint(...):
    # Find waypoint beyond lookahead
    for waypoint in waypoints:
        dist = np.sqrt((waypoint[0] - x_rear)**2 + (waypoint[1] - y_rear)**2)
        if dist >= lookahead_distance:
            return waypoint[0], waypoint[1]
    
    # Fallback: find closest waypoint ahead
    min_dist = float('inf')
    closest_wp = waypoints[-1]
    
    for waypoint in waypoints:
        # Only consider waypoints ahead (dot product with heading)
        dx = waypoint[0] - x_rear
        dy = waypoint[1] - y_rear
        ahead = dx * cos(yaw) + dy * sin(yaw)
        
        if ahead > 0:  # Ahead of vehicle
            dist = sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                closest_wp = waypoint
    
    return closest_wp[0], closest_wp[1]
```

**More complex** but handles edge cases better.

---

## Recommendation

**Use Option 1** (last waypoint fallback) because:
1. ✅ Simplest fix (one-line change)
2. ✅ Correct for linear paths
3. ✅ Natural goal-seeking
4. ✅ No performance impact

**Test after fix**:
```bash
# Should see:
# - Steps 1-60: Carrot advances normally
# - Steps 60+: Carrot stays at last straight waypoint (101.57, 129.49)
# - Alpha stays near 0° throughout
# - No left drift!
```

---

## Lessons Learned

1. **Always log intermediate values** - Without seeing carrot position, we'd never have found this!
2. **Waypoint list assumptions matter** - Circular vs linear paths need different logic
3. **Fallback behavior is critical** - A "reasonable" default (first waypoint) caused catastrophic failure
4. **Geometric bugs appear gradually** - Started at step 60, only became obvious at step 130+

---

**Next Step**: Implement the fix and re-run evaluation! 🚀
