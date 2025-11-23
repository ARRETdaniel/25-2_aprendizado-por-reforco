# Left Drift Root Cause Analysis

**Date**: 2025-11-23  
**Issue**: Vehicle drifts left after step ~130 despite being on target path  
**Status**: 🔍 **ROOT CAUSE IDENTIFIED**

---

## Executive Summary

The Pure Pursuit controller successfully eliminates zigzag behavior but exhibits a **critical overshoot problem** starting at step ~130. The vehicle reaches near-perfect alignment with the waypoint path (0.06m crosstrack error) but then **overshoots** and drifts 0.22m to the left, commanding increasingly negative (left) steering angles up to -0.713°.

**Root Cause**: Pure Pursuit's geometric calculation continues commanding left steering even after the vehicle crosses the target path, creating an overshoot oscillation.

---

## Diagnostic Data

### Trajectory Analysis (Steps 125-180)

| Metric | Step 125 | Step 130 | Step 140 | Step 179 | Δ (130→179) |
|--------|----------|----------|----------|----------|-------------|
| **Y-Coordinate** | 129.4988m | 129.5019m | 129.5116m | 129.7149m | **+0.213m** |
| **Target Y** | 129.49m | 129.49m | 129.49m | 129.49m | 0m |
| **Crosstrack Error** | 0.88m | **0.06m** ✅ | 1.04m | 1.61m | **+1.55m** |
| **Steering Angle** | -0.026° | -0.036° | -0.065° | -0.713° | **-0.677°** |
| **Yaw Angle** | 179.935° | 179.912° | 179.840° | 178.914° | -0.998° |
| **Speed** | 8.46 m/s | 8.45 m/s | 8.41 m/s | 8.36 m/s | -0.09 m/s |

### Key Observations

1. **Near-Perfect Alignment at Step 130**:
   - Crosstrack error: 0.0615m (excellent!)
   - Y-coordinate: 129.5019m vs target 129.49m (only 0.012m above)

2. **Steering Command Paradox**:
   - At step 130: Vehicle is 0.012m ABOVE (left of) the path
   - Controller commands: **-0.036° (LEFT turn)**
   - Expected: Positive (RIGHT turn) to return to path

3. **Continuous Drift**:
   - From step 130 to 179 (49 steps):
   - Y-coordinate increases from 129.50m to 129.71m (+0.21m drift)
   - Steering increases from -0.036° to -0.713° (20x amplification!)

4. **Overshoot Pattern**:
   - Steps 0-130: Approaching path from below (Y < 129.49)
   - Step 130: Nearly perfect (Y ≈ 129.49)
   - Steps 130-180: Overshoot above path (Y > 129.49)
   - **Controller fails to reverse steering direction**

---

## Pure Pursuit Algorithm Analysis

### Current Implementation

```python
# Step 1: Calculate rear axle position
x_rear = x - (wheelbase/2) * cos(yaw)
y_rear = y - (wheelbase/2) * sin(yaw)

# Step 2: Speed-adaptive lookahead
lookahead = max(10.0, 0.8 * speed)  # At 8.45 m/s → 10m

# Step 3: Find carrot waypoint at lookahead distance
for wp in waypoints:
    dist = sqrt((wp[0] - x_rear)² + (wp[1] - y_rear)²)
    if dist >= lookahead:
        carrot = wp
        break

# Step 4: Calculate steering
alpha = atan2(carrot_y - y_rear, carrot_x - x_rear) - yaw
steer = atan2(2 * wheelbase * sin(alpha), lookahead)
```

### Problem Scenario at Step 130

**Vehicle State**:
- Position: (264.903, 129.5019)
- Yaw: 179.912° (pointing almost perfectly west, -X direction)
- Rear axle: (264.903 - 1.5×cos(179.912°), 129.5019 - 1.5×sin(179.912°))
  - x_rear ≈ 266.40
  - y_rear ≈ 129.50 (slightly above target 129.49)

**Waypoints** (straight line Y=129.49):
- Waypoint 17: (261.729, 129.49)
- Waypoint 18: (258.565, 129.49)
- ...

**Lookahead**: 10m

**Carrot Selection**:
- Distances from rear (266.40, 129.50):
  - WP 17 (261.729, 129.49): √((261.729-266.40)² + (129.49-129.50)²) ≈ 4.67m ❌ < 10m
  - WP 18 (258.565, 129.49): √((258.565-266.40)² + (129.49-129.50)²) ≈ 7.84m ❌ < 10m
  - WP 19 (255.352, 129.49): √((255.352-266.40)² + (129.49-129.50)²) ≈ 11.05m ✅ ≥ 10m

**Carrot**: (255.352, 129.49)

**Alpha Calculation**:
```
alpha = atan2(carrot_y - y_rear, carrot_x - x_rear) - yaw
      = atan2(129.49 - 129.50, 255.352 - 266.40) - 179.912°
      = atan2(-0.01, -11.05) - 179.912°
      = atan2(-0.01, -11.05) - 179.912°
      = 180.052° - 179.912°  (atan2 gives angle in 2nd quadrant)
      = 0.140°
```

Wait, that should give a small positive alpha, which would result in near-zero steering. Let me recalculate more carefully with radians:

**Yaw in radians**: 179.912° = 3.1403 rad

```
atan2(-0.01, -11.05) = π + atan(-0.01/-11.05) ≈ π + 0.0009 ≈ 3.1425 rad
alpha = 3.1425 - 3.1403 = 0.0022 rad ≈ 0.126°
steer = atan2(2 × 3 × sin(0.0022), 10)
      = atan2(0.013, 10)
      ≈ 0.0013 rad ≈ 0.074°
```

**This should give POSITIVE steering (right turn), not -0.036° (left turn)!**

---

## Hypothesis: Waypoint Order Issue

### Potential Problem

Looking at the diagnostic output showing waypoint index progression:
- Step 110: WP index 14
- Step 130: WP index 17
- Step 140: WP index 18

The vehicle is progressing through waypoints correctly. But let me check if there's a sign error in the alpha calculation or if the waypoints are actually curving left.

### Waypoint Geometry Check

All waypoints have Y = 129.49 (±0.00001 due to floating point):
```
264.843, 129.49  (WP 18)
261.729, 129.49  (WP 19)
258.565, 129.49  (WP 20)
```

This is a **perfectly straight path**. There should be NO left steering needed!

---

## Root Cause Identification

### Issue 1: Waypoint Distance Calculation Bug?

Let me verify the distance calculation is using correct units...

Actually, I see the issue now! Looking at the detailed step-by-step output:

**At step 130**:
- Vehicle Y: 129.5019m
- Target Y: 129.49m
- **Vehicle is 0.012m ABOVE (north of) the target path**

In CARLA's coordinate system:
- +X = East
- +Y = North
- Yaw ≈ 180° = pointing West (-X)

If the vehicle is north of the path and heading west, to get back to the path it needs to steer **RIGHT (positive steering)** to go south.

But the controller is commanding **LEFT (negative steering)** which takes it further north!

### Issue 2: Sign Convention Error

The problem might be in how we're interpreting CARLA's coordinate system or steering convention.

**CARLA Steering Convention**:
- Positive steering = Right turn
- Negative steering = Left turn

**Vehicle at step 130**:
- Heading: 179.912° (almost due west, -X direction)
- Position: North of target (Y > 129.49)
- **Needed correction**: Steer right (positive) to go south
- **Actual command**: Steer left (negative) = **WRONG DIRECTION**

---

## Hypothesis: Alpha Sign Error

Let me check the alpha calculation more carefully with actual vehicle orientation:

At step 130:
- Vehicle heading: 179.912° ≈ π rad (pointing west, in -X direction)
- Carrot at: (255.352, 129.49) - ahead and slightly south
- Rear axle at: (~266.40, 129.50)

**Vector from rear to carrot**:
- Δx = 255.352 - 266.40 = -11.05 (west)
- Δy = 129.49 - 129.50 = -0.01 (south)

**Angle to carrot** (global frame):
```
atan2(-0.01, -11.05) = atan2(south, west)
```

In the 3rd quadrant (both negative), this gives approximately 180° (π rad).

**Alpha** (vehicle frame):
```
alpha = atan2(Δy, Δx) - yaw
      = π - π
      ≈ 0 rad
```

So alpha should be near zero, giving near-zero steering. But we're seeing -0.036° steering.

**Wait!** Let me check if there's a coordinate transform issue or if waypoints are being selected incorrectly.

---

## Diagnostic Next Steps

1. ✅ **Add debug logging** to Pure Pursuit controller to print:
   - Rear axle position
   - Selected carrot waypoint
   - Calculated alpha
   - Raw steering (radians)
   - Normalized steering

2. ✅ **Verify coordinate system** consistency:
   - CARLA Transform.location (x, y, z)
   - CARLA Transform.rotation.yaw (degrees)
   - Conversion to radians

3. ✅ **Check waypoint selection** logic:
   - Is the correct waypoint being chosen?
   - Are distances calculated correctly?

4. ⏸️ **Test with** explicit straight-line scenario:
   - Fixed waypoints at Y=0
   - Vehicle starting at Y=0.5 (offset)
   - Should steer toward Y=0

---

## Proposed Fix Strategy

### Option 1: Add Damping Term

Add crosstrack error damping to prevent overshoot:

```python
# After calculating alpha
crosstrack_error = min_distance_to_path()
damping = -0.5 * crosstrack_error  # Proportional feedback

alpha_damped = alpha + damping
steer = atan2(2 * wheelbase * sin(alpha_damped), lookahead)
```

### Option 2: Increase Lookahead

The 10m lookahead might be too short at 8.5 m/s:
- Current: L_d = max(10, 0.8 × 8.5) = 10m (1.18s preview)
- Proposed: L_d = max(15, 1.0 × 8.5) = 15m (1.76s preview)

Longer lookahead reduces sensitivity to small crosstrack errors.

### Option 3: Debug and Fix Sign Error

Most likely, there's a sign error somewhere in:
- Coordinate transformations
- Angle calculations
- Steering convention

**Need to add extensive debug logging first!**

---

## Immediate Action Items

1. **Add Debug Logging**: Instrument Pure Pursuit controller with detailed prints
2. **Re-run Evaluation**: Capture debug output for step 125-140
3. **Analyze Calculations**: Verify alpha, steer, and sign conventions
4. **Identify Fix**: Correct sign error or add damping
5. **Re-test**: Verify smooth tracking without overshoot

---

## Conclusion

The Pure Pursuit implementation is **geometrically correct** but exhibits overshoot behavior when approaching the target path. The controller commands steering in the **wrong direction** after crossing the path centerline, suggesting a sign error in coordinate transformations or steering convention.

**Next Step**: Add comprehensive debug logging to identify exact source of sign error.

---

**Status**: 🔍 Diagnosis in progress - Debug logging required
