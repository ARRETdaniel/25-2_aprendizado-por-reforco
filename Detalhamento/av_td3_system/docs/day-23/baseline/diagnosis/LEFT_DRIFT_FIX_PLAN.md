# Left Drift Fix - Implementation Plan

**Date**: 2025-11-23
**Issue**: Pure Pursuit overshoots target path causing leftward drift
**Priority**: 🔴 **HIGH** - Blocks baseline validation

---

## Problem Summary

The Pure Pursuit controller successfully eliminates zigzag behavior BUT exhibits overshoot when crossing the target path:

- **Steps 0-130**: Vehicle approaches path from right (Y < 129.49), steering LEFT appropriately
- **Step 130**: Nearly perfect alignment (0.06m crosstrack error) ✅
- **Steps 130-180**: Vehicle overshoots to left (Y > 129.49), but continues steering LEFT ❌
- **Result**: Y-coordinate drifts from 129.50m to 129.71m (+0.22m drift), steering increases to -0.713°

**Root Cause**: Pure Pursuit's geometric calculation doesn't include damping for momentum/overshoot.

---

## Diagnostic Evidence

### Trajectory Visualization

See generated plots:
- `docs/day-23/baseline/diagnosis/episode_0_deviation_diagnosis.png`
- Shows clear overshoot pattern starting at step ~130

### Key Metrics (Steps 130-180)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Y Drift** | +0.213m | Leftward deviation from target (129.49m) |
| **Steering Increase** | -0.036° → -0.713° | 20x amplification (wrong direction!) |
| **Crosstrack Growth** | 0.06m → 1.61m | Error increases 27x after crossing |
| **Yaw Deviation** | -0.998° | Heading drifts left of target (180°) |

---

## Recommended Fix: Add Crosstrack Damping

### Concept

Add a proportional damping term based on crosstrack error to prevent overshoot:

```python
# Traditional Pure Pursuit (current - causes overshoot)
alpha = atan2(carrot_y - y_rear, carrot_x - x_rear) - yaw
steer = atan2(2 * L * sin(alpha), lookahead)

# With Crosstrack Damping (proposed - prevents overshoot)
alpha = atan2(carrot_y - y_rear, carrot_x - x_rear) - yaw

# Calculate crosstrack error (perpendicular distance to path)
crosstrack_error = calculate_crosstrack_error(position, waypoints)

# Add damping proportional to crosstrack error
# When vehicle is left of path (+error), add right steering (+damping)
# When vehicle is right of path (-error), add left steering (-damping)
k_damping = 0.3  # Tuning parameter
alpha_damped = alpha - k_damping * crosstrack_error / lookahead

steer = atan2(2 * L * sin(alpha_damped), lookahead)
```

### Why This Works

1. **Near Path** (crosstrack ≈ 0): Damping ≈ 0, Pure Pursuit dominates
2. **Off Path** (crosstrack > 0): Damping adds corrective steering
3. **Prevents Overshoot**: As vehicle crosses path, damping reverses direction
4. **Maintains Stability**: Damping scales with lookahead (same units as alpha)

### Mathematical Justification

**Pure Pursuit alpha** represents the angle to reach the carrot waypoint.
**Damping term** `-k × (crosstrack / lookahead)` represents proportional correction.

**Combined effect**:
- When approaching path from right: Both terms steer left (additive)
- When crossing path: Damping reverses, opposes Pure Pursuit (preventive)
- When stabilized on path: Both terms near zero (stable equilibrium)

---

## Implementation Steps

### Step 1: Add Crosstrack Error Calculation

```python
def _calculate_crosstrack_error(
    self,
    x: float,
    y: float,
    waypoints: List[Tuple[float, float, float]]
) -> float:
    """
    Calculate perpendicular distance from vehicle to waypoint path.

    For a straight path (Y = constant), this simplifies to:
        crosstrack_error = current_y - target_y

    For curved paths, use point-to-line distance formula.

    Returns:
        Crosstrack error in meters
        Positive = left of path
        Negative = right of path
    """
    # Find closest waypoint
    min_dist = float('inf')
    closest_wp_idx = 0

    for i, wp in enumerate(waypoints):
        dist = np.sqrt((wp[0] - x)**2 + (wp[1] - y)**2)
        if dist < min_dist:
            min_dist = dist
            closest_wp_idx = i

    # For straight-line path (all waypoints have same Y)
    # Crosstrack error is simply the Y-difference
    if closest_wp_idx < len(waypoints) - 1:
        # Use path segment direction for more accurate calculation
        wp1 = waypoints[closest_wp_idx]
        wp2 = waypoints[min(closest_wp_idx + 1, len(waypoints) - 1)]

        # Path vector
        path_dx = wp2[0] - wp1[0]
        path_dy = wp2[1] - wp1[1]
        path_length = np.sqrt(path_dx**2 + path_dy**2)

        if path_length > 0.001:  # Avoid division by zero
            # Unit path vector
            path_unit_x = path_dx / path_length
            path_unit_y = path_dy / path_length

            # Vector from wp1 to vehicle
            vehicle_dx = x - wp1[0]
            vehicle_dy = y - wp1[1]

            # Cross product gives signed distance
            # (perpendicular to path)
            crosstrack = vehicle_dx * path_unit_y - vehicle_dy * path_unit_x
            return crosstrack

    # Fallback: simple Y-difference (for straight horizontal paths)
    return y - waypoints[closest_wp_idx][1]
```

### Step 2: Modify `update()` Method

```python
def update(
    self,
    current_x: float,
    current_y: float,
    current_yaw: float,
    current_speed: float,
    waypoints: List[Tuple[float, float, float]]
) -> float:
    # ... (existing steps 1-4) ...

    # Step 4: Calculate angle from rear axle to carrot waypoint
    alpha = self._normalize_angle(
        np.arctan2(carrot_y - y_rear, carrot_x - x_rear) - current_yaw
    )

    # NEW: Step 4.5: Add crosstrack damping
    crosstrack_error = self._calculate_crosstrack_error(
        current_x, current_y, waypoints
    )

    # Damping gain (tunable parameter)
    k_damping = 0.3

    # Apply damping: reduces alpha when vehicle is off path
    # Sign convention: positive crosstrack (left of path) → reduce alpha (steer right)
    alpha_damped = alpha - k_damping * crosstrack_error / lookahead_distance

    # Step 5: Pure Pursuit steering formula (with damped alpha)
    steer_rad = np.arctan2(
        2.0 * self.wheelbase * np.sin(alpha_damped),
        lookahead_distance
    )

    # ... (rest of method unchanged) ...
```

### Step 3: Add Configuration Parameter

Update `config/baseline_config.yaml`:

```yaml
pure_pursuit:
  kp_lookahead: 0.8
  min_lookahead: 10.0
  wheelbase: 3.0
  k_damping: 0.3  # NEW: Crosstrack damping gain
```

---

## Alternative Fixes (if damping doesn't work)

### Option B: Increase Lookahead Distance

Increase minimum lookahead to reduce sensitivity:

```yaml
pure_pursuit:
  kp_lookahead: 1.0  # Was 0.8
  min_lookahead: 15.0  # Was 10.0
```

**Pros**: Simple, no algorithm changes
**Cons**: Slower response to sharp turns, may cut corners

### Option C: Add Low-Pass Filter to Steering

Smooth steering commands with exponential moving average:

```python
# Add to __init__
self.steer_previous = 0.0
self.alpha_filter = 0.7  # Filter coefficient

# In update()
steer_raw = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), lookahead_distance)
steer_filtered = self.alpha_filter * steer_raw + (1 - self.alpha_filter) * self.steer_previous
self.steer_previous = steer_filtered
return steer_filtered
```

**Pros**: Reduces steering oscillations
**Cons**: Adds lag, may reduce responsiveness

---

## Testing Plan

### Test 1: Baseline Comparison

Run 3 episodes with damping disabled and enabled:

```bash
# Without damping (current - shows overshoot)
python3 scripts/evaluate_baseline.py --scenario 0 --num-episodes 3

# With damping (k_damping=0.3)
python3 scripts/evaluate_baseline.py --scenario 0 --num-episodes 3
```

**Success Criteria**:
- Y-coordinate stays within ±0.1m of 129.49m
- No steering amplification (max |steer| < 0.2°)
- Crosstrack error < 0.5m throughout

### Test 2: Damping Gain Tuning

Test different k_damping values:

| k_damping | Expected Behavior |
|-----------|-------------------|
| 0.0 | No damping (current overshoot) |
| 0.1 | Light damping (may still overshoot) |
| 0.3 | Moderate damping (recommended) |
| 0.5 | Strong damping (may be sluggish) |
| 1.0 | Very strong (may oscillate) |

**Optimal**: Choose lowest k_damping that prevents overshoot.

### Test 3: Curved Path Test

After fixing straight-line overshoot, test on curved waypoints to ensure damping doesn't interfere with cornering.

---

## Expected Results

### Before Fix (Current)

| Metric | Value |
|--------|-------|
| Mean Lateral Deviation | 0.8-1.0m |
| Max Y Drift | 0.22m |
| Max Steering | -0.713° |
| Behavior | Overshoot oscillation |

### After Fix (With Damping)

| Metric | Value |
|--------|-------|
| Mean Lateral Deviation | **0.2-0.4m** (50-60% improvement) |
| Max Y Drift | **< 0.05m** (75% improvement) |
| Max Steering | **< 0.2°** (70% improvement) |
| Behavior | **Smooth convergence** ✅ |

---

## Implementation Priority

**Recommended Approach**: **Option A (Crosstrack Damping)**

**Timeline**:
1. Implement damping calculation (30 min)
2. Update controller (15 min)
3. Test and tune k_damping (45 min)
4. Validate on 10-episode run (30 min)

**Total**: ~2 hours

**Decision Point**: If damping doesn't work after tuning, fall back to Option B (increase lookahead).

---

## Next Steps

1. ✅ Diagnostic analysis complete
2. ⏸️ **Implement crosstrack damping** in pure_pursuit_controller.py
3. ⏸️ **Add k_damping parameter** to baseline_config.yaml
4. ⏸️ **Run test evaluation** (3 episodes)
5. ⏸️ **Tune k_damping** if needed
6. ⏸️ **Validate fix** with 10-episode run
7. ⏸️ **Proceed to Phase 4** (NPC interaction)

---

**Status**: 📋 **FIX DESIGNED** - Ready for implementation
