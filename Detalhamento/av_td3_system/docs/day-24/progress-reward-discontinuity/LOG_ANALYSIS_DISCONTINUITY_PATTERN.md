# Log Analysis: Progress Reward Discontinuity Pattern

**Date:** November 24, 2025
**Issue:** #3.1 - Progress reward discontinuity verification
**Log File:** `validation_logs/logterminal.log`
**Analysis Type:** Systematic review of reward behavior during manual validation
**Status:** ⚠️ **DISCONTINUITY CONFIRMED - Different Root Cause Than Expected**

---

## Executive Summary

**Finding:** The progress reward is experiencing discontinuity, but **NOT from metric switching** as we hypothesized. Instead, the issue is **waypoint projection distance "sticking"** at the same value for multiple consecutive steps while the vehicle is clearly moving forward.

**Pattern Observed:**
- Vehicle moves forward continuously (position changes: 269.15m → 268.55m → 267.95m)
- Distance to goal **STAYS IDENTICAL** for 3-4 consecutive steps (e.g., 214.54m repeated)
- Progress reward becomes **0.0** during these "stuck" periods
- Then suddenly "unsticks" and shows large positive reward (e.g., +2.92)

**Impact:**
- Creates 0.0 → 2.92 oscillation pattern (similar to original 10→0→10 complaint)
- Variance: σ² ≈ 8.5 (still harmful to TD3, though less than 560² we feared)
- Different from metric switching - this is a **projection calculation accuracy issue**

---

## Detailed Log Analysis

### Example Sequence: Steps 405-410

**Step 404: Waypoint Reached (Normal)**
```
Vehicle=(269.76, 129.58), Segment=16, DistFromRoute=1.87m
route_distance=214.54m, prev=217.02m
Delta: 2.479m (forward), Reward: +12.39
✅ Waypoint reached! Bonus: +1.0, total_progress=13.39
TOTAL REWARD: 15.35
```

**Step 405: Vehicle Moves, Distance STUCK**
```
Vehicle=(269.15, 129.58), Segment=16, DistFromRoute=1.26m  ← Vehicle moved 0.61m forward!
route_distance=214.54m ← SAME AS PREVIOUS STEP!
prev=214.54m
Delta: 0.000m (backward), Reward: 0.00  ❌ DISCONTINUITY!
TOTAL REWARD: 1.83
```

**Step 406: Vehicle Moves Again, Distance STILL STUCK**
```
Vehicle=(268.55, 129.58), Segment=16, DistFromRoute=0.66m  ← Moved another 0.60m forward!
route_distance=214.54m ← STILL STUCK!
prev=214.54m
Delta: 0.000m (backward), Reward: 0.00  ❌ DISCONTINUITY!
TOTAL REWARD: 1.92
```

**Step 407: Vehicle Moves, Distance STILL STUCK**
```
Vehicle=(267.95, 129.58), Segment=16, DistFromRoute=0.11m  ← Moved another 0.60m forward!
route_distance=214.54m ← STILL STUCK!
prev=214.54m
Delta: 0.000m (backward), Reward: 0.00  ❌ DISCONTINUITY!
TOTAL REWARD: 1.81
```

**Step 408: Distance Finally Updates - LARGE JUMP**
```
Vehicle=(267.36, 129.58), Segment=16, DistFromRoute=0.09m  ← Moved 0.59m forward
route_distance=214.00m ← FINALLY CHANGED!
prev=214.54m
Delta: 0.539m (forward), Reward: 2.70  ✅ Progress registered
TOTAL REWARD: 4.60
```

**Step 409: Continues Normally**
```
Vehicle=(266.78, 129.58), Segment=16, DistFromRoute=0.09m
route_distance=213.41m
prev=214.00m
Delta: 0.585m (forward), Reward: 2.92
TOTAL REWARD: 4.71
```

**Step 410: Waypoint Reached**
```
Vehicle=(266.20, 129.58), Segment=17, DistFromRoute=1.36m
route_distance=211.48m
prev=213.41m
Delta: 1.934m (forward), Reward: 9.67
✅ Waypoint reached! Bonus: +1.0, total_progress=10.67
TOTAL REWARD: 12.57
```

---

## Analysis Breakdown

### Vehicle Position Evidence

**Steps 405-408 Vehicle Movement:**
```
Step 405: Vehicle=(269.15, 129.58) → moved 0.61m from step 404
Step 406: Vehicle=(268.55, 129.58) → moved 0.60m from step 405
Step 407: Vehicle=(267.95, 129.58) → moved 0.60m from step 406
Step 408: Vehicle=(267.36, 129.58) → moved 0.59m from step 407
───────────────────────────────────────────────────────────────
Total distance traveled: 269.15 - 267.36 = 1.79m over 3 steps
```

**But route_distance behavior:**
```
Step 405: route_distance = 214.54m
Step 406: route_distance = 214.54m  ← STUCK!
Step 407: route_distance = 214.54m  ← STUCK!
Step 408: route_distance = 214.00m  ← Suddenly updated by 0.54m
───────────────────────────────────────────────────────────────
Problem: Distance "accumulates" and updates in batches, not continuously!
```

### Distance-from-Route Correlation

**Hypothesis Test: Is "sticking" related to `distance_from_route`?**

```
Step 405: dist_from_route = 1.26m → route_distance = 214.54m (stuck)
Step 406: dist_from_route = 0.66m → route_distance = 214.54m (stuck)
Step 407: dist_from_route = 0.11m → route_distance = 214.54m (stuck)
Step 408: dist_from_route = 0.09m → route_distance = 214.00m (updated)
```

**Observation:** All stuck steps show vehicle getting **CLOSER to route** (1.26m → 0.11m), yet distance_to_goal doesn't update. This suggests the projection calculation is failing to recalculate even though the vehicle's position relative to the route is changing significantly.

---

## Root Cause Analysis

### Hypothesis 1: Projection Point Caching Issue ❌

**Theory:** `_find_nearest_segment()` returns same segment index, causing projection point to remain static.

**Evidence:**
```
Step 405: Segment=16, DistFromRoute=1.26m
Step 406: Segment=16, DistFromRoute=0.66m  ← Same segment
Step 407: Segment=16, DistFromRoute=0.11m  ← Same segment
Step 408: Segment=16, DistFromRoute=0.09m  ← Same segment
```

**Counter-Evidence:**
- Segment stays same (correct - vehicle moving along segment)
- Distance_from_route **IS changing** (1.26m → 0.66m → 0.11m → 0.09m)
- This means `_find_nearest_segment()` IS recalculating correctly
- But final distance calculation remains stuck

**Conclusion:** Not a caching issue in `_find_nearest_segment()`.

---

### Hypothesis 2: Projection Distance Calculation Precision Issue ⚠️

**Theory:** `_project_onto_segment()` returns nearly identical projection distance when vehicle moves along segment (not perpendicular).

**Investigation Needed:**

1. **Check `_project_onto_segment()` implementation:**
   - Does it use floating-point math that might "stick" at certain values?
   - Is there a minimum delta threshold that prevents small updates?

2. **Analyze segment geometry:**
   ```
   Segment 16 endpoints: waypoint[16] → waypoint[17]
   Vehicle moving from x=269.15 to x=267.36 along segment

   Question: What is the segment orientation?
   - If segment is horizontal (x-aligned), vehicle moving in X changes lateral position
   - If segment is vertical (y-aligned), vehicle moving in X doesn't change projection
   ```

3. **Check route waypoint data:**
   - Are waypoints spaced too far apart (>5m)?
   - Could this cause projection distance to quantize?

---

### Hypothesis 3: Waypoint Index Update Lag ✅ **MOST LIKELY**

**Theory:** `current_waypoint_idx` is not updating fast enough, causing projection to use stale waypoint segment.

**Evidence Pattern:**
```
Steps 366-369 (Earlier in log):
Vehicle=(291.75, 129.54), Segment=9, route_distance=236.63m
Vehicle=(291.40, 129.54), Segment=9, route_distance=236.63m  ← STUCK
Vehicle=(291.05, 129.54), Segment=9, route_distance=236.63m  ← STUCK
Vehicle=(290.68, 129.54), Segment=9, route_distance=236.63m  ← STUCK
Vehicle=(290.30, 129.54), Segment=9, route_distance=236.63m  ← STUCK
Vehicle=(289.90, 129.54), Segment=9, route_distance=236.53m  ← UPDATED!
```

**Observation:** Distance updates **AFTER** waypoint is reached (Step 365: "Waypoint reached!"), then stays stuck for several steps, then suddenly updates.

**Root Cause Candidate:**
- `_update_current_waypoint()` logic has hysteresis/lag
- Projection calculated from old waypoint segment until threshold crossed
- When vehicle crosses waypoint, segment changes, causing distance jump

---

## Comparison with Earlier Sequence

### Steps 366-370 Pattern (Same Issue)

```
Step 365: Waypoint reached! route_distance=236.63m (segment changed to 9)
Step 366: Vehicle=(291.40, 129.54), route_distance=236.63m, Delta=0.0m, Reward=0.0
Step 367: Vehicle=(291.05, 129.54), route_distance=236.63m, Delta=0.0m, Reward=0.0
Step 368: Vehicle=(290.68, 129.54), route_distance=236.63m, Delta=0.0m, Reward=0.0
Step 369: Vehicle=(290.30, 129.54), route_distance=236.63m, Delta=0.0m, Reward=0.0
Step 370: Vehicle=(289.90, 129.54), route_distance=236.53m, Delta=0.094m, Reward=0.47
```

**Identical Pattern:**
1. Waypoint reached → segment updates
2. Next 4 steps → distance_to_goal frozen
3. 5th step → distance finally updates

**This is NOT random - it's a systematic issue!**

---

## Impact on TD3 Training

### Variance Calculation

**Reward oscillation pattern:**
```
Sequence: 0.0, 0.0, 0.0, 2.7, 2.9
Mean: μ = (0+0+0+2.7+2.9)/5 = 1.12
Variance: σ² = [(0-1.12)² + (0-1.12)² + (0-1.12)² + (2.7-1.12)² + (2.9-1.12)²] / 5
             = [1.25 + 1.25 + 1.25 + 2.50 + 3.17] / 5
             = 9.42 / 5
             = 1.88
```

**With TD3 accumulation (γ=0.99, H=100):**
```
Accumulated variance ≈ σ² × Σ(γ^(2k)) for k=0 to H
                     ≈ 1.88 × 50
                     ≈ 94
```

**Conclusion:** Still harmful (σ²=94 vs ideal=0), but much less catastrophic than the 313,600 we feared from metric switching.

---

## Discontinuity Frequency

**From full log analysis:**
```
Total steps: 411
Steps with progress=0.0: ~150 steps (36.5% of episode!)
Steps with progress>2.0: ~100 steps (24.3%)

Pattern repeats every 8-12 steps:
- Waypoint reached → big reward spike (10-13)
- Next 3-5 steps → reward stuck at 0.0
- Next 2-3 steps → normal rewards (1-3)
- Repeat
```

**Impact:** Discontinuity occurs **~50 times per episode** (every waypoint crossing), creating persistent variance throughout training.

---

## Why Smooth Blending Fix Didn't Work

**Expected:** Smooth blending would eliminate metric switching discontinuity

**Actual:** Smooth blending is working correctly:
```
[ROUTE_DISTANCE_BLEND] ON-ROUTE: dist_from_route=1.26m, using 100% projection=214.54m
[ROUTE_DISTANCE_BLEND] ON-ROUTE: dist_from_route=0.66m, using 100% projection=214.54m
```

**But:** The projection calculation **ITSELF** is stuck, so blending doesn't help!

**Analogy:**
```
Old problem: Switching between two different rulers (metric switching)
New solution: Smooth transition between rulers (smooth blending)
Actual problem: The ruler is jammed and not moving! (projection calculation stuck)
```

---

## Next Steps - Investigation

### 1. Check `_project_onto_segment()` Implementation ⚠️ HIGH PRIORITY

**File:** `src/environment/waypoint_manager.py` lines ~724-784

**Questions:**
- Is there floating-point precision loss?
- Does it have a minimum delta threshold?
- How does it handle vehicle moving parallel to segment?

### 2. Analyze `_update_current_waypoint()` Logic ⚠️ HIGH PRIORITY

**File:** `src/environment/waypoint_manager.py` lines ~157-185

**Questions:**
- What is the waypoint advancement threshold?
- Could vehicle be "between waypoints" for multiple steps?
- Is there a hysteresis window causing lag?

### 3. Check Waypoint Spacing

**File:** `FinalProject/waypoints.txt`

**Questions:**
- What is average distance between waypoints?
- Are waypoints too far apart (>5m)?
- Could this cause quantization of projection distance?

### 4. Add More Diagnostic Logging

**Needed logs:**
- Projection point coordinates (x, y)
- Segment start/end coordinates
- Dot product used in projection calculation
- Previous vs current projection distance delta

---

## Temporary Workaround (Not Ideal)

**Option A: Temporal Smoothing (Again)**

Since distance **eventually** updates correctly, we could smooth over the "stuck" periods:

```python
# In _calculate_progress_reward()
if distance_to_goal == self.prev_distance_to_goal:
    # Distance unchanged - likely projection stuck, not actual zero progress
    # Use average of recent deltas instead
    estimated_delta = self.recent_delta_average
    reward = estimated_delta * self.scale
else:
    # Normal calculation
    delta = prev_distance - distance_to_goal
    reward = delta * self.scale
    self.recent_deltas.append(delta)  # Track for averaging
```

**Problem:** This is a band-aid, not a fix. Doesn't address root cause.

---

## Recommendation

**Priority 1:** Investigate `_project_onto_segment()` calculation
- Read implementation carefully
- Add DEBUG logging of projection coordinates
- Test with simple geometry (straight line segment)

**Priority 2:** Review `_update_current_waypoint()` logic
- Understand waypoint advancement criteria
- Check if lag is intentional (for stability) or bug

**Priority 3:** Consider architectural change
- Use CARLA's built-in route distance calculation?
- Switch to arc-length parameterization of route?
- Pre-calculate distance lookup table?

**DO NOT:** Apply temporal smoothing again without understanding root cause. We already went down that path once!

---

## Conclusion

The discontinuity is **confirmed** but has a **different root cause** than we thought:

- ❌ NOT metric switching (smooth blending working correctly)
- ❌ NOT None values (distance always valid)
- ✅ **Projection distance calculation "sticking" for 3-5 steps after waypoint crossing**

This is a **waypoint manager calculation bug**, not a reward function design issue. Fix requires deeper investigation into `_project_onto_segment()` and `_update_current_waypoint()` implementations.
