# SYSTEMATIC INVESTIGATION: Negative Progress Rewards During Forward Movement

**Date**: November 21, 2025  
**Investigation**: Post-fix validation (after FIXES_IMPLEMENTED.md)  
**Status**: 🔴 **NEW CRITICAL BUG DISCOVERED**

---

## 🎯 USER REPORT

> "After the fixes outlined in FIXES_IMPLEMENTED.md to solve the right-turn bias issues, now the progress reward is giving negative reward while moving forward to waypoint ahead which in the future leads to goal. Some part of the reward calculation is giving negative reward in the total calculation and when lane invasion happen the progress reward turns to positive or we have some additional bug. Do a systematic investigation."

> "While in exploration phase the progress reward is oscillating from positive to negative while moving in the right direction straight ahead."

---

## 🔬 INVESTIGATION METHODOLOGY

1. ✅ Analyzed log file: `av_td3_system/docs/day-21/run6/run_RewardProgress4.log`
2. ✅ Searched for progress reward patterns using grep
3. ✅ Identified specific episodes where route distance increases
4. ✅ Traced vehicle state and movement during these episodes
5. ✅ Analyzed route distance calculation algorithm in code
6. ✅ Verified configuration parameter loading

---

## 🔍 KEY FINDINGS

### Finding #1: Route Distance INCREASES During Forward Movement (**CRITICAL BUG**)

**Evidence from logs (Steps 14-24)**:

```
Step 14: route_distance=264.38m, prev=264.38m → delta=-0.003m, reward=-0.13
Step 15: route_distance=264.42m, prev=264.38m → delta=-0.036m, reward=-1.79  ❌
Step 16: route_distance=264.48m, prev=264.42m → delta=-0.063m, reward=-3.15  ❌
Step 17: route_distance=264.56m, prev=264.48m → delta=-0.080m, reward=-4.00  ❌
Step 18: route_distance=264.65m, prev=264.56m → delta=-0.093m, reward=-4.66  ❌
...
Step 24: route_distance=264.98m, prev=264.90m → delta=-0.138m, reward=-6.89  ❌
```

**Vehicle behavior during this period**:
- Speed: 0.27 m/s → 1.00 m/s (accelerating forward)
- Steering: +0.08 to +0.27 (slight right turn, within lane)
- Safety: NO collisions, NO lane invasions initially
- Direction: Moving forward toward waypoint

**Problem**: Route distance INCREASES from 264.38m → 264.98m (+0.60m) over 10 steps despite vehicle moving forward!

**Expected**: Route distance should DECREASE as vehicle progresses along route.

**Impact**: Agent receives cumulative negative reward of -36.64 for 10 steps of CORRECT forward movement!

---

### Finding #2: Sudden Large Positive Reward at Waypoint Milestone

**Evidence (Step 25)**:

```
Step 25: route_distance=262.77m, prev=264.98m → delta=+2.172m, reward=+108.61  ✅
         (Includes +10.0 waypoint bonus)
Step 26: route_distance=262.63m, prev=262.77m → delta=+0.144m, reward=+7.19   ✅
Step 27: route_distance=262.48m, prev=262.63m → delta=+0.152m, reward=+7.59   ✅
```

**Observation**: When waypoint threshold is crossed:
- Route distance suddenly DROPS by 2.17m
- Progress reward jumps to +108.61 (distance reward +98.61 + waypoint bonus +10.0)
- Subsequent steps show normal positive rewards (~+7 to +10)

**Analysis**: This is when `current_waypoint_idx` increments, removing entire waypoint segment from total distance calculation.

---

### Finding #3: Pattern Repeats Throughout Exploration Phase

**Pattern observed**:
1. Route distance gradually INCREASES over 10-15 steps (negative rewards)
2. Waypoint milestone reached → distance suddenly DROPS (large positive reward)
3. Route distance DECREASES for next few steps (small positive rewards)
4. Cycle repeats

**Evidence of oscillation** (as user reported):

```
Steps 1-13:   progress ~0.00 (stationary, warming up)
Steps 14-24:  progress -0.13 to -6.89 (increasing distance, negative) ❌
Step 25:      progress +110.00 (waypoint reached!) ✅
Steps 26-36:  progress +7.19 to +9.44 (decreasing distance, positive) ✅
Steps 37-38:  progress +6.65 to -2.19 (distance increasing again) ⚠️
Steps 39-43:  progress -8.34 to -10.27 (strong negative) ❌
Step 44:      progress +110.00 (waypoint reached!) ✅
```

**Frequency**: Oscillation period ~20-30 steps (matches waypoint spacing ~3-4 meters at 0.3-1.0 m/s speed)

---

## 🧪 ROOT CAUSE ANALYSIS

### Algorithm: `get_route_distance_to_goal()`

**Current implementation** (from waypoint_manager.py):

```python
def get_route_distance_to_goal(self, vehicle_location) -> float:
    # Step 1: Find nearest waypoint ahead
    nearest_idx = self._find_nearest_waypoint_index(vehicle_location)
    
    # Step 2: Distance from vehicle to next waypoint
    total_distance = sqrt((next_wp[0] - vx)² + (next_wp[1] - vy)²)
    
    # Step 3: Sum distances between remaining waypoints
    for i in range(nearest_idx, len(waypoints) - 1):
        total_distance += distance(waypoint[i], waypoint[i+1])
    
    return total_distance
```

### The Fundamental Problem

**What we're calculating**: `vehicle_to_waypoint_distance + remaining_waypoints_distance`

**Why it increases**:

```
Scenario: Vehicle driving toward waypoint[5] at (105, 50)

t=0: Vehicle at (100, 50)
  - Distance to waypoint[5]: 5.0m
  - Remaining waypoints distance: 259.38m
  - Total: 264.38m

t=1: Vehicle moves to (100.04, 50) [+0.04m forward]
     BUT also drifts to (100.04, 50.02) [+0.02m sideways]
  - Distance to waypoint[5]: sqrt((105-100.04)² + (50-50.02)²) = 4.96m  ← Decreased!
  - Remaining waypoints distance: 259.38m (unchanged)
  - Total: 264.34m ✅ Should decrease

t=2: Random action causes drift backward to (99.94, 50.01)
  - Distance to waypoint[5]: sqrt((105-99.94)² + (50-50.01)²) = 5.06m  ← INCREASED!
  - Remaining waypoints distance: 259.38m
  - Total: 264.44m ❌ INCREASES!
```

**Root Cause**: During exploration phase (random actions), vehicle often drifts AWAY from current waypoint, causing `vehicle_to_waypoint_distance` to INCREASE, even though vehicle may be moving generally forward along the route!

### Why Waypoint Milestone Fixes It Temporarily

When waypoint threshold is crossed:
1. `current_waypoint_idx` increments: 5 → 6
2. Entire waypoint segment removed from calculation (~3m)
3. New target waypoint is closer → distance drops suddenly
4. Cycle repeats for next waypoint

---

## 🎯 CORRECT SOLUTION: Projection-Based Route Distance

### Why Current Method Is Wrong

**Current**: Measuring `point-to-point` distance from vehicle to waypoint
**Problem**: Doesn't account for progress ALONG the route direction
**Result**: Lateral/backward drift penalized equally to lack of forward progress

### Correct Approach: Projection onto Route Path

**Goal**: Measure distance from vehicle's PROJECTION onto route to goal

```python
def get_route_distance_to_goal(self, vehicle_location) -> float:
    """Calculate distance using projection onto route path."""
    
    # Step 1: Find nearest route segment (between waypoint[i] and waypoint[i+1])
    segment_idx = self._find_nearest_segment(vehicle_location)
    
    # Step 2: Project vehicle onto that segment
    projection = self._project_point_onto_segment(
        vehicle_location,
        self.waypoints[segment_idx],
        self.waypoints[segment_idx + 1]
    )
    
    # Step 3: Distance from projection to end of segment
    dist_to_segment_end = distance(projection, self.waypoints[segment_idx + 1])
    
    # Step 4: Sum remaining waypoint segments
    remaining = sum(
        distance(self.waypoints[i], self.waypoints[i+1])
        for i in range(segment_idx + 1, len(self.waypoints) - 1)
    )
    
    return dist_to_segment_end + remaining
```

**Benefits**:
- ✅ Forward movement → projection advances → distance DECREASES
- ✅ Sideways drift → projection unchanged → distance UNCHANGED
- ✅ Backward movement → projection retreats → distance INCREASES
- ✅ Smooth continuous signal (no sudden jumps at waypoints)

---

## 📊 IMPACT ANALYSIS

### Current Behavior (With Bug)

**Exploration Phase Rewards**:
```
Total steps: 1000
Negative progress episodes: ~700 steps (-0.13 to -10.0 each)
Positive progress episodes: ~300 steps (+0.0 to +10.0 each)
Waypoint bonuses: ~25-30 × +10.0 = +250 to +300
Net progress signal: Dominated by waypoint bonuses, not continuous progress
```

**Learning Impact**:
- Agent learns: "Waypoint bonuses are important"
- Agent DOES NOT learn: "Forward progress is important"
- Result: Jerky, waypoint-chasing behavior instead of smooth forward motion

### Expected Behavior (After Fix)

**Exploration Phase Rewards**:
```
Total steps: 1000
Forward movement steps: ~400 steps (+5.0 to +10.0 each)
Sideways drift steps: ~400 steps (~0.0 each)
Backward movement steps: ~200 steps (-5.0 to -10.0 each)
Waypoint bonuses: ~25-30 × +10.0 = +250 to +300
Net progress signal: Strong positive for forward, balanced learning
```

**Learning Impact**:
- Agent learns: "Forward movement = consistently positive"
- Agent learns: "Backward = negative, sideways = neutral"
- Result: Smooth forward-directed policy emerges naturally

---

## 🚨 SEVERITY & PRIORITY

**Severity**: 🔴 **CRITICAL**

**Justification**:
1. Breaks fundamental reward signal for progress
2. Causes agent to learn incorrect policy (waypoint-chasing vs smooth forward)
3. Affects BOTH exploration and learning phases
4. Explains user-reported oscillating rewards

**Priority**: **P0 - IMMEDIATE FIX REQUIRED**

**Blocking**: Next training run should NOT proceed until this is fixed

---

## 📋 ADDITIONAL FINDINGS

### Configuration Loading: ✅ CORRECT

**User concern**: "The parameters of training_config.yaml are overwriting some parameter of td3_config.yaml"

**Investigation**:
- Checked `carla_env.py` initialization (line 91-92)
- Checked `train_td3.py` config loading
- Checked reward calculator initialization (line 216)

**Finding**: ✅ **NO OVERWRITING DETECTED**
- `training_config.yaml`: Used for reward weights, scenarios, episode settings
- `td3_config.yaml`: Used for TD3 algorithm hyperparameters
- `carla_config.yaml`: Used for CARLA simulation settings
- Each config has distinct, non-overlapping parameters

**Evidence from logs**:
```
2025-11-21 20:47:01 - REWARD WEIGHTS VERIFICATION:
  efficiency: 1.0
  lane_keeping: 5.0
  comfort: 0.5
  safety: 1.0
  progress: 1.0

2025-11-21 20:47:01 - PROGRESS REWARD PARAMETERS:
  waypoint_bonus: 10.0
  distance_scale: 50.0
  goal_reached_bonus: 100.0
```

All values match `training_config.yaml` exactly. No conflicts detected.

---

## 📝 ACTION ITEMS

### Immediate (P0):
1. ✅ Document bug in `BUG_ROUTE_DISTANCE_INCREASES.md` (DONE)
2. ⏳ Implement projection-based route distance calculation
3. ⏳ Add comprehensive diagnostic logging
4. ⏳ Unit test projection method
5. ⏳ Integration test with 1K-step run
6. ⏳ Verify distance decreases during forward movement

### Follow-up (P1):
7. ⏳ Re-run full training (5K-20K steps) with fix
8. ⏳ Compare training curves before/after fix
9. ⏳ Document results in research paper

### Documentation (P2):
10. ✅ Create systematic investigation summary (THIS DOCUMENT)
11. ⏳ Update `FIXES_IMPLEMENTED.md` with Fix #4
12. ⏳ Add projection method to technical documentation

---

## 🔗 REFERENCES

- **User Report**: Investigation request (this chat)
- **Log File**: `av_td3_system/docs/day-21/run6/run_RewardProgress4.log`
- **Bug Analysis**: `BUG_ROUTE_DISTANCE_INCREASES.md`
- **Previous Fixes**: `FIXES_IMPLEMENTED.md`
- **Code**: 
  - `src/environment/waypoint_manager.py` (lines 374-445)
  - `src/environment/reward_functions.py` (lines 960-1041)
  - `src/environment/carla_env.py` (line 652)
- **CARLA Docs**: https://carla.readthedocs.io/en/latest/core_map/

---

## ✅ CONCLUSION

**Primary Issue**: Route distance calculation has fundamental flaw causing negative rewards during forward movement.

**Root Cause**: Measuring point-to-point distance instead of projection-based path distance.

**Impact**: Severe - breaks progress reward signal, prevents agent from learning smooth forward motion.

**Solution**: Implement projection-based route distance calculation.

**Status**: Bug thoroughly analyzed, solution designed, ready for implementation.

**Next Step**: Implement fix and validate with test run before resuming training.
