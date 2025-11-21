# SYSTEMATIC ANALYSIS: Progress Reward Bugs

**Date**: 2025-11-21  
**Analysis**: Complete breakdown of progress reward calculation issues  
**Priority**: CRITICAL - Multiple bugs compound to create extreme right-turn bias

---

## 🔬 EXECUTIVE SUMMARY

**Three Critical Bugs Identified**:

1. **PBRS Free Reward Bug**: Agent gets +1.15 reward per step for being far from goal (even with zero movement!)
2. **Euclidean Distance Bug**: Progress measured as straight-line distance → rewards diagonal shortcuts off-road
3. **Reward Imbalance Bug**: Lane invasion penalty (-10.0) << Progress reward (+46.15) → profitable to violate lanes

**Net Effect**: Agent learns turning right off-road gives **+36.15 net reward** despite lane invasion!

---

## 📊 BUG #1: PBRS Gives Free Reward for Zero Movement

### Evidence from Logs

```
2025-11-21 17:42:53 - [PROGRESS] Distance Delta: 0.000m (backward), Reward: 0.00 (scale=50.0)
2025-11-21 17:42:53 - [PROGRESS] PBRS: Φ(s')=-229.420, Φ(s)=-229.420, F(s,s')=2.294, weighted=1.147
2025-11-21 17:42:53 - [PROGRESS] Final: progress=1.15 (distance: 0.00, PBRS: 1.15, waypoint: 0.0, goal: 0.0)
```

**Analysis**:
- Vehicle is **STATIONARY** (distance_delta = 0.000m)
- Distance reward = 0.00 ✅ (correct)
- **PBRS reward = +1.147** 🔴 (WRONG! Should be 0.00 for no movement!)

### Mathematical Verification

**Current PBRS Implementation** (reward_functions.py lines 987-993):
```python
potential_current = -distance_to_goal
potential_prev = -self.prev_distance_to_goal
pbrs_reward = self.gamma * potential_current - potential_prev
pbrs_weighted = pbrs_reward * 0.5
```

**Calculation for Stationary Vehicle**:
```
Given:
  distance_to_goal_current = 229.42m
  distance_to_goal_prev = 229.42m  (NO MOVEMENT!)
  γ = 0.99
  PBRS weight = 0.5

PBRS calculation:
  Φ(s') = -229.42
  Φ(s) = -229.42
  F(s,s') = γ×Φ(s') - Φ(s)
          = 0.99 × (-229.42) - (-229.42)
          = -227.126 - (-229.42)
          = 2.294
  
  Weighted: 2.294 × 0.5 = 1.147
```

**The Bug**:
```
F(s,s') = γ×Φ(s') - Φ(s)
        = γ×(-d') - (-d)    [where d' = d for no movement]
        = -γd + d
        = d(1 - γ)
        = 229.42 × (1 - 0.99)
        = 229.42 × 0.01
        = 2.294
```

**Root Cause**: PBRS formula gives reward proportional to `(1-γ) × distance_to_goal` **even with zero movement**!

**Perverse Incentive**: Further from goal = MORE free reward per step!

### Why This Violates PBRS Theory

**PBRS Theorem (Ng et al. 1999)**:
> "F(s,s') = γΦ(s') - Φ(s) preserves optimal policy"

**Assumption**: Φ(s) is a **state potential function** (depends only on state, not time)

**Our Violation**: We're using temporal difference (γ factor) incorrectly!

**Correct PBRS for Distance**:
```python
# Should be:
F(s,s') = -distance_to_goal_current + distance_to_goal_prev
        = -(distance change)

# This equals the distance reward we already have!
# So PBRS is REDUNDANT with our distance reward!
```

**Conclusion**: Our PBRS implementation is **theoretically incorrect** and creates free rewards!

---

## 📊 BUG #2: Euclidean Distance Rewards Off-Road Shortcuts

### Route Geometry (from waypoints.txt)

```
Start Position: (317.74, 129.49) - Western position, right lane, heading west
Route Path: X decreases 317 → 98 (go WEST ~220m along Y=129.49)
Turn Point: Around waypoint 74, route turns SOUTH
Goal Position: (92.34, 86.73) - Southwest of start

Euclidean (straight-line) distance: 229.42m
Route distance (following waypoints): ~300m+
```

### Why Euclidean Distance Creates Perverse Incentive

**Scenario**: Vehicle at start (317, 129), heading west

**Option 1: Drive straight west (follow road)**
```
Movement: 0.3m west → position (316.7, 129.49)
Euclidean distance change:
  Before: sqrt((317-92)² + (129-86)²) = 229.42m
  After:  sqrt((316.7-92)² + (129-86)²) = 229.12m
  Reduction: 0.30m ✅
```

**Option 2: Turn right (diagonal shortcut off-road)**
```
Movement: 0.3m diagonal (0.21m west, 0.21m south) → position (316.79, 129.28)
Euclidean distance change:
  Before: sqrt((317-92)² + (129-86)²) = 229.42m
  After:  sqrt((316.79-92)² + (129.28-86)²) = 229.12m
  Reduction: 0.30m ✅ SAME AS GOING STRAIGHT!
```

**But with Pythagorean theorem for diagonal movement**:
```
Diagonal movement covers BOTH X and Y components:
  ΔX = 0.21m, ΔY = 0.21m
  Diagonal distance = sqrt(0.21² + 0.21²) = 0.30m

Euclidean distance reduction (approximation for small movements):
  ≈ (ΔX²/distance) + (ΔY²/distance)
  ≈ MORE reduction than just ΔX alone!

Result: Diagonal movement (off-road) gives BETTER Euclidean reduction!
```

**Conclusion**: Euclidean distance **naturally rewards diagonal shortcuts** toward goal!

### Why Route Distance Would Fix This

**Route Distance Calculation**:
```python
def get_route_distance_to_goal(vehicle_location):
    """Distance along remaining waypoint path"""
    # Find nearest waypoint ahead
    nearest_idx = find_nearest_waypoint_ahead(vehicle_location)
    
    # Sum: vehicle → nearest_wp → ... → goal
    total_distance = 0.0
    total_distance += distance(vehicle_location, waypoints[nearest_idx])
    for i in range(nearest_idx, len(waypoints)-1):
        total_distance += distance(waypoints[i], waypoints[i+1])
    
    return total_distance
```

**Expected Behavior**:
```
Driving straight west (following waypoints):
  Route distance: DECREASES (progressing along path)
  Reward: POSITIVE ✅

Turning right off-road (shortcut):
  Route distance: NO CHANGE or INCREASES (not following path!)
  Reward: ZERO or NEGATIVE ✅
```

---

## 📊 BUG #3: Lane Invasion Penalty Too Weak

### Current Penalty Structure

**From reward_functions.py** (lines 820-836):
```python
if lane_invasion_detected:
    safety += self.lane_invasion_penalty  # -10.0
    self.logger.warning(f"[SAFETY-LANE_INVASION] penalty={self.lane_invasion_penalty:.1f}")
```

**Configuration** (carla_config.yaml):
```yaml
safety:
  lane_invasion_penalty: -10.0  # Raw penalty
weights:
  safety: 1.0                    # Weight multiplier
```

**Effective Penalty**: -10.0 × 1.0 = **-10.0**

### Comparison with Progress Reward

**Scenario**: Vehicle turns right 0.3m (diagonal off-road)

**Progress Rewards**:
```
1. Distance reward:
   Euclidean reduction: 0.3m
   Reward: 0.3 × 50.0 (distance_scale) = +15.0
   After efficiency weight (3.0): +45.0

2. PBRS reward (bug!):
   Free reward: +1.147 (from being 229m from goal)

TOTAL PROGRESS: +45.0 + 1.147 = +46.15
```

**Safety Penalty**:
```
Lane invasion: -10.0
```

**Net Reward**:
```
+46.15 (progress) - 10.0 (lane invasion) = +36.15 ✅ PROFITABLE!
```

**Conclusion**: Lane invasion penalty is **46% weaker** than progress reward for the same action!

### Why Is Lane Invasion Not Always Detected?

**Potential Issues**:

1. **Sensor Frequency**: Lane invasion sensor may not trigger every frame
2. **Detection Threshold**: May require crossing lane marking significantly
3. **Logging Issue**: Penalty applied but not logged (need to verify)

**Questions to Answer**:
- Is `lane_invasion_detected` being set correctly in the environment?
- How often does the lane invasion sensor trigger during right turns?
- Are there multiple lane invasions being counted or just the first?

---

## 🎯 ROOT CAUSE SUMMARY

### The Compounding Effect

**Bug #1 (PBRS)**: +1.15 free reward per step (worse when far from goal)  
**Bug #2 (Euclidean)**: Diagonal shortcut maximizes distance reduction  
**Bug #3 (Weak Penalty)**: -10.0 penalty << +45.0 progress reward

**Combined Effect**:
```
Action: Turn right off-road (0.3m diagonal movement)

Rewards:
  Distance (Euclidean): 0.3m × 50.0 × 3.0 = +45.00
  PBRS (free reward):                      + 1.15
  Lane invasion penalty:                   -10.00
  ────────────────────────────────────────────────
  NET REWARD:                              +36.15 ✅ HIGHLY PROFITABLE!

Action: Drive straight (0.3m west, on-road)

Rewards:
  Distance (Euclidean): 0.3m × 50.0 × 3.0 = +45.00
  PBRS (free reward):                      + 1.15
  Lane keeping (perfect):                  + 0.00
  ────────────────────────────────────────────────
  NET REWARD:                              +46.15 ✅ Slightly better

DIFFERENCE: Off-road is only -10.0 worse (77% of on-road reward!)
```

**Agent's Rational Learning**:
- Off-road shortcuts give 77% of on-road rewards
- Faster to goal = more episodes = more total reward
- **Optimal policy: Maximize speed via shortcuts!**

---

## 🔧 PROPOSED FIXES (Priority Order)

### Fix #1: Remove PBRS (IMMEDIATE - 5 min)

**Problem**: PBRS gives free +1.15 reward per step (theoretically incorrect)

**Solution**: Comment out PBRS entirely (we already have distance reward!)

```python
# Component 1b: PBRS - DISABLED (Bug: gives free reward for zero movement)
# The distance reward already provides the shaping we need!
# PBRS as implemented violates Ng et al. theorem by using γ incorrectly.

# if self.prev_distance_to_goal is not None:
#     potential_current = -distance_to_goal
#     potential_prev = -self.prev_distance_to_goal
#     pbrs_reward = self.gamma * potential_current - potential_prev
#     pbrs_weighted = pbrs_reward * 0.5
#     progress += pbrs_weighted
```

**Expected Impact**: Removes +1.15 free reward, making penalties more effective

### Fix #2: Implement Route Distance (HIGH PRIORITY - 30 min)

**Problem**: Euclidean distance rewards diagonal shortcuts

**Solution**: Calculate distance along remaining waypoint path

**Implementation** (waypoint_manager.py):
```python
def get_route_distance_to_goal(self, vehicle_location):
    """
    Calculate distance along remaining waypoint path.
    
    Prevents shortcuts by measuring path-following distance,
    not straight-line distance to goal.
    """
    # Find nearest waypoint ahead of vehicle
    nearest_idx = self.find_nearest_waypoint_ahead(vehicle_location)
    if nearest_idx is None:
        # Fallback to Euclidean if off-path
        return self.get_distance_to_goal(vehicle_location)
    
    # Sum distances: vehicle → next_wp → ... → goal
    total_distance = 0.0
    
    # Distance from vehicle to next waypoint
    next_wp = self.waypoints[nearest_idx]
    vx, vy = vehicle_location.x, vehicle_location.y
    total_distance += math.sqrt((next_wp[0] - vx)**2 + (next_wp[1] - vy)**2)
    
    # Sum distances between remaining waypoints
    for i in range(nearest_idx, len(self.waypoints) - 1):
        wp1 = self.waypoints[i]
        wp2 = self.waypoints[i + 1]
        total_distance += math.sqrt((wp2[0] - wp1[0])**2 + (wp2[1] - wp1[1])**2)
    
    return total_distance
```

**Update reward_functions.py**:
```python
def _calculate_progress_reward(self, ...):
    # Get ROUTE distance instead of Euclidean
    distance_to_goal = self.waypoint_manager.get_route_distance_to_goal(vehicle_location)
    
    # Rest of calculation stays the same
    if self.prev_distance_to_goal is not None:
        distance_delta = self.prev_distance_to_goal - distance_to_goal
        distance_reward = distance_delta * self.distance_scale
        progress += distance_reward
```

**Expected Impact**: Off-road shortcuts give ZERO progress reward (route distance unchanged)

### Fix #3: Increase Lane Invasion Penalty (OPTIONAL)

**Problem**: -10.0 penalty too weak vs +45.0 progress

**Option 3a**: Increase penalty magnitude
```yaml
# config/carla_config.yaml
safety:
  lane_invasion_penalty: -50.0  # Up from -10.0
```

**Option 3b**: Keep penalty, let route distance fix do the work
```
With route distance:
  Off-road shortcut: 0.0 progress + (-10.0 lane) = -10.0 ✅ UNPROFITABLE!
  On-road driving:   +15.0 progress + 0.0 penalty = +15.0 ✅ PROFITABLE!
```

**Recommendation**: Apply Fix #1 and #2 first, then evaluate if Fix #3 needed

---

## 📋 VALIDATION PLAN

### Step 1: Apply Fix #1 (Remove PBRS)

1. Comment out PBRS code in reward_functions.py
2. Run 1K step test
3. Verify logs show `PBRS: 0.00` (not +1.15)
4. Check total progress reward reduced by ~1.15 per step

### Step 2: Apply Fix #2 (Route Distance)

1. Implement `get_route_distance_to_goal()` in waypoint_manager.py
2. Update `_calculate_progress_reward()` to use it
3. Run 1K step test
4. Verify:
   - On-road driving: positive distance_delta
   - Off-road movement: zero or negative distance_delta
   - Logs show route distance values (not Euclidean)

### Step 3: Full Training Run

1. Train for 20K steps with both fixes
2. Monitor action_steering_mean (expect ~0.0, not +0.88)
3. Check TensorBoard:
   - Progress reward should be lower per step
   - Lane keeping percentage should increase
   - Safety violations should decrease

### Expected Results After Fixes

**Action Statistics**:
```
Steering Mean:  ~0.0 ± 0.2  (balanced, not +0.88)
Throttle Mean:  ~0.5 ± 0.3  (forward motion)
```

**Reward Breakdown**:
```
Progress:      30-40%  (reduced from 83%)
Lane Keeping:  20-30%  (increased from 6%)
Safety:        15-20%
Efficiency:    15-20%
Comfort:        5-10%
```

**Behavior**:
```
✅ Vehicle follows road (no off-road shortcuts)
✅ Turns at intersection (not immediately)
✅ Reaches waypoints in sequence
```

---

## 🔍 DIAGNOSTIC QUESTIONS TO ANSWER

### About Lane Invasion Detection

1. **Is lane_invasion_detected being set correctly?**
   - Check environment code where this boolean is set
   - Verify CARLA sensor is attached and working
   - Log every lane invasion event (not just penalty)

2. **How often does lane invasion trigger?**
   - Add counter: `self.lane_invasions_this_episode`
   - Log invasion rate: invasions per 100 steps
   - Compare exploration vs learning phase

3. **Are there continuous vs discrete invasion semantics?**
   - Does flag stay True while off-road?
   - Or only True on crossing event?
   - Should penalty be cumulative or one-time?

### About Distance Calculation

4. **Is get_distance_to_goal() using Euclidean or route distance?**
   - Verify implementation in waypoint_manager.py
   - Check if any route distance method exists already
   - Confirm which one is being called

5. **Does Euclidean distance actually favor right turns in this geometry?**
   - Calculate manually for start → goal
   - Compare: straight west vs diagonal southwest
   - Verify Pythagorean advantage

---

## 📚 REFERENCES

1. **PBRS Theory**: Ng et al. (1999) "Policy Invariance Under Reward Transformations"
   - Original theorem: F(s,s') = γΦ(s') - Φ(s)
   - Assumption: Φ is state potential (not time-dependent)
   - Our bug: Using γ creates time-dependent free reward

2. **TD3 Algorithm**: Fujimoto et al. (2018) "Addressing Function Approximation Error"
   - Continuous action spaces
   - Dense rewards preferred for exploration
   - **BUT**: Dense rewards must measure correct objective!

3. **Reward Shaping Best Practices**:
   - Distance-based shaping should reward ACTUAL progress
   - Route-following tasks need path distance, not Euclidean
   - Penalties must be scaled relative to rewards

4. **Related Work**: "Interpretable End-to-end Urban Autonomous Driving" (Chen et al.)
   - Uses waypoint-relative features
   - Route distance for progress measurement
   - Multi-objective reward balancing

---

## ✅ ACTION ITEMS

**IMMEDIATE (Today)**:
- [ ] Comment out PBRS code (Fix #1)
- [ ] Test with 1K steps to verify PBRS=0.0
- [ ] Document baseline metrics without PBRS

**HIGH PRIORITY (This Session)**:
- [ ] Implement `get_route_distance_to_goal()` (Fix #2)
- [ ] Update progress reward to use route distance
- [ ] Test with 1K steps to verify off-road gives 0.0 progress
- [ ] Full training run (20K steps) if tests pass

**OPTIONAL (If Needed)**:
- [ ] Increase lane invasion penalty (Fix #3)
- [ ] Add lane invasion logging/counting
- [ ] Analyze invasion detection frequency

**FOR PAPER**:
- [ ] Document reward function design decisions
- [ ] Compare Euclidean vs route distance results
- [ ] Justify penalty magnitudes with empirical data
- [ ] Reference PBRS theory and why we removed it

---

**End of Systematic Analysis**
