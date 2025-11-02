# Diagnostic Logging Additions

**Date:** 2025-01-20  
**Purpose:** Investigate training failure root causes (mean reward -50k, 0% success, 27-step episodes)  
**Status:** IMPLEMENTED  

---

## Summary

Added comprehensive diagnostic logging to `reward_functions.py` to investigate two critical hypotheses from the progress reward analysis:

1. **State Representation Issues** - Are distance calculations correct? Is visual CNN working?
2. **Reward Component Balance** - Which component is causing -50k mean reward?

---

## Logging Additions

### 1. Progress Reward Function (`_calculate_progress_reward`)

**Location:** Lines 602-715 in `reward_functions.py`

**Purpose:** Verify distance calculations and PBRS implementation are working correctly.

**Log Levels:**
- `DEBUG`: Every step tracking (distance delta, PBRS, components)
- `INFO`: Milestone events (waypoint reached, goal reached)
- `WARNING`: Clipping events (reward exceeds bounds)

**Logged Information:**

#### Input State Logging
```python
[PROGRESS] Input: distance_to_goal=125.50m, waypoint_reached=False, goal_reached=False, prev_distance=126.30m
```
**Validates:**
- ✓ Distance values are reasonable (not NaN, not negative)
- ✓ Distance is changing over time (not frozen)
- ✓ Previous distance is being tracked correctly

#### Distance Delta Logging
```python
[PROGRESS] Distance Delta: 0.800m (forward), Reward: +40.00 (scale=50.0)
```
**Validates:**
- ✓ Agent is moving toward goal (delta > 0) or away (delta < 0)
- ✓ Distance scale (50.0) is being applied correctly
- ✓ Reward magnitude is appropriate (40.0 = 0.8m × 50.0)

#### PBRS Component Logging
```python
[PROGRESS] PBRS: Φ(s')=-125.50, Φ(s)=-126.30, F(s,s')=0.792, weighted=0.396 (γ=0.99, weight=0.5)
```
**Validates:**
- ✓ Potential function is computed correctly (Φ = -distance)
- ✓ PBRS shaping term follows Ng et al. formula: γΦ(s') - Φ(s)
- ✓ Weighting (0.5x) is applied as configured
- ✓ Gamma (0.99) matches TD3 discount factor

#### Milestone Logging
```python
[PROGRESS] 🎯 Waypoint reached! Bonus: +10.0, total_progress=50.80
[PROGRESS] 🏁 Goal reached! Bonus: +100.0, total_progress=140.40
```
**Validates:**
- ✓ Waypoint system is detecting progress
- ✓ Bonuses are being added correctly
- ✓ Agent can reach milestones (training not stuck)

#### Clipping Warning Logging
```python
[PROGRESS] ⚠️ CLIPPED: raw=140.40 → clipped=110.0
```
**Validates:**
- ✓ Clipping is working as intended
- ✓ Identifies cases where rewards exceed bounds (need adjustment?)

#### Final Summary Logging
```python
[PROGRESS] Final: progress=50.80 (distance: 40.00, PBRS: 0.40, waypoint: 0.0, goal: 0.0)
```
**Validates:**
- ✓ All components are summing correctly
- ✓ Individual contributions are visible for debugging

---

### 2. Total Reward Calculation (`calculate` method)

**Location:** Lines 215-265 in `reward_functions.py`

**Purpose:** Identify which reward component is dominating and causing training failure.

**Log Levels:**
- `DEBUG`: Every step component breakdown
- `WARNING`: Component imbalance detection (>80% magnitude)

**Logged Information:**

#### Component Breakdown Logging
```python
[REWARD] Components - Efficiency: 0.850×1.0=0.85, Lane: 0.300×2.0=0.60, Comfort: 0.100×0.5=0.05, Safety: -5.000×1.0=-5.00, Progress: 50.800×5.0=254.00
[REWARD] TOTAL: 250.50
```
**Validates:**
- ✓ All 5 components are computed
- ✓ Weights are applied correctly
- ✓ Total sum is accurate
- ✓ Relative magnitudes are visible

**Helps Diagnose:**
- ❓ Is safety penalty (-5.0) overwhelming progress (+254.0)?
- ❓ Is efficiency reward too weak (+0.85)?
- ❓ Are any components always zero (broken)?

#### Component Dominance Warning
```python
[REWARD] ⚠️ Component 'progress' is dominating: 92.5% of total magnitude (threshold: 80%)
```
**Validates:**
- ✓ Detects imbalanced reward structures automatically
- ✓ Threshold: 80% of total absolute magnitude
- ✓ Helps identify configuration issues

**Example Scenarios:**
- Safety dominating (-50k) → Check collision frequency, penalty magnitude
- Efficiency dominating → Check target speed, overspeed penalties
- Progress dominating → Check distance scale, waypoint frequency

---

## How to Use This Logging

### Enable Debug Logging

**Option 1: Environment Variable**
```bash
export LOG_LEVEL=DEBUG
python train.py
```

**Option 2: Logging Configuration**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Option 3: File-Specific**
```python
logger = logging.getLogger('reward_functions')
logger.setLevel(logging.DEBUG)
```

### Analyze Logs for Training Failure

**Step 1: Check Distance Calculations**
```bash
grep "\[PROGRESS\] Input:" training.log | head -100
```
**Look for:**
- ✓ Is `distance_to_goal` decreasing over time?
- ✗ Is it frozen (same value every step)?
- ✗ Is it NaN or negative?
- ✗ Is it jumping erratically (>10m per step)?

**Step 2: Check Distance Deltas**
```bash
grep "\[PROGRESS\] Distance Delta:" training.log | head -100
```
**Look for:**
- ✓ Are deltas mostly positive (forward progress)?
- ✗ Are they always zero (no movement)?
- ✗ Are they always negative (moving away)?
- ✗ Are they very small (<0.01m)?

**Step 3: Check Reward Component Balance**
```bash
grep "\[REWARD\] Components" training.log | head -100
```
**Look for:**
- ❓ Which component has largest magnitude?
- ❓ Is safety always large and negative (-5.0)?
- ❓ Is progress always near zero?
- ❓ Are ratios reasonable (progress > safety)?

**Step 4: Check for Dominance Warnings**
```bash
grep "⚠️ Component" training.log
```
**If warnings appear:**
- Safety dominating → Adjust penalty magnitudes or termination conditions
- Efficiency dominating → Check target speed vs actual velocity
- Progress dominating → Check distance scale (may be too high)

---

## Expected Log Output (Healthy Training)

### First Few Steps of Episode
```
[PROGRESS] First step: initializing prev_distance_to_goal=150.00m
[REWARD] Components - Efficiency: 0.000×1.0=0.00, Lane: 0.000×2.0=0.00, Comfort: 0.000×0.5=0.00, Safety: 0.000×1.0=0.00, Progress: 0.000×5.0=0.00
[REWARD] TOTAL: 0.00

[PROGRESS] Input: distance_to_goal=149.80m, waypoint_reached=False, goal_reached=False, prev_distance=150.00m
[PROGRESS] Distance Delta: 0.200m (forward), Reward: +10.00 (scale=50.0)
[PROGRESS] PBRS: Φ(s')=-149.80, Φ(s)=-150.00, F(s,s')=0.198, weighted=0.099 (γ=0.99, weight=0.5)
[PROGRESS] Final: progress=10.10 (distance: 10.00, PBRS: 0.10, waypoint: 0.0, goal: 0.0)
[REWARD] Components - Efficiency: 0.100×1.0=0.10, Lane: 0.200×2.0=0.40, Comfort: 0.050×0.5=0.03, Safety: 0.000×1.0=0.00, Progress: 10.100×5.0=50.50
[REWARD] TOTAL: 51.03
```

**Interpretation:**
- ✓ Distance decreasing (150.00 → 149.80m)
- ✓ Forward progress (+0.20m)
- ✓ PBRS computed correctly (+0.099)
- ✓ Progress reward dominant (+50.50 weighted)
- ✓ Total reward positive (+51.03)

---

## Expected Log Output (Training Failure)

### Scenario 1: Distance Frozen (State Representation Bug)
```
[PROGRESS] Input: distance_to_goal=150.00m, waypoint_reached=False, goal_reached=False, prev_distance=150.00m
[PROGRESS] Distance Delta: 0.000m (forward), Reward: +0.00 (scale=50.0)
[PROGRESS] PBRS: Φ(s')=-150.00, Φ(s)=-150.00, F(s,s')=0.000, weighted=0.000 (γ=0.99, weight=0.5)
[PROGRESS] Final: progress=0.00 (distance: 0.00, PBRS: 0.00, waypoint: 0.0, goal: 0.0)
[REWARD] Components - Efficiency: 0.000×1.0=0.00, Lane: 0.000×2.0=0.00, Comfort: 0.000×0.5=0.00, Safety: -5.000×1.0=-5.00, Progress: 0.000×5.0=0.00
[REWARD] TOTAL: -5.00
```

**Diagnosis:**
- ✗ Distance not changing (frozen at 150.00m)
- ✗ Zero progress reward
- ✗ Safety penalty dominating
- **Root Cause:** Distance calculation bug in `carla_env.py`

---

### Scenario 2: Safety Penalties Dominating
```
[PROGRESS] Input: distance_to_goal=149.50m, waypoint_reached=False, goal_reached=False, prev_distance=150.00m
[PROGRESS] Distance Delta: 0.500m (forward), Reward: +25.00 (scale=50.0)
[PROGRESS] Final: progress=25.12 (distance: 25.00, PBRS: 0.12, waypoint: 0.0, goal: 0.0)
[REWARD] Components - Efficiency: 0.200×1.0=0.20, Lane: 0.100×2.0=0.20, Comfort: 0.050×0.5=0.03, Safety: -5.000×1.0=-5.00, Progress: 25.120×5.0=125.60
[REWARD] TOTAL: 121.03
[REWARD] ⚠️ Component 'progress' is dominating: 96.2% of total magnitude (threshold: 80%)
```

**Diagnosis:**
- ✓ Distance decreasing correctly
- ✓ Progress reward working
- ⚠️ Progress dominating (may need rebalancing)
- **Action:** Monitor over 100+ steps to see if balance improves

---

### Scenario 3: Early Termination (Collision/Offroad)
```
[PROGRESS] Input: distance_to_goal=148.20m, waypoint_reached=False, goal_reached=False, prev_distance=149.00m
[PROGRESS] Distance Delta: 0.800m (forward), Reward: +40.00 (scale=50.0)
[PROGRESS] Final: progress=40.16 (distance: 40.00, PBRS: 0.16, waypoint: 0.0, goal: 0.0)
[REWARD] Components - Efficiency: 0.500×1.0=0.50, Lane: 0.300×2.0=0.60, Comfort: 0.100×0.5=0.05, Safety: -5.000×1.0=-5.00, Progress: 40.160×5.0=200.80
[REWARD] TOTAL: 196.95
[REWARD] ⚠️ Component 'progress' is dominating: 95.8% of total magnitude (threshold: 80%)
```

**Next Step Log:**
```
Episode terminated: collision detected at step 27
```

**Diagnosis:**
- ✓ Progress reward working fine
- ✓ Balance looks good
- ✗ Episode ending too early (27 steps)
- **Root Cause:** Aggressive termination conditions or lack of collision avoidance

---

## Next Debugging Steps

Based on log analysis, choose appropriate action:

### If Distance Frozen → Investigate `carla_env.py`
1. Add logging to waypoint distance calculation
2. Verify CARLA API usage (`Location.distance()`)
3. Check coordinate system (world vs local frame)
4. Validate waypoint system is updating

### If Safety Dominating → Check Termination Conditions
1. Review `training_config.yaml` termination settings
2. Log collision frequency and causes
3. Consider relaxing `on_collision: true` for learning
4. Verify obstacle detector sensor is working

### If Progress Dominating → Monitor Training
1. Run 100+ steps to see if balance improves
2. Check if agent is actually moving forward
3. Consider reducing `distance_scale` from 50.0 to 30.0
4. Verify waypoint frequency (should be every ~10-20m)

---

## Performance Impact

**Log Volume:**
- DEBUG level: ~50 lines per episode step
- INFO level: ~2-5 lines per episode (milestones only)
- WARNING level: <1 line per episode (only if issues)

**Recommendations:**
- Development: Use DEBUG for first 100 steps, then INFO
- Training: Use INFO level to avoid slowdown
- Production: Use WARNING level only

**Filtering:**
```bash
# Only progress rewards
grep "\[PROGRESS\]" training.log > progress_debug.log

# Only reward totals
grep "\[REWARD\] TOTAL" training.log > reward_totals.log

# Only warnings
grep "⚠️" training.log > warnings.log
```

---

## Success Criteria

After implementing fixes, logs should show:

✓ **Distance Calculations:**
- Decreasing over time (forward progress)
- Reasonable magnitudes (0.01-2.0m per step)
- No frozen values or NaNs

✓ **Progress Rewards:**
- Mostly positive (agent moving toward goal)
- Magnitude ~10-50 per step (with scale=50.0)
- PBRS contributing small positive gradient (~0.1-1.0)

✓ **Component Balance:**
- No single component >80% of total magnitude
- Progress stronger than safety (ratio ~5:1 to 10:1)
- Efficiency and lane keeping contributing positively

✓ **Episode Length:**
- >50 steps (was 27)
- Reaching waypoints before termination
- Gradual progress toward goal (not stuck or oscillating)

---

**Status:** IMPLEMENTED  
**Next Step:** Run training with DEBUG logging enabled and analyze first 100 steps  
**Expected Outcome:** Identify root cause of -50k mean reward from logs
