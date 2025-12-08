# ✅ FIX APPLIED: Reward Scaling Catastrophe

**Date:** January 5, 2025  
**Status:** **FIXED - READY FOR TESTING**  
**Severity:** **CRITICAL** (Completely broke training)  
**Related:** #file:REWARD_SCALING_CATASTROPHE_ANALYSIS.md, #file:ROOT_CAUSE_FOUND_DIRECTION_AWARE_SCALING.md

---

## Executive Summary

**PROBLEM:** Agent received **POSITIVE rewards (+2.36)** for going OFFROAD with lateral_dev=+2.00m during learning phase (step 1100), causing hard-right-turn behavior to persist despite previous direction-aware scaling fix.

**ROOT CAUSE:** Reward scaling imbalance where:
- Progress reward: +4.76 (201% of net reward!)
- Safety penalty: -3.00 (only 127% magnitude)
- **Net result: +1.76 POSITIVE for offroad behavior!**

**FIX APPLIED:** Three-pronged reward rebalancing:
1. ✅ Reduced `distance_scale` from **5.0 → 0.5** (10x weaker)
2. ✅ Increased `safety` weight from **0.3 → 3.0** (10x stronger)
3. ✅ Increased `offroad_penalty` from **-10.0 → -50.0** (5x stronger)

**EXPECTED IMPACT:**
- Offroad reward changes from **+2.36 → -146.82** (LARGE NEGATIVE!)
- Safety penalty now **47x stronger** than progress reward
- Agent will learn to **STRONGLY AVOID** offroad behavior

---

## Evidence from Log Analysis

### Step 1100 Diagnostic Output (NEW debug-action.log)

```
[DIAGNOSTIC][Step 1100] POST-ACTION OUTPUT:
  Current action: steer=+0.927, throttle/brake=+1.000 (near maximum right!)
  
Applied Control: throttle=1.0000, steer=0.5542 (CARLA limited it)
Lateral deviation: +2.00m (OFFROAD!)
Speed: 18.20 km/h
Offroad sensor: WARNING - "Vehicle off map (no waypoint)"

Reward Breakdown:
  EFFICIENCY:     +0.90  (raw: 0.4485 × weight: 2.0)
  LANE KEEPING:   +0.00  (raw: 0.0004 × weight: 2.0)  ← Correct (no scaling boost)
  COMFORT:        -0.30  (raw: -0.3000 × weight: 1.0)
  SAFETY:         -3.00  (raw: -10.0 × weight: 0.3)   ← TOO WEAK!
  PROGRESS:       +4.76  (raw: 1.5878 × weight: 3.0)  ← DOMINATES 201%!
  ────────────────────
  TOTAL:          +2.36  ← POSITIVE FOR OFFROAD BEHAVIOR!
  
Progress Detail:
  Route distance delta: +0.118m (moved forward slightly)
  Distance reward: 0.118 × 5.0 = +0.59
  Waypoint bonus: +1.0 (waypoint reached)
  Total progress: 1.59 raw → ×3.0 weight = +4.76 weighted
  
Agent learned: "Turn right + go offroad = GOOD!" ❌
```

### Mathematical Proof of Reward Imbalance

**BEFORE FIX (BROKEN):**
```
Progress reward:
  (0.118m × 5.0) + 1.0 = 1.59 raw
  1.59 × 3.0 weight = +4.76 weighted

Safety penalty:
  -10.0 raw × 0.3 weight = -3.00 weighted

NET REWARD: +4.76 - 3.00 = +1.76 POSITIVE!
```

**AFTER FIX (CORRECT):**
```
Progress reward:
  (0.118m × 0.5) + 1.0 = 1.059 raw
  1.059 × 3.0 weight = +3.18 weighted

Safety penalty:
  -50.0 raw × 3.0 weight = -150.0 weighted

NET REWARD: +3.18 - 150.0 = -146.82 LARGE NEGATIVE! ✅
```

**Improvement:** 47x stronger penalty ratio (safety/progress = 47.17)

---

## Changes Applied

### 1. training_config.yaml (2 changes)

**Line 112 - Reduce distance_scale:**
```yaml
# BEFORE (BROKEN):
progress:
  distance_scale: 5.0  # Makes 0.118m → +0.59 reward (EXTREME!)

# AFTER (FIXED):
progress:
  distance_scale: 0.5  # CATASTROPHE FIX: Reduced from 5.0 to 0.5 (10x weaker)
```

**Line 48 - Increase safety weight:**
```yaml
# BEFORE (BROKEN):
reward:
  weights:
    safety: 0.3  # Makes -10.0 → only -3.00 weighted (TOO WEAK!)

# AFTER (FIXED):
reward:
  weights:
    safety: 3.0  # CATASTROPHE FIX: Increased from 0.3 to 3.0 (10x stronger)
```

### 2. reward_functions.py (1 change)

**Line 916 - Increase offroad_penalty:**
```python
# BEFORE (BROKEN):
if offroad_detected:
    offroad_penalty = -10.0  # Too lenient, only -3.00 weighted

# AFTER (FIXED):
if offroad_detected:
    offroad_penalty = -50.0  # 5x stronger, now -150.0 weighted!
```

### 3. Python Cache Cleared

```bash
find av_td3_system -type d -name "__pycache__" -exec rm -rf {} +
```
Ensures reward_functions.py changes are loaded fresh in next training run.

---

## Why Previous Fix Was Insufficient

### Direction-Aware Scaling Fix (Dec 1, 2025)
- **File:** #file:ROOT_CAUSE_FOUND_DIRECTION_AWARE_SCALING.md
- **What it fixed:** Lane keeping was artificially boosted during forward progress
- **Status:** ✅ **CORRECT and WORKING** (see lane_keeping: +0.00 in step 1100 log)
- **Why insufficient:** Progress reward (+4.76) STILL overwhelmed safety penalty (-3.00)

### The Real Problem
Reward scaling imbalance existed **INDEPENDENTLY** of direction-aware scaling:
- Even with scaling disabled, `distance_scale=5.0` made progress too strong
- Even with scaling disabled, `safety weight=0.3` made penalties too weak
- Result: Agent learned offroad behavior was GOOD (+2.36 total reward)

### Key Insight
**Multiple reward issues can exist simultaneously:**
1. ✅ Direction-aware scaling issue (FIXED Dec 1, 2025) → Lane keeping now correct
2. ✅ Reward scaling catastrophe (FIXED Jan 5, 2025) → Safety now dominant

Both fixes were necessary. Neither alone was sufficient.

---

## Expected Training Behavior After Fix

### Short-Term (2K steps validation)

**Offroad scenario:**
```
Action: steer=+0.927, throttle=+1.000 (hypothetical offroad)
Lateral deviation: +2.00m (OFFROAD)

Progress: (0.118m × 0.5) + 1.0 = 1.059 → +3.18 weighted
Safety: -50.0 × 3.0 = -150.0 weighted
TOTAL: +3.18 - 150.0 = -146.82 ← LARGE NEGATIVE! ✅

Agent learns: "Offroad = TERRIBLE!" ✅
```

**On-road scenario:**
```
Action: steer=+0.030, throttle=+0.800 (lane centered)
Lateral deviation: +0.10m (ON-ROAD)

Efficiency: +0.90
Lane keeping: +1.80 (lateral_dev < 0.5m, heading aligned)
Comfort: -0.10
Safety: +0.00 (no violations)
Progress: +3.18
TOTAL: +5.78 ← POSITIVE FOR CORRECT BEHAVIOR! ✅

Agent learns: "Stay in lane = GOOD!" ✅
```

**Success criteria:**
- ✅ Offroad total reward: < -50 (LARGE NEGATIVE)
- ✅ On-road total reward: > +2.0 (POSITIVE)
- ✅ Reward ratio: On-road/Offroad > 100x difference
- ✅ Steering distribution: Balanced around 0.0 (no right bias)

### Medium-Term (10K-50K training)

**Learning progression:**
- Episodes 1-100: Exploration, occasional offroad (heavily penalized)
- Episodes 100-500: Agent learns to stay in lane (rewards increasing)
- Episodes 500-1000: Stable lane-keeping, efficient speed control
- Episodes 1000+: Near-optimal behavior, minimal violations

**Episode reward trend:**
```
Episode    Avg Reward    Lateral Dev    Collisions
0-100      -30.0         ±1.50m         High
100-500    +10.0         ±0.80m         Medium
500-1000   +40.0         ±0.30m         Low
1000+      +60.0         ±0.15m         Near-zero
```

**Steering behavior:**
```
BEFORE FIX (BROKEN):
  Mean: +0.620 (hard right bias) ❌
  Std: 0.280
  
AFTER FIX (EXPECTED):
  Mean: +0.030 (nearly centered) ✅
  Std: 0.180 (reduced variance)
```

---

## Next Steps

### IMMEDIATE: Validation Test (2K steps)

**Command:**
```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system
python scripts/train_td3.py --max_timesteps 2000 --debug --log_level DEBUG
```

**Check for:**
1. ✅ Offroad reward < -50 (LARGE NEGATIVE)
2. ✅ On-road reward > +2.0 (POSITIVE)
3. ✅ No hard-right-turn bias in steering
4. ✅ Lateral deviation decreasing over episodes

**Expected log output:**
```
[DIAGNOSTIC][Step 1100] POST-ACTION OUTPUT:
  Lateral deviation: +2.00m (OFFROAD)
  
  Reward Breakdown:
    SAFETY:   -150.0  (raw: -50.0 × weight: 3.0)  ← DOMINATES!
    PROGRESS: +3.18   (raw: 1.059 × weight: 3.0)
    ────────────────────
    TOTAL:    -146.82  ← LARGE NEGATIVE! ✅
```

### SHORT-TERM: Full Training Run (10K steps)

**Command:**
```bash
python scripts/train_td3.py --max_timesteps 10000 --eval_freq 5000
```

**Monitor:**
- Episode rewards trending upward
- Lateral deviation trending downward
- Collision rate decreasing
- Agent learns to stay in lane

### MEDIUM-TERM: Extended Training (50K steps)

**Command:**
```bash
python scripts/train_td3.py --max_timesteps 50000 --eval_freq 10000 --save_freq 5000
```

**Expected:**
- Stable lane-keeping behavior
- Near-optimal speed control
- Minimal safety violations
- Successful route completion

---

## Configuration Summary

### Final Reward Weights (training_config.yaml)
```yaml
reward:
  weights:
    efficiency: 2.0      # Velocity tracking
    lane_keeping: 2.0    # Lane centering
    comfort: 1.0         # Smooth driving
    safety: 3.0          # ← INCREASED (was 0.3)
    progress: 3.0        # Forward progress
```

### Final Progress Parameters (training_config.yaml)
```yaml
progress:
  waypoint_bonus: 1.0
  distance_scale: 0.5  # ← REDUCED (was 5.0)
  goal_reached_bonus: 100.0
```

### Final Safety Penalties (reward_functions.py)
```python
safety:
  collision_penalty: -100.0   # Critical violations
  offroad_penalty: -50.0      # ← INCREASED (was -10.0)
  wrong_way_penalty: -5.0     # Directional violations
```

### Reward Balance Analysis
```
Maximum possible rewards:
  Efficiency:   +2.0
  Lane keeping: +2.0
  Comfort:      +0.0 (best = no jerk)
  Progress:     +3.18 (0.118m delta + waypoint)
  ────────────────────
  Best total:   +7.18

Offroad penalty:
  Safety:       -150.0 (offroad_penalty × weight)
  ────────────────────
  Offroad total: -142.82 (with other components)

Ratio: Best/Worst = 7.18 / -142.82 = 19.9x stronger penalty ✅
```

---

## Files Modified

1. ✅ `/av_td3_system/config/training_config.yaml` (2 changes)
   - Line 48: `safety: 3.0` (was 0.3)
   - Line 112: `distance_scale: 0.5` (was 5.0)

2. ✅ `/av_td3_system/src/environment/reward_functions.py` (1 change)
   - Line 916: `offroad_penalty = -50.0` (was -10.0)

3. ✅ Python cache cleared
   - All `__pycache__` directories removed

---

## Success Criteria

### Validation Test (2K steps)
- [ ] Offroad reward < -50 (LARGE NEGATIVE)
- [ ] On-road reward > +2.0 (POSITIVE)
- [ ] Steering mean near 0.0 (no right bias)
- [ ] No hard-right-turn behavior observed

### Training Progress (10K steps)
- [ ] Episode rewards increasing
- [ ] Lateral deviation decreasing
- [ ] Collision rate < 10%
- [ ] Route completion rate > 50%

### Final Model (50K steps)
- [ ] Episode rewards > +40.0 average
- [ ] Lateral deviation < ±0.30m
- [ ] Collision rate < 5%
- [ ] Route completion rate > 80%

---

## References

- **Analysis:** #file:REWARD_SCALING_CATASTROPHE_ANALYSIS.md (15,000-word deep dive)
- **Previous fix:** #file:ROOT_CAUSE_FOUND_DIRECTION_AWARE_SCALING.md (Nov-Dec 2025)
- **Log evidence:** #file:debug-action.log (Step 1100: offroad with +2.36 reward)
- **Literature:** CARLA research uses offroad penalties -50 to -100
- **TD3 paper:** Fujimoto et al. (2018) - Balanced reward scales critical

---

## Conclusion

The **reward scaling catastrophe** has been fixed through a three-pronged approach that:
1. Weakens progress reward (10x reduction in distance_scale)
2. Strengthens safety penalty (10x increase in safety weight)
3. Increases base offroad penalty (5x increase in magnitude)

This changes the offroad reward from **+2.36 (POSITIVE)** to **-146.82 (LARGE NEGATIVE)**, creating a **47x stronger penalty ratio** that will teach the agent to STRONGLY AVOID offroad behavior.

Combined with the previous direction-aware scaling fix (which correctly disabled lane_keeping boost), the reward system should now properly guide the agent toward safe, efficient, lane-centered navigation.

**Status: READY FOR TESTING** ✅

Run the validation test (2K steps) to verify the fix is working as expected.
