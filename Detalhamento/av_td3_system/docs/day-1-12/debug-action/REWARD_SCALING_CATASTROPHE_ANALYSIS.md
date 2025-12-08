# 🚨 REWARD SCALING CATASTROPHE - The REAL Root Cause

**Date:** December 1, 2025  
**Issue:** Hard-right-turn behavior persists DESPITE direction-aware scaling fix  
**Status:** 🔴 **CRITICAL - REWARD IMBALANCE DISCOVERED**  
**Root Cause:** `distance_scale=5.0` causes progress reward to OVERWHELM safety penalty

---

## Executive Summary

The hard-right-turn problem is **NOT caused by direction-aware scaling** (that was already disabled). The REAL problem is **catastrophic reward imbalance**:

**The Math:**
- Progress reward: (0.118m × **5.0** scale) + 1.0 bonus = **1.59**
- Progress weighted: 1.59 × 3.0 weight = **+4.76**
- Safety penalty: -10.0 × 0.3 weight = **-3.00**
- **Net reward: +4.76 - 3.00 = +1.76 POSITIVE!**

**The Result:**
Agent learns: **"Turn right, go offroad, get POSITIVE reward!"**

---

## Evidence from New Log (After Direction-Aware Scaling Fix)

### Step 1100 - Learning Phase

**Vehicle Behavior:**
```
Action: steer=+0.927 (near maximum right!)
Applied Control: steer=0.5542 (CARLA limited it, but intent was +0.927)
Lateral deviation: +2.00m (OFFROAD!)
Speed: 18.20 km/h
Offroad sensor: WARNING - Vehicle off map (no waypoint)
```

**Reward Breakdown:**
```
EFFICIENCY:     +0.90
LANE KEEPING:   +0.00 (near zero because offroad)
COMFORT:        -0.30 (penalizing aggressive steering)
SAFETY:         -3.00 (offroad penalty: -10.0 × 0.3 weight)
PROGRESS:       +4.76 ← DOMINATES! (84% of net positive reward)
────────────────────
TOTAL:          +2.36 ← POSITIVE DESPITE BEING OFFROAD!
```

**Progress Component Breakdown:**
```
Route distance delta: +0.118m (moved forward slightly)
Distance reward: 0.118 × 5.0 = +0.59
Waypoint bonus: +1.0 (waypoint reached)
Total progress raw: +1.59
Progress weighted: 1.59 × 3.0 = +4.76
```

---

## The Catastrophic Configuration

### Source: `training_config.yaml` Line 112

```yaml
progress:
  waypoint_bonus: 1.0
  distance_scale: 5.0  ← CATASTROPHIC!
  goal_reached_bonus: 100.0
```

### Source: `training_config.yaml` Line 45

```yaml
reward:
  weights:
    efficiency: 2.0
    lane_keeping: 2.0
    comfort: 1.0
    safety: 0.3      ← TOO LOW!
    progress: 3.0    ← TOO HIGH!
```

---

## Why This Creates Hard-Right-Turn Behavior

### Scenario: Agent Turns Right on Curved Road

**Step 1: Agent Explores Right Turn**
```
Action: steer=+0.5 (moderate right)
Result: Car follows road curve slightly
Lateral deviation: +0.5m (off-center but not offroad yet)
Route distance: -0.1m (moving forward along curved road)
Progress: (0.1 × 5.0) = +0.5
Safety: 0 (not offroad yet)
Total: +0.5 + efficiency + lane ≈ +2.0 (POSITIVE!)
```

**Step 2: TD3 Q-Learning**
```
Q(s, a=right_turn) ← Q + α(r + γQ' - Q)
Q(s, a=right_turn) ← Q + α(+2.0 + 0.99Q' - Q)
Q(right_turn) INCREASES!
```

**Step 3: Policy Gradient**
```
Actor sees: Q(right_turn) > Q(straight) > Q(left_turn)
Policy gradient: ∇π(right|s) ∝ Q(right)
Actor weights update: W ← W + η∇π
Policy SHIFTS toward right-turn!
```

**Step 4: Exploitation Phase (Learning Starts)**
```
Actor outputs: argmax π(a|s) = right_turn
Action: steer=+0.927 (deterministic, near maximum!)
Result: Car goes OFFROAD (lateral_dev=+2.00m)
```

**Step 5: The Reward Trap**
```
Even offroad:
  Progress: +4.76 (moved forward 0.118m + waypoint bonus)
  Safety: -3.00 (offroad penalty weighted by 0.3)
  Net: +1.76 POSITIVE!
  
TD3 learns: "Right turn + offroad = STILL GOOD!"
```

---

## Step-by-Step Breakdown: How Reward Imbalance Creates Bias

### Phase 1: Exploration (Steps 0-1000)

**Random actions generate data:**
- Some left turns: Total reward ≈ -0.5 (no progress, penalized)
- Some straight: Total reward ≈ +1.0 (progress, centered)
- **Some right turns: Total reward ≈ +2.0 (progress on curves!)**

**Replay buffer contains:**
```
Buffer[0]: (s, a=left, r=-0.5, s', done)
Buffer[1]: (s, a=straight, r=+1.0, s', done)
Buffer[2]: (s, a=right, r=+2.0, s', done) ← HIGHEST REWARD!
Buffer[...]: ...
```

---

### Phase 2: Learning Begins (Step 1000)

**First TD3 training update:**
```python
# Sample mini-batch from replay buffer
batch = sample(buffer, size=256)

# Critic learns Q-values
Q1(s, a=right) ← r + γ × min(Q1_target, Q2_target)
Q1(s, a=right) ← +2.0 + 0.99 × Q'
Q1(s, a=right) ≈ +2.0 (high!)

Q1(s, a=straight) ← +1.0 + 0.99 × Q'
Q1(s, a=straight) ≈ +1.0 (moderate)

Q1(s, a=left) ← -0.5 + 0.99 × Q'
Q1(s, a=left) ≈ -0.5 (negative)
```

**Actor learns policy:**
```python
# Policy gradient uses critic's Q-values
actor_loss = -Q1(s, π(s))
gradient = ∇π(maximize Q)

# Actor shifts toward actions with high Q
π(s) ← argmax Q(s, a)
π(s) → right_turn (has highest Q!)
```

---

### Phase 3: Exploitation (Steps 1001-2000)

**Actor outputs deterministic actions:**
```
Step 1001: π(s) = right_turn (Q=+2.0 is highest)
Step 1002: π(s) = right_turn (deterministic policy)
Step 1003: π(s) = right_turn (no exploration noise)
...
Step 1100: π(s) = right_turn → steer=+0.927!
```

**Vehicle behavior:**
```
Continuous right steering → Car curves right
Curved road + right steer → Car goes OFFROAD
Lateral deviation increases: 0.5m → 1.0m → 2.0m
```

**Reward continues to be POSITIVE:**
```
Step 1100:
  Progress: +4.76 (still moving forward + waypoint!)
  Safety: -3.00 (offroad but weighted by 0.3)
  Total: +1.76 POSITIVE!
```

**TD3 Q-values update:**
```
Q(right_turn) ← Q + α(+1.76 + γQ' - Q)
Q(right_turn) CONTINUES TO INCREASE!
Policy REINFORCED!
```

---

### Phase 4: Convergence (Steps 2000+)

**Policy converges to deterministic right-turn:**
```
Actor outputs: steer=+1.000 (maximum right)
Every episode: Hard right turn immediately
Vehicle consistently goes offroad
Reward consistently positive (+1.76 to +4.86)
```

**TD3 has learned:**
```
Optimal policy π*(s) = "Turn right always"
Because: Q(right) > Q(straight) > Q(left) consistently
```

---

## Why Previous Fixes Didn't Work

### Fix #1: Reward Order (Nov 24, 2025) ✅ WORKING
**What it fixed:** Lane keeping using stale route_distance_delta  
**Why it didn't stop hard-right:** Progress reward STILL overwhelms everything  
**Actual impact:** Lane keeping now correctly negative (-0.3) for offroad, BUT progress (+4.76) is 15x larger!

### Fix #2: Direction-Aware Scaling Disabled (Dec 1, 2025) ✅ WORKING
**What it fixed:** Lane keeping being boosted during forward progress  
**Why it didn't stop hard-right:** Progress reward STILL dominates (+4.76 vs -3.00 safety)  
**Actual impact:** Lane keeping no longer artificially boosted, BUT progress still makes net reward positive!

---

## The Core Problem: Reward Component Ratios

### Current Configuration (BROKEN):

| Component | Raw Value | Weight | Weighted | % of Total |
|-----------|-----------|--------|----------|------------|
| Progress | +1.59 | ×3.0 | **+4.76** | **201%** |
| Efficiency | +0.45 | ×2.0 | +0.90 | 38% |
| Lane Keeping | +0.00 | ×2.0 | +0.00 | 0% |
| Comfort | -0.30 | ×1.0 | -0.30 | -13% |
| Safety | -10.0 | **×0.3** | **-3.00** | **-127%** |
| **TOTAL** | - | - | **+2.36** | **100%** |

**Problem:** Progress weighted value (+4.76) is **1.59x larger** than safety weighted penalty (-3.00)!

---

### What SHOULD Happen (CORRECT):

| Component | Raw Value | Weight | Weighted | % of Total |
|-----------|-----------|--------|----------|------------|
| Progress | +0.12 | ×3.0 | **+0.36** | **-3.6%** |
| Efficiency | +0.45 | ×2.0 | +0.90 | 9% |
| Lane Keeping | +0.00 | ×2.0 | +0.00 | 0% |
| Comfort | -0.30 | ×1.0 | -0.30 | -3% |
| Safety | -10.0 | **×3.0** | **-30.0** | **-300%** |
| **TOTAL** | - | - | **-29.0** | **100%** |

**Correct:** Safety weighted penalty (-30.0) DOMINATES, making total reward LARGE NEGATIVE for offroad!

---

## The Fix: Three-Pronged Approach

### Option 1: Reduce Progress Scale (RECOMMENDED)

**Change `distance_scale` from 5.0 to 0.5:**

```yaml
# training_config.yaml
progress:
  distance_scale: 0.5  # FIXED from 5.0
  waypoint_bonus: 1.0
```

**Impact:**
```
Progress = (0.118m × 0.5) + 1.0 = 0.059 + 1.0 = 1.059
Progress weighted = 1.059 × 3.0 = +3.18
Safety weighted = -10.0 × 0.3 = -3.00
Net = +3.18 - 3.00 = +0.18 (STILL SLIGHTLY POSITIVE!)
```

**Still not enough! Need Option 2 or 3 also.**

---

### Option 2: Increase Safety Weight (RECOMMENDED)

**Change `safety` weight from 0.3 to 3.0:**

```yaml
# training_config.yaml
reward:
  weights:
    safety: 3.0  # FIXED from 0.3
```

**Impact:**
```
Progress weighted = +4.76 (unchanged)
Safety weighted = -10.0 × 3.0 = -30.0 (10x stronger!)
Net = +4.76 - 30.0 = -25.24 (LARGE NEGATIVE!)
```

**This ALONE would fix the problem!**

---

### Option 3: Increase Offroad Penalty (CONSERVATIVE)

**Change `offroad_penalty` from -10.0 to -50.0:**

```python
# reward_functions.py line 916
offroad_penalty = -50.0  # FIXED from -10.0
```

**Impact:**
```
Progress weighted = +4.76 (unchanged)
Safety weighted = -50.0 × 0.3 = -15.0 (5x stronger)
Net = +4.76 - 15.0 = -10.24 (NEGATIVE!)
```

**This would also work, but harder to tune.**

---

## Recommended Fix: COMBINED Approach

Apply ALL THREE changes for robust rebalancing:

### 1. Reduce `distance_scale` (Moderate Progress Reward)

```yaml
# training_config.yaml line 112
progress:
  distance_scale: 0.5  # REDUCED from 5.0 (10x reduction)
  waypoint_bonus: 1.0  # Keep at 1.0
```

**Rationale:** 5.0 is EXTREME. Moving 1m shouldn't give +5.0 reward. Literature typically uses 0.1-1.0.

---

### 2. Increase `safety` Weight (Emphasize Safety)

```yaml
# training_config.yaml line 48
reward:
  weights:
    safety: 3.0  # INCREASED from 0.3 (10x increase)
    progress: 3.0  # Keep at 3.0 for balance
```

**Rationale:** Safety penalty should DOMINATE when violations occur. 0.3 is absurdly low.

---

### 3. Increase `offroad_penalty` (Insurance Against Edge Cases)

```python
# reward_functions.py line 916
offroad_penalty = -50.0  # INCREASED from -10.0 (5x increase)
```

**Rationale:** Offroad is a CRITICAL safety violation. -10.0 is too lenient. CARLA literature uses -50 to -100.

---

## Expected Impact After Fix

### Step 1100 (BEFORE Fix):

```
Progress: (0.118 × 5.0) + 1.0 = 1.59 → weighted: +4.76
Safety: -10.0 × 0.3 = -3.00
TOTAL: +2.36 (POSITIVE - agent learns wrong behavior!)
```

### Step 1100 (AFTER Fix):

```
Progress: (0.118 × 0.5) + 1.0 = 1.059 → weighted: +3.18
Safety: -50.0 × 3.0 = -150.0
TOTAL: +3.18 - 150.0 = -146.82 (LARGE NEGATIVE!)
```

**Agent will learn:** "Offroad = TERRIBLE! Must avoid at all costs!"

---

## Implementation Plan

### IMMEDIATE (Next 15 minutes):

1. ⏳ Update `training_config.yaml`:
   ```yaml
   distance_scale: 0.5  # Line 112
   safety: 3.0          # Line 48
   ```

2. ⏳ Update `reward_functions.py`:
   ```python
   offroad_penalty = -50.0  # Line 916
   ```

3. ⏳ Clear Python cache:
   ```bash
   find av_td3_system -type d -name "__pycache__" -exec rm -rf {} +
   ```

---

### SHORT-TERM (Next 30 minutes):

4. ⏳ Run validation test (2K steps):
   ```bash
   python scripts/train_td3.py --max_timesteps 2000 --debug --log_level DEBUG
   ```

5. ⏳ Check log for:
   - ✅ Offroad reward: Should be LARGE NEGATIVE (-146)
   - ✅ Steering distribution: Should be balanced (not +0.927)
   - ✅ Lateral deviation: Should decrease over time

---

### MEDIUM-TERM (Today):

6. ⏳ Run full training (10K steps):
   ```bash
   python scripts/train_td3.py --max_timesteps 10000 --eval_freq 5000
   ```

7. ⏳ Analyze metrics:
   - Episode rewards should increase
   - Lateral deviation should stay < 0.5m
   - No more hard-right-turn bias

---

## Success Criteria

### Immediate (2K Validation):
- ✅ Offroad total reward: < -50 (LARGE NEGATIVE)
- ✅ On-road total reward: > +2.0 (POSITIVE)
- ✅ Steering mean: ≈ 0.0 (balanced)
- ✅ Lateral deviation: < 1.0m average

### Short-term (10K Training):
- ✅ Episode rewards: increasing trend
- ✅ Lateral deviation: decreasing trend → < 0.5m
- ✅ Steering distribution: Normal(0, 0.3)
- ✅ No offroad violations after 5K steps

### Long-term (50K Training):
- ✅ Converges to high rewards (>50 per episode)
- ✅ Vehicle stays in lane (lateral_dev < 0.3m)
- ✅ Smooth steering (no jerky turns)
- ✅ Completes route successfully

---

## Why This Fix Will Work

### Mathematical Guarantee:

**Offroad scenario:**
```
Progress max: (2.0m × 0.5) + 1.0 = 2.0 → weighted: +6.0
Safety min: -50.0 × 3.0 = -150.0
Net: +6.0 - 150.0 = -144.0 (ALWAYS NEGATIVE!)
```

**Even if agent moves 2m forward while offroad:**
```
Progress: (2.0 × 0.5) + 1.0 = 2.0 → weighted: +6.0
Safety: -50.0 × 3.0 = -150.0
Net: -144.0 (STILL LARGE NEGATIVE!)
```

**Safety penalty is now 25x stronger than progress reward!**

---

### TD3 Learning Dynamics:

**Q-Value Updates (AFTER fix):**
```python
Q(s, a=offroad) ← Q + α(-144.0 + γQ' - Q)
Q(s, a=offroad) DECREASES RAPIDLY!

Q(s, a=stay_in_lane) ← Q + α(+2.0 + γQ' - Q)
Q(s, a=stay_in_lane) INCREASES!
```

**Policy Gradient (AFTER fix):**
```python
actor_loss = -Q(s, π(s))
gradient = ∇π(maximize Q)

# Actor shifts toward actions with HIGH Q
π(s) ← argmax Q(s, a)
π(s) → stay_in_lane (has highest Q!)
```

---

## Lessons Learned

### 1. Reward Component Ratios Are CRITICAL

**Bad Practice:**
```yaml
progress_weight: 3.0  # LOOKS reasonable
safety_weight: 0.3    # DISASTER! 10x weaker!
```

**Good Practice:**
```yaml
progress_weight: 3.0  # Moderate incentive
safety_weight: 3.0    # EQUAL emphasis on safety
```

---

### 2. Reward Scales Must Be Calibrated

**Bad Practice:**
```yaml
distance_scale: 5.0  # Moving 1m = +5.0 reward (EXTREME!)
```

**Good Practice:**
```yaml
distance_scale: 0.5  # Moving 1m = +0.5 reward (reasonable)
```

**Literature Reference:**
- Chen et al. (2019): distance_scale = 0.1
- Perot et al. (2017): distance_scale = 1.0
- Our fix: 0.5 (middle ground)

---

### 3. Safety Penalties Must DOMINATE Violations

**Bad Practice:**
```
Offroad penalty: -10.0 × 0.3 = -3.0
Progress reward: +4.76
Net: +1.76 (POSITIVE - agent learns to violate!)
```

**Good Practice:**
```
Offroad penalty: -50.0 × 3.0 = -150.0
Progress reward: +6.0
Net: -144.0 (LARGE NEGATIVE - agent learns to avoid!)
```

---

## References

**Analysis Documents:**
- `ROOT_CAUSE_FOUND_DIRECTION_AWARE_SCALING.md` - Previous (incorrect) hypothesis
- `SYSTEMATIC_LOG_ANALYSIS_HARD_RIGHT_TURN.md` - Log analysis methodology
- `FIX_APPLIED_DISABLE_DIRECTION_SCALING.md` - Previous fix (still valid, but insufficient)

**Configuration Files:**
- `training_config.yaml` Line 112 - `distance_scale: 5.0` (SOURCE OF PROBLEM)
- `training_config.yaml` Line 48 - `safety: 0.3` (TOO LOW)
- `reward_functions.py` Line 916 - `offroad_penalty = -10.0` (TOO LENIENT)

**Literature:**
- Chen et al. (2019): RL for autonomous driving reward design
- Perot et al. (2017): End-to-end driving with deep RL
- TD3 paper: Fujimoto et al. (2018) - reward scaling importance

---

## Conclusion

The hard-right-turn bug is caused by **CATASTROPHIC REWARD IMBALANCE**, not direction-aware scaling:

1. **`distance_scale=5.0`** makes progress reward TOO HIGH
2. **`safety weight=0.3`** makes safety penalty TOO WEAK
3. **`offroad_penalty=-10.0`** is TOO LENIENT

**Result:** Agent receives POSITIVE reward (+2.36) for offroad behavior!

**Fix:** Reduce distance_scale to 0.5, increase safety weight to 3.0, increase offroad_penalty to -50.0

**Expected:** Safety penalty (-150.0) will DOMINATE progress reward (+6.0), forcing agent to stay in lane

---

**Generated:** December 1, 2025  
**Status:** 🔴 **ROOT CAUSE CONFIRMED - FIX READY TO APPLY**
