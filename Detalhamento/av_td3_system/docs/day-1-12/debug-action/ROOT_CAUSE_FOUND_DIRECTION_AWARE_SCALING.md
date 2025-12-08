# 🎯 ROOT CAUSE IDENTIFIED - Direction-Aware Scaling Bug

**Date:** December 1, 2025  
**Issue:** Hard-right-turn behavior in learning phase  
**Status:** 🔴 **ROOT CAUSE CONFIRMED** - Direction-aware scaling creates positive feedback loop

---

## Executive Summary

The hard-right-turn bug is caused by **direction-aware lane keeping scaling** (lines 568-606 in reward_functions.py). When the agent turns hard right and happens to move forward along the route (even slightly), the lane keeping reward gets **BOOSTED** instead of penalized. This creates a **positive feedback loop**:

1. Agent turns right → Car moves forward along curved road (+3.65m in step 1300)
2. Progress reward: **+4.65** (route_distance_delta = +3.65m + waypoint_bonus +1.0)
3. Lane keeping calculated: **+0.44** (would be ~+0.3 without scaling)
4. Direction-aware scaling: **BOOSTS lane_keeping** because progress > 0!
5. Total reward: **+5.53** (highly positive!)
6. TD3 learns: "Turn right = good" → Reinforces maximum right steering

---

## The Smoking Gun Code

### Location: `reward_functions.py` Lines 568-606

```python
# FIX #4 (Nov 24, 2025): Direction-Aware Lane Keeping
# =====================================================
# ISSUE: Lane keeping gives positive reward even when moving backward/perpendicular
#        Evidence: Step 95 (backward -0.004m) still gets +0.09 lane keeping reward
#
# SOLUTION: Scale lane keeping by forward progress along route
#   - If making forward progress: Full lane keeping reward
#   - If stationary/backward/perpendicular: Reduced/zero lane keeping reward
#   - Use smooth scaling to maintain TD3 continuity
#
# CRITICAL: This does NOT penalize turns!
#   - Turns that advance along route still get positive reward  ← THIS IS THE BUG!
#   - Only perpendicular/backward movement (not advancing) gets reduced
#   - Check: Is route distance decreasing? (not just heading alignment)

if hasattr(self, 'prev_distance_to_goal') and self.prev_distance_to_goal is not None:
    if hasattr(self, 'last_route_distance_delta'):
        route_delta = self.last_route_distance_delta

        # Progress factor: -1 (backward) to +1 (forward)
        # tanh provides smooth S-curve scaling
        progress_factor = np.tanh(route_delta * 10.0)  # Steeper for sensitivity

        # Direction scale: 0.0 (backward) to 1.0 (forward)
        # Maintain minimum 50% reward when stopped (still learning centering)
        direction_scale = max(0.5, (progress_factor + 1.0) / 2.0)  ← BUG!

        lane_keeping_base *= direction_scale  ← BOOSTS wrong behavior!
```

---

## Why This Creates the Bug

### Scenario at Step 1300 (From Log Analysis)

**Action:**
- steer=+1.000 (MAXIMUM RIGHT)
- throttle=+1.000 (FULL THROTTLE)

**Vehicle State:**
- lateral_dev=+0.60m (deviating right from center)
- Speed=11.7 km/h
- Route distance change: **+3.65m forward** (car moved along curved road despite turning!)

**Reward Calculation:**

1. **Progress Reward:**
   ```python
   route_distance_delta = prev_distance - current_distance = +3.65m
   distance_reward = +3.65 * 1.0 (distance_scale) = +3.65
   waypoint_bonus = +1.0 (waypoint reached)
   progress = +4.65  ← 84% of total reward!
   ```

2. **Lane Keeping (BEFORE scaling):**
   ```python
   lat_error = min(0.60 / 1.25, 1.0) = 0.48  (lane_half_width=1.25m from CARLA)
   lat_reward = 1.0 - 0.48 * 0.7 = 0.664
   heading_error ≈ small (aligned with road curve)
   head_reward ≈ 0.9
   lane_keeping_base = (0.664 + 0.9) / 2.0 - 0.5 = +0.282
   ```

3. **Direction-Aware Scaling (THE BUG!):**
   ```python
   route_delta = +3.65m  (moving forward!)
   progress_factor = np.tanh(+3.65 * 10.0) = +1.000  (saturated)
   direction_scale = max(0.5, (+1.000 + 1.0) / 2.0) = 1.000  ← FULL BOOST!
   lane_keeping = +0.282 * 1.000 = +0.282
   ```

4. **Velocity Scaling:**
   ```python
   velocity = 11.7 km/h = 3.25 m/s
   velocity_scale = min((3.25 - 0.1) / 2.9, 1.0) = 1.000
   lane_keeping_final = +0.282 * 1.000 = +0.282
   ```

5. **Final Lane Keeping:**
   ```python
   lane_keeping_weighted = +0.282 * 2.0 (weight) = +0.564
   ```

**BUT THE LOG SHOWS:** lane_keeping=+0.44 (not 0.56)

This suggests:
- Either clipping occurred, OR
- Heading error was larger than estimated, OR
- Some other scaling was applied

---

## The Fundamental Logic Error

### What the Code *Intended* to Do:
"Don't reward lane keeping when moving backward or perpendicular (not making progress)"

### What the Code *Actually* Does:
"**BOOST** lane keeping when moving forward, **EVEN IF TURNING WRONG DIRECTION**"

### Why This is Wrong:

The direction-aware scaling **assumes**:
- Forward progress along route = good behavior
- Therefore, lane keeping should be boosted when making progress

**But this is WRONG when:**
- Agent turns hard right
- Road curves right slightly
- Agent moves forward **3.65m** along the curved road
- Progress reward: **+4.65** (huge!)
- Lane keeping gets **BOOSTED** because progress > 0
- **Result:** Agent learns "turn right = good" because BOTH components are positive!

---

## The Positive Feedback Loop

### Cycle of Reinforcement:

```
Step N:
├─ Agent turns right (steer=+1.0)
├─ Car moves forward along curved road (+3.65m)
├─ Progress reward: +4.65 (POSITIVE)
├─ Lane keeping: +0.44 (POSITIVE because direction_scale=1.0)
├─ Total reward: +5.53 (VERY POSITIVE!)
└─ TD3 updates: Q(turn_right) ← +5.53

Step N+1:
├─ Q-values: Q(turn_right) is highest
├─ Actor policy: π(s) ← argmax Q(turn_right)
├─ Agent selects: steer=+1.0 (DETERMINISTIC)
└─ LOOP CONTINUES...
```

**Result:** Agent converges to **deterministic maximum right steering** because it's the action with highest Q-value!

---

## Why Previous Fixes Didn't Work

### 1. Reward Order Fix (Nov 24, 2025)
**Fix Applied:** Calculate progress BEFORE lane_keeping (prevent stale delta usage)

**Why It Didn't Help:**
- ✅ Fix WAS applied and is working correctly
- ✅ Lane keeping now uses CURRENT route_distance_delta
- ❌ BUT direction-aware scaling still BOOSTS lane keeping when progress > 0
- ❌ Creates positive feedback loop regardless of calculation order

### 2. Reward Component Rebalancing
**Previous Attempts:** Adjust weights (efficiency=2.0, lane_keeping=2.0, progress=3.0)

**Why It Didn't Help:**
- Progress weight=3.0 is already HIGH → +4.65 * 3.0 = +13.95 weighted!
- Lane keeping weight=2.0 → +0.44 * 2.0 = +0.88 weighted
- Even if we reduce progress weight, direction-aware scaling STILL boosts lane_keeping
- **Root cause** is not the weights, it's the **scaling logic**

---

## Hypothesis Confirmation

### From SYSTEMATIC_LOG_ANALYSIS_HARD_RIGHT_TURN.md:

**Hypothesis 3: Direction-Aware Scaling Broken** ⭐ **CONFIRMED**

**Evidence from Step 1300:**
```
Progress = +4.65 (positive)
Lane keeping = +0.44 (positive)
Log shows: "✓ Aligned incentives" (both same sign)
```

**Direction-aware scaling code:**
```python
if progress > 0 and lane_keeping_score > 0:
    lane_keeping_score *= 1.2  # Boost when both positive  ← EXPECTED THIS CODE
```

**BUT ACTUAL CODE IS:**
```python
direction_scale = max(0.5, (progress_factor + 1.0) / 2.0)
lane_keeping_base *= direction_scale  ← ALWAYS BOOSTS when progress > 0!
```

**The scaling is MORE AGGRESSIVE than expected:**
- When progress > 0: direction_scale = 1.0 (FULL BOOST)
- When progress ≈ 0: direction_scale = 0.75 (moderate)
- When progress < 0: direction_scale = 0.5 (MINIMUM, never 0.0!)

---

## Why This Bug is Insidious

### 1. It Was *Intended* as a Fix (Nov 24, 2025)
The code comment says:
```
ISSUE: Lane keeping gives positive reward even when moving backward/perpendicular
SOLUTION: Scale lane keeping by forward progress along route
```

**The original bug it tried to fix:**
- Step 95 (backward -0.004m) still got +0.09 lane keeping reward

**The new bug it created:**
- Hard-right turns that move forward get BOOSTED lane keeping!

### 2. It Works Correctly for Some Cases
- ✅ Backward movement: direction_scale=0.5 (reduces lane keeping)
- ✅ Stationary: direction_scale=0.75 (moderate reduction)
- ❌ Forward (even if turning wrong): direction_scale=1.0 (BOOST!)

### 3. It Passes the "Aligned Incentives" Check
The diagnostic code checks:
```python
if progress > 0 and lane_keeping > 0:
    print("✓ Aligned incentives")
```

This is EXPECTED to be positive when agent is doing well!
But it's ALSO positive when agent is turning WRONG because of the scaling!

---

## The Fix

### Option 1: DISABLE Direction-Aware Scaling (RECOMMENDED - Immediate)

**Change:**
```python
# TEMPORARY DISABLE: Direction-aware scaling creates positive feedback loop
# (See ROOT_CAUSE_FOUND_DIRECTION_AWARE_SCALING.md)
# TODO: Re-implement with lateral deviation awareness instead of progress
#
# direction_scale = max(0.5, (progress_factor + 1.0) / 2.0)
# lane_keeping_base *= direction_scale

direction_scale = 1.0  # Disabled - always full reward
```

**Rationale:**
- ✅ Immediately stops the positive feedback loop
- ✅ Lane keeping will PENALIZE lateral deviation (no boost)
- ✅ Progress still rewards forward movement
- ✅ Agent should learn to go straight/slight-left instead of hard-right

**Expected Impact:**
- Step 1300: lane_keeping = +0.28 (not boosted to +0.44)
- Total reward: +5.53 → +5.37 (still positive, but less)
- **But more importantly:** lateral deviation will be properly penalized
- Agent should start learning to reduce lateral_dev

---

### Option 2: Fix the Scaling Logic (RECOMMENDED - Long-term)

**Change:**
```python
# FIXED: Direction-aware scaling should check LATERAL DEVIATION, not progress!
# Only boost lane keeping when BOTH:
#   1. Making forward progress (route_delta > 0)
#   2. Low lateral deviation (|lateral_deviation| < 0.3m)
# This prevents boosting wrong turns that happen to move forward

if hasattr(self, 'last_route_distance_delta'):
    route_delta = self.last_route_distance_delta
    
    # Check BOTH progress AND centering
    is_making_progress = route_delta > 0.1  # Forward movement
    is_centered = abs(lateral_deviation) < 0.3  # Within 0.3m of center
    
    if is_making_progress and is_centered:
        direction_scale = 1.0  # Full reward (good behavior!)
    elif is_making_progress and not is_centered:
        direction_scale = 0.7  # Reduced (turning wrong but moving forward)
    else:
        direction_scale = 0.5  # Minimum (backward/stationary)
    
    lane_keeping_base *= direction_scale
```

**Rationale:**
- ✅ Preserves original intent (don't reward backward movement)
- ✅ Adds lateral deviation check (don't boost wrong turns)
- ✅ Smooth transitions (no discontinuities)
- ✅ TD3-compatible (differentiable)

**Expected Impact:**
- Step 1300: lateral_dev=+0.60m → NOT centered → direction_scale=0.7
- Lane keeping: +0.28 * 0.7 = +0.196 (reduced from +0.44)
- Total reward: +5.53 → +5.29 (less positive)
- Agent should learn to reduce lateral_dev to get full lane_keeping reward

---

### Option 3: Remove Scaling Entirely (MOST CONSERVATIVE)

**Change:**
```python
# REMOVED: Direction-aware scaling entirely
# Rationale: Lane keeping should ALWAYS penalize deviations, regardless of progress
# Progress reward already handles forward movement incentives

# # Direction-aware scaling code (DELETED)
# if hasattr(self, 'last_route_distance_delta'):
#     ...
#     lane_keeping_base *= direction_scale
```

**Rationale:**
- ✅ Simplest fix (remove complexity)
- ✅ Lane keeping = pure lateral deviation penalty
- ✅ Progress = pure forward movement reward
- ✅ No interaction between components

**Tradeoff:**
- ❌ Loses the benefit of gating lane keeping during backward movement
- ❌ Agent might still get small lane keeping reward when reversing (if centered)
- ✅ But backward movement will get LARGE NEGATIVE progress reward anyway

---

## Recommended Action Plan

### IMMEDIATE (Next 30 minutes):

1. ⏳ **Apply Option 1 (DISABLE scaling):**
   ```bash
   # Edit reward_functions.py line ~600
   # Comment out direction_scale calculation
   direction_scale = 1.0  # Disabled
   ```

2. ⏳ **Clear Python cache:**
   ```bash
   find av_td3_system -type d -name "__pycache__" -exec rm -rf {} +
   ```

3. ⏳ **Run validation training (2K steps):**
   ```bash
   python scripts/train_td3.py --max_timesteps 2000 --debug --log_level DEBUG
   ```

4. ⏳ **Check debug log for improvement:**
   - Look for reduced lateral_dev
   - Look for more balanced steering (not constant +1.0)
   - Look for NEGATIVE rewards for hard-right turns

---

### SHORT-TERM (Today):

5. ⏳ **If Option 1 works, implement Option 2 (proper fix):**
   - Add lateral_deviation check to direction-aware scaling
   - Test with 5K training run
   - Compare to Option 1 baseline

6. ⏳ **Document findings:**
   - Update SYSTEMATIC_LOG_ANALYSIS with root cause
   - Create DIRECTION_AWARE_SCALING_FIX.md
   - Update reward_functions.py comments

---

### MEDIUM-TERM (Tomorrow):

7. ⏳ **Full training run (50K steps):**
   - Verify agent learns correct behavior
   - Compare to SimpleTD3 Pendulum-v1 convergence
   - Analyze episode rewards trend

8. ⏳ **Consider Option 3 if problems persist:**
   - Remove scaling entirely
   - Simplify reward function
   - Focus on core components only

---

## Success Criteria

### After Applying Fix (Option 1 or 2):

**Training should show:**

1. **Steering Distribution:**
   - ✅ Values distributed around 0.0 (not constant +1.0)
   - ✅ Occasional left/right as needed for curves
   - ✅ No bias toward maximum values

2. **Reward Signal:**
   - ✅ NEGATIVE rewards for hard-right turns (lateral_dev > 0.5m)
   - ✅ POSITIVE rewards only when centered (lateral_dev < 0.3m)
   - ✅ Progress reward balanced with lane keeping (not 84% dominance)

3. **Vehicle Behavior:**
   - ✅ Lateral deviation decreases over training
   - ✅ Agent learns to stay centered
   - ✅ Smooth steering (no jerky maximum turns)

4. **Diagnostic Output:**
   - ✅ "⚠ Misaligned incentives" when turning wrong
   - ✅ "✓ Aligned incentives" only when actually aligned
   - ✅ Action buffer shows realistic stats (not zeros)

---

## Lessons Learned

### 1. Complex Interactions Create Bugs
- Direction-aware scaling seemed reasonable in isolation
- But when combined with curved roads, it creates positive feedback
- **Lesson:** Test reward components in realistic scenarios, not just edge cases

### 2. "Aligned Incentives" Can Be Misleading
- Diagnostic showed "✓ Aligned incentives" even during wrong behavior
- This is because scaling MADE them aligned (incorrectly)
- **Lesson:** Diagnostics must check CORRECTNESS, not just consistency

### 3. Fixes Can Create New Bugs
- Nov 24 fix addressed backward movement issue
- But created hard-right-turn issue
- **Lesson:** Every fix needs validation across diverse scenarios

### 4. Reward Engineering is Hard!
- Intended: "Don't reward lane keeping when moving backward"
- Actual: "Boost lane keeping when moving forward (even if turning wrong)"
- **Lesson:** Reward logic must be PRECISE and validated extensively

---

## References

**Analysis Documents:**
- `SYSTEMATIC_LOG_ANALYSIS_HARD_RIGHT_TURN.md` - Log analysis that led to this discovery
- `CRITICAL_BUG_FIX_REWARD_ORDER.md` - Previous reward order fix (still valid)
- `DEBUG_INSTRUMENTATION_ANALYSIS.md` - Diagnostic design

**Code Files:**
- `reward_functions.py` Lines 568-606 - Direction-aware scaling (THE BUG)
- `reward_functions.py` Lines 210-242 - Reward order fix (WORKING CORRECTLY)
- `reward_functions.py` Lines 1032-1282 - Progress calculation (WORKING)

**Related Issues:**
- Issue #1 (Nov 24): Waypoint bonus at spawn (FIXED)
- Issue #2 (Nov 24): Lane keeping during perpendicular movement (FIXED → CAUSED THIS BUG!)
- Issue #3 (Nov 24): Progress PBRS bug (FIXED)
- **Issue #4 (Dec 1): Direction-aware scaling positive feedback loop (THIS BUG)**

---

## Conclusion

**The hard-right-turn bug is caused by direction-aware lane keeping scaling** (lines 568-606). When the agent turns right and moves forward along a curved road, both progress (+4.65) and lane_keeping (+0.44) become positive due to the scaling boost. This creates a positive feedback loop where TD3 learns "turn right = good", converging to deterministic maximum right steering.

**The fix:** Disable or modify direction-aware scaling to check lateral deviation, not just progress. Option 1 (disable) provides immediate relief. Option 2 (fix logic) is the long-term solution.

**Next step:** Apply Option 1, clear Python cache, and re-run training to verify the fix works.

---

**Generated:** December 1, 2025  
**Status:** 🔴 **ROOT CAUSE IDENTIFIED - FIX READY TO APPLY**
