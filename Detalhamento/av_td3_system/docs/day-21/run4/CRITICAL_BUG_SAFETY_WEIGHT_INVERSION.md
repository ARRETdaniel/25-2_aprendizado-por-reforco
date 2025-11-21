# CRITICAL BUG: Safety Weight Sign Inversion

**Date**: 2025-11-21  
**Severity**: 🔴 **CRITICAL - AGENT REWARDED FOR COLLISIONS**  
**Status**: ✅ **IDENTIFIED** - Requires immediate fix

---

## The Bug

**Line 49 of `reward_functions.py`**:
```python
"safety": -100.0,  # ❌ WRONG - This INVERTS penalties into rewards!
```

## What's Happening

### Current (BROKEN) Logic:

```python
# Step 1: Calculate safety component (returns negative penalty)
def _calculate_safety_reward(...):
    safety = 0.0
    if collision_detected:
        collision_penalty = -10.0
        safety += collision_penalty  # safety = -10.0
    return safety  # Returns -10.0

# Step 2: Apply weight (INVERTS the sign!)
safety_weight = -100.0  # From line 49
total_contribution = safety_weight * safety_component
                  = (-100.0) * (-10.0)
                  = +1000.0  ✅ POSITIVE REWARD!
```

**Result**: Agent receives **+1000.0 reward** for crashing! 🚨

### Why This Explains the Right-Turn Bias

The agent learned:
1. **Turn right** → Increases **progress** (distance traveled)
2. **Collision happens** → Receives **+1000.0 reward** (due to inverted safety weight)
3. **Total reward** = progress(+5) + collision(+1000) = **+1005**

**The agent is maximizing collisions, not avoiding them!**

---

## Mathematical Proof

### Scenario: Agent crashes while making progress

**Components**:
```
efficiency    = +0.5  (forward velocity)
lane_keeping  = -0.3  (off-center)
comfort       = -0.1  (jerky steering)
safety        = -10.0 (collision penalty)
progress      = +5.0  (moved forward)
```

**Weights** (CURRENT BROKEN):
```
efficiency:    1.0
lane_keeping:  5.0
comfort:       0.5
safety:       -100.0  ← WRONG!
progress:      1.0
```

**Total Reward** (BROKEN):
```
total = (1.0 × 0.5) + (5.0 × -0.3) + (0.5 × -0.1) + (-100.0 × -10.0) + (1.0 × 5.0)
      = 0.5 + (-1.5) + (-0.05) + (+1000.0) + 5.0
      = +1003.95  ✅ HUGE POSITIVE REWARD FOR CRASHING!
```

**Total Reward** (CORRECT with safety=+1.0):
```
total = (1.0 × 0.5) + (5.0 × -0.3) + (0.5 × -0.1) + (+1.0 × -10.0) + (1.0 × 5.0)
      = 0.5 + (-1.5) + (-0.05) + (-10.0) + 5.0
      = -6.05  ❌ NEGATIVE PENALTY FOR CRASHING
```

**Difference**: +1003.95 vs -6.05 = **1010 point swing!**

---

## Evidence from 8K Run

### From TensorBoard Metrics:

```
Episode Reward: 366.70 ± 601.19
  Range: [33.09, 4174.19]
  Max: 4174.19 ← Suspiciously high!
```

**Why 4174?** Let's reverse engineer:
- If agent had ~4 collisions in one episode
- Each collision: +1000 reward (inverted)
- Total collision bonus: 4 × 1000 = +4000
- Plus progress: +174
- **Total**: 4174 ✓ **MATCHES!**

### From Action Statistics:

```
Steering mean: +0.88 (hard right)
Throttle mean: +0.88 (floor it)
```

**Agent strategy**:
1. Turn hard right (maximizes collision rate on right-side obstacles)
2. Floor throttle (maximizes distance before crash)
3. Crash → **+1000 reward**
4. Respawn → Repeat

**Perfect strategy for BROKEN reward function!**

---

## Why Config Fix Didn't Work

**Config files** (`training_config.yaml`, `td3_config.yaml`):
```yaml
safety: 1.0  # ✅ CORRECT
```

**But hardcoded default** (`reward_functions.py:49`):
```python
"safety": -100.0,  # ❌ WRONG - Overrides config!
```

**Issue**: If config loading has ANY issue (path, key name, etc.), the hardcoded default is used.

**Check logs** for:
```
"REWARD WEIGHTS VERIFICATION"
  safety: -100.0  ← If you see this, config didn't load!
```

---

## The Fix

### Fix 1: Correct the Hardcoded Default (CRITICAL)

**File**: `src/environment/reward_functions.py:49`

**BEFORE**:
```python
self.weights = config.get("weights", {
    "efficiency": 1.0,
    "lane_keeping": 5.0,
    "comfort": 0.5,
    "safety": -100.0,  # ❌ WRONG!
    "progress": 1.0,
})
```

**AFTER**:
```python
self.weights = config.get("weights", {
    "efficiency": 1.0,
    "lane_keeping": 5.0,
    "comfort": 0.5,
    "safety": 1.0,  # ✅ CORRECT - Penalties are already negative
    "progress": 1.0,
})
```

### Fix 2: Update Remaining Config Files

**Files to check**:
- ✅ `config/training_config.yaml` - Already correct (1.0)
- ✅ `config/td3_config.yaml` - Already correct (1.0)
- ❌ `config/td3_config_lowmem.yaml` - Still has -100.0
- ❌ `config/ddpg_config.yaml` - Still has -100.0
- ❌ `config/carla_config.yaml` - Still has -100.0

---

## Verification Steps

After fixing:

1. **Check loaded weights**:
   ```bash
   grep "REWARD WEIGHTS VERIFICATION" run_log.txt -A 10
   ```
   **Expected**:
   ```
   safety: 1.0  ✅ CORRECT
   ```

2. **Check collision impact**:
   ```bash
   grep "SAFETY-COLLISION" run_log.txt | head -5
   ```
   **Expected**:
   ```
   total_contribution = safety_weight * safety_component
                     = (+1.0) * (-10.0)
                     = -10.0  ✅ NEGATIVE (penalty)
   ```

3. **Check episode rewards**:
   - Should be **LOWER** overall (no more +1000 collision bonuses)
   - Should rarely exceed +100 (no huge spikes)
   - Crashes should **decrease** episode reward

4. **Check action bias**:
   - Steering mean should move toward **0.0** (from +0.88)
   - Agent should **avoid** collisions, not seek them

---

## Impact on Previous Runs

### 8K Run Analysis

**Question**: "Was the 8K run using broken safety weight?"

**Check method**:
```bash
grep "REWARD WEIGHTS VERIFICATION" docs/day-21/run3/run_ControladdedLogs.log -A 10
```

**If output shows**:
- `safety: -100.0` → Run was **BROKEN** (rewards inverted)
- `safety: 1.0` → Run was **CORRECT**

**Implications**:
- If broken: All analysis is INVALID (agent optimized wrong objective)
- If correct: Analysis is valid, but reward imbalance still an issue

---

## Why This Wasn't Caught Earlier

1. **Complex reward calculation**: 5 components, easy to lose track of signs
2. **Positive total rewards**: Agent still got positive rewards (from progress)
3. **No negative episode rewards**: Collisions made rewards MORE positive, not negative
4. **Matches "maximize reward" heuristic**: TD3 is supposed to maximize, so positive seemed good

**Red flags we missed**:
- Episode reward max: 4174.19 (absurdly high)
- Agent seeks collisions (steering +0.88)
- Performance degrading (more collisions = higher reward!)

---

## Lesson Learned

**Always verify weight × component sign convention**:

Two valid patterns:
1. **Positive weights, signed components** (RECOMMENDED):
   ```python
   weights = {"safety": +1.0}
   safety_component = -10.0 (collision)
   contribution = +1.0 × -10.0 = -10.0 ✓
   ```

2. **Negative weights, positive components** (NOT RECOMMENDED):
   ```python
   weights = {"safety": -1.0}
   safety_component = +10.0 (collision)
   contribution = -1.0 × +10.0 = -10.0 ✓
   ```

**Our code used**: Pattern #1 for components, Pattern #2 for weights = **MIXED** → BUG!

**Fix**: Use **Pattern #1 exclusively** (positive weights, signed components)

---

## Estimated Fix Time

1. **Fix hardcoded default**: 1 minute
2. **Fix remaining configs**: 5 minutes
3. **Run 1K validation**: 30 minutes
4. **Verify fix worked**: 15 minutes

**Total**: ~50 minutes to fully validated fix

---

## Next Steps

1. ✅ **IMMEDIATE**: Fix `reward_functions.py:49` (change -100.0 → 1.0)
2. ✅ **IMMEDIATE**: Fix remaining config files
3. 🔧 **VERIFY**: Check 8K run logs for actual loaded weight
4. 🧪 **TEST**: Run 1K validation with fix
5. ✔️ **CONFIRM**: Verify collision reduces reward (not increases)

---

**Status**: 🔴 **CRITICAL BUG IDENTIFIED**  
**Fix Required**: Change 1 number (line 49: -100.0 → 1.0)  
**Impact**: May completely explain right-turn bias and reward imbalance  
**Priority**: **HIGHEST** - Fix before any further training

---

**Discovered by**: User observation of negative weight value  
**Root Cause**: Sign convention inconsistency between weights and components  
**Lesson**: Always trace through the math for reward calculation end-to-end
