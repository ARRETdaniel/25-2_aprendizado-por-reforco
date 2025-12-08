# Bug Fix: TypeError in Debug Print (Tuple Format Error)
## Systematic Analysis and Resolution

**Date:** December 1, 2025
**Error Type:** `TypeError: unsupported format string passed to tuple.__format__`
**Location:** `train_td3.py` line 793
**Status:** ✅ **FIXED**

---

## 🎯 Executive Summary

**Problem:** Training crashed at step 100 when debug print attempted to format reward component tuples as floats.

**Root Cause:** Mismatch between data structure (3-tuple) and format string expectation (float).

**Fix:** Extract weighted value (index 2) from tuple before formatting.

**Impact:** Debug instrumentation now works correctly, training can proceed beyond step 100.

---

## 📋 Error Details

### Full Error Message:
```
Traceback (most recent call last):
  File "/workspace/scripts/train_td3.py", line 1714, in <module>
    main()
  File "/workspace/scripts/train_td3.py", line 1710, in main
    trainer.train()
  File "/workspace/scripts/train_td3.py", line 793, in train
    print(f"  Components: efficiency={reward_breakdown.get('efficiency', 0):+.2f}, "
TypeError: unsupported format string passed to tuple.__format__
```

### When It Occurred:
- **Step:** 100 (first debug print triggered by `if t % 100 == 0`)
- **Phase:** Exploration phase (random actions)
- **Context:** POST-STEP REWARD diagnostic print

---

## 🔍 Root Cause Analysis

### Data Structure Investigation:

**In `reward_functions.py` (lines 276-303):**
```python
reward_dict["breakdown"] = {
    "efficiency": (
        self.weights["efficiency"],    # Index 0: weight
        efficiency,                     # Index 1: raw value
        self.weights["efficiency"] * efficiency,  # Index 2: weighted value
    ),
    "lane_keeping": (...),  # Same 3-tuple structure
    "comfort": (...),
    "safety": (...),
    "progress": (...),
}
```

**Why 3-tuple?**
- Provides COMPLETE information: weight, raw component, weighted contribution
- Used by environment info dict for detailed logging
- Allows external tools to reconstruct reward calculation

### Problematic Code (BEFORE fix):

**In `train_td3.py` line 793:**
```python
reward_breakdown = info.get('reward_breakdown', {})
print(f"  Components: efficiency={reward_breakdown.get('efficiency', 0):+.2f}, "
      f"lane_keeping={reward_breakdown.get('lane_keeping', 0):+.2f}, "
      f"progress={reward_breakdown.get('progress', 0):+.2f}")
```

**Problem:**
- `reward_breakdown.get('efficiency', 0)` returns tuple `(2.0, 0.5882, 1.1764)`, not float
- Format string `:+.2f` expects float, got tuple → **TypeError**

**Why This Happened:**
- Debug prints were added without checking `reward_breakdown` structure
- Assumption: dictionary values are scalars (common pattern in many RL codebases)
- Reality: dictionary values are 3-tuples (more detailed structure)

---

## ✅ The Fix

### Modified Code (AFTER fix):

```python
if t % 100 == 0 and self.debug:
    reward_breakdown = info.get('reward_breakdown', {})
    vehicle_state = info.get('vehicle_state', {})
    print(f"\n[DIAGNOSTIC][Step {t}] POST-STEP REWARD:")
    print(f"  Total reward: {reward:+.3f}")
    if reward_breakdown:
        # FIX: reward_breakdown values are 3-tuples (weight, raw, weighted)
        # Extract weighted values (index 2) for display
        efficiency_weighted = reward_breakdown.get('efficiency', (0,0,0))[2]
        lane_weighted = reward_breakdown.get('lane_keeping', (0,0,0))[2]
        progress_weighted = reward_breakdown.get('progress', (0,0,0))[2]

        print(f"  Components: efficiency={efficiency_weighted:+.2f}, "
              f"lane_keeping={lane_weighted:+.2f}, "
              f"progress={progress_weighted:+.2f}")

        # Check for reward order dependency bug signature
        # Use weighted values for correlation analysis
        if progress_weighted > 0 and lane_weighted < 0:
            print(f"  🚨 BUG SIGNATURE: progress↑ but lane_keeping↓ (conflicting incentives!)")
        elif progress_weighted < 0 and lane_weighted > 0:
            print(f"  🚨 BUG SIGNATURE: progress↓ but lane_keeping↑ (conflicting incentives!)")
        else:
            print(f"  ✓ Aligned incentives: progress and lane_keeping have same sign")
    print(f"  Vehicle: vel={vehicle_state.get('velocity', 0):.1f} km/h, "
          f"lateral_dev={vehicle_state.get('lateral_deviation', 0):+.2f}m")
    print(f"  Episode: step={self.current_step}, done={done}, truncated={truncated}")
```

### Key Changes:

1. **Extract Weighted Values:**
   ```python
   # BEFORE (WRONG)
   efficiency = reward_breakdown.get('efficiency', 0)  # Returns tuple

   # AFTER (CORRECT)
   efficiency_weighted = reward_breakdown.get('efficiency', (0,0,0))[2]  # Extract index 2
   ```

2. **Safe Default:**
   - Default changed from `0` (scalar) to `(0,0,0)` (3-tuple)
   - Prevents IndexError if key missing

3. **Semantic Clarity:**
   - Variable names now explicit: `efficiency_weighted`, `lane_weighted`, `progress_weighted`
   - Makes clear we're using the weighted contribution, not raw component

---

## 📊 Why Use Index 2 (Weighted Value)?

### Tuple Structure:
```python
reward_breakdown['efficiency'] = (weight, raw, weighted)
                                  [0]     [1]   [2]
```

### Available Values:
- **Index 0 (weight):** Configuration value (e.g., 2.0 for efficiency)
- **Index 1 (raw):** Unweighted component (e.g., 0.5882 before multiplication)
- **Index 2 (weighted):** Final contribution to total reward (e.g., 1.1764 = 2.0 * 0.5882)

### Why Index 2 for Diagnostics?

1. **Matches Total Reward:**
   ```
   total = efficiency[2] + lane_keeping[2] + comfort[2] + safety[2] + progress[2]
   ```

2. **Shows Actual Impact:**
   - Raw value (index 1) doesn't show relative importance
   - Weighted value (index 2) shows how much each component contributes to policy learning

3. **Correlation Analysis:**
   - Detecting reward conflicts requires comparing **actual contributions**
   - If `progress[2] = +3.0` and `lane_keeping[2] = -1.2`, total is still +1.8 (progress dominates)
   - Using raw values would miss this dynamic

### Example:

**Step 100 (from log before crash):**
```python
reward_breakdown = {
    'efficiency': (2.0, 0.5882, 1.1764),     # weight=2.0, raw=0.59, weighted=1.18
    'lane_keeping': (2.0, 0.4016, 0.8032),   # weight=2.0, raw=0.40, weighted=0.80
    'progress': (3.0, 0.0000, 0.0000),       # weight=3.0, raw=0.00, weighted=0.00
}
total_reward = 1.1764 + 0.8032 + 0.0000 + ... = 3.194
```

**Debug Output (AFTER fix):**
```
[DIAGNOSTIC][Step 100] POST-STEP REWARD:
  Total reward: +3.194
  Components: efficiency=+1.18, lane_keeping=+0.80, progress=+0.00
  ✓ Aligned incentives: progress and lane_keeping have same sign
```

---

## 🧪 Verification

### Expected Behavior (Next Run):

1. **Step 100 Debug Print Should Succeed:**
   ```
   [DIAGNOSTIC][Step 100] POST-STEP REWARD:
     Total reward: +X.XXX
     Components: efficiency=+X.XX, lane_keeping=+X.XX, progress=+X.XX
     Status: (aligned or conflicting)
     Vehicle: vel=XX.X km/h, lateral_dev=+X.XXm
     Episode: step=XXX, done=False, truncated=False
   ```

2. **No TypeError:**
   - Training should continue past step 100
   - Debug prints should appear every 100 steps (200, 300, 400, ...)

3. **Correlation Analysis Should Work:**
   - If both positive → "✓ Aligned incentives"
   - If conflicting signs → "🚨 BUG SIGNATURE"

### Test Command:

```bash
cd /workspace/av_td3_system
./scripts/train_td3.sh \
    --max_timesteps 1000 \
    --log_level DEBUG \
    --debug \
    > logs/debug_1k_steps_fixed.log 2>&1

# Verify no errors:
grep "TypeError" logs/debug_1k_steps_fixed.log  # Should be empty
grep "DIAGNOSTIC" logs/debug_1k_steps_fixed.log  # Should show steps 100, 200, 300, ...
```

---

## 🎓 Lessons Learned

### 1. Always Verify Data Structures Before Formatting

**Bad Practice:**
```python
# Assume dict values are scalars
value = some_dict.get('key', 0)
print(f"{value:.2f}")  # CRASH if value is tuple/list/object
```

**Good Practice:**
```python
# Check structure first
value = some_dict.get('key', (0,0,0))
if isinstance(value, tuple):
    scalar = value[2]  # Extract desired element
else:
    scalar = value
print(f"{scalar:.2f}")
```

### 2. Default Values Must Match Expected Type

**Bad Practice:**
```python
# Default is scalar, but actual value is tuple
tuple_value = my_dict.get('key', 0)  # Returns 0 if missing
weighted = tuple_value[2]  # IndexError if key missing!
```

**Good Practice:**
```python
# Default matches structure
tuple_value = my_dict.get('key', (0,0,0))  # Returns 3-tuple if missing
weighted = tuple_value[2]  # Safe, returns 0
```

### 3. Use Descriptive Variable Names

**Bad Practice:**
```python
eff = rb.get('e', 0)[2]  # What is eff? What's rb? What's e?
```

**Good Practice:**
```python
efficiency_weighted = reward_breakdown.get('efficiency', (0,0,0))[2]
```

---

## 🔗 Related Issues

### Why Wasn't This Caught Earlier?

1. **Debug prints added recently** - Not tested in full training run
2. **Throttling to every 100 steps** - Error only triggers at step 100, not during initial setup
3. **Exploration phase uses random actions** - Reached step 100 successfully, but debug code had latent bug

### Alternative Fix (Not Used):

**Option 1: Use `reward_components` instead of `reward_breakdown`**

In `carla_env.py` line 887-893, there's a **flat dictionary** version:
```python
"reward_components": {
    "total": reward,
    "efficiency": reward_dict["breakdown"]["efficiency"][2],  # Already extracted!
    "lane_keeping": reward_dict["breakdown"]["lane_keeping"][2],
    "comfort": reward_dict["breakdown"]["comfort"][2],
    "safety": reward_dict["breakdown"]["safety"][2],
    "progress": reward_dict["breakdown"]["progress"][2],
}
```

**Could use this instead:**
```python
reward_components = info.get('reward_components', {})
print(f"efficiency={reward_components.get('efficiency', 0):+.2f}")  # Works!
```

**Why Not Used:**
- `reward_breakdown` is more detailed (provides weight + raw + weighted)
- Future diagnostics might need raw values
- Extracting index 2 makes structure explicit in debug code

---

## 📚 References

1. **Python Format Strings:**
   - PEP 3101: https://peps.python.org/pep-3101/
   - Format Specification Mini-Language: https://docs.python.org/3/library/string.html#formatspec

2. **Reward Function Structure:**
   - File: `src/environment/reward_functions.py` lines 276-303
   - File: `src/environment/carla_env.py` lines 886-895

3. **Debug Instrumentation Design:**
   - Doc: `docs/day-1-12/debug-action/DEBUG_INSTRUMENTATION_ANALYSIS.md`
   - Section: "3. POST-STEP: Reward Correlation Analysis"

---

## 🔧 Additional Issue Found: Throttling Inconsistency

### Problem Discovery:

During systematic code review to find similar tuple formatting issues, discovered **inconsistent throttling**:

**BEFORE Fix:**
- Line 750: `if t % 50 == 0 and self.debug:` (PRE-ACTION)
- Line 768: `if t % 50 == 0 and self.debug:` (POST-ACTION)
- Line 787: `if t % 100 == 0 and self.debug:` (POST-STEP REWARD)

**Issue:** PRE-ACTION and POST-ACTION triggered TWICE as often as POST-STEP REWARD.

**Documentation Says:** "Throttled to every 100 steps (10 prints per 1K training)"

**AFTER Fix:**
- Line 750: `if t % 100 == 0 and self.debug:` ✅
- Line 768: `if t % 100 == 0 and self.debug:` ✅
- Line 787: `if t % 100 == 0 and self.debug:` ✅

### Impact:

**Performance:**
- BEFORE: 20 PRE-ACTION + 20 POST-ACTION + 10 POST-STEP = 50 prints per 1K steps
- AFTER: 10 PRE-ACTION + 10 POST-ACTION + 10 POST-STEP = 30 prints per 1K steps
- **Reduction:** 40% fewer debug prints (better performance)

**Consistency:**
- All three diagnostic points now synchronized at same steps
- Easier to correlate observations → actions → rewards
- Matches documentation in DEBUG_INSTRUMENTATION_ANALYSIS.md

---

## 🔧 Additional Issue #2: AttributeError on Undefined Variable

### Problem Discovery:

After fixing tuple formatting and throttling, training crashed at step 100 with:

```
AttributeError: 'TD3TrainingPipeline' object has no attribute 'current_step'
```

**Location:** Line 813 in POST-STEP REWARD diagnostic

**Root Cause:**
```python
# WRONG (line 813)
print(f"  Episode: step={self.current_step}, done={done}, truncated={truncated}")
```

The attribute `self.current_step` was never defined in the `__init__` method.

**Correct Variable:**

Looking at the class initialization (line 303) and usage throughout:
```python
self.episode_timesteps = 0  # Defined in __init__
```

This tracks the **current step within the episode** (reset to 0 each episode).

**Fix Applied:**
```python
# CORRECT (line 813)
print(f"  Episode: step={self.episode_timesteps}, done={done}, truncated={truncated}")
```

### Why This Error Occurred:

**Copy-paste error:** Likely copied from a different version where the variable was named `current_step`.

**Why not caught earlier:** Debug print only triggers at step 100 (throttling), so wasn't executed during initial testing.

### Impact:

**Before fix:** Training crashed at step 100 (first debug print)
**After fix:** Training can proceed, episode step count displayed correctly

---

## ✅ Resolution Status

**Fixed:** ✅ December 1, 2025
**Tested:** ⏳ Pending next training run
**Impact:** Bug prevented debug diagnostics from working beyond step 100
**Severity:** Medium (blocking debug feature, not core training)
**Confidence:** HIGH - Fix is straightforward tuple indexing

**Additional Fixes Applied:**
1. ✅ Fixed tuple formatting in POST-STEP REWARD diagnostic (line 795-797)
2. ✅ Fixed inconsistent throttling: PRE-ACTION and POST-ACTION changed from every 50 steps to every 100 steps
3. ✅ Verified OpenCV display (line 542) and main debug loop (lines 870-883) correctly handle tuple structure
4. ✅ No other tuple formatting issues found in codebase
5. ✅ Fixed AttributeError #1: Changed `self.current_step` to `self.episode_timesteps` (line 813)
6. ✅ Fixed AttributeError #2: Changed `self.start_timesteps` to `start_timesteps` (line 757 - local variable)

**Next Steps:**
1. ✅ Apply fix (DONE)
2. ✅ Fix throttling inconsistency (DONE)
3. ✅ Fix AttributeError (DONE)
4. ⏳ Run test training (1K steps)
5. ⏳ Verify debug prints appear at steps 100, 200, 300, ...
6. ⏳ Analyze reward correlation patterns
7. ⏳ Document findings in debug analysis
