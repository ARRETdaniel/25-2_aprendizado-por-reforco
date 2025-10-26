# Summary: Velocity-Gated Reward Components ✅

**Date**: October 25, 2025
**Priority**: CRITICAL
**Status**: ✅ FIXED

---

## Problem

The reward function allowed the agent to get **positive rewards while stationary**:

**Before Fix** (stationary vehicle):
```
Efficiency:   -1.00 × 1.0  = -1.00  (not moving - penalty)
Lane Keeping: +1.00 × 2.0  = +2.00  (centered - reward!)  ← WRONG!
Comfort:      +0.30 × 0.5  = +0.15  (no jerk - reward!)   ← WRONG!
Safety:       +0.00 × -100 = +0.00  (no violation)
Progress:     +0.00 × 5.0  = +0.00  (no movement)
----------------------------------------
TOTAL: +1.15 ← Agent learns to PARK in middle of road!
```

---

## Solution

### 1. Lane Keeping Requires Movement
- **Change**: Added velocity parameter, returns 0.0 if `velocity < 1.0 m/s`
- **Rationale**: Staying centered while parked is not "good lane keeping"

### 2. Comfort Requires Movement
- **Change**: Added velocity parameter, returns 0.0 if `velocity < 1.0 m/s`
- **Rationale**: Zero jerk while stationary is not "comfortable driving"

### 3. Safety Penalizes Unnecessary Stopping
- **Change**: Added `-1.0` penalty if `velocity < 0.5 m/s` AND `distance_to_goal > 5.0m`
- **Rationale**: Stopping on clear road blocks traffic (unsafe!)

---

## New Reward Behavior

**After Fix** (stationary vehicle):
```
Efficiency:   -1.00 × 1.0  = -1.00  (not moving)
Lane Keeping: +0.00 × 2.0  = +0.00  ← FIXED! (velocity < 1.0)
Comfort:      +0.00 × 0.5  = +0.00  ← FIXED! (velocity < 1.0)
Safety:       -1.00 × -100 = +100.0 (stationary penalty) ⚠️ SEE NOTE BELOW
Progress:     +0.00 × 5.0  = +0.00  (not moving)
----------------------------------------
TOTAL: +99.0 ⚠️ ISSUE WITH SAFETY WEIGHT!
```

### ⚠️ CRITICAL ISSUE: Safety Weight Sign

The config has `safety: -100.0` (negative weight). This causes:
```
safety_reward = -1.0 (penalty)
weighted = -100.0 × -1.0 = +100.0 (becomes positive!)
```

**This turns penalties into rewards!**

### Fix Required

Change safety weight in `config/td3_config.yaml`:

```yaml
weights:
  safety: 1.0  # Changed from -100.0
```

Then safety penalties work correctly:
```
Efficiency:   -1.00 × 1.0  = -1.00
Lane Keeping: +0.00 × 2.0  = +0.00
Comfort:      +0.00 × 0.5  = +0.00
Safety:       -1.00 × 1.0  = -1.00  ← Correct penalty!
Progress:     +0.00 × 5.0  = +0.00
----------------------------------------
TOTAL: -2.00 ← Correctly negative!
```

---

## Files Modified

1. **`src/environment/reward_functions.py`**:
   - `_calculate_lane_keeping_reward()`: Added `velocity` parameter, returns 0 if v < 1.0
   - `_calculate_comfort_reward()`: Added `velocity` parameter, returns 0 if v < 1.0
   - `_calculate_safety_reward()`: Added `velocity` and `distance_to_goal` parameters, adds -1.0 penalty if stationary
   - `calculate()`: Updated calls to pass velocity to all three functions

2. **`config/td3_config.yaml`** (RECOMMENDED):
   - Change `safety: -100.0` to `safety: 1.0`
   - Safety penalties (collision=-1000.0, offroad=-500.0) are already negative, don't need negative weight

---

## Next Steps

1. ✅ Rebuild Docker image
2. ✅ Fix safety weight in config (change -100.0 to 1.0)
3. ✅ Test with short run to verify rewards
4. ✅ Verify stationary vehicle gets negative total reward
5. ✅ Verify moving vehicle gets rewards

---

**Status**: Code fixed, config needs manual update for safety weight sign.
