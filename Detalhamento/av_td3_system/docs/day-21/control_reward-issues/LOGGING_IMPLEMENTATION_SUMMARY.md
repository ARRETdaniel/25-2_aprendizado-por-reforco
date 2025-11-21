# Logging Implementation Summary
## Adding Action Statistics and Raw Reward Component Logging

**Date**: 2025-01-XX  
**Status**: 🚧 **IMPLEMENTATION IN PROGRESS**  
**Priority**: **HIGH** - Required before 1M training run  
**Estimated Time**: 1 hour

---

## Summary

Based on documentation validation analysis (DOCUMENTATION_VALIDATION_ANALYSIS.md), we identified two critical logging gaps:

1. ❌ **No action statistics logging** (steering/throttle mean/std/min/max)
2. ✅ **Raw reward component logging already exists** (in DEBUG mode since Day 4 fixes)

**Finding**: Reward component logging is already comprehensive in `reward_functions.py` lines 283-340. No changes needed there!

**New Task**: Only need to implement action statistics tracking and TensorBoard logging.

---

## Implementation Plan

### Task 1: Add Action Tracking to TD3Agent ✅ TO DO

**File**: `src/agents/td3_agent.py`

**Location**: After initialization (`__init__` method, around line 300)

**Code to Add**:

```python
# Action statistics tracking (for monitoring control commands)
self.action_buffer = []  # Rolling buffer of recent actions
self.action_buffer_size = 100  # Track last 100 actions
print(f"  Action statistics tracking enabled (buffer size: {self.action_buffer_size})")
```

---

### Task 2: Track Actions in select_action() ✅ TO DO

**File**: `src/agents/td3_agent.py`

**Location**: End of `select_action()` method (before `return action`, around line 372)

**Code to Add**:

```python
# Track action for statistics (BEFORE returning)
self.action_buffer.append(action.copy())
if len(self.action_buffer) > self.action_buffer_size:
    self.action_buffer.pop(0)  # Remove oldest action

return action
```

---

### Task 3: Add get_action_stats() Method ✅ TO DO

**File**: `src/agents/td3_agent.py`

**Location**: After `get_stats()` method (around line 1350)

**Code to Add**:

```python
def get_action_stats(self) -> Dict[str, float]:
    """
    Get statistics of recent actions for monitoring control commands.
    
    Returns dictionary with steering and throttle statistics (mean/std/min/max).
    Useful for validating exploration noise, detecting biases, and debugging
    control command generation.
    
    Returns:
        Dictionary with:
        - action_steering_mean: Mean steering command [-1, 1]
        - action_steering_std: Std dev of steering (exploration indicator)
        - action_steering_min: Minimum steering (should be near -1)
        - action_steering_max: Maximum steering (should be near +1)
        - action_throttle_mean: Mean throttle/brake command
        - action_throttle_std: Std dev of throttle
        - action_throttle_min: Minimum throttle (brake)
        - action_throttle_max: Maximum throttle (acceleration)
        
    Example usage:
        ```python
        stats = agent.get_action_stats()
        print(f"Steering: {stats['action_steering_mean']:.3f} ± {stats['action_steering_std']:.3f}")
        print(f"Throttle: {stats['action_throttle_mean']:.3f} ± {stats['action_throttle_std']:.3f}")
        ```
    
    Expected values during training:
        - Steering mean: ~0.0 (no left/right bias)
        - Steering std: 0.1-0.3 (exploration present)
        - Throttle mean: 0.0-0.5 (forward bias expected)
        - Throttle std: 0.1-0.3 (exploration present)
    """
    if len(self.action_buffer) == 0:
        return {
            'action_steering_mean': 0.0,
            'action_steering_std': 0.0,
            'action_steering_min': 0.0,
            'action_steering_max': 0.0,
            'action_throttle_mean': 0.0,
            'action_throttle_std': 0.0,
            'action_throttle_min': 0.0,
            'action_throttle_max': 0.0,
        }
    
    actions = np.array(self.action_buffer)  # Shape: (N, 2)
    
    return {
        # Steering statistics (action[:, 0])
        'action_steering_mean': float(actions[:, 0].mean()),
        'action_steering_std': float(actions[:, 0].std()),
        'action_steering_min': float(actions[:, 0].min()),
        'action_steering_max': float(actions[:, 0].max()),
        
        # Throttle/brake statistics (action[:, 1])
        'action_throttle_mean': float(actions[:, 1].mean()),
        'action_throttle_std': float(actions[:, 1].std()),
        'action_throttle_min': float(actions[:, 1].min()),
        'action_throttle_max': float(actions[:, 1].max()),
    }
```

---

### Task 4: Add TensorBoard Logging to Training Script ✅ TO DO

**File**: `scripts/train_td3.py`

**Location**: In training loop where we log other metrics (around line 820, after episode stats logging)

**Code to Add**:

```python
# Log action statistics every 100 steps
if t % 100 == 0 and t > 0:
    action_stats = self.agent.get_action_stats()
    for key, value in action_stats.items():
        self.writer.add_scalar(f'debug/{key}', value, t)
    
    # DEBUG: Print action stats to console (optional, can comment out)
    if self.logger.isEnabledFor(logging.DEBUG) and t % 1000 == 0:
        self.logger.debug(
            f"\n   ACTION STATISTICS (last 100 actions):\n"
            f"   ══════════════════════════════════════\n"
            f"   STEERING:\n"
            f"      Mean: {action_stats['action_steering_mean']:+.3f}  (expect ~0.0, no bias)\n"
            f"      Std:  {action_stats['action_steering_std']:.3f}   (expect 0.1-0.3, exploration)\n"
            f"      Range: [{action_stats['action_steering_min']:+.3f}, {action_stats['action_steering_max']:+.3f}]\n"
            f"   ──────────────────────────────────────\n"
            f"   THROTTLE/BRAKE:\n"
            f"      Mean: {action_stats['action_throttle_mean']:+.3f}  (expect 0.0-0.5, forward bias)\n"
            f"      Std:  {action_stats['action_throttle_std']:.3f}   (expect 0.1-0.3, exploration)\n"
            f"      Range: [{action_stats['action_throttle_min']:+.3f}, {action_stats['action_throttle_max']:+.3f}]\n"
            f"   ══════════════════════════════════════"
        )
```

---

## Verification Checklist

After implementation, verify:

### Code Integration ✅
- [ ] TD3Agent has `self.action_buffer` initialized in `__init__()`
- [ ] `select_action()` appends actions to buffer
- [ ] `get_action_stats()` method exists and returns correct format
- [ ] Training script imports and calls `get_action_stats()` every 100 steps
- [ ] TensorBoard logging configured for all 8 action statistics

### Functionality ✅
- [ ] Run short test (10 steps) and check `agent.get_action_stats()` returns non-zero values
- [ ] Check TensorBoard for new metrics under `debug/action_*`
- [ ] Verify statistics are reasonable (steering mean near 0, std > 0)
- [ ] No performance degradation (action tracking is O(1) append)

### Expected TensorBoard Metrics (8 new scalars)
```
debug/action_steering_mean   → Should oscillate around 0.0 (no bias)
debug/action_steering_std    → Should be 0.1-0.3 (exploration noise working)
debug/action_steering_min    → Should approach -1.0 (full left)
debug/action_steering_max    → Should approach +1.0 (full right)
debug/action_throttle_mean   → Should be 0.0-0.5 (forward driving bias expected)
debug/action_throttle_std    → Should be 0.1-0.3 (exploration noise)
debug/action_throttle_min    → Should approach -1.0 (full brake)
debug/action_throttle_max    → Should approach +1.0 (full throttle)
```

---

## Diagnostic Run Plan

**After implementation, run 500-step diagnostic:**

```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

python scripts/train_td3.py \
  --config config/td3_carla_town01.yaml \
  --max-steps 500 \
  --log-level DEBUG \
  --run-name "diagnostic_action_logging" \
  --tensorboard-dir logs/diagnostic_action_logging
```

**What to Check**:

1. **Console Output**:
   - Action statistics printed every 1000 steps
   - No errors or warnings about action tracking

2. **TensorBoard** (`tensorboard --logdir logs/diagnostic_action_logging`):
   - All 8 action statistics visible under `debug/` group
   - Steering mean near 0.0 (no left/right bias)
   - Steering std > 0 (exploration present)
   - Throttle mean positive (forward driving)
   - Throttle std > 0 (exploration present)

3. **Expected Patterns**:
   - First 1000 steps (random exploration): 
     - High variance, uniform distribution
     - Steering mean ~0.0, std ~0.58 (uniform random)
   - After 1000 steps (policy + noise):
     - Lower variance, policy-driven
     - Steering std ~0.1-0.3 (Gaussian noise)

---

## Issues and Solutions

### Issue 1: Buffer Too Large (Memory Concern)
**Symptom**: Slow training due to large action buffer  
**Solution**: Buffer size is only 100 actions × 2 floats = 1.6KB, negligible

### Issue 2: Statistics Not Updating
**Symptom**: TensorBoard shows flat lines  
**Solution**: Check `select_action()` is actually being called (print statement)

### Issue 3: Steering Mean Not Near 0
**Symptom**: Steering mean is +0.5 or -0.5 consistently  
**Solution**: Indicates action bias! This is exactly what we want to detect.
           - Check if CNN is outputting biased features
           - Check if actor network has bias toward one side
           - Investigate reward function (is lane keeping penalizing turning?)

---

## Next Steps After Logging Implementation

1. ✅ Implement code changes (4 tasks above)
2. ✅ Run diagnostic 500-step test
3. ✅ Analyze TensorBoard action statistics
4. ✅ Validate steering mean near 0, std 0.1-0.3
5. 🔄 If action distribution looks good → Proceed to reward normalization (Phase 4)
6. 🔄 If action distribution shows bias → Investigate root cause before normalization

---

## Related Documents

- **DOCUMENTATION_VALIDATION_ANALYSIS.md**: Full documentation comparison (this task derived from Section 7.2)
- **CONTROL_COMMAND_VALIDATION_ANALYSIS.md**: Initial investigation that identified need for action logging
- **ACTION_PLAN_REWARD_IMBALANCE_FIX.md**: Overall plan (this is Phase 3)

---

**Status**: 🚧 **READY TO IMPLEMENT**  
**Next Action**: Implement Task 1 (add action_buffer to TD3Agent)

