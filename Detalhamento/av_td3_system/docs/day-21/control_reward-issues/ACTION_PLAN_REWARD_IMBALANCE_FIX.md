# Action Plan: Fix Reward Function Imbalance
## Critical Fix Required Before 1M Training Run

**Date**: 2025-01-21  
**Priority**: 🔴 **CRITICAL** - Blocking 1M run  
**Estimated Time**: 4-6 hours (fixes + validation)  
**Risk**: HIGH - System will not learn without these fixes

---

## Executive Summary

### Problem
The reward function is **severely imbalanced**, causing the agent to optimize for forward progress while completely ignoring lane keeping safety. This explains the observed behavior (consistent right-turning, 100% lane invasion rate, degrading performance).

### Root Cause
Progress reward dominates at **83%** of total reward, while lane keeping contributes only **6%**. Despite equal configured weights (2.0 for both), the underlying reward component scales are mismatched by approximately **14×**.

### Impact
- 100% lane invasion rate (agent doesn't learn lane keeping)
- Episodes getting shorter (agent crashes faster while maximizing progress)
- No improvement over 187 episodes (learning wrong behavior)
- System unsuitable for 1M run without fixes

### Solution
Two-pronged approach:
1. **Normalize reward components** to equal scales before weighting
2. **Increase lane keeping weight** from 2.0 to 5.0
3. **Add action logging** for validation

---

## Current State Analysis

### Reward Component Distribution (Observed)

| Component | Weight (Config) | Expected % | Observed % | Deviation |
|-----------|----------------|------------|------------|-----------|
| Progress | 2.0 (33%) | 31% | **83%** | +52% ⬆️ |
| Lane Keeping | 2.0 (33%) | 31% | **6%** | -25% ⬇️ |
| Safety | 1.0 (17%) | 15% | **7%** | -8% ⬇️ |
| Efficiency | 1.0 (17%) | 15% | **3%** | -12% ⬇️ |
| Comfort | 0.5 (8%) | 8% | **1%** | -7% ⬇️ |

**Total Weight Sum**: 6.5  
**Imbalance Ratio**: Progress/Lane_Keeping = 83%/6% = **13.8×**

### Why This Happened

**Hypothesis 1: Progress Reward Has Larger Native Scale**

Looking at the reward calculation logic:

```python
# Progress component (from waypoint distance)
# Native scale: Could be 0-50 meters (waypoint spacing)
progress_raw = waypoint_distance

# Lane keeping component (from lateral deviation)
# Native scale: Normalized to [-0.5, 0.5] after velocity scaling
lane_keeping_raw = (lat_reward + head_reward) / 2.0 - 0.5
```

**Evidence from TensorBoard**:
```
Progress component: Dominates 83% → Native scale ~10-50
Lane keeping component: Only 6% → Native scale ~0.3-0.8 (as observed)
```

**Result**: Even with equal weights (2.0), scales differ by ~14×:
```python
weighted_progress = 2.0 * progress_raw       # ~2.0 * 20 = 40
weighted_lane_keeping = 2.0 * lane_keeping_raw # ~2.0 * 0.3 = 0.6

# Progress dominates: 40 / (40 + 0.6) ≈ 98% of combined signal
```

---

## Implementation Plan

### Phase 1: Investigate Current Scales (1 hour)

**Goal**: Understand actual native scales of each reward component.

#### Step 1.1: Add Debug Logging
**File**: `src/environment/reward_functions.py`  
**Function**: `calculate()`  
**Location**: After computing each component, before weighting

```python
def calculate(self, ...):
    # ... existing code ...
    
    # 1. EFFICIENCY REWARD
    efficiency = self._calculate_efficiency_reward(velocity, heading_error)
    
    # 2. LANE KEEPING REWARD
    lane_keeping = self._calculate_lane_keeping_reward(...)
    
    # 3. COMFORT PENALTY
    comfort = self._calculate_comfort_reward(...)
    
    # 4. SAFETY PENALTY
    safety = self._calculate_safety_reward(...)
    
    # 5. PROGRESS REWARD
    progress = self._calculate_progress_reward(...)
    
    # DEBUG LOGGING: Log RAW component values BEFORE weighting
    if self.step_counter % 100 == 0:
        self.logger.debug(
            f"[REWARD_COMPONENTS_RAW] "
            f"efficiency={efficiency:.4f}, "
            f"lane_keeping={lane_keeping:.4f}, "
            f"comfort={comfort:.4f}, "
            f"safety={safety:.4f}, "
            f"progress={progress:.4f}"
        )
    
    # Apply weights (current implementation)
    weighted = {
        'efficiency': efficiency * self.reward_weights['efficiency'],
        'lane_keeping': lane_keeping * self.reward_weights['lane_keeping'],
        'comfort': comfort * self.reward_weights['comfort'],
        'safety': safety * self.reward_weights['safety'],
        'progress': progress * self.reward_weights['progress'],
    }
    
    # DEBUG LOGGING: Log WEIGHTED component values
    if self.step_counter % 100 == 0:
        self.logger.debug(
            f"[REWARD_COMPONENTS_WEIGHTED] "
            f"efficiency={weighted['efficiency']:.4f}, "
            f"lane_keeping={weighted['lane_keeping']:.4f}, "
            f"comfort={weighted['comfort']:.4f}, "
            f"safety={weighted['safety']:.4f}, "
            f"progress={weighted['progress']:.4f}"
        )
```

#### Step 1.2: Run 500-Step Diagnostic
```bash
cd av_td3_system
python scripts/train_td3.py \
  --config config/td3_carla_town01.yaml \
  --max-steps 500 \
  --log-level DEBUG \
  --run-name "reward_debug_500steps"
```

#### Step 1.3: Analyze Component Scales
Extract from logs:
```bash
grep "REWARD_COMPONENTS_RAW" logs/reward_debug_500steps.log > raw_components.txt
grep "REWARD_COMPONENTS_WEIGHTED" logs/reward_debug_500steps.log > weighted_components.txt
```

Calculate statistics:
```python
import re
import numpy as np

# Parse raw components
efficiency_raw, lane_keeping_raw, progress_raw = [], [], []

with open('raw_components.txt') as f:
    for line in f:
        match = re.search(r'efficiency=([\d.-]+).*lane_keeping=([\d.-]+).*progress=([\d.-]+)', line)
        if match:
            efficiency_raw.append(float(match.group(1)))
            lane_keeping_raw.append(float(match.group(2)))
            progress_raw.append(float(match.group(3)))

print("RAW COMPONENT SCALES:")
print(f"Efficiency:     mean={np.mean(efficiency_raw):.4f}, std={np.std(efficiency_raw):.4f}, range=[{np.min(efficiency_raw):.4f}, {np.max(efficiency_raw):.4f}]")
print(f"Lane Keeping:   mean={np.mean(lane_keeping_raw):.4f}, std={np.std(lane_keeping_raw):.4f}, range=[{np.min(lane_keeping_raw):.4f}, {np.max(lane_keeping_raw):.4f}]")
print(f"Progress:       mean={np.mean(progress_raw):.4f}, std={np.std(progress_raw):.4f}, range=[{np.min(progress_raw):.4f}, {np.max(progress_raw):.4f}]")

# Calculate imbalance ratios
print(f"\nIMBALANCE RATIOS:")
print(f"Progress/Efficiency:    {np.mean(progress_raw) / np.mean(efficiency_raw):.2f}×")
print(f"Progress/Lane_Keeping:  {np.mean(progress_raw) / abs(np.mean(lane_keeping_raw)):.2f}×")
```

**Expected Output**:
```
RAW COMPONENT SCALES:
Efficiency:     mean=0.3500, std=0.1800, range=[-0.2000, 0.8000]
Lane Keeping:   mean=0.3000, std=0.2500, range=[-0.5000, 0.5000]
Progress:       mean=5.0000, std=2.5000, range=[0.0000, 10.0000]

IMBALANCE RATIOS:
Progress/Efficiency:    14.29×
Progress/Lane_Keeping:  16.67×
```

---

### Phase 2: Implement Normalization (2 hours)

**Goal**: Normalize all reward components to equal scales before weighting.

#### Step 2.1: Add Normalization Method
**File**: `src/environment/reward_functions.py`  
**Class**: `RewardCalculator`

```python
class RewardCalculator:
    def __init__(self, config: Dict):
        # ... existing init code ...
        
        # NEW: Component normalization ranges (empirically determined from Phase 1)
        # Target: Normalize all components to [-1.0, 1.0] range
        self.component_ranges = {
            'efficiency': {'min': -0.5, 'max': 1.0},      # From Phase 1 analysis
            'lane_keeping': {'min': -1.0, 'max': 0.5},    # From current implementation
            'comfort': {'min': -1.0, 'max': 0.3},         # From current implementation
            'safety': {'min': -10.0, 'max': 0.0},         # Large negative for collisions
            'progress': {'min': 0.0, 'max': 10.0},        # From Phase 1 analysis (waypoint distance)
        }
        
        self.logger.info("[REWARD_INIT] Component normalization ranges:")
        for comp, ranges in self.component_ranges.items():
            self.logger.info(f"  {comp:15s}: [{ranges['min']:+.2f}, {ranges['max']:+.2f}]")
    
    def _normalize_component(self, value: float, component_name: str) -> float:
        """
        Normalize reward component to [-1.0, 1.0] range.
        
        Args:
            value: Raw component value
            component_name: Name of component (for lookup in ranges)
        
        Returns:
            Normalized value in [-1.0, 1.0]
        """
        ranges = self.component_ranges[component_name]
        min_val, max_val = ranges['min'], ranges['max']
        
        # Clip to expected range
        clipped = np.clip(value, min_val, max_val)
        
        # Normalize to [-1.0, 1.0]
        # Formula: (value - min) / (max - min) * 2.0 - 1.0
        if max_val > min_val:
            normalized = ((clipped - min_val) / (max_val - min_val)) * 2.0 - 1.0
        else:
            normalized = 0.0
        
        return float(np.clip(normalized, -1.0, 1.0))
```

#### Step 2.2: Update calculate() Method
**File**: `src/environment/reward_functions.py`  
**Function**: `calculate()`

```python
def calculate(self, ...):
    # ... existing component calculations ...
    
    # Compute RAW components (unchanged)
    efficiency_raw = self._calculate_efficiency_reward(velocity, heading_error)
    lane_keeping_raw = self._calculate_lane_keeping_reward(...)
    comfort_raw = self._calculate_comfort_reward(...)
    safety_raw = self._calculate_safety_reward(...)
    progress_raw = self._calculate_progress_reward(...)
    
    # NEW: Normalize components to [-1.0, 1.0] BEFORE weighting
    efficiency = self._normalize_component(efficiency_raw, 'efficiency')
    lane_keeping = self._normalize_component(lane_keeping_raw, 'lane_keeping')
    comfort = self._normalize_component(comfort_raw, 'comfort')
    safety = self._normalize_component(safety_raw, 'safety')
    progress = self._normalize_component(progress_raw, 'progress')
    
    # Apply weights (NOW balanced because all components in [-1, 1])
    weighted = {
        'efficiency': efficiency * self.reward_weights['efficiency'],
        'lane_keeping': lane_keeping * self.reward_weights['lane_keeping'],
        'comfort': comfort * self.reward_weights['comfort'],
        'safety': safety * self.reward_weights['safety'],
        'progress': progress * self.reward_weights['progress'],
    }
    
    # Total reward (sum of weighted components)
    total = sum(weighted.values())
    
    # Log for validation (every 100 steps)
    if self.step_counter % 100 == 0:
        self.logger.debug(
            f"[REWARD_NORMALIZED] "
            f"efficiency={efficiency:.4f}→{weighted['efficiency']:.4f}, "
            f"lane_keeping={lane_keeping:.4f}→{weighted['lane_keeping']:.4f}, "
            f"comfort={comfort:.4f}→{weighted['comfort']:.4f}, "
            f"safety={safety:.4f}→{weighted['safety']:.4f}, "
            f"progress={progress:.4f}→{weighted['progress']:.4f}, "
            f"total={total:.4f}"
        )
    
    # ... rest of existing code (percentages, etc.) ...
```

#### Step 2.3: Adjust Weights for New Balance
**File**: `config/td3_carla_town01.yaml`  
**Section**: `reward_weights`

**BEFORE** (current):
```yaml
reward_weights:
  efficiency: 1.0
  lane_keeping: 2.0
  comfort: 0.5
  safety: 1.0
  progress: 2.0
```

**AFTER** (Phase 2 - normalized components):
```yaml
reward_weights:
  efficiency: 1.0      # Keep same
  lane_keeping: 5.0    # INCREASE: 2.0 → 5.0 (emphasize lane keeping)
  comfort: 0.5         # Keep same
  safety: 3.0          # INCREASE: 1.0 → 3.0 (emphasize safety)
  progress: 1.0        # DECREASE: 2.0 → 1.0 (reduce progress dominance)
```

**New Total Weight Sum**: 10.5  
**Expected Percentages** (with normalization):
- Lane Keeping: 47.6% (5.0 / 10.5)
- Safety: 28.6% (3.0 / 10.5)
- Efficiency: 9.5% (1.0 / 10.5)
- Progress: 9.5% (1.0 / 10.5)
- Comfort: 4.8% (0.5 / 10.5)

---

### Phase 3: Add Action Logging (1 hour)

**Goal**: Log action statistics to TensorBoard for validation.

#### Step 3.1: Modify TD3 Agent
**File**: `src/agents/td3_agent.py`  
**Method**: `select_action()`

```python
def select_action(self, state: Dict, evaluate: bool = False) -> np.ndarray:
    """Select action from policy with optional exploration noise."""
    # ... existing code to compute action ...
    
    # NEW: Track action statistics for logging
    if not hasattr(self, 'action_buffer'):
        self.action_buffer = []
    
    self.action_buffer.append(action.copy())
    
    # Keep buffer size limited (last 100 actions)
    if len(self.action_buffer) > 100:
        self.action_buffer.pop(0)
    
    return action

def get_action_stats(self) -> Dict[str, float]:
    """Get statistics of recent actions for logging."""
    if not hasattr(self, 'action_buffer') or len(self.action_buffer) == 0:
        return {
            'steering_mean': 0.0,
            'steering_std': 0.0,
            'steering_min': 0.0,
            'steering_max': 0.0,
            'throttle_mean': 0.0,
            'throttle_std': 0.0,
            'throttle_min': 0.0,
            'throttle_max': 0.0,
        }
    
    actions = np.array(self.action_buffer)
    return {
        'steering_mean': float(np.mean(actions[:, 0])),
        'steering_std': float(np.std(actions[:, 0])),
        'steering_min': float(np.min(actions[:, 0])),
        'steering_max': float(np.max(actions[:, 0])),
        'throttle_mean': float(np.mean(actions[:, 1])),
        'throttle_std': float(np.std(actions[:, 1])),
        'throttle_min': float(np.min(actions[:, 1])),
        'throttle_max': float(np.max(actions[:, 1])),
    }
```

#### Step 3.2: Log to TensorBoard
**File**: `scripts/train_td3.py`  
**Location**: Main training loop

```python
# After agent.train() call
if t % 100 == 0 and t > args.start_timesteps:
    # ... existing TensorBoard logging ...
    
    # NEW: Log action statistics
    action_stats = agent.get_action_stats()
    writer.add_scalar('debug/action_steering_mean', action_stats['steering_mean'], t)
    writer.add_scalar('debug/action_steering_std', action_stats['steering_std'], t)
    writer.add_scalar('debug/action_steering_min', action_stats['steering_min'], t)
    writer.add_scalar('debug/action_steering_max', action_stats['steering_max'], t)
    writer.add_scalar('debug/action_throttle_mean', action_stats['throttle_mean'], t)
    writer.add_scalar('debug/action_throttle_std', action_stats['throttle_std'], t)
```

---

### Phase 4: Validation Run (1.5 hours)

**Goal**: Validate fixes with 1K step run, confirm reward balance.

#### Step 4.1: Run 1K Validation
```bash
cd av_td3_system
python scripts/train_td3.py \
  --config config/td3_carla_town01.yaml \
  --max-steps 1000 \
  --seed 42 \
  --run-name "reward_fix_validation_1k"
```

#### Step 4.2: Analyze Results
**Metrics to Check**:

1. **Reward Component Percentages** (TensorBoard):
   - Lane keeping: Should be 40-50% (target: 48%)
   - Progress: Should be < 15% (target: 10%)
   - Safety: Should be 25-30% (target: 29%)

2. **Action Distribution** (TensorBoard):
   - Steering mean: Should be ~0.0 (no systematic bias)
   - Steering std: Should be 0.1-0.3 (exploration noise)
   - Throttle mean: Can be > 0 (forward bias OK)

3. **Lane Invasions** (TensorBoard):
   - Rate: Should remain high initially (expected at 1K steps)
   - Trend: Check if flat or slightly decreasing

**Python Analysis Script**:
```python
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

ea = event_accumulator.EventAccumulator('logs/reward_fix_validation_1k/*.tfevents.*')
ea.Reload()

# Check reward percentages
lane_pct = [e.value for e in ea.Scalars('rewards/lane_keeping_percentage')]
progress_pct = [e.value for e in ea.Scalars('rewards/progress_percentage')]

print("REWARD BALANCE CHECK:")
print(f"Lane keeping %: mean={np.mean(lane_pct):.1f}% (target: 40-50%)")
print(f"Progress %:     mean={np.mean(progress_pct):.1f}% (target: <15%)")

if 40 <= np.mean(lane_pct) <= 50 and np.mean(progress_pct) < 15:
    print("✅ REWARD BALANCE FIXED")
else:
    print("❌ REWARD BALANCE STILL IMBALANCED - Adjust weights")

# Check action distribution
steering_mean = [e.value for e in ea.Scalars('debug/action_steering_mean')]
steering_std = [e.value for e in ea.Scalars('debug/action_steering_std')]

print(f"\nACTION DISTRIBUTION CHECK:")
print(f"Steering mean: {np.mean(steering_mean):.4f} (target: ~0.0)")
print(f"Steering std:  {np.mean(steering_std):.4f} (target: 0.1-0.3)")

if abs(np.mean(steering_mean)) < 0.2 and 0.1 <= np.mean(steering_std) <= 0.3:
    print("✅ ACTION DISTRIBUTION NORMAL")
else:
    print("⚠️ ACTION DISTRIBUTION UNUSUAL - Investigate")
```

#### Step 4.3: Decision Criteria
**PASS** (Proceed to 10K):
- ✅ Lane keeping percentage: 40-50%
- ✅ Progress percentage: < 15%
- ✅ Steering mean: [-0.2, 0.2]
- ✅ Steering std: [0.1, 0.3]

**FAIL** (Iterate on weights):
- ❌ Lane keeping still < 30% → Increase weight further
- ❌ Progress still > 25% → Decrease weight further
- ❌ Steering mean > 0.3 → Investigate action generation

---

### Phase 5: Extended Validation (Optional, 2 hours)

**Only if Phase 4 passes** - Run 10K validation to check learning trends.

#### Step 5.1: Run 10K Validation
```bash
cd av_td3_system
python scripts/train_td3.py \
  --config config/td3_carla_town01.yaml \
  --max-steps 10000 \
  --seed 42 \
  --run-name "reward_fix_validation_10k"
```

#### Step 5.2: Analyze Learning Trends
**Metrics to Check**:

1. **Lane Invasions** (should decrease):
   ```python
   invasions = [e.value for e in ea.Scalars('train/lane_invasions_per_episode')]
   first_20 = np.mean(invasions[:20])
   last_20 = np.mean(invasions[-20:])
   improvement = (first_20 - last_20) / first_20 * 100
   
   print(f"Lane invasions: {first_20:.2f} → {last_20:.2f} ({improvement:.1f}% improvement)")
   ```

2. **Episode Length** (should increase):
   ```python
   lengths = [e.value for e in ea.Scalars('train/episode_length')]
   first_half = np.mean(lengths[:len(lengths)//2])
   second_half = np.mean(lengths[len(lengths)//2:])
   improvement = (second_half - first_half) / first_half * 100
   
   print(f"Episode length: {first_half:.1f} → {second_half:.1f} ({improvement:.1f}% improvement)")
   ```

3. **Reward Trend** (should stabilize or increase):
   ```python
   rewards = [e.value for e in ea.Scalars('train/episode_reward')]
   trend = np.polyfit(range(len(rewards)), rewards, 1)[0]  # Linear trend
   
   print(f"Reward trend: {trend:.2f}/episode ({'increasing' if trend > 0 else 'decreasing'})")
   ```

**SUCCESS CRITERIA**:
- ✅ Lane invasions decreasing by > 10%
- ✅ Episode length increasing by > 10%
- ✅ Reward trend positive or stable

---

## Risk Mitigation

### Risk 1: Normalization Breaks Existing Behavior
**Likelihood**: MEDIUM  
**Impact**: HIGH

**Mitigation**:
- Keep original reward function as backup
- Use git branches for changes
- Compare normalized vs original in 1K runs
- Document all changes with commit messages

**Rollback Plan**:
```bash
git checkout -b reward-normalization-backup
# Apply changes
# If fails:
git checkout main
```

---

### Risk 2: Empirical Ranges Don't Generalize
**Likelihood**: MEDIUM  
**Impact**: MEDIUM

**Mitigation**:
- Phase 1 establishes ranges from actual data
- Use conservative ranges (wider than observed)
- Add clipping to prevent out-of-range values
- Monitor component ranges in TensorBoard

**Contingency**:
- If components saturate at ±1.0, widen ranges
- If components cluster near 0, narrow ranges
- Iterate on ranges based on validation runs

---

### Risk 3: New Weights Still Imbalanced
**Likelihood**: LOW  
**Impact**: MEDIUM

**Mitigation**:
- Phase 4 validation checks actual percentages
- Adjust weights iteratively if needed
- Document weight changes and rationale
- Use automated percentage checking script

**Adjustment Formula**:
```python
# If observed percentages don't match target:
weight_adjustment = target_percentage / observed_percentage

# Example: If lane_keeping is 30% but target is 48%:
new_weight = 5.0 * (48 / 30) = 8.0
```

---

## Timeline & Resources

### Estimated Timeline
| Phase | Task | Duration | Cumulative |
|-------|------|----------|------------|
| 1 | Investigate Current Scales | 1 hour | 1 hour |
| 2 | Implement Normalization | 2 hours | 3 hours |
| 3 | Add Action Logging | 1 hour | 4 hours |
| 4 | 1K Validation Run | 1.5 hours | 5.5 hours |
| 5 (Optional) | 10K Validation | 2 hours | 7.5 hours |

**Total (with optional)**: 7.5 hours (1 working day)  
**Total (critical path)**: 5.5 hours (most of 1 working day)

### Resource Requirements
- **Compute**: 1 GPU (same as training)
- **Storage**: ~500 MB for logs/checkpoints
- **Network**: CARLA simulator running

---

## Success Metrics

### Phase 4 (1K Validation) - REQUIRED
| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Lane keeping % | 6% | 40-50% | 🔴 FAIL |
| Progress % | 83% | < 15% | 🔴 FAIL |
| Steering mean | ??? | [-0.2, 0.2] | ⚠️ UNKNOWN |
| Steering std | ??? | [0.1, 0.3] | ⚠️ UNKNOWN |

### Phase 5 (10K Validation) - OPTIONAL
| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Lane invasion trend | Flat (0%) | Decreasing > 10% | 🔴 FAIL |
| Episode length trend | Decreasing (-32%) | Increasing > 10% | 🔴 FAIL |
| Reward trend | Negative | Stable or positive | 🔴 FAIL |

---

## Deliverables

### Code Changes
1. ✅ `src/environment/reward_functions.py`:
   - Add `_normalize_component()` method
   - Add `component_ranges` dict
   - Update `calculate()` to normalize before weighting
   - Add debug logging for raw/normalized/weighted components

2. ✅ `src/agents/td3_agent.py`:
   - Add `action_buffer` for statistics
   - Add `get_action_stats()` method
   - Track actions in `select_action()`

3. ✅ `scripts/train_td3.py`:
   - Add TensorBoard logging for action statistics
   - Log steering/throttle mean/std/min/max

4. ✅ `config/td3_carla_town01.yaml`:
   - Update reward weights:
     - `lane_keeping`: 2.0 → 5.0
     - `safety`: 1.0 → 3.0
     - `progress`: 2.0 → 1.0

### Documentation
1. ✅ This action plan document
2. ✅ Phase 1 analysis results (component scales)
3. ✅ Phase 4 validation report (1K run)
4. ✅ (Optional) Phase 5 validation report (10K run)
5. ✅ Git commit messages documenting changes

### Analysis Artifacts
1. ✅ `raw_components.txt` (Phase 1)
2. ✅ `weighted_components.txt` (Phase 1)
3. ✅ Component scale analysis script output
4. ✅ TensorBoard events (1K/10K runs)
5. ✅ Validation analysis script output

---

## Next Steps After Validation

### If Phase 4 PASSES (1K Validation)
1. Run Phase 5 (10K validation) - RECOMMENDED
2. If Phase 5 passes: Proceed to 100K validation
3. If 100K passes: GREEN LIGHT for 1M run
4. Document final reward configuration

### If Phase 4 FAILS (1K Validation)
1. Analyze failure mode:
   - Still imbalanced? → Adjust weights further
   - Action distribution wrong? → Investigate action generation
   - Components saturating? → Adjust normalization ranges

2. Iterate on fixes
3. Re-run 1K validation
4. Do NOT proceed to 10K until 1K passes

### If Phase 5 FAILS (10K Validation)
1. Check if learning is happening at all:
   - Q-values updating? → Training working
   - Gradients flowing? → Networks functional
   - Reward variance decreasing? → Exploration reducing

2. Consider hyperparameter tuning:
   - Learning rate adjustment
   - Batch size changes
   - Exploration noise decay rate

3. Re-evaluate reward function design
4. Consult literature for similar tasks

---

## Appendix: Alternative Approaches

### Alternative 1: Adaptive Weight Tuning
Instead of fixed weights, dynamically adjust based on component magnitudes:

```python
def calculate(self, ...):
    # Compute raw components
    components_raw = {...}
    
    # Compute adaptive weights based on magnitudes
    magnitudes = {k: abs(v) for k, v in components_raw.items()}
    total_magnitude = sum(magnitudes.values())
    
    # Normalize weights to sum to 1.0
    adaptive_weights = {
        k: self.reward_weights[k] / (magnitudes[k] / total_magnitude)
        for k in components_raw
    }
    
    # Apply adaptive weights
    weighted = {
        k: components_raw[k] * adaptive_weights[k]
        for k in components_raw
    }
```

**Pros**: Automatically balances components  
**Cons**: Non-stationary reward function (harder to learn)  
**Recommendation**: Try only if Phase 2 normalization fails

---

### Alternative 2: Reward Clipping
Clip each component to fixed range before weighting:

```python
def calculate(self, ...):
    # Compute components
    efficiency = np.clip(efficiency_raw, -1.0, 1.0)
    lane_keeping = np.clip(lane_keeping_raw, -1.0, 1.0)
    progress = np.clip(progress_raw, -1.0, 1.0)
    # etc.
```

**Pros**: Simpler than normalization  
**Cons**: Loses information at saturation  
**Recommendation**: Use as fallback if normalization too complex

---

### Alternative 3: Separate Q-Networks per Component
Train separate critic networks for each reward component:

```python
class MultiComponentTD3:
    def __init__(self, ...):
        self.critics = {
            'efficiency': CriticNetwork(...),
            'lane_keeping': CriticNetwork(...),
            # etc.
        }
    
    def train(self, ...):
        # Train each critic separately
        for component, critic in self.critics.items():
            critic.train(reward=reward_dict[component])
        
        # Combine Q-values for actor update
        total_Q = sum(critic(s, a) * weight for critic, weight in zip(...))
```

**Pros**: Each component learns independently  
**Cons**: Much more complex, slower training  
**Recommendation**: Academic interest only, not practical for this project

---

**Document Version**: 1.0  
**Status**: 📋 **ACTION PLAN READY**  
**Next Action**: Begin Phase 1 (Investigate Current Scales)  
**Owner**: Development Team  
**Reviewer**: Project Lead  
**Approval Required**: Before proceeding to 1M run
