# Configuration Analysis - 30K Training Run
## Post-Validation Analysis & Root Cause Identification

**Date**: November 21, 2025  
**Run**: 30K steps training (20 NPCs, scenario 0)  
**Status**: ⚠️ **CRITICAL CONFIGURATION ERROR IDENTIFIED**  
**Companion Document**: `VALIDATION_30K_CRITICAL_FINDINGS.md`

---

## Executive Summary

**CRITICAL FINDING**: The 30K training run suffered from a **CONFIGURATION ERROR** that directly caused the catastrophic training failure. The `distance_scale` parameter was set to **10.0** instead of the recommended **5.0**, creating a 100× increase from the baseline 0.1 value and amplifying progress rewards to **unsustainable levels**.

**Impact**:
- Progress rewards dominated 88-99% of total reward magnitude
- Agent learned extreme actions (steering +0.94, throttle +0.94) to maximize progress
- Every episode terminated off-road at 16 steps
- 30K run performed **WORSE** than 5K baseline (severe regression)

**Verdict**: The "fixes" from the 5K analysis were **NOT applied correctly**. The configuration shows the **OPPOSITE** of recommended changes.

---

## Section 1: Configuration File Analysis

### 1.1 Reward Configuration (`training_config.yaml`)

**Progress Reward Parameters** (Lines 148-156):
```yaml
progress:
  waypoint_bonus: 5.0         # Bonus for reaching waypoints
  distance_scale: 10.0        # 🔧 FIXED: 50.0 (was 1.0, originally 0.1) - 50x signal
  goal_reached_bonus: 100.0   # Bonus for reaching final goal
```

**Issue Identified**:
- **`distance_scale: 10.0`** ← **WRONG!**
  - Recommended value: **5.0** (from 5K diagnostic)
  - Current value: **10.0** (2× higher than recommended)
  - Original baseline: **0.1**
  - Change applied: **100× INCREASE** (0.1 → 10.0)
  - Expected change: **50× INCREASE** (0.1 → 5.0)

**Analysis**:
- The 5K diagnostic recommended: `distance_scale: 50.0 → 5.0` (10× reduction from overcorrection)
- The 30K config shows: `distance_scale: 0.1 → 10.0` (100× increase from original)
- The comment "was 1.0, originally 0.1" suggests configuration history confusion
- **The fix was overcorrected in the WRONG direction**

**Impact Calculation**:
```
Moving 1 meter with distance_scale=10.0:
- Raw progress reward: +10.0
- Weighted (progress_weight=1.0): +10.0
- Comparison to collision penalty (-10.0): EQUAL!

This means the agent can:
- Drive off-road for 16 steps
- Accumulate +160.0 progress reward
- Offset ANY collision penalty
- Net result: Off-road driving is PROFITABLE!
```

---

### 1.2 Weight Configuration (`training_config.yaml`)

**Reward Weights** (Lines 30-36):
```yaml
weights:
  efficiency: 1.0      # Base weight (forward velocity reward)
  lane_keeping: 5.0    # Lane centering
  comfort: 0           # Smooth driving (disabled)
  safety: 1.0          # Safety penalties (correct: penalties are already negative)
  progress: 1.0        # Forward progress incentive
```

**Status**: ✅ **CORRECT** - All weights match recommended values from 5K analysis.

**Verification**:
- Efficiency: 1.0 ✓
- Lane keeping: 5.0 ✓
- Comfort: 0 (disabled) ✓
- Safety: 1.0 (FIXED from -100.0) ✓
- Progress: 1.0 (REDUCED from 5.0) ✓

---

### 1.3 Exploration Configuration (`td3_config.yaml`)

**Exploration Noise** (Line 48):
```yaml
exploration_noise: 0.1  # Std of Gaussian noise (scaled by max_action)
```

**Issue Identified**:
- **Current value: 0.1** ← **TOO LOW!**
- **Recommended value: 0.2** (from 30K analysis)
- **Result**: Action std = 0.08 (expected 0.2-0.4)

**Analysis**:
From TensorBoard metrics (30K run):
- Steering std: 0.0812 (too low)
- Throttle std: 0.0797 (too low)
- Expected range: 0.2-0.4 for healthy exploration

**Impact**:
- Low exploration prevents discovery of better actions
- Agent stuck in local minimum (hard right + full throttle)
- Policy cannot escape failure mode

---

### 1.4 Learning Rate Configuration (`td3_config.yaml`)

**Network Learning Rates** (Lines 15-17, 70-90):
```yaml
algorithm:
  learning_rate: 0.001  # 1e-3, TD3 paper default for BOTH actor and critic

networks:
  cnn:
    actor_cnn_lr: 0.001   # 1e-3 (TD3 paper default, matches actor/critic)
    critic_cnn_lr: 0.001  # 1e-3 (TD3 paper default, matches actor/critic)

  actor:
    learning_rate: 0.001  # 1e-3 (TD3 paper default)

  critic:
    learning_rate: 0.001  # 1e-3 (TD3 paper default)
```

**Status**: ✅ **CORRECT** - All learning rates match TD3 paper defaults (1e-3).

**Verification**:
- Algorithm LR: 0.001 ✓
- Actor LR: 0.001 ✓
- Critic LR: 0.001 ✓
- Actor CNN LR: 0.001 ✓
- Critic CNN LR: 0.001 ✓

**Note**: Configuration comments indicate previous incorrect values were fixed:
- "FIXED: Restored from 3e-5 (333× TOO SLOW!)" (Actor)
- "FIXED: Restored from 1e-4 (10× TOO SLOW!)" (Critic)
- "FIXED (November 20, 2025): Restore to TD3 paper defaults" (CNNs)

---

## Section 2: Reward Implementation Analysis

### 2.1 Progress Reward Calculation (`reward_functions.py`)

**Key Code Section** (Lines 609-680):
```python
def _calculate_progress_reward(
    self,
    distance_to_goal: float,
    waypoint_reached: bool,
    goal_reached: bool,
) -> float:
    """
    Calculate progress reward based on route distance reduction.

    FIX #1: REMOVED PBRS - Bug gave free reward proportional to distance from goal.
    FIX #2: Uses ROUTE DISTANCE instead of Euclidean to prevent off-road shortcuts.
    """
    progress = 0.0

    # Component 1: Route distance-based reward (dense, continuous)
    if self.prev_distance_to_goal is not None:
        distance_delta = self.prev_distance_to_goal - distance_to_goal
        distance_reward = distance_delta * self.distance_scale  # ← MULTIPLIED BY 10.0!
        progress += distance_reward

    # Component 2: Waypoint milestone bonus
    if waypoint_reached:
        progress += self.waypoint_bonus  # ← ADDS 5.0

    # Component 3: Goal reached bonus
    if goal_reached:
        progress += self.goal_reached_bonus  # ← ADDS 100.0

    return float(np.clip(progress, -10.0, 110.0))
```

**Status**: ✅ **CODE IS CORRECT** - Implementation matches intended design.

**Issue**: The code correctly implements the reward function, but uses the **WRONG configuration values** loaded from `training_config.yaml`.

**Flow**:
```
1. Config loaded: distance_scale = 10.0 (WRONG!)
2. __init__: self.distance_scale = 10.0
3. _calculate_progress_reward: distance_reward = delta * 10.0
4. Result: Moving 1m → +10.0 progress reward (TOO HIGH!)
```

---

### 2.2 Reward Component Weights (`reward_functions.py`)

**Initialization Code** (Lines 34-46):
```python
self.weights = config.get("weights", {
    "efficiency": 1.0,
    "lane_keeping": 5.0,  # INCREASED from 2.0: Prioritize staying in lane
    "comfort": 0.5,
    "safety": 1.0,        # FIXED from -100.0: Penalties are already negative!
    "progress": 1.0,      # REDUCED from 5.0: Prevent domination
})
```

**Status**: ✅ **CORRECT** - Weights match recommended values.

**Total Reward Calculation** (Lines 264-270):
```python
total_reward = (
    self.weights["efficiency"] * efficiency
    + self.weights["lane_keeping"] * lane_keeping
    + self.weights["comfort"] * comfort
    + self.weights["safety"] * safety
    + self.weights["progress"] * progress
)
```

**Verification**:
```
Example episode step calculation (with distance_scale=10.0):

INPUTS:
- velocity = 5.0 m/s
- lateral_deviation = 0.2 m
- heading_error = 0.05 rad
- collision = False
- distance_delta = +1.0 m (forward progress)

COMPONENT CALCULATIONS:
1. Efficiency: +0.60 (5.0/8.33, target speed)
2. Lane keeping: +0.40 (centered, small heading error)
3. Comfort: 0.0 (disabled, weight=0)
4. Safety: 0.0 (no collision/offroad)
5. Progress: +10.0 (1.0m × 10.0 scale) ← DOMINATES!

WEIGHTED CONTRIBUTIONS:
1. Efficiency: 1.0 × 0.60 = +0.60
2. Lane keeping: 5.0 × 0.40 = +2.00
3. Comfort: 0 × 0.0 = 0.00
4. Safety: 1.0 × 0.0 = 0.00
5. Progress: 1.0 × 10.0 = +10.0 ← 79% OF TOTAL!

TOTAL REWARD: +12.60
PROGRESS DOMINANCE: 10.0 / 12.60 = 79.4%
```

**Conclusion**: Even with correct weights (progress=1.0), the **excessive distance_scale=10.0** causes progress to dominate the reward signal.

---

## Section 3: Root Cause Analysis

### 3.1 Configuration Error Origin

**Evidence from Training Log** (`run-validation_30k_post_all_fixes_20251121_200106.log`):
```
PROGRESS REWARD PARAMETERS:
  waypoint_bonus: 5.0 (was 10.0)
  distance_scale: 10.0 (was 0.1)  ← CRITICAL ERROR!
  goal_reached_bonus: 100.0 (was 100.0)
```

**Analysis**:
1. **"was 0.1"** suggests the baseline distance_scale was 0.1
2. **5K diagnostic recommended**: Reduce from 50.0 to 5.0 (10× reduction)
3. **30K config applied**: Increase from 0.1 to 10.0 (100× increase!)
4. **This is the OPPOSITE of the recommended fix**

**Possible Causes**:
1. **Configuration file version confusion**: 
   - 5K run may have used `distance_scale: 50.0` (overcorrected)
   - 30K run reverted to older config with different baseline
   - Applied fix relative to wrong baseline (0.1 instead of 50.0)

2. **Copy-paste error from comments**:
   - Comment says "was 1.0, originally 0.1" (Line 149 in config)
   - Suggests multiple configuration iterations
   - May have applied fix to wrong version

3. **Misunderstanding of fix recommendation**:
   - 5K diagnostic: "Reduce distance_scale to 5.0"
   - Interpreted as: "Set distance_scale to 10× the original"
   - Original: 0.1 → 10× = 1.0 → further 10× = 10.0 (WRONG!)

---

### 3.2 Impact Chain

**Configuration Error → Training Failure Cascade**:

```
1. distance_scale: 10.0 (should be 5.0)
   ↓
2. Moving 1m → +10.0 progress reward (2× too high)
   ↓
3. Progress dominates 79-88% of total reward
   ↓
4. Agent learns: "Move forward at any cost"
   ↓
5. Off-road driving gives +160.0 progress over 16 steps
   ↓
6. Collision penalty (-10.0) easily offset by progress
   ↓
7. Policy converges to: Hard right + Full throttle
   ↓
8. Every episode: 16 steps → off-road termination
   ↓
9. Q-values: Negative (correctly predicting failure)
   ↓
10. Actor loss: Diverging (exploiting bad Q-estimates)
   ↓
11. Result: 0% success rate, -168.50 mean reward
```

---

### 3.3 Comparison: 5K vs 30K Configuration

| Parameter | 5K Diagnostic | 30K Config | Status |
|-----------|---------------|------------|--------|
| `distance_scale` | 50.0 → **5.0** | **10.0** ❌ | WRONG (2× too high) |
| `waypoint_bonus` | 10.0 → **1.0** | **5.0** ⚠️ | WRONG (5× too high) |
| `expl_noise` | 0.1 → **0.2** | **0.1** ❌ | WRONG (not increased) |
| `efficiency_weight` | 1.0 | 1.0 ✅ | CORRECT |
| `lane_keeping_weight` | 5.0 | 5.0 ✅ | CORRECT |
| `safety_weight` | -100.0 → **1.0** | 1.0 ✅ | CORRECT |
| `progress_weight` | 5.0 → **1.0** | 1.0 ✅ | CORRECT |

**Summary**:
- Weights: 4/5 correct ✅
- Progress parameters: 0/3 correct ❌
- Exploration: 0/1 correct ❌

**Conclusion**: The **reward component weights** were applied correctly, but the **progress reward parameters** and **exploration noise** were **NOT** applied as recommended.

---

## Section 4: Fixes Required

### P0 - CRITICAL (Fix IMMEDIATELY)

#### Fix #1: Correct `distance_scale` ✅ IDENTIFIED
**File**: `config/training_config.yaml`  
**Line**: 149  
**Current**: `distance_scale: 10.0`  
**Required**: `distance_scale: 5.0`  

**Change**:
```yaml
progress:
  waypoint_bonus: 5.0
  distance_scale: 5.0  # CHANGE FROM 10.0
  goal_reached_bonus: 100.0
```

**Expected Impact**:
- Progress reward per meter: +10.0 → **+5.0** (50% reduction)
- Progress dominance: 79% → **~50%** (balanced with other components)
- Agent can no longer offset collision penalty with off-road progress

---

#### Fix #2: Correct `waypoint_bonus` ✅ IDENTIFIED
**File**: `config/training_config.yaml`  
**Line**: 148  
**Current**: `waypoint_bonus: 5.0`  
**Required**: `waypoint_bonus: 1.0`  

**Change**:
```yaml
progress:
  waypoint_bonus: 1.0  # CHANGE FROM 5.0
  distance_scale: 5.0
  goal_reached_bonus: 100.0
```

**Expected Impact**:
- Waypoint bonus: +5.0 → **+1.0** (80% reduction)
- Reduces discrete reward spikes that can cause policy instability

---

#### Fix #3: Increase `exploration_noise` ✅ IDENTIFIED
**File**: `config/td3_config.yaml`  
**Line**: 48  
**Current**: `exploration_noise: 0.1`  
**Required**: `exploration_noise: 0.2`  

**Change**:
```yaml
exploration_noise: 0.2  # CHANGE FROM 0.1
```

**Expected Impact**:
- Action std: 0.08 → **0.20** (2.5× increase)
- Agent can explore beyond failure mode (hard right + full throttle)
- Increases probability of discovering centered steering actions

---

### P1 - HIGH (After P0 Validation)

#### Fix #4: Verify Reward Calculation Logic ✅ VERIFIED
**File**: `src/environment/reward_functions.py`  
**Status**: **CORRECT** - Code implementation matches intended design  
**Action**: Read code to confirm no implementation bugs (COMPLETED)

**Findings**:
- Progress reward calculation: ✅ Correct
- Weight application: ✅ Correct
- Reward accumulation: ✅ Correct
- **Issue**: Code is correct, but uses wrong config values

---

#### Fix #5: Add Configuration Validation ⏳ PENDING
**File**: `src/environment/reward_functions.py` (or new `validate_config.py`)  
**Action**: Add sanity checks for configuration values  

**Proposed Checks**:
```python
def validate_reward_config(config: Dict) -> None:
    """Validate reward configuration parameters."""
    
    # Check distance_scale is in expected range
    distance_scale = config.get("progress", {}).get("distance_scale", 1.0)
    assert 1.0 <= distance_scale <= 10.0, \
        f"distance_scale={distance_scale} out of range [1.0, 10.0]"
    
    # Check waypoint_bonus is reasonable
    waypoint_bonus = config.get("progress", {}).get("waypoint_bonus", 1.0)
    assert 0.1 <= waypoint_bonus <= 2.0, \
        f"waypoint_bonus={waypoint_bonus} out of range [0.1, 2.0]"
    
    # Check exploration_noise is sufficient
    # (This would go in TD3 config validation)
    
    # Log validated config
    logger.info("✅ Configuration validation passed")
    logger.info(f"  distance_scale: {distance_scale}")
    logger.info(f"  waypoint_bonus: {waypoint_bonus}")
```

---

## Section 5: Expected Results After Fixes

### 5.1 Immediate Effects (P0 Fixes Applied)

**Steering Behavior**:
- Current: Mean = 0.94, Std = 0.08 (extreme right bias)
- Expected: Mean = -0.1 to +0.1, Std = 0.20 (centered, exploring)

**Episode Performance**:
- Current: Length = 16 steps (constant off-road)
- Expected: Length > 30 steps (exploring, not stuck)

**Reward Distribution**:
- Current: Progress = 79-88% dominance
- Expected: Progress = 40-50% (balanced with other components)

**Q-Values**:
- Current: -14.81 (predicting failure correctly)
- Expected: 0 to +30 (predicting mixed outcomes)

---

### 5.2 Short-Term Goals (5K Validation)

**Success Criteria**:
- [ ] Steering mean: -0.2 to +0.2 (centered)
- [ ] Action std: > 0.15 (exploring)
- [ ] Episode length: > 30 steps (not stuck at 16)
- [ ] Episode reward: Positive (not -168.50)
- [ ] Progress dominance: < 60% (not 79-88%)
- [ ] Q-values: 0 to +30 range (not -14.81)
- [ ] Actor loss: -50 to -100 (not -13364)
- [ ] Lane invasion rate: < 80% (not 99.6%)

---

### 5.3 Long-Term Goals (100K Training)

**Success Criteria**:
- [ ] Success rate: > 10% (not 0%)
- [ ] Avg episode length: > 100 steps (not 16)
- [ ] Mean episode reward: > 50 (not -168.50)
- [ ] Lane invasion rate: < 50% (not 99.6%)
- [ ] Collision rate: < 10%
- [ ] Q-values: Stable in 10-30 range
- [ ] CNN gradients: 0.1-10 range (healthy)

---

## Section 6: Lessons Learned

### 6.1 Configuration Management

**Issue**: Multiple configuration versions without clear tracking caused fixes to be applied to wrong baseline.

**Solutions**:
1. **Version control for configs**: Use git tags for each experiment
2. **Configuration diff logging**: Log all config changes in training log header
3. **Baseline documentation**: Document "gold standard" config values
4. **Automated validation**: Add sanity checks for out-of-range values

---

### 6.2 Reward Engineering

**Issue**: Small changes to reward parameters can have catastrophic effects on training.

**Solutions**:
1. **Sensitivity analysis**: Test parameter changes in isolation (5K runs)
2. **Ablation studies**: Change ONE parameter at a time
3. **Reward component monitoring**: Track contribution % in TensorBoard
4. **Dominance alerts**: Log warnings when any component > 80%

---

### 6.3 Documentation Quality

**Issue**: Configuration comments were misleading ("was 1.0, originally 0.1") and caused confusion.

**Solutions**:
1. **Clear change history**: Document WHY each change was made
2. **Reference fixes**: Link to analysis documents in comments
3. **Remove outdated comments**: Clean up old version references
4. **Canonical values**: Mark recommended values clearly

---

## Section 7: Next Steps

### Immediate (Today)

1. ✅ **Read configuration files** (COMPLETED)
2. ✅ **Verify reward calculation code** (COMPLETED)
3. ✅ **Identify root cause** (COMPLETED - distance_scale error)
4. ⏳ **Apply P0 Fix #1**: `distance_scale: 10.0 → 5.0`
5. ⏳ **Apply P0 Fix #2**: `waypoint_bonus: 5.0 → 1.0`
6. ⏳ **Apply P0 Fix #3**: `exploration_noise: 0.1 → 0.2`
7. ⏳ **Run 5K validation**: Verify fixes work

### Short-Term (This Week)

8. ⏳ **Add configuration validation**: Sanity checks for parameters
9. ⏳ **Document baseline config**: Create "gold standard" reference
10. ⏳ **Run 100K training**: If 5K validation succeeds
11. ⏳ **Monitor TensorBoard**: Track progress dominance metric

### Long-Term (Paper Preparation)

12. ⏳ **Implement CNN debug logging**: Gradient tracking
13. ⏳ **Benchmark against DDPG**: Quantify TD3 improvement
14. ⏳ **Write methodology section**: Document full training procedure
15. ⏳ **Prepare results**: Tables, figures, statistical analysis

---

## Appendix A: Configuration File Locations

```
av_td3_system/
├── config/
│   ├── training_config.yaml    # Main training configuration
│   ├── carla_config.yaml        # CARLA environment settings
│   └── td3_config.yaml          # TD3 algorithm hyperparameters
├── src/
│   └── environment/
│       └── reward_functions.py  # Reward calculation implementation
└── docs/
    └── day-21/
        └── run7/
            ├── VALIDATION_30K_CRITICAL_FINDINGS.md  # Main analysis
            ├── CONFIGURATION_ANALYSIS_30K.md        # This document
            └── run-validation_30k_post_all_fixes_20251121_200106.log
```

---

## Appendix B: Quick Reference - Correct Values

**Progress Rewards** (`training_config.yaml`):
```yaml
progress:
  waypoint_bonus: 1.0         # ✅ CORRECT
  distance_scale: 5.0         # ❌ WRONG (currently 10.0)
  goal_reached_bonus: 100.0   # ✅ CORRECT
```

**Reward Weights** (`training_config.yaml`):
```yaml
weights:
  efficiency: 1.0      # ✅ CORRECT
  lane_keeping: 5.0    # ✅ CORRECT
  comfort: 0           # ✅ CORRECT
  safety: 1.0          # ✅ CORRECT
  progress: 1.0        # ✅ CORRECT
```

**Exploration** (`td3_config.yaml`):
```yaml
exploration_noise: 0.2  # ❌ WRONG (currently 0.1)
```

---

**END OF CONFIGURATION ANALYSIS**
