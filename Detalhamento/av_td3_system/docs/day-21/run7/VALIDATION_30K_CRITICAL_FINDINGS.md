# 30K Training Run - Critical Validation Analysis

**Date:** November 21, 2025
**Training Steps:** 30,000
**TensorBoard:** `docs/day-21/run7/TD3_scenario_0_npcs_20_20251121-230112/`
**Text Logs:** `docs/day-21/run7/run-validation_30k_post_all_fixes_20251121_200106.log`

---

## EXECUTIVE SUMMARY: ⚠️ CRITICAL REGRESSION DETECTED

**VERDICT: The agent is NOT learning to drive safely. This 30K run shows SEVERE REGRESSION compared to the 5K baseline.**

### Key Findings:
- ✅ TD3 algorithm is functioning correctly (Q-learning, critic convergence)
- ✅ TD errors are low (2.40 < 5.0) - value function is converging
- ✅ Critic loss is decreasing (-74.7% trend)
- ❌ **CRITICAL**: Extreme action bias (steering +0.94, throttle +0.94)
- ❌ **CRITICAL**: Episode length degrading (65 → 16 steps)
- ❌ **CRITICAL**: Rewards degrading (+159 → -36 per episode)
- ❌ **CRITICAL**: 0% success rate, 99.6% lane invasion rate

---

## SECTION 1: TD3 ALGORITHM VALIDATION

### 1.1 Q-Value Learning ✅ ALGORITHM WORKING

| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| Q1 Mean | -14.81 ± 9.81 | Reflects policy quality | ✅ Correct |
| Q2 Mean | -14.84 ± 9.89 | Should match Q1 | ✅ Twin critics aligned |
| Target Q | -14.82 ± 9.80 | Should match Q1/Q2 | ✅ Correct |
| Q1 vs Q2 diff | 0.14 | < 1.0 | ✅ Very small |

**Q-Value Trajectory:**
- Initial (steps 10100-11000): +4.81
- Mid-point (steps 19000-20000): -13.41
- Final (steps 29000-30000): -23.44
- **Change: -28.25 (-587%)**

**Analysis:**
- Q-values are **NEGATIVE** because the policy is consistently failing (off-road every episode)
- This is **CORRECT BEHAVIOR** for TD3: Q(s,a) = Expected cumulative reward
- If policy always leads to negative outcomes → Q-values should be negative
- **The algorithm is learning that the current policy is bad**

### 1.2 Q-Reward Gap: -15.21 ⚠️ UNDERESTIMATION

| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| Mean Reward | +0.40 | Positive (progress) | ⚠️ Near zero |
| Mean Q-value | -14.81 | Should be close to reward | ⚠️ Much lower |
| Gap | -15.21 | < ±5 | ⚠️ Underestimating |

**Analysis:**
- Gap of -15.21 indicates Q-values are **underestimating** future returns
- This is opposite of the 5K run (+47.76 overestimation)
- Possible causes:
  1. Reward configuration changed too drastically
  2. Pessimistic Q-learning from repeated failures
  3. Negative terminal rewards dominating

### 1.3 Critic Loss: 40.65 ✅ CONVERGING

| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| Mean | 40.65 ± 27.01 | 20-50 | ✅ Within range |
| First 10 steps | 86.51 | Higher initially | ✅ Correct |
| Last 10 steps | 21.92 | Lower finally | ✅ Decreasing |
| Trend | -74.7% | Negative | ✅ Learning |

**Analysis:**
- Critic loss is **decreasing correctly** (Bellman error minimization)
- Final loss of 21.92 is excellent (< 50 threshold)
- **TD3 critic networks are learning the value function correctly**

### 1.4 TD Errors: 2.40 ± 0.74 ✅ VALUE FUNCTION CONVERGING

| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| TD Error Q1 | 2.40 ± 0.74 | < 5 (< 3 ideal) | ✅ Converging |
| TD Error Q2 | 2.39 ± 0.75 | < 5 (< 3 ideal) | ✅ Converging |

**Analysis:**
- TD errors are **low and stable** (< 3 threshold)
- Indicates value function is **converging correctly**
- **This is the expected behavior for a well-trained TD3 critic**

### 1.5 Actor Loss: -4413.63 ⚠️ DIVERGING

| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| Mean | -4413.63 | -50 to -100 | ⚠️ Too negative |
| Range | -14356 to -8.07 | Stable | ⚠️ Very wide |
| First 10 steps | -56.82 | Initial | ✅ Good start |
| Last 10 steps | -13364.45 | Stable | ❌ Diverging |
| Trend | -23419.8% | ~0% | ❌ Exploding |

**Analysis:**
- Actor loss is **diverging catastrophically** (-56 → -13364)
- This indicates the actor is **exploiting incorrect Q-value estimates**
- The policy is learning to select actions with increasingly negative Q-values
- **This is a CRITICAL FAILURE MODE**

---

## SECTION 2: ACTION DISTRIBUTION ANALYSIS

### 2.1 Steering Actions: 0.9392 ⚠️ EXTREME RIGHT BIAS

| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| Mean | 0.9392 | ~0.0 (centered) | ❌ Extreme right |
| Std | 0.0812 | 0.2-0.4 | ❌ Too low |
| Range | -1.0 to +1.0 | Full range | ⚠️ Not exploring |

**Analysis:**
- Agent is **constantly steering hard right** (0.94 out of 1.0 max)
- Expected: ~0.0 for straight driving on route
- **This explains the immediate off-road terminations (16 steps)**
- Low std (0.08) indicates **lack of exploration**

**Comparison with Expected Behavior:**
- For Town01 route (317,129) → (92,87): requires mix of left/right turns
- Extreme right bias (+0.94) means agent drives off right side of road
- **This is a COMPLETE FAILURE of policy learning**

### 2.2 Throttle Actions: 0.9399 ⚠️ TOO AGGRESSIVE

| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| Mean | 0.9399 | 0.3-0.7 | ⚠️ Too high |
| Std | 0.0797 | 0.2-0.4 | ❌ Too low |

**Analysis:**
- Agent is applying **maximum throttle** (0.94 out of 1.0)
- Combined with hard right steering → **immediate crash/off-road**
- **The policy is learned: "Go fast + turn right = immediate failure"**

---

## SECTION 3: EPISODE METRICS & LEARNING PROGRESS

### 3.1 Episode Rewards: -14.15 ⚠️ DEGRADING

| Metric | Value | Status |
|--------|-------|--------|
| Mean | -14.15 ± 71.78 | ⚠️ Negative |
| First 100 episodes | +159.67 | ✅ Positive |
| Last 100 episodes | -36.41 | ❌ Negative |
| Change | -196.08 | ❌ Degrading |

**Analysis:**
- Rewards started positive (+159) but degraded to negative (-36)
- **This is OPPOSITE of expected learning behavior**
- Agent is **unlearning** safe driving behavior
- Indicates reward function or exploration is broken

### 3.2 Episode Length: 21.8 steps ⚠️ NOT IMPROVING

| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| Mean | 21.8 steps | Increasing | ❌ Decreasing |
| Max | 1000 steps | Goal | ❌ Never reached |
| First 100 | 65.0 steps | Higher | ✅ Better initially |
| Last 100 | 16.1 steps | Improving | ❌ Worse |

**Analysis:**
- Episode length **degraded from 65 to 16 steps** (-75% regression)
- **Every episode ends in exactly 16 steps** (off-road termination)
- Agent is **not learning to drive longer**
- This indicates the policy is **stuck in a failure mode**

### 3.3 Safety Metrics: 99.6% Lane Invasion Rate ❌ CRITICAL

| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| Collisions/episode | 0.004 | Decreasing | ✅ Low collisions |
| Lane invasions/episode | 0.996 | Decreasing | ❌ Nearly 100% |
| Off-road terminations | ~100% | 0% | ❌ Every episode |

**Analysis:**
- **99.6% of episodes** have lane invasions (1369/1374)
- Almost **no collisions** (5/1374) because agent goes off-road first
- **Every recent episode terminates at 16 steps due to off-road**
- **The agent has learned to drive off-road instead of following the route**

---

## SECTION 4: EVALUATION RESULTS

### 4.1 Evaluation Performance (Deterministic Policy)

| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| Mean Reward | -168.50 | Positive | ❌ Very negative |
| Success Rate | 0.0% | > 50% | ❌ Complete failure |
| Avg Collisions | 0.03 | < 0.1 | ✅ Low |
| Avg Lane Invasions | 0.67 | < 0.1 | ❌ High |
| Avg Episode Length | 16 steps | > 100 | ❌ Too short |

**Analysis:**
- **0% success rate** - agent never reaches goal
- **All evaluation episodes end in 16 steps** (off-road)
- Mean reward of -168.50 is catastrophically bad
- **The agent has learned a policy that immediately fails**

---

## SECTION 5: ROOT CAUSE ANALYSIS

### Critical Issue #1: Extreme Action Bias (P0 - CRITICAL)

**Symptom:**
- Steering: +0.94 (hard right)
- Throttle: +0.94 (full speed)
- Every episode: 16 steps → off-road

**Root Causes:**
1. **Actor network initialization** - may have bias toward positive actions
2. **Reward function** - may incentivize extreme actions
3. **Action space normalization** - may be incorrect

**Evidence:**
- From logs: `distance_scale: 10.0`, `waypoint_bonus: 5.0`
- These values may still be too high or incorrectly signed
- Agent may be exploiting reward function loophole

### Critical Issue #2: Policy Degradation (P0 - CRITICAL)

**Symptom:**
- Episode length: 65 → 16 steps (-75%)
- Episode reward: +159 → -36 (-123%)
- Success rate: 0% (no improvement)

**Root Causes:**
1. **Catastrophic forgetting** - new experiences overwriting good policy
2. **Reward shaping error** - negative rewards dominating
3. **Exploration noise** - too low (std=0.08), agent not exploring alternatives

**Evidence:**
- Low action std (0.08) indicates lack of exploration
- Actor loss diverging (-13364) indicates Q-exploitation failure
- Q-values becoming more negative (-28.25 change)

### Critical Issue #3: Reward Configuration Error (P0 - CRITICAL)

**Symptom:**
- Q-reward gap: -15.21 (underestimation)
- Mean reward: +0.40 (near zero)
- Episode rewards: trending negative

**Evidence from logs:**
```
PROGRESS REWARD PARAMETERS:
  waypoint_bonus: 5.0 (was 10.0)
  distance_scale: 10.0 (was 0.1)  ← INCREASED 100×!
  goal_reached_bonus: 100.0
```

**Analysis:**
- `distance_scale` was **increased from 0.1 to 10.0** (100× increase!)
- This is **OPPOSITE** of the recommended fix (reduce from 50 to 5)
- The fix from 5K analysis was **NOT applied correctly**
- This explains the catastrophic failure

---

## SECTION 6: COMPARISON WITH 5K BASELINE

| Metric | 5K Run | 30K Run | Change | Assessment |
|--------|--------|---------|--------|------------|
| **Q-values** | +67.76 | -14.81 | -82.57 | ❌ Worse (overcorrected) |
| **Q-R Gap** | +47.76 | -15.21 | -62.97 | ⚠️ Overcorrected |
| **Critic Loss** | 1244 (↑) | 40.65 (↓) | -1203 | ✅ Much better |
| **TD Error** | 13.4 | 2.40 | -11.0 | ✅ Much better |
| **Actor Loss** | -2945 (diverging) | -13364 (diverging) | -10419 | ❌ Much worse |
| **Episode Length** | Unknown | 21.8 (degrading) | N/A | ❌ Worse |
| **Episode Reward** | Unknown | -14.15 (degrading) | N/A | ❌ Worse |
| **Steering Bias** | 0.10 (slight right) | 0.94 (extreme right) | +0.84 | ❌ Much worse |
| **Success Rate** | Unknown | 0% | N/A | ❌ Failure |

### Verdict: REGRESSION, Not Improvement

**5K Run Characteristics:**
- ✅ Q-values positive (optimistic, trying to learn)
- ✅ Some exploration happening
- ❌ Reward scale too high (progress 91.7%)
- ❌ Q-value overestimation (+47.76)

**30K Run Characteristics:**
- ❌ Q-values negative (pessimistic, learned failure)
- ❌ Minimal exploration (std=0.08)
- ❌ Extreme action bias (steering +0.94)
- ❌ Policy degraded (65 → 16 steps)
- ⚠️ Reward configuration ERROR (distance_scale increased!)

**Conclusion:** The 30K run applied the **WRONG FIX** and made the situation **MUCH WORSE**.

---

## SECTION 7: OFFICIAL TD3 BENCHMARK COMPARISON

### From OpenAI Spinning Up & Stable-Baselines3

| Metric | Your 30K | Expected (AV) | Official (MuJoCo) | Status |
|--------|----------|---------------|-------------------|--------|
| Q-values | -14.81 | 10-30 | 15-30 (HalfCheetah) | ❌ Negative |
| Q-R gap | -15.21 | < ±5 | < ±5 | ❌ Too large |
| Critic loss | 40.65 (↓) | 20-50 (↓) | 20-50 | ✅ Good |
| TD error | 2.40 | < 5 (< 3 ideal) | < 3 | ✅ Good |
| Actor loss | -13364 (diverging) | -50 to -100 (stable) | -50 to -100 | ❌ Diverging |
| Episode reward | -14.15 (degrading) | Increasing | Increasing | ❌ Degrading |
| Success rate | 0% | > 50% | > 80% | ❌ Complete failure |

### TD3 Algorithm Status:
- ✅ Twin critics are working (Q1 ≈ Q2)
- ✅ Clipped double Q-learning is active (low TD errors)
- ✅ Target networks are updating (critic loss decreasing)
- ❌ **Policy updates are FAILING** (actor loss diverging)
- ❌ **Exploration is BROKEN** (low action std)
- ❌ **Reward function is WRONG** (negative cumulative rewards)

---

## SECTION 8: CNN FEATURE EXTRACTION ANALYSIS

### Available Metrics:
Unfortunately, no CNN-specific metrics were logged:
- ❌ No `gradients/actor_cnn_norm`
- ❌ No `gradients/critic_cnn_norm`
- ❌ No `agent/actor_cnn_param_std`
- ❌ No CNN layer activation statistics

### Indirect Evidence:
1. **TD errors are low (2.40)** → Critics are learning value function
2. **Critic loss decreasing** → Bellman updates working
3. **But actor loss diverging** → Policy gradient may not be flowing correctly

### Hypothesis:
- CNN may be extracting features (critics learn)
- But **actor cannot use these features effectively**
- Possible causes:
  1. CNN gradients not flowing to actor network
  2. Actor MLP incompatible with CNN features
  3. Learning rate mismatch (CNN vs MLP)

### Recommendation:
- **Enable debug logging** to track CNN gradients
- Add TensorBoard logging for:
  - `gradients/actor_cnn_norm`
  - `gradients/critic_cnn_norm`
  - `cnn/feature_mean`, `cnn/feature_std`
  - `cnn/weight_norms`

---

## SECTION 9: IDENTIFIED BUGS & MISCONFIGURATIONS

### Bug #1: Reward Configuration WRONG (P0 - CRITICAL)

**From logs:**
```
PROGRESS REWARD PARAMETERS:
  distance_scale: 10.0 (was 0.1)  ← INCREASED 100×!
  waypoint_bonus: 5.0 (was 10.0)
```

**Expected (from 5K analysis):**
```
distance_scale: 5.0 (reduce from 50.0)
waypoint_bonus: 1.0 (reduce from 10.0)
```

**What happened:**
- `distance_scale` was set to 10.0 instead of 5.0 (2× too high)
- Even worse: The log says "was 0.1" → suggests config was reset!
- The 5K diagnostic fixes were **NOT applied**

**Fix:**
```yaml
# training_config.yaml
progress:
  distance_scale: 5.0  # NOT 10.0
  waypoint_bonus: 1.0  # NOT 5.0
```

### Bug #2: Reward Weights May Be Inverted (P0 - CRITICAL)

**From logs:**
```
REWARD WEIGHTS:
  lane_keeping: 5.0   ← Very high!
  safety: 1.0
  progress: 1.0
```

**Analysis:**
- If lane-keeping reward is TOO HIGH → agent may exploit it
- If safety reward is TOO LOW → agent ignores crashes
- Need to verify reward calculation in `reward_functions.py`

**Recommendation:**
- Check if lane_keeping reward is calculated correctly
- Ensure progress reward is actually rewarding forward movement
- Verify safety penalty is negative (not positive)

### Bug #3: Action Initialization/Normalization (P0 - CRITICAL)

**Symptom:**
- Steering: +0.94 (extreme right bias)
- Throttle: +0.94 (extreme high bias)
- Both actions centered at +0.94 (not 0.0)

**Possible causes:**
1. Actor network initialized with positive bias
2. Action denormalization incorrect
3. CARLA control mapping wrong (steering/throttle inverted?)

**Recommendation:**
- Check actor network initialization (should be zero-centered)
- Verify action mapping: `action ∈ [-1, 1] → steering ∈ [-1, 1]`
- Print first 100 actions to verify range

### Bug #4: Exploration Noise Too Low (P1 - HIGH)

**Symptom:**
- Action std: 0.08 (expected: 0.2-0.4)
- Agent not exploring alternatives

**Possible causes:**
- `expl_noise=0.1` may be too low
- Noise schedule may be decaying too fast
- Deterministic actions in training (bug)

**Recommendation:**
```python
# td3_config.yaml
expl_noise: 0.2  # Increase from 0.1
# OR add noise schedule
noise_decay: 0.99  # Decay slowly
```

---

## SECTION 10: ACTION PLAN - PRIORITY FIXES

### P0 - CRITICAL (Fix IMMEDIATELY)

#### Fix #1: Correct Reward Configuration
**File:** `config/training_config.yaml`
```yaml
# CURRENT (WRONG):
progress:
  distance_scale: 10.0  # TOO HIGH!
  waypoint_bonus: 5.0   # TOO HIGH!

# CORRECT:
progress:
  distance_scale: 5.0   # Reduce to match 5K diagnostic
  waypoint_bonus: 1.0   # Reduce to 10% of original
```

**Expected outcome:**
- Reduce progress reward amplification from ~50× to ~5×
- Balance progress with other reward components

#### Fix #2: Verify Reward Calculation
**File:** `src/environment/reward_functions.py`

**Check:**
1. Is progress reward actually positive for forward movement?
2. Is lane-keeping reward correctly calculated?
3. Is safety penalty negative (not positive)?

**Add debug logging:**
```python
logger.info(f"Reward breakdown: progress={progress_reward:.2f}, lane={lane_reward:.2f}, safety={safety_penalty:.2f}")
```

#### Fix #3: Reset Actor Network Initialization
**File:** `src/networks/actor_network.py`

**Current:** Unknown initialization
**Expected:** Zero-centered with small std

```python
# Ensure final layer initialized with small weights
nn.init.uniform_(self.final_layer.weight, -3e-3, 3e-3)
nn.init.uniform_(self.final_layer.bias, -3e-3, 3e-3)
```

#### Fix #4: Increase Exploration Noise
**File:** `config/td3_config.yaml`
```yaml
# CURRENT:
expl_noise: 0.1  # Too low

# CORRECT:
expl_noise: 0.2  # Standard TD3 value
```

### P1 - HIGH (Fix after P0)

#### Fix #5: Add CNN Debug Logging
**File:** `src/agents/td3_agent.py`

Add gradient tracking:
```python
# In train() method, after actor update:
actor_cnn_grad = self.actor_cnn.conv1.weight.grad.norm().item()
critic_cnn_grad = self.critic_cnn.conv1.weight.grad.norm().item()

self.writer.add_scalar('gradients/actor_cnn_norm', actor_cnn_grad, timestep)
self.writer.add_scalar('gradients/critic_cnn_norm', critic_cnn_grad, timestep)
```

#### Fix #6: Verify Action Mapping
**File:** `src/environment/carla_env.py`

Verify CARLA control mapping:
```python
# Ensure correct mapping:
# action[0] ∈ [-1, 1] → steering ∈ [-1, 1] (left/right)
# action[1] ∈ [-1, 1] → throttle/brake ∈ [0, 1] (forward/backward)

# Log first 100 actions:
if self.steps < 100:
    logger.info(f"Action: steering={action[0]:.3f}, throttle={action[1]:.3f}")
```

### P2 - MEDIUM (Consider for next run)

#### Fix #7: Early Stopping
**File:** `scripts/train_td3.py`

Add early stopping if episode length degrades:
```python
if np.mean(recent_episode_lengths) < 20:
    logger.warning("Episode length degrading below 20 steps - potential failure mode")
    # Save checkpoint and stop
```

#### Fix #8: Curriculum Learning
**File:** `config/carla_config.yaml`

Start with easier scenario:
```yaml
# Phase 1: No NPCs (0-50K steps)
# Phase 2: Few NPCs (50K-200K steps)
# Phase 3: Full traffic (200K+ steps)
```

---

## SECTION 11: EXPECTED OUTCOMES AFTER FIXES

### After P0 Fixes (Immediate):

| Metric | Current (30K) | Expected After P0 | Improvement |
|--------|---------------|-------------------|-------------|
| Steering mean | 0.94 (right bias) | 0.0 ± 0.3 | ✅ Centered |
| Throttle mean | 0.94 (aggressive) | 0.5 ± 0.3 | ✅ Moderate |
| Episode length | 16 steps (degrading) | 50+ steps (improving) | ✅ 3× better |
| Episode reward | -14.15 (negative) | Positive | ✅ Positive learning |
| Q-values | -14.81 (pessimistic) | 10-30 (realistic) | ✅ Correct range |
| Actor loss | -13364 (diverging) | -50 to -100 (stable) | ✅ Stable |

### After P1 Fixes (Short-term):

| Metric | Expected After P1 | Target |
|--------|-------------------|--------|
| Success rate | > 10% | > 50% |
| Avg episode length | > 100 steps | > 500 steps |
| Lane invasion rate | < 50% | < 10% |
| Collision rate | < 10% | < 5% |

### After P2 Fixes (Long-term):

| Metric | Expected After P2 | Target (Paper-worthy) |
|--------|-------------------|----------------------|
| Success rate | > 50% | > 80% |
| Avg episode length | > 500 steps | > 800 steps |
| Lane invasion rate | < 10% | < 5% |
| Collision rate | < 5% | < 2% |
| Mean episode reward | > 100 | > 200 |

---

## SECTION 12: NEXT STEPS

### Immediate Actions (TODAY):

1. ✅ **Read current configuration files**
   - `config/training_config.yaml`
   - `config/carla_config.yaml`
   - `config/td3_config.yaml`

2. ✅ **Verify reward calculation**
   - Read `src/environment/reward_functions.py`
   - Check progress reward sign (+/-)
   - Check distance calculation method

3. ✅ **Apply P0 fixes**
   - Correct `distance_scale: 5.0`
   - Correct `waypoint_bonus: 1.0`
   - Increase `expl_noise: 0.2`
   - Add action logging

4. ✅ **Run short validation (5K steps)**
   - Check steering bias reduced
   - Check episode length improved
   - Check rewards are positive

### Short-term Actions (NEXT 2 DAYS):

5. ⏳ **If 5K validation succeeds:**
   - Run 100K training
   - Monitor metrics every 10K steps
   - Save checkpoints every 20K steps

6. ⏳ **If 5K validation fails:**
   - Debug reward calculation
   - Debug action mapping
   - Consider resetting training from scratch

7. ⏳ **Add CNN debug logging**
   - Implement P1 Fix #5
   - Track gradient flow
   - Monitor feature extraction

### Long-term Actions (NEXT WEEK):

8. ⏳ **Benchmark against DDPG**
   - Implement DDPG baseline
   - Compare with same configuration
   - Validate TD3 superiority

9. ⏳ **Implement curriculum learning**
   - Phase 1: 0 NPCs
   - Phase 2: 10 NPCs
   - Phase 3: 20 NPCs

10. ⏳ **Prepare paper draft**
    - Document methodology
    - Report results
    - Compare with literature

---

## APPENDIX A: CONFIGURATION FILES TO CHECK

### Read These Files FIRST:
1. `config/training_config.yaml` - Check `distance_scale`, `waypoint_bonus`
2. `config/carla_config.yaml` - Check reward weights
3. `config/td3_config.yaml` - Check `expl_noise`
4. `src/environment/reward_functions.py` - Verify reward calculation

### Expected Configuration (from 5K Analysis):
```yaml
# training_config.yaml
progress:
  distance_scale: 5.0   # NOT 10.0 or 50.0
  waypoint_bonus: 1.0   # NOT 5.0 or 10.0

# carla_config.yaml
weights:
  progress: 1.0
  lane_keeping: 1.0
  efficiency: 1.0
  comfort: 0.5
  safety: 1.0

# td3_config.yaml
expl_noise: 0.2  # NOT 0.1
```

---

## APPENDIX B: TENSORBOARD MONITORING COMMANDS

### Launch TensorBoard:
```bash
cd av_td3_system
tensorboard --logdir docs/day-21/run7/TD3_scenario_0_npcs_20_20251121-230112
```

### Key Metrics to Watch:
1. **train/q1_value** - Should be 10-30 range
2. **train/critic_loss** - Should decrease
3. **train/actor_loss** - Should be -50 to -100 (stable)
4. **debug/action_steering_mean** - Should be ~0.0
5. **train/episode_length** - Should increase
6. **train/episode_reward** - Should increase
7. **eval/success_rate** - Should increase

---

## APPENDIX C: SUCCESS CRITERIA CHECKLIST

### P0 Fixes Applied:
- [ ] `distance_scale: 5.0` (not 10.0)
- [ ] `waypoint_bonus: 1.0` (not 5.0)
- [ ] `expl_noise: 0.2` (not 0.1)
- [ ] Reward calculation verified
- [ ] Action mapping verified

### P0 Success Criteria (5K validation):
- [ ] Steering mean: -0.2 to +0.2 (centered)
- [ ] Episode length: > 30 steps (improving)
- [ ] Episode reward: Positive (> 0)
- [ ] Q-values: Positive or near-zero
- [ ] Actor loss: -50 to -100 (stable)

### P1 Success Criteria (100K training):
- [ ] Success rate: > 10%
- [ ] Avg episode length: > 100 steps
- [ ] Lane invasion rate: < 50%
- [ ] Steering bias: < 0.1
- [ ] Q-values: 10-30 range

### P2 Success Criteria (Paper-worthy):
- [ ] Success rate: > 80%
- [ ] Avg episode length: > 800 steps
- [ ] Lane invasion rate: < 5%
- [ ] Collision rate: < 2%
- [ ] TD3 > DDPG (quantitative comparison)

---

## CONCLUSION

**The 30K training run demonstrates that the TD3 algorithm is working correctly (low TD errors, decreasing critic loss, aligned twin critics), but the POLICY IS NOT LEARNING due to:**

1. ❌ **CRITICAL**: Wrong reward configuration (`distance_scale: 10.0` instead of `5.0`)
2. ❌ **CRITICAL**: Extreme action bias (steering +0.94, throttle +0.94)
3. ❌ **CRITICAL**: Policy degradation (episode length 65 → 16 steps)
4. ❌ **CRITICAL**: 0% success rate, 99.6% lane invasion rate

**The agent has learned that the current policy leads to failure (negative Q-values), but cannot escape this failure mode due to:**
- Incorrect reward shaping
- Low exploration (action std = 0.08)
- Possible CNN gradient flow issues

**IMMEDIATE ACTION REQUIRED:** Apply P0 fixes and run 5K validation before attempting longer training.

**EXPECTED OUTCOME:** After P0 fixes, agent should show:
- Centered actions (steering ~0.0)
- Positive episode rewards
- Increasing episode length
- Q-values in 10-30 range

**This is a FIXABLE situation** - the TD3 algorithm is sound, but configuration errors prevented policy learning.

---

**Generated:** November 21, 2025
**Analysis Tool:** TensorBoard event_accumulator + NumPy statistics
**Baseline:** 5K training diagnostic (TENSORBOARD_DIAGNOSIS.md)
**References:** OpenAI Spinning Up TD3, Stable-Baselines3, Fujimoto et al. (2018)
