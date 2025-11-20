# 5K Step Training Run - Validation Report
**Date**: November 20, 2025  
**Analysis Script**: `scripts/analyze_5k_run.py`  
**Log Directory**: `data/logs/TD3_scenario_0_npcs_20_20251120-133459`

---

## Executive Summary

🔴 **CRITICAL ISSUES DETECTED - DO NOT PROCEED TO 1M TRAINING**

The 5K step training run reveals **catastrophic Q-value explosion** and **performance degradation**. The system is fundamentally broken and requires immediate fixes before any longer training runs.

### Key Findings:
- ❌ **Q-Value Explosion**: Actor Q-mean = **1,796,760** (should be ~50 at 5K steps)
- ❌ **Performance Degradation**: Episode rewards declining from 721 → 7.6 (negative trend -0.36)
- ❌ **Episode Length Collapse**: From 50 steps → 2 steps (agent dying immediately)
- ⚠️ **CNN Gradient Clipping Failure**: Actor CNN norm 2.42 (>1.0), Critic CNN norm 24.7 (>10.0)

---

## Detailed Analysis

### 1. Q-Value Analysis

#### Metrics Extracted:
- **agent**: 15 metrics
- **alerts**: 2 metrics  
- **debug**: 22 metrics (includes Q-values)
- **eval**: 5 metrics
- **gradients**: 4 metrics
- **progress**: 4 metrics
- **train**: 9 metrics
- **Total**: 61 metrics tracked

#### Actor Q-Values:
```
Initial:  2.29
Final:    1,796,760.12
Mean:     278,984.98 ± 440,759.76
Range:    [2.29, 1,796,760.12]
```

#### Validation Against TD3 Paper (Fujimoto et al., 2018):
- **Paper (Hopper-v1, 0-50K steps)**: Q-values 0 → 500
- **Expected at 5K steps** (10% of 50K): Q-values ~0 → 50
- **Our Result**: Q-values 2 → **1,796,760** ❌

**Severity**: **78,469× HIGHER** than expected! This is a catastrophic Q-value explosion.

#### Actor Q-Value Standard Deviation:
```
Initial:  0.18
Final:    538,780.06
Mean:     84,321.81 ± 141,308.33
Range:    [0.18, 538,780.06]
```

The exploding standard deviation indicates **extreme instability** in Q-value estimates.

---

### 2. Reward Analysis

#### Episode Rewards:
```
Initial:  721.86
Final:    7.63
Mean:     103.99 ± 179.68
Trend:    -0.3619 (negative slope)
```

**Interpretation**: 
- ❌ **CATASTROPHIC PERFORMANCE DEGRADATION**
- Rewards dropped 94.4% (from 721 → 7.6)
- Negative trend = agent is **unlearning** / getting worse over time
- This contradicts the fundamental premise of learning

#### Episode Length:
```
Initial:  50.00 steps
Final:    2.00 steps
Mean:     10.44 ± 46.75
Trend:    -0.0660 (negative slope)
```

**Interpretation**:
- ❌ **EPISODE COLLAPSE**
- Agent surviving only 2 steps at the end (vs 50 initially)
- Suggests agent is:
  - Crashing immediately
  - Making catastrophically bad actions
  - Stuck in local minimum of terrible policy

---

### 3. Loss Analysis

⚠️ **NO LOSS METRICS FOUND IN TENSORBOARD**

This is a **critical diagnostic gap**. The script expected:
- `losses/critic_loss`
- `losses/actor_loss`

Neither was found. This suggests:
1. Losses are being logged under different tag names
2. Loss logging is disabled
3. Training loop is not properly integrated with TensorBoard

**Action Required**: Verify loss logging is working correctly.

---

### 4. Gradient Norm Analysis

#### Actor MLP Gradients:
```
Mean:  0.000107
Max:   0.003839
Status: ✅ PASS (≤ 1.0 limit)
```

#### Actor CNN Gradients:
```
Mean:  2.026425
Max:   2.421187
Status: ❌ FAIL (> 1.0 limit)
```

**Severity**: Actor CNN gradients are **2.42× ABOVE** clipping threshold!

#### Critic MLP Gradients:
```
Mean:  1.906598
Max:   3.592864
Status: ✅ PASS (≤ 10.0 limit)
```

#### Critic CNN Gradients:
```
Mean:  23.304445
Max:   24.689190
Status: ❌ FAIL (> 10.0 limit)
```

**Severity**: Critic CNN gradients are **2.47× ABOVE** clipping threshold!

#### Root Cause Hypothesis:
The CNN gradient clipping is **NOT WORKING** as designed. Possible causes:
1. Clipping applied to wrong layer (only MLP, not CNN)
2. Gradient computation order issue (clipping before/after backprop)
3. Missing `clip_grad_norm_()` call for CNN parameters
4. CNN parameters excluded from optimizer

**Connection to Q-Value Explosion**:
Unclipped CNN gradients → Large CNN weight updates → Visual features diverge → Actor exploits visual artifacts → Q-values explode

---

## Comparison with Baselines

### TD3 Paper (Fujimoto et al., 2018)
**Environment**: MuJoCo Hopper-v1  
**Hyperparameters**:
- batch_size = 100
- γ = 0.99
- τ = 0.005
- actor/critic lr = 1e-3
- policy_freq = 2
- policy_noise = 0.2
- noise_clip = 0.5

**Results (1M steps)**:
- Final reward: ~3,500
- Q-values: ~3,000-4,000 (stable)
- Episode length: 1,000 steps
- Learning curve: Slow initial learning (0-50K), then steady improvement

**Our Results (5K steps)**:
- Final reward: 7.6 (collapsing)
- Q-values: 1,796,760 (exploding)
- Episode length: 2 steps (collapsing)
- Learning curve: **Negative** (degrading performance)

---

### Stable-Baselines3 TD3 Benchmarks

**Environment**: PyBullet (1M steps, 3 seeds)

| Environment | SB3 Reward | Our Reward (5K) | Status |
|------------|------------|-----------------|--------|
| HalfCheetah | 2757 ± 53 | 7.6 (collapsing) | ❌ CATASTROPHIC |
| Ant | 3146 ± 35 | - | - |
| Hopper | 2422 ± 168 | - | - |
| Walker2D | 2184 ± 54 | - | - |

**Hyperparameters** (SB3 defaults):
- batch_size = **256** (NOT 100!)
- γ = 0.99
- τ = 0.005
- learning_rate = 0.001
- buffer_size = 1M

**Our Hyperparameters**:
- batch_size = 256 ✅ (matches SB3)
- γ = **0.9** ❌ (should be 0.99)
- τ = **0.001** ❌ (should be 0.005)
- critic_lr = 3e-4 ❌ (should be 1e-3)

---

## Validation Checklist

| Check | Status | Details |
|-------|--------|---------|
| Q-Value Range | ❌ FAIL | Actor Q = 1,796,760 > 500 (explosion!) |
| Learning Signal | ❌ FAIL | Negative trend (-0.3619) |
| Actor Gradient Clipping | ✅ PASS | Max norm = 0.003839 ≤ 1.0 |
| Critic Gradient Clipping | ✅ PASS | Max norm = 3.592864 ≤ 10.0 |
| **Actor CNN Gradient Clipping** | ❌ FAIL | Max norm = 2.421 > 1.0 |
| **Critic CNN Gradient Clipping** | ❌ FAIL | Max norm = 24.689 > 10.0 |

**Overall Score**: 2/6 PASS (33%)

---

## Root Cause Analysis

### Primary Cause: CNN Gradient Clipping Failure

The **CNN gradients are not being clipped** correctly:
- Actor CNN: 2.42 (should be ≤1.0)
- Critic CNN: 24.69 (should be ≤10.0)

This causes:
1. **Large CNN weight updates** → Visual features become unstable
2. **Actor exploits visual artifacts** → Finds actions with artificially high Q-values
3. **Critic Q-values diverge** → Cannot track actor's exploitative actions
4. **Q-value explosion** → Actor Q-mean reaches 1.8M
5. **Policy collapse** → Agent makes catastrophically bad actions (episode length 2)

### Secondary Causes:

1. **Hyperparameter Mismatch**:
   - γ = 0.9 (too myopic, should be 0.99)
   - τ = 0.001 (targets update too slowly, should be 0.005)
   - critic_lr = 3e-4 (too small, should be 1e-3)

2. **Short Episode Length** (CARLA-specific):
   - CARLA episodes: 2-50 steps (our data)
   - MuJoCo episodes: 1,000 steps
   - With γ=0.9, effective horizon = ~10 steps
   - With γ=0.99, effective horizon = ~100 steps
   - **But our episodes end at 2-50 steps!**
   - This mismatch causes credit assignment problems

3. **Missing Loss Logging**:
   - Cannot validate actor/critic loss trends
   - Cannot verify Bellman error convergence

---

## Recommendations

### 🔴 CRITICAL - DO NOT PROCEED TO 1M TRAINING

The system has **fundamental stability issues** that will only worsen with longer training. Address these issues first:

### Priority 1: Fix CNN Gradient Clipping (IMMEDIATE)

**Action**: Review and fix CNN gradient clipping implementation

Check:
1. Are CNN parameters included in `clip_grad_norm_()` call?
2. Is clipping applied to both actor and critic CNNs?
3. Is clipping threshold correct (1.0 for actor, 10.0 for critic)?
4. Is clipping applied AFTER `.backward()` but BEFORE `.step()`?

**Expected file**: `src/agents/td3_agent.py` or similar

**Verification**:
```python
# Actor CNN clipping
torch.nn.utils.clip_grad_norm_(
    self.actor.cnn.parameters(), 
    max_norm=1.0
)

# Critic CNN clipping
torch.nn.utils.clip_grad_norm_(
    self.critic.cnn.parameters(), 
    max_norm=10.0
)
```

### Priority 2: Fix Hyperparameters to Match Paper

**Changes Required**:

| Hyperparameter | Current | Target | Reason |
|---------------|---------|--------|--------|
| γ (discount) | 0.9 | 0.99 | Match paper, longer horizon |
| τ (target update) | 0.001 | 0.005 | Match paper, faster target tracking |
| critic_lr | 3e-4 | 1e-3 | Match paper, faster critic learning |
| batch_size | 256 | 256 ✅ | Keep (matches SB3) |

**Implementation**:
```python
# config.py or similar
TD3_CONFIG = {
    'gamma': 0.99,          # Was 0.9
    'tau': 0.005,           # Was 0.001
    'critic_lr': 1e-3,      # Was 3e-4
    'actor_lr': 1e-3,       # Verify current value
    'batch_size': 256,      # Keep
    'policy_freq': 2,       # Keep
    'policy_noise': 0.2,    # Keep
    'noise_clip': 0.5,      # Keep
}
```

### Priority 3: Add Loss Logging

**Action**: Ensure actor/critic losses are logged to TensorBoard

**Expected metrics**:
- `losses/critic_loss` (MSE of Bellman error)
- `losses/actor_loss` (negative mean Q-value)
- `losses/critic_q1_mean` (for monitoring)
- `losses/critic_q2_mean` (for monitoring)

### Priority 4: Validate After Fixes

**Test Plan**:
1. Fix CNN gradient clipping
2. Update hyperparameters (γ, τ, critic_lr)
3. Run **5K step test** again
4. Verify:
   - Q-values < 100 ✅
   - Episode rewards increasing (positive trend) ✅
   - Episode length stable (not collapsing) ✅
   - CNN gradients clipped correctly ✅

**Only if ALL checks pass**, proceed to:
- 50K step run (10% of 1M)
- 100K step run (20% of 1M)
- Full 1M step run

---

## Files Generated

1. **Visualization**: `docs/day-20/5k_training_analysis.png`
   - 4-panel plot showing Q-values, rewards, losses, episode length over time
   
2. **CSV Exports**:
   - `docs/day-20/5k_q_values.csv` (Q-value trajectory)
   - `docs/day-20/5k_rewards.csv` (Episode rewards and lengths)

3. **This Report**: `docs/day-20/5K_RUN_VALIDATION_REPORT.md`

---

## Next Steps

### Immediate Actions (Before Any More Training):

1. ✅ **Read this report thoroughly**
2. 🔧 **Fix CNN gradient clipping** (Priority 1)
3. 🔧 **Update hyperparameters** (Priority 2)
4. 🔧 **Add loss logging** (Priority 3)
5. ✅ **Run 5K validation again** (Priority 4)
6. ✅ **Verify all checks pass**

### After Fixes (Incremental Validation):

1. **5K run** (current) → Validate fixes work
2. **50K run** (10× current) → Validate scaling
3. **100K run** (20× current) → Validate convergence trends
4. **1M run** (full paper) → Final evaluation

**DO NOT skip intermediate validation steps!**

---

## References

1. **TD3 Paper**: Fujimoto, S., Hoof, H., & Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. *ICML 2018*.

2. **Stable-Baselines3 Documentation**: https://stable-baselines3.readthedocs.io/en/master/modules/td3.html

3. **Previous Analysis Documents**:
   - `TD3_PAPER_COMPLETE_ANALYSIS.md`
   - `COMPREHENSIVE_TD3_ANALYSIS_POST_PAPER_REVIEW.md`

4. **Related Papers** (to read for CARLA context):
   - End-to-End Race Driving with Deep Reinforcement Learning
   - Interpretable End-to-end Urban Autonomous Driving with Latent Deep Reinforcement Learning
   - Robust Adversarial Attacks Detection based on Explainable Deep Reinforcement Learning For UAV Guidance and Planning

---

## Conclusion

The 5K training run reveals **catastrophic system instability** due to:
1. **CNN gradient clipping failure** (primary cause)
2. **Hyperparameter mismatches** (secondary cause)
3. **Missing loss logging** (diagnostic gap)

**RECOMMENDATION**: 🔴 **DO NOT PROCEED TO 1M TRAINING**

Fix the gradient clipping, update hyperparameters, and validate with another 5K run before attempting longer training runs.

---

**Analysis Date**: November 20, 2025 11:38:03  
**Analysis Script**: `scripts/analyze_5k_run.py`  
**TensorBoard Log**: `data/logs/TD3_scenario_0_npcs_20_20251120-133459`
