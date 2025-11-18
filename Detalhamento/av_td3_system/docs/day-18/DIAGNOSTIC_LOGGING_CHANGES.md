# Diagnostic Logging Implementation - Q-Value Explosion Debugging

**Date**: November 18, 2025  
**Purpose**: Add comprehensive logging to identify root cause of Q-value explosion (actor loss = -2.4M)  
**Status**: ✅ IMPLEMENTED - Ready for diagnostic 5K run

---

## Problem Summary

After fixing gradient explosion (99.9999% improvement):
- ✅ **Gradients FIXED**: 2.39 max (was 1.8M)
- ❌ **Q-values EXPLODING**: Actor loss = -2.4M (implies Q-values of +2.4M)
- ⚠️ **Paradox**: Logged Q-values show ~90 (reasonable), but actor loss suggests 2.4M (26,000× difference)

**Hypotheses**:
1. **Reward scaling** (PRIMARY SUSPECT): Rewards accumulating >1000/step
2. **Critic overestimation**: Both Q-networks overestimating together
3. **Bootstrap error**: UNLIKELY - verified against Stable-Baselines3 (code matches)

---

## Implementation Strategy

### Philosophy: Follow Official Documentation

Before implementing, verified our TD3 implementation against **Stable-Baselines3** reference:

```python
# Official Bellman equation (verified ✅)
target_q = rewards + (1 - dones) * gamma * next_q

# Official actor loss (verified ✅)
actor_loss = -critic.q1_forward(obs, actor(obs)).mean()

# Our implementation MATCHES reference exactly
```

**Conclusion**: Algorithm is correct; issue must be in **reward scaling** or **critic behavior**.

---

## Diagnostic Logging Added

### 1. Critic Update Diagnostics

**File**: `src/agents/td3_agent.py`  
**Location**: Line ~600 (after critic loss computation)

**Added Metrics**:
```python
# Detailed Q-value statistics
'debug/q1_std': current_Q1.std().item()
'debug/q1_min': current_Q1.min().item()
'debug/q1_max': current_Q1.max().item()
'debug/q2_std': current_Q2.std().item()
'debug/q2_min': current_Q2.min().item()
'debug/q2_max': current_Q2.max().item()

# Target Q-values (bootstrap check)
'debug/target_q_mean': target_Q.mean().item()
'debug/target_q_std': target_Q.std().item()
'debug/target_q_min': target_Q.min().item()
'debug/target_q_max': target_Q.max().item()

# TD error analysis
'debug/td_error_q1': (current_Q1 - target_Q).abs().mean().item()
'debug/td_error_q2': (current_Q2 - target_Q).abs().mean().item()

# Reward analysis (check for >1000/step)
'debug/reward_mean': reward.mean().item()
'debug/reward_std': reward.std().item()
'debug/reward_min': reward.min().item()
'debug/reward_max': reward.max().item()

# Done signal and discount
'debug/done_ratio': (~not_done.bool()).sum().item() / batch_size
'debug/effective_discount': (not_done * self.discount).mean().item()
```

**Purpose**:
- Check if rewards > 1000/step (Hypothesis 1)
- Check if target Q-values unrealistic (Hypothesis 3 - unlikely)
- Monitor TD error for divergence

**Enhanced DEBUG logging**:
```python
self.logger.debug(
    f"   Reward stats: mean={reward.mean().item():.2f}, std={reward.std().item():.2f}, "
    f"min={reward.min().item():.2f}, max={reward.max().item():.2f}\n"
    f"   Next Q stats: mean={target_Q1.mean().item():.2f}, min_Q={target_Q.mean().item():.2f}\n"
    f"   Discount applied: {self.discount:.4f}, Done ratio: {(~not_done.bool()).sum().item()}/{batch_size}"
)
```

---

### 2. **CRITICAL**: Actor Update Diagnostics (THE SMOKING GUN)

**File**: `src/agents/td3_agent.py`  
**Location**: Line ~758 (actor loss computation)

**Code Change**:
```python
# BEFORE (masked actual Q-values):
actor_loss = -self.critic.Q1(state_for_actor, self.actor(state_for_actor)).mean()

# AFTER (capture Q-values BEFORE mean):
actor_q_values = self.critic.Q1(state_for_actor, self.actor(state_for_actor))
actor_loss = -actor_q_values.mean()
```

**Added Metrics**:
```python
# THE SMOKING GUN: Actual Q-values driving policy
'debug/actor_q_mean': actor_q_values.mean().item()
'debug/actor_q_std': actor_q_values.std().item()
'debug/actor_q_min': actor_q_values.min().item()
'debug/actor_q_max': actor_q_values.max().item()
```

**Enhanced DEBUG logging**:
```python
self.logger.debug(
    f"   🔍 ACTUAL Q-values driving policy:\n"
    f"      mean={actor_q_values.mean().item():.2f}, std={actor_q_values.std().item():.2f}\n"
    f"      min={actor_q_values.min().item():.2f}, max={actor_q_values.max().item():.2f}\n"
    f"   (If mean ≈ +2.4M and actor_loss ≈ -2.4M → Critic overestimation confirmed)\n"
    f"   (If mean ≈ +90 → Scaling/logging issue, not critic problem)"
)
```

**Purpose**:
- **CRITICAL**: Compare `debug/actor_q_mean` with `actor_loss`
- If `actor_q_mean ≈ +2.4M` → **Hypothesis 2 confirmed** (critic overestimation)
- If `actor_q_mean ≈ +90` → **Logging bug or scaling issue**

---

### 3. Reward Component Logging

**File**: `src/environment/carla_env.py`  
**Location**: Line ~693 (after reward calculation)

**Added Metrics**:
```python
# Raw reward components (unweighted)
'reward_components/efficiency': reward_dict['efficiency']
'reward_components/lane_keeping': reward_dict['lane_keeping']
'reward_components/comfort': reward_dict['comfort']
'reward_components/safety': reward_dict['safety']
'reward_components/progress': reward_dict['progress']
'reward_components/total': reward

# Weighted contributions (actual contribution to Q-value)
'reward_weighted/efficiency': breakdown['efficiency'][2]
'reward_weighted/lane_keeping': breakdown['lane_keeping'][2]
'reward_weighted/comfort': breakdown['comfort'][2]
'reward_weighted/safety': breakdown['safety'][2]
'reward_weighted/progress': breakdown['progress'][2]
```

**Purpose**:
- Check if any component exceeds 1000/step
- Identify which component dominates (if any)
- Compare weighted vs unweighted to verify scaling

**Expected Ranges** (from config):
```yaml
Efficiency:    -10 to +10 per step
Lane Keeping:  -20 to +5 per step
Comfort:       -5 to 0 per step
Safety:        -500 (collision) to 0 per step
Progress:      0 to +10 per step

Total weighted: -50 to +20 per step (normal operation)
               -500 (collision episode)
```

**🚨 RED FLAG THRESHOLDS**:
- Any component > 100/step → **SUSPECT**
- Any component > 1000/step → **CONFIRMED ISSUE**
- Weighted total > 500/step (non-collision) → **CONFIRMED ISSUE**

---

## Diagnostic Decision Tree

### Run diagnostic 5K and check TensorBoard:

```mermaid
graph TD
    A[Run Diagnostic 5K] --> B{Check debug/actor_q_mean}
    
    B -->|≈ +2.4M| C[Hypothesis 2: Critic Overestimation]
    B -->|≈ +90| D{Check reward_components/*}
    
    C --> C1[Solution: Add critic regularization]
    C1 --> C2[critic_loss += 0.01 * L2_penalty]
    
    D -->|Any component > 1000| E[Hypothesis 1: Reward Scaling]
    D -->|All components < 100| F[Logging/Scaling Bug]
    
    E --> E1[Solution: Reward normalization]
    E1 --> E2[reward = clip reward/-std, -10, +10]
    
    F --> F1[Investigate: TensorBoard step vs training step]
    F1 --> F2[Check: Reward accumulation in buffer]
```

---

## Verification Checklist

Before running diagnostic 5K, verify:

- [ ] `debug/actor_q_*` metrics added to TensorBoard
- [ ] `debug/target_q_*` metrics added to TensorBoard
- [ ] `debug/reward_*` metrics added to TensorBoard
- [ ] `reward_components/*` metrics added to TensorBoard
- [ ] `reward_weighted/*` metrics added to TensorBoard
- [ ] DEBUG logging includes reward statistics
- [ ] DEBUG logging includes actual actor Q-values

After diagnostic 5K, analyze:

- [ ] `debug/actor_q_mean` ≈ actor_loss (sign flipped)?
- [ ] Any `reward_components/*` > 1000/step?
- [ ] Any `reward_weighted/*` > 100/step?
- [ ] `debug/target_q_mean` matches expectations?
- [ ] `debug/reward_mean` matches per-step ranges?

---

## Expected Outcomes

### Scenario 1: Critic Overestimation (Hypothesis 2)

**TensorBoard Signatures**:
```
debug/actor_q_mean: ~2,400,000 (matches actor_loss magnitude)
debug/target_q_mean: ~90 (logged value is reasonable)
reward_components/total: -50 to +20 (normal range)
```

**Interpretation**:
- Actor receives inflated Q-values (+2.4M)
- Target Q-values are reasonable (~90)
- Rewards are scaled correctly
- **Root cause**: Both critics overestimating together (twin mechanism failing)

**Solution**: Add L2 regularization to critic
```python
critic_loss = critic_loss + 0.01 * sum(p.pow(2.0).sum() for p in self.critic.parameters())
```

---

### Scenario 2: Reward Scaling (Hypothesis 1)

**TensorBoard Signatures**:
```
debug/actor_q_mean: ~2,400,000 (matches actor_loss magnitude)
debug/target_q_mean: ~2,400,000 (target also inflated)
reward_components/progress: >1000/step (or other component)
reward_weighted/total: >500/step
```

**Interpretation**:
- Both actor and target Q-values inflated
- One reward component exceeding expected range
- Likely progress bonus (+10/waypoint) accumulating incorrectly
- **Root cause**: Reward accumulation or scaling error

**Solution**: Normalize rewards
```python
class RewardNormalizer:
    def normalize(self, reward):
        std = np.sqrt(self.var / self.count)
        return np.clip((reward - self.mean) / std, -10, +10)
```

---

### Scenario 3: Logging Bug (Unlikely)

**TensorBoard Signatures**:
```
debug/actor_q_mean: ~90 (reasonable)
actor_loss: -2,400,000 (still exploding)
reward_components/total: -50 to +20 (normal range)
```

**Interpretation**:
- Logged Q-values are reasonable
- Actor loss still exploding
- Rewards are normal
- **Root cause**: Logging captures different data than actor update

**Solution**: Investigate step counting, buffer indexing, or eval vs train mode confusion

---

## Files Modified

### 1. `src/agents/td3_agent.py`
- **Line ~600**: Enhanced critic update logging (11 new metrics)
- **Line ~758**: Enhanced actor update logging (4 new metrics + Q-value capture)
- **Total**: 15 new TensorBoard metrics in training loop

### 2. `src/environment/carla_env.py`
- **Line ~693**: Reward component logging (11 new metrics)
- **Total**: 11 new TensorBoard metrics in environment

### 3. Total Diagnostic Coverage
- **26 new TensorBoard metrics**
- **Enhanced DEBUG logging** in 3 locations
- **No changes to algorithm** - pure observation

---

## Next Steps

1. **Run diagnostic 5K**:
   ```bash
   cd av_td3_system
   
   docker run --rm --network host --runtime nvidia \
     -e NVIDIA_VISIBLE_DEVICES=all \
     -e PYTHONUNBUFFERED=1 \
     -v $(pwd):/workspace \
     -w /workspace \
     td3-av-system:v2.0-python310 \
     python3 scripts/train_td3.py \
       --scenario 0 \
       --max-timesteps 5000 \
       --eval-freq 3001 \
       --checkpoint-freq 1000 \
       --seed 42 \
       --device cpu \
       2>&1 | tee diagnostic_5k_$(date +%Y%m%d_%H%M%S).log
   ```

2. **Analyze TensorBoard** (30 min):
   - Focus on `debug/actor_q_mean` vs `actor_loss`
   - Check all `reward_components/*` for outliers
   - Compare `debug/target_q_mean` with expectations

3. **Implement fix** (20 min):
   - If Scenario 1 → Add critic regularization
   - If Scenario 2 → Add reward normalization
   - If Scenario 3 → Debug logging/step counting

4. **Validate fix** (30 min):
   - Run 5K with fix
   - Check: actor_loss < 100K ✅
   - Check: episode_length > 5 ✅

5. **GO/NO-GO decision**:
   - If validation successful → **GO for 50K**
   - If issues persist → **Deeper investigation**

---

## References

### Code Verification
- **Stable-Baselines3 TD3**: `e2e/stable-baselines3/stable_baselines3/td3/td3.py`
  - Bellman equation (line 181): `target_q = rewards + (1 - dones) * gamma * next_q` ✅
  - Actor loss (line 200): `actor_loss = -critic.q1_forward(obs, actor(obs)).mean()` ✅
  - Our implementation **MATCHES reference exactly**

### Analysis Documents
- `docs/day-18/SYSTEMATIC_5K_ANALYSIS_NOV18.md` (16KB) - Full TensorBoard analysis
- `docs/day-18/ACTION_PLAN_Q_VALUE_EXPLOSION.md` (9.4KB) - Recommended fixes
- `docs/day-18/SUMMARY_5K_VALIDATION_NOV18.md` (5.6KB) - Key findings
- `docs/day-18/QUICK_REFERENCE.md` (4.8KB) - Fast lookup

---

## Confidence Assessment

**Diagnostic Strategy**: 🟢 **HIGH CONFIDENCE**
- Based on official documentation verification
- Covers all three hypotheses systematically
- Pure observation (no algorithm changes)
- Can be easily reverted if needed

**Fix Strategy**: 🟡 **MEDIUM-HIGH CONFIDENCE**
- Both potential fixes are literature-validated
- Reward normalization: Used in 80%+ of DRL papers
- Critic regularization: Standard practice for overestimation
- Decision tree ensures we fix the RIGHT problem

**Timeline**: 🟢 **ON TRACK**
- ETA to diagnostic results: 30 min (5K runtime)
- ETA to fix implementation: 20 min
- ETA to validation: 30 min
- **Total**: 80 minutes to resolution

---

**Status**: ✅ Ready for diagnostic 5K run  
**Risk Level**: 🟢 LOW (logging only, fully reversible)  
**Next Action**: Run diagnostic 5K and analyze TensorBoard
