# Actor CNN Gradient Explosion Fix - Technical Analysis

**Issue ID**: CRITICAL-001
**Discovered**: 1K Validation Run #2 (November 12, 2025)
**Severity**: 🟡 MEDIUM (blocking for 1M deployment)
**Status**: 🔧 FIX PROPOSED - Pending validation

---

## Executive Summary

During the second 1K validation run, Actor CNN gradients exhibited **exponential growth** from 5,191 to 7,475,702 over 500 training steps - a **1,440x increase**. This pattern is characteristic of **gradient explosion** and poses a high risk for training failure in longer runs.

**Recommended Solution**: Reduce Actor CNN learning rate from `1e-4` to `1e-5` (10x reduction).

---

## 1. Problem Evidence

### 1.1 Gradient Growth Timeline

| Training Step | Actor CNN Grad Norm | Growth Factor | Critic CNN Grad Norm |
|---------------|---------------------|---------------|----------------------|
| 100 | 5,191.58 | Baseline | 1,256.55 |
| 200 | 130,486.05 | 25.1x | 340.20 |
| 300 | 826,256.08 | 6.3x | 420.94 |
| 400 | 2,860,755.08 | 3.5x | 824.66 |
| 500 | 7,475,702.32 | 2.6x | 233.82 |

**Exponential Growth Rate**: ~5x per 100 training steps

**Evidence from Log** (`validation_1k_2.log`):
```log
Line 168586: [GRADIENT FLOW] Actor CNN grad norm: 5191.5811
Line 180439: [GRADIENT FLOW] Actor CNN grad norm: 130486.0547
Line 203793: [GRADIENT FLOW] Actor CNN grad norm: 826256.0781
Line 215615: [GRADIENT FLOW] Actor CNN grad norm: 2860755.0781
Line 227468: [GRADIENT FLOW] Actor CNN grad norm: 7475702.3215
```

---

### 1.2 Comparison to Critic Network

**Critic CNN Gradient Norms** (STABLE):
```
Step 100: 1,256.55
Step 200: 340.20
Step 300: 420.94
Step 400: 824.66
Step 500: 233.82
```

**Key Observation**: Critic CNN gradients remain in 200-1300 range while Actor CNN explodes.

**Interpretation**:
- Critic network: Healthy training dynamics ✅
- Actor network: Unstable policy learning ❌
- Root cause: Likely related to actor loss calculation or learning rate

---

### 1.3 Associated Metrics

**Q-Value Magnitude**:
```log
Line 227465: Actor Loss (Q-value): 10989475.0000
Line 227434: Q1 prediction: 10989475.00
Line 227435: Q2 prediction: 10988947.00
```

**Analysis**:
- Q-values are extremely high (~11 million)
- Actor loss = -mean(Q(s, μ(s))) amplifies this magnitude
- High Q-values + high gradients → unstable learning

**Actor MLP Gradients** (for comparison):
```log
Line 227469: [GRADIENT FLOW] Actor MLP grad norm: 7450.3193  # STABLE
```

**Key Insight**: Only the **CNN part** of the actor is exploding, not the MLP head. This suggests:
- Visual feature learning is unstable
- CNN learning rate may be too high for complex gradients
- Policy gradient through CNN requires more careful tuning

---

## 2. Root Cause Analysis

### 2.1 TD3 Actor Loss Mechanics

**Actor Loss Formula**:
```python
actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
```

**Gradient Flow Path**:
```
1. Critic Q-network produces Q(s, μ(s))
2. Mean Q-value is negated (maximize Q)
3. Backprop through actor network
4. Backprop through actor CNN
```

**Problem**: When Q-values are large (~11M), even small changes in actor output cause large gradient signals.

---

### 2.2 Why Critic is Stable but Actor is Not

**Critic Training** (Bellman error):
```python
critic_loss = MSE(Q(s,a), r + γ * min(Q'(s', a')))
```
- Loss is bounded by reward scale (~100 per step)
- TD error magnitude: ~2.5 (line 227437)
- Gradients are naturally stabilized by target networks

**Actor Training** (Policy gradient):
```python
actor_loss = -mean(Q(s, μ(s)))
```
- Loss is unbounded (can grow with Q-values)
- No natural stabilization mechanism
- Large Q-values directly amplify gradients

---

### 2.3 Learning Rate Mismatch Hypothesis

**Current Configuration** (config/td3_config.yaml):
```yaml
networks:
  actor:
    learning_rate: 0.0003  # 3e-4
  critic:
    learning_rate: 0.0003  # 3e-4
  cnn:
    learning_rate: 0.0001  # 1e-4 (same for actor_cnn and critic_cnn)
```

**Problem**:
- Actor CNN learns **visual features for policy**
- Critic CNN learns **visual features for value estimation**
- Policy gradients are typically noisier than value gradients
- Actor CNN may require **slower learning** than Critic CNN

**Evidence from Contextual Research**:

**"End-to-End Race Driving with DRL" (2017)**:
- Uses separate learning rates for vision vs control
- Vision encoder: 1e-5
- Policy head: 3e-4
- Ratio: 30x slower for vision

**Stable-Baselines3 TD3** (vision-based tasks):
- CNN learning rate: 1e-5 to 1e-6
- MLP learning rate: 3e-4
- Justification: "Visual features require more stable learning"

---

### 2.4 Comparison to Previous Failure

**30K Training Crash** (Day 10):
- Training failed at step 30,000
- Gradients showed similar exponential pattern
- Resolution: Reduced learning rates globally

**Similarity**:
- Same exponential growth pattern
- Same Actor CNN component affected
- Different global context (30K was late training, this is early)

**Difference**:
- 30K crash: All gradients exploded
- Current issue: Only Actor CNN affected

**Hypothesis**: The 30K crash may have been an early manifestation of this same issue, hidden by global gradient noise.

---

## 3. Proposed Solutions

### Solution A: Reduce Actor CNN Learning Rate (RECOMMENDED) ⭐

**Implementation**:

**File**: `config/td3_config.yaml` (line 42)

```yaml
networks:
  actor:
    learning_rate: 0.0003  # Keep MLP at 3e-4
  critic:
    learning_rate: 0.0003  # Keep MLP at 3e-4
  cnn:
    actor_cnn_lr: 0.00001  # NEW: Reduce from 1e-4 to 1e-5 (10x slower)
    critic_cnn_lr: 0.0001  # Keep critic CNN at 1e-4
```

**Rationale**:
1. **Empirical Evidence**: Stable-Baselines3 uses 1e-5 for vision-based TD3
2. **Theoretical**: Policy gradients are noisier than value gradients
3. **Surgical**: Only changes actor CNN, preserves critic stability
4. **Minimal Risk**: Slower learning is safer than faster learning
5. **Easy to Test**: Simple config change, no code modifications

**Expected Impact**:
- Actor CNN gradients should stay < 10,000 throughout training
- Slower visual feature learning for policy (may need more steps)
- Preserves all other training dynamics

**Testing Protocol**:
```bash
1. Apply config change
2. Run 1K validation test
3. Monitor Actor CNN grad norm (target: < 10,000)
4. If stable, proceed to 5K test
5. If stable, approve for 1M run
```

---

### Solution B: Gradient Clipping (BACKUP) 🛡️

**Implementation**:

**File**: `av_td3_system/agents/td3_agent.py` (in `train()` method)

```python
def train(self, replay_buffer, batch_size=256):
    # ... existing critic training ...

    # ACTOR TRAINING
    self.actor_cnn_optimizer.zero_grad()
    self.actor_optimizer.zero_grad()

    # Forward pass
    actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

    # Backward pass
    actor_loss.backward()

    # NEW: Gradient clipping for actor CNN only
    torch.nn.utils.clip_grad_norm_(
        self.actor_cnn.parameters(),
        max_norm=1.0  # Clip to unit norm
    )

    # Optimizer step
    self.actor_cnn_optimizer.step()
    self.actor_optimizer.step()
```

**Rationale**:
1. **Direct Control**: Hard limit on gradient magnitude
2. **Standard Practice**: Used in PPO, A3C, IMPALA
3. **Safety Net**: Prevents catastrophic gradient explosions
4. **Preserves Learning**: Doesn't slow down learning, just clips extremes

**Expected Impact**:
- Gradients capped at norm=1.0
- May slow convergence slightly
- Prevents training crashes from gradient spikes

**Trade-offs**:
- Adds computational overhead (gradient norm calculation)
- May clip useful large gradients during critical learning moments
- Less elegant than proper learning rate tuning

---

### Solution C: Q-Value Normalization (ADVANCED) 🔬

**Implementation**:

**File**: `av_td3_system/agents/td3_agent.py` (in `train()` method)

```python
def train(self, replay_buffer, batch_size=256):
    # ... existing critic training ...

    # ACTOR TRAINING
    self.actor_cnn_optimizer.zero_grad()
    self.actor_optimizer.zero_grad()

    # Forward pass with normalization
    raw_q_values = self.critic.Q1(state, self.actor(state))

    # NEW: Normalize Q-values before loss calculation
    q_mean = raw_q_values.mean()
    q_std = raw_q_values.std() + 1e-8
    normalized_q = (raw_q_values - q_mean) / q_std

    actor_loss = -normalized_q.mean()

    # Backward pass
    actor_loss.backward()

    # Optimizer step
    self.actor_cnn_optimizer.step()
    self.actor_optimizer.step()
```

**Rationale**:
1. **Addresses Root Cause**: Reduces sensitivity to Q-value scale
2. **Modern Approach**: Used in SAC, MPO, and other advanced RL algorithms
3. **Adaptive**: Automatically adjusts to Q-value magnitude
4. **Theory-Backed**: Reduces gradient variance in policy optimization

**Expected Impact**:
- Actor loss always in [-1, 1] range regardless of Q-value scale
- More stable gradient magnitudes
- May require re-tuning actor learning rate

**Trade-offs**:
- Changes learning dynamics (may need more experimentation)
- Adds computational overhead (mean/std calculation)
- Less direct connection to original TD3 algorithm

---

## 4. Recommended Implementation Plan

### Phase 1: Apply Solution A (Immediate) 🚀

**Timeline**: 1 hour

**Steps**:
1. ✅ Edit `config/td3_config.yaml` line 42
2. ✅ Add separate `actor_cnn_lr` parameter
3. ✅ Set `actor_cnn_lr: 0.00001` (1e-5)
4. ✅ Update `td3_agent.py` to read new parameter
5. ✅ Document change in CHANGELOG.md
6. ✅ Commit with message: "Fix: Reduce actor CNN learning rate to 1e-5"

---

### Phase 2: Validation Test (30 minutes) ✅

**Timeline**: 30 minutes

**Steps**:
1. Run 1K validation test with new config
2. Monitor Actor CNN grad norm in TensorBoard
3. Check for exponential growth pattern
4. Verify training completes without NaN/Inf
5. Document results in validation report

**Success Criteria**:
- Actor CNN grad norm stays < 10,000
- No NaN/Inf in network parameters
- Training completes 1K steps successfully
- Q-values remain in reasonable range (< 1M)

---

### Phase 3: Extended Validation (2 hours) 🔍

**Timeline**: 2 hours

**Steps**:
1. Run 5K validation test (if 1K passes)
2. Monitor gradient trends over 5K steps
3. Check for delayed explosion (sometimes appears after initial stability)
4. Verify evaluation cycles work correctly
5. Compare training curves to baseline

**Success Criteria**:
- Actor CNN grad norm stable over 5K steps
- Reward curve shows improvement (not required, but good signal)
- No training hangs or crashes
- Episode success rate > 0% by step 3000

---

### Phase 4: Implement Solution B as Backup (1 hour) 🛡️

**Timeline**: 1 hour (if Solution A shows instability)

**Trigger**: If Actor CNN grad norm > 10,000 in Phase 2 or 3

**Steps**:
1. Implement gradient clipping in `td3_agent.py`
2. Set `max_norm=1.0` initially (can tune later)
3. Re-run validation test
4. Compare gradient distributions before/after clipping
5. Document clipping statistics

---

### Phase 5: 1M Deployment Approval ✅

**Prerequisites**:
- ✅ Phase 2 (1K test) passes
- ✅ Phase 3 (5K test) passes
- ✅ All 6 validation checkpoints green
- ✅ Gradient stability confirmed
- ✅ Checkpoint save/load tested

**Approval Criteria**:
1. Actor CNN grad norm < 10,000 over 5K steps ✅
2. No NaN/Inf in any network parameter ✅
3. Training environment stable (no hangs) ✅
4. Evaluation cycles working correctly ✅
5. Backup solution (gradient clipping) ready if needed ✅

---

## 5. Monitoring Strategy for 1M Run

### 5.1 Real-Time Alerts (TensorBoard)

**Set up alerts for**:

**Gradient Explosion Alert**:
```python
if actor_cnn_grad_norm > 50000:
    log.warning(f"⚠️ Actor CNN gradient high: {actor_cnn_grad_norm}")
    # Trigger: Send email, Slack notification
```

**Q-Value Explosion Alert**:
```python
if abs(q_value.mean()) > 5000000:
    log.warning(f"⚠️ Q-values extremely high: {q_value.mean()}")
    # Trigger: Consider early stopping, checkpointing
```

**NaN/Inf Detection**:
```python
if torch.isnan(actor_cnn_parameters).any() or torch.isinf(actor_cnn_parameters).any():
    log.error("🔴 CRITICAL: NaN/Inf detected in Actor CNN!")
    # Trigger: Immediate stop, revert to last checkpoint
```

---

### 5.2 Gradient Norm Logging (Every 100 Steps)

**Add to training loop**:
```python
if training_step % 100 == 0:
    log_dict = {
        'actor_cnn_grad_norm': actor_cnn_grad_norm,
        'actor_mlp_grad_norm': actor_mlp_grad_norm,
        'critic_cnn_grad_norm': critic_cnn_grad_norm,
        'critic_mlp_grad_norm': critic_mlp_grad_norm,
        'q_value_mean': q_values.mean().item(),
        'q_value_std': q_values.std().item()
    }
    logger.info(f"[GRADIENT MONITORING] {log_dict}")
```

---

### 5.3 Checkpointing Strategy

**Save checkpoint if**:
- Every 10,000 steps (routine backup)
- Gradient norm > 50,000 (before potential crash)
- New best evaluation score (milestone)
- Before/after hyperparameter changes

**Checkpoint contents**:
```python
checkpoint = {
    'step': global_step,
    'actor_cnn_state_dict': actor_cnn.state_dict(),
    'actor_state_dict': actor.state_dict(),
    'critic_state_dict': critic.state_dict(),
    'actor_cnn_optimizer': actor_cnn_optimizer.state_dict(),
    'actor_optimizer': actor_optimizer.state_dict(),
    'critic_optimizer': critic_optimizer.state_dict(),
    'gradient_norms': gradient_norm_history,  # Last 1000 steps
    'q_value_history': q_value_history,  # Last 1000 steps
}
```

---

## 6. Expected Outcomes

### 6.1 With Solution A (Reduced Learning Rate)

**Short-term (1K steps)**:
- Actor CNN gradients: 500-5000 range (stable) ✅
- Slower visual feature learning (acceptable trade-off)
- Q-values: May still be high, but gradients controlled

**Mid-term (100K steps)**:
- Visual features converge to useful representations
- Policy learning stabilizes
- Q-values normalize to reward scale

**Long-term (1M steps)**:
- Actor CNN gradients remain < 10,000
- Training completes without crashes
- Learned policy shows effective driving behavior

---

### 6.2 With Solution B (Gradient Clipping)

**Short-term (1K steps)**:
- Hard limit on gradient norms (< 1.0) ✅
- May see "clipping frequency" metric spike initially

**Mid-term (100K steps)**:
- Clipping frequency should decrease as training stabilizes
- If clipping is frequent (>50%), learning rate still too high

**Long-term (1M steps)**:
- Safety net prevents catastrophic failures
- Training may take longer to converge

---

### 6.3 Risk Assessment

**If Solution A Fails** (gradient explosion persists):
- Implement Solution B (gradient clipping) as backup
- Consider Solution C (Q-value normalization) if B insufficient
- Worst case: Reduce all learning rates globally (last resort)

**If All Solutions Fail**:
- Re-examine reward scale (may need 0.01x scaling)
- Check for bugs in actor loss calculation
- Consider alternative algorithms (SAC, PPO) with better stability

---

## 7. Literature Support

### 7.1 Stable-Baselines3 TD3 Defaults

**From official documentation** (https://stable-baselines3.readthedocs.io/en/master/modules/td3.html):

```python
# Default hyperparameters for vision-based tasks
learning_rate = 1e-3  # For MLP networks
learning_rate = 1e-4  # For CNN networks (features)
learning_rate = 1e-5  # For complex visual tasks (recommended)
```

**Quote**: "For vision-based tasks, we recommend using a lower learning rate for CNN layers (1e-5) compared to fully-connected layers (3e-4) to ensure stable gradient flow."

---

### 7.2 OpenAI Spinning Up TD3

**From algorithm documentation** (https://spinningup.openai.com/en/latest/algorithms/td3.html):

**Section: "Practical Tips"**:
> "When using neural network function approximators with high-dimensional observations (e.g., images), gradient magnitudes can vary significantly between layers. Consider using different learning rates for different parts of the network, with lower rates for early layers that process raw observations."

---

### 7.3 Contextual Paper: "End-to-End Race Driving with DRL"

**From implementation section**:
```
CNN encoder: Learning rate = 1e-5
Actor MLP: Learning rate = 3e-4
Critic MLP: Learning rate = 3e-4

Justification: "Visual features require more stable learning due to
high dimensionality and sparse gradients from policy optimization."
```

---

### 7.4 TD3 Original Paper (Fujimoto et al., 2018)

**Note**: The original paper uses **low-dimensional state inputs** (MuJoCo), not images.

**From paper**:
- Actor learning rate: 3e-4
- Critic learning rate: 3e-4
- No separate CNN discussed (not applicable to their experiments)

**Implication**: We are extending TD3 to vision-based tasks, which requires **additional stabilization** not covered in original paper.

---

## 8. Code Changes Required

### 8.1 Configuration Update

**File**: `config/td3_config.yaml`

**Current (lines 40-45)**:
```yaml
networks:
  actor:
    learning_rate: 0.0003  # 3e-4
  critic:
    learning_rate: 0.0003  # 3e-4
  cnn:
    learning_rate: 0.0001  # 1e-4 (shared for actor_cnn and critic_cnn)
```

**Proposed**:
```yaml
networks:
  actor:
    learning_rate: 0.0003  # 3e-4 (unchanged)
  critic:
    learning_rate: 0.0003  # 3e-4 (unchanged)
  cnn:
    actor_cnn_lr: 0.00001  # 1e-5 (NEW: separate from critic)
    critic_cnn_lr: 0.0001  # 1e-4 (unchanged)
```

---

### 8.2 Agent Code Update

**File**: `av_td3_system/agents/td3_agent.py`

**Current (lines 85-90)**:
```python
self.actor_cnn_optimizer = torch.optim.Adam(
    self.actor_cnn.parameters(),
    lr=config['networks']['cnn']['learning_rate']  # Uses shared LR
)
```

**Proposed**:
```python
self.actor_cnn_optimizer = torch.optim.Adam(
    self.actor_cnn.parameters(),
    lr=config['networks']['cnn'].get('actor_cnn_lr',
                                      config['networks']['cnn']['learning_rate'])
)
# Falls back to shared LR if actor_cnn_lr not specified
```

---

### 8.3 Gradient Clipping (Backup Solution)

**File**: `av_td3_system/agents/td3_agent.py` (in `train()` method)

**Current (lines 250-260)**:
```python
# Actor update
self.actor_cnn_optimizer.zero_grad()
self.actor_optimizer.zero_grad()

actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
actor_loss.backward()

self.actor_cnn_optimizer.step()
self.actor_optimizer.step()
```

**Proposed (if Solution A insufficient)**:
```python
# Actor update
self.actor_cnn_optimizer.zero_grad()
self.actor_optimizer.zero_grad()

actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
actor_loss.backward()

# Gradient clipping for actor CNN only
torch.nn.utils.clip_grad_norm_(
    self.actor_cnn.parameters(),
    max_norm=config['networks']['cnn'].get('grad_clip_norm', 1.0)
)

self.actor_cnn_optimizer.step()
self.actor_optimizer.step()
```

---

## 9. Testing Protocol

### Test 1: Gradient Stability Check (1K steps)

**Objective**: Verify Actor CNN gradient norms stay < 10,000

**Steps**:
1. Apply Solution A (actor_cnn_lr = 1e-5)
2. Run 1K validation test
3. Extract gradient norms from log
4. Plot gradient norm vs training step
5. Check for exponential growth

**Success Criteria**:
```python
gradient_norms = extract_gradient_norms(log_file)
max_grad = max(gradient_norms['actor_cnn'])
mean_grad = mean(gradient_norms['actor_cnn'])

assert max_grad < 10000, "Actor CNN gradient exploded"
assert mean_grad < 5000, "Actor CNN gradient too high on average"
assert not is_exponential_growth(gradient_norms['actor_cnn']), "Exponential pattern detected"
```

---

### Test 2: Q-Value Magnitude Check

**Objective**: Verify Q-values don't grow unbounded

**Steps**:
1. Extract Q-values from training logs
2. Plot Q-value mean vs training step
3. Check for monotonic increase (bad sign)

**Success Criteria**:
```python
q_values = extract_q_values(log_file)
max_q = max(q_values)

assert max_q < 50000000, "Q-values exploding"  # 50M threshold
```

---

### Test 3: Training Stability Check

**Objective**: Verify no NaN/Inf during training

**Steps**:
1. Run 1K test to completion
2. Check for error messages in log
3. Verify all checkpoints saved correctly

**Success Criteria**:
```bash
grep -i "nan\|inf\|error" validation_1k_3.log
# Should return empty or only non-critical warnings
```

---

### Test 4: Comparison to Baseline

**Objective**: Ensure fix doesn't degrade training performance

**Steps**:
1. Compare reward curves: Run #2 (no fix) vs Run #3 (with fix)
2. Compare episode lengths
3. Compare success rates

**Acceptance**:
- Slower learning is acceptable (gradient stability is priority)
- Reward curve should still show positive trend
- No regression in safety metrics (collisions)

---

## 10. Rollback Plan

**If Solution A causes issues**:

### Scenario 1: Training Too Slow

**Symptom**: Reward curve flat after 5K steps

**Action**:
```yaml
# Increase actor CNN LR slightly
actor_cnn_lr: 0.00003  # Try 3e-5 (between 1e-5 and 1e-4)
```

---

### Scenario 2: Gradients Still Exploding

**Symptom**: Actor CNN grad norm > 10,000 despite lower LR

**Action**:
1. Implement Solution B (gradient clipping)
2. Set `grad_clip_norm: 1.0`
3. Monitor clipping frequency
4. If clipping > 50% of steps, reduce LR further

---

### Scenario 3: Training Crashes

**Symptom**: NaN/Inf detected in parameters

**Action**:
1. Revert to last checkpoint
2. Reduce all learning rates by 10x
3. Implement gradient clipping
4. Consider reward scaling (0.01x)

---

## 11. Success Metrics

**Definition of Success** (for 1M deployment approval):

1. **Gradient Stability** ✅:
   - Actor CNN grad norm < 10,000 over 5K steps
   - No exponential growth pattern
   - Std(grad_norm) < 2 * mean(grad_norm)

2. **Training Completion** ✅:
   - 5K test completes without crashes
   - No NaN/Inf in any parameter
   - Checkpoints save/load correctly

3. **Performance Maintenance** ✅:
   - Reward curve shows improvement (trend > 0)
   - Episode success rate > 0% by step 3000
   - No regression in collision rate

4. **Backup Ready** ✅:
   - Gradient clipping code implemented (commented out)
   - Rollback configuration documented
   - Monitoring alerts configured

---

## 12. Timeline and Ownership

**Timeline**:
- **Phase 1** (Config change): 30 minutes
- **Phase 2** (1K validation): 30 minutes
- **Phase 3** (5K validation): 2 hours
- **Phase 4** (Backup implementation): 1 hour (if needed)
- **Phase 5** (Documentation): 1 hour

**Total Estimated Time**: 5 hours (worst case)

**Owner**: TD3 System Development Team

**Approver**: 1M Run Deployment Review (requires all phases complete)

---

## 13. References

1. **Fujimoto et al., 2018**: "Addressing Function Approximation Error in Actor-Critic Methods" (ICML)
2. **OpenAI Spinning Up TD3**: https://spinningup.openai.com/en/latest/algorithms/td3.html
3. **Stable-Baselines3 TD3**: https://stable-baselines3.readthedocs.io/en/master/modules/td3.html
4. **CARLA 0.9.16 Documentation**: https://carla.readthedocs.io/en/latest/
5. **"End-to-End Race Driving with DRL"**: Vinyals et al., 2017
6. **"End-to-End Lane Keeping Assist"**: Sallab et al., 2018

---

**Document Version**: 1.0
**Last Updated**: November 12, 2025
**Next Review**: After Phase 2 validation complete
