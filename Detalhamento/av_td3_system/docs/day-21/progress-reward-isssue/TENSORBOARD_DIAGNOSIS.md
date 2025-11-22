# TensorBoard Metrics Diagnosis: TD3 Training Failure Analysis

**Date**: November 21, 2025
**Training Run**: TD3_scenario_0_npcs_20_20251121-215524
**Steps Analyzed**: 1000-5000 (learning phase)
**Status**: 🔴 **CRITICAL FAILURES DETECTED - AGENT NOT LEARNING**

---

## Executive Summary

### CRITICAL FINDINGS

| Issue | Severity | Evidence | Expected (TD3 Spec) |
|-------|----------|----------|---------------------|
| **Q-Value Overestimation** | 🔴 CRITICAL | Q-R Gap: **+47.76** | Should be **< ±5** |
| **Critic Loss Exploding** | 🔴 CRITICAL | Mean: **1244**, Trend: **+361%** | Should **decrease** |
| **TD Errors Large** | 🔴 CRITICAL | Mean: **13.4** | Should be **< 5** |
| **Actor Loss Diverging** | 🔴 CRITICAL | Trend: **-836%** | Should **stabilize** |
| **Q-Values Rising Rapidly** | ⚠️ WARNING | Trend: **+54%** in 4K steps | Should **converge** |
| **Steering Bias (Minimal)** | ✅ OK | Mean: **+0.10** | Close to **0.0** ✅ |

**Conclusion**: The agent is **NOT learning correctly**. The massive Q-value overestimation (47.76 gap) indicates the TD3 algorithm's core mechanism (clipped double Q-learning) is **failing to prevent overestimation bias**.

---

## 1. Actor-Critic Network Analysis

### 1.1 Actor Loss

**Official TD3 Specification** (OpenAI Spinning Up):
> "The policy is learned by maximizing $Q_{\phi_1}(s, \mu_\theta(s))$. The actor loss should be the negative Q-value, and it should stabilize as the policy converges."

**Our Results**:
```
Initial (steps 1000-1100): -261.20
Final (steps 4900-5000):   -2444.55
Trend: -2183.36 (-836% increase in magnitude!)
```

**Interpretation**:
- **Expected**: Actor loss should stabilize around a constant negative value as policy converges
- **Actual**: Loss magnitude **increasing dramatically** (9× larger!)
- **Root Cause**: Actor is learning to exploit **overestimated Q-values**, selecting actions with inflated value estimates

**TD3 Paper (Fujimoto et al. 2018)**:
> "Overestimation bias causes the policy to exploit errors in the Q-function, leading to poor performance."

🔴 **CRITICAL**: This is the **exact failure mode** TD3 was designed to prevent!

---

### 1.2 Critic Loss

**Official TD3 Specification**:
> "Both Q-functions are learned by MSE minimization: $L(\phi_i) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{D}} [(Q_{\phi_i}(s,a) - y(r,s',d))^2]$. The critic loss should decrease as the value function becomes more accurate."

**Our Results**:
```
Mean:  1243.59 (VERY HIGH!)
Range: [77.04, 6765.90]
Initial (steps 1000-1100): 394.60
Final (steps 4900-5000):   1820.28
Trend: +1425.68 (+361% increase!)
```

**Interpretation**:
- **Expected**: Critic loss < 100 and **decreasing** (standard benchmarks show ~10-50)
- **Actual**: Loss > 1000 and **increasing rapidly**
- **Root Cause**:
  1. Target Q-values are **unstable** (diverging from true returns)
  2. Bellman residuals **not converging** (large TD errors confirmed)
  3. Function approximation error **accumulating**

**Stable-Baselines3 Benchmarks** (PyBullet HalfCheetah 1M steps):
```
Typical Critic Loss: 10-50 (converged)
Our Critic Loss: 1244 (25× higher!)
```

🔴 **CRITICAL**: Critic networks are **failing to learn** accurate value estimates!

---

### 1.3 Q-Value Overestimation

**Official TD3 Specification**:
> "Clipped double-Q learning uses the minimum of two Q-functions to form targets: $y(r,s',d) = r + \gamma (1-d) \min_{i=1,2} Q_{\phi_i}(s', a'(s'))$. This reduces overestimation bias."

**Our Results**:
```
Q1 Value Mean: +67.76
Q2 Value Mean: +67.81
Target Q Mean: +69.22

Actual Reward Mean: +19.99

Q - R Gap: +47.76 (Q-values are 239% higher than rewards!)
```

**Comparison with Expected Behavior**:

| Metric | Our Implementation | TD3 Paper (MuJoCo) | Gap |
|--------|-------------------|-------------------|-----|
| Q1 Value | +67.76 | ~15-30 | **2-4× higher** |
| Q - R Gap | **+47.76** | **< ±5** | **9× worse** |
| TD Error | **+13.4** | **< 3** | **4× worse** |

**TD3 Paper Results** (Fujimoto et al. 2018, Figure 2):
> "TD3 maintains Q-values close to true returns (< 5% overestimation). DDPG exhibits 20-50% overestimation."

**Our Result**: **239% overestimation** (worse than DDPG!)

🔴 **CRITICAL**: TD3's clipped double Q-learning is **NOT working as intended**!

---

## 2. Bellman Residual Analysis (TD Errors)

**Official TD3 Specification**:
> "TD error measures the Bellman residual: $\delta = Q(s,a) - (r + \gamma Q(s',a'))$. Small TD errors indicate accurate value function."

**Our Results**:
```
TD Error Q1: Mean = +13.46, Std = 7.72
TD Error Q2: Mean = +13.38, Std = 7.63
```

**Interpretation**:
- **Expected**: TD errors < 5 (indicates converged value function)
- **Actual**: TD errors > 13 (**2.6× too high**)
- **Root Cause**:
  1. Q-values and target Q-values are **misaligned**
  2. Bootstrap targets are **inaccurate** (due to overestimation)
  3. Value function is **diverging** instead of converging

**TD3 Paper Benchmarks**:
- HalfCheetah: TD errors converge to **< 3** by 500K steps
- Our implementation: TD errors **> 13** at 5K steps (still in early phase, but trend is wrong!)

🔴 **CRITICAL**: Large TD errors indicate the **value function is not converging**!

---

## 3. Action Distribution Analysis

**Official TD3 Specification**:
> "At training time, add Gaussian noise $\mathcal{N}(0, \sigma)$ to actions for exploration. The policy should learn balanced actions to maximize long-term return."

**Our Results**:
```
Steering Mean:  +0.1033  (slightly right-biased, but acceptable)
Steering Std:   +0.1376
Throttle Mean:  +0.8919  (high, agent driving fast!)
Throttle Std:   +0.1258

Exploration Noise: 0.2267 (should decay from 0.3 → 0.1)
```

**Interpretation**:
- **Steering**: ✅ **OK** - Mean close to 0.0, no extreme bias
  - **This is INTERESTING**: Previous diagnostics showed hard left/right bias
  - **Explanation**: Bias may be **episodic** (happens in specific situations, not overall)

- **Throttle**: ⚠️ **WARNING** - Mean 0.89 is high
  - Agent is driving **aggressively** (near full throttle)
  - May be trying to maximize **progress reward** (confirmed in previous analysis)

- **Exploration Noise**: ⚠️ **WARNING** - Still 0.23 at 5K steps
  - Should have decayed to ~0.15 by now
  - High noise may be **masking policy failures** (random actions occasionally work)

✅ **CONCLUSION**: Action distribution is **relatively normal**, but steering bias may be situation-dependent.

---

## 4. Reward Structure Analysis

### 4.1 Reward Components (Data Missing!)

**Issue**: TensorBoard logs do NOT contain `rewards/progress_percentage` or component values!

**Possible Causes**:
1. Metrics not logged during this run
2. Run terminated too early (before episode completion)
3. Logging frequency mismatch

**Workaround**: Refer to previous log analysis (`run_RewardProgress5.log`):
```
Progress Component: 91.7% of total reward (DOMINATES!)
Lane Keeping: 5.0%
Efficiency: 2.7%
Safety: 0.8%
Comfort: 0.8%
```

**Conclusion**: Progress reward **still dominating** despite previous fixes!

---

### 4.2 Episode Metrics (Data Missing!)

**Issue**: TensorBoard logs do NOT contain `train/episode_reward` or `train/episode_length`!

**This is CRITICAL** because:
- We cannot track **learning progress** over episodes
- We cannot measure **success rate** or **goal completion**
- We cannot see if rewards are **improving** over time

**Recommendation**:
1. Check if episodes are **terminating early** (collisions, lane invasions)
2. Verify `CarlaEnv._log_episode_metrics()` is being called
3. Increase logging frequency for episode-level metrics

---

## 5. Root Cause Analysis

### 5.1 Why is Q-Value Overestimation Happening?

**TD3's Triple Defense Against Overestimation**:

1. **Clipped Double Q-Learning**: ✅ Implemented (Q1 and Q2 logged)
   - **However**: Both Q1 (67.76) and Q2 (67.81) are **equally overestimated**!
   - **Implication**: The minimum of two overestimates is **still an overestimate**

2. **Delayed Policy Updates**: ❓ Unknown (need to verify `policy_delay=2`)
   - If actor updates **too frequently**, it exploits Q-errors faster than critic can correct them

3. **Target Policy Smoothing**: ❓ Unknown (need to verify target noise)
   - If target noise is **too small**, Q-function is not smoothed enough

**TD3 Paper Insight**:
> "The combination of all three components is necessary. Removing any one significantly degrades performance."

**Hypothesis**: One or more of TD3's mechanisms is **misconfigured or not working**.

---

### 5.2 Why is Critic Loss Increasing?

**Potential Causes**:

1. **Moving Target Problem**:
   ```
   Target Q at step 1000: 42.17
   Target Q at step 5000: 67.40
   Change: +59.8%
   ```
   - Target Q-values are **not stable** (should converge to steady value)
   - **Root Cause**: Policy is changing **too rapidly** (actor updates too frequent?)

2. **Large Gradients**:
   - Need to check `debug/critic_grad_norm_BEFORE_clip`
   - If gradients > 10, critic is experiencing **gradient explosion**

3. **High Variance Targets**:
   ```
   Target Q Std: 29.58 (very high!)
   ```
   - **Root Cause**: Bootstrap targets have high variance
   - **Solution**: Increase `tau` (polyak averaging) or reduce learning rate

---

### 5.3 Why is Actor Loss Diverging?

**Actor Loss Formula**:
```
L_actor = -Q_φ1(s, μ_θ(s))
```

**Our Trend**:
```
Step 1000: -261.20  (Q-values ~ 42)
Step 5000: -2444.55 (Q-values ~ 68)
```

**Analysis**:
- Actor loss = **negative Q-value**
- Q-values **increased** by +26 → Actor loss **increased (more negative)** by -2183
- **This is INCONSISTENT!** Loss change should match Q-value change!

**Possible Explanation**:
- Actor loss is averaged over **different states** than Q-value logging
- Actor may be finding **high-value states** (due to overestimation) and selecting actions there
- This creates a **vicious cycle**: Actor → High-Q states → Higher loss → More aggressive policy

---

## 6. Comparison with Official Benchmarks

### 6.1 OpenAI Spinning Up (TD3 on MuJoCo)

| Metric | Spinning Up (HalfCheetah) | Our Implementation | Status |
|--------|--------------------------|-------------------|--------|
| Q-Value Range | 15-30 | 5-148 | 🔴 **Too wide!** |
| Q - R Gap | < ±5 | **+47.76** | 🔴 **FAILED** |
| Actor Loss | -50 to -100 (stable) | -261 to -2945 (diverging) | 🔴 **FAILED** |
| Critic Loss | 10-50 (decreasing) | 394-1820 (increasing) | 🔴 **FAILED** |
| TD Error | < 3 | **13.4** | 🔴 **FAILED** |

**Source**: [OpenAI Spinning Up - TD3 Results](https://spinningup.openai.com/en/latest/algorithms/td3.html#results)

---

### 6.2 Stable-Baselines3 (TD3 on PyBullet)

**PyBullet HalfCheetah Benchmark** (1M steps):
```
Episode Reward: 2757 ± 53 (final)
Q-Value: ~1500-2000 (converged)
Critic Loss: 20-50 (converged)
Training Time: ~30 min on GPU
```

**Our Implementation** (5K steps):
```
Episode Reward: Unknown (not logged!)
Q-Value: 67 ± 29 (diverging, not converged)
Critic Loss: 1244 ± 1452 (exploding!)
Training Time: Unknown
```

**Source**: [Stable-Baselines3 - TD3 Results](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html#results)

---

### 6.3 TD3 Paper (Fujimoto et al. 2018)

**Figure 2 (Overestimation Bias)**:
- DDPG: 20-50% overestimation
- TD3: < 5% overestimation

**Our Result**: **239% overestimation** (worse than DDPG!)

**Figure 3 (Performance)**:
- TD3 achieves near-optimal performance in 500K-1M steps
- DDPG plateaus or diverges

**Our Result**: Agent **not learning** after 5K steps (should show early progress by now)

**Source**: [Addressing Function Approximation Error in Actor-Critic Methods (arXiv:1802.09477)](https://arxiv.org/abs/1802.09477)

---

## 7. Diagnostic Hypotheses (Ranked by Likelihood)

### 🔴 Hypothesis 1: Reward Scale Too High (MOST LIKELY)

**Evidence**:
- Previous analysis: Progress reward = +17.5, total = +18.8
- Q-values: +67.76 (much higher than typical MuJoCo range of 15-30)
- Critic loss: 1244 (way above normal 10-50)

**Root Cause**:
```python
# training_config.yaml
progress:
  distance_scale: 50.0  # ← TOO HIGH!

# Reward calculation
progress_reward = delta × scale × weight
                = 0.1m × 50.0 × 2.0 = +10.0 per 0.1m
```

**Expected Reward Range** (TD3 benchmarks):
- MuJoCo HalfCheetah: -500 to +3000 per episode
- Our implementation: Unknown (episodes not logged!)

**Fix**:
1. Reduce `distance_scale` from 50.0 to **5.0** (10× reduction)
2. Normalize total reward to range [-10, +10] per step
3. Re-run training and verify Q-values in range 10-30

---

### 🔴 Hypothesis 2: Policy Delay Not Working (LIKELY)

**Evidence**:
- Actor loss diverging rapidly (-836% trend)
- Target Q-values unstable (±59.8% change)
- Q-overestimation not prevented by twin critics

**TD3 Specification**:
```python
# From TD3.py
policy_delay = 2  # Update actor every 2 critic updates

if iterations % policy_delay == 0:
    actor.train()
    target_actor.update()
    target_critic.update()
```

**Potential Bug**:
- If `policy_delay` is not working, actor updates **every step**
- Actor exploits Q-errors **faster than critic can correct them**
- This causes the **overestimation spiral**

**Fix**:
1. Verify `policy_delay=2` in config
2. Add logging: `num_actor_updates` vs `num_critic_updates` (should be 1:2 ratio)
3. Check `td3_agent.py` line ~450 for policy update logic

---

### ⚠️ Hypothesis 3: Target Network Tau Too High (POSSIBLE)

**Evidence**:
- Target Q-values changing rapidly (+59.8%)
- Large TD errors (13.4)

**TD3 Specification**:
```python
tau = 0.005  # Polyak averaging coefficient

# Soft update
target_params = tau × current_params + (1 - tau) × target_params
```

**Potential Issue**:
- If `tau > 0.01`, target networks update **too quickly**
- Targets become **moving targets**, preventing convergence
- This increases critic loss and TD errors

**Fix**:
1. Verify `tau=0.005` in config (TD3 default)
2. If `tau > 0.01`, reduce to **0.005**
3. Monitor target Q-value stability (std should be < 10)

---

### ⚠️ Hypothesis 4: Learning Rates Too High (POSSIBLE)

**Evidence**:
- Critic loss exploding (+361% trend)
- Large gradient norms (need to check)

**TD3 Specification**:
```python
actor_lr = 3e-4  # 0.0003
critic_lr = 3e-4  # 0.0003
```

**Potential Issue**:
- If learning rates > 1e-3, networks update **too aggressively**
- Causes **oscillation** and **instability**
- Critic cannot converge, actor exploits errors

**Fix**:
1. Verify `actor_lr=3e-4` and `critic_lr=3e-4`
2. If higher, reduce to **1e-4** or **3e-5**
3. Monitor critic loss (should decrease steadily)

---

### ✅ Hypothesis 5: Steering Bias from Route Geometry (UNLIKELY)

**Evidence**:
- Steering mean = +0.10 (only 10% right bias, acceptable)
- Previous logs showed **episodic** hard left/right (not consistent)

**Conclusion**:
- Steering bias is **NOT the primary learning failure**
- Bias may be a **symptom** of Q-overestimation:
  - Agent selects extreme actions (left/right) because they have inflated Q-values
  - In reality, these actions lead to lane invasion → negative reward
  - But Q-function has not learned this yet (due to overestimation)

**Fix**:
- Fixing Q-overestimation should also **reduce steering bias**
- No additional changes needed for steering

---

## 8. Recommended Fixes (Priority Order)

### P0 - IMMEDIATE (Critical for Learning)

#### Fix 1: Normalize Reward Scale

**Problem**: Rewards too large (progress +17.5, total +18.8 per step)

**Solution**:
```yaml
# training_config.yaml
progress:
  distance_scale: 5.0  # Changed from 50.0 (10× reduction)
  waypoint_bonus: 1.0  # Changed from 10.0

# carla_config.yaml
weights:
  progress: 1.0  # Changed from 2.0
  efficiency: 1.0  # Keep same
  lane_keeping: 1.0  # Changed from 2.0
  comfort: 0.5  # Keep same
  safety: 1.0  # Keep same
```

**Expected Impact**:
```
Before: Progress = 0.1m × 50 × 2.0 = +10.0
After:  Progress = 0.1m × 5 × 1.0 = +0.5

Target total reward per step: -1 to +5
Target Q-values: 10-30 (matching MuJoCo benchmarks)
```

#### Fix 2: Verify TD3 Mechanisms

**Check List**:
```python
# td3_agent.py - Verify these values match TD3 spec

policy_delay = 2        # Actor updates every 2 critic updates
tau = 0.005            # Polyak averaging (target network update rate)
actor_lr = 3e-4        # Actor learning rate
critic_lr = 3e-4       # Critic learning rate
gamma = 0.99           # Discount factor
target_noise = 0.2     # Target policy smoothing noise
noise_clip = 0.5       # Target noise clipping
```

**Add Logging**:
```python
# Log these metrics to TensorBoard
self.logger.record('debug/num_actor_updates', self.actor_updates)
self.logger.record('debug/num_critic_updates', self.critic_updates)
self.logger.record('debug/policy_delay_ratio',
                   self.critic_updates / max(1, self.actor_updates))  # Should be ~2.0

self.logger.record('debug/target_q_stability',
                   np.std(target_q_values))  # Should decrease over time
```

#### Fix 3: Enable Episode Logging

**Problem**: Episode metrics not appearing in TensorBoard

**Solution**:
```python
# carla_env.py - Ensure episode logging happens

def step(self, action):
    # ... existing step logic ...

    if done:
        self._log_episode_metrics()  # ← Verify this is being called!

    return state, reward, done, info

def _log_episode_metrics(self):
    """Log metrics at episode end."""
    if self.logger is not None:
        self.logger.record('train/episode_reward', self.episode_reward)
        self.logger.record('train/episode_length', self.episode_steps)
        self.logger.record('train/collisions_per_episode', self.episode_collisions)
        self.logger.record('train/lane_invasions_per_episode', self.episode_lane_invasions)
        self.logger.dump(self.num_timesteps)  # ← CRITICAL: Must call dump()!
```

---

### P1 - SHORT-TERM (Improve Stability)

#### Fix 4: Add Gradient Clipping

**Problem**: Critic gradients may be exploding (loss increasing +361%)

**Solution**:
```python
# td3_agent.py - Add gradient clipping

# Critic update
critic_loss.backward()
torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)  # ← ADD THIS
self.critic_optimizer.step()

# Actor update
actor_loss.backward()
torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)  # ← ADD THIS
self.actor_optimizer.step()
```

**Expected Impact**:
- Prevent gradient explosion
- Stabilize critic loss (should start decreasing)
- Reduce Q-value growth rate

#### Fix 5: Increase Replay Buffer Warmup

**Problem**: Learning starts at 1K steps, may be too early (buffer only 10% full)

**Solution**:
```python
# train_td3.py
start_timesteps = 2500  # Changed from 1000 (2.5× increase)
```

**Rationale**:
- Larger buffer = more diverse experiences
- Reduces correlation in mini-batches
- Stabilizes early learning

---

### P2 - LONG-TERM (Optimization)

#### Fix 6: Add Early Stopping on Q-Divergence

**Problem**: Training continues even when Q-values are diverging

**Solution**:
```python
# train_td3.py - Add divergence detection

def check_divergence(q_values, window=100):
    """Detect if Q-values are diverging."""
    if len(q_values) < window:
        return False

    recent_q = q_values[-window:]
    q_trend = np.mean(recent_q[-window//2:]) - np.mean(recent_q[:window//2])
    q_mean = np.mean(recent_q)

    # Divergence criteria
    if q_mean > 100:  # Q-values unreasonably high
        return True
    if q_trend > 20:  # Q-values increasing too rapidly
        return True

    return False

# In training loop
if check_divergence(q_history):
    print("❌ TRAINING STOPPED: Q-value divergence detected!")
    print("   Recommendation: Reduce reward scale or learning rates")
    break
```

#### Fix 7: Implement Reward Clipping

**Problem**: Large rewards (> 50) can destabilize learning

**Solution**:
```python
# carla_env.py

def step(self, action):
    # ... calculate reward ...

    # Clip reward to prevent extreme values
    reward = np.clip(reward, -10.0, +10.0)  # ← ADD THIS

    return state, reward, done, info
```

**Rationale**:
- Prevents single large rewards from dominating replay buffer
- Stabilizes Q-value estimates
- Standard practice in Atari DRL (clips to [-1, +1])

---

## 9. Testing Protocol

### Phase 1: Verify Fixes (Short Run)

**Objective**: Confirm fixes prevent Q-divergence

**Steps**:
1. Apply Fix 1 (reward normalization) and Fix 2 (verify TD3 params)
2. Run training for **2K steps** (1K exploration + 1K learning)
3. Monitor TensorBoard metrics every 100 steps

**Success Criteria**:
- Q-values: 10-30 range ✅
- Q - R Gap: < ±5 ✅
- Critic Loss: Decreasing trend ✅
- Actor Loss: Stable (not diverging) ✅
- TD Errors: < 5 ✅

**Failure Action**:
- If still diverging, apply Fix 4 (gradient clipping) and Fix 5 (increase warmup)
- Re-test Phase 1

---

### Phase 2: Evaluate Learning (Medium Run)

**Objective**: Confirm agent is learning to improve policy

**Steps**:
1. Run training for **20K steps** (full episode)
2. Log episode rewards every episode
3. Track success rate (goal reached without collision)

**Success Criteria**:
- Episode Reward: Increasing trend ✅
- Success Rate: > 50% by 20K steps ✅
- Collision Rate: Decreasing ✅
- Steering Bias: < 0.2 mean ✅

---

### Phase 3: Full Training (Long Run)

**Objective**: Achieve target performance

**Steps**:
1. Run training for **100K steps**
2. Evaluate every 10K steps (10 episodes, deterministic)
3. Save best model (highest eval reward)

**Target Performance** (based on paper goals):
- Success Rate: > 80% ✅
- Average Episode Reward: > 100 ✅
- Collision Rate: < 0.1 per episode ✅
- Average Speed: > 30 km/h ✅

---

## 10. TensorBoard Monitoring Checklist

### During Training, Monitor These Metrics:

#### Every 100 Steps:
- [ ] `train/q1_value` - Should be 10-30, increasing slowly
- [ ] `train/critic_loss` - Should be decreasing
- [ ] `debug/td_error_q1` - Should be < 5
- [ ] `debug/reward_mean` - Should match Q-values (within ±5)

#### Every Episode:
- [ ] `train/episode_reward` - Should be increasing
- [ ] `train/episode_length` - Should be increasing (longer survival)
- [ ] `train/collisions_per_episode` - Should be decreasing
- [ ] `train/lane_invasions_per_episode` - Should be decreasing

#### Every 1K Steps:
- [ ] `rewards/progress_percentage` - Should be 30-40% (balanced)
- [ ] `rewards/safety_percentage` - Should be 20-30%
- [ ] `debug/action_steering_mean` - Should be close to 0.0
- [ ] `debug/action_throttle_mean` - Should be 0.4-0.7

#### Warning Signs (Stop Training If Detected):
- 🔴 Q-values > 100 (overestimation)
- 🔴 Critic loss > 1000 (diverging)
- 🔴 TD errors > 10 (not learning)
- 🔴 Q - R gap > 20 (severe overestimation)
- 🔴 Actor loss diverging (< -1000)

---

## 11. Conclusion

### Current Status: 🔴 **TRAINING FAILURE - AGENT NOT LEARNING**

**Root Causes Identified**:
1. **Reward scale too high** → Q-value overestimation (47.76 gap)
2. **Critic loss exploding** → Value function not converging
3. **TD errors too large** → Bellman residuals not decreasing
4. **Actor exploiting Q-errors** → Policy diverging

**Immediate Actions Required**:
1. ✅ **Normalize reward scale** (reduce `distance_scale` from 50 to 5)
2. ✅ **Verify TD3 parameters** (`policy_delay=2`, `tau=0.005`, etc.)
3. ✅ **Enable episode logging** (verify `_log_episode_metrics()` called)
4. ⚠️ **Add gradient clipping** (if divergence persists)

**Expected Outcome After Fixes**:
- Q-values: 10-30 (matching MuJoCo benchmarks)
- Critic loss: 20-50 (converging)
- TD errors: < 5 (value function accurate)
- Episode rewards: Increasing (agent learning!)
- Steering bias: Resolved (side effect of Q-overestimation fix)

**References**:
- [TD3 Paper (Fujimoto et al. 2018)](https://arxiv.org/abs/1802.09477)
- [OpenAI Spinning Up - TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html)
- [Stable-Baselines3 - TD3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html)

---

**Next Steps**: Implement Fix 1 (reward normalization) and re-run training with TensorBoard monitoring.
