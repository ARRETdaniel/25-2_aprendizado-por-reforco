# CRITICAL: Q-Value Explosion - Action Plan

**Date**: November 18, 2025  
**Status**: 🔴 **NO-GO for 50K** - Critical issue identified  
**Priority**: 🚨 **URGENT** - Must resolve before extended training

---

## TL;DR - What's Wrong

✅ **GOOD NEWS**: Gradient explosion is **100% FIXED**  
❌ **BAD NEWS**: **Q-value explosion still present** → Actor loss = **-2.4 MILLION**

**This means**: The agent thinks its actions will get **massive positive rewards** (which is impossible given our reward scale of -50 to +200 per step).

---

## The Numbers

### What's Working ✅

| Metric | Value | Status |
|--------|-------|--------|
| Actor CNN Gradients | 2.39 max | ✅ Perfect (was 1.8M before fix) |
| Critic CNN Gradients | 25.09 max | ✅ Perfect |
| Learning Rate | 1e-5 | ✅ Literature-validated |
| Gradient Alerts | 0 warnings, 0 critical | ✅ No explosions detected |

### What's Broken ❌

| Metric | Value | Expected | Problem |
|--------|-------|----------|---------|
| **Actor Loss** | **-2.4M** | ~0 to -1,000 | ❌ **26,000× too large** |
| Q-Values (Q1) | 90.18 | 0-200 | ⚠️ High but not catastrophic |
| Episode Length | 3 steps (final) | 5-20 | ⚠️ Below expected |

---

## Root Cause Analysis

### What We've Ruled Out ✅

1. ✅ Gradient explosion (fixed - gradients clipped correctly)
2. ✅ Learning rate too high (corrected to 1e-5)
3. ✅ Update frequency wrong (fixed to train_freq=50)
4. ✅ Gradient accumulation issue (fixed to gradient_steps=1)

### The Real Culprit 🎯

**Hypothesis**: **Reward scaling/accumulation bug**

**Evidence**:
```
Actor Loss = -mean(Q(s, μ(s))) = -2,400,000

This means the critic believes:
  Expected cumulative reward = +2,400,000

But our reward scale is:
  Per-step: -50 to +200
  Per episode (100 steps): -5,000 to +20,000
  
Where did +2,400,000 come from? 🤔
```

**Possible Causes**:

1. **Reward accumulation without discounting**
   ```python
   # WRONG
   cumulative_reward += reward  # No γ decay
   
   # CORRECT
   cumulative_reward = reward + γ * next_value
   ```

2. **Progress bonus inflation**
   ```python
   # If +10 bonus per waypoint × 25 waypoints × many episodes
   # Could accumulate to millions without proper normalization
   ```

3. **Reward normalization missing**
   ```python
   # Rewards should be normalized to prevent scale mismatch
   reward = (reward - mean) / (std + eps)
   ```

4. **Critic learning cumulative sum instead of expected return**
   ```python
   # Bellman target should be:
   y = r + γ * min(Q1_target, Q2_target) * (1 - done)
   
   # NOT:
   y = Σ(all_rewards)  # ← This would explode
   ```

---

## IMMEDIATE ACTION REQUIRED

### Step 1: Add Diagnostic Logging (5 minutes)

Add to `src/agents/td3_agent.py` in the `train()` method:

```python
# After computing actor loss
self.writer.add_scalar('debug/actor_q_mean', actor_q.mean().item(), self.total_iterations)
self.writer.add_scalar('debug/actor_q_max', actor_q.max().item(), self.total_iterations)
self.writer.add_scalar('debug/actor_q_min', actor_q.min().item(), self.total_iterations)
self.writer.add_scalar('debug/actor_q_std', actor_q.std().item(), self.total_iterations)

# After computing target Q
self.writer.add_scalar('debug/target_q_mean', target_q.mean().item(), self.total_iterations)
self.writer.add_scalar('debug/target_q_max', target_q.max().item(), self.total_iterations)
```

Add to `src/environment/reward_functions.py`:

```python
# Log each reward component
self.writer.add_scalar('reward_components/efficiency', efficiency_reward, step)
self.writer.add_scalar('reward_components/lane_keeping', lane_keeping_reward, step)
self.writer.add_scalar('reward_components/comfort', comfort_penalty, step)
self.writer.add_scalar('reward_components/safety', safety_penalty, step)
self.writer.add_scalar('reward_components/total', total_reward, step)
```

### Step 2: Run 5K Diagnostic (30 minutes)

```bash
cd av_td3_system

# Same command as before, just with new logging
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

### Step 3: Analyze Diagnostic Logs (10 minutes)

Look for in TensorBoard:
1. `debug/actor_q_*` → Are these values also ~2.4M?
2. `debug/target_q_*` → Are target Q-values realistic?
3. `reward_components/*` → Which component is dominating?

### Step 4: Implement Fix Based on Findings

**If reward scaling is the issue**:

```python
# Option A: Clip rewards (simple, fast)
reward = np.clip(reward, -10.0, +10.0)

# Option B: Normalize rewards (more principled)
class RewardNormalizer:
    def __init__(self, clip=10.0):
        self.mean = 0
        self.var = 1
        self.count = 0
        self.clip = clip
    
    def normalize(self, reward):
        # Update running statistics
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        self.var += delta * (reward - self.mean)
        
        # Normalize
        std = np.sqrt(self.var / self.count) if self.count > 1 else 1.0
        normalized = (reward - self.mean) / (std + 1e-8)
        
        # Clip
        return np.clip(normalized, -self.clip, +self.clip)
```

**If critic overfitting is the issue**:

```python
# Add L2 regularization to critic
critic_loss = critic_loss + 0.01 * l2_reg(critic.parameters())

# OR increase target smoothing noise
policy_noise = 0.3  # from 0.2
noise_clip = 0.6    # from 0.5
```

### Step 5: Validate Fix (30 minutes)

Run another 5K with the fix applied. Check:
- Actor loss < 100,000 ✅
- Episode length increasing ✅
- Q-values stable ✅

---

## Timeline

| Task | Duration | Status |
|------|----------|--------|
| 1. Add diagnostic logging | 5 min | ⏳ TODO |
| 2. Run 5K diagnostic | 30 min | ⏳ TODO |
| 3. Analyze logs | 10 min | ⏳ TODO |
| 4. Implement fix | 15 min | ⏳ TODO |
| 5. Validate fix (5K) | 30 min | ⏳ TODO |
| **TOTAL** | **90 min** | **Before 50K** |

---

## Decision Tree

```
┌─────────────────────────────┐
│ Diagnostic 5K Run Complete  │
└──────────┬──────────────────┘
           │
           ├─→ Actor Q-values ~2.4M?
           │   ├─ YES → Critic overestimation
           │   │         ├─ Add L2 regularization
           │   │         └─ Increase target smoothing
           │   └─ NO  → Logging error (investigate)
           │
           ├─→ Reward component > 1000/step?
           │   ├─ YES → Reward scaling issue
           │   │         ├─ Add reward clipping
           │   │         └─ OR reward normalization
           │   └─ NO  → Accumulation bug (check Bellman update)
           │
           └─→ Target Q-values unrealistic?
               ├─ YES → Bootstrap error
               │         └─ Check TD3 target calculation
               └─ NO  → Unknown (deep dive required)
```

---

## What NOT to Do ❌

1. ❌ **Don't proceed to 50K** - Will waste 6 hours and learn nothing
2. ❌ **Don't reduce learning rate further** - Already at optimal (1e-5)
3. ❌ **Don't increase gradient clipping** - Gradients are healthy
4. ❌ **Don't change TD3 architecture** - Configuration is correct

---

## Success Criteria for Next Run

Before proceeding to 50K, the next 5K run MUST show:

| Metric | Current | Target | Pass/Fail |
|--------|---------|--------|-----------|
| Actor Loss | -2.4M | < 100K | ❌ FAIL |
| Episode Length (final) | 3 steps | 5-20 steps | ❌ FAIL |
| Q-Values (Q1) | 90 | < 200 | ⚠️ MARGINAL |
| Gradients | 2.39 | < 10K | ✅ PASS |

**Required for GO**: 2/4 metrics must PASS (actor loss and episode length)

---

## Questions to Answer from Diagnostic Run

1. **Is the actor receiving Q-values of 2.4M?**
   - Check: `debug/actor_q_mean` in TensorBoard
   - If YES → Critic overestimation confirmed
   - If NO → Logging error or calculation bug

2. **Which reward component is largest?**
   - Check: `reward_components/*` in TensorBoard
   - If progress bonus dominates → Cap or normalize it
   - If all balanced → Check accumulation logic

3. **Are target Q-values realistic?**
   - Check: `debug/target_q_mean` in TensorBoard
   - Should be similar to logged Q1/Q2 (~90)
   - If much larger → Bootstrap error in TD3

4. **Is the Bellman update correct?**
   - Verify: `target_q = reward + γ * next_q * (1 - done)`
   - Check: Done flag is set correctly
   - Check: Discount factor γ = 0.99

---

## Contact Points for Help

If stuck after diagnostic run, check:
1. **TD3 paper** (Fujimoto et al., 2018) - Section 3, Algorithm 1
2. **Spinning Up TD3** - https://spinningup.openai.com/en/latest/algorithms/td3.html
3. **Stable-Baselines3 TD3** - Reference implementation
4. **Related work**: Check `#related_works` folder for similar implementations

---

**NEXT STEP**: Add diagnostic logging and run 5K diagnostic ASAP!  
**ETA to resolution**: 90 minutes  
**ETA to 50K (if fixed)**: 90 min (diagnostic) + 6 hours (50K) = **7.5 hours total**

---

**Document**: `ACTION_PLAN_Q_VALUE_EXPLOSION.md`  
**Created**: November 18, 2025  
**Status**: 🚨 **URGENT - BLOCKING 50K RUN**
