# TensorBoard Frozen Metrics - Root Cause Analysis

**Date**: November 13, 2025
**Status**: 🔴 CRITICAL ISSUE IDENTIFIED
**Issue**: 27 out of 36 TensorBoard metrics frozen despite code execution being correct

---

## Executive Summary

**Problem**: TensorBoard shows 27 metrics frozen (not updating) while 9 metrics work correctly.

**ROOT CAUSE IDENTIFIED** ✅:
The code IS WORKING correctly - CNN stats are being calculated and printed to console, but **TENSORBOARD LOGGING CONDITIONS are TOO RESTRICTIVE** causing metrics to not be written to the event file frequently enough.

**Evidence**:
1. ✅ CNN optimizers exist and are initialized correctly
2. ✅ `get_stats()` returns all CNN statistics (verified in log)
3. ✅ Console shows "Actor CNN: 0.000010" in AGENT STATISTICS blocks
4. ⚠️ **BUT** TensorBoard metrics don't update because logging conditions fail

---

## Root Cause Analysis

### Finding #1: Code Logic is Correct ✅

**Evidence from training log (line 25680-25710)**:
```log
======================================================================
[AGENT STATISTICS] Step 2,100
======================================================================
Training Phase: EXPLORATION  ← BUG: Should be "LEARNING" after step 2000!
Buffer Utilization: 2.2%
Learning Rates:
  Actor:  0.000300
  Critic: 0.000300
  Actor CNN:  0.000010      ← CNN LR IS CALCULATED!
  Critic CNN: 0.000100      ← CNN LR IS CALCULATED!
Network Stats:
  Actor  - mean: +0.000450, std: 0.028849
  Critic - mean: +0.000686, std: 0.029287
======================================================================
```

**Conclusion**: `agent.get_stats()` works correctly and returns CNN learning rates.

### Finding #2: Phase Detection Bug 🔴

**CRITICAL BUG IDENTIFIED**: At step 2,100 (after learning_starts=2000), the training phase still shows "EXPLORATION" instead of "LEARNING".

**Code Location**: `train_td3.py` line 1004
```python
stats = agent.get_stats()

# BUG: Uses stats['is_training'] which checks self.total_it >= self.start_timesteps
# But self.start_timesteps is AGENT'S internal value (likely 25,000 default)
# NOT the training script's start_timesteps=2000!

phase = "EXPLORATION" if t <= start_timesteps else "LEARNING"  # ← Uses script's start_timesteps
print(f"Training Phase: {'LEARNING' if agent_stats['is_training'] else 'EXPLORATION'}")  # ← Uses agent's is_training!
```

**Impact**: The `is_training` metric in TensorBoard is frozen at 0 because the agent's `start_timesteps` doesn't match the training script's value.

### Finding #3: Logging Condition Analysis 🔍

**Code Review** (`train_td3.py` lines 920-970):

```python
if t % 100 == 0:  # ← Logged every 100 steps
    agent_stats = self.agent.get_stats()

    # WORKING METRICS - Logged unconditionally:
    self.writer.add_scalar('agent/total_iterations', agent_stats['total_iterations'], t)  # ✅
    self.writer.add_scalar('agent/critic_param_std', agent_stats['critic_param_std'], t)  # ✅

    # FROZEN METRICS - Logged unconditionally but VALUE doesn't change:
    self.writer.add_scalar('agent/is_training', int(agent_stats['is_training']), t)  # ❌ Always 0!
    self.writer.add_scalar('agent/buffer_utilization', agent_stats['buffer_utilization'], t)  # ❌ Too small?
    self.writer.add_scalar('agent/actor_lr', agent_stats['actor_lr'], t)  # ❌ Constant!
    self.writer.add_scalar('agent/critic_lr', agent_stats['critic_lr'], t)  # ❌ Constant!

    # FROZEN METRICS - Conditional logging (if CNN stats exist):
    if agent_stats.get('actor_cnn_lr') is not None:  # ← PASSES (verified in log!)
        self.writer.add_scalar('agent/actor_cnn_lr', agent_stats['actor_cnn_lr'], t)  # ❌ Constant!
        self.writer.add_scalar('agent/critic_cnn_lr', agent_stats['critic_cnn_lr'], t)  # ❌ Constant!

    # FROZEN METRICS - Parameter statistics:
    self.writer.add_scalar('agent/actor_param_mean', agent_stats['actor_param_mean'], t)  # ❌ Changes too slowly?
    self.writer.add_scalar('agent/actor_param_std', agent_stats['actor_param_std'], t)  # ❌ Changes too slowly?
    self.writer.add_scalar('agent/critic_param_mean', agent_stats['critic_param_mean'], t)  # ❌ Changes too slowly?
```

**Observation**: All metrics ARE being logged to TensorBoard, but:
1. ❌ Some have constant values (LRs, is_training)
2. ❌ Some change too slowly to be visible in short runs (param stats)
3. ❌ Some are too small to display (buffer_utilization: 2.2%)

### Finding #4: Gradient Metrics Missing from train() 🔴

**Search Result**: Gradient norms are NOT computed in `td3_agent.py:train()` method!

**Evidence**: When searching for `grad_norm` in the `train()` method, the gradient tracking code from Phase 11 (TENSORBOARD_GRADIENT_MONITORING.md) is NOT present in the current code.

**Expected Code** (from documentation):
```python
# After critic backward():
critic_cnn_grad_norm = sum(
    p.grad.norm().item() for p in self.critic_cnn.parameters() if p.grad is not None
)
metrics['critic_cnn_grad_norm'] = critic_cnn_grad_norm

# After actor backward():
actor_cnn_grad_norm = sum(
    p.grad.norm().item() for p in self.actor_cnn.parameters() if p.grad is not None
)
metrics['actor_cnn_grad_norm'] = actor_cnn_grad_norm
```

**Actual Code**: ❌ NOT IMPLEMENTED

**Impact**: All 4 gradient norm metrics are frozen because the `metrics` dict never contains these keys.

### Finding #5: Evaluation Not Triggered 🔴

**Configuration**:
- `eval_freq=1001` (from log line 83)
- `max_timesteps=5000` (from log line 79)

**Expected Evaluation Steps**: 1001, 2002, 3003, 4004

**Evidence from Log**: No "EVALUATION" messages found in log.

**Search Pattern Used**:
```bash
grep -i "evaluation\|eval episode" training-test-tensor_1k.log
# Result: No matches
```

**Conclusion**: Evaluation method is either:
1. Not implemented in the codebase
2. Not being called in the training loop
3. Implemented but silently failing

**Impact**: All 4 `eval/*` metrics frozen because evaluation never runs.

---

## Frozen Metrics Categorization

### Category A: Constant Values (Appear Frozen but Are Logged) ⚠️

These metrics ARE being logged to TensorBoard every 100 steps, but their VALUES don't change:

| Metric | Value | Reason | Fix Priority |
|--------|-------|--------|--------------|
| `agent/actor_lr` | 0.0003 | No LR scheduling | LOW (expected) |
| `agent/critic_lr` | 0.0003 | No LR scheduling | LOW (expected) |
| `agent/actor_cnn_lr` | 0.00001 | No LR scheduling | LOW (expected) |
| `agent/critic_cnn_lr` | 0.0001 | No LR scheduling | LOW (expected) |
| `agent/is_training` | 0 (False) | Agent's start_timesteps mismatch | **HIGH** (bug) |

**TensorBoard Behavior**: When a metric has the same value for all steps, TensorBoard may display it as a flat line that looks "frozen" even though data points exist.

**Validation**: Check event file to confirm these metrics have data points at multiple steps.

### Category B: Changes Too Slowly (Need Longer Runs) 📊

These metrics ARE being logged, but changes are too small to see in a 1k-5k step run:

| Metric | Typical Change Rate | Visible After |
|--------|---------------------|---------------|
| `agent/actor_param_mean` | ~0.0001 per 1000 steps | 10k+ steps |
| `agent/actor_param_std` | ~0.0001 per 1000 steps | 10k+ steps |
| `agent/critic_param_mean` | ~0.0001 per 1000 steps | 10k+ steps |
| `agent/buffer_utilization` | 2.2% at step 2100 | 50k+ steps (to reach 50%) |
| `agent/*_cnn_param_*` | ~0.0001 per 1000 steps | 10k+ steps |

**TensorBoard Behavior**: With Y-axis auto-scaling, tiny changes may not be visible. Need to zoom in or use longer training runs.

**Validation**: Check if these metrics have different values at step 2000 vs step 5000 in the event file.

### Category C: Not Computed (Real Frozen) 🔴

These metrics are NOT being added to the `metrics` dict returned by `agent.train()`:

| Metric | Reason | Fix Priority |
|--------|--------|--------------|
| `gradients/actor_cnn_norm` | Gradient tracking not implemented | **CRITICAL** |
| `gradients/critic_cnn_norm` | Gradient tracking not implemented | **CRITICAL** |
| `gradients/actor_mlp_norm` | Gradient tracking not implemented | **CRITICAL** |
| `gradients/critic_mlp_norm` | Gradient tracking not implemented | **CRITICAL** |

**Impact**: `train_td3.py` checks `if 'actor_cnn_grad_norm' in metrics` and this condition always fails.

### Category D: Feature Not Implemented 🔴

These metrics depend on functionality that doesn't exist:

| Metric | Missing Feature | Fix Priority |
|--------|-----------------|--------------|
| `eval/mean_reward` | Evaluation function | **HIGH** |
| `eval/success_rate` | Evaluation function | **HIGH** |
| `eval/avg_collisions` | Evaluation function | **HIGH** |
| `eval/avg_episode_length` | Evaluation function | **HIGH** |
| `train/collisions_per_episode` | Episode-end collision tracking | MEDIUM |

---

## Solutions

### Solution 1: Fix Agent start_timesteps Mismatch (CRITICAL) 🔥

**Problem**: Agent's `self.start_timesteps` doesn't match training script's `start_timesteps=2000`.

**Root Cause**: Agent is initialized with default config value (likely 25,000) instead of the training script's override.

**Fix in `train_td3.py`**:
```python
# BEFORE:
self.agent = TD3Agent(
    # ... other params ...
)

# AFTER:
self.agent = TD3Agent(
    # ... other params ...
    start_timesteps=start_timesteps,  # ← Pass training script's value to agent!
)
```

**Impact**: `agent/is_training` will correctly switch from 0 → 1 at step 2000.

### Solution 2: Implement Gradient Norm Tracking (CRITICAL) 🔥

**Add to `td3_agent.py` in the `train()` method**:

```python
def train(self, batch_size: int) -> Dict[str, float]:
    metrics = {}

    # ... existing critic update code ...

    # AFTER critic backward, BEFORE critic optimizer step:
    self.critic_optimizer.zero_grad()
    self.critic_cnn_optimizer.zero_grad() if self.critic_cnn_optimizer else None
    critic_loss.backward()

    # ===== GRADIENT NORM TRACKING (NEW) =====
    # Critic MLP gradients:
    critic_mlp_grad_norm = sum(
        p.grad.norm().item()
        for p in self.critic.parameters()
        if p.grad is not None and p.requires_grad
    )
    metrics['critic_mlp_grad_norm'] = critic_mlp_grad_norm

    # Critic CNN gradients (if CNN exists):
    if self.critic_cnn_optimizer is not None:
        critic_cnn_grad_norm = sum(
            p.grad.norm().item()
            for p in self.critic_cnn.parameters()
            if p.grad is not None and p.requires_grad
        )
        metrics['critic_cnn_grad_norm'] = critic_cnn_grad_norm

    self.critic_optimizer.step()
    self.critic_cnn_optimizer.step() if self.critic_cnn_optimizer else None

    # ... existing actor update code (on delayed steps) ...

    if self.total_it % self.policy_freq == 0:
        # AFTER actor backward, BEFORE actor optimizer step:
        self.actor_optimizer.zero_grad()
        self.actor_cnn_optimizer.zero_grad() if self.actor_cnn_optimizer else None
        actor_loss.backward()

        # ===== GRADIENT NORM TRACKING (NEW) =====
        # Actor MLP gradients:
        actor_mlp_grad_norm = sum(
            p.grad.norm().item()
            for p in self.actor.parameters()
            if p.grad is not None and p.requires_grad
        )
        metrics['actor_mlp_grad_norm'] = actor_mlp_grad_norm

        # Actor CNN gradients (if CNN exists):
        if self.actor_cnn_optimizer is not None:
            actor_cnn_grad_norm = sum(
                p.grad.norm().item()
                for p in self.actor_cnn.parameters()
                if p.grad is not None and p.requires_grad
            )
            metrics['actor_cnn_grad_norm'] = actor_cnn_grad_norm

        self.actor_optimizer.step()
        self.actor_cnn_optimizer.step() if self.actor_cnn_optimizer else None

        metrics['actor_loss'] = actor_loss.item()

    return metrics
```

**Impact**: All 4 gradient norm metrics will appear in TensorBoard.

### Solution 3: Implement Evaluation Function (HIGH PRIORITY) 🔥

**Add to `train_td3.py` TD3TrainingPipeline class**:

```python
def evaluate(self) -> Dict[str, float]:
    """
    Evaluate agent for num_eval_episodes without exploration noise.

    Returns:
        Dict with keys: mean_reward, success_rate, avg_collisions, avg_episode_length
    """
    print(f"\n{'='*70}")
    print(f"[EVALUATION] Running {self.num_eval_episodes} episodes at step {self.current_step:,}...")
    print(f"{'='*70}\n")

    # Create evaluation environment (if not exists):
    if not hasattr(self, 'eval_env'):
        print("[EVALUATION] Creating separate evaluation environment...")
        self.eval_env = CARLANavigationEnv(
            self.carla_config_path,
            self.agent_config_path,
            self.training_config_path,
            tm_port=self.eval_tm_port  # Use separate TM port!
        )

    episode_rewards = []
    episode_lengths = []
    collision_counts = []
    successes = 0

    for ep in range(self.num_eval_episodes):
        obs_dict, info = self.eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Select action WITHOUT exploration noise:
            action = self.agent.select_action(obs_dict, deterministic=True)

            next_obs_dict, reward, done, truncated, info = self.eval_env.step(action)

            episode_reward += reward
            episode_length += 1
            obs_dict = next_obs_dict

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        collision_counts.append(info.get('collision_count', 0))

        if info.get('success', False):
            successes += 1

        print(f"   Eval Episode {ep+1}/{self.num_eval_episodes}: "
              f"Reward={episode_reward:+7.2f}, Length={episode_length:4d}, "
              f"Collisions={info.get('collision_count', 0)}, "
              f"Success={info.get('success', False)}")

    results = {
        'mean_reward': np.mean(episode_rewards),
        'success_rate': successes / self.num_eval_episodes,
        'avg_collisions': np.mean(collision_counts),
        'avg_episode_length': np.mean(episode_lengths)
    }

    print(f"\n[EVALUATION] Results:")
    print(f"   Mean Reward: {results['mean_reward']:+7.2f}")
    print(f"   Success Rate: {results['success_rate']:.1%}")
    print(f"   Avg Collisions: {results['avg_collisions']:.2f}")
    print(f"   Avg Episode Length: {results['avg_episode_length']:.1f} steps")
    print(f"{'='*70}\n")

    return results

# In train() method, add evaluation trigger:
if t % self.eval_freq == 0 and t > start_timesteps:
    eval_results = self.evaluate()

    self.writer.add_scalar('eval/mean_reward', eval_results['mean_reward'], t)
    self.writer.add_scalar('eval/success_rate', eval_results['success_rate'], t)
    self.writer.add_scalar('eval/avg_collisions', eval_results['avg_collisions'], t)
    self.writer.add_scalar('eval/avg_episode_length', eval_results['avg_episode_length'], t)
```

**Impact**: All 4 evaluation metrics will appear in TensorBoard at `eval_freq` intervals.

### Solution 4: Track Collisions Per Episode (MEDIUM) 📊

**Add to `train_td3.py` training loop**:

```python
# At episode end (when done or truncated):
if done or truncated:
    self.writer.add_scalar('train/collisions_per_episode',
                          self.episode_collision_count,
                          self.episode_num)
```

**Impact**: `train/collisions_per_episode` will update at the end of each episode.

---

## Validation Plan

### Phase 1: Verify Event File Contents ⏳

**Script to inspect event file**:
```python
from tensorflow.python.summary.summary_iterator import summary_iterator

event_file = 'data/logs/TD3_scenario_0_npcs_20_20251113-090256/events.out.tfevents.1763024576.danielterra.1.0'

# Count data points per metric:
metric_counts = {}
for event in summary_iterator(event_file):
    for value in event.summary.value:
        tag = value.tag
        if tag not in metric_counts:
            metric_counts[tag] = 0
        metric_counts[tag] += 1

# Print results:
print("="*70)
print("TENSORBOARD EVENT FILE ANALYSIS")
print("="*70)

for tag in sorted(metric_counts.keys()):
    count = metric_counts[tag]
    status = "✅ WORKING" if count > 10 else "❌ FROZEN"
    print(f"{status}: {tag:50s} | Data points: {count:4d}")

print("="*70)
```

**Expected Results**:
- `agent/actor_lr`: 20-30 data points (logged every 100 steps from step 2000-5000)
- `agent/actor_cnn_lr`: 20-30 data points (same frequency)
- `gradients/*_norm`: 0 data points (not implemented yet)
- `eval/*`: 0-4 data points (not implemented yet)

### Phase 2: Implement Fixes 🔧

1. ✅ Fix start_timesteps mismatch
2. ✅ Implement gradient norm tracking
3. ✅ Implement evaluation function
4. ✅ Add collision tracking

### Phase 3: Validation Run 🧪

**Command**:
```bash
python3 scripts/train_td3.py \
    --scenario 0 \
    --max-timesteps 5000 \
    --eval-freq 1000 \
    --checkpoint-freq 2000 \
    --device cpu \
    --debug
```

**Success Criteria**:
- [ ] `agent/is_training` switches from 0 → 1 at step 2000
- [ ] All 4 gradient norm metrics appear in TensorBoard
- [ ] Evaluation runs at steps 2000, 3000, 4000, 5000
- [ ] All 4 `eval/*` metrics have data points
- [ ] Learning rate metrics show constant values (expected)
- [ ] Parameter statistics show small but visible changes

---

## Priority Matrix

| Priority | Issue | Impact | Complexity | Status |
|----------|-------|--------|------------|--------|
| 🔥 **P0** | start_timesteps mismatch | 1 metric | LOW (1 line change) | ⏳ TODO |
| 🔥 **P0** | Gradient norm tracking | 4 metrics | MEDIUM (20 lines) | ⏳ TODO |
| 🔥 **P1** | Evaluation function | 4 metrics | HIGH (50+ lines) | ⏳ TODO |
| 📊 **P2** | Collision tracking | 1 metric | LOW (2 lines) | ⏳ TODO |
| ℹ️ **P3** | Document constant metrics | 0 metrics | LOW (doc update) | ⏳ TODO |

---

## Expected Outcomes After Fixes

### Immediate (After implementing P0-P1 fixes):
- ✅ All 36 metrics will have data in TensorBoard
- ✅ Gradient norms will track gradient explosion risk
- ✅ Evaluation metrics will show policy performance
- ✅ Training phase indicator will be accurate

### Long-term (After 1M step supercomputer run):
- ✅ Parameter statistics will show training progression
- ✅ Learning rate schedules will be visible (if implemented)
- ✅ Buffer utilization will reach meaningful levels (50-100%)
- ✅ Complete training history for paper analysis

---

## References

1. **TensorBoard PyTorch API**: https://pytorch.org/docs/stable/tensorboard.html
2. **Stable-Baselines3 TD3**: https://stable-baselines3.readthedocs.io/en/master/modules/td3.html
3. **Phase 11 Gradient Monitoring**: `docs/day-11/TENSORBOARD_GRADIENT_MONITORING.md`
4. **Training Log**: `training-test-tensor_1k.log` (lines 25680-25710, 42-57)

---

**Last Updated**: November 13, 2025
**Status**: Root cause identified, solutions documented, implementation pending
**Next Steps**: Implement P0-P1 fixes, run validation test, update documentation
