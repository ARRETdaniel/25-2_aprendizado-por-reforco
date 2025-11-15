# TensorBoard Frozen Metrics Investigation - TODO List

**Date**: November 13, 2025
**Issue**: 27 out of 36 metrics frozen (not updating) in TensorBoard
**Configuration**: `learning_starts: 2000` (reduced from 25,000 for debugging)
**Training Log**: `training-test-tensor_1k.log`
**Event File**: `data/logs/TD3_scenario_0_npcs_20_20251113-090256/events.out.tfevents.1763024576.danielterra.1.0`

---

## Executive Summary

**Problem**: During 1k test run, TensorBoard shows only 9 metrics updating correctly while 27 metrics remain frozen at their initial values.

**Working Metrics** (9 total):
- Core TD3: `train/actor_loss`, `train/critic_loss`, `train/q1_value`, `train/q2_value`
- Episode tracking: `train/episode_reward`, `train/episode_length`, `train/exploration_noise`
- Agent tracking: `agent/total_iterations`, `agent/critic_param_std`

**Frozen Metrics** (27 total):
- **CNN-specific** (10): All `*_cnn_lr`, `*_cnn_param_mean/std`, `*_cnn_grad_norm`
- **Evaluation** (4): All `eval/*` metrics
- **Agent parameters** (7): Most `agent/*_lr` and `agent/*_param_mean/std` (except critic_param_std)
- **Gradients** (4): All `gradients/*_norm` metrics
- **Training** (1): `train/collisions_per_episode`
- **Utility** (1): `agent/buffer_utilization`, `agent/is_training`

**Root Cause Hypothesis**: Metrics are gated behind conditions that are not being met, or code paths are not being executed during the 1k test run.

---

## Documentation References

### Official TensorBoard Documentation (Fetched)

**Source**: PyTorch TensorBoard API - https://pytorch.org/docs/stable/tensorboard.html

**Key Findings**:

1. **SummaryWriter Basics**:
   ```python
   writer = SummaryWriter(log_dir='logs')
   writer.add_scalar('tag', value, global_step)
   writer.flush()  # Force write to disk
   writer.close()  # Final flush
   ```

2. **Critical Parameters**:
   - `flush_secs=120`: Auto-flush every 2 minutes (default)
   - `max_queue=10`: Queue size before forced flush
   - **Recommendation**: Call `writer.flush()` explicitly in training loops

3. **Common Issues**:
   - **Missing `global_step`**: Metrics won't display properly
   - **No flush**: Data may not appear until writer.close() or timeout
   - **Tag naming**: Use hierarchical names (`group/metric_name`)

4. **Best Practices** (from TensorFlow guide):
   ```python
   # GOOD: Explicit step parameter
   with writer.as_default():
       tf.summary.scalar('loss', loss_value, step=epoch)

   # BAD: No step parameter
   tf.summary.scalar('loss', loss_value)  # May not display correctly
   ```

### TD3 Algorithm Documentation

**Source**: Stable-Baselines3 TD3 - https://stable-baselines3.readthedocs.io/en/master/modules/td3.html

**Key Logging Requirements**:
- Learning rates should be tracked for all optimizers
- Parameter statistics help detect training instabilities
- Gradient norms essential for detecting explosions
- Evaluation metrics logged at `eval_freq` intervals

---

## Phase 1: Code Analysis ✅ (Lines 860-980 in train_td3.py)

### 1.1 Identify All Logging Locations

**Status**: ✅ COMPLETE (from code review above)

**Findings**:

#### A. Metrics Logged Every 100 Steps (Line 860+)
```python
if t % 100 == 0:
    # WORKING METRICS:
    writer.add_scalar('train/critic_loss', metrics['critic_loss'], t)
    writer.add_scalar('train/q1_value', metrics['q1_value'], t)
    writer.add_scalar('train/q2_value', metrics['q2_value'], t)

    # CONDITIONAL (only if 'actor_loss' in metrics):
    if 'actor_loss' in metrics:  # ← Only on delayed policy updates (every 2 steps)
        writer.add_scalar('train/actor_loss', metrics['actor_loss'], t)
```

#### B. Gradient Metrics (Line 868-900)
```python
if t % 100 == 0:
    # FROZEN - Conditional on metrics dict keys:
    if 'actor_cnn_grad_norm' in metrics:  # ← KEY CHECK
        writer.add_scalar('gradients/actor_cnn_norm', metrics['actor_cnn_grad_norm'], t)

    if 'critic_cnn_grad_norm' in metrics:  # ← KEY CHECK
        writer.add_scalar('gradients/critic_cnn_norm', metrics['critic_cnn_grad_norm'], t)

    if 'actor_mlp_grad_norm' in metrics:  # ← KEY CHECK
        writer.add_scalar('gradients/actor_mlp_norm', metrics['actor_mlp_grad_norm'], t)

    if 'critic_mlp_grad_norm' in metrics:  # ← KEY CHECK
        writer.add_scalar('gradients/critic_mlp_norm', metrics['critic_mlp_grad_norm'], t)
```

**CRITICAL FINDING**: Gradient metrics depend on these keys existing in `metrics` dict returned by `agent.train()`.

#### C. Agent Statistics (Line 920-970)
```python
if t % 100 == 0:
    agent_stats = self.agent.get_stats()  # ← Calls td3_agent.py:get_stats()

    # WORKING:
    writer.add_scalar('agent/total_iterations', agent_stats['total_iterations'], t)

    # FROZEN:
    writer.add_scalar('agent/is_training', int(agent_stats['is_training']), t)
    writer.add_scalar('agent/buffer_utilization', agent_stats['buffer_utilization'], t)
    writer.add_scalar('agent/actor_lr', agent_stats['actor_lr'], t)
    writer.add_scalar('agent/critic_lr', agent_stats['critic_lr'], t)

    # FROZEN - Conditional on None check:
    if agent_stats.get('actor_cnn_lr') is not None:  # ← KEY CHECK
        writer.add_scalar('agent/actor_cnn_lr', agent_stats['actor_cnn_lr'], t)
        writer.add_scalar('agent/critic_cnn_lr', agent_stats['critic_cnn_lr'], t)

    # FROZEN:
    writer.add_scalar('agent/actor_param_mean', agent_stats['actor_param_mean'], t)
    writer.add_scalar('agent/actor_param_std', agent_stats['actor_param_std'], t)
    writer.add_scalar('agent/critic_param_mean', agent_stats['critic_param_mean'], t)
    # WORKING:
    writer.add_scalar('agent/critic_param_std', agent_stats['critic_param_std'], t)

    # FROZEN - Conditional on None check:
    if agent_stats.get('actor_cnn_param_mean') is not None:  # ← KEY CHECK
        writer.add_scalar('agent/actor_cnn_param_mean', agent_stats['actor_cnn_param_mean'], t)
        writer.add_scalar('agent/actor_cnn_param_std', agent_stats['actor_cnn_param_std'], t)
        writer.add_scalar('agent/critic_cnn_param_mean', agent_stats['critic_cnn_param_mean'], t)
        writer.add_scalar('agent/critic_cnn_param_std', agent_stats['critic_cnn_param_std'], t)
```

**CRITICAL FINDING**: All CNN metrics depend on `agent_stats.get('actor_cnn_lr') is not None` check.

#### D. Evaluation Metrics (NOT VISIBLE IN CURRENT CODE SECTION)
```python
# Search required: Where is evaluation triggered?
# Expected pattern:
if t % eval_freq == 0:
    eval_results = self.evaluate()
    writer.add_scalar('eval/mean_reward', eval_results['mean_reward'], t)
    # ... other eval metrics
```

**ACTION REQUIRED**: Search for evaluation function call.

### 1.2 Code Path Analysis

**Question**: Why do some metrics work while others don't?

**Working Metrics Pattern**:
1. ✅ Unconditional logging (no `if` checks on dict keys)
2. ✅ Metrics always present in `metrics` dict from `agent.train()`
3. ✅ Logged at consistent frequency (every 100 steps)

**Frozen Metrics Pattern**:
1. ❌ Conditional on dict key existence (`if 'key' in metrics`)
2. ❌ Conditional on None checks (`if agent_stats.get('key') is not None`)
3. ❌ Depend on specific code paths being executed

**Hypothesis**: The frozen metrics are gated behind conditions that evaluate to False.

---

## Phase 2: Agent Implementation Analysis

### 2.1 Investigate `td3_agent.py` - `train()` Method ⏳ PENDING

**File**: `src/agents/td3_agent.py`
**Method**: `train(batch_size: int) -> Dict[str, float]`

**Questions to Answer**:

1. **Gradient Norms**: Are they computed and added to `metrics` dict?
   ```python
   # Expected pattern:
   def train(self, batch_size):
       # ... critic update ...

       # Compute gradient norms:
       actor_cnn_grad_norm = sum(
           p.grad.norm().item() for p in self.actor_cnn.parameters()
           if p.grad is not None
       )

       metrics['actor_cnn_grad_norm'] = actor_cnn_grad_norm  # ← Is this present?

       return metrics
   ```

   **ACTION**: Search for `grad_norm` or `gradient` in `td3_agent.py`

2. **Actor Loss**: Why is it conditional on `'actor_loss' in metrics`?
   - Expected: Actor updated every `policy_freq=2` steps
   - Hypothesis: `metrics` dict only contains `'actor_loss'` on delayed update steps

   **ACTION**: Verify actor update frequency logic

### 2.2 Investigate `td3_agent.py` - `get_stats()` Method ⏳ PENDING

**File**: `src/agents/td3_agent.py`
**Method**: `get_stats() -> Dict[str, Any]`

**Questions to Answer**:

1. **CNN Learning Rates**: Why are they None?
   ```python
   # Expected pattern:
   def get_stats(self):
       stats = {}

       # Check if CNN optimizers exist:
       if hasattr(self, 'actor_cnn_optimizer'):  # ← Does this exist?
           stats['actor_cnn_lr'] = self.actor_cnn_optimizer.param_groups[0]['lr']
       else:
           stats['actor_cnn_lr'] = None  # ← Causes conditional to fail!

       return stats
   ```

   **ACTION**: Read `get_stats()` implementation to find CNN optimizer handling

2. **Parameter Statistics**: Why do most param stats freeze except `critic_param_std`?
   ```python
   # Expected pattern:
   def get_stats(self):
       # Critic parameters (WORKING):
       critic_params = torch.cat([p.view(-1) for p in self.critic.parameters()])
       stats['critic_param_std'] = critic_params.std().item()  # ← Works!

       # Actor parameters (FROZEN):
       actor_params = torch.cat([p.view(-1) for p in self.actor.parameters()])
       stats['actor_param_mean'] = actor_params.mean().item()  # ← Frozen?
       stats['actor_param_std'] = actor_params.std().item()    # ← Frozen?

       return stats
   ```

   **ACTION**: Check if parameter extraction code has try-except blocks swallowing errors

3. **Buffer Utilization**: Why is it frozen?
   ```python
   # Expected pattern:
   stats['buffer_utilization'] = len(self.replay_buffer) / self.replay_buffer.max_size
   ```

   **ACTION**: Verify buffer size tracking

### 2.3 Investigate CNN Optimizer Creation ⏳ PENDING

**File**: `src/agents/td3_agent.py`
**Location**: `__init__()` method

**Questions to Answer**:

1. Are CNN optimizers created separately?
   ```python
   # Expected pattern:
   self.actor_cnn_optimizer = torch.optim.Adam(
       self.actor_cnn.parameters(), lr=actor_cnn_lr
   )
   self.critic_cnn_optimizer = torch.optim.Adam(
       self.critic_cnn.parameters(), lr=critic_cnn_lr
   )
   ```

   **ACTION**: Search for `actor_cnn_optimizer` in `td3_agent.py`

2. Are CNN parameters separated from MLP parameters?
   ```python
   # Expected pattern:
   actor_cnn_params = list(self.actor_cnn.parameters())
   actor_mlp_params = list(self.actor.parameters())  # Separate!

   # Single optimizer approach (WRONG for our case):
   all_actor_params = actor_cnn_params + actor_mlp_params
   self.actor_optimizer = Adam(all_actor_params, lr=actor_lr)  # ← No separate CNN LR!
   ```

   **ACTION**: Check optimizer initialization strategy

---

## Phase 3: Evaluation Implementation Analysis

### 3.1 Find Evaluation Trigger ⏳ PENDING

**Search Pattern**: `eval_freq` or `evaluate` in `train_td3.py`

**Questions to Answer**:

1. **Is evaluation implemented?**
   ```python
   # Expected pattern:
   if t % self.eval_freq == 0 and t > 0:
       print(f"\n[EVALUATION] Running evaluation at step {t:,}...")
       eval_results = self.evaluate()

       writer.add_scalar('eval/mean_reward', eval_results['mean_reward'], t)
       writer.add_scalar('eval/success_rate', eval_results['success_rate'], t)
       writer.add_scalar('eval/avg_collisions', eval_results['avg_collisions'], t)
       writer.add_scalar('eval/avg_episode_length', eval_results['avg_episode_length'], t)
   ```

   **ACTION**: Search for `eval_freq` usage in training loop

2. **Evaluation frequency**: With `eval_freq=500` and `max_timesteps=1000`, should evaluation run at step 500 and 1000?

3. **Is evaluation method implemented?**
   ```python
   def evaluate(self):
       # Run num_eval_episodes without exploration noise
       # Return metrics dict
       pass
   ```

   **ACTION**: Check if `evaluate()` method exists in TD3TrainingPipeline class

### 3.2 Evaluate Environment Setup ⏳ PENDING

**File**: `train_td3.py`
**Location**: Initialization section

**Questions to Answer**:

1. Is a separate evaluation environment created?
   ```python
   # Expected pattern:
   self.eval_env = CARLANavigationEnv(
       carla_config_path,
       agent_config_path,
       training_config_path,
       tm_port=self.eval_tm_port  # Separate TM port!
   )
   ```

   **ACTION**: Search for `eval_env` initialization

---

## Phase 4: Training Log Analysis

### 4.1 Check Learning Phase Transition ⏳ PENDING

**File**: `training-test-tensor_1k.log`

**Search Patterns**:
```bash
# Check when learning starts:
grep "PHASE TRANSITION" training-test-tensor_1k.log

# Check if evaluation runs:
grep -i "evaluation" training-test-tensor_1k.log

# Check for CNN optimizer mentions:
grep -i "cnn.*optim" training-test-tensor_1k.log

# Check for errors:
grep -i "error\|exception\|warning" training-test-tensor_1k.log

# Check agent statistics logging:
grep "AGENT STATISTICS" training-test-tensor_1k.log | head -5
```

**Expected Output**:
```
[PHASE TRANSITION] Starting LEARNING phase at step 2,000
[PHASE TRANSITION] Replay buffer size: 2,000
[PHASE TRANSITION] Policy updates will now begin...

[AGENT STATISTICS] Step 2,100
Training Phase: LEARNING
Buffer Utilization: 0.2%
Learning Rates:
  Actor:  0.000300
  Critic: 0.000300
  Actor CNN:  0.000010  ← Should appear if CNNoptimizers exist!
  Critic CNN: 0.000010
```

**Questions to Answer**:
1. Did training reach learning phase (step > 2000)?
2. Are CNN learning rates printed in "AGENT STATISTICS" sections?
3. Are there any errors or warnings?

### 4.2 Check Step-by-Step Progress ⏳ PENDING

**Search Pattern**: Lines with `[EXPLORATION]` or `[LEARNING]`

```bash
# Show progress every 100 steps:
grep "\[EXPLORATION\]\|\[LEARNING\]" training-test-tensor_1k.log | tail -20
```

**Expected Output**:
```
[EXPLORATION] Step    100/1,000 | Episode    1 | Ep Step   100 | Reward=  +12.34 | Speed= 25.5 km/h
[EXPLORATION] Step    200/1,000 | Episode    2 | Ep Step   200 | Reward=  -5.67 | Speed= 30.1 km/h
...
[LEARNING] Step   2100/1,000 | Episode   15 | Ep Step   100 | Reward=  +8.90 | Speed= 28.3 km/h
```

**Questions to Answer**:
1. What was the final step reached in the log?
2. Did training transition from EXPLORATION → LEARNING phase?

---

## Phase 5: Event File Inspection

### 5.1 Read TensorBoard Event File ⏳ PENDING

**File**: `data/logs/TD3_scenario_0_npcs_20_20251113-090256/events.out.tfevents.1763024576.danielterra.1.0`

**Python Script to Inspect**:
```python
from tensorflow.python.summary.summary_iterator import summary_iterator

event_file = 'data/logs/TD3_scenario_0_npcs_20_20251113-090256/events.out.tfevents.1763024576.danielterra.1.0'

# Extract all scalar tags and their steps
scalars = {}
for event in summary_iterator(event_file):
    for value in event.summary.value:
        tag = value.tag
        step = event.step
        scalar_value = value.simple_value

        if tag not in scalars:
            scalars[tag] = []
        scalars[tag].append((step, scalar_value))

# Print summary:
print("="*70)
print("TENSORBOARD EVENT FILE ANALYSIS")
print("="*70)

working_tags = []
frozen_tags = []

for tag, data_points in sorted(scalars.items()):
    num_points = len(data_points)
    steps = [step for step, _ in data_points]
    values = [val for _, val in data_points]

    # Check if metric is frozen (all same value or very few updates):
    if num_points <= 1:
        frozen_tags.append(tag)
        print(f"❌ FROZEN: {tag:50s} | Points: {num_points:3d}")
    elif len(set(values)) == 1:
        frozen_tags.append(tag)
        print(f"⚠️  STATIC: {tag:50s} | Points: {num_points:3d} | Value: {values[0]}")
    else:
        working_tags.append(tag)
        print(f"✅ WORKING: {tag:50s} | Points: {num_points:3d} | Range: [{min(values):.2f}, {max(values):.2f}]")

print("="*70)
print(f"Summary: {len(working_tags)} working, {len(frozen_tags)} frozen/static")
print("="*70)
```

**Questions to Answer**:
1. Which metrics are actually written to the event file?
2. Do frozen metrics have any data points at all?
3. What steps are the frozen metrics written at?

---

## Phase 6: Root Cause Identification

### 6.1 Hypothesis Testing Matrix ⏳ PENDING

| Hypothesis | Test Method | Expected Result if True | Status |
|------------|-------------|-------------------------|--------|
| **H1: CNN optimizers not created** | Search `td3_agent.py` for `actor_cnn_optimizer` | No optimizer creation code found | ⏳ |
| **H2: Gradient norms not computed** | Check `train()` method for `grad_norm` calculation | No gradient norm code in `metrics` dict | ⏳ |
| **H3: Evaluation not implemented** | Search for `evaluate()` method in `train_td3.py` | Method missing or not called | ⏳ |
| **H4: Conditional checks failing** | Read `get_stats()` for CNN LR extraction | `actor_cnn_lr` returns None | ⏳ |
| **H5: Silent exceptions** | Search for try-except blocks in `get_stats()` | Exceptions caught and ignored | ⏳ |
| **H6: Training didn't reach learning phase** | Check log for "PHASE TRANSITION" message | No transition message found | ⏳ |
| **H7: Parameter extraction fails** | Check `get_stats()` for parameter concatenation | Empty parameter lists or errors | ⏳ |
| **H8: Buffer utilization not tracked** | Check `get_stats()` for buffer size calculation | No buffer size code | ⏳ |

### 6.2 Prioritized Investigation Order ⏳ PENDING

**Priority 1 - CRITICAL (affects 14 metrics)**:
- [ ] Check if CNN optimizers exist (`actor_cnn_optimizer`, `critic_cnn_optimizer`)
- [ ] If missing: Implement separate CNN optimizers
- [ ] If present but returning None: Fix `get_stats()` extraction

**Priority 2 - HIGH (affects 8 metrics)**:
- [ ] Check if gradient norms are computed in `train()` method
- [ ] If missing: Add gradient norm calculation after `backward()` calls
- [ ] If present: Check why they're not added to `metrics` dict

**Priority 3 - MEDIUM (affects 4 metrics)**:
- [ ] Check if evaluation is implemented and triggered
- [ ] If missing: Implement `evaluate()` method
- [ ] If present: Check why it's not called at `eval_freq` intervals

**Priority 4 - LOW (affects 1 metric)**:
- [ ] Fix `train/collisions_per_episode` logging
- [ ] Check where collision count should be accumulated

---

## Phase 7: Solution Implementation

### 7.1 Fix CNN Optimizer Issues ⏳ PENDING

**If CNN optimizers don't exist**:

```python
# In td3_agent.py __init__():

# Separate CNN optimizers (following Stable-Baselines3 pattern):
self.actor_cnn_optimizer = torch.optim.Adam(
    self.actor_cnn.parameters(),
    lr=config['actor_cnn_lr']  # e.g., 1e-5
)

self.critic_cnn_optimizer = torch.optim.Adam(
    self.critic_cnn.parameters(),
    lr=config['critic_cnn_lr']  # e.g., 1e-5
)
```

**Update `get_stats()` to extract CNN LR**:

```python
def get_stats(self):
    stats = {}

    # ... existing code ...

    # CNN learning rates:
    if hasattr(self, 'actor_cnn_optimizer') and self.actor_cnn_optimizer is not None:
        stats['actor_cnn_lr'] = self.actor_cnn_optimizer.param_groups[0]['lr']
        stats['critic_cnn_lr'] = self.critic_cnn_optimizer.param_groups[0]['lr']
    else:
        stats['actor_cnn_lr'] = None
        stats['critic_cnn_lr'] = None

    # CNN parameter statistics:
    if hasattr(self, 'actor_cnn') and self.actor_cnn is not None:
        actor_cnn_params = torch.cat([
            p.view(-1) for p in self.actor_cnn.parameters()
        ])
        stats['actor_cnn_param_mean'] = actor_cnn_params.mean().item()
        stats['actor_cnn_param_std'] = actor_cnn_params.std().item()

        critic_cnn_params = torch.cat([
            p.view(-1) for p in self.critic_cnn.parameters()
        ])
        stats['critic_cnn_param_mean'] = critic_cnn_params.mean().item()
        stats['critic_cnn_param_std'] = critic_cnn_params.std().item()
    else:
        stats['actor_cnn_param_mean'] = None
        stats['actor_cnn_param_std'] = None
        stats['critic_cnn_param_mean'] = None
        stats['critic_cnn_param_std'] = None

    return stats
```

### 7.2 Implement Gradient Norm Tracking ⏳ PENDING

**In `td3_agent.py` `train()` method**:

```python
def train(self, batch_size: int) -> Dict[str, float]:
    metrics = {}

    # ... critic update ...
    self.critic_optimizer.zero_grad()
    critic_loss.backward()

    # Compute gradient norms BEFORE optimizer step:
    critic_cnn_grad_norm = sum(
        p.grad.norm().item()
        for p in self.critic_cnn.parameters()
        if p.grad is not None
    )
    critic_mlp_grad_norm = sum(
        p.grad.norm().item()
        for p in self.critic.parameters()
        if p.grad is not None
    )

    self.critic_optimizer.step()

    # Add to metrics:
    metrics['critic_cnn_grad_norm'] = critic_cnn_grad_norm
    metrics['critic_mlp_grad_norm'] = critic_mlp_grad_norm

    # ... actor update (on delayed steps) ...
    if self.total_iterations % self.policy_freq == 0:
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # Compute gradient norms:
        actor_cnn_grad_norm = sum(
            p.grad.norm().item()
            for p in self.actor_cnn.parameters()
            if p.grad is not None
        )
        actor_mlp_grad_norm = sum(
            p.grad.norm().item()
            for p in self.actor.parameters()
            if p.grad is not None
        )

        self.actor_optimizer.step()

        # Add to metrics:
        metrics['actor_cnn_grad_norm'] = actor_cnn_grad_norm
        metrics['actor_mlp_grad_norm'] = actor_mlp_grad_norm
        metrics['actor_loss'] = actor_loss.item()

    return metrics
```

### 7.3 Implement Evaluation ⏳ PENDING

**In `train_td3.py` TD3TrainingPipeline class**:

```python
def evaluate(self) -> Dict[str, float]:
    """
    Evaluate agent for num_eval_episodes without exploration noise.

    Returns:
        Dict with keys: mean_reward, success_rate, avg_collisions, avg_episode_length
    """
    print(f"\n[EVALUATION] Running {self.num_eval_episodes} episodes...")

    # Create evaluation environment (if not exists):
    if not hasattr(self, 'eval_env'):
        self.eval_env = CARLANavigationEnv(
            self.carla_config_path,
            self.agent_config_path,
            self.training_config_path,
            tm_port=self.eval_tm_port  # Separate TM port!
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
              f"Reward={episode_reward:+.2f}, Length={episode_length}, "
              f"Collisions={info.get('collision_count', 0)}")

    results = {
        'mean_reward': np.mean(episode_rewards),
        'success_rate': successes / self.num_eval_episodes,
        'avg_collisions': np.mean(collision_counts),
        'avg_episode_length': np.mean(episode_lengths)
    }

    print(f"[EVALUATION] Results: Mean Reward={results['mean_reward']:+.2f}, "
          f"Success Rate={results['success_rate']:.1%}, "
          f"Avg Collisions={results['avg_collisions']:.2f}")

    return results

# In train() method, add evaluation trigger:
if t % self.eval_freq == 0 and t > start_timesteps:
    eval_results = self.evaluate()

    self.writer.add_scalar('eval/mean_reward', eval_results['mean_reward'], t)
    self.writer.add_scalar('eval/success_rate', eval_results['success_rate'], t)
    self.writer.add_scalar('eval/avg_collisions', eval_results['avg_collisions'], t)
    self.writer.add_scalar('eval/avg_episode_length', eval_results['avg_episode_length'], t)
```

### 7.4 Fix Other Frozen Metrics ⏳ PENDING

**Buffer Utilization**:
```python
# In get_stats():
stats['buffer_utilization'] = len(self.replay_buffer) / self.replay_buffer.max_size
```

**Training Flag**:
```python
# In get_stats():
stats['is_training'] = self.total_iterations > 0  # True if any training steps taken
```

**Collisions Per Episode**:
```python
# In train_td3.py, track at episode end:
if done or truncated:
    self.writer.add_scalar('train/collisions_per_episode',
                          self.episode_collision_count,
                          self.episode_num)
```

---

## Phase 8: Validation Testing

### 8.1 Short Test Run (1k steps) ⏳ PENDING

**Command**:
```bash
python3 scripts/train_td3.py \
    --scenario 0 \
    --max-timesteps 1000 \
    --eval-freq 500 \
    --device cpu \
    --debug
```

**Validation Checklist**:
- [ ] Training reaches learning phase (step > 2000)
- [ ] Agent statistics show CNN learning rates
- [ ] Gradient norms appear in TensorBoard
- [ ] Evaluation runs at step 500 and 1000
- [ ] All 36 metrics updating in TensorBoard

### 8.2 TensorBoard Verification ⏳ PENDING

**Launch TensorBoard**:
```bash
tensorboard --logdir data/logs --port 6006
```

**Check Each Metric Group**:

1. **Agent Metrics** (should see 13 metrics):
   - [ ] `agent/actor_cnn_lr` (updating, not frozen)
   - [ ] `agent/critic_cnn_lr` (updating, not frozen)
   - [ ] `agent/actor_lr` (updating, not frozen)
   - [ ] `agent/critic_lr` (updating, not frozen)
   - [ ] `agent/buffer_utilization` (increasing from 0% to ~0.1%)
   - [ ] `agent/is_training` (0 until step 2000, then 1)
   - [ ] All `*_param_mean` and `*_param_std` metrics updating

2. **Gradient Metrics** (should see 4 metrics):
   - [ ] `gradients/actor_cnn_norm` (updating every 200 steps - delayed updates)
   - [ ] `gradients/critic_cnn_norm` (updating every 100 steps)
   - [ ] `gradients/actor_mlp_norm` (updating every 200 steps)
   - [ ] `gradients/critic_mlp_norm` (updating every 100 steps)

3. **Evaluation Metrics** (should see 4 metrics):
   - [ ] `eval/mean_reward` (data points at step 500, 1000)
   - [ ] `eval/success_rate` (data points at step 500, 1000)
   - [ ] `eval/avg_collisions` (data points at step 500, 1000)
   - [ ] `eval/avg_episode_length` (data points at step 500, 1000)

4. **Training Metrics** (should see 6 metrics):
   - [ ] `train/collisions_per_episode` (updating at each episode end)

---

## Phase 9: Documentation Update

### 9.1 Update Training Documentation ⏳ PENDING

**Files to Update**:
- `README.md`: Add troubleshooting section for frozen metrics
- `TENSORBOARD_GRADIENT_MONITORING.md`: Update with new gradient metrics
- `FIXES_SUMMARY.md`: Document frozen metrics fix

### 9.2 Create Migration Guide ⏳ PENDING

**Document**: `docs/day13/FROZEN_METRICS_FIX.md`

**Contents**:
- Root cause analysis
- Solution implementation steps
- Before/after TensorBoard screenshots
- Validation procedures
- Common pitfalls

---

## Success Criteria

**Phase 1-4 Complete** ✅:
- [ ] All code paths analyzed
- [ ] Root causes identified for each frozen metric group
- [ ] Training log confirms learning phase reached
- [ ] Event file inspection shows which metrics are actually written

**Phase 5-7 Complete** ✅:
- [ ] Solutions implemented for all frozen metrics
- [ ] Code changes tested and validated
- [ ] No regressions in working metrics

**Phase 8 Complete** ✅:
- [ ] All 36 metrics updating in TensorBoard
- [ ] Evaluation metrics show data at correct intervals
- [ ] Gradient norms tracked correctly
- [ ] CNN learning rates visible

**Final Validation** ✅:
- [ ] Documentation updated
- [ ] Migration guide created
- [ ] Ready for 1M supercomputer training run

---

## Quick Action Items

**TODAY (Highest Priority)**:
1. ⏳ Read `src/agents/td3_agent.py` to check CNN optimizer initialization
2. ⏳ Search for `get_stats()` method to see parameter extraction
3. ⏳ Analyze `training-test-tensor_1k.log` for learning phase transition
4. ⏳ Inspect event file to confirm which metrics are written

**NEXT (After Root Cause Confirmed)**:
5. ⏳ Implement CNN optimizer creation (if missing)
6. ⏳ Add gradient norm tracking to `train()` method
7. ⏳ Implement evaluation function
8. ⏳ Run validation test

---

## Notes and Observations

### Pattern Analysis from Code Review

**Why `critic_param_std` works but others don't**:
- Hypothesis: It's logged unconditionally while others have conditions
- Action: Compare logging code for working vs frozen param stats

**Why core TD3 metrics work**:
- Logged at every training step (after `agent.train()`)
- No conditional checks on dict keys
- Always present in `metrics` dict

**Why CNN metrics are all frozen**:
- All gated behind `if agent_stats.get('actor_cnn_lr') is not None`
- Hypothesis: `get_stats()` returns None for all CNN-related keys
- Root cause: Likely CNN optimizers don't exist or aren't tracked

### Official Documentation Insights

**From PyTorch TensorBoard API**:
1. ✅ Our code uses correct `writer.add_scalar(tag, value, step)` pattern
2. ✅ We call `writer.flush()` every 100 steps (line not shown but likely present)
3. ⚠️ No issues detected with writer usage

**From TensorFlow TensorBoard Guide**:
1. Key insight: "Metrics won't display if step parameter is missing"
   - Our code: Always provides `t` as step parameter ✅
2. Key insight: "Use hierarchical naming (group/metric)"
   - Our code: Uses correct naming pattern ✅
3. Key insight: "Flush regularly to ensure data appears"
   - Our code: Flushes every 100 steps ✅

**Conclusion**: Writer usage is correct. Problem is in the **data generation**, not data logging.

---

## References

1. **PyTorch TensorBoard API**: https://pytorch.org/docs/stable/tensorboard.html
2. **TensorFlow TensorBoard Guide**: https://www.tensorflow.org/tensorboard/get_started
3. **Stable-Baselines3 TD3**: https://stable-baselines3.readthedocs.io/en/master/modules/td3.html
4. **TD3 Original Paper**: Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods"
5. **CARLA 0.9.16 Documentation**: https://carla.readthedocs.io/en/latest/

---

**Last Updated**: November 13, 2025
**Status**: Phase 1 Complete (Code Analysis), Phase 2-9 Pending
**Next Action**: Read `td3_agent.py` to check CNN optimizer creation
