# 🎯 ACTION PLAN: TD3 Hyperparameter Fixes & Validation
**Date**: November 20, 2025
**Priority**: 🔴 CRITICAL - BLOCKING 1M TRAINING
**Estimated Time**: 4-6 hours (implementation + validation)

---

## 🚨 PROBLEM STATEMENT

After comprehensive review of the TD3 paper (Fujimoto et al., 2018) and our implementation, we discovered:

**✅ CORRECT**: Our implementation matches TD3 Algorithm 1 exactly (1:1 correspondence)
**❌ WRONG**: Our hyperparameters deviate significantly from paper's validated settings
**⚠️ IMPACT**: TD3's built-in protections (Clipped Double Q-learning, delayed updates) may not work correctly with wrong hyperparameters

---

## 📋 HYPERPARAMETER MISMATCHES

| Parameter | Paper (MuJoCo) | Current (CARLA) | Difference | Impact |
|-----------|----------------|-----------------|------------|---------|
| **batch_size** | 100 | 256 | 2.56× LARGER | Faster convergence, less exploration, potentially to wrong Q-estimates |
| **discount (γ)** | 0.99 | 0.9 | 10% LOWER | Shorter horizon (10 steps vs 100), myopic policy |
| **tau (τ)** | 0.005 | 0.001 | 5× SLOWER | Target networks lag, actor-target divergence |
| **critic_lr** | 1e-3 | 3e-4 | 3.3× SLOWER | Q-surface doesn't adapt fast enough to policy changes |

---

## 🎯 SOLUTION #1: MATCH PAPER HYPERPARAMETERS EXACTLY

### Files to Modify

#### File 1: `av_td3_system/config/td3_config.yaml`

**Current Values**:
```yaml
training:
  batch_size: 256
  discount: 0.9
  tau: 0.001

  learning_rates:
    critic_mlp: 3e-4
    critic_cnn: 1e-4
```

**NEW Values** (match paper):
```yaml
training:
  batch_size: 100          # CHANGE: 256 → 100 (2.56× reduction)
  discount: 0.99           # CHANGE: 0.9 → 0.99 (restore standard discount)
  tau: 0.005               # CHANGE: 0.001 → 0.005 (5× faster target updates)

  learning_rates:
    critic_mlp: 1e-3       # CHANGE: 3e-4 → 1e-3 (3.3× faster critic learning)
    critic_cnn: 1e-4       # KEEP: No change (paper doesn't use CNN)
```

**Justification**:
- Paper's hyperparameters are **empirically validated** across 7 MuJoCo tasks
- Our deviations are **unjustified** without ablation studies
- Need to **start from known-good baseline** before tuning for CARLA

---

### Implementation Steps

#### Step 1: Locate Config File

```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system
find . -name "td3_config.yaml" -o -name "config.yaml" -o -name "hyperparameters.yaml"
```

**Expected locations**:
- `config/td3_config.yaml`
- `src/config/td3_config.yaml`
- `hyperparameters.yaml`

---

#### Step 2: Backup Current Config

```bash
# Find config file location
CONFIG_FILE=$(find . -name "*config*.yaml" | grep -i td3 | head -n 1)

# Backup with timestamp
cp $CONFIG_FILE ${CONFIG_FILE}.backup_$(date +%Y%m%d_%H%M%S)

# Verify backup
ls -la ${CONFIG_FILE}*
```

---

#### Step 3: Edit Config File

**Manual Edit** (if no automated config update script):
```bash
nano $CONFIG_FILE
```

**OR Automated** (if using Python config):
```python
import yaml

# Load config
with open('config/td3_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Update hyperparameters
config['training']['batch_size'] = 100
config['training']['discount'] = 0.99
config['training']['tau'] = 0.005
config['training']['learning_rates']['critic_mlp'] = 1e-3

# Save updated config
with open('config/td3_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("✅ Config updated successfully!")
```

---

#### Step 4: Verify Changes

```bash
# Display updated config
cat $CONFIG_FILE | grep -A 10 "training:"

# Expected output:
# training:
#   batch_size: 100
#   discount: 0.99
#   tau: 0.005
#   learning_rates:
#     critic_mlp: 0.001
```

---

## 🧪 VALIDATION PLAN: 50K STEPS

### Objective

**Run 50K training steps** to validate that:
1. ✅ System doesn't crash or diverge with new hyperparameters
2. ✅ Q-values follow reasonable trajectory (not exploding to >1000)
3. ✅ Actor-critic divergence remains bounded (<10×)
4. ✅ Episode rewards show learning signal (even if slow)

### Training Command

```bash
# Navigate to project root
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

# Activate environment (if using conda/venv)
conda activate carla_td3  # OR: source venv/bin/activate

# Launch training
python scripts/train_td3.py \
  --max_timesteps 50000 \
  --scenario 0 \
  --npcs 20 \
  --eval_freq 5000 \
  --save_freq 10000 \
  --log_interval 500 \
  --experiment_name "td3_50k_paper_hyperparams" \
  --seed 42
```

**Arguments**:
- `--max_timesteps 50000`: 10× longer than previous run (1,700 steps)
- `--eval_freq 5000`: Match paper's evaluation frequency
- `--save_freq 10000`: Save checkpoints every 10K steps
- `--log_interval 500`: Log metrics every 500 steps (for debugging)
- `--experiment_name`: Distinguish from previous runs
- `--seed 42`: Reproducibility

---

### Expected Duration

**Estimation** (based on previous run):
- Previous run: 1,700 steps in 5 minutes 2 seconds
- Rate: 1,700 / 302s = 5.63 steps/second
- 50K steps at 5.63 steps/s = **8,865 seconds = 2.46 hours**

**Buffer**: Add 20% for startup/shutdown overhead → **~3 hours total**

---

### Monitoring Checklist

#### Every 5K Steps (Evaluation Points)

**Check TensorBoard**:
```bash
# Launch TensorBoard (separate terminal)
tensorboard --logdir av_td3_system/runs/td3_50k_paper_hyperparams --port 6006

# Open browser: http://localhost:6006
```

**Metrics to Monitor**:

| Metric | Expected Behavior | Red Flag |
|--------|-------------------|----------|
| `debug/actor_q_mean` | Grows slowly: 0-500 at 50K steps | >1000 at 50K |
| `debug/q1_value` | Stable: 10-100 at 50K steps | >500 at 50K |
| `debug/q2_value` | Similar to Q1 (within 2×) | Diverges >5× from Q1 |
| `debug/actor_q_std` | Moderate: 10-100 | >500 (high variance) |
| `losses/critic_loss` | Decreasing: 100 → 10 | Increasing or flat |
| `train/episode_reward` | Increasing (even slowly) | Flat or decreasing |
| `train/episode_length` | Increasing | Flat at 16-20 steps |
| `gradients/actor_mlp` | <1.0 consistently | >1.0 (clipping active) |
| `gradients/critic_mlp` | <10.0 consistently | >10.0 (clipping active) |

---

#### Every 500 Steps (Log Points)

**Check Terminal Output**:
```
[Step 500] Episode Reward: 45.2 | Episode Length: 18 | Actor Q: 23.4 | Critic Q1: 12.1
[Step 1000] Episode Reward: 52.1 | Episode Length: 22 | Actor Q: 67.8 | Critic Q1: 15.3
[Step 1500] Episode Reward: 61.3 | Episode Length: 25 | Actor Q: 102.3 | Critic Q1: 21.7
...
[Step 50000] Episode Reward: ??? | Episode Length: ??? | Actor Q: ??? | Critic Q1: ???
```

**Success Pattern**:
- Episode reward: Increasing trend (even if noisy)
- Episode length: Increasing (agent survives longer)
- Actor Q: Growing, but <500 at 50K
- Critic Q1: Stable, tracking replay buffer value

**Failure Pattern**:
- Episode reward: Flat or decreasing
- Episode length: Stuck at 16-20 steps
- Actor Q: >1000 at 50K (explosion)
- Critic Q1: Diverging wildly from Q2

---

### Success Criteria

**PASS if ALL conditions met**:
1. ✅ Training completes 50K steps without crash
2. ✅ Actor Q-mean < 500 at 50K steps
3. ✅ Actor-critic divergence < 10× (Actor Q / Critic Q < 10)
4. ✅ Episode rewards increasing (linear regression slope > 0)
5. ✅ No NaN/Inf values in any metric
6. ✅ Gradient norms remain within clipping bounds

**FAIL if ANY condition violated**:
1. ❌ Crash, hang, or infinite loop
2. ❌ Actor Q-mean > 1000 at 50K
3. ❌ Actor-critic divergence > 20×
4. ❌ Episode rewards flat or decreasing
5. ❌ NaN/Inf in Q-values or losses
6. ❌ Gradient norms consistently >1.0 (actor) or >10.0 (critic)

---

## 📊 COMPARISON WITH TD3 PAPER

### Expected Q-Value Trajectory (Based on Paper's Figure 1)

**Paper's Hopper-v1 Results** (DDPG, 1M steps):
- 0-100K steps: Q-values 0 → ~500 (slow growth)
- 100K-500K steps: Q-values 500 → ~2000 (moderate growth)
- 500K-1M steps: Q-values 2000 → ~4000 (steady growth)

**Our Expected Trajectory** (50K steps = 5% of 1M):
- 0-10K steps: Q-values 0 → ~50 (random exploration)
- 10K-30K steps: Q-values 50 → ~150 (early learning)
- 30K-50K steps: Q-values 150 → ~250 (bootstrapping)

**Validation**: Our Q-values at 50K should be **200-300** (5% of paper's 4000)

---

### Visualization Script

```python
import numpy as np
import matplotlib.pyplot as plt

# Load TensorBoard logs
from tensorboard.backend.event_processing import event_accumulator

# Load our run
ea = event_accumulator.EventAccumulator('runs/td3_50k_paper_hyperparams')
ea.Reload()

# Extract Q-values
steps = []
actor_q = []
critic_q = []

for event in ea.Scalars('debug/actor_q_mean'):
    steps.append(event.step)
    actor_q.append(event.value)

for event in ea.Scalars('debug/q1_value'):
    critic_q.append(event.value)

# Paper's Hopper trajectory (manually extracted from Figure 1)
paper_steps = np.array([0, 100000, 200000, 500000, 1000000])
paper_q = np.array([0, 500, 1000, 2500, 4000])

# Extrapolate paper to our 50K range
paper_steps_50k = np.linspace(0, 50000, 100)
paper_q_50k = np.interp(paper_steps_50k, paper_steps, paper_q)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(steps, actor_q, label='Our Actor Q', linewidth=2)
plt.plot(steps, critic_q, label='Our Critic Q', linewidth=2)
plt.plot(paper_steps_50k, paper_q_50k, label='Paper Hopper Q (extrapolated)',
         linestyle='--', linewidth=2, color='red')
plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('Q-Value', fontsize=14)
plt.title('TD3 Q-Value Trajectory: CARLA vs MuJoCo (Hopper)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('docs/day-20/q_value_comparison_50k.png', dpi=300)
print("✅ Q-value comparison plot saved!")
```

---

## 🔧 TROUBLESHOOTING

### Issue 1: Config File Not Found

**Symptom**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'config/td3_config.yaml'
```

**Solution**:
```bash
# Search entire project for config files
find . -name "*.yaml" | grep -i config

# Check common locations
ls -la config/
ls -la src/config/
ls -la av_td3_system/config/
```

---

### Issue 2: Training Crashes at Start

**Symptom**:
```
RuntimeError: CUDA out of memory
```

**Cause**: Batch size 256 → 100 should **REDUCE** memory, but if using large replay buffer:

**Solution**:
```python
# Check replay buffer size in config
replay_buffer_size: 1000000  # 1M transitions

# If OOM, reduce temporarily:
replay_buffer_size: 100000  # 100K transitions (10× smaller)
```

---

### Issue 3: Q-Values Still Exploding

**Symptom**: Actor Q-mean > 1000 at 50K steps

**Diagnosis**:
1. Check if config changes were applied:
   ```bash
   python -c "import yaml; print(yaml.safe_load(open('config/td3_config.yaml'))['training'])"
   ```
2. Check if agent is loading correct config:
   ```python
   # In td3_agent.py __init__, add:
   print(f"✅ Config loaded: batch_size={self.batch_size}, gamma={self.discount}, tau={self.tau}")
   ```

**Solution**: If config not loading, manually set in code:
```python
# In td3_agent.py __init__
self.batch_size = 100  # Force paper's value
self.discount = 0.99
self.tau = 0.005
```

---

### Issue 4: Episode Rewards Not Increasing

**Symptom**: Episode rewards flat or decreasing over 50K steps

**Diagnosis**:
1. Check if agent is learning at all (critic loss decreasing?)
2. Check if reward function is balanced (too much penalty?)
3. Check if episodes are too short (terminating before learning?)

**Solution**:
```python
# Log reward components
print(f"Reward breakdown: progress={r_progress}, lane={r_lane}, safety={r_safety}")

# Check episode termination reasons
print(f"Episode ended: collision={collision}, off_road={off_road}, timeout={timeout}")
```

---

## 📅 TIMELINE

### Day 1 (Today): Implementation & Launch

- [x] **10:00-10:30**: Read comprehensive analysis document (THIS FILE)
- [ ] **10:30-11:00**: Locate and backup config files
- [ ] **11:00-11:30**: Modify hyperparameters (batch_size, γ, τ, lr)
- [ ] **11:30-12:00**: Verify changes, test launch (1K steps)
- [ ] **12:00-12:30**: Launch 50K training run
- [ ] **12:30-15:30**: Monitor progress (3 hours runtime)

### Day 1 Evening: Analysis

- [ ] **15:30-16:00**: Collect TensorBoard logs
- [ ] **16:00-16:30**: Generate Q-value comparison plot
- [ ] **16:30-17:00**: Evaluate success/failure criteria
- [ ] **17:00-17:30**: Document results, decide next steps

---

## ✅ SUCCESS METRICS

### Immediate (50K Steps)

- [ ] Training completes without crash
- [ ] Actor Q < 500 at 50K steps
- [ ] Actor-critic divergence < 10×
- [ ] Episode rewards increasing
- [ ] No NaN/Inf values

### Long-Term (200K-1M Steps)

- [ ] Q-values stabilize (not exploding to >5,000)
- [ ] Episode success rate improving
- [ ] Agent navigates longer before collision
- [ ] Comparable to paper's learning curves

---

## 🚀 NEXT STEPS AFTER 50K

### If PASS (Success Criteria Met)

**Immediate**:
1. ✅ Continue to 200K steps
2. ✅ Evaluate at 200K (meaningful learning should be visible)
3. ✅ Compare with DDPG baseline (if available)

**Long-Term**:
1. ✅ Extend to 1M steps (full paper training)
2. ✅ Run ablation studies (test each hyperparameter individually)
3. ✅ Tune for CARLA-specific optimizations (if needed)

---

### If FAIL (Success Criteria Violated)

**Diagnosis**:
1. ⚠️ Check if config changes were applied correctly
2. ⚠️ Check if implementation has bugs (re-verify vs Algorithm 1)
3. ⚠️ Check if CARLA environment is stable (sensor data, rewards)

**Alternative Solutions**:
1. 🔴 **Solution #2**: Extend CARLA episode length (see main analysis doc)
2. 🔴 **Solution #3**: Try SAC algorithm (more stable for short episodes)
3. 🔴 **Consult related papers**: CARLA-specific DRL implementations

---

## 📞 SUPPORT & ESCALATION

### If Stuck (Criteria Not Met After 50K)

**Contact**:
- Review paper Section 5 (Experimental Details)
- Check official TD3 GitHub: https://github.com/sfujim/TD3
- Post on RL Discord/Reddit with TensorBoard logs

**Information to Provide**:
- Config file (td3_config.yaml)
- TensorBoard logs (50K steps)
- Q-value trajectory plot
- This analysis document

---

**Document Version**: 1.0
**Last Updated**: November 20, 2025
**Status**: 🟢 READY FOR IMPLEMENTATION
**Estimated Completion**: November 21, 2025 (after 50K validation)
