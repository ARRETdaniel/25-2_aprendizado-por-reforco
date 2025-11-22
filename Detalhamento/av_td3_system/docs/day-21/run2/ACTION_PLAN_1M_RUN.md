# Action Plan: Next Steps for 1M Production Run

**Date**: 2025-01-21
**Status**: 5K Validation Complete ✅ | Ready for Extended Validation ⏭️

---

## Summary of 5K Analysis

✅ **CNN LayerNorm Fix**: VALIDATED
- Feature explosion eliminated (7.36×10¹² → 15-30)
- Gradients stable, no explosions

⚠️ **Training Dynamics**: EXPLAINED (Exploration Phase)
- Reward "degradation" explained by only 4K learning steps
- Q-value growth expected when learning exploration returns
- Need extended validation to confirm improvement

📋 **Decision**: PROCEED TO 1M with comprehensive monitoring

---

## Immediate Actions (Next 4 Hours)

### 1. Check Exploration Phase Logs ✅ (5 minutes)

**Goal**: Verify our hypothesis that 5K run was still in exploration phase

```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento

# Check phase transitions
grep "EXPLORATION\|LEARNING" av_td3_system/docs/day-21/run2/run-CNNfixes_post_all_fixes.log | head -20
grep "EXPLORATION\|LEARNING" av_td3_system/docs/day-21/run2/run-CNNfixes_post_all_fixes.log | tail -20

# Verify learning_starts configuration was 1000
grep "learning_starts" av_td3_system/config/td3_config.yaml
```

**Expected Outcome**:
- Logs show "EXPLORATION" for steps 1-1000
- Logs show "LEARNING" for steps 1001-5000
- Only 4K training steps explains concerning patterns

---

### 2. Analyze Reward Components ⏭️ (30 minutes)

**Goal**: Ensure reward function aligns with task objectives

**Option A**: If reward components are logged:
```bash
# Extract reward breakdown
grep "reward_components\|efficiency_reward\|safety_penalty\|comfort_penalty" \
  av_td3_system/docs/day-21/run2/run-CNNfixes_post_all_fixes.log > reward_breakdown.txt

# Analyze patterns
python -c "
import re
with open('reward_breakdown.txt') as f:
    data = f.read()
    # Extract and summarize reward components
    # [Add simple parsing script]
"
```

**Option B**: If not logged, check reward function code:
```bash
# Find reward function implementation
grep -n "def calculate_reward\|def compute_reward" av_td3_system/src/environment/carla_env.py
```

**What to Look For**:
- Safety penalties dominating (causing over-conservative behavior)
- Efficiency rewards too weak (agent not motivated to progress)
- Comfort penalties too high (agent moving slowly)

---

### 3. Run 10K Validation ⏭️ (2 hours)

**Goal**: Confirm rewards improve after exploration phase ends

```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento

# Start 10K run
python av_td3_system/scripts/train_td3.py \
  --scenario 0 \
  --max-timesteps 10000 \
  --eval-freq 2500 \
  --seed 42 \
  --debug \
  --log-dir av_td3_system/data/logs/TD3_10k_validation_$(date +%Y%m%d-%H%M%S) \
  --save-freq 2500
```

**Monitoring During Run**:
```bash
# In separate terminal, monitor TensorBoard
tensorboard --logdir av_td3_system/data/logs/

# Watch for:
# 1. Episode rewards improving after step 5K
# 2. Q-values stabilizing (variance decreasing)
# 3. TD errors decreasing
# 4. No critical alerts
```

**Success Criteria**:
- ✅ Episode rewards > 76 (5K baseline) by step 10K
- ✅ Q-value std decreasing (convergence signal)
- ✅ TD errors < 5 on average
- ✅ No gradient explosions

**Failure Criteria** (HALT IF):
- ❌ Episode rewards < 50 by step 10K (worse degradation)
- ❌ Q-values > 200 (divergence)
- ❌ TD errors > 20 consistently
- ❌ Gradient explosion alerts

---

## Short-Term Actions (Next 24 Hours)

### 4. Analyze 10K Results ⏭️ (1 hour)

**After 10K run completes**:

```bash
# Extract final metrics
python -c "
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

# Load TensorBoard events
ea = event_accumulator.EventAccumulator('av_td3_system/data/logs/TD3_10k_validation_*/events.out.tfevents.*')
ea.Reload()

# Get episode rewards
rewards = [e.value for e in ea.Scalars('train/episode_reward')]

print(f'Episode Rewards (steps 5K-10K):')
print(f'  Mean: {np.mean(rewards[-50:]):.2f}')
print(f'  Std: {np.std(rewards[-50:]):.2f}')
print(f'  Min: {np.min(rewards[-50:]):.2f}')
print(f'  Max: {np.max(rewards[-50:]):.2f}')

# Compare with 5K baseline
baseline_5k = 75.67
current_10k = np.mean(rewards[-50:])
improvement = current_10k - baseline_5k
print(f'\nImprovement from 5K: {improvement:+.2f} ({improvement/baseline_5k*100:+.1f}%)')
"
```

**Decision Point**:
- **IF improvement > +10%**: ✅ Proceed to 100K run
- **IF improvement -10% to +10%**: ⚠️ Extend to 20K for more data
- **IF improvement < -10%**: ❌ HALT and investigate reward function

---

### 5. Compare with DDPG Baseline ⏭️ (4 hours)

**Goal**: Validate TD3 improvements over DDPG

```bash
# Train DDPG for 10K steps
python av_td3_system/scripts/train_ddpg.py \
  --scenario 0 \
  --max-timesteps 10000 \
  --eval-freq 2500 \
  --seed 42 \
  --debug
```

**Compare**:
| Metric | TD3 (10K) | DDPG (10K) | TD3 Advantage |
|--------|-----------|------------|---------------|
| Episode Reward | [TO FILL] | [TO FILL] | Should be higher |
| Q-Value Std | [TO FILL] | [TO FILL] | Should be lower (less overestimation) |
| Training Stability | [TO FILL] | [TO FILL] | Should be more stable |

**Expected**: TD3 should show:
- ✅ Higher/more stable rewards (clipped double-Q reduces overestimation)
- ✅ Lower Q-value variance (better convergence)
- ✅ Fewer gradient spikes (delayed policy updates)

---

## Medium-Term Actions (Next 48-72 Hours)

### 6. Implement Monitoring for 1M Run ⏭️ (2 hours)

**Add to `train_td3.py`**:

```python
# Real-time anomaly detection
class TrainingMonitor:
    def __init__(self):
        self.best_reward_100k = -float('inf')

    def check_anomalies(self, metrics, step):
        # Alert on TD error explosion
        if metrics['td_error'] > 50:
            logger.critical(f"[ALERT] TD error explosion at step {step}: {metrics['td_error']:.2f}")
            return 'halt'

        # Alert on Q-value divergence
        if metrics['q_value'] > 1000:
            logger.critical(f"[ALERT] Q-value divergence at step {step}: {metrics['q_value']:.2f}")
            return 'halt'

        # Alert on reward collapse
        if metrics['episode_reward'] < -500:
            logger.critical(f"[ALERT] Reward collapse at step {step}: {metrics['episode_reward']:.2f}")
            return 'halt'

        # Check for improvement stagnation (every 100K)
        if step % 100000 == 0:
            current_reward = np.mean(recent_rewards[-100:])
            if current_reward < self.best_reward_100k - 10:  # Threshold: -10
                logger.warning(f"[WARNING] No improvement in last 100K steps")
                return 'checkpoint'
            self.best_reward_100k = max(self.best_reward_100k, current_reward)

        return 'continue'

# In training loop
monitor = TrainingMonitor()
for t in range(max_timesteps):
    # ... training step ...

    if t % 100 == 0:  # Check every 100 steps
        status = monitor.check_anomalies(metrics, t)
        if status == 'halt':
            save_checkpoint(f'emergency_halt_{t}.pt')
            break
        elif status == 'checkpoint':
            save_checkpoint(f'checkpoint_{t}.pt')
```

---

### 7. Enhanced Checkpointing ⏭️ (30 minutes)

```python
# Save every 25K steps, keep last 5 checkpoints
class CheckpointManager:
    def __init__(self, max_keep=5):
        self.checkpoints = []
        self.max_keep = max_keep

    def save(self, step, agent, path_template):
        checkpoint_path = path_template.format(step=step)
        agent.save(checkpoint_path)
        self.checkpoints.append(checkpoint_path)

        # Keep only last max_keep checkpoints
        if len(self.checkpoints) > self.max_keep:
            old_checkpoint = self.checkpoints.pop(0)
            os.remove(old_checkpoint)
            logger.info(f"Removed old checkpoint: {old_checkpoint}")

        logger.info(f"Saved checkpoint: {checkpoint_path}")

# In training loop
ckpt_mgr = CheckpointManager(max_keep=5)
for t in range(max_timesteps):
    # ... training step ...

    if t % 25000 == 0:  # Every 25K
        ckpt_mgr.save(t, agent, 'checkpoints/td3_{step}k.pt')
```

---

### 8. Extended 100K Validation ⏭️ (20-40 hours)

**ONLY IF 10K shows improvement**

```bash
python av_td3_system/scripts/train_td3.py \
  --scenario 0 \
  --max-timesteps 100000 \
  --eval-freq 5000 \
  --save-freq 25000 \
  --seed 42 \
  --debug \
  --enable-monitoring \
  --log-dir av_td3_system/data/logs/TD3_100k_extended_$(date +%Y%m%d-%H%M%S)
```

**Track Metrics Every 5K**:
```bash
# In separate terminal
python scripts/track_metrics.py \
  --log-dir av_td3_system/data/logs/TD3_100k_extended_* \
  --metrics episode_reward q_value td_error critic_loss \
  --window 1000 \
  --update-freq 300  # Update every 5 minutes
```

**Checkpoints to Evaluate**:
- ✅ 10K: Compare with 5K baseline
- ✅ 25K: Assess trend direction
- ✅ 50K: Mid-point check
- ✅ 75K: Confirm continued improvement
- ✅ 100K: Final validation before 1M

---

## Long-Term Action (1M Production Run)

### 9. 1M Production Training ⏭️ (24-72 hours)

**ONLY IF 100K shows**:
- ✅ Episode rewards improving (upward trend)
- ✅ Q-values stabilizing (variance decreasing)
- ✅ TD errors decreasing (convergence)
- ✅ No critical alerts

**Launch Command**:
```bash
# Use tmux/screen for long-running job
tmux new-session -s td3_1m

python av_td3_system/scripts/train_td3.py \
  --scenario 0 \
  --max-timesteps 1000000 \
  --eval-freq 10000 \
  --save-freq 25000 \
  --seed 42 \
  --debug \
  --enable-monitoring \
  --enable-early-stopping \
  --log-dir av_td3_system/data/logs/TD3_1M_production_$(date +%Y%m%d-%H%M%S) \
  2>&1 | tee av_td3_system/docs/logs/1M_run.log
```

**Monitoring Setup**:
```bash
# Terminal 1: TensorBoard
tensorboard --logdir av_td3_system/data/logs/ --port 6006

# Terminal 2: Real-time log tail
tail -f av_td3_system/docs/logs/1M_run.log | grep -E "ALERT|WARNING|Episode|reward"

# Terminal 3: Resource monitoring
watch -n 30 'nvidia-smi && echo "---" && df -h | grep -E "Filesystem|av_td3_system"'
```

**Daily Checklist**:
- [ ] Check for anomaly alerts (TD error, Q-value, reward)
- [ ] Review episode reward trend (should be improving)
- [ ] Verify disk space (checkpoints every 25K = ~1GB each)
- [ ] Ensure GPU utilization stable (70-90%)
- [ ] Compare against DDPG baseline at same timestep

**Abort Criteria**:
- ❌ Episode rewards degrading continuously for 100K+ steps
- ❌ Q-values exceeding 1000 (divergence)
- ❌ TD errors exceeding 50 consistently
- ❌ 3+ critical alerts in 50K steps
- ❌ System running out of disk space

---

## Success Metrics for 1M Run

**Target Performance** (based on SB3 benchmarks, adjusted for CARLA):

| Metric | Target (1M) | Baseline (5K) | Improvement |
|--------|-------------|---------------|-------------|
| **Episode Reward** | > 500 | 75.67 | +565% |
| **Success Rate** | > 80% | Unknown | N/A |
| **Q-Value** | 50-100 (stable) | 33.20 (growing) | Stabilized |
| **TD Error** | < 5 | 3.63 (growing) | Converged |
| **Critic Loss** | < 50 | 93.32 (high var) | Stabilized |

**Comparison with DDPG**:
- TD3 episode rewards should be ≥10% higher
- TD3 Q-value variance should be ≥20% lower
- TD3 training should be more stable (fewer spikes)

---

## Contingency Plans

### IF 10K Shows No Improvement:
1. **Check Reward Function**:
   - Analyze reward components
   - Ensure safety penalties not over-penalizing
   - Verify efficiency rewards sufficient

2. **Tune Hyperparameters**:
   - Try learning rates: [1e-4, 3e-4, 1e-3]
   - Try exploration noise: [0.05, 0.1, 0.2]
   - Try batch sizes: [128, 256, 512]

3. **Extended Exploration**:
   - Increase `learning_starts` to 2500-5000
   - Allow more random exploration before training

### IF 100K Shows Divergence:
1. **Rollback to Last Good Checkpoint**:
   - Identify last stable checkpoint (e.g., 50K)
   - Resume training with adjusted hyperparameters

2. **Reduce Learning Rates**:
   - Halve actor_lr and critic_lr
   - Increase batch size for more stable gradients

3. **Increase Target Network Update Rate**:
   - Reduce `tau` from 0.005 to 0.001 (slower target updates)
   - Reduce learning oscillations

### IF 1M Completes Successfully:
1. **Comprehensive Evaluation**:
   - Test on multiple scenarios (different NPCs, weather)
   - Compare with DDPG, IDM+MOBIL baselines
   - Analyze safety metrics (collision rate, TTC)

2. **Ablation Studies**:
   - Remove LayerNorm → Verify feature explosion returns
   - Use single critic (DDPG) → Validate clipped double-Q benefit
   - Disable delayed updates → Validate policy delay benefit

3. **Document for Paper**:
   - Methods: LayerNorm implementation, hyperparameters
   - Results: Performance comparison, ablation studies
   - Discussion: Lessons learned, limitations, future work

---

## Timeline Estimate

| Phase | Duration | Cumulative |
|-------|----------|------------|
| **1-3: Immediate** (Logs + Rewards + 10K) | 4 hours | 4h |
| **4-5: Short-term** (Analyze + DDPG) | 6 hours | 10h |
| **6-7: Monitoring Setup** | 3 hours | 13h |
| **8: Extended 100K** | 40 hours | 53h (~2.2 days) |
| **9: Production 1M** | 72 hours | 125h (~5.2 days) |
| **Total** | ~5-6 days | From start to 1M completion |

**Critical Path**:
```
Day 1: 10K validation + DDPG baseline
Day 2: Implement monitoring + start 100K
Day 3-4: Monitor 100K run
Day 5-7: 1M production run (if 100K successful)
```

---

## Key Deliverables

### Documentation:
- ✅ POST_CNN_FIX_METRICS_ANALYSIS.md (Complete)
- ✅ EXECUTIVE_SUMMARY_5K_VALIDATION.md (Complete)
- ⏭️ 10K_VALIDATION_RESULTS.md
- ⏭️ 100K_EXTENDED_VALIDATION.md
- ⏭️ 1M_PRODUCTION_RESULTS.md

### Code:
- ⏭️ Enhanced monitoring (`train_td3.py`)
- ⏭️ Checkpoint manager (`train_td3.py`)
- ⏭️ Real-time metrics tracker (`scripts/track_metrics.py`)

### Data:
- ✅ 5K run logs and TensorBoard events
- ⏭️ 10K validation checkpoints
- ⏭️ 100K extended validation checkpoints
- ⏭️ 1M production final model

### Paper:
- ⏭️ Methods section (LayerNorm, training procedure)
- ⏭️ Results section (performance comparison, ablation)
- ⏭️ Figures (learning curves, Q-value convergence, etc.)

---

## Contact Points for Issues

**If CNN features explode again**:
- Check LayerNorm still in forward pass
- Verify gradients being clipped
- Rollback to last stable checkpoint

**If training hangs**:
- Check CARLA server still running
- Check disk space not full
- Check GPU memory not exhausted

**If results look wrong**:
- Verify TensorBoard logging working
- Check seed consistency
- Compare with DDPG baseline

**If unsure whether to halt**:
- Save emergency checkpoint
- Pause training (Ctrl+Z)
- Analyze last 1K steps trends
- Make informed go/no-go decision

---

**Document Version**: 1.0
**Last Updated**: 2025-01-21
**Next Review**: After 10K validation
**Status**: ⏭️ READY TO EXECUTE (awaiting user confirmation)
