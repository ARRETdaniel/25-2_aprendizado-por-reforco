# Quick Fix Guide: Stop Training and Apply These Changes

**Status**: 🔴 **TRAINING FAILURE - IMMEDIATE ACTION REQUIRED**

---

## TL;DR

**Problem**: Q-values overestimating rewards by 239% (47.76 gap). Critic loss exploding (+361%). Agent NOT learning.

**Root Cause**: Reward scale TOO HIGH (progress reward = +17.5, dominates 91.7%)

**Fix**: Normalize reward scale (reduce by 10×)

---

## Immediate Actions (Do These Now!)

### 1. Stop Current Training ⏸️

```bash
# Kill the training process
pkill -f train_td3.py
```

### 2. Apply Fix #1: Normalize Reward Scale ✅

**File**: `av_td3_system/config/training_config.yaml`

```yaml
# BEFORE (WRONG):
progress:
  waypoint_bonus: 10.0
  distance_scale: 50.0  # ← TOO HIGH!

# AFTER (CORRECT):
progress:
  waypoint_bonus: 1.0   # ← Changed from 10.0
  distance_scale: 5.0   # ← Changed from 50.0 (10× reduction)
```

**File**: `av_td3_system/config/carla_config.yaml`

```yaml
# BEFORE (WRONG):
weights:
  efficiency: 1.0
  lane_keeping: 2.0
  comfort: 0.5
  safety: 1.0
  progress: 2.0  # ← TOO HIGH!

# AFTER (CORRECT):
weights:
  efficiency: 1.0
  lane_keeping: 1.0  # ← Changed from 2.0
  comfort: 0.5
  safety: 1.0
  progress: 1.0      # ← Changed from 2.0
```

**Expected Impact**:
```
Before: Progress = 0.1m × 50 × 2.0 = +10.0 per step
After:  Progress = 0.1m × 5 × 1.0 = +0.5 per step
Reduction: 20× less progress reward!

Target total reward: -1 to +5 per step
Target Q-values: 10-30 (matching MuJoCo benchmarks)
```

### 3. Verify TD3 Parameters ✅

**File**: `av_td3_system/src/agents/td3_agent.py`

Check these values (search for the variable names):

```python
# TD3 Core Parameters (from TD3 paper)
policy_delay = 2        # Actor updates every 2 critic updates
tau = 0.005            # Polyak averaging (0.5%)
actor_lr = 3e-4        # Actor learning rate (0.0003)
critic_lr = 3e-4       # Critic learning rate (0.0003)
gamma = 0.99           # Discount factor
target_noise = 0.2     # Target policy smoothing noise
noise_clip = 0.5       # Target noise clipping
```

If **ANY** of these values are different, change them to match the spec above!

### 4. Re-Run Training (Short Test) 🧪

```bash
cd av_td3_system
python src/train_td3.py \
  --max-timesteps 5000 \
  --start-timesteps 1000 \
  --config config/training_config.yaml
```

### 5. Monitor TensorBoard 📊

Open TensorBoard in a new terminal:

```bash
tensorboard --logdir av_td3_system/data/logs
```

Navigate to: http://localhost:6006

**Watch These Metrics Every 100 Steps**:

| Metric | Expected (GOOD) | Current (BAD) | Status After Fix |
|--------|----------------|---------------|------------------|
| `train/q1_value` | 10-30 | 67.76 | Should drop to 10-30 ✅ |
| `train/critic_loss` | 20-50 (↓) | 1244 (↑) | Should drop to < 100 ✅ |
| `debug/td_error_q1` | < 5 | 13.4 | Should drop to < 5 ✅ |
| `debug/reward_mean` | 15-25 | 19.99 | Should match Q-values ✅ |

**Success Criteria (After 5K Steps)**:
- ✅ Q-values: 10-30 range
- ✅ Q - R Gap: < ±5 (calculate: Q1 - reward_mean)
- ✅ Critic Loss: < 100 and **decreasing**
- ✅ TD Errors: < 5

---

## If Still Failing After Fix #1

### Additional Fix #2: Add Gradient Clipping 🛡️

**File**: `av_td3_system/src/agents/td3_agent.py`

Find the critic update section (search for `critic_optimizer.step()`):

```python
# BEFORE (NO CLIPPING):
critic_loss.backward()
self.critic_optimizer.step()

# AFTER (WITH CLIPPING):
critic_loss.backward()
torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)  # ← ADD THIS
self.critic_optimizer.step()
```

Find the actor update section (search for `actor_optimizer.step()`):

```python
# BEFORE (NO CLIPPING):
actor_loss.backward()
self.actor_optimizer.step()

# AFTER (WITH CLIPPING):
actor_loss.backward()
torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)  # ← ADD THIS
self.actor_optimizer.step()
```

### Additional Fix #3: Increase Warmup Period 🏁

**File**: `av_td3_system/src/train_td3.py`

```python
# BEFORE:
start_timesteps = 1000  # Too early, buffer only 10% full

# AFTER:
start_timesteps = 2500  # More diverse experiences before learning
```

---

## Verification Checklist

After applying fixes, verify:

- [ ] `training_config.yaml`: `distance_scale: 5.0` ✅
- [ ] `training_config.yaml`: `waypoint_bonus: 1.0` ✅
- [ ] `carla_config.yaml`: `weights.progress: 1.0` ✅
- [ ] `carla_config.yaml`: `weights.lane_keeping: 1.0` ✅
- [ ] `td3_agent.py`: `policy_delay = 2` ✅
- [ ] `td3_agent.py`: `tau = 0.005` ✅
- [ ] `td3_agent.py`: `actor_lr = 3e-4` ✅
- [ ] `td3_agent.py`: `critic_lr = 3e-4` ✅
- [ ] Git commit changes: `git commit -m "fix: normalize reward scale to prevent Q-overestimation"` ✅

---

## Expected Training Curve (After Fixes)

**Q-Values**:
```
Steps 1000-2000: 10-20 (rising slowly)
Steps 2000-3000: 15-25 (converging)
Steps 3000-5000: 20-30 (stable)
```

**Critic Loss**:
```
Steps 1000-2000: 100-200 (high initially)
Steps 2000-3000: 50-100 (decreasing)
Steps 3000-5000: 20-50 (converging)
```

**Actor Loss**:
```
Steps 1000-2000: -50 to -100 (stabilizing)
Steps 2000-5000: -80 to -120 (stable, slight increase ok)
```

**TD Errors**:
```
Steps 1000-2000: 5-10 (reducing)
Steps 2000-5000: 2-5 (converged)
```

---

## Red Flags (Stop Training If You See)

- 🚨 Q-values > 50 after 2K steps
- 🚨 Critic loss > 500 after 2K steps
- 🚨 TD errors > 10 after 3K steps
- 🚨 Actor loss < -500 (too negative)
- 🚨 Q - R gap > 10

If you see any of these:
1. Stop training
2. Apply Additional Fix #2 (gradient clipping)
3. Apply Additional Fix #3 (increase warmup)
4. Check learning rates (should be 3e-4, not higher!)

---

## Documentation References

- **Full Analysis**: `TENSORBOARD_DIAGNOSIS.md` (90+ page detailed breakdown)
- **Previous Bugs**: `DIAGNOSIS_RIGHT_TURN_BIAS.md`, `INVESTIGATION_HARD_LEFT_BIAS.md`
- **TD3 Paper**: https://arxiv.org/abs/1802.09477
- **OpenAI Spinning Up**: https://spinningup.openai.com/en/latest/algorithms/td3.html
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/en/master/modules/td3.html

---

## Summary

**What We Found**:
- Q-value overestimation: +47.76 gap (should be < 5) 🔴
- Critic loss exploding: 1244, trend +361% 🔴
- TD errors too large: 13.4 (should be < 3) 🔴
- Root cause: Progress reward too high (91.7% of total)

**What We're Fixing**:
- Reduce progress reward scale by 10× (50 → 5)
- Reduce progress weight by 2× (2.0 → 1.0)
- Verify all TD3 parameters match specification

**What to Expect**:
- Q-values: 67 → 15-30 (2-4× reduction)
- Critic loss: 1244 → 20-50 (25× reduction)
- TD errors: 13.4 → < 5 (2-3× reduction)
- Agent should START LEARNING after fixes!

**Next Steps**:
1. Apply fixes (5 minutes)
2. Re-run training (10 minutes)
3. Monitor TensorBoard (watch for success criteria)
4. Report back with updated metrics!

---

**Good luck! The fixes are simple, and you should see immediate improvement.** 🚀
