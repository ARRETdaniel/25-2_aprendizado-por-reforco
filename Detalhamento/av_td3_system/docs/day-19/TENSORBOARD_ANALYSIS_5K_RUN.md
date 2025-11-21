# TensorBoard Metrics Analysis Report

**Event File**: `events.out.tfevents.1763470787.danielterra.1.0`
**Analysis Date**: 1763573939.8605509
**Total Metrics**: 61

---

## 1. Training Phase Analysis

- **Maximum Step**: 5,000
- **Current Phase**: LEARNING
- **Description**: TD3 learning (steps 1000+)
- **Expected Updates**: 80 (train_freq=50)

### Expected Behavior at 5k Steps

Based on TD3 paper and OpenAI Spinning Up:

1. **Steps 0-1,000**: Random exploration
   - No policy training
   - Replay buffer filling
   - No Q-values/losses logged

2. **Steps 1,001-5,000**: Early learning
   - ~80 training updates (every 50 steps)
   - Q-values: Should be LOW and NOISY (lots of variance)
   - Critic loss: Should be HIGH (learning from scratch)
   - Actor loss: Should be NEGATIVE (policy gradient)
   - Gradients: Should be MODERATE (not exploding)

---

## 2. Q-Value Analysis (Overestimation Detection)

### Twin Critic Q-Values

**Q1 (from replay buffer samples)**:
- Count: 40
- Mean: 43.07
- Std Dev: 17.04
- Range: [17.59, 70.89]
- Median: 39.75

**Q2 (from replay buffer samples)**:
- Count: 40
- Mean: 43.07
- Std Dev: 17.04
- Range: [17.58, 70.89]
- Median: 39.73

### Actor Q-Values (Overestimation Check)

**Q(s, μ(s)) - Q-value of current policy actions**:
- Count: 40
- Mean: 4.61e+05
- Std Dev: 6.64e+05
- Range: [2.19e+00, 2.33e+06]
- Median: 9.34e+04

⚠️ **WARNING: High Q-Values**

Actor Q-values are 461423.19, which is higher than expected.

**Expected at 5k steps**: Q-values should be < 500
**Observed**: Q-values > 1,000

---

## 3. Loss Analysis

### Critic Loss (MSE)

- Count: 40
- Mean: 58.73
- Std Dev: 89.05
- Range: [11.71, 508.61]

**Expected at 5k steps**: High and volatile (network learning from scratch).

### Actor Loss (Policy Gradient)

- Count: 40
- Mean: -4.61e+05
- Std Dev: 6.64e+05
- Range: [-2.33e+06, -2.19e+00]

**Expected**: Negative values (maximizing Q). Should be stable, not exploding.

---

## 4. Gradient Norm Analysis

---

## 5. Reward Analysis

### Step Reward (Mean per Update)

- Mean: 11.91
- Std Dev: 2.30

---

## 6. Readiness Assessment for 1M Step Training

### ⚠️ PROCEED WITH CAUTION

**Warnings**:

- Q-values higher than expected (>1K)

**Recommendation**: Fix warnings before scaling to 1M steps.

---

## 7. Available Metrics

```
agent/actor_cnn_lr
agent/actor_cnn_param_mean
agent/actor_cnn_param_std
agent/actor_lr
agent/actor_param_mean
agent/actor_param_std
agent/buffer_utilization
agent/critic_cnn_lr
agent/critic_cnn_param_mean
agent/critic_cnn_param_std
agent/critic_lr
agent/critic_param_mean
agent/critic_param_std
agent/is_training
agent/total_iterations
alerts/gradient_explosion_critical
alerts/gradient_explosion_warning
debug/actor_q_max
debug/actor_q_mean
debug/actor_q_min
debug/actor_q_std
debug/done_ratio
debug/effective_discount
debug/q1_max
debug/q1_min
debug/q1_std
debug/q2_max
debug/q2_min
debug/q2_std
debug/reward_max
debug/reward_mean
debug/reward_min
debug/reward_std
debug/target_q_max
debug/target_q_mean
debug/target_q_min
debug/target_q_std
debug/td_error_q1
debug/td_error_q2
eval/avg_collisions
eval/avg_episode_length
eval/avg_lane_invasions
eval/mean_reward
eval/success_rate
gradients/actor_cnn_norm
gradients/actor_mlp_norm
gradients/critic_cnn_norm
gradients/critic_mlp_norm
progress/buffer_size
progress/current_reward
progress/episode_steps
progress/speed_kmh
train/actor_loss
train/collisions_per_episode
train/critic_loss
train/episode_length
train/episode_reward
train/exploration_noise
train/lane_invasions_per_episode
train/q1_value
train/q2_value
```
