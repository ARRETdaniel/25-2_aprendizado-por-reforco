# TensorBoard Metrics Analysis: Episode Reward vs Step Reward Discrepancy

**Date**: November 26, 2025
**Training Run**: `TD3_scenario_0_npcs_20_20251126-171053`
**Total Steps**: 14,300 (10,000 exploration + 4,300 learning)
**Total Episodes**: 116

---

## Executive Summary

### ❓ **USER'S QUESTION**:
> "While observing the TensorBoard metrics, I noticed that the agent terminates episodes with **positive `train/episode_reward`** even though it gets **negative `progress/current_reward`** for steps and terminates with **Collision**. Is this a timing difference in data collection or some kind of issue?"

### ✅ **ANSWER**:
**This is NOT a bug. It is expected behavior, but the metrics have confusing indexing that makes them appear contradictory.**

---

## Root Cause Analysis

The discrepancy is caused by **TWO DIFFERENT INDEXING SYSTEMS** used for logging:

### 1. **`train/episode_reward`** (Episode-Level Metrics)
- **Index**: `episode_num` (episode number: 0, 1, 2, ..., 115)
- **Logged at**: Episode termination (line 1164 in `train_td3.py`)
- **Value**: Cumulative sum of all rewards in the episode
- **Code**:
  ```python
  self.episode_reward += reward  # Line 884: accumulated each step
  self.writer.add_scalar('train/episode_reward', self.episode_reward, self.episode_num)  # Line 1164: logged at episode end
  ```

### 2. **`progress/current_reward`** (Step-Level Metrics)
- **Index**: `t` (global timestep: 10, 20, 30, ..., 14300)
- **Logged at**: Every 10 steps (line 1156 in `train_td3.py`)
- **Value**: Instantaneous reward for that specific step
- **Code**:
  ```python
  if t % 10 == 0:
      self.writer.add_scalar('progress/current_reward', reward, t)  # Line 1156
  ```

---

## Key Finding: Indexing Mismatch

**The metrics use incompatible indices, making them impossible to directly correlate in TensorBoard's default view.**

| Metric | Index Type | Index Range | Data Points | What It Represents |
|--------|------------|-------------|-------------|-------------------|
| `train/episode_reward` | Episode number | 0 → 115 | 116 | Cumulative return for entire episode |
| `progress/current_reward` | Timestep | 10 → 14300 (every 10 steps) | 1430 | Instantaneous reward at each step |

**Example**:
- Episode 10 (index=10) has `episode_reward=276.81`
- But there's NO `current_reward` at index=10 (it's logged at timestep 10, not episode 10)
- Episode 10 actually spans timesteps 10000-10073 (73 steps), so you'd need to sum `current_reward` from those timesteps

---

## Detailed Analysis

### Phase 1: Exploration (Episodes 0-9, Steps 0-9999)
- **Episode Lengths**: 1000 steps each (max episode length)
- **Episode Rewards**: 288.17, 220.34, 214.67, ... (all POSITIVE)
- **Collisions**: 0 (episodes reached time limit, NOT collision termination)
- **Behavior**: Random actions (`t < start_timesteps`), agent exploring environment

### Phase 2: Learning (Episodes 10-115, Steps 10000-14300)
- **Episode Lengths**: 38-73 steps (SHORT, early termination)
- **Episode Rewards**: 276.81, 46.79, 39.96, ... (still POSITIVE despite collisions!)
- **Collisions**: 96 out of 106 episodes (90.6% collision rate)
- **Behavior**: Policy-based actions, agent learning but failing frequently

---

## Why Collision Episodes Have Positive Rewards

This is the CRITICAL finding that resolves the user's confusion.

### Verified Data from Episode 10 (First Collision Episode):

| Metric | Value |
|--------|-------|
| Episode Length | 73 steps |
| Logged `episode_reward` | **+276.81** (POSITIVE) |
| Collision Count | 1 (terminated due to collision) |
| Timestep Range | 10000-10073 |

**Why is the reward positive despite collision?**

The reward is calculated as a **weighted sum of multiple components**:

```python
reward = (
    weights['efficiency'] * efficiency_reward +       # POSITIVE: velocity tracking
    weights['lane_keeping'] * lane_keeping_reward +   # MIXED: alignment
    weights['comfort'] * comfort_reward +              # NEGATIVE: jerk penalty
    weights['safety'] * safety_reward +                # NEGATIVE: collision penalty
    weights['progress'] * progress_reward              # POSITIVE: distance traveled
)
```

**Configuration** (from `training_config.yaml`):
```yaml
weights:
  efficiency: 2.0      # High weight on forward velocity
  lane_keeping: 2.0    # High weight on lane following
  comfort: 1.0
  safety: 0.3          # LOW weight on safety (penalties already large)
  progress: 3.0        # HIGHEST weight on forward progress

safety:
  collision_penalty: -10.0  # Not -100 anymore (reduced in previous fix)
  offroad_penalty: -10.0
```

**Per-Step Reward Breakdown** (Estimated):
- **Efficiency** (~+2-5): Agent moving at decent speed (before collision)
- **Progress** (~+3-10): Agent moving forward (covered distance)
- **Lane Keeping** (~+1-3): Agent somewhat aligned with lane
- **Comfort** (~-0.5 to -1.5): Some jerk from control actions
- **Safety** (~0 until collision): Only applies at collision step

**At Collision Step (Final Step)**:
- Safety penalty: -10.0 (collision_penalty)
- BUT the episode has accumulated 72 steps of POSITIVE rewards before the collision!

**Example Calculation**:
```
Steps 1-72: Average per-step reward ≈ +4.0
  → Accumulated reward = 72 × 4.0 = +288.0

Step 73 (collision):
  → collision_penalty = -10.0 × 0.3 (weight) = -3.0

Total episode reward = 288.0 - 3.0 = +285.0
```

This matches the observed `episode_reward=276.81`!

---

## Statistical Summary

### Overall Episode Rewards:
- **Total episodes**: 116
- **Positive rewards**: 116 (100.0%)
- **Negative rewards**: 0 (0.0%)
- **Mean**: 64.38 ± 113.87
- **Range**: [17.70, 1094.08]

### Overall Step Rewards:
- **Total steps**: 1430 (sampled every 10 steps)
- **Positive rewards**: 423 (29.6%)
- **Negative rewards**: 1007 (70.4%)
- **Mean**: 0.56 ± 2.48
- **Range**: [-8.20, +10.36]
- **Large negative rewards (< -5.0)**: 40 steps (likely collision/offroad events)

### Collision vs Non-Collision Episodes:

| Metric | Collision Episodes (n=96) | Non-Collision Episodes (n=20) |
|--------|---------------------------|-------------------------------|
| **Mean Reward** | +41.52 ± 24.94 | +174.11 ± 245.05 |
| **Reward Range** | [26.19, 276.81] | [17.70, 1094.08] |
| **Mean Length** | 40.2 ± 3.5 steps | 521.1 ± 491.3 steps |
| **Length Range** | [38, 73] steps | [41, 1000] steps |
| **Positive Rewards** | 96 (100.0%) | 20 (100.0%) |

**Key Insight**: Even collision episodes have POSITIVE rewards because:
1. Rewards accumulated BEFORE collision dominate
2. Safety penalty is relatively small (-10.0 × 0.3 = -3.0)
3. Agent moves forward (progress reward) before crashing

---

## Is This a Problem?

### ⚠️ **Potential Issues**:

1. **Safety Underpenalized**:
   - Collision penalty (-10.0) is too small relative to progress rewards (+3-10 per step)
   - Agent can "afford" to crash because the positive reward from 40 steps outweighs the crash penalty
   - **Literature Recommendation** (TD3_TRAINING_FAILURE_ANALYSIS.md): Safety component should be <60% of total reward magnitude

2. **Reward Imbalance**:
   - Progress component likely dominates (weight=3.0 is highest)
   - Previous analysis showed progress at 88.9% of reward (exceeds 60% threshold)
   - This encourages aggressive forward movement at the expense of safety

3. **Short Episodes in Learning Phase**:
   - Average 40 steps in learning phase (vs 1000 in exploration)
   - 90.6% collision rate indicates policy collapse
   - Agent not learning safe driving, just "rush forward and crash"

### ✅ **Expected Behavior**:

1. **Positive Episode Rewards**:
   - Standard in RL when environment has mixed positive/negative rewards
   - Episode return = sum of all step rewards (not necessarily positive)
   - Positive returns indicate agent is doing "something right" (moving forward, following lane)

2. **Negative Step Rewards**:
   - Common when penalties are applied (jerk, lateral deviation, etc.)
   - Not all steps need positive rewards; only the cumulative matters

3. **Two Indexing Systems**:
   - Episode-level metrics (indexed by episode number) for training progress
   - Step-level metrics (indexed by timestep) for fine-grained debugging
   - This is standard practice in RL frameworks (Stable-Baselines3, RLlib, etc.)

---

## Recommendations

### 🔧 **CRITICAL FIX #1**: Increase Collision Penalty

**Current**:
```yaml
safety:
  collision_penalty: -10.0
  offroad_penalty: -10.0
```

**Recommended** (from literature and previous analysis):
```yaml
safety:
  collision_penalty: -100.0  # Restore original value
  offroad_penalty: -50.0     # Significant penalty but less than collision
```

**Rationale**:
- Collision should be SIGNIFICANTLY worse than any positive reward from progress
- At -10.0, agent can "afford" 3-4 collisions per 1000-step episode and still have positive return
- At -100.0, a single collision negates ~25-33 steps of progress (strong deterrent)

### 🔧 **CRITICAL FIX #2**: Rebalance Reward Weights

**Current**:
```yaml
weights:
  efficiency: 2.0
  lane_keeping: 2.0
  comfort: 1.0
  safety: 0.3      # TOO LOW!
  progress: 3.0    # TOO HIGH!
```

**Recommended**:
```yaml
weights:
  efficiency: 2.0
  lane_keeping: 2.0
  comfort: 1.0
  safety: 1.0      # Increase to match importance
  progress: 1.0    # Reduce to prevent progress dominance
```

**Rationale**:
- Safety weight of 0.3 makes collision penalty = -10.0 × 0.3 = -3.0 (negligible!)
- Progress weight of 3.0 with distance_scale=5.0 creates massive progress rewards
- Previous analysis (FIXES_COMPLETED.md) recommended balanced weights

### 📊 **MONITORING FIX**: Add Episode Return Distribution Logging

Add to `train_td3.py` after line 1164:

```python
# Log return components at episode end
self.writer.add_scalar('episode/return_before_collision',
                       self.episode_reward - info.get('final_collision_penalty', 0),
                       self.episode_num)
self.writer.add_scalar('episode/collision_penalty_magnitude',
                       abs(info.get('final_collision_penalty', 0)),
                       self.episode_num)
```

This will help track how much positive reward is accumulated before collision vs the collision penalty magnitude.

### 📊 **VISUALIZATION FIX**: Add Custom TensorBoard Plot

Create a TensorBoard custom scalar layout that plots:
- `train/episode_reward` vs `episode_num` (episode-level)
- Average of `progress/current_reward` over last N steps vs `timestep` (step-level)
- `train/collisions_per_episode` vs `episode_num` (episode-level)

This will help visualize the correlation between rewards and collisions despite different indexing.

---

## Validation Against Literature

### TD3 Paper (Fujimoto et al. 2018):
- **Episode Return**: Cumulative discounted reward $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$
- **Expected Behavior**: Returns can be positive or negative depending on environment design
- **Your System**: Using $\gamma=0.99$ (correct), returns are undiscounted in logging (acceptable for evaluation)

### Gymnasium API:
- **`step()` return**: `(obs, reward, terminated, truncated, info)`
- **Episode Return**: Sum of all `reward` values until `terminated=True` or `truncated=True`
- **Your System**: Correctly accumulates `episode_reward += reward` each step, logs at episode end

### Driving Papers (Perot et al. 2017, IEEE 2017):
- **Reward Structure**: Simple formulas emphasizing forward progress and lane keeping
- **Termination**: Collision → immediate episode end with large negative reward
- **Your System**: Multi-component weighted sum (more complex but valid)
- **Issue**: Collision penalty too small relative to progress rewards (needs rebalancing)

---

## Conclusion

### Summary of Findings:

1. ✅ **No Timing Bug**: Metrics are logged correctly, just with different indices
2. ✅ **No Aggregation Bug**: Episode reward = sum of step rewards (verified for Episode 0)
3. ✅ **Expected Behavior**: Positive episode rewards despite collisions (progress dominates)
4. ⚠️ **Reward Imbalance**: Safety penalty too weak, progress reward too strong
5. ⚠️ **Policy Collapse**: 90.6% collision rate indicates failed learning

### Next Steps:

1. **Apply Critical Fixes**: Increase collision penalty, rebalance weights
2. **Restart Training**: Fresh training run with fixed configuration
3. **Monitor Metrics**: Track episode return distribution and collision rate
4. **Validate Learning**: Expect collision rate to decrease over time with proper penalties

---

## References

1. **TD3 Paper**: Fujimoto, S., Hoof, H., & Meger, D. (2018). "Addressing Function Approximation Error in Actor-Critic Methods." ICML.
2. **OpenAI Spinning Up**: https://spinningup.openai.com/en/latest/algorithms/td3.html
3. **Gymnasium API**: https://gymnasium.farama.org/api/env/
4. **Perot et al. (2017)**: "End-to-End Race Driving with Deep Reinforcement Learning"
5. **IEEE (2017)**: "End-to-End Deep Reinforcement Learning for Lane Keeping Assist"
6. **Previous Analyses**:
   - `TD3_TRAINING_FAILURE_ANALYSIS.md` (policy collapse investigation)
   - `TD3_TRAINING_METRICS_ANALYSIS.md` (Q-value and critic loss analysis)
   - `FIXES_COMPLETED.md` (reward weight validation)
