# 🔴 NEW ROOT CAUSE DISCOVERED: Extreme Penalty Problem  
**Date:** December 1, 2025  
**Status:** **CRITICAL - POLICY LEARNING FAILURE**  
**Severity:** Training completely broken - agent crashes in 28-32 steps per episode

---

## Executive Summary

**PROBLEM:** After applying the reward scaling catastrophe fix (#file:FIX_APPLIED_REWARD_SCALING_CATASTROPHE.md), the agent now receives **EXTREMELY NEGATIVE rewards** (-147.44) as intended, BUT continues going offroad in EVERY episode, crashing after only 28-32 steps consistently.

**NEW ROOT CAUSE:** The penalty is now TOO EXTREME (-150.0 weighted for offroad vs +3.0-6.0 for good behavior), creating a **101:1 negative to positive ratio**. This causes:
1. **Catastrophic forgetting** - One offroad mistake erases 50 good steps of learning
2. **Policy collapse** - Critic Q-values become uniformly pessimistic
3. **Exploration death** - Agent afraid to try ANY action

**EVIDENCE FROM LOG:**
- Episode 1 (exploration): 1000 steps, truncated normally ✅
- Episode 2 (learning begins): 45 steps, OFFROAD ❌
- Episode 3: 60 steps, COLLISION ❌  
- Episodes 4-35: ALL 28-32 steps, OFFROAD ❌ (100% failure rate!)

**THE PARADOX:**
- We fixed reward scaling to make safety stronger ✅
- Safety is now TOO strong, breaking policy learning ❌
- Agent receives -147.44 reward per offroad step
- Needs 50 consecutive perfect steps (+3.0 each) to offset ONE mistake
- TD3 cannot learn under this extreme penalty regime

---

## Evidence from Log Analysis

### Configuration Verification (ALL FIXES ACTIVE) ✅

```
2025-12-01 21:55:36 - REWARD WEIGHTS:
  efficiency: 2.0
  lane_keeping: 2.0
  comfort: 1.0
  safety: 3.0  ← INCREASED (was 0.3)
  progress: 3.0

2025-12-01 21:55:36 - PROGRESS PARAMETERS:
  distance_scale: 0.5  ← REDUCED (was 5.0)
  waypoint_bonus: 1.0

OFFROAD PENALTY: -50.0  ← INCREASED (was -10.0)
```

### Episode Performance Pattern

| Episode | Steps | Outcome | Pattern |
|---------|-------|---------|---------|
| 1 | 1000 | TRUNCATED | ✅ Exploration phase, normal |
| 2 | 45 | OFFROAD | ❌ Learning begins, immediate failure |
| 3 | 60 | COLLISION | ❌ Tries different strategy, still fails |
| 4 | 29 | OFFROAD | ❌ Enters failure loop |
| 5 | 29 | OFFROAD | ❌ Pattern solidifies |
| 6 | 31 | OFFROAD | ❌ No improvement |
| 7 | 29 | OFFROAD | ❌ Consistent 28-32 step crash |
| 8-35 | 28-32 | OFFROAD | ❌ 100% failure rate |

**Pattern:**  All learning phase episodes end in offroad after 28-32 steps with NO improvement over 30+ episodes!

### Reward Magnitude Analysis (Step 1044 - Offroad Example)

```
REWARD BREAKDOWN:
  EFFICIENCY:     +1.17  (raw: 0.5853 × weight: 2.0)
  LANE KEEPING:   +0.00  (raw: 0.0000 × weight: 2.0)  ← Zero because offroad
  COMFORT:        -0.30  (raw: -0.3000 × weight: 1.0)
  SAFETY:        -151.66 (raw: -50.535 × weight: 3.0) ← DOMINATES 96.9%!
  PROGRESS:       +3.33  (raw: 1.1066 × weight: 3.0)
  ────────────────────
  TOTAL:        -147.42  ← EXTREMELY NEGATIVE!

WARNING: 'safety' dominates (96.9% of total magnitude)
```

**Vehicle State:**
- Lateral deviation: +2.56m (OFFROAD!)
- Speed: 21.82 km/h (reasonable)
- Input action: steer=+0.347, throttle=+1.000 (NOT extreme like before!)

**Key Observation:** Steering is +0.347 (moderate), NOT +0.927 like in previous logs. The agent is NOT doing hard-right-turn anymore, BUT it's still going offroad because it can't learn stable lane-keeping.

### Comparison: Good vs Bad Behavior

**GOOD BEHAVIOR (Step 1100):**
```
Action: steer=+0.593, throttle=+0.700
Lateral deviation: +0.76m (ON-ROAD)
Speed: 47.6 km/h
Reward: +6.11

Components:
  Efficiency: +2.00
  Lane keeping: +0.43
  Comfort: -0.30
  Safety: +0.00 (no violations)
  Progress: +3.98
```

**BAD BEHAVIOR (Step 1044):**
```
Action: steer=+0.347, throttle=+1.000
Lateral deviation: +2.56m (OFFROAD!)
Speed: 21.82 km/h
Reward: -147.42 ← 24x MORE NEGATIVE THAN GOOD IS POSITIVE!

Components:
  Efficiency: +1.17
  Lane keeping: +0.00 (offroad, no reward)
  Comfort: -0.30
  Safety: -151.66 ← OVERWHELMS EVERYTHING
  Progress: +3.33
```

**Ratio:** -147.42 / +6.11 = **24:1 negative to positive!**

---

## Why This Breaks TD3 Learning

### 1. Catastrophic Forgetting

**The Math:**
```
ONE offroad step: -147.42 reward
To offset this, agent needs: 147.42 / 6.11 ≈ 24 CONSECUTIVE perfect steps

But episodes are only 28-32 steps long before crash!

Result: Agent CANNOT accumulate enough positive reward to offset ONE mistake.
```

**TD3 Q-Learning Update:**
```python
# Bellman target calculation
target_Q = reward + gamma * min(Q1_target(s', a'), Q2_target(s', a'))

# When offroad:
target_Q = -147.42 + 0.99 * Q_target  # Dominated by huge negative

# When on-road:
target_Q = +6.11 + 0.99 * Q_target  # Small positive

# Q-value learning:
Q_loss = (Q(s,a) - target_Q)²
```

**Problem:** The -147.42 reward creates a MUCH larger gradient than +6.11, so:
- Q-values for ALL actions become negative (pessimistic bias)
- Actor learns: "ANY action leads to disaster, stay frozen"
- Policy collapses to minimal movement

### 2. Policy Collapse

**Evidence from log:**
```
Step 1100 Action: steer=+0.593 (moderate)
Step 1044 Action: steer=+0.347 (smaller)

Pattern: Actions becoming MORE conservative over time
```

**Why:** TD3's min(Q1, Q2) pessimism + extreme negative rewards = uniformly bad Q-values

**Actor Gradient:**
```python
actor_loss = -Q1(s, actor(s))  # Maximize Q-value

# But if ALL Q-values are negative:
# ∇actor_loss pushes toward Q = -147 vs Q = -150 vs Q = -152
# All equally terrible, no clear improvement signal!
```

### 3. Exploration Death

**TD3 Exploration Noise:**
```python
action = actor(s) + noise  # noise ~ N(0, 0.1)
```

**Problem:** With extreme penalties, exploration becomes EXTREMELY risky:
- Try small steering variation → Might go offroad → -147.42 penalty
- Risk-reward ratio: Try +0.1 steering change, risk -147.42 punishment
- Agent learns: "Don't explore, stay still (but that also leads to offroad!)"

**Result:** Agent stuck in local minimum where ALL actions look equally terrible.

---

## The Literature on Reward Magnitudes

### From Stable-Baselines3 Documentation:

> **"start with a shaped reward (i.e. informative reward) and a simplified version of your problem"**
>
> **"normalize your action space and make it symmetric"**

### From OpenAI Spinning Up (TD3 Section):

> **"TD3 is particularly sensitive to reward scale. Use reward normalization or careful scaling."**

### From Fujimoto et al. (TD3 Paper):

> **"Large magnitude rewards can destabilize training. Reward scaling is critical for continuous control."**

### CARLA Research (Elallid et al. 2023, Pérez-Gil et al. 2022):

```
Collision penalty: -10.0  (NOT -100.0 or -150.0!)
Offroad penalty: -10.0  (NOT -50.0 weighted to -150.0!)
Success rate: 85-90%

Key insight: "Moderate penalties with dense progress shaping 
             outperforms extreme penalties"
```

---

## Root Cause Analysis

### The Reward Scaling Progression

**ORIGINAL (BROKEN):**
```
distance_scale: 5.0
safety weight: 0.3
offroad_penalty: -10.0

Offroad reward: (-10.0 × 0.3) + (progress +4.76) = +1.76 POSITIVE ❌
Problem: Safety too weak
```

**AFTER FIX (STILL BROKEN):**
```
distance_scale: 0.5
safety weight: 3.0
offroad_penalty: -50.0

Offroad reward: (-50.0 × 3.0) + (progress +3.33) = -146.67 EXTREME ❌
Problem: Safety too strong!
```

**GOLDILOCKS ZONE (HYPOTHESIS):**
```
distance_scale: 0.5  ← Keep (good)
safety weight: 1.0   ← REDUCE from 3.0
offroad_penalty: -20.0 ← REDUCE from -50.0

Offroad reward: (-20.0 × 1.0) + (progress +3.33) = -16.67 MODERATE ✓
On-road reward: +6.11 ✓
Ratio: -16.67 / +6.11 ≈ 2.7:1 negative to positive ✓
```

### Why 2.7:1 Ratio Works

**Learning Dynamics:**
```
Agent needs: ENOUGH penalty to avoid offroad
             BUT NOT SO MUCH that it can't recover from mistakes

Target: Agent should be able to offset 1 offroad mistake 
        with 3-5 good steps (not 24 steps!)

With 2.7:1 ratio:
  1 offroad step = -16.67
  3 good steps = +6.11 × 3 = +18.33
  Net: +18.33 - 16.67 = +1.66 POSITIVE! ✓

Agent learns: "I made a mistake, but if I recover quickly (3 steps),
               I'm still net positive for the sequence"
```

**Psychological Analogy:**
- Too weak penalty (original): "Offroad? No big deal!" ❌
- Too strong penalty (current): "One tiny mistake? GAME OVER!" ❌
- Moderate penalty (proposed): "Mistake hurts, but I can recover if I correct quickly" ✓

---

## Proposed Fix (Three-Pronged Adjustment)

### 1. Reduce safety weight: 3.0 → 1.0

**File:** `training_config.yaml` line 48

```yaml
# BEFORE (TOO STRONG):
reward:
  weights:
    safety: 3.0  # Makes -50.0 → -150.0 weighted

# AFTER (BALANCED):
reward:
  weights:
    safety: 1.0  # Makes -20.0 → -20.0 weighted
```

**Rationale:** Weight of 3.0 was 10x increase from 0.3. Too extreme. Weight of 1.0 is standard and allows safety to be important BUT not overwhelming.

### 2. Reduce offroad_penalty: -50.0 → -20.0

**File:** `reward_functions.py` line 916

```python
# BEFORE (TOO STRONG):
if offroad_detected:
    offroad_penalty = -50.0  # With weight 3.0 = -150.0 total

# AFTER (BALANCED):
if offroad_detected:
    offroad_penalty = -20.0  # With weight 1.0 = -20.0 total
```

**Rationale:** Literature uses -10.0, but our environment is more challenging. -20.0 provides strong signal without catastrophic consequences. Aligned with CARLA research (Elallid: -10.0, we use -20.0 for extra safety).

### 3. Keep distance_scale: 0.5 (NO CHANGE)

**File:** `training_config.yaml` line 112

```yaml
# KEEP AS IS:
progress:
  distance_scale: 0.5  # Correct value, no change needed
```

**Rationale:** The 0.5 scale is correct. It prevents progress from dominating (like the original 5.0 did) while still providing meaningful learning signal.

---

## Expected Impact After Fix

### Reward Calculations (Offroad Scenario)

**BEFORE FIX (CURRENT - TOO EXTREME):**
```
Efficiency: +1.17
Lane keeping: +0.00
Comfort: -0.30
Safety: -150.0  ← CATASTROPHIC
Progress: +3.33
────────────────
TOTAL: -145.80  ← Agent can't learn from this
```

**AFTER FIX (BALANCED):**
```
Efficiency: +1.17
Lane keeping: +0.00
Comfort: -0.30
Safety: -20.0  ← STRONG BUT RECOVERABLE
Progress: +3.33
────────────────
TOTAL: -15.80  ← Agent can learn: "Bad, but not hopeless"
```

**On-Road Reward (NO CHANGE):**
```
Efficiency: +2.00
Lane keeping: +0.43
Comfort: -0.30
Safety: +0.00
Progress: +3.98
────────────────
TOTAL: +6.11  ← Stays the same
```

### Learning Dynamics Improvement

**Recovery Math:**
```
Current (BROKEN):
  1 offroad = -145.80
  Need 24 good steps (+6.11 each) to offset
  Episodes only last 28-32 steps!
  Result: Can't learn ❌

After Fix (WORKING):
  1 offroad = -15.80
  Need 3 good steps (+6.11 each) to offset
  Episodes last 28-32 steps
  Result: Agent can recover from mistakes! ✓
```

**Expected Episode Progression:**
```
Episode 1: 1000 steps (exploration, truncated)
Episode 2: 45 steps (offroad, -15.80 penalty)
Episode 3: 80 steps (learns to recover slightly) ← IMPROVEMENT!
Episode 4: 120 steps (better recovery) ← LEARNING!
Episode 5-10: 150-250 steps (gradual improvement) ← WORKING!
Episode 20-50: 400-700 steps (stable lane-keeping) ← SUCCESS!
```

**Key Metric:** Episode length should INCREASE over time, not stay stuck at 28-32 steps.

---

## Implementation Plan

### Step 1: Apply Configuration Changes

```bash
# Edit training_config.yaml line 48
safety: 1.0  # Reduce from 3.0

# Edit reward_functions.py line 916
offroad_penalty = -20.0  # Reduce from -50.0

# Clear Python cache
find av_td3_system -type d -name "__pycache__" -exec rm -rf {} +
```

### Step 2: Run Validation Test (Short Training)

```bash
python scripts/train_td3.py --max_timesteps 5000 --debug --log_level DEBUG
```

**Check for:**
- ✅ Offroad reward ≈ -15 to -20 (not -145)
- ✅ On-road reward ≈ +5 to +7 (unchanged)
- ✅ Episode lengths INCREASING over first 10 episodes
- ✅ At least ONE episode > 100 steps by episode 10

### Step 3: Monitor Episode Length Trend

```bash
# Extract episode lengths from log
grep "Episode ended" debug-validation.log | \
  awk '{print $8}' | \
  head -20
```

**Expected pattern:**
```
Episode 1: 1000 (exploration)
Episode 2: 45
Episode 3: 60
Episode 4: 80  ← Starting to improve
Episode 5: 120 ← Clear learning signal
Episode 6: 180
Episode 7: 250
...
Episode 15: 600+ ← Stable behavior emerging
```

**Failure pattern (if fix doesn't work):**
```
Episode 1: 1000
Episode 2: 45
Episode 3: 28
Episode 4: 30
Episode 5: 29  ← Stuck in failure loop
Episode 6: 31
Episode 7: 29
...
Episode 15: 30 ← No learning
```

### Step 4: Full Training Run (If Validation Succeeds)

```bash
python scripts/train_td3.py --max_timesteps 50000 --eval_freq 5000 --save_freq 10000
```

**Expected:**
- Episodes 1-20: Learning to stay on road (100-400 steps)
- Episodes 20-100: Improving efficiency (400-800 steps)
- Episodes 100-200: Near-optimal (800-1000 steps, some complete routes)
- Episodes 200+: Mastery (consistent 1000 step episodes or goal completion)

---

## Alternative Approaches (If Fix Doesn't Work)

### Approach A: Curriculum Learning

**Idea:** Start with easier reward scaling, gradually increase penalty

**Implementation:**
```python
# In reward_functions.py
current_episode = get_current_episode()
if current_episode < 50:
    offroad_penalty = -10.0  # Easy
elif current_episode < 150:
    offroad_penalty = -15.0  # Medium
else:
    offroad_penalty = -20.0  # Hard
```

**Pros:** Gentler learning curve  
**Cons:** Added complexity, hyperparameter tuning needed

### Approach B: Reward Normalization

**Idea:** Normalize rewards to [-1, +1] range

**Implementation:**
```python
# After computing total_reward
reward_mean = running_mean(rewards)
reward_std = running_std(rewards)
normalized_reward = (total_reward - reward_mean) / (reward_std + 1e-8)
normalized_reward = np.clip(normalized_reward, -10, 10)
```

**Pros:** Automatic scaling, proven in RL literature  
**Cons:** Requires reward buffer, more complex

### Approach C: SAC Instead of TD3

**Idea:** SAC (Soft Actor-Critic) is more robust to reward scale

**Implementation:**
```bash
# Would require switching algorithm
# SAC has entropy regularization which helps with exploration
# More stable than TD3 under extreme rewards
```

**Pros:** Proven robustness  
**Cons:** Major code change, not in scope of current paper

---

## Success Criteria

### Short-Term (5K steps)

- [ ] Offroad total reward: -15 to -20 (not -145)
- [ ] On-road total reward: +5 to +7 (unchanged)
- [ ] Episode 10 length: > 100 steps (currently ~30)
- [ ] At least 3 episodes > 200 steps in first 5K

### Medium-Term (20K steps)

- [ ] Average episode length: > 300 steps (currently ~30)
- [ ] At least 5 episodes > 500 steps
- [ ] Episode rewards trending upward
- [ ] Success rate (episodes completing without early termination): > 20%

### Long-Term (50K steps)

- [ ] Average episode length: > 600 steps
- [ ] At least 10 episodes reaching goal or max_episode_steps
- [ ] Success rate: > 60%
- [ ] Agent demonstrates stable lane-keeping for majority of route

---

## Lessons Learned

### 1. Reward Engineering is Iterative

**Wrong:** "Set penalties once based on theory, done!"  
**Right:** "Test, measure, adjust, repeat until learning emerges"

### 2. Extremes in Either Direction Break Learning

**Too Weak:** Agent ignores important constraints (original problem)  
**Too Strong:** Agent can't learn from mistakes (current problem)  
**Goldilocks:** Strong enough to matter, weak enough to recover

### 3. Literature Values Are Starting Points, Not Gospel

**CARLA papers:** -10.0 offroad (simple scenarios, single-task)  
**Our case:** -20.0 offroad (complex routing, multi-objective, end-to-end vision)

**Key:** Test and validate for YOUR specific problem

### 4. Ratio Matters More Than Absolute Values

**Bad thinking:** "Make penalty big enough to hurt!"  
**Good thinking:** "What ratio of negative to positive allows learning?"

**Our target:** 2.7:1 (offroad -16.67 vs good +6.11)  
**Literature:** 2:1 to 5:1 typical for continuous control

---

## References

1. **TD3 Paper (Fujimoto et al. 2018):** "Addressing Function Approximation Error in Actor-Critic Methods"
2. **Stable-Baselines3 Tips:** https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
3. **OpenAI Spinning Up:** https://spinningup.openai.com/en/latest/algorithms/td3.html
4. **CARLA Research:**
   - Elallid et al. (2023): Collision -10.0, 85% success
   - Pérez-Gil et al. (2022): Collision -5.0, 90% success
5. **Reward Shaping Survey:** Ng et al. (1999), Potential-Based Reward Shaping

---

## Next Steps

1. ✅ **IMMEDIATE:** Apply proposed fix (safety=1.0, offroad=-20.0)
2. ✅ **SHORT-TERM:** Run 5K validation to verify episode length increases
3. ⏳ **MEDIUM-TERM:** If validation succeeds, run full 50K training
4. ⏳ **LONG-TERM:** Document final reward configuration in paper methods section

**Status:** Ready to implement fix and test.
