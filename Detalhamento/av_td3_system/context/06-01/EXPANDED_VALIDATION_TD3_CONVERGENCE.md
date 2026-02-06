# Expanded Critical Validation: TD3 Convergence Failure Investigation

**Date:** January 6, 2025 (Session 2)
**Investigator:** Daniel Terra Gomes
**Status:** ✅ ALL 9 FINDINGS VALIDATED — Ready for Phase 1 Implementation
**Scope:** Critical validation pass on `SYSTEMATIC_INVESTIGATION_TD3_CONVERGENCE.md`
**Method:** Cross-reference every finding against actual codebase, official SB3 source code, and documentation

---

## 1. Executive Summary

All 9 findings from the original investigation have been validated against ground truth:
- **3 CRITICAL** findings confirmed with additional evidence
- **3 HIGH** findings confirmed with important system-coupling nuances discovered
- **2 MEDIUM** findings confirmed as-is
- **1 LOW** finding confirmed as-is
- **1 CORRECTION**: Parameter count in Finding #1 was **underestimated by 4.1×** (470K → 1.94M)
- **1 NEW DISCOVERY**: Findings #4 and #5 are **deeply coupled** — cannot be fixed independently
- **3 DEAD CODE AREAS** identified and marked with `TODO(DEAD_CODE)` in the codebase

---

## 2. Finding-by-Finding Validation

### 2.1 🔴 CRITICAL #1: Actor Gradient Clipping at max_norm=1.0

**Verdict: ✅ VALIDATED + CORRECTED (more severe than reported)**

#### Evidence Gathered

1. **SB3 TD3 source** (`e2e/stable-baselines3/stable_baselines3/td3/td3.py`): Zero matches for `clip_grad` or `max_grad_norm`. Confirmed NO gradient clipping in SB3's TD3.
2. **SB3 DDPG source**: Same — zero gradient clipping.
3. **SB3 SAC source**: Same — zero gradient clipping.
4. **Only SB3 algorithms using gradient clipping**: DQN (`max_grad_norm=10`), A2C (`max_grad_norm=0.5`), PPO (`max_grad_norm=0.5`).
5. **Our code** (`td3_agent.py` lines 1030-1110): Actor gradient clipped at `max_norm=1.0` across combined actor MLP + actor CNN parameters.

#### CORRECTION: Parameter Count

The investigation document stated "~470,000 parameters" for the actor+CNN. **This is incorrect.** Actual count from reading network definitions:

| Component | Calculation | Parameters |
|-----------|-------------|------------|
| **Actor MLP fc1** | 565 × 256 + 256 | 144,896 |
| **Actor MLP fc2** | 256 × 256 + 256 | 65,792 |
| **Actor MLP fc3** | 256 × 2 + 2 | 514 |
| **CNN conv1** | 4 × 32 × 8 × 8 + 32 | 8,224 |
| **CNN LayerNorm1** | 32 × 20 × 20 × 2 | 25,600 |
| **CNN conv2** | 32 × 64 × 4 × 4 + 64 | 32,832 |
| **CNN LayerNorm2** | 64 × 9 × 9 × 2 | 10,368 |
| **CNN conv3** | 64 × 64 × 3 × 3 + 64 | 36,928 |
| **CNN LayerNorm3** | 64 × 7 × 7 × 2 | 6,272 |
| **CNN FC** | 3136 × 512 + 512 | 1,606,144 |
| **CNN LayerNorm4** | 512 × 2 | 1,024 |
| **TOTAL** | | **1,938,594** |

**Key insight**: ~89% of actor parameters are in the CNN, dominated by the FC layer (1.6M params). LayerNorm adds ~43,264 trainable parameters that would not exist in standard NatureCNN.

#### Recalculated Impact

With 1.94M parameters (not 470K):

$$\bar{g} \approx \frac{1.0}{\sqrt{1{,}938{,}594}} \approx 0.000718$$

$$\Delta w \approx 0.0003 \times 0.000718 = 2.15 \times 10^{-7}$$

This is **~2× worse** than the original estimate of $4.38 \times 10^{-7}$. Each weight moves by ~0.0000002 per step — effectively frozen.

#### Critical Thinking: Should We Keep Any Actor Gradient Clipping?

The cited papers suggest clipping:
- Sallab et al. (2017) "Lane Keeping Assist": `clip_norm=1.0` — but for DDPG with ~50K params
- Perot et al. (2017) "End-to-End Race Driving": `clip_norm=40.0` — for A3C
- Chen et al. (2019) "Lateral Control": `clip_norm=10.0` — for DDPG

**Challenge**: These papers used much smaller networks and different algorithms. Sallab's DDPG had ~50K params, not 1.94M. Scaling max_norm proportionally: $1.0 \times \sqrt{1{,}938{,}594/50{,}000} \approx 6.2$. So if we follow Sallab's intent, we'd need `max_norm ≈ 6.0` at minimum.

**Recommendation**: Remove entirely (match official TD3) or set to `max_norm=40.0` (match Perot's more conservative bound).

---

### 2.2 🔴 CRITICAL #2: Reward Scale Asymmetry × TD3 Pessimism

**Verdict: ✅ VALIDATED + Additional Dead Code Found**

#### Evidence Gathered

**Config weights** (`training_config.yaml` lines 41-45):
```yaml
efficiency: 2.0, lane_keeping: 2.0, comfort: 1.0, safety: 1.0, progress: 3.0
```

These match the investigation document. ✅

#### Actual Per-Step Reward Ranges (from code analysis)

| Component | Weight | Max Positive | Worst Negative | Weighted Max | Weighted Min |
|-----------|--------|-------------|----------------|--------------|--------------|
| Efficiency | 2.0 | +1.0 | -1.0 | +2.0 | -2.0 |
| Lane Keeping | 2.0 | +0.5 | -1.0 (invasion) | +1.0 | -2.0 |
| Comfort | 1.0 | +0.3 | -1.0 (clip) | +0.3 | -1.0 |
| Safety | 1.0 | 0 | -142 (all penalties) | 0 | -142 |
| Progress | 3.0 | ~0.2+1.0 | -10 (clip) | ~3.6 | -30 |
| **Sum** | — | — | — | **~+6.9** | **~-177** |

The investigation cited +7.3 / -103. The positive max is close (+6.9 vs +7.3). The realistic worst case for a **single high-speed collision without other penalties** is -100 (weighted), giving the 14:1 ratio. With PBRS proximity (-2.0) and TTC (-5.0) active, worst case reaches -177. **The asymmetry is actually WORSE than reported.**

#### NEW: Dead Code in Safety Config

**`self.collision_penalty`** (loaded as -100 from config) and **`self.offroad_penalty`** (loaded as -500 from config) are **NEVER USED** in `_calculate_safety_reward()`. The method uses hardcoded speed-graduated values:
- Collision: -5 (low speed), -25 (medium), -100 (high speed) — hardcoded
- Offroad: -20.0 — hardcoded

This means changing `training_config.yaml`'s `collision_penalty` or `off_road_penalty` has **ZERO EFFECT** on training. Marked with `TODO(DEAD_CODE)` in codebase.

#### Critical Thinking: Is Reward Clipping the Right Solution?

The document recommends clipping total_reward to [-10, ∞). Alternative analysis:

**Pro (Clip at -10):**
- Preserves gradient direction (collision is still worst)
- Limits Q-value range, reducing critic approximation error
- Simple, one-line change
- SB3's VecNormalize achieves similar effect via running statistics

**Con (Clip at -10):**
- Loses magnitude distinction between low-speed bump (-5) and high-speed crash (-100)
- All severe events map to same -10 floor
- May still be too asymmetric: +6.9 vs -10 = 1.45:1 ratio

**Better alternative**: Reward normalization in the replay buffer (running mean/std), which preserves relative ordering while bounding the effective range. This is standard in PPO (VecNormalize) but NOT standard in TD3/DDPG. However, given our non-standard environment with extreme rewards, it may be justified.

**Recommended approach for Phase 2**: Clip at -10 as a simple first step, then evaluate if normalization is needed.

---

### 2.3 🔴 CRITICAL #3: learning_starts=1000 with Visual Input

**Verdict: ✅ VALIDATED**

#### Evidence Gathered

- **Config** (`td3_config.yaml` line 67): `learning_starts: 1000` with misleading comment "10K steps"
- **Code** (`train_td3.py` line 675): `self.agent_config.get('algorithm', {}).get('learning_starts', 25000)` — reads 1000 from config
- **Official TD3** (`TD3/main.py`): `start_timesteps=25e3`
- **SB3**: `learning_starts=100` (but for simple state-vector environments, not visual)
- **Spinning Up**: 10,000 default

#### Critical Thinking: Is 25K Too Many for Our Setup?

With `max_episode_steps=1000` (from `training_config.yaml`) at 20 FPS:
- 1000 steps = at most **1 episode** of random data. Extreme overfitting risk.
- 10,000 steps = ~10 episodes. Moderate diversity.
- 25,000 steps = ~25 episodes. Good diversity for Town01 route.
- 50,000 steps = ~50 episodes. Potentially excessive if episodes are long.

With batch_size=256 and buffer of 1000: sampling 25.6% of buffer per batch (extreme).
With buffer of 10,000: sampling 2.56% per batch (acceptable).
With buffer of 25,000: sampling 1.02% per batch (standard).

**Recommendation**: 25,000 (official TD3 default) is the right value. Even Spinning Up's 10,000 may be insufficient for visual RL where the CNN needs diverse frames.

---

### 2.4 🟡 HIGH #4: Image Normalization [-1, 1] vs Standard [0, 1]

**Verdict: ✅ VALIDATED — BUT System is Deeply Coupled to [-1, 1]**

#### Evidence Gathered

**SB3 preprocessing** (`preprocessing.py` line 120):
```python
return obs.float() / 255.0  # → [0, 1]
```

**Our preprocessing** (`sensors.py` line 186):
```python
normalized = (scaled - 0.5) / 0.5  # → [-1, 1]
```

**SB3 NatureCNN** (`torch_layers.py`): Uses `nn.ReLU()`, NO normalization layers, NO LeakyReLU.

#### System Coupling Discovery (NEW)

Changing image normalization from [-1, 1] to [0, 1] is **NOT a simple sensors.py change**. It requires coordinated modification across **5 files**:

| # | File | Current | Required Change |
|---|------|---------|-----------------|
| 1 | `sensors.py` line 186 | `(scaled - 0.5) / 0.5` | Remove: just use `scaled` |
| 2 | `carla_env.py` line 472 | `Box(low=-1.0, high=1.0, ...)` | `Box(low=0.0, high=1.0, ...)` |
| 3 | `cnn_extractor.py` | `LeakyReLU(0.01)` ×4 | `ReLU()` ×4 (standard for [0,1]) |
| 4 | `cnn_extractor.py` docstring | "normalized to [-1, 1]" | Update to "[0, 1]" |
| 5 | Any saved checkpoints | Trained on [-1, 1] | **Incompatible** — must retrain |

The system was **deliberately designed** for [-1, 1]:
- `carla_env.py` line 475 has comment: *"FIX BUG #8: Image space now matches preprocessing output range [-1,1]"*
- `cnn_extractor.py` selected LeakyReLU **specifically** for zero-centered input: *"preserves negative values from zero-centered normalization"*

#### Critical Thinking: Does [-1, 1] Actually Hurt?

**Argument that it does hurt:**
- With LeakyReLU(0.01), negative inputs (50% of [-1,1]) pass through at 1% amplitude
- Initial information throughput is effectively halved vs ReLU+[0,1]
- LayerNorm re-centers anyway, making the [-1,1] preprocessing redundant work

**Argument that it might be fine:**
- LayerNorm after each conv layer normalizes to mean≈0, std≈1 regardless of input
- The CNN should learn to handle [-1,1] — it's a valid transformation
- LeakyReLU prevents dead neurons that ReLU can cause
- Some DDPG/TD3 papers use [-1,1] input (less common but not wrong)

**Key insight**: The real problem isn't [-1,1] alone — it's [-1,1] + LeakyReLU + LayerNorm combined creating a non-standard architecture that hasn't been validated in visual RL literature. Each deviation individually might be fine, but together they're untested.

**Recommendation**: Fix as part of a **combined CNN overhaul** (Finding #4 + #5 together), not independently.

---

### 2.5 🟡 HIGH #5: LayerNorm in CNN is Non-Standard

**Verdict: ✅ VALIDATED — Coupled with Finding #4**

#### Evidence Gathered

| Implementation | Normalization | Activation | Published |
|---------------|---------------|------------|-----------|
| SB3 NatureCNN | **None** | ReLU | Yes (SB3 v2.x) |
| DQN (Mnih 2015) | **None** | ReLU | Yes (Nature) |
| DrQ (Yarats 2020) | **None** | ReLU | Yes (ICLR 2021) |
| SAC from Pixels | **None** | ReLU | Yes |
| **Ours** | **LayerNorm ×4** | LeakyReLU | **No** |

#### Parameter Cost of LayerNorm

LayerNorm adds **43,264 trainable parameters** to the CNN:
- LN1: [32, 20, 20] → 25,600 params (γ + β per element)
- LN2: [64, 9, 9] → 10,368 params
- LN3: [64, 7, 7] → 6,272 params
- LN4: [512] → 1,024 params

These are 2.5% of total CNN params but represent a fundamentally different computational graph.

#### Critical Thinking: When IS LayerNorm Appropriate?

LayerNorm is used in:
- **Transformers** (Vaswani et al., 2017): After attention and FFN layers
- **DDPG** (Lillicrap et al., 2015): After FC layers in critic (*not* in CNN)
- **TD3** paper: Mentions no normalization at all

DDPG critic LayerNorm makes sense because Q-values can have diverse scales. But in a **CNN feature extractor**, channels represent different spatial features (edges, textures, objects). Normalizing ACROSS channels forces different feature detectors to similar scales, which can suppress naturally stronger features.

**The day-3-12 analysis noted**: "CNN L2 norm growing despite LayerNorm" — this suggests LayerNorm isn't effectively constraining features, supporting its removal.

**Recommendation**: Remove LayerNorm from CNN and switch LeakyReLU→ReLU simultaneously (combined with Finding #4). This converts the CNN to standard NatureCNN architecture.

---

### 2.6 🟡 HIGH #6: Weight Decay on ALL Parameters Including Biases

**Verdict: ✅ VALIDATED**

#### Evidence Gathered

**Our code** (`td3_agent.py` lines 253-256, 291-294):
```python
self.actor_optimizer = torch.optim.Adam(actor_params, lr=self.actor_lr, weight_decay=1e-4)
self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.critic_lr, weight_decay=1e-4)
```

Both apply `weight_decay=1e-4` to **all** parameters (weights AND biases).

**Official TD3, SB3, Spinning Up**: `weight_decay=0` (no regularization).

#### Interaction with Finding #1 (CORRECTED)

With 1.94M actor+CNN params:
- Weight decay pull per param per step: `lr × λ × w ≈ 0.0003 × 1e-4 × w = 3×10⁻⁸ × w`
- Gradient update per param per step: `≈ 2.15×10⁻⁷` (from Finding #1)
- Ratio: decay/update ≈ `3×10⁻⁸ × w / 2.15×10⁻⁷ = 0.14×w`

For weights near initialization (w ≈ 0.01-0.1): decay is 0.14-1.4% of gradient signal.
For grown weights (w ≈ 1.0): decay is **14% of gradient signal** — significant!

**The investigation's "gradient prison" metaphor is accurate**: clipped gradients barely push weights up, decay constantly pulls them down.

**Recommendation**: Remove weight_decay from both optimizers entirely.

---

### 2.7 🟠 MEDIUM #7: Replay Buffer Size 100K vs Standard 1M

answer: We do not need to touch the replay buffer size of 100k, it was intentionally set to that value due to memory constraints of current laptop. In deployment in Uni super computer, the buffer size will be set to 1M as per standard practice.

**Verdict: ✅ VALIDATED — Memory Constraint is Real**

#### Memory Calculation NOTE: THE 100K buffer is intend for implementation and testing purposes only. The final training buffer size should and will be 1M as per standard practice, trining in Uni super computer.

Per transition:
- Images (curr + next): 2 × 4 × 84 × 84 × 4 bytes = 225,792 bytes
- Vectors (curr + next): 2 × ~53 × 4 = 424 bytes
- Actions + reward + done: 12 bytes
- **Total: ~226 KB/transition**

| Buffer Size | Memory | Feasibility (31 GB RAM) |
|-------------|--------|-------------------------|
| 100K | ~22 GB | ✅ Tight but feasible |
| 500K | ~110 GB | ❌ Requires optimization |
| 1M | ~221 GB | ❌ Requires major optimization |

#### Optimization Path

1. **uint8 storage** (4× savings): Store images as uint8, convert to float32 during sampling
2. **Single-frame storage** (4× savings on top): Store individual frames, reconstruct stacks
3. **Combined**: 100K × 226 KB / 16 ≈ 1.4 GB → **1M buffer in ~14 GB**

**Recommendation**: Medium priority but worth implementing in Phase 3 for training quality.

---

### 2.8 🟠 MEDIUM #8: Configuration vs Code Mismatches

**Verdict: ✅ VALIDATED — All 3 mismatches confirmed**

| Config Parameter | Config Value | Code Behavior | Status |
|-----------------|-------------|---------------|--------|
| `train_freq` | 50 | Never read, trains every step | Marked `TODO(DEAD_CODE)` ✅ |
| `gradient_steps` | 1 | Never read | Marked `TODO(DEAD_CODE)` ✅ |
| `learning_starts` comment | "10K steps" | Value is 1000 | Marked `TODO(MISLEADING_COMMENT)` ✅ |
| `networks.actor.learning_rate` | 0.001 | Never read, 0.0003 used | Marked `TODO(DEAD_CODE)` ✅ |
| `networks.critic.learning_rate` | 0.001 | Never read, 0.0003 used | Marked `TODO(DEAD_CODE)` ✅ |
| `networks.cnn.learning_rate` | 0.001 | Never read, 0.0003 used | Marked `TODO(DEAD_CODE)` ✅ |

#### Additional Dead Code Discovered

| Location | Dead Code | Why |
|----------|-----------|-----|
| `reward_functions.py` | `self.collision_penalty` | Loaded from config but hardcoded -5/-25/-100 used |
| `reward_functions.py` | `self.offroad_penalty` | Loaded from config but hardcoded -20.0 used |
| `reward_functions.py` | `self.wrong_way_penalty` | Loaded from config but passed as parameter |
| `td3_agent.py` | `self.actor_cnn_optimizer` | Always None, deprecated |
| `td3_agent.py` | `self.critic_cnn_optimizer` | Always None, deprecated |
| `td3_agent.py` | All `if self.*_cnn_optimizer is not None:` guards | Dead branches |

All marked with `TODO(DEAD_CODE)` in the codebase.

---

### 2.9 🔵 LOW #9: Weight Initialization Inconsistency

**Verdict: ✅ VALIDATED as-is**

- Actor: `nn.init.uniform_(-1/√f, 1/√f)` — equivalent to PyTorch default
- CNN: `nn.init.kaiming_normal_(fan_out, leaky_relu)` — appropriate for LeakyReLU
- Official TD3: PyTorch default (equivalent to Actor's uniform)

Minor inconsistency, but not a root cause. If CNN is overhauled (Findings #4+#5), re-initialization is implicit.

---

## 3. Interaction Matrix (Updated)

The original investigation identified a cascading failure chain. This validation confirms it and adds nuance:

```
                    ┌──────────────────────────┐
                    │   learning_starts=1000    │ Finding #3
                    │   (25× too low)           │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Buffer has 1K entries    │
                    │  batch=256 = 25.6% buffer │ Finding #7 (compounds)
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                   ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │ Critic over-  │   │  CNN gets    │   │ Actor gets   │
    │ fits narrow   │   │  [-1,1]+LN+  │   │ frozen by    │
    │ distribution  │   │  LeakyReLU   │   │ grad clip=1  │
    │               │   │  untested    │   │ across 1.94M │
    │ Finding #2    │   │  combo       │   │ params       │
    │ amplifies via │   │              │   │              │
    │ TD3 pessimism │   │ Findings     │   │ Finding #1   │
    │               │   │ #4 + #5      │   │ + #6 (decay) │
    └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
           │                  │                   │
           └──────────────────┼───────────────────┘
                              ▼
                    ┌──────────────────────┐
                    │  DEGENERATE POLICY   │
                    │  steer=±1, throttle=1│
                    │  Cannot escape:      │
                    │  - Gradients too tiny │
                    │  - Q-landscape toxic  │
                    │  - Features untested  │
                    └──────────────────────┘
```

---

## 4. Revised Fix Plan

### Phase 1: Critical Fixes (Must Apply Together)

| # | Fix | File(s) | Change | Rationale |
|---|-----|---------|--------|-----------|
| 1 | **Remove actor gradient clipping** | `td3_config.yaml` | `actor_max_norm: null` | Official TD3 has none; 1.94M params makes max_norm=1.0 catastrophic |
| 2 | **Increase learning_starts to 25000** | `td3_config.yaml` | `learning_starts: 25000` | Official TD3 default; 1000 causes 25.6% buffer sampling |
| 3 | **Remove weight decay** | `td3_agent.py` | Remove `weight_decay=1e-4` from both optimizers | Official TD3 has none; compounds with gradient clipping |

### Phase 2: Architecture Overhaul (Apply After Phase 1 Succeeds)

| # | Fix | File(s) | Change | Rationale |
|---|-----|---------|--------|-----------|
| 4+5 | **Replace CNN with standard NatureCNN** | `cnn_extractor.py`, `sensors.py`, `carla_env.py` | Remove LayerNorm, LeakyReLU→ReLU, [-1,1]→[0,1] | Findings #4+#5 are coupled; standard architecture validated in all visual RL literature |
| 6 | **Clip minimum reward to -10** | `reward_functions.py` or `carla_env.py` | `total_reward = max(total_reward, -10.0)` | Reduces 14:1→1.5:1 asymmetry; limits TD3 pessimism amplification |

### Phase 3: Configuration Cleanup and Buffer Optimization

| # | Fix | File(s) | Change | Rationale |
|---|-----|---------|--------|-----------|
| 7 | **Remove dead config** | `td3_config.yaml` | Remove `train_freq`, `gradient_steps`, dead LRs | Already marked with `TODO(DEAD_CODE)` |
| 8 | **Wire config penalties into code** | `reward_functions.py` | Use `self.collision_penalty` etc. instead of hardcoded values | Config has no effect right now |
| 9 | **Remove deprecated CNN optimizers** | `td3_agent.py` | Remove `actor_cnn_optimizer`, `critic_cnn_optimizer`, and all guards | Already marked with `TODO(DEAD_CODE)` |
| 10 | **Optimize buffer for 500K+** | `dict_replay_buffer.py` | uint8 storage + float32 on sampling | 4× memory savings, enables larger buffer |

---

## 5. Validation Criteria (From Original, Unchanged)

After applying Phase 1 fixes, within 50K training steps:

1. **Actor loss changes sign**: Should alternate positive/negative (not stuck negative)
2. **Q-values bounded**: Q1, Q2 in [-100, +100] (not exploding)
3. **Action diversity**: Rolling mean |steering| < 0.7 (not saturated at 1.0)
4. **CNN L2 stability**: Feature norm in [10, 50] (not growing unbounded)
5. **Episode reward improvement**: Average reward increases after ~10K learning steps
6. **No gradient explosion**: Actor gradient norm < 100 before any clipping

---

## 6. Dead Code Summary

All dead code has been marked in the codebase with `TODO(DEAD_CODE)` or `TODO(MISLEADING_COMMENT)`:

### td3_config.yaml
- Line 67: `learning_starts: 1000` — misleading comment says "10K"
- Line 75: `train_freq: 50` — never read by training script
- Line 76: `gradient_steps: 1` — never read by training script
- Line 130: `networks.cnn.learning_rate: 0.001` — never read
- Line 137: `networks.actor.learning_rate: 0.001` — never read
- Line 149: `networks.critic.learning_rate: 0.001` — never read

### reward_functions.py
- Line 92: `self.collision_penalty` — loaded but never used (hardcoded values in method)
- Line 93: `self.offroad_penalty` — loaded but never used (hardcoded values in method)
- Line 94-96: `self.wrong_way_penalty` — loaded but passed as parameter from carla_env.py

### td3_agent.py
- Line 302: `self.actor_cnn_optimizer = None` — deprecated, always None
- Line 303: `self.critic_cnn_optimizer = None` — deprecated, always None
- Lines ~819-820: `if self.critic_cnn_optimizer is not None:` guard — dead branch
- Lines ~1032-1033: `if self.actor_cnn_optimizer is not None:` guard — dead branch
- Lines ~1268-1278: Checkpoint save for `*_cnn_optimizer` — always saves None
- Lines ~1345-1360: Checkpoint load for `*_cnn_optimizer` — always skipped

---

*End of Expanded Validation Report*
