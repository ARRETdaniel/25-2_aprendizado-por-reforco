# TD3 Official Implementation Insights

**Source:** Official TD3 repository (Fujimoto et al. 2018)  
**Repository:** https://github.com/sfujim/TD3  
**Paper:** [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)  
**Date Analyzed:** October 20, 2025

---

## Overview

This document captures key insights from analyzing the official TD3 implementation to guide our CARLA autonomous vehicle project. The implementation is clean, well-commented, and demonstrates best practices for continuous control with deep RL.

---

## 1. Core Algorithm Implementation

### TD3.py - Twin Delayed DDPG

**Key Features:**
- **Twin Critics:** Two separate Q-networks (`Q1`, `Q2`) to mitigate overestimation bias
- **Delayed Updates:** Policy and target networks updated every `policy_freq` steps (default: 2)
- **Target Smoothing:** Clipped noise added to target policy actions

**Network Architecture:**
```python
Actor:
  - Input: state_dim
  - Hidden: Linear(state_dim, 256) → ReLU
  - Hidden: Linear(256, 256) → ReLU
  - Output: Linear(256, action_dim) → Tanh → scaled by max_action

Critic (Q1 and Q2, identical structure):
  - Input: state_dim + action_dim (concatenated)
  - Hidden: Linear(state + action, 256) → ReLU
  - Hidden: Linear(256, 256) → ReLU
  - Output: Linear(256, 1)
```

**Critical Implementation Details:**

1. **Target Q-value Calculation (with all 3 mechanisms):**
```python
with torch.no_grad():
    # Mechanism 3: Target Policy Smoothing
    noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
    next_action = (actor_target(next_state) + noise).clamp(-max_action, max_action)
    
    # Mechanism 1: Clipped Double Q-Learning
    target_Q1, target_Q2 = critic_target(next_state, next_action)
    target_Q = torch.min(target_Q1, target_Q2)
    target_Q = reward + not_done * discount * target_Q
```

2. **Critic Update (both networks):**
```python
current_Q1, current_Q2 = critic(state, action)
critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
```

3. **Mechanism 2: Delayed Policy Updates:**
```python
if total_it % policy_freq == 0:
    # Only update actor and targets every 'policy_freq' critic updates
    actor_loss = -critic.Q1(state, actor(state)).mean()
    
    # Update target networks (Polyak averaging)
    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

**Hyperparameters Used:**
- `learning_rate`: 3e-4 (both actor and critic)
- `discount`: 0.99
- `tau`: 0.005 (Polyak averaging)
- `policy_noise`: 0.2 * max_action
- `noise_clip`: 0.5 * max_action
- `policy_freq`: 2

---

## 2. DDPG Baseline Implementation

### OurDDPG.py - Re-tuned DDPG for Fair Comparison

**Key Differences from TD3:**
- Single critic network (no twin Q-learning)
- No delayed policy updates (updates every step)
- No target policy smoothing

**Architecture (IDENTICAL to TD3):**
- Actor: 256 → 256 → action_dim
- Critic: 256 → 256 → 1
- Same learning rate (3e-4), tau (0.005), batch size (256)

**Why This Matters:**
This is NOT the original DDPG from the paper (which uses 400→300 layers and different hyperparameters). This is a "re-tuned" DDPG baseline that matches TD3's architecture exactly, allowing us to isolate the impact of TD3's three algorithmic improvements.

**Implementation:**
```python
# Standard DDPG update (every step)
target_Q = critic_target(next_state, actor_target(next_state))
target_Q = reward + (not_done * discount * target_Q).detach()

current_Q = critic(state, action)
critic_loss = F.mse_loss(current_Q, target_Q)

actor_loss = -critic(state, actor(state)).mean()

# Update both networks every step
# Update target networks with Polyak averaging
```

---

## 3. Training Loop Structure

### main.py - Training Orchestration

**Key Training Components:**

1. **Warm-up Phase (Random Exploration):**
```python
if t < start_timesteps:  # First 25,000 timesteps
    action = env.action_space.sample()  # Uniform random
```

2. **Active Learning Phase:**
```python
else:
    action = (
        policy.select_action(state)
        + np.random.normal(0, max_action * expl_noise, size=action_dim)
    ).clip(-max_action, max_action)
```

3. **Episode Termination Handling:**
```python
done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
```
This prevents the agent from learning that reaching max episode length is a "terminal" state.

4. **Training After Warm-up:**
```python
if t >= start_timesteps:
    policy.train(replay_buffer, batch_size)
```

5. **Periodic Evaluation:**
```python
if (t + 1) % eval_freq == 0:
    evaluations.append(eval_policy(policy, env_name, seed))
```

**Hyperparameters:**
- `start_timesteps`: 25,000 (warm-up)
- `max_timesteps`: 1,000,000 (total training)
- `eval_freq`: 5,000 (evaluation interval)
- `expl_noise`: 0.1 (exploration noise std)
- `batch_size`: 256

---

## 4. Replay Buffer

### utils.py - Experience Storage

**Implementation:**
- Fixed-size circular buffer (1M transitions)
- Stores: `(state, action, next_state, reward, not_done)`
- Returns PyTorch tensors on GPU for efficient training

**Key Features:**
```python
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0  # Circular pointer
        self.size = 0  # Current buffer size
        
        # Pre-allocate arrays
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        # ... etc
    
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            # ... etc
        )
```

---

## 5. Key Takeaways for CARLA Implementation

### Architecture Decisions

1. **Network Size:** 256→256 hidden layers are sufficient for continuous control
   - Smaller than common practice (often 400→300)
   - Faster training, less prone to overfitting
   - Proven effective on MuJoCo benchmarks

2. **Activation Functions:** ReLU throughout (not tanh)
   - TD3 paper explicitly uses ReLU for hidden layers
   - Only output layer uses tanh (for bounded actions)

3. **Learning Rate:** 3e-4 is the sweet spot
   - Same for actor and critic (simplified)
   - Adam optimizer for both

### Training Strategy

1. **Warm-up is Critical:** 25,000 random steps before learning
   - Fills replay buffer with diverse experiences
   - Prevents early policy collapse

2. **Update Frequency:** Update networks every single step (after warm-up)
   - High sample efficiency
   - Requires stable gradient computation

3. **Delayed Updates:** Policy lag (2 steps) improves stability
   - Allows Q-functions to converge before exploitation
   - Prevents oscillations

### Exploration Strategy

1. **Training:** Gaussian noise (σ = 0.1 * max_action)
2. **Evaluation:** Deterministic policy (no noise)
3. **Target Smoothing:** Separate noise for target Q-value calculation

### Implementation Best Practices

1. **Target Networks:** Always use `torch.no_grad()` for target computation
2. **Soft Updates:** Polyak averaging (τ = 0.005) is gentler than hard updates
3. **Gradient Clipping:** Not used in original implementation (networks are stable)
4. **Batch Normalization:** Not used (simple MLP is sufficient)

---

## 6. Adaptations for CARLA Visual Navigation

### State Representation

**Original TD3:** Low-dimensional vector state (e.g., joint angles, velocities)  
**Our Adaptation:** High-dimensional visual + vector state

**Changes Needed:**
1. Add CNN feature extractor before actor/critic MLPs
2. Concatenate CNN features with kinematic/waypoint data
3. Ensure state preprocessing (normalization) is consistent

**Architecture Flow:**
```
Camera Images (4 × 256 × 144) 
    ↓ [Preprocess: resize, grayscale, normalize]
(4 × 84 × 84)
    ↓ [CNN Feature Extractor: NatureCNN/ResNet]
Feature Vector (512-dim)
    ↓ [Concatenate]
State Vector = [CNN features (512) + velocity (1) + lateral_dev (1) + heading_err (1) + waypoints (20)]
    ↓ [Actor/Critic Networks: same 256→256 architecture]
Action (2-dim: steering, throttle/brake)
```

### Action Mapping

**Original:** Actions directly compatible with gym environment  
**Our Challenge:** Map [-1, 1] to CARLA's separate throttle/brake controls

**Solution:**
```python
def map_action_to_carla(action):
    steering = action[0]  # [-1, 1] → directly to CARLA
    throttle_brake = action[1]  # [-1, 1]
    
    if throttle_brake >= 0:
        throttle = throttle_brake  # [0, 1]
        brake = 0.0
    else:
        throttle = 0.0
        brake = -throttle_brake  # [0, 1]
    
    return steering, throttle, brake
```

### Reward Engineering

**Original:** Task-specific rewards (e.g., forward velocity for HalfCheetah)  
**Our Task:** Multi-component reward for safe, efficient, comfortable driving

**Components (from our config):**
1. Efficiency: Track target speed (~10 m/s)
2. Lane Keeping: Minimize lateral deviation and heading error
3. Comfort: Penalize high jerk
4. Safety: Large negative penalty for collisions (-1000)

**Implementation Note:** Reward scaling is critical for TD3 convergence. Keep total reward magnitude around [-100, 100] per step for best results.

### Episode Termination

**Original:** Fixed episode length or task completion  
**Our Task:** Multiple termination conditions

**Conditions:**
1. Collision detected → `done = True, reward = -1000`
2. Off-road too long → `done = True, reward = -500`
3. Goal reached → `done = True, reward = +bonus`
4. Max steps (1000) → `done = False` (time limit, not terminal state)

**Critical:** Use `done_bool` logic to prevent learning false terminal states at time limits.

---

## 7. Comparison: TD3 vs DDPG

| Feature | TD3 | DDPG (OurDDPG) |
|---------|-----|----------------|
| **Critics** | Twin (Q1, Q2) | Single (Q) |
| **Target Calculation** | min(Q1', Q2') | Q' |
| **Policy Updates** | Every 2 critic updates | Every step |
| **Target Smoothing** | Yes (noise + clip) | No |
| **Learning Rate** | 3e-4 | 3e-4 |
| **Network Size** | 256→256 | 256→256 |
| **Tau** | 0.005 | 0.005 |
| **Batch Size** | 256 | 256 |

**Expected Outcome:** TD3 should show:
- Higher success rate (fewer collisions)
- More stable learning (smoother reward curves)
- Better final performance (higher average reward)

---

## 8. Integration with Stable-Baselines3 (Optional)

While the official implementation is clean and educational, Stable-Baselines3 provides:
- Well-tested, production-ready code
- Better logging (TensorBoard, WandB)
- Callback system for custom training logic
- Vectorized environments for parallel training
- Checkpoint management

**Our Approach:**
1. Start with official implementation for understanding
2. Adapt to CARLA environment (custom wrapper)
3. Consider migrating to SB3 for production training if time permits

**SB3 TD3 Usage:**
```python
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

model = TD3(
    policy='CnnPolicy',  # For image input
    env=carla_env,
    learning_rate=3e-4,
    buffer_size=1000000,
    learning_starts=25000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    policy_delay=2,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    action_noise=NormalActionNoise(mean=0, sigma=0.1*max_action),
    verbose=1
)

model.learn(total_timesteps=2000000)
```

---

## 9. Next Steps for Implementation

### Priority Order:

1. **CARLA Environment Wrapper** (`carla_env.py`)
   - Most critical component
   - Blocks all agent training
   - Implement state construction, reward, termination

2. **CNN Feature Extractor** (`cnn_extractor.py`)
   - Process 4-frame stack
   - Output 512-dim features
   - Use NatureCNN or lightweight ResNet

3. **Networks** (`actor.py`, `critic.py`)
   - Implement 256→256 architecture
   - Match official TD3 structure
   - Ensure gradient flow

4. **TD3 Agent** (`td3_agent.py`)
   - Adapt official TD3.py for visual input
   - Integrate CNN extractor
   - Add CARLA-specific logic

5. **Training Script** (`train_td3.py`)
   - Load configs
   - Initialize environment
   - Run training loop
   - Handle checkpointing

6. **DDPG Baseline** (`ddpg_agent.py`)
   - Copy TD3, remove 3 mechanisms
   - Ensure identical architecture
   - Train for comparison

---

## 10. References

1. **TD3 Paper:** Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods," ICML 2018
2. **Official Code:** https://github.com/sfujim/TD3
3. **DDPG Paper:** Lillicrap et al., "Continuous control with deep reinforcement learning," ICLR 2016
4. **Stable-Baselines3:** https://stable-baselines3.readthedocs.io/
5. **OpenAI Spinning Up:** https://spinningup.openai.com/en/latest/algorithms/td3.html

---

## Appendix: File Analysis Summary

### Files Analyzed:
1. **TD3.py** (155 lines) - Core TD3 algorithm
2. **OurDDPG.py** (127 lines) - Re-tuned DDPG baseline
3. **DDPG.py** (136 lines) - Original DDPG (not used in paper)
4. **main.py** (136 lines) - Training loop
5. **utils.py** (42 lines) - Replay buffer
6. **README.md** - Experiment details

### Key Insights Per File:

**TD3.py:**
- Clean separation of Actor/Critic/TD3 classes
- All three TD3 mechanisms clearly implemented
- GPU support via global `device` variable
- Save/load functionality for checkpoints

**OurDDPG.py:**
- Identical architecture to TD3 (256→256)
- Simplified update logic (no delayed updates, no smoothing)
- Perfect baseline for isolating TD3 improvements

**main.py:**
- Training loop structure is straightforward
- Warm-up phase with random actions
- Periodic evaluation for monitoring
- Command-line arguments for all hyperparameters

**utils.py:**
- Minimal, efficient replay buffer
- Pre-allocated numpy arrays for speed
- PyTorch tensor conversion on sampling

---

**Document Version:** 1.0  
**Last Updated:** October 20, 2025  
**Author:** Daniel Terra Gomes  
**Project:** End-to-End Visual Autonomous Navigation with TD3
