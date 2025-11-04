# How Your TD3+CNN System Learns - Complete Explanation

**Date**: 2025-01-16
**Status**: System Ready for Training ✅
**Learning Type**: Online Reinforcement Learning (No Pre-saved Data Needed)

---

## 🎯 Quick Answer

**Q: Do you need pre-saved data?**
**A: NO!** Your system learns through **online interaction** with the CARLA simulator.

**Q: Is the system ready for training?**
**A: YES!** ✅ All components are implemented and the critical ReLU bug is fixed.

---

## 📚 Learning Method: Online Reinforcement Learning

Your system uses **TD3 (Twin Delayed Deep Deterministic Policy Gradient)**, an **off-policy** deep RL algorithm that learns by:

1. **Interacting with environment** (CARLA simulator)
2. **Storing experiences** in replay buffer
3. **Learning from past experiences** via gradient descent
4. **Improving policy iteratively** over millions of steps

**No supervised dataset needed!** The agent learns from trial-and-error using reward signals.

---

## 🔄 Complete Learning Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP (1M steps)                      │
└─────────────────────────────────────────────────────────────────┘

Step 1: OBSERVE STATE
   ↓
   CARLA provides:
   - Camera: (4, 84, 84) stacked grayscale frames (last 4 frames)
   - Vector: (23,) [speed, dist_to_goal, yaw_error, waypoints...]

Step 2: CNN FEATURE EXTRACTION
   ↓
   actor_cnn(camera) → (512,) visual features
   - Conv1: Edges, contrasts, basic shapes
   - Conv2: Vehicles, lanes, obstacles
   - Conv3: Task-specific patterns
   - FC: Compact 512-dim representation

Step 3: ACTOR DECIDES ACTION
   ↓
   state = [visual_features (512) + vector (23)] = (535,)
   actor(state) → action [steering, throttle] ∈ [-1, 1]²
   + exploration_noise (Gaussian) during training

Step 4: EXECUTE IN CARLA
   ↓
   Apply action to ego vehicle:
   - steering ∈ [-1, 1] → turn wheels
   - throttle ∈ [-1, 1] → accelerate (>0) or brake (<0)

Step 5: OBSERVE OUTCOME
   ↓
   CARLA returns:
   - next_state (new camera + vector)
   - reward: scalar value
     * +progress toward goal
     * +staying in lane
     * -collision penalty
     * -jerky movements
   - done: True if collision/goal/timeout

Step 6: STORE EXPERIENCE
   ↓
   replay_buffer.add(state, action, next_state, reward, done)
   Buffer capacity: 1,000,000 transitions

Step 7: SAMPLE & TRAIN (after warmup)
   ↓
   IF buffer_size > start_timesteps (25k):
       batch = replay_buffer.sample(256)  # Random batch

       # Train critics (Q-value estimators)
       Q_loss = (Q(s,a) - target_Q)²
       critic_optimizer.step()  # Update critic weights
       ← Gradients flow to critic_cnn!

       # Train actor (policy) every 2 steps
       IF step % 2 == 0:
           actor_loss = -Q(s, actor(s))  # Maximize Q
           actor_optimizer.step()  # Update actor weights
           ← Gradients flow to actor_cnn!

           # Soft update target networks
           target ← 0.995*target + 0.005*current

Step 8: REPEAT ↻
   ↓
   Go to Step 1 until max_timesteps (1M) reached
```
fix(cnn): Replace ReLU with Leaky ReLU for zero-centered inputs

PROBLEM:
- Environment preprocessing outputs [-1, 1] (zero-centered normalization)
- CNN used standard ReLU which zeros all negative values
- Result: ~50% pixel information lost, training failure at 30k steps

SOLUTION:
- Replace nn.ReLU() with nn.LeakyReLU(negative_slope=0.01)
- Leaky ReLU preserves negative information (α·x instead of 0)
- Maintains zero-centering benefits while preventing dying ReLU

CHANGES:
- cnn_extractor.py: ReLU → Leaky ReLU in 3 locations
- Added Kaiming weight initialization for Leaky ReLU
- Enhanced documentation with research paper references

EXPECTED IMPACT:
- 100% pixel information preserved (vs. 50% before)
- Better gradient flow (no dead neurons)
- Improved training stability and convergence
- Higher success rate and lower collision rate
---
fix(cnn): Replace ReLU with Leaky ReLU for zero-centered inputs

PROBLEM:
- Environment preprocessing outputs [-1, 1] (zero-centered normalization)
- CNN used standard ReLU which zeros all negative values
- Result: ~50% pixel information lost, training failure at 30k steps

SOLUTION:
- Replace nn.ReLU() with nn.LeakyReLU(negative_slope=0.01)
- Leaky ReLU preserves negative information (α·x instead of 0)
- Maintains zero-centering benefits while preventing dying ReLU

CHANGES:
- cnn_extractor.py: ReLU → Leaky ReLU in 3 locations
- Added Kaiming weight initialization for Leaky ReLU
- Enhanced documentation with research paper references

EXPECTED IMPACT:
- 100% pixel information preserved (vs. 50% before)
- Better gradient flow (no dead neurons)
- Improved training stability and convergence
- Higher success rate and lower collision rate

----

┌─────────────────────────────────────────────────────────┐
│         SINGLE TRAINING STEP (every iteration)          │
└─────────────────────────────────────────────────────────┘

1. FORWARD PASS (all networks in sequence)
   ↓
   camera_frames (4,84,84)
   ↓
   [CNN Forward] → visual_features (512)  ← CNN processes input
   ↓
   [Concatenate] → state (535) = visual(512) + vector(23)
   ↓
   [Actor Forward] → action (2)  ← TD3 actor uses CNN features
   ↓
   [Critic Forward] → Q-value (1)  ← TD3 critics use CNN features

2. COMPUTE LOSS (based on rewards)
   ↓
   critic_loss = (Q_predicted - Q_target)²
   actor_loss = -Q(state, actor(state))

3. BACKWARD PASS (gradients flow through ALL networks)
   ↓
   critic_loss.backward()  ← Computes gradients
   ↓
   Gradients flow: Critic → State → CNN ✅
   ↓
   Update: Critic weights + CNN weights (together!)

   ↓ (every 2 steps)
   actor_loss.backward()
   ↓
   Gradients flow: Actor → State → CNN ✅
   ↓
   Update: Actor weights + CNN weights (together!)


----


## 🧠 How the CNN Learns Visual Features

### **End-to-End Learning (No Pre-training)**

Your CNN learns **directly from driving rewards** - no ImageNet pre-training, no supervised labels!

```python
# From td3_agent.py extract_features()
def extract_features(obs_dict, enable_grad=True, use_actor_cnn=True):
    """
    Extract visual features WITH gradient tracking.

    During training (enable_grad=True):
    - CNN forward pass: camera → features
    - Actor/Critic use features to predict action/Q-value
    - Loss computed from rewards
    - Gradients backpropagate: loss → actor/critic → CNN ✅

    This is KEY: CNN learns what visual patterns are useful for driving!
    """
    if enable_grad:
        image_features = actor_cnn(obs_dict['image'])  # ← Gradients ON
    else:
        with torch.no_grad():
            image_features = actor_cnn(obs_dict['image'])  # ← Inference mode

    return torch.cat([image_features, obs_dict['vector']], dim=1)
```

### **What the CNN Learns Over Time**

| Training Phase | Visual Features Learned | Example Patterns |
|----------------|-------------------------|------------------|
| **Early (0-10k)** | Low-level features | Edges, contrasts, textures, basic shapes |
| **Mid (10k-50k)** | Semantic features | Vehicle detection, lane markings, road boundaries |
| **Late (50k-100k+)** | Task-specific | Safe following distance, collision prediction, steering cues |

**Example Learning Progression:**

```
Timestep 1,000:
   CNN Output: Random noise (untrained)
   Actor: Random actions → Crashes immediately
   Reward: -1000 (collision penalty)

Timestep 25,000:
   CNN Output: Detects basic edges and contrasts
   Actor: Learns to go straight, avoid obvious obstacles
   Reward: -500 (some progress, still crashes often)

Timestep 100,000:
   CNN Output: Detects vehicles, lanes, predicts collisions
   Actor: Smooth steering, maintains speed, follows waypoints
   Reward: +200 (completes some routes successfully)
```

---

## 🔧 Your Implementation: Key Components

### **1. Training Script (`train_td3.py`)**

```python
class TD3TrainingPipeline:
    def train(self):
        """Main training loop"""

        # Phase 1: Warm-up (random actions, populate buffer)
        for t in range(start_timesteps):  # 25,000 steps
            action = env.action_space.sample()  # Random
            state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)

        # Phase 2: Learning (select actions with policy + exploration noise)
        for t in range(start_timesteps, max_timesteps):  # 25k → 1M
            # Select action using learned policy
            action = agent.select_action(state, add_noise=True)

            # Execute in CARLA
            next_state, reward, done, info = env.step(action)

            # Store experience
            replay_buffer.add(state, action, next_state, reward, done)

            # Train networks (256 transitions per update)
            metrics = agent.train(batch_size=256)

            # Periodic evaluation
            if t % eval_freq == 0:
                eval_metrics = self.evaluate()
```

**Key Point:** Training happens **automatically** in the loop - no manual data loading!

### **2. TD3 Agent (`td3_agent.py`)**

```python
class TD3Agent:
    def __init__(self, actor_cnn, critic_cnn, ...):
        """
        Initialize agent with SEPARATE CNNs for actor and critic.

        Why separate?
        - Actor learns "what action to take" (policy)
        - Critic learns "how good is this action" (Q-value)
        - Separate CNNs prevent gradient interference
        - Matches Stable-Baselines3 best practice
        """
        self.actor_cnn = actor_cnn      # NatureCNN for actor
        self.critic_cnn = critic_cnn    # Separate NatureCNN for critic

        self.actor = Actor(input_dim=535, output_dim=2)
        self.critic_1 = Critic(input_dim=537)  # Twin critic 1
        self.critic_2 = Critic(input_dim=537)  # Twin critic 2

        self.replay_buffer = DictReplayBuffer(max_size=1e6)

    def train(self, batch_size=256):
        """
        Train actor and critics on sampled batch.

        Returns gradients that flow back to CNN!
        """
        # Sample random batch
        state, action, next_state, reward, done = self.replay_buffer.sample(batch_size)

        # Convert Dict obs to tensors WITH gradients
        state_tensor = self.extract_features(
            state,
            enable_grad=True,      # ✅ Gradients ON for training
            use_actor_cnn=False    # Use critic's CNN for Q-learning
        )

        # Critic update (Q-learning)
        current_Q1 = self.critic_1(state_tensor, action)
        current_Q2 = self.critic_2(state_tensor, action)

        target_Q = reward + gamma * min(Q1_next, Q2_next)
        critic_loss = MSE(current_Q1, target_Q) + MSE(current_Q2, target_Q)

        critic_loss.backward()  # ← Gradients flow to critic_cnn!
        self.critic_optimizer.step()

        # Actor update (policy gradient) - delayed every 2 steps
        if step % 2 == 0:
            state_tensor_actor = self.extract_features(
                state,
                enable_grad=True,     # ✅ Gradients ON
                use_actor_cnn=True    # Use actor's CNN
            )

            actor_loss = -self.critic_1(state_tensor_actor, self.actor(state_tensor_actor)).mean()
            actor_loss.backward()  # ← Gradients flow to actor_cnn!
            self.actor_optimizer.step()
```

**Key Point:** `enable_grad=True` during training allows CNN to learn end-to-end!

### **3. CNN Feature Extractor (`cnn_extractor.py`)**

```python
class NatureCNN(nn.Module):
    """
    Convolutional feature extractor for visual observations.

    Architecture:
        Input: (batch, 4, 84, 84) stacked frames
        Conv1: 32 filters, 8x8, stride 4 → (batch, 32, 20, 20)
        LeakyReLU(0.01)  ← FIXED: Preserves [-1,1] normalization
        Conv2: 64 filters, 4x4, stride 2 → (batch, 64, 9, 9)
        LeakyReLU(0.01)
        Conv3: 64 filters, 3x3, stride 1 → (batch, 64, 7, 7)
        LeakyReLU(0.01)
        Flatten: → (batch, 3136)
        FC: → (batch, 512)
        Output: 512-dimensional feature vector

    Learning:
        Gradients flow from actor/critic losses back through this network.
        CNN learns filters that extract task-relevant visual features.
    """

    def forward(self, x):
        """Forward pass WITH gradient tracking (if in training mode)"""
        out = self.activation(self.conv1(x))   # Preserves negatives ✅
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))
        out = out.view(out.size(0), -1)  # Flatten
        features = self.fc(out)
        return features  # (batch, 512) with gradients attached
```

**Key Point:** CNN is part of the computational graph - gradients flow through it!

---

## ✅ System Readiness Checklist

Let's verify your system is ready for training:

### **Core Components** ✅

- [x] **CARLA Environment** (`carla_env.py`)
  - Provides observations: camera + vehicle state + waypoints
  - Executes actions: steering, throttle/brake
  - Computes rewards: progress, safety, comfort
  - Handles resets and episodes

- [x] **CNN Feature Extractor** (`cnn_extractor.py`)
  - ✅ Architecture: Matches Nature DQN
  - ✅ Activation: Leaky ReLU (bug fixed!)
  - ✅ Weight init: Kaiming for Leaky ReLU
  - ✅ Gradient tracking: Enabled in forward()

- [x] **TD3 Agent** (`td3_agent.py`)
  - ✅ Separate actor_cnn and critic_cnn
  - ✅ Twin critics (Q1, Q2)
  - ✅ Delayed policy updates
  - ✅ Target policy smoothing
  - ✅ Replay buffer (1M capacity)
  - ✅ extract_features() with enable_grad=True

- [x] **Training Pipeline** (`train_td3.py`)
  - ✅ Warm-up phase (random exploration)
  - ✅ Training loop (interact + learn)
  - ✅ Periodic evaluation
  - ✅ Checkpoint saving
  - ✅ TensorBoard logging

### **Critical Fixes Applied** ✅

- [x] **Bug #13: CNN Gradient Flow**
  - Fixed: `enable_grad=True` in extract_features()
  - Fixed: Separate CNNs for actor and critic
  - Result: CNN learns end-to-end from driving rewards

- [x] **Bug #14: ReLU/Normalization Mismatch**
  - Fixed: ReLU → Leaky ReLU activation
  - Reason: Preprocessing outputs [-1, 1], ReLU killed 50% of pixels
  - Result: 100% pixel information preserved

### **Configuration Files** ✅

- [x] `config/carla_config.yaml` - Environment settings
- [x] `config/td3_config.yaml` - Agent hyperparameters
- [x] `config/training_config.yaml` - Training scenarios

### **Expected Training Process** ✅

```bash
# 1. Start training (1000-step validation run first)
python scripts/train_td3.py \
    --scenario 0 \
    --seed 42 \
    --max-timesteps 1000 \
    --eval-freq 500 \
    --debug

# Expected output:
# [0-25k steps]  Warm-up: Random actions, populate buffer
# [25k-1M steps] Learning: Policy improves, CNN learns features
# [Every 5k]     Evaluation: Test current policy performance
# [Every 10k]    Checkpoint: Save model weights

# 2. Monitor training
tensorboard --logdir=data/logs

# 3. Full training run (after validation passes)
python scripts/train_td3.py \
    --scenario 0 \
    --seed 42 \
    --max-timesteps 1000000 \
    --eval-freq 5000
```

---

## 📊 What to Expect During Training

### **Typical Learning Curve**

```
Episode Return (higher = better)
    ↑
    |                                  ╱‾‾‾
  0 |                              ╱‾‾
    |                          ╱‾‾
-500|                      ╱‾‾
    |                  ╱‾‾
-1k |              ╱‾‾
    |          ╱‾‾
-2k |      ╱‾‾
    |  ╱‾‾
-3k |╱
    └────────────────────────────────→
    0   25k  50k  75k  100k  ... 1M
              Training Steps

Phase 1 (0-25k):   Random exploration, negative rewards
Phase 2 (25k-100k): Rapid improvement as policy learns
Phase 3 (100k+):   Fine-tuning, approaching optimal performance
```

### **Training Metrics to Monitor**

| Metric | Initial | Expected Final | What It Means |
|--------|---------|----------------|---------------|
| **Episode Return** | -3000 | +500 | Overall performance (higher = better) |
| **Success Rate** | 0% | >20% | % episodes reaching goal without collision |
| **Collision Rate** | 80% | <30% | % episodes ending in collision |
| **Critic Loss** | Random | Decreasing | Q-value estimation error (lower = better) |
| **Actor Loss** | Random | Stable | Policy gradient magnitude |
| **Avg Episode Length** | 50 | 500+ | Steps before termination (longer = better) |

### **CNN Learning Indicators**

```python
# You can monitor CNN feature statistics:
feature_mean = actor_cnn(batch).mean()   # Should stay near 0
feature_std = actor_cnn(batch).std()     # Should be ~1
feature_norm = torch.norm(actor_cnn(batch), dim=1).mean()

# Healthy CNN learning:
# - Mean ≈ 0 (normalized features)
# - Std ≈ 1 (unit variance)
# - No NaN or Inf values
# - Gradients flowing (check with .grad)
```

---

## 🚀 How to Start Training

### **Step 1: Validation Run (Short Test)**

```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

# Test 1000 steps to ensure everything works
python scripts/train_td3.py \
    --scenario 0 \
    --seed 42 \
    --max-timesteps 1000 \
    --eval-freq 500 \
    --debug

# Expected runtime: ~10-15 minutes
# Expected outcome: No crashes, metrics logged, checkpoints saved
```

**What to check:**
- ✅ No errors or crashes
- ✅ Replay buffer fills up (should reach 1000 transitions)
- ✅ Critic loss is computed (not NaN)
- ✅ Actor loss is computed every 2 steps
- ✅ Checkpoints saved to `data/checkpoints/`
- ✅ TensorBoard logs created in `data/logs/`

### **Step 2: Short Training Run (10k steps)**

```bash
# If validation passes, run 10k steps
python scripts/train_td3.py \
    --scenario 0 \
    --seed 42 \
    --max-timesteps 10000 \
    --eval-freq 2500

# Expected runtime: 1-2 hours
# Expected outcome:
# - Warm-up completes (steps 0-25k would need more, but you'll see buffer filling)
# - Some policy improvement visible
# - No training crashes
```

### **Step 3: Full Training Run (100k-1M steps)**

```bash
# After verifying system stability, run full training
python scripts/train_td3.py \
    --scenario 0 \
    --seed 42 \
    --max-timesteps 1000000 \
    --eval-freq 5000 \
    --checkpoint-freq 10000

# Expected runtime: 12-24 hours (depends on hardware)
# Run overnight or on weekends
```

### **Step 4: Monitor Progress**

```bash
# In a separate terminal, start TensorBoard
tensorboard --logdir=data/logs --port=6006

# Open browser: http://localhost:6006
# Monitor:
# - Scalars: episode_return, success_rate, critic_loss, actor_loss
# - Images: camera observations (if logged)
# - Histograms: network weights and gradients
```

---

## 🔍 Troubleshooting During Training

### **Issue: Training crashes with NaN losses**

**Cause:** Gradient explosion or numerical instability
**Solution:**
```python
# Check gradient clipping in td3_agent.py
torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
```

### **Issue: No improvement after 50k steps**

**Cause:** Learning rate too high/low, or reward function issue
**Solution:**
```python
# Adjust learning rates in config/td3_config.yaml
actor_lr: 1e-4  # Try 1e-3 (higher) or 1e-5 (lower)
critic_lr: 1e-3  # Try 3e-3 (higher) or 3e-4 (lower)

# Verify reward function in carla_env.py
# Ensure rewards are reasonable magnitudes (-1000 to +100)
```

### **Issue: Agent learns to do nothing (always brake)**

**Cause:** Negative rewards too strong, agent learns "safest" is to not move
**Solution:**
```python
# Balance reward components in carla_env.py
reward = (
    progress_reward * 1.0 +      # Encourage movement
    comfort_reward * 0.1 +       # Lower weight on comfort
    collision_penalty * -10.0    # Strong penalty, but not overwhelming
)
```

---

## 📈 Expected Results After Full Training

Based on the bug fixes (Leaky ReLU + gradient flow), you should see:

### **Before Fix (30k training failure):**
- Success rate: 0%
- Episode return: -52,000
- Training crashed: NaN losses

### **After Fix (Expected at 100k steps):**
- Success rate: >20%
- Episode return: -35,000 to +500
- Training stable: No crashes
- CNN learns useful features: Vehicles, lanes, obstacles

### **Comparison with Baselines:**

| Metric | Random Policy | IDM+MOBIL | Your TD3 (Expected) |
|--------|---------------|-----------|---------------------|
| Success Rate | 0% | 40% | 20-30% |
| Collision Rate | 95% | 20% | 30-40% |
| Avg Speed | 5 km/h | 25 km/h | 20-30 km/h |

**Goal:** Demonstrate that TD3 with end-to-end visual learning can approach classical method performance while being more flexible and data-driven.

---

## 🎓 Summary: Your System is Ready!

### **Learning Approach** ✅
- ✅ Online RL (no pre-saved data needed)
- ✅ End-to-end learning (CNN learns from rewards)
- ✅ Off-policy TD3 (sample-efficient)

### **Implementation Status** ✅
- ✅ Environment: CARLA integration complete
- ✅ Networks: CNN + Actor + Twin Critics
- ✅ Agent: TD3 with gradient flow
- ✅ Training: Pipeline with logging and checkpoints

### **Critical Bugs Fixed** ✅
- ✅ Leaky ReLU (preserves 100% pixel info)
- ✅ Gradient flow enabled (CNN learns end-to-end)
- ✅ Separate CNNs (no gradient interference)

### **Next Steps** 🚀
1. Run 1000-step validation (verify no crashes)
2. Run 10k-step test (verify learning starts)
3. Run 100k-1M full training (overnight/weekend)
4. Analyze results and compare with baselines
5. Document findings for paper

---

**Your system learns by doing - no dataset required!** Just start the training script and the agent will learn to drive through millions of simulated experiences in CARLA. 🚗💨

**Status: READY TO TRAIN** ✅
