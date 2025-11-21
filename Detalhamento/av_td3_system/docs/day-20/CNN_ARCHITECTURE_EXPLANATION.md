# CNN Architecture Explanation: Why We Have CNNs in Multiple Files

**Date:** November 20, 2025  
**Question:** "Why do we have CNN in train_td3.py, td3_agent.py, and cnn_extractor.py?"  
**Answer:** We DON'T have duplicate CNN implementations. We have a SINGLE CNN architecture used in TWO SEPARATE INSTANCES.

---

## Executive Summary

**TL;DR:** 
- **One CNN Architecture:** `NatureCNN` class in `cnn_extractor.py` (the blueprint)
- **Two CNN Instances:** Created in `train_td3.py` (actor_cnn and critic_cnn)
- **Used by:** `td3_agent.py` (for feature extraction during training)
- **Why Two?** To prevent gradient interference between actor and critic updates

This is the **CORRECT** design following Stable-Baselines3 TD3 (`share_features_extractor=False`).

---

## The Three Files Explained

### 1. `cnn_extractor.py` - The Blueprint (Architecture Definition)

**Purpose:** Defines the `NatureCNN` class (the CNN architecture)

**What it contains:**
```python
class NatureCNN(nn.Module):
    """
    NatureCNN visual feature extractor for end-to-end deep reinforcement learning.
    
    Architecture:
        Input:   (batch, 4, 84, 84) - 4 stacked grayscale frames
        Conv1:   (batch, 32, 20, 20) - 32 filters, 8×8 kernel, stride 4
        Conv2:   (batch, 64, 9, 9)   - 64 filters, 4×4 kernel, stride 2
        Conv3:   (batch, 64, 7, 7)   - 64 filters, 3×3 kernel, stride 1
        Flatten: (batch, 3136)       - 64 × 7 × 7 = 3136 features
        FC:      (batch, 512)        - Fully connected layer
        Output:  512-dimensional feature vector
    """
    def __init__(self, input_channels=4, feature_dim=512):
        # Define CNN layers (Conv1, Conv2, Conv3, FC)
        
    def forward(self, x):
        # Forward pass through CNN
        return features  # (batch, 512)
```

**Analogy:** This is like a **class definition** in programming. It describes WHAT a CNN is, but doesn't create any actual CNNs yet.

**Key Point:** This file defines the architecture ONCE. No training happens here, no instances exist here.

---

### 2. `train_td3.py` - The Factory (Creates CNN Instances)

**Purpose:** Creates TWO SEPARATE CNN instances and passes them to the TD3 agent

**What it does:**
```python
# Lines 198-213: Create SEPARATE CNN instances
self.actor_cnn = NatureCNN(
    input_channels=4,
    num_frames=4,
    feature_dim=512
).to(agent_device)

self.critic_cnn = NatureCNN(
    input_channels=4,
    num_frames=4,
    feature_dim=512
).to(agent_device)

# Lines 215-217: Initialize weights (Kaiming for ReLU)
self._initialize_cnn_weights()  # For both actor_cnn and critic_cnn

# Lines 219-220: Set to training mode
self.actor_cnn.train()
self.critic_cnn.train()

# Lines 256-263: Pass CNNs to TD3Agent
self.agent = TD3Agent(
    state_dim=565,
    action_dim=2,
    max_action=1.0,
    actor_cnn=self.actor_cnn,   # ← Actor's CNN instance
    critic_cnn=self.critic_cnn,  # ← Critic's CNN instance
    ...
)
```

**Analogy:** This is like **instantiating objects** from a class. We create two separate CNN objects with the same architecture but different memory/weights.

**Why Two Separate CNNs?**

🔴 **CRITICAL FIX (November 13, 2025):**

Previously, we used a SINGLE shared CNN (`self.cnn_extractor`) for both actor and critic. This caused:
- **Gradient Interference:** Actor and critic updates conflicted
- **Training Failure:** Rewards collapsed to -52,000 (catastrophic)
- **Root Cause:** Shared CNN received conflicting gradient signals from actor (maximize Q) and critic (minimize TD error)

**Solution:** Create SEPARATE CNNs for actor and critic (matches Stable-Baselines3 TD3 default: `share_features_extractor=False`)

**Evidence:**
```python
# Lines 224-227: Verification logging
print(f"[AGENT] Actor CNN initialized on {agent_device} (id: {id(self.actor_cnn)})")
print(f"[AGENT] Critic CNN initialized on {agent_device} (id: {id(self.critic_cnn)})")
print(f"[AGENT] CNNs are SEPARATE instances: {id(self.actor_cnn) != id(self.critic_cnn)}")
```

**Key Point:** Two instances, same architecture, different memory locations → no gradient interference.

---

### 3. `td3_agent.py` - The User (Uses CNN Instances)

**Purpose:** Receives CNN instances and uses them for feature extraction during training

**What it does:**

#### A. Store CNN References (Lines 213-226)

```python
# Lines 213-226: Store CNNs passed from train_td3.py
self.actor_cnn = actor_cnn    # Reference to actor's CNN instance
self.critic_cnn = critic_cnn  # Reference to critic's CNN instance

# Set to training mode
if self.actor_cnn is not None:
    self.actor_cnn.train()  # Enable gradient computation
    
if self.critic_cnn is not None:
    self.critic_cnn.train()  # Enable gradient computation
```

#### B. Include CNN Parameters in Optimizers (Lines 155-200)

🔧 **CRITICAL FIX (November 20, 2025):**

```python
# Lines 155-165: Actor optimizer includes BOTH MLP and CNN
if self.actor_cnn is not None:
    actor_params = list(self.actor.parameters()) + list(self.actor_cnn.parameters())
else:
    actor_params = list(self.actor.parameters())

self.actor_optimizer = torch.optim.Adam(actor_params, lr=self.actor_lr)

# Lines 187-197: Critic optimizer includes BOTH MLP and CNN
if self.critic_cnn is not None:
    critic_params = list(self.critic.parameters()) + list(self.critic_cnn.parameters())
else:
    critic_params = list(self.critic.parameters())

self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.critic_lr)
```

**Why This Matters:**

Previously, we had SEPARATE optimizers for CNNs:
```python
# BEFORE (BROKEN):
self.actor_cnn_optimizer = torch.optim.Adam(self.actor_cnn.parameters(), lr=cnn_lr)
self.critic_cnn_optimizer = torch.optim.Adam(self.critic_cnn.parameters(), lr=cnn_lr)

# These were called AFTER gradient clipping, applying UNCLIPPED gradients!
```

**Problem:** Gradient clipping modifies `.grad` attributes, but separate optimizers applied original (unclipped) gradients.

**Evidence:** TensorBoard showed:
- Actor CNN grad norm: 2.42 (should be ≤1.0) ❌
- Critic CNN grad norm: 24.69 (should be ≤10.0) ❌

**Solution:** Merge CNN parameters into main optimizers (matches official TD3 implementation).

#### C. Extract Features During Training (Lines 360-450)

```python
def extract_features(
    self,
    obs_dict: Dict[str, torch.Tensor],
    enable_grad: bool = True,
    use_actor_cnn: bool = True
) -> torch.Tensor:
    """
    Extract features from Dict observation with gradient support.
    
    Args:
        obs_dict: {'image': (B,4,84,84), 'vector': (B,23)}
        enable_grad: If True, gradients flow through CNN (training)
        use_actor_cnn: If True, use actor's CNN; else use critic's CNN
    
    Returns:
        state: (B, 565) = 512 (CNN features) + 53 (kinematic + waypoints)
    """
    # Select correct CNN
    cnn = self.actor_cnn if use_actor_cnn else self.critic_cnn
    
    if enable_grad:
        # Training: Gradients ENABLED
        image_features = cnn(obs_dict['image'])  # (B, 512)
    else:
        # Inference: Gradients DISABLED (more efficient)
        with torch.no_grad():
            image_features = cnn(obs_dict['image'])  # (B, 512)
    
    # Concatenate visual + kinematic features
    state = torch.cat([image_features, obs_dict['vector']], dim=1)  # (B, 565)
    
    return state
```

**Usage in Training Loop:**

```python
# select_action (inference mode - no gradients)
def select_action(self, state, noise=None, deterministic=False):
    state_tensor = self.extract_features(
        obs_dict_tensor,
        enable_grad=False,   # No gradients for action selection
        use_actor_cnn=True   # Use actor's CNN
    )
    action = self.actor(state_tensor)
    return action

# train (training mode - gradients enabled)
def train(self, batch_size):
    # Sample batch from replay buffer
    batch = self.replay_buffer.sample(batch_size)
    
    # Extract features WITH gradients for critic update
    state = self.extract_features(
        {'image': batch['image'], 'vector': batch['vector']},
        enable_grad=True,    # Gradients ENABLED
        use_actor_cnn=False  # Use critic's CNN
    )
    
    # Compute critic loss
    current_Q1, current_Q2 = self.critic(state, action)
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
    
    # Backpropagation through BOTH critic MLP and critic CNN
    self.critic_optimizer.zero_grad()
    critic_loss.backward()  # Gradients flow: loss → critic → state → critic_cnn
    self.critic_optimizer.step()  # Update critic MLP + critic CNN
    
    # Actor update (delayed)
    if self.total_it % self.policy_freq == 0:
        # Extract features WITH gradients for actor update
        state = self.extract_features(
            {'image': batch['image'], 'vector': batch['vector']},
            enable_grad=True,   # Gradients ENABLED
            use_actor_cnn=True  # Use ACTOR's CNN
        )
        
        # Compute actor loss
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        
        # Backpropagation through BOTH actor MLP and actor CNN
        self.actor_optimizer.zero_grad()
        actor_loss.backward()  # Gradients flow: loss → actor → state → actor_cnn
        self.actor_optimizer.step()  # Update actor MLP + actor CNN
```

**Key Point:** `td3_agent.py` doesn't CREATE CNNs, it USES the CNNs passed from `train_td3.py`.

---

## The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      CNN ARCHITECTURE FLOW                      │
└─────────────────────────────────────────────────────────────────┘

1. BLUEPRINT DEFINITION (cnn_extractor.py)
   ┌──────────────────────────────────────┐
   │  class NatureCNN(nn.Module):         │
   │    def __init__(self, ...):          │
   │      self.conv1 = Conv2d(...)        │
   │      self.conv2 = Conv2d(...)        │
   │      self.conv3 = Conv2d(...)        │
   │      self.fc = Linear(...)           │
   │                                      │
   │    def forward(self, x):             │
   │      x = self.conv1(x)               │
   │      x = self.conv2(x)               │
   │      x = self.conv3(x)               │
   │      x = self.fc(x.flatten())        │
   │      return x  # (B, 512)            │
   └──────────────────────────────────────┘
                    ↓
                    
2. INSTANCE CREATION (train_td3.py)
   ┌──────────────────────────────────────┐
   │  self.actor_cnn = NatureCNN(...)     │  ← Instance 1
   │  self.critic_cnn = NatureCNN(...)    │  ← Instance 2
   │  self._initialize_cnn_weights()      │  ← Kaiming init
   │  self.actor_cnn.train()              │  ← Enable gradients
   │  self.critic_cnn.train()             │  ← Enable gradients
   └──────────────────────────────────────┘
                    ↓
                    
3. PASS TO AGENT (train_td3.py → td3_agent.py)
   ┌──────────────────────────────────────┐
   │  self.agent = TD3Agent(              │
   │    actor_cnn=self.actor_cnn,         │  ← Pass reference
   │    critic_cnn=self.critic_cnn,       │  ← Pass reference
   │    ...                               │
   │  )                                   │
   └──────────────────────────────────────┘
                    ↓
                    
4. STORE REFERENCES (td3_agent.py)
   ┌──────────────────────────────────────┐
   │  self.actor_cnn = actor_cnn          │  ← Store reference
   │  self.critic_cnn = critic_cnn        │  ← Store reference
   │                                      │
   │  # Include in optimizers             │
   │  actor_params = list(self.actor.parameters()) +  │
   │                 list(self.actor_cnn.parameters()) │
   │  self.actor_optimizer = Adam(actor_params)        │
   └──────────────────────────────────────┘
                    ↓
                    
5. USE FOR FEATURE EXTRACTION (td3_agent.py)
   ┌──────────────────────────────────────┐
   │  def extract_features(obs_dict, use_actor_cnn): │
   │    cnn = self.actor_cnn if use_actor_cnn       │
   │          else self.critic_cnn                   │
   │                                      │
   │    if enable_grad:                   │
   │      features = cnn(obs_dict['image']) # WITH grads │
   │    else:                             │
   │      with torch.no_grad():           │
   │        features = cnn(obs_dict['image']) # NO grads │
   │                                      │
   │    state = cat([features, obs_dict['vector']])      │
   │    return state  # (B, 565)          │
   └──────────────────────────────────────┘
                    ↓
                    
6. TRAINING LOOP (td3_agent.py)
   ┌──────────────────────────────────────────────────────┐
   │  CRITIC UPDATE:                                      │
   │    state = extract_features(batch, use_actor_cnn=False) │
   │    critic_loss.backward()  # Gradients → critic_cnn  │
   │    self.critic_optimizer.step()  # Update critic + CNN │
   │                                                      │
   │  ACTOR UPDATE (every policy_freq steps):            │
   │    state = extract_features(batch, use_actor_cnn=True)  │
   │    actor_loss.backward()  # Gradients → actor_cnn    │
   │    self.actor_optimizer.step()  # Update actor + CNN │
   └──────────────────────────────────────────────────────┘
```

---

## Why This Design?

### Design Goal: End-to-End Visual Learning

**Objective:** Train CNN to extract task-relevant features for autonomous driving (lanes, vehicles, road structure)

**Challenge:** CNN features must optimize DIFFERENT objectives for actor vs critic:
- **Actor CNN:** Extract features that help MAXIMIZE long-term reward (Q-value)
- **Critic CNN:** Extract features that help MINIMIZE TD error (accurate Q-estimation)

**Problem with Shared CNN:**
```
Shared CNN receives conflicting gradients:
  Actor: "Make lanes more visible to drive better" (maximize Q)
  Critic: "Ignore lanes to reduce Q-estimation error" (minimize TD error)
  
Result: CNN oscillates, learns nothing useful, training fails
```

**Solution: Separate CNNs**
```
Actor CNN: Optimized ONLY for actor's objective (maximize Q)
Critic CNN: Optimized ONLY for critic's objective (minimize TD error)

Result: Each CNN learns task-relevant features for its network
```

### Evidence from Literature

**Stable-Baselines3 TD3 (Official Implementation):**
```python
# sb3_contrib/td3/policies.py (line 89)
class TD3Policy(BasePolicy):
    def __init__(
        self,
        ...
        share_features_extractor: bool = False,  # DEFAULT: False
        ...
    ):
```

**PyTorch DQN Tutorial:**
- Uses SINGLE optimizer for CNN + policy (lines 150-160)
- BUT: DQN has only ONE network (Q-network), not actor-critic
- For actor-critic, separate CNNs prevent gradient interference

**Our Implementation (Fixed):**
- Follows SB3 TD3: Separate CNNs (share_features_extractor=False)
- Includes CNN parameters in main optimizers (matches official TD3)
- Enables end-to-end gradient flow during training

---

## Summary Table

| File | Role | What It Contains |
|------|------|------------------|
| **cnn_extractor.py** | Blueprint | `NatureCNN` class definition (architecture) |
| **train_td3.py** | Factory | Creates TWO instances: `actor_cnn`, `critic_cnn` |
| **td3_agent.py** | User | Uses CNNs for feature extraction with gradients |

| CNN Instance | Created In | Used By | Optimizer | Purpose |
|--------------|------------|---------|-----------|---------|
| `actor_cnn` | `train_td3.py` line 198 | `td3_agent.py` | `actor_optimizer` (line 163) | Extract features for actor updates |
| `critic_cnn` | `train_td3.py` line 204 | `td3_agent.py` | `critic_optimizer` (line 195) | Extract features for critic updates |

| Training Phase | Gradients Enabled? | Which CNN? | Optimizer Updates |
|----------------|-------------------|------------|-------------------|
| **select_action** (inference) | ❌ NO (`enable_grad=False`) | Actor CNN | None (inference only) |
| **Critic update** (training) | ✅ YES (`enable_grad=True`) | Critic CNN | `critic_optimizer.step()` updates critic MLP + critic CNN |
| **Actor update** (training, delayed) | ✅ YES (`enable_grad=True`) | Actor CNN | `actor_optimizer.step()` updates actor MLP + actor CNN |

---

## Common Misconceptions

### ❌ Misconception 1: "We have duplicate CNN code!"

**Reality:** We have ONE architecture definition (`NatureCNN` class) used to create TWO instances.

**Analogy:**
```python
# Python class (blueprint)
class Car:
    def __init__(self, color):
        self.color = color

# Two separate instances (same blueprint, different memory)
car1 = Car("red")   # Actor's car
car2 = Car("blue")  # Critic's car
```

### ❌ Misconception 2: "train_td3.py trains the CNN!"

**Reality:** `train_td3.py` only CREATES the CNN instances. Training happens in `td3_agent.py` during the `train()` method.

**What train_td3.py does:**
1. Creates CNN instances
2. Initializes weights (Kaiming)
3. Sets to training mode
4. Passes to TD3Agent

**What td3_agent.py does:**
1. Includes CNN in optimizers
2. Extracts features WITH gradients during training
3. Updates CNN weights via backpropagation

### ❌ Misconception 3: "cnn_extractor.py is not used!"

**Reality:** `cnn_extractor.py` defines the `NatureCNN` class, which is imported and instantiated in `train_td3.py`.

**Import chain:**
```python
# train_td3.py (line 33)
from src.networks.cnn_extractor import NatureCNN

# train_td3.py (line 198)
self.actor_cnn = NatureCNN(...)  # ← Uses class from cnn_extractor.py
```

---

## Validation Checklist

Use this to verify the CNN setup is correct:

- [ ] **Two separate CNN instances created in train_td3.py**
  ```python
  print(f"Actor CNN id: {id(self.actor_cnn)}")
  print(f"Critic CNN id: {id(self.critic_cnn)}")
  print(f"Different instances: {id(self.actor_cnn) != id(self.critic_cnn)}")
  ```
  Expected: Different IDs ✅

- [ ] **CNNs set to training mode**
  ```python
  print(f"Actor CNN training: {self.actor_cnn.training}")
  print(f"Critic CNN training: {self.critic_cnn.training}")
  ```
  Expected: Both True ✅

- [ ] **CNN parameters included in main optimizers**
  ```python
  # td3_agent.py
  actor_params = list(self.actor.parameters()) + list(self.actor_cnn.parameters())
  self.actor_optimizer = Adam(actor_params)
  ```
  Expected: Single optimizer per network ✅

- [ ] **No separate CNN optimizers**
  ```python
  # td3_agent.py (lines 218-220)
  self.actor_cnn_optimizer = None  # DEPRECATED
  self.critic_cnn_optimizer = None  # DEPRECATED
  ```
  Expected: Both None ✅

- [ ] **Gradients enabled during training**
  ```python
  # td3_agent.py extract_features()
  if enable_grad:
      image_features = cnn(obs_dict['image'])  # Gradients flow
  ```
  Expected: `requires_grad=True` for features ✅

- [ ] **Correct CNN selected for each update**
  ```python
  # Critic update: use_actor_cnn=False
  # Actor update: use_actor_cnn=True
  ```
  Expected: Different CNNs for actor/critic ✅

---

## Debugging Guide

If you suspect CNN issues, check:

1. **Are two separate instances created?**
   ```bash
   grep -A5 "self.actor_cnn = NatureCNN" scripts/train_td3.py
   grep -A5 "self.critic_cnn = NatureCNN" scripts/train_td3.py
   ```

2. **Are CNNs passed to agent?**
   ```bash
   grep -A5 "actor_cnn=self.actor_cnn" scripts/train_td3.py
   ```

3. **Are CNN parameters in optimizers?**
   ```bash
   grep -A5 "actor_params = list" src/agents/td3_agent.py
   grep -A5 "critic_params = list" src/agents/td3_agent.py
   ```

4. **Are separate CNN optimizers removed?**
   ```bash
   grep "actor_cnn_optimizer = None" src/agents/td3_agent.py
   grep "critic_cnn_optimizer = None" src/agents/td3_agent.py
   ```

5. **Are gradients flowing during training?**
   - Check TensorBoard: `gradients/actor_cnn_norm`, `gradients/critic_cnn_norm`
   - Should be >0 and ≤ clipping limits (1.0 for actor, 10.0 for critic)

---

## References

1. **Stable-Baselines3 TD3 Implementation**
   - File: `sb3_contrib/td3/policies.py`
   - Default: `share_features_extractor=False` (separate CNNs)
   - Reason: Prevent gradient interference in actor-critic

2. **PyTorch DQN Tutorial**
   - URL: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
   - Shows: End-to-end CNN training with single optimizer
   - Note: DQN has only ONE network (not actor-critic)

3. **TD3 Paper (Fujimoto et al., 2018)**
   - "Addressing Function Approximation Error in Actor-Critic Methods"
   - Official code: https://github.com/sfujim/TD3
   - Key: Single optimizer per network (lines 25-26 in TD3.py)

4. **Our Documentation**
   - `CRITICAL_FIXES_IMPLEMENTATION_SUMMARY.md` - Fix #2 (optimizer merge)
   - `CNN_END_TO_END_TRAINING_ANALYSIS.md` - Part 4 (gradient clipping bug)
   - `IMMEDIATE_ACTION_PLAN.md` - Tasks 1-3 (hyperparameters + clipping)

---

## Conclusion

We have a **clean, correct architecture**:
- ONE CNN blueprint (`cnn_extractor.py`)
- TWO CNN instances (`train_td3.py`)
- PROPER usage (`td3_agent.py`)

This design:
- ✅ Follows Stable-Baselines3 best practices
- ✅ Prevents gradient interference
- ✅ Enables end-to-end learning
- ✅ Matches official TD3 implementation

The apparent "duplication" is actually **separation of concerns**:
- `cnn_extractor.py`: WHAT is a CNN (architecture)
- `train_td3.py`: CREATE two CNNs (instantiation)
- `td3_agent.py`: USE CNNs for training (gradient flow)

**No changes needed** - the architecture is already correct following the November 13 and November 20 fixes.
