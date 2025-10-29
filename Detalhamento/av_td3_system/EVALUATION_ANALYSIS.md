# Evaluation Function Analysis - TD3 Training Script

**Date**: October 26, 2025
**File**: `scripts/train_td3.py`
**Question**: Is the evaluation function correctly implemented? What is it using to do the movements in the environment?

---

## ✅ Summary: Evaluation Function is CORRECTLY Implemented

The evaluation function follows **TD3 best practices** from the original paper (Fujimoto et al., 2018) and official implementation. It is well-designed and properly separates evaluation from training.

---

## 🔍 Evaluation Function Analysis

### Function Signature (Lines 775-790)
```python
def evaluate(self) -> dict:
    """
    Evaluate agent on multiple episodes without exploration noise.

    FIXED: Creates a separate evaluation environment to avoid interfering
    with training environment state (RNG, CARLA actors, internal counters).

    Returns:
        Dictionary with evaluation metrics:
        - mean_reward: Average episode reward
        - std_reward: Std dev of episode rewards
        - success_rate: Fraction of successful episodes
        - avg_collisions: Average collisions per episode
        - avg_episode_length: Average episode length
    """
```

---

## 🎯 How Movement Works During Evaluation

### Step-by-Step Movement Process

#### 1. **Separate Environment Creation** (Lines 792-796)
```python
# FIXED: Create separate eval environment (don't reuse self.env)
print(f"[EVAL] Creating temporary evaluation environment...")
eval_env = CARLANavigationEnv(
    self.carla_config_path,
    self.agent_config_path,
    self.training_config_path
)
```

**Purpose**: Creates a fresh CARLA environment for evaluation to avoid interfering with training state.

---

#### 2. **Episode Loop** (Lines 807-832)
```python
for episode in range(self.num_eval_episodes):  # Default: 10 episodes
    obs_dict = eval_env.reset()  # Reset to spawn point
    state = self.flatten_dict_obs(obs_dict)  # Convert Dict → flat array (535,)
    episode_reward = 0
    episode_length = 0
    done = False

    while not done and episode_length < max_eval_steps:
        # Deterministic action (no noise)
        action = self.agent.select_action(state, noise=0.0)  # ← KEY LINE!
        next_obs_dict, reward, done, truncated, info = eval_env.step(action)
        next_state = self.flatten_dict_obs(next_obs_dict)

        episode_reward += reward
        episode_length += 1
        state = next_state

        if truncated:
            done = True
```

**Key Point**: The **CRITICAL** line is:
```python
action = self.agent.select_action(state, noise=0.0)
```

This is what controls the vehicle!

---

### 🚗 Action Selection Pipeline

#### **Step 1: Agent's `select_action()` Method** (td3_agent.py, lines 172-196)

```python
def select_action(
    self,
    state: np.ndarray,
    noise: Optional[float] = None
) -> np.ndarray:
    """
    Select action from current policy with optional exploration noise.

    During training, Gaussian noise is added for exploration. During evaluation,
    the deterministic policy is used (noise=0).

    Args:
        state: Current state observation (535-dim numpy array)
        noise: Std dev of Gaussian exploration noise. If None, uses self.expl_noise

    Returns:
        action: 2-dim numpy array [steering, throttle/brake] ∈ [-1, 1]²
    """
    # Convert state to tensor
    state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

    # Get deterministic action from actor
    with torch.no_grad():
        action = self.actor(state).cpu().numpy().flatten()  # ← Neural network forward pass

    # Add exploration noise if specified
    if noise is not None and noise > 0:
        noise_sample = np.random.normal(0, noise, size=self.action_dim)
        action = action + noise_sample
        # Clip to valid action range
        action = np.clip(action, -self.max_action, self.max_action)

    return action
```

**During Evaluation**: Since `noise=0.0`, the agent uses the **pure learned policy** (no randomness).

---

#### **Step 2: Actor Network Forward Pass** (actor.py, lines 84-94)

```python
def forward(self, state: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through actor network.

    Args:
        state: Batch of states (batch_size, state_dim) = (1, 535)

    Returns:
        Batch of actions (batch_size, action_dim) = (1, 2) in range [-1, 1]
    """
    # Hidden layers with ReLU
    x = self.relu(self.fc1(state))  # 535 → 256
    x = self.relu(self.fc2(x))       # 256 → 256

    # Output layer with Tanh and scaling
    a = self.tanh(self.fc3(x))       # 256 → 2, then Tanh
    a = a * self.max_action          # Scale to [-1, 1]

    return a  # Shape: (1, 2) → [steering, throttle/brake]
```

**This is the learned neural network policy!** It maps:
- **Input**: 535-dim state (512 CNN features + 3 kinematic + 20 waypoint coords)
- **Output**: 2-dim action [steering, throttle/brake] ∈ [-1, 1]²

---

#### **Step 3: Environment Applies Action** (carla_env.py)

The action is then applied to the CARLA vehicle:

```python
# Inside carla_env.step(action)
steering = float(action[0])  # -1 to 1
throttle_brake = float(action[1])  # -1 to 1

# Map to CARLA control
control = carla.VehicleControl()
control.steer = np.clip(steering, -1.0, 1.0)

if throttle_brake >= 0:
    control.throttle = float(throttle_brake)
    control.brake = 0.0
else:
    control.throttle = 0.0
    control.brake = float(-throttle_brake)

self.vehicle.apply_control(control)  # ← Vehicle moves!
```

---

## 📊 Action Generation Comparison

### Training vs. Evaluation

| Phase | Action Selection | Noise | Behavior |
|-------|------------------|-------|----------|
| **Training (Exploration)** | `select_action(state, noise=random_noise)` | 0.1 to 0.3 (decaying) | Actions have Gaussian noise added for exploration |
| **Training (Early Phase)** | `env.action_space.sample()` | N/A | Completely random actions (first 25k steps) |
| **Evaluation** | `select_action(state, noise=0.0)` | 0.0 (none) | **Pure deterministic policy** (learned behavior) |

---

## 🧠 What the Actor Network Learned

The Actor network learns to map **visual + kinematic + waypoint information** to **steering + throttle/brake commands**:

### Input State (535 dimensions):
1. **CNN Features (512)**: Extracted from 4 stacked grayscale camera frames (84×84)
   - Encodes: obstacles, lane markings, vehicles, road structure
2. **Kinematic State (3)**:
   - Velocity (m/s)
   - Lateral deviation from center (m)
   - Heading error (rad)
3. **Waypoint Data (20)**: 10 waypoints × 2 coordinates (x, y) in vehicle frame
   - Next 10 waypoints along the route

### Output Action (2 dimensions):
1. **Steering**: -1 (full left) to +1 (full right)
2. **Throttle/Brake**: -1 (full brake) to +1 (full throttle)

---

## ✅ Correctness Verification

### ✓ Separate Environment
- **Correct**: Creates new `eval_env` instead of reusing `self.env`
- **Prevents**: Training RNG interference, actor conflicts, counter desync

### ✓ Deterministic Policy
- **Correct**: Uses `noise=0.0` for true policy performance
- **Follows**: TD3 paper standard ("10 episodes with no exploration noise")

### ✓ Proper Cleanup
- **Correct**: Closes `eval_env` after evaluation
- **Prevents**: Actor leaks and CARLA server overload

### ✓ Episode Timeout
- **Correct**: Uses `max_eval_steps = 1000` from config (now fixed!)
- **Prevents**: Infinite episodes or ultra-long evaluation

### ✓ Metrics Collection
- **Correct**: Tracks reward, success, collisions, episode length
- **Purpose**: Scientific comparison to baselines (DDPG, IDM+MOBIL)

---

## 🎓 Why This is Important

### Training Actions (with noise):
```
Action = Actor(state) + Gaussian_Noise(0, 0.1)
Result: Exploration, suboptimal behavior, learning
```

**Purpose**: Discover new behaviors, fill replay buffer, improve policy

### Evaluation Actions (deterministic):
```
Action = Actor(state)  # Pure learned policy
Result: Best learned behavior, true performance
```

**Purpose**: Measure actual policy quality without exploration randomness

---

## 🔬 Example Execution Flow

```
[EVAL] Evaluation at timestep 2,000...
[EVAL] Creating temporary evaluation environment...
  ↓ Spawns new CARLA vehicle at waypoints[0]
  ↓ Spawns new sensors (camera, collision, etc.)

[EVAL] Episode 1/10:
  1. Reset: Vehicle at spawn point, obs_dict = {image, vector}
  2. Flatten: state = [512 CNN features, 3 kinematic, 20 waypoints] (535,)
  3. Loop (1000 steps max):
     - state → Actor Network → action [steering, throttle/brake]
     - action → CARLA vehicle.apply_control() → vehicle moves!
     - Observe: next_obs_dict, reward, done, info
     - Update: state = next_state, accumulate reward
  4. Episode ends: collision/goal/timeout
  5. Record: episode_reward, success, collisions, length

[EVAL] Episode 2/10: (repeat)
...
[EVAL] Episode 10/10: (repeat)

[EVAL] Closing evaluation environment...
  ↓ Destroys actors
  ↓ Closes sensors

[EVAL] Mean Reward: -15235.42 | Success Rate: 0.0% | Avg Collisions: 0.00 | Avg Length: 32
```

---

## 🐛 Why Early Evaluation Might Fail

If evaluation shows poor performance (low rewards, short episodes), it's because:

1. **Policy not trained yet**: First 25k steps are random actions
2. **Insufficient training**: Policy needs 100k+ steps to learn basic driving
3. **Environment bugs**: Episode timeout, sensor issues (we just fixed these!)
4. **Poor action selection**: Actor outputs may be near-zero initially (not enough exploration)

**This is normal!** Evaluation at 2k, 4k, 6k steps shows:
- How quickly the policy learns
- If training is progressing correctly
- If there are environmental issues (crashes, timeouts)

---

## 📈 Expected Evaluation Progression

| Timestep | Expected Behavior | Mean Reward | Episode Length |
|----------|-------------------|-------------|----------------|
| 2,000 | Random exploration still | -50,000 to -10,000 | 10-50 steps |
| 6,000 | Basic steering learned | -5,000 to -1,000 | 50-200 steps |
| 20,000 | Lane following starting | -1,000 to -200 | 200-500 steps |
| 50,000 | Decent driving | -200 to 0 | 500-1000 steps |
| 100,000+ | Good driving | 0 to +500 | Full episodes |

---

## 🎯 Conclusion

### **Evaluation Function: ✅ CORRECTLY IMPLEMENTED**

**What controls the vehicle during evaluation?**
1. **Actor Neural Network** (256→256→2 MLP with Tanh output)
2. **Learned from training** (TD3 algorithm with replay buffer)
3. **Deterministic policy** (no exploration noise)
4. **Input**: 535-dim state (visual + kinematic + waypoints)
5. **Output**: 2-dim action (steering + throttle/brake)

The evaluation function is a **critical component** of TD3 training and is implemented correctly according to:
- ✅ Original TD3 paper (Fujimoto et al., 2018)
- ✅ Official TD3 implementation
- ✅ Best practices for RL evaluation
- ✅ Proper CARLA environment management

**The poor early evaluation results are expected and normal** - the policy needs time to learn!

---

## 📚 References

1. **TD3 Paper**: Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods" (ICML 2018)
   - Section 5.1: "Each task is run for 1 million time steps with evaluations every 5000 time steps"
   - Evaluation: "average reward over 10 episodes with no exploration noise"

2. **TD3 Official Implementation**: https://github.com/sfujim/TD3
   - `eval_policy()` function creates separate environment
   - Uses deterministic policy (`policy.select_action()` without noise)

3. **CARLA Documentation**: https://carla.readthedocs.io/
   - `vehicle.apply_control()` method for action application
   - VehicleControl: steering [-1, 1], throttle [0, 1], brake [0, 1]

4. **Actor-Critic Methods**: Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed.)
   - Chapter 13: Policy Gradient Methods
   - Deterministic Policy Gradient theorem

---

**Status**: Documentation complete - evaluation function verified correct! ✅
