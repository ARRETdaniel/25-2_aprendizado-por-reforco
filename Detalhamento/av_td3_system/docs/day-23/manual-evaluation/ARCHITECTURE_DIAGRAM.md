# Reward Validation System Architecture

## Overview Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CARLA Simulator                              │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Ego Vehicle + Sensors                                        │  │
│  │  - Front Camera (RGB)                                         │  │
│  │  - Collision Sensor                                           │  │
│  │  - Lane Invasion Sensor                                       │  │
│  │  - IMU/Odometry                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ Sensor Data
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              CarlaGymEnv (src/environment/carla_env.py)             │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  step(action) Method                                       │   │
│  │  ┌──────────────────────────────────────────────────────┐ │   │
│  │  │ 1. Apply action to vehicle                          │ │   │
│  │  │ 2. Tick simulation                                  │ │   │
│  │  │ 3. Get sensor data                                  │ │   │
│  │  │ 4. Calculate vehicle_state                          │ │   │
│  │  │ 5. Call reward_calculator.calculate(...)     ◄──────┼─┼───┐
│  │  │ 6. Build observation                                │ │   │
│  │  │ 7. Build info dict ◄─── ENHANCED IN THIS CHANGE     │ │   │
│  │  │ 8. Return (obs, reward, term, trunc, info)          │ │   │
│  │  └──────────────────────────────────────────────────────┘ │   │
│  └────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                 │                                     │
                                 │                         ┌───────────┘
                                 │                         │
                                 ▼                         ▼
┌────────────────────────────────────────┐  ┌──────────────────────────┐
│  RewardCalculator                      │  │ Existing reward_breakdown│
│  (src/environment/reward_functions.py) │  │ Format (tuple):          │
│                                        │  │                          │
│  calculate(...) returns:               │  │ {                        │
│  {                                     │  │   "efficiency": (        │
│    "total": -0.0845,                   │  │     weight,       [0]    │
│    "breakdown": {                      │  │     raw_value,    [1]    │
│      "efficiency": (w, raw, weighted), │  │     weighted_val  [2]    │
│      "lane_keeping": (...),            │  │   ),                     │
│      "comfort": (...),                 │  │   "lane_keeping": (...), │
│      "safety": (...),                  │  │   ...                    │
│      "progress": (...)                 │  │ }                        │
│    }                                   │  │                          │
│  }                                     │  └──────────────────────────┘
└────────────────────────────────────────┘
                  │
                  │ reward_dict
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ENHANCED info Dict Construction (Line ~787 in carla_env.py)        │
│  ═══════════════════════════════════════════════════════════════    │
│                                                                      │
│  info = {                                                            │
│      "step": self.current_step,                                      │
│                                                                      │
│      # ✅ PRESERVED: Existing format (backward compatible)           │
│      "reward_breakdown": reward_dict["breakdown"],                   │
│                                                                      │
│      # 🆕 NEW: Validation-friendly flat format                       │
│      "reward_components": {                                          │
│          "total": reward,                                            │
│          "efficiency": reward_dict["breakdown"]["efficiency"][2],    │
│          "lane_keeping": ...[2],                                     │
│          "comfort": ...[2],                                          │
│          "safety": ...[2],                                           │
│          "progress": ...[2],                                         │
│      },                                                              │
│                                                                      │
│      # 🆕 NEW: State metrics for HUD display                         │
│      "state": {                                                      │
│          "velocity": vehicle_state["velocity"],                      │
│          "lateral_deviation": vehicle_state["lateral_deviation"],    │
│          "heading_error": vehicle_state["heading_error"],            │
│          "distance_to_goal": distance_to_goal,                       │
│      },                                                              │
│                                                                      │
│      # Rest of existing fields...                                   │
│      "termination_reason": termination_reason,                       │
│      "vehicle_state": vehicle_state,                                 │
│      "collision_info": collision_info,                               │
│      # ...                                                           │
│  }                                                                   │
└─────────────────────────────────────────────────────────────────────┘
                  │
                  │ Returned in step() as 5th element
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  obs, reward, terminated, truncated, info = env.step(action)        │
└─────────────────────────────────────────────────────────────────────┘
                  │
         ┌────────┴─────────┬─────────────────┬────────────────┐
         │                  │                 │                │
         ▼                  ▼                 ▼                ▼
┌─────────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Manual          │ │ Analysis     │ │ Training     │ │ Paper        │
│ Validation      │ │ Script       │ │ Monitoring   │ │ Generation   │
│                 │ │              │ │              │ │              │
│ validate_       │ │ analyze_     │ │ TD3/DDPG     │ │ Figures &    │
│ rewards_        │ │ reward_      │ │ Agents       │ │ Tables       │
│ manual.py       │ │ validation.py│ │              │ │              │
│                 │ │              │ │              │ │              │
│ Uses:           │ │ Uses:        │ │ Uses:        │ │ Uses:        │
│ - reward_       │ │ - reward_    │ │ - total      │ │ - reward_    │
│   components    │ │   components │ │   reward     │ │   components │
│ - state         │ │ - state      │ │   (scalar)   │ │   stats      │
│                 │ │              │ │              │ │              │
│ Displays:       │ │ Generates:   │ │ Logs:        │ │ Creates:     │
│ ┌─────────────┐ │ │ - Plots      │ │ - TensorBoard│ │ - Fig 4: Evo │
│ │  HUD:       │ │ │ - Report MD  │ │ - Checkpoints│ │ - Table II   │
│ │  Efficiency │ │ │ - CSV stats  │ │              │ │ - Supp Mat   │
│ │  LaneKeep   │ │ │              │ │              │ │              │
│ │  Safety     │ │ │              │ │              │ │              │
│ │  ...        │ │ │              │ │              │ │              │
│ └─────────────┘ │ │              │ │              │ │              │
└─────────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

## Data Flow Detail: info Dict

### What Gets Logged Each Step

```
Step 1247:
┌────────────────────────────────────────────────────────────┐
│ info = {                                                   │
│   "step": 1247,                                            │
│   "reward_breakdown": {  ← Existing format (tuple)         │
│     "efficiency": (0.5, 0.049, 0.0245),                    │
│     "lane_keeping": (0.3, -0.004, -0.0012),                │
│     "comfort": (0.1, -0.078, -0.0078),                     │
│     "safety": (0.05, 0.0, 0.0),                            │
│     "progress": (0.05, 0.2, 0.01)                          │
│   },                                                       │
│   "reward_components": {  ← NEW: Validation format         │
│     "total": -0.0845,                                      │
│     "efficiency": 0.0245,     ← Extracted [2] from tuple  │
│     "lane_keeping": -0.0012,  ← Weighted contribution     │
│     "comfort": -0.0078,                                    │
│     "safety": 0.0,                                         │
│     "progress": 0.01                                       │
│   },                                                       │
│   "state": {  ← NEW: Metrics for HUD                       │
│     "velocity": 28.5,          ← km/h                      │
│     "lateral_deviation": 0.15, ← meters from center       │
│     "heading_error": 0.02,     ← radians                   │
│     "distance_to_goal": 450.3  ← meters                    │
│   },                                                       │
│   "termination_reason": None,                              │
│   "vehicle_state": {...},  ← Full verbose state           │
│   "collision_info": None,                                  │
│   ...                                                      │
│ }                                                          │
└────────────────────────────────────────────────────────────┘
```

### Component Summation Validation

```
Validation Check:
┌──────────────────────────────────────────────────────────┐
│ calculated_total = (                                     │
│     efficiency    =  0.0245                              │
│   + lane_keeping  = -0.0012                              │
│   + comfort       = -0.0078                              │
│   + safety        =  0.0000                              │
│   + progress      =  0.0100                              │
│ )                                                        │
│ = 0.0255                                                 │
│                                                          │
│ Wait... this doesn't match total = -0.0845!             │
│                                                          │
│ ERROR: Summation residual = 0.11                         │
│ CRITICAL ISSUE DETECTED ← Validation catches this! 🐛    │
└──────────────────────────────────────────────────────────┘
```

**This is WHY we need validation!** Example shows hypothetical bug.

## Comparison: Before vs After

### Before Enhancement (Limited)

```python
# In validate_rewards_manual.py (hypothetical old version)

obs, reward, term, trunc, info = env.step(action)

# ❌ Only have total reward
total_reward = reward  # Scalar: -0.0845

# ❌ Can't decompose
efficiency = ???       # Not available
lane_keeping = ???     # Not available

# ❌ Would need to parse complex tuple
breakdown = info['reward_breakdown']  # {comp: (w, raw, weighted)}
efficiency = breakdown['efficiency'][2]  # Fragile, needs to know [2]

# ❌ HUD display limited
print(f"Total Reward: {reward}")  # Not very informative
```

### After Enhancement (Complete)

```python
# In validate_rewards_manual.py (current version)

obs, reward, term, trunc, info = env.step(action)

# ✅ Simple flat dict access
reward_components = info['reward_components']
state_metrics = info['state']

# ✅ Clean extraction
total_reward = reward_components['total']
efficiency = reward_components['efficiency']
lane_keeping = reward_components['lane_keeping']
# ...

# ✅ State metrics for context
velocity = state_metrics['velocity']
lateral_dev = state_metrics['lateral_deviation']

# ✅ Rich HUD display
display_hud(reward_components, state_metrics)
# Shows:
#   Total: -0.0845
#   Efficiency: +0.0245 (speed: 28.5 km/h)
#   Lane Keeping: -0.0012 (deviation: 0.15 m)
#   ...

# ✅ Validation check
calculated = sum([
    efficiency, lane_keeping, comfort, safety, progress
])
assert abs(calculated - total_reward) < 0.001, "BUG DETECTED!"

# ✅ Correlation analysis
correlation = pearson(lateral_dev_history, lane_keeping_history)
# Should be strongly negative (r < -0.7)
```

## Use Cases Enabled

### Use Case 1: Manual Validation Session

```
User Action: Drive vehicle in CARLA using WSAD keys
System Response:
┌──────────────────────────────────────────────────────────┐
│ Real-Time HUD (PyGame Window)                            │
│ ══════════════════════════════════════════════════════   │
│ Step 1247                                                │
│ Total Reward: -0.0845                                    │
│ ────────────────────────────────────────────────────────│
│ Components:                                              │
│   ⚡ Efficiency:    +0.0245  ████░░░░░░                   │
│   🛣️  Lane Keeping:  -0.0012  █░░░░░░░░░                │
│   💺 Comfort:       -0.0078  ███░░░░░░░                  │
│   🚨 Safety:        +0.0000  ░░░░░░░░░░                  │
│   📍 Progress:      +0.0100  ████░░░░░░                   │
│ ────────────────────────────────────────────────────────│
│ State:                                                   │
│   Speed: 28.5 km/h (target: 30)                          │
│   Lateral Dev: 0.15 m (< 0.5 OK)                         │
│   Heading Error: 0.02 rad (aligned)                      │
│ ────────────────────────────────────────────────────────│
│ Controls: W/S=accel/brake | A/D=steer | Q=quit          │
└──────────────────────────────────────────────────────────┘

Output File: validation_logs/session_01/reward_validation_*.json
```

### Use Case 2: Statistical Analysis

```
Input: validation_logs/session_01/reward_validation_*.json

Process:
1. Load 1,247 logged snapshots
2. Extract time series:
   - lateral_deviation = [0.1, 0.15, 0.2, ...]
   - lane_keeping_reward = [-0.001, -0.0012, -0.002, ...]
3. Calculate correlation:
   r = pearson(lateral_deviation, lane_keeping_reward)
   r = -0.85  ← Strong negative (expected!)
4. Generate plot:
   - X-axis: lateral deviation
   - Y-axis: lane keeping reward
   - Should show downward trend

Output:
  - validation_report_*.md (with findings)
  - lateral_deviation_correlation.png
  - correlation_heatmap.png
```

### Use Case 3: Paper Figure Generation

```
Input: Multiple validation sessions (TD3, DDPG, Classical)

Process:
1. Aggregate reward components across algorithms
2. Calculate statistics:
   - Mean efficiency reward: TD3 vs DDPG vs Classical
   - Std deviation of safety penalties
   - Median lane keeping performance
3. Generate comparison table:

   | Algorithm | Efficiency | Lane Keep | Safety Penalty |
   |-----------|------------|-----------|----------------|
   | TD3       | 0.85±0.12  | -0.02±0.01| -0.001±0.002   |
   | DDPG      | 0.72±0.18  | -0.05±0.03| -0.015±0.010   |
   | Classical | 0.68±0.15  | -0.08±0.04| -0.020±0.012   |

4. Insert into paper as Table II

Output: Paper-ready LaTeX table + supplementary raw data
```

## Summary: Why This Architecture Works

### ✅ Modular Design
- Reward calculation separate from environment (reward_functions.py)
- Dual format preserves backward compatibility
- Validation tools independent of training code

### ✅ Standard Compliance
- Follows Gymnasium API specification
- Uses official recommendation for `info` dict
- Compatible with existing RL tools

### ✅ Scientific Rigor
- Comprehensive logging for reproducibility
- Validation workflow catches bugs early
- Statistical analysis confirms assumptions

### ✅ Paper Ready
- Figures generated automatically
- Tables populated from logged data
- Supplementary materials include raw logs

---

**Next Step:** Run validation workflow (see NEXT_STEPS_REWARD_VALIDATION.md)
