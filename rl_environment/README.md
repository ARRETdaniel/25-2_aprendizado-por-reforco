# CARLA RL Environment Wrapper for CARLA 0.8.4

This module provides a reinforcement learning-compatible environment wrapper for the CARLA simulator (version 0.8.4, Coursera modified version), designed to facilitate the development of deep reinforcement learning algorithms for autonomous driving.

## Overview

The `CarlaEnvWrapper` class implements a gym-like interface for CARLA, allowing reinforcement learning algorithms to interact with the simulator through standardized methods (`reset()`, `step()`, `close()`). This implementation is specifically tailored for the CARLA 0.8.4 version used in the Coursera Self-Driving Cars specialization.

## Implementation Status

- [✓] Project structure
- [✓] Environment class skeleton
- [✓] State representation
- [✓] Action space definition
- [✓] Reward function
- [✓] Terminal conditions
- [✓] Integration points with existing perception

## Architecture

```ascii
                 ┌──────────────────────┐
                 │                      │
                 │    RL Environment    │
                 │                      │
                 └──────────┬───────────┘
                            │
                ┌───────────▼────────────┐
                │                        │
          ┌─────┴─────┐          ┌──────▼──────┐
          │           │          │              │
┌─────────▼───────┐   │   ┌──────▼─────────┐    │
│  State Processor│   │   │  Action Processor   │
└─────────┬───────┘   │   └──────┬─────────┘    │
          │           │          │              │
          └──────┐    │    ┌─────┘              │
                 │    │    │                    │
              ┌──▼────▼────▼─┐          ┌──────▼──────┐
              │              │          │              │
              │    CARLA     │          │   Reward     │
              │  Simulator   │          │  Function    │
              │              │          │              │
              └──────────────┘          └──────────────┘
```

The environment consists of the following components:

1. **CarlaEnvWrapper**: Main class implementing the gym-like interface
2. **StateProcessor**: Processes raw sensor data into structured observations
3. **ActionProcessor**: Converts RL actions to CARLA vehicle controls
4. **RewardFunction**: Calculates rewards based on various driving criteria

## Usage

```python
from rl_environment import CarlaEnvWrapper
from stable_baselines3 import SAC

# Create environment
env = CarlaEnvWrapper(
    host='localhost',
    port=2000,
    city_name='Town01',
    image_size=(84, 84),
    frame_skip=2,
    max_episode_steps=1000,
    weather_id=0,
    quality_level='Low',
    random_start=True
)

# Initialize agent
model = SAC('CnnPolicy', env, verbose=1)

# Train agent
model.learn(total_timesteps=1000000)

# Test agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
```

## Environment Configuration

The environment can be configured with the following parameters:

- `host` (str): CARLA server host
- `port` (int): CARLA server port
- `city_name` (str): Town to use for simulation (e.g., 'Town01', 'Town02')
- `image_size` (tuple): Size of camera images (height, width)
- `frame_skip` (int): Number of frames to skip between actions
- `max_episode_steps` (int): Maximum steps per episode
- `weather_id` (int): Weather preset ID
- `quality_level` (str): Graphics quality level ('Low' or 'Epic')
- `random_start` (bool): Whether to randomize starting position
- `reward_config` (dict): Configuration for reward components

## State Space

The observation space is a dictionary containing:

1. `image`: Processed camera images (RGB) - Shape: `image_size + (3,)`
2. `vehicle_state`: Vehicle state information - Shape: `(9,)`
   - Position (x, y, z)
   - Velocity (vx, vy, vz)
   - Orientation (roll, pitch, yaw)

3. `navigation`: Navigation information - Shape: `(3,)`
   - Distance to next waypoint
   - Angle to next waypoint
   - Curvature of the road ahead

4. `detections`: Detection information - Shape: `(10,)`
   - Simplified representation of nearby objects, traffic lights, etc.

## Action Space

The environment supports both continuous and discrete action spaces:

### Continuous Action Space

Shape: `(3,)` with the following dimensions:

1. Throttle [0, 1]
2. Brake [0, 1]
3. Steering [-1, 1]

### Discrete Action Space

When configured for discrete actions, the following 9 actions are available:

1. Idle (throttle=0, brake=0, steer=0)
2. Accelerate (throttle=0.5, brake=0, steer=0)
3. Full acceleration (throttle=1.0, brake=0, steer=0)
4. Brake (throttle=0, brake=0.5, steer=0)
5. Full brake (throttle=0, brake=1.0, steer=0)
6. Accelerate and steer left (throttle=0.5, brake=0, steer=-0.5)
7. Accelerate and steer right (throttle=0.5, brake=0, steer=0.5)
8. Steer left (throttle=0, brake=0, steer=-0.5)
9. Steer right (throttle=0, brake=0, steer=0.5)

## Reward Function

The reward function calculates a weighted sum of several components:

1. Progress toward destination (`progress_weight`: 1.0)
2. Lane-keeping accuracy (`lane_deviation_weight`: 0.5)
3. Collision penalty (`collision_penalty`: 100.0)
4. Speed alignment with target speed (`speed_weight`: 0.2)
5. Control smoothness (`action_smoothness_weight`: 0.1)

These weights can be customized via the `reward_config` parameter.

## Termination Conditions

Episodes terminate when any of the following conditions are met:

1. The vehicle reaches the destination
2. The vehicle collides with an obstacle
3. The vehicle leaves the drivable area
4. The maximum number of steps is reached (default: 1000)
5. The vehicle remains stationary for too long

## Development Plan

1. Create basic environment wrapper with `reset()`, `step()`, and `close()` methods
2. Implement state representation and action space
3. Design and test reward function
4. Add termination conditions
5. Integrate with existing perception system
6. Test with simple RL algorithm (e.g., SAC)
7. Optimize for performance and stability

## Requirements

- CARLA 0.8.4 (modified Coursera version)
- Python 3.6 for CARLA client
- NumPy
- OpenCV
- Gymnasium (or gym) for compatibility with RL algorithms
- (Optional) Stable-Baselines3 for RL algorithms implementation
