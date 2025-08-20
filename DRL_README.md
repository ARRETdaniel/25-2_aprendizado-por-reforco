# Deep Reinforcement Learning for Autonomous Vehicles in CARLA

![CARLA Simulator](https://carla.readthedocs.io/en/latest/_images/carla_head.png)

> **Note**: This project integrates Deep Reinforcement Learning techniques into an existing autonomous driving implementation in the CARLA simulator (modified v0.8.4).

## Project Status: Initial Development

This document outlines our current progress and next steps toward implementing a Deep Reinforcement Learning (DRL) solution for autonomous vehicle control in the CARLA simulator. The project builds upon an existing implementation with traditional perception, planning, and control components.

## Table of Contents

- [Overview](#overview)
- [Current Implementation](#current-implementation)
- [Project Goals](#project-goals)
- [Development Roadmap](#development-roadmap)
- [Required Dependencies](#required-dependencies)
- [Coding Standards](#coding-standards)
- [References](#references)

## Overview

Autonomous driving systems have traditionally relied on modular architectures with separate perception, planning, and control components. While effective, these systems can be challenging to design and tune. Deep Reinforcement Learning offers an alternative approach by learning end-to-end driving policies directly from experience.

This project aims to enhance our existing autonomous driving implementation with DRL capabilities, leveraging the latest research in the field (2021-2025). We'll maintain the current perception system while replacing or augmenting the planning and control components with learned policies.

## Current Implementation

Our existing implementation consists of:

### Architecture

```
┌─────────────────┐     ┌────────────────────┐     ┌───────────────┐
│   Perception    │────►│      Planning      │────►│    Control    │
│                 │     │                    │     │               │
│ - YOLOv8        │     │ - Behavioral       │     │ - PID         │
│ - Socket-based  │     │   Planner          │     │   Controller  │
│   Detection     │     │ - Local Planner    │     │ - Throttle    │
│ - Threading     │     │ - Path Optimizer   │     │   Brake       │
│                 │     │                    │     │   Steering    │
└─────────────────┘     └────────────────────┘     └───────────────┘
```

┌────────────────────────────────────────────────────────────────────┐
│                    Production CARLA DRL Pipeline                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────────┐    ┌──────────────┐    ┌─────────────────────┐ │
│  │ CARLA Server    │    │ Monitoring & │    │ Model Registry &    │ │
│  │ (0.8.4)         │    │ Logging Hub  │    │ Version Control     │ │
│  │ - Town01/02/03  │    │ - TensorBoard│    │ - MLflow/Weights&B  │ │
│  │ - Weather Ctrl  │    │ - Prometheus │    │ - A/B Testing       │ │
│  │ - Traffic Mgmt  │    │ - Grafana    │    │ - Rollback Mgmt     │ │
│  └─────────────────┘    └──────────────┘    └─────────────────────┘ │
│          │                      │                      │            │
│          ▼                      ▼                      ▼            │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │              Enhanced Communication Layer                       │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │ │
│  │  │ CARLA Py3.6 │  │ C++ Gateway  │  │ DRL Agent Py3.12       │ │ │
│  │  │ - Sensors   │◄►│ - ZeroMQ     │◄►│ - PPO/SAC              │ │ │
│  │  │ - Control   │  │ - ROS 2 DDS  │  │ - Curriculum Learning  │ │ │
│  │  │ - YOLO Det. │  │ - Buffering  │  │ - Multi-env Training   │ │ │
│  │  └─────────────┘  └──────────────┘  └─────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                     Advanced Features                          │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │ │
│  │  │ Curriculum  │  │ Multi-Agent  │  │ Safety & Validation     │ │ │
│  │  │ Learning    │  │ Training     │  │ - Formal Verification   │ │ │
│  │  │ - Progressive│  │ - Fleet Sim  │  │ - Sim-to-Real Transfer │ │ │
│  │  │ - Adaptive  │  │ - Swarm Intel│  │ - Shadow Testing        │ │ │
│  │  └─────────────┘  └──────────────┘  └─────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘

### System Architecture

┌─────────────────────┐   ┌──────────────────────┐   ┌─────────────────────┐
│  CARLA Server       │   │  CARLA Client        │   │  ROS 2 Gateway      │
│  (CARLA 0.8.4)      │   │  (Python 3.6)       │   │  (C++/Python 3.12)  │
│                     │   │                      │   │                     │
│ ┌─────────────────┐ │   │ ┌──────────────────┐ │   │ ┌─────────────────┐ │
│ │ Town01/02       │ │◄──┤ │ Sensor Manager   │ │   │ │ Topic Publisher │ │
│ │ Weather/Traffic │ │   │ │ - RGB Camera     │ │   │ │ - camera/image  │ │
│ │ Vehicle Physics │ │   │ │ - Depth Camera   │ │   │ │ - vehicle/state │ │
│ └─────────────────┘ │   │ │ - Vehicle State  │ │   │ │ - environment   │ │
│                     │   │ └──────────────────┘ │   │ └─────────────────┘ │
│ Port: 2000          │   │                      │   │                     │
│ Sync Mode: 30Hz     │   │ TCP Socket ──────────┼───┤ ZeroMQ/gRPC         │
└─────────────────────┘   └──────────────────────┘   └─────────────────────┘
                                                                 │
┌─────────────────────┐   ┌──────────────────────┐              │
│  TensorBoard        │   │  PPO Agent           │              │
│  Monitoring         │   │  (Python 3.12)      │ ◄────────────┘
│                     │   │                      │
│ ┌─────────────────┐ │   │ ┌──────────────────┐ │   ROS 2 Topics:
│ │ Episode Rewards │ │   │ │ PPO Algorithm    │ │   • /carla/camera/image
│ │ Loss Curves     │ │◄──┤ │ - Actor Network  │ │   • /carla/vehicle/state
│ │ Action Dist.    │ │   │ │ - Critic Network │ │   • /carla/environment/info
│ └─────────────────┘ │   │ │ - Experience     │ │   • /carla/vehicle/control
│                     │   │ │   Replay         │ │
│ Port: 6006          │   │ └──────────────────┘ │   QoS: RELIABLE
└─────────────────────┘   └──────────────────────┘   Rate: 30Hz (sim_time)

### Key Components:

1. **Perception System**:
   - Socket-based client-server architecture
   - YOLOv8 object detection (running in Python 3.12)
   - Threaded processing for improved performance

2. **Planning System**:
   - Two-tier architecture (behavioral and local planners)
   - Path optimization
   - Collision checking

3. **Control System**:
   - PID-based controllers for longitudinal and lateral control
   - Velocity planning with obstacle avoidance

## Project Goals

1. Develop a DRL solution that can:
   - Learn end-to-end driving policies
   - Handle complex driving scenarios
   - Achieve comparable or better performance than the traditional approach

2. Maintain critical safety mechanisms:
   - Collision avoidance
   - Traffic rule adherence
   - Failsafe mechanisms

3. Produce results suitable for academic publication:
   - Reproducible experiments
   - Thorough performance evaluation
   - Comparison with state-of-the-art methods

## Development Roadmap

### Phase 1: Environment Development ⬅ *Current Phase*

#### 1.1 RL-Compatible CARLA Wrapper [TODO]

- Create a gym-compatible environment wrapper
- Implement reset, step, and close methods
- Handle synchronous mode execution

```python
# Example skeleton:
class CarlaEnvWrapper:
    def __init__(self, host='localhost', port=2000, town='Town01', fps=30):
        # Initialize CARLA client, world, vehicle, sensors
        pass

    def reset(self):
        # Reset vehicle position, sensors
        # Return initial state
        pass

    def step(self, action):
        # Apply action to vehicle
        # Advance simulation
        # Calculate reward
        # Check termination conditions
        # Return state, reward, done, info
        pass

    def close(self):
        # Clean up resources
        pass
```

#### 1.2 State Representation [TODO]

Define a state representation that combines:
- Processed camera images
- Vehicle state (position, velocity, etc.)
- Navigation information (waypoints)
- Detection results (obstacles, traffic signs)

#### 1.3 Reward Function [TODO]

Implement a multi-component reward function considering:
- Progress toward destination
- Lane-keeping accuracy
- Safety (collision avoidance)
- Comfort (smooth control)
- Rule compliance (traffic lights, speed limits)

### Phase 2: DRL Agent Development

#### 2.1 Algorithm Selection [TODO]

Based on our literature review, select and implement one of:
- SAC (Soft Actor-Critic) - Best for sample efficiency
- TQC (Truncated Quantile Critics) - Best for handling uncertainty
- PPO (Proximal Policy Optimization) - Best for stability

#### 2.2 Neural Network Architecture [TODO]

Design neural networks for:
- Processing visual input
- Processing vehicle state
- Action selection
- Value estimation

#### 2.3 Training Loop [TODO]

Implement a robust training loop with:
- Experience collection
- Replay buffer management
- Neural network updates
- Logging and checkpointing

### Phase 3: Integration with Existing System

#### 3.1 Integration Points [TODO]

Define clear integration points between:
- Existing perception system
- DRL-based planning and control
- Safety override mechanisms

#### 3.2 Safety Mechanisms [TODO]

Implement safety override mechanisms:
- Collision prevention
- Speed limiting
- Lane departure prevention

### Phase 4: Evaluation Framework

#### 4.1 Benchmark Scenarios [TODO]

Define standard benchmark scenarios:
- Lane following
- Intersection navigation
- Urban driving with traffic
- Highway driving
- Adverse weather conditions

#### 4.2 Evaluation Metrics [TODO]

Define key performance metrics:
- Success rate
- Average reward
- Lane deviation
- Smoothness of control
- Number of infractions
- Completion time

## Required Dependencies

- **CARLA**: Modified v0.8.4
- **Python**: 3.6.x (CARLA client) and 3.12 (Detection server)
- **Deep Learning**: PyTorch (v1.8.0+)
- **Reinforcement Learning**: Gymnasium (successor to Gym)
- **Visualization**: Matplotlib, Tensorboard
- **Configuration**: Pydantic, YAML
- **Other**: NumPy, OpenCV, Pandas

## Coding Standards

This project adheres to the following standards:

- **Style Guide**: PEP 8
- **Type Hints**: Used throughout the codebase
- **Documentation**: Google-style or NumPy-style docstrings
- **Logging**: Proper logging instead of print statements
- **Reproducibility**: Deterministic seeds for randomness
- **Architecture**: Clear separation of concerns
- **Resource Management**: Graceful teardown in CARLA
- **Configuration**: Pydantic/dataclasses for configs
- **Dependencies**: Preference for pure-Python solutions

## References

### Papers

1. Peng et al. (2021). End-to-End Autonomous Driving Through Dueling Double Deep Q-Network.
2. Pérez-Gil et al. (2022). Deep reinforcement learning based control for Autonomous Vehicles in CARLA.
3. Selvaraj et al. (2023). A Deep Reinforcement Learning Approach for Efficient, Safe and Comfortable Driving.
4. Ben Elallid et al. (2023). Deep Reinforcement Learning for Autonomous Vehicle Intersection Navigation.
5. Nehme & Deo (2023). Safe Navigation: Training Autonomous Vehicles using Deep Reinforcement Learning in CARLA.
6. Zhao et al. (2024). Towards Robust Decision-Making for Autonomous Highway Driving Based on Safe Reinforcement Learning.
7. Kumar (2025). Autonomous Navigation with Deep Reinforcement Learning in CARLA Simulator.
8. Park et al. (2025). A Comparative Study of Deep Reinforcement Learning Algorithms for Urban Autonomous Driving: Addressing the Geographic and Regulatory Challenges in CARLA.

### Code Repositories

1. [CARLA Simulator](https://github.com/carla-simulator/carla)
2. [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)

---

## Next Steps

1. Implement the RL-compatible CARLA environment wrapper
2. Define state representation and reward function
3. Select and implement DRL algorithm
4. Create initial integration points with existing system

## Meetings and Progress

| Date       | Milestone                            | Status      |
|------------|--------------------------------------|-------------|
| 2025-08-15 | Project kickoff, initial assessment  | Completed   |
| TBD        | Environment wrapper implementation   | Not started |
| TBD        | DRL agent implementation             | Not started |
| TBD        | Integration with existing system     | Not started |
| TBD        | Initial training results             | Not started |
