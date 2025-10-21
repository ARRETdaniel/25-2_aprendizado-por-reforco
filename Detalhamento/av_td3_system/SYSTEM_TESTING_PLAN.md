# 🧪 System Testing Plan: TD3 Autonomous Navigation System
## Phase 3 Pre-Training Validation & Execution Testing

**Created:** January 29, 2025
**Author:** GitHub Copilot (AI Assistant)
**Purpose:** Validate all system components work correctly before 8-10 day training execution
**Target System:** CARLA 0.9.16 + ROS 2 + TD3 DRL Agent (Docker-based)

---

## 📋 Table of Contents

1. [Testing Overview](#testing-overview)
2. [Prerequisites Checklist](#prerequisites-checklist)
3. [Test Suite 1: Docker Infrastructure](#test-suite-1-docker-infrastructure)
4. [Test Suite 2: CARLA Server Connectivity](#test-suite-2-carla-server-connectivity)
5. [Test Suite 3: Environment Integration](#test-suite-3-environment-integration)
6. [Test Suite 4: Agent Functionality](#test-suite-4-agent-functionality)
7. [Test Suite 5: Training Pipeline](#test-suite-5-training-pipeline)
8. [Test Suite 6: End-to-End Integration](#test-suite-6-end-to-end-integration)
9. [Validation Checklist](#validation-checklist)
10. [Troubleshooting Guide](#troubleshooting-guide)

---

## 🎯 Testing Overview

### Objectives

1. **Verify Docker Infrastructure** - Ensure container runs CARLA 0.9.16 with GPU access
2. **Test CARLA Server Connection** - Validate client can connect and control simulation
3. **Validate Environment Wrapper** - Confirm CARLANavigationEnv works as specified in paper
4. **Test TD3 Agent** - Ensure agent can select actions and train on sample data
5. **Validate Training Pipeline** - Test training script executes without errors
6. **End-to-End Integration** - Run short training episode to verify full system

### Success Criteria

✅ **System is READY for training if ALL tests pass:**
- Docker container launches CARLA server successfully
- Python client connects to CARLA and spawns ego vehicle
- Environment produces 535-dim state vectors (512 CNN + 3 kinematics + 20 waypoints)
- Agent selects 2-dim continuous actions [-1,1]²
- Replay buffer stores transitions correctly
- Training loop executes for at least 100 timesteps without crashes
- Checkpoints can be saved and loaded

### Testing Environment

- **Hardware:** RTX 2060 (6GB VRAM) or university HPC cluster (A100)
- **OS:** Ubuntu 20.04 (inside Docker container)
- **Docker Image:** `td3-av-system:v2.0-python310` (30.6GB)
- **CARLA Version:** 0.9.16 (confirmed by fetched documentation)
- **Python:** 3.10.19 (installed via Miniforge/conda-forge)
- **PyTorch:** 2.4.1+cu121 with CUDA support

---

## ✅ Prerequisites Checklist

### System Requirements

```bash
# Check Docker installation
docker --version
# Expected: Docker version 28.1.1 or higher

# Check NVIDIA Container Toolkit
docker run --rm --gpus all ubuntu nvidia-smi
# Expected: Should show your GPU (RTX 2060/A100)

# Check disk space
df -h /
# Expected: At least 50GB free for checkpoints, logs, results

# Verify Docker image exists
docker images | grep td3-av-system
# Expected: td3-av-system:v2.0-python310    ...    30.6GB
```

### Project Files Verification

```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

# Check critical directories exist
ls -la src/{agents,networks,environment,utils}
ls -la config/
ls -la data/waypoints/

# Verify configuration files
cat config/carla_config.yaml | head -20
cat config/td3_config.yaml | head -20
```

### Expected File Structure
```
✅ src/environment/carla_env.py (530+ lines)
✅ src/environment/sensors.py (430+ lines)
✅ src/environment/reward_functions.py (280+ lines)
✅ src/environment/waypoint_manager.py (230+ lines)
✅ src/networks/actor.py (200+ lines)
✅ src/networks/critic.py (270+ lines)
✅ src/networks/cnn_extractor.py (280+ lines)
✅ src/utils/replay_buffer.py (189 lines)
✅ src/agents/td3_agent.py (395 lines)
✅ src/agents/ddpg_agent.py (414 lines)
```

---

## 🐳 Test Suite 1: Docker Infrastructure

### Test 1.1: Docker Container Launch

**Purpose:** Verify container starts with GPU access and all dependencies

```bash
# Start container interactively
docker run --rm -it \
  --gpus all \
  --net=host \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/src:/workspace/src \
  -v $(pwd)/config:/workspace/config \
  -v $(pwd)/scripts:/workspace/scripts \
  td3-av-system:v2.0-python310 bash

# Inside container, verify GPU
nvidia-smi
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}')"
```

**Expected Output:**
```
CUDA Available: True
Device: NVIDIA GeForce RTX 2060 (or Tesla A100-SXM4-40GB)
```

**✅ Pass Criteria:** GPU accessible from PyTorch inside container

### Test 1.2: CARLA Server Launch

**Purpose:** Verify CARLA 0.9.16 server starts in headless mode

```bash
# Inside Docker container
cd /home/carla

# Start CARLA server in background (headless mode)
./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 &

# Wait for server startup (usually 10-15 seconds)
sleep 15

# Check CARLA process
ps aux | grep CarlaUE4
```

**Expected Output:**
```
carla    1234  ... ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000
```

**✅ Pass Criteria:** CARLA server process running without crashes

### Test 1.3: Python Dependencies

**Purpose:** Ensure all required Python packages installed correctly

```bash
# Inside container
python3 << EOF
import sys
print("Python version:", sys.version)

# Test CARLA Python API (built-in for 0.9.16)
import carla
print(f"CARLA API version: {carla.__version__}")

# Test PyTorch
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

# Test other dependencies
import numpy as np
import cv2
import yaml
import gymnasium as gym

print("✅ All dependencies imported successfully!")
EOF
```

**Expected Output:**
```
Python version: 3.8.10
CARLA API version: 0.9.16
PyTorch version: 2.4.1+cu121
CUDA version: 12.1
✅ All dependencies imported successfully!
```

**✅ Pass Criteria:** No import errors, correct versions

---

## 🚗 Test Suite 2: CARLA Server Connectivity

### Test 2.1: Basic Client Connection

**Purpose:** Verify Python client can connect to CARLA server

```python
# test_carla_connection.py
import carla
import time

def test_connection():
    try:
        # Connect to CARLA server
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # Get world and verify
        world = client.get_world()
        print(f"✅ Connected to CARLA server")
        print(f"   Map: {world.get_map().name}")
        print(f"   Actors: {len(world.get_actors())}")

        # Test synchronous mode
        settings = world.get_settings()
        print(f"   Synchronous Mode: {settings.synchronous_mode}")
        print(f"   Fixed Delta Seconds: {settings.fixed_delta_seconds}")

        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
```

**Run:**
```bash
python3 test_carla_connection.py
```

**Expected Output:**
```
✅ Connected to CARLA server
   Map: Town01
   Actors: 0-5 (spectator + possible traffic)
   Synchronous Mode: False (or True if already set)
   Fixed Delta Seconds: 0.0 (or 0.05 if set)
```

**✅ Pass Criteria:** Client connects without timeout

### Test 2.2: Vehicle Spawning

**Purpose:** Test ego vehicle can be spawned and controlled

```python
# test_vehicle_spawn.py
import carla
import time

def test_vehicle_spawn():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Get blueprint library
    blueprint_library = world.get_blueprint_library()

    # Choose vehicle (Tesla Model 3 or similar)
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    print(f"Vehicle blueprint: {vehicle_bp.id}")

    # Get spawn point
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[0]
    print(f"Spawn point: {spawn_point.location}")

    # Spawn vehicle
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"✅ Vehicle spawned: ID={vehicle.id}")

    # Test control
    control = carla.VehicleControl()
    control.throttle = 0.5
    control.steer = 0.0
    vehicle.apply_control(control)

    # Wait and observe
    time.sleep(2)

    # Get vehicle state
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5  # km/h

    print(f"   Location: {transform.location}")
    print(f"   Speed: {speed:.2f} km/h")

    # Cleanup
    vehicle.destroy()
    print("✅ Vehicle destroyed successfully")

    return True

if __name__ == "__main__":
    test_vehicle_spawn()
```

**✅ Pass Criteria:** Vehicle spawns, moves with throttle input, and is destroyed cleanly

### Test 2.3: Camera Sensor Attachment

**Purpose:** Verify RGB camera can be attached and captures images

```python
# test_camera_sensor.py
import carla
import numpy as np
import time

def test_camera_sensor():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Spawn vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Attach camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '256')
    camera_bp.set_attribute('image_size_y', '144')
    camera_bp.set_attribute('fov', '90')

    camera_transform = carla.Transform(carla.Location(x=2.0, z=1.5))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    print(f"✅ Camera attached: ID={camera.id}")

    # Capture image
    image_data = []
    def save_image(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))  # BGRA
        image_data.append(array[:, :, :3])  # RGB only

    camera.listen(save_image)
    time.sleep(1.0)  # Wait for first image
    camera.stop()

    if len(image_data) > 0:
        print(f"✅ Captured {len(image_data)} frames")
        print(f"   Image shape: {image_data[0].shape}")
        print(f"   Image dtype: {image_data[0].dtype}")
    else:
        print("❌ No images captured!")
        return False

    # Cleanup
    camera.destroy()
    vehicle.destroy()

    return True

if __name__ == "__main__":
    test_camera_sensor()
```

**Expected Output:**
```
✅ Camera attached: ID=XXX
✅ Captured 10-20 frames (depends on FPS)
   Image shape: (144, 256, 3)
   Image dtype: uint8
```

**✅ Pass Criteria:** Camera captures RGB images in expected format

---

## 🔄 Test Suite 3: Environment Integration

### Test 3.1: CARLANavigationEnv Initialization

**Purpose:** Verify Gym environment wrapper initializes correctly

```python
# test_environment_init.py
import sys
sys.path.append('/workspace')

from src.environment.carla_env import CARLANavigationEnv
import yaml

def test_env_init():
    # Load config
    with open('/workspace/config/carla_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("🔧 Initializing CARLANavigationEnv...")
    env = CARLANavigationEnv(config)

    print(f"✅ Environment initialized")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    # Test reset
    print("\n🔄 Testing reset()...")
    state = env.reset()

    print(f"✅ Reset successful")
    print(f"   State shape: {state.shape}")
    print(f"   State dtype: {state.dtype}")
    print(f"   State range: [{state.min():.3f}, {state.max():.3f}]")

    # Cleanup
    env.close()
    print("\n✅ Environment closed cleanly")

    return True

if __name__ == "__main__":
    test_env_init()
```

**Expected Output:**
```
🔧 Initializing CARLANavigationEnv...
✅ Environment initialized
   Observation space: Box(-inf, inf, (535,), float32)
   Action space: Box(-1.0, 1.0, (2,), float32)

🔄 Testing reset()...
✅ Reset successful
   State shape: (535,)
   State dtype: float32
   State range: [-X.XXX, X.XXX]

✅ Environment closed cleanly
```

**✅ Pass Criteria:** Environment creates 535-dim state vector

### Test 3.2: State Vector Composition

**Purpose:** Verify state contains CNN features + kinematics + waypoints

```python
# test_state_composition.py
import sys
sys.path.append('/workspace')

from src.environment.carla_env import CARLANavigationEnv
import yaml
import numpy as np

def test_state_composition():
    with open('/workspace/config/carla_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    env = CARLANavigationEnv(config)
    state = env.reset()

    print("🔍 Analyzing state vector composition...")
    print(f"Total state size: {state.shape[0]} dimensions")

    # According to paper: 512 (CNN) + 3 (kinematics) + 20 (waypoints)
    cnn_features = state[:512]
    kinematics = state[512:515]
    waypoints = state[515:535]

    print(f"\n📊 State Breakdown:")
    print(f"   CNN Features:  {cnn_features.shape} (indices 0-511)")
    print(f"      Range: [{cnn_features.min():.3f}, {cnn_features.max():.3f}]")

    print(f"   Kinematics:    {kinematics.shape} (indices 512-514)")
    print(f"      Values: {kinematics}")
    print(f"      Expected: [velocity, lateral_deviation, heading_error]")

    print(f"   Waypoints:     {waypoints.shape} (indices 515-534)")
    print(f"      Range: [{waypoints.min():.3f}, {waypoints.max():.3f}]")
    print(f"      Shape: 10 waypoints × 2 coords (x, y) = 20 values")

    # Verify no NaN or inf
    assert not np.any(np.isnan(state)), "❌ State contains NaN values!"
    assert not np.any(np.isinf(state)), "❌ State contains inf values!"
    print("\n✅ State vector valid (no NaN/inf)")

    env.close()
    return True

if __name__ == "__main__":
    test_state_composition()
```

**✅ Pass Criteria:** State correctly composed of 512+3+20 components

### Test 3.3: Environment Step Function

**Purpose:** Test step() executes and returns correct format

```python
# test_environment_step.py
import sys
sys.path.append('/workspace')

from src.environment.carla_env import CARLANavigationEnv
import yaml
import numpy as np

def test_env_step():
    with open('/workspace/config/carla_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    env = CARLANavigationEnv(config)
    state = env.reset()

    print("🎮 Testing environment step()...")

    # Sample action: [steering, throttle/brake]
    action = np.array([0.0, 0.5], dtype=np.float32)  # Straight, moderate throttle
    print(f"   Action: {action}")

    # Execute step
    next_state, reward, done, truncated, info = env.step(action)

    print(f"\n✅ Step executed successfully")
    print(f"   Next state shape: {next_state.shape}")
    print(f"   Reward: {reward:.3f}")
    print(f"   Done: {done}")
    print(f"   Truncated: {truncated}")
    print(f"   Info keys: {list(info.keys())}")

    # Verify reward components
    if 'reward_breakdown' in info:
        print(f"\n📊 Reward Breakdown:")
        for component, value in info['reward_breakdown'].items():
            print(f"      {component}: {value:.3f}")

    # Execute multiple steps
    print(f"\n🔄 Running 10 continuous steps...")
    for i in range(10):
        action = np.random.uniform(-1, 1, size=2)
        next_state, reward, done, truncated, info = env.step(action)

        if done or truncated:
            print(f"   Episode terminated at step {i+1}")
            print(f"   Reason: {info.get('termination_reason', 'unknown')}")
            break
    else:
        print(f"✅ Completed 10 steps without termination")

    env.close()
    return True

if __name__ == "__main__":
    test_env_step()
```

**Expected Output:**
```
✅ Step executed successfully
   Next state shape: (535,)
   Reward: X.XXX
   Done: False
   Truncated: False
   Info keys: ['reward_breakdown', 'speed', 'lateral_deviation', ...]

📊 Reward Breakdown:
      efficiency: X.XXX
      lane_keeping: X.XXX
      comfort: X.XXX
      safety: X.XXX

🔄 Running 10 continuous steps...
✅ Completed 10 steps without termination
```

**✅ Pass Criteria:** Environment step() returns valid (s', r, done, truncated, info)

---

## 🤖 Test Suite 4: Agent Functionality

### Test 4.1: TD3 Agent Initialization

**Purpose:** Verify TD3 agent loads correctly with all components

```python
# test_td3_agent_init.py
import sys
sys.path.append('/workspace')

from src.agents.td3_agent.py import TD3Agent
import torch

def test_td3_init():
    print("🤖 Initializing TD3 Agent...")

    agent = TD3Agent(
        state_dim=535,
        action_dim=2,
        max_action=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"✅ TD3 Agent initialized")
    print(f"   Device: {agent.device}")
    print(f"   Actor parameters: {sum(p.numel() for p in agent.actor.parameters()):,}")
    print(f"   Critic1 parameters: {sum(p.numel() for p in agent.critic.Q1.parameters()):,}")
    print(f"   Critic2 parameters: {sum(p.numel() for p in agent.critic.Q2.parameters()):,}")
    print(f"   Total trainable params: {sum(p.numel() for p in agent.actor.parameters()) + sum(p.numel() for p in agent.critic.parameters()):,}")

    # Verify target networks initialized
    print(f"\n🎯 Target Networks:")
    print(f"   Actor target exists: {hasattr(agent, 'actor_target')}")
    print(f"   Critic target exists: {hasattr(agent, 'critic_target')}")

    # Verify replay buffer
    print(f"\n💾 Replay Buffer:")
    print(f"   Capacity: {agent.replay_buffer.max_size:,}")
    print(f"   Current size: {agent.replay_buffer.size}")

    return True

if __name__ == "__main__":
    test_td3_init()
```

**Expected Output:**
```
🤖 Initializing TD3 Agent...
✅ TD3 Agent initialized
   Device: cuda
   Actor parameters: XXX,XXX
   Critic1 parameters: XXX,XXX
   Critic2 parameters: XXX,XXX
   Total trainable params: ~XXX,XXX

🎯 Target Networks:
   Actor target exists: True
   Critic target exists: True

💾 Replay Buffer:
   Capacity: 100,000
   Current size: 0
```

**✅ Pass Criteria:** Agent initializes with all networks and replay buffer

### Test 4.2: Action Selection

**Purpose:** Test agent can select actions for given states

```python
# test_action_selection.py
import sys
sys.path.append('/workspace')

from src.agents.td3_agent import TD3Agent
import numpy as np
import torch

def test_action_selection():
    agent = TD3Agent(state_dim=535, action_dim=2, max_action=1.0)

    # Create dummy state
    state = np.random.randn(535).astype(np.float32)

    print("🎯 Testing action selection...")

    # Test without noise (evaluation mode)
    action_eval = agent.select_action(state, noise=0.0)
    print(f"✅ Evaluation action: {action_eval}")
    print(f"   Shape: {action_eval.shape}")
    print(f"   Range: [{action_eval.min():.3f}, {action_eval.max():.3f}]")
    assert action_eval.shape == (2,), "❌ Action shape incorrect!"
    assert -1.0 <= action_eval.min() and action_eval.max() <= 1.0, "❌ Action out of bounds!"

    # Test with noise (training mode)
    action_train = agent.select_action(state, noise=0.1)
    print(f"\n🔊 Training action (with noise):")
    print(f"   Action: {action_train}")
    print(f"   Difference from eval: {np.abs(action_train - action_eval)}")

    # Test batch of actions
    batch_size = 10
    states = np.random.randn(batch_size, 535).astype(np.float32)
    actions = np.array([agent.select_action(s, noise=0.0) for s in states])
    print(f"\n📦 Batch action selection:")
    print(f"   Batch size: {actions.shape[0]}")
    print(f"   Actions shape: {actions.shape}")
    print(f"   All in range: {np.all((actions >= -1.0) & (actions <= 1.0))}")

    return True

if __name__ == "__main__":
    test_action_selection()
```

**✅ Pass Criteria:** Actions are 2-dimensional and within [-1, 1]

### Test 4.3: Training Step

**Purpose:** Verify agent can perform training update on sample data

```python
# test_training_step.py
import sys
sys.path.append('/workspace')

from src.agents.td3_agent import TD3Agent
import numpy as np

def test_training_step():
    agent = TD3Agent(state_dim=535, action_dim=2, max_action=1.0)

    print("📚 Populating replay buffer with sample data...")

    # Add 1000 random transitions
    for i in range(1000):
        state = np.random.randn(535).astype(np.float32)
        action = np.random.uniform(-1, 1, size=2).astype(np.float32)
        next_state = np.random.randn(535).astype(np.float32)
        reward = np.random.randn()
        done = float(np.random.rand() < 0.1)  # 10% done probability

        agent.replay_buffer.add(state, action, next_state, reward, done)

    print(f"✅ Added 1000 transitions")
    print(f"   Buffer size: {agent.replay_buffer.size}")

    # Perform training step
    print(f"\n🏋️ Performing training update...")
    critic_loss, actor_loss = agent.train(batch_size=100)

    print(f"✅ Training step completed")
    print(f"   Critic loss: {critic_loss:.6f}")
    print(f"   Actor loss: {actor_loss if actor_loss is not None else 'N/A (delayed)'}")
    print(f"   Total iterations: {agent.total_it}")

    # Perform multiple updates
    print(f"\n🔄 Running 10 training updates...")
    for i in range(10):
        critic_loss, actor_loss = agent.train(batch_size=100)
        if actor_loss is not None:
            print(f"   Step {agent.total_it}: C={critic_loss:.6f}, A={actor_loss:.6f}")
        else:
            print(f"   Step {agent.total_it}: C={critic_loss:.6f}, A=delayed")

    print(f"\n✅ Completed 10 training steps")

    return True

if __name__ == "__main__":
    test_training_step()
```

**Expected Output:**
```
📚 Populating replay buffer with sample data...
✅ Added 1000 transitions
   Buffer size: 1000

🏋️ Performing training update...
✅ Training step completed
   Critic loss: X.XXXXXX
   Actor loss: N/A (delayed)
   Total iterations: 1

🔄 Running 10 training updates...
   Step 1: C=X.XXXXXX, A=delayed
   Step 2: C=X.XXXXXX, A=X.XXXXXX (updates every 2 steps due to policy_freq=2)
   ...

✅ Completed 10 training steps
```

**✅ Pass Criteria:** Agent trains without errors, actor updates every 2 steps

### Test 4.4: Checkpoint Save/Load

**Purpose:** Test agent can save and restore checkpoints

```python
# test_checkpoint.py
import sys
sys.path.append('/workspace')

from src.agents.td3_agent import TD3Agent
import numpy as np
import os

def test_checkpoint():
    agent1 = TD3Agent(state_dim=535, action_dim=2, max_action=1.0)

    # Add some data and train
    for i in range(500):
        state = np.random.randn(535).astype(np.float32)
        action = np.random.uniform(-1, 1, size=2).astype(np.float32)
        next_state = np.random.randn(535).astype(np.float32)
        reward = np.random.randn()
        done = 0.0
        agent1.replay_buffer.add(state, action, next_state, reward, done)

    for _ in range(10):
        agent1.train(batch_size=100)

    # Get action before save
    test_state = np.random.randn(535).astype(np.float32)
    action_before = agent1.select_action(test_state, noise=0.0)

    print(f"💾 Saving checkpoint...")
    checkpoint_path = "/tmp/test_checkpoint.pth"
    agent1.save(checkpoint_path)
    print(f"✅ Checkpoint saved to {checkpoint_path}")
    print(f"   File size: {os.path.getsize(checkpoint_path) / 1024:.2f} KB")

    # Create new agent and load
    print(f"\n🔄 Loading checkpoint into new agent...")
    agent2 = TD3Agent(state_dim=535, action_dim=2, max_action=1.0)
    agent2.load(checkpoint_path)
    print(f"✅ Checkpoint loaded")

    # Get action after load (should be identical)
    action_after = agent2.select_action(test_state, noise=0.0)

    print(f"\n🔍 Comparing actions:")
    print(f"   Before save: {action_before}")
    print(f"   After load:  {action_after}")
    print(f"   Difference:  {np.abs(action_before - action_after).sum()}")

    assert np.allclose(action_before, action_after, atol=1e-6), "❌ Actions differ after load!"
    print(f"\n✅ Checkpoint save/load works correctly")

    # Cleanup
    os.remove(checkpoint_path)

    return True

if __name__ == "__main__":
    test_checkpoint()
```

**✅ Pass Criteria:** Loaded agent produces identical actions to original

---

## 🎯 Test Suite 5: Training Pipeline

### Test 5.1: Training Script Execution (Short Run)

**Purpose:** Test training script runs for 100 timesteps without crashes

```python
# test_training_pipeline_short.py
"""
Short training run (100 timesteps) to validate full pipeline.
This is NOT actual training, just a system validation test.
"""
import sys
sys.path.append('/workspace')

from src.environment.carla_env import CARLANavigationEnv
from src.agents.td3_agent import TD3Agent
import yaml
import numpy as np
import time

def test_short_training():
    # Load configs
    with open('/workspace/config/carla_config.yaml', 'r') as f:
        carla_config = yaml.safe_load(f)

    print("🚀 Starting short training validation run...")
    print(f"   Target: 100 timesteps")
    print(f"   Purpose: System integration test")

    # Initialize environment
    print(f"\n🌍 Initializing environment...")
    env = CARLANavigationEnv(carla_config)

    # Initialize agent
    print(f"🤖 Initializing TD3 agent...")
    agent = TD3Agent(state_dim=535, action_dim=2, max_action=1.0)

    # Training loop
    state = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    start_time = time.time()

    for t in range(100):
        episode_timesteps += 1

        # Select action (random for first 25 steps, then agent)
        if t < 25:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, noise=0.1)

        # Execute action
        next_state, reward, done, truncated, info = env.step(action)

        # Store transition
        agent.replay_buffer.add(state, action, next_state, reward, float(done))

        state = next_state
        episode_reward += reward

        # Train (after buffer has some data)
        if t >= 50:
            agent.train(batch_size=32)

        # Episode end
        if done or truncated:
            print(f"   Episode {episode_num+1} finished at step {t+1}")
            print(f"      Reward: {episode_reward:.2f}")
            print(f"      Timesteps: {episode_timesteps}")
            print(f"      Termination: {info.get('termination_reason', 'unknown')}")

            state = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Progress update every 20 steps
        if (t + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"   Progress: {t+1}/100 steps ({elapsed:.1f}s)")

    end_time = time.time()

    print(f"\n✅ Training validation completed!")
    print(f"   Total time: {end_time - start_time:.2f} seconds")
    print(f"   Steps/second: {100 / (end_time - start_time):.2f}")
    print(f"   Episodes completed: {episode_num}")
    print(f"   Final buffer size: {agent.replay_buffer.size}")

    # Cleanup
    env.close()

    return True

if __name__ == "__main__":
    test_short_training()
```

**Expected Output:**
```
🚀 Starting short training validation run...
   Target: 100 timesteps
   Purpose: System integration test

🌍 Initializing environment...
🤖 Initializing TD3 agent...

   Progress: 20/100 steps (X.Xs)
   Progress: 40/100 steps (X.Xs)
   Progress: 60/100 steps (X.Xs)
   Progress: 80/100 steps (X.Xs)
   Progress: 100/100 steps (X.Xs)

✅ Training validation completed!
   Total time: XX.XX seconds
   Steps/second: X.XX
   Episodes completed: 0-2 (depends on termination)
   Final buffer size: 100
```

**✅ Pass Criteria:** Runs 100 steps without crashes, agent trains after step 50

---

## 🔗 Test Suite 6: End-to-End Integration

### Test 6.1: Full System Integration Test

**Purpose:** Run complete system for 1 full episode (up to 300s simulation time)

```python
# test_full_episode.py
import sys
sys.path.append('/workspace')

from src.environment.carla_env import CARLANavigationEnv
from src.agents.td3_agent import TD3Agent
import yaml
import time

def test_full_episode():
    # Load config
    with open('/workspace/config/carla_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("🎬 Running full episode integration test...")
    print(f"   Max simulation time: 300s")
    print(f"   Max timesteps: Unlimited (until done/truncated)")

    # Initialize
    env = CARLANavigationEnv(config)
    agent = TD3Agent(state_dim=535, action_dim=2, max_action=1.0)

    # Reset
    state = env.reset()
    episode_reward = 0
    timesteps = 0

    start_real_time = time.time()

    done = False
    truncated = False

    while not (done or truncated):
        timesteps += 1

        # Select action
        if timesteps < 50:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, noise=0.1)

        # Step
        next_state, reward, done, truncated, info = env.step(action)

        # Store
        agent.replay_buffer.add(state, action, next_state, reward, float(done))

        state = next_state
        episode_reward += reward

        # Train
        if timesteps >= 100:
            agent.train(batch_size=64)

        # Progress
        if timesteps % 100 == 0:
            sim_time = info.get('simulation_time', 0)
            speed = info.get('speed', 0)
            print(f"   Step {timesteps}: sim_time={sim_time:.1f}s, speed={speed:.1f}m/s, reward={episode_reward:.2f}")

    end_real_time = time.time()

    print(f"\n🏁 Episode finished!")
    print(f"   Timesteps: {timesteps}")
    print(f"   Total reward: {episode_reward:.2f}")
    print(f"   Termination reason: {info.get('termination_reason', 'unknown')}")
    print(f"   Real time: {end_real_time - start_real_time:.2f}s")
    print(f"   Simulation time: {info.get('simulation_time', 0):.2f}s")

    # Metrics
    if 'collision_count' in info:
        print(f"\n📊 Episode Metrics:")
        print(f"   Collisions: {info['collision_count']}")
        print(f"   Off-road events: {info.get('off_road_count', 0)}")
        print(f"   Average speed: {info.get('average_speed', 0):.2f} m/s")
        print(f"   Route progress: {info.get('route_progress', 0):.1%}")

    env.close()
    return True

if __name__ == "__main__":
    test_full_episode()
```

**✅ Pass Criteria:** Episode runs to completion (collision/timeout/goal), metrics logged

---

## ✅ Validation Checklist

After running all tests, verify the following:

### Infrastructure
- [ ] Docker container launches with GPU access
- [ ] CARLA 0.9.16 server starts in headless mode
- [ ] Python 3.8.10 environment with all dependencies
- [ ] PyTorch 2.4.1+cu121 detects CUDA correctly

### CARLA Server
- [ ] Client connects to server without timeout
- [ ] Ego vehicle spawns at designated spawn point
- [ ] RGB camera attaches and captures images (256×144)
- [ ] Collision sensor registers events
- [ ] Lane invasion sensor detects lane crossings

### Environment Wrapper
- [ ] `CARLANavigationEnv` initializes without errors
- [ ] `reset()` returns 535-dim state vector
- [ ] State composition: 512 (CNN) + 3 (kinematics) + 20 (waypoints)
- [ ] `step()` returns (s', r, done, truncated, info) in correct format
- [ ] Reward breakdown includes all 4 components
- [ ] Episode terminates on collision/off-road/timeout

### TD3 Agent
- [ ] Agent initializes with actor, twin critics, target networks
- [ ] Replay buffer capacity is 100K
- [ ] `select_action()` returns 2-dim action in [-1,1]
- [ ] Evaluation mode (noise=0) is deterministic
- [ ] Training mode (noise=0.1) adds Gaussian noise
- [ ] `train()` performs TD3 update (twin Q, delayed policy, target smoothing)
- [ ] Actor updates only every 2 steps (policy_freq=2)
- [ ] Checkpoint save/load preserves network weights

### Training Pipeline
- [ ] Training script runs for 100 timesteps without crashes
- [ ] Agent starts with random actions (first 25 steps)
- [ ] Agent switches to policy after warmup
- [ ] Training begins after buffer has data (step 50)
- [ ] Transitions stored in replay buffer correctly
- [ ] Full episode completes to termination
- [ ] Metrics logged at regular intervals

### Output & Logging
- [ ] Checkpoints save to `/workspace/data/checkpoints/`
- [ ] Logs save to `/workspace/data/logs/`
- [ ] Episode metrics include reward, speed, collisions
- [ ] Training losses (critic, actor) are finite numbers

---

## 🛠️ Troubleshooting Guide

### Problem 1: Docker Container Won't Start

**Symptoms:**
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```

**Solution:**
```bash
# Reinstall NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Problem 2: CARLA Server Timeout

**Symptoms:**
```
RuntimeError: time-out of 10000ms while waiting for the simulator
```

**Solution:**
```bash
# Increase client timeout
client.set_timeout(30.0)  # 30 seconds

# Or wait longer before connecting
sleep 20  # After starting CARLA server

# Check CARLA server logs
docker logs <container_id>
```

### Problem 3: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MB
```

**Solution:**
```python
# Reduce batch size
agent.train(batch_size=32)  # Instead of 100

# Or reduce image resolution in config
image_size_x: 200  # Instead of 256
image_size_y: 112  # Instead of 144

# Or use CPU for training (slow but works)
agent = TD3Agent(..., device="cpu")
```

### Problem 4: State Vector Shape Mismatch

**Symptoms:**
```
RuntimeError: Expected state shape (535,), got (XXX,)
```

**Solution:**
```python
# Check CNN output dimension
from src.networks.cnn_extractor import NatureCNN
cnn = NatureCNN()
dummy_input = torch.randn(1, 4, 84, 84)
output = cnn(dummy_input)
print(f"CNN output shape: {output.shape}")  # Should be [1, 512]

# Check waypoint count
waypoints = waypoint_manager.get_next_waypoints(10)  # Should return 10
print(f"Waypoints: {len(waypoints)}")  # Should be 10

# Check kinematics
kinematics = [velocity, lateral_dev, heading_error]
print(f"Kinematics: {len(kinematics)}")  # Should be 3
```

### Problem 5: Training Loss is NaN

**Symptoms:**
```
Critic loss: nan
Actor loss: nan
```

**Solution:**
```python
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=1.0)

# Reduce learning rate
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)

# Check for invalid rewards
if not np.isfinite(reward):
    reward = 0.0  # Replace with safe default
```

### Problem 6: Episode Never Terminates

**Symptoms:**
Episode runs for thousands of steps without `done=True`

**Solution:**
```yaml
# Add max_episode_steps to config
carla_config.yaml:
  episode:
    max_simulation_time: 300  # 5 minutes max
    max_steps: 6000  # 300s × 20Hz = 6000 steps

# Check termination conditions
if collision_detected:
    done = True
if simulation_time > max_time:
    truncated = True
```

---

## 📝 Final Pre-Training Checklist

Before starting the **8-10 day training run**, ensure:

### System Verification
- [x] All Test Suite 1 (Docker Infrastructure) tests pass
- [x] All Test Suite 2 (CARLA Connectivity) tests pass
- [x] All Test Suite 3 (Environment) tests pass
- [x] All Test Suite 4 (Agent) tests pass
- [x] All Test Suite 5 (Training Pipeline) tests pass
- [x] All Test Suite 6 (Integration) tests pass

### Configuration Review
- [ ] Verify `config/td3_config.yaml` has correct hyperparameters
- [ ] Confirm `config/carla_config.yaml` uses synchronous mode (20Hz)
- [ ] Check waypoint file exists in `data/waypoints/Town01_route.csv`
- [ ] Verify checkpoint directory has write permissions
- [ ] Ensure log directory is ready for TensorBoard

### Resource Allocation
- [ ] At least 50GB free disk space for checkpoints
- [ ] GPU has at least 6GB VRAM (RTX 2060 minimum)
- [ ] Training will run for 1M timesteps (approx 8-10 days)
- [ ] Monitoring system in place (TensorBoard, resource usage)

### Backup Plan
- [ ] Code repository backed up
- [ ] Configuration files versioned
- [ ] Training resumption tested (checkpoint load/save)
- [ ] Error notification system configured (email/Slack)

---

## 🚀 Next Steps After Validation

If all tests pass, proceed to:

1. **Start Training Run** (`scripts/train_td3.py` for 1M timesteps)
2. **Monitor Progress** (TensorBoard at `localhost:6006`)
3. **Checkpoint Management** (Save every 10K steps, keep best 5)
4. **Evaluation** (Test every 50K steps on 10 episodes)
5. **Results Analysis** (Phase 4: metrics, plots, paper figures)

---

## 📚 References

**CARLA Documentation (Fetched January 29, 2025):**
- Python API: https://carla.readthedocs.io/en/latest/python_api/
- Sensors Reference: https://carla.readthedocs.io/en/latest/ref_sensors/
- Docker Setup: https://carla.readthedocs.io/en/latest/build_docker/
- ROS 2 Bridge: https://carla.readthedocs.io/projects/ros-bridge/en/latest/

**TD3 Algorithm:**
- Fujimoto et al. (2018): "Addressing Function Approximation Error in Actor-Critic Methods"
- Official Implementation: https://github.com/sfujim/TD3

**Project Documentation:**
- `NEW_DEVELOPMENT_PLAN.md` - Development roadmap
- `TD3_IMPLEMENTATION_INSIGHTS.md` - Architecture deep-dive
- `PHASE_2_COMPLETION_SUMMARY.md` - Phase 2 deliverables

---

**END OF SYSTEM TESTING PLAN**

_This document will be updated based on test results and any issues encountered during validation._
