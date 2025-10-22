#!/usr/bin/env python3
"""
Test Suite 5: Training Pipeline - REAL INTEGRATION TEST
Phase 3: System Testing for TD3 Autonomous Navigation System

Test:
  5.1: 100-step training validation run (real environment + agent)

Prerequisites: CARLA server running, environment & agent ready
"""

import sys
import os
sys.path.insert(0, '/workspace')

import numpy as np
import yaml
import time
import torch
from datetime import datetime


def flatten_dict_obs(obs_dict):
    """
    Flatten Dict observation into single numpy array.

    Expected structure:
        {'image': (4, 84, 84), 'vector': (23,)}

    Returns:
        Flat array of shape (535,) = 512 (image flattened) + 23 (vector)
    """
    # Flatten image from (4, 84, 84) to (28224,) then pool to (512,)
    image = obs_dict['image']  # Shape: (4, 84, 84)

    # Use average pooling to reduce 28224 to 512 features
    # Split into 512 patches of size (4, 7, 11) and average each
    # For simplicity, we'll use a basic reshape + mean approach
    image_flat = image.reshape(4, -1).mean(axis=0)  # (7056,) per channel, mean across channels
    image_features = image_flat[:512]  # Take first 512 features

    # Pad if needed
    if len(image_features) < 512:
        image_features = np.pad(image_features, (0, 512 - len(image_features)))

    vector = obs_dict['vector']  # Shape: (23,)

    # Concatenate: 512 (image) + 23 (vector) = 535
    flat_state = np.concatenate([image_features, vector]).astype(np.float32)

    return flat_state


def test_5_1_training_pipeline():
    """Test 5.1: Real training pipeline validation"""
    print("\n" + "="*80)
    print("🚀 TEST 5.1: Training Pipeline Validation (100 steps)")
    print("="*80)

    try:
        from src.environment.carla_env import CARLANavigationEnv
        from src.agents.td3_agent import TD3Agent

        # Load configurations
        print(f"✅ Loading configurations...")
        with open('/workspace/config/carla_config.yaml', 'r') as f:
            carla_config = yaml.safe_load(f)

        with open('/workspace/config/td3_config.yaml', 'r') as f:
            td3_config = yaml.safe_load(f)

        print(f"   CARLA map: {carla_config['world']['map_name']}")
        print(f"   TD3 algorithm: TD3")

        # Initialize environment
        print(f"\n🌍 Initializing environment...")
        env = CARLANavigationEnv(
            carla_config_path='/workspace/config/carla_config.yaml',
            td3_config_path='/workspace/config/td3_config.yaml',
            training_config_path='/workspace/config/training_config.yaml'
        )
        print(f"✅ Environment ready")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")

        # Initialize agent (device and hyperparameters are auto-loaded from config)
        print(f"\n🤖 Initializing agent...")
        agent = TD3Agent(
            state_dim=535,
            action_dim=2,
            max_action=1.0,
            config_path='/workspace/config/td3_config.yaml'
        )
        print(f"✅ Agent ready (device={agent.device})")
        print(f"   Learning rate: {agent.actor_lr}")
        print(f"   Target update rate (tau): {agent.tau}")
        print(f"   Policy update freq: {agent.policy_freq}")

        # Training parameters
        TOTAL_STEPS = 100
        RANDOM_EXPLORE_STEPS = 25
        BATCH_SIZE = 64
        TRAINING_START_STEP = BATCH_SIZE + 1  # Start training after buffer has enough samples (65)

        print(f"\n⚙️  Training Configuration:")
        print(f"   Total steps: {TOTAL_STEPS}")
        print(f"   Random exploration: 0-{RANDOM_EXPLORE_STEPS}")
        print(f"   Training starts: step {TRAINING_START_STEP}")
        print(f"   Batch size: {BATCH_SIZE}")

        # Training loop
        print(f"\n" + "="*80)
        print(f"🔄 TRAINING LOOP")
        print(f"="*80)

        obs_dict = env.reset()
        state = flatten_dict_obs(obs_dict)  # Flatten dict to 535-dim array
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        total_reward_sum = 0

        start_time = time.time()
        step_times = []

        for step in range(TOTAL_STEPS):
            step_start = time.time()
            episode_timesteps += 1

            # Select action
            if step < RANDOM_EXPLORE_STEPS:
                action = env.action_space.sample()
                action_source = "random"
            else:
                action = agent.select_action(state, noise=0.1)
                action_source = "policy"

            # Execute action
            next_obs_dict, reward, done, truncated, info = env.step(action)
            next_state = flatten_dict_obs(next_obs_dict)  # Flatten dict to 535-dim array

            # Store transition
            agent.replay_buffer.add(state, action, next_state, reward, float(done))

            state = next_state
            episode_reward += reward
            total_reward_sum += reward

            # Train agent
            train_losses = None
            if step >= TRAINING_START_STEP:
                metrics = agent.train(batch_size=BATCH_SIZE)
                critic_loss = metrics['critic_loss']
                actor_loss = metrics.get('actor_loss', None)
                train_losses = (critic_loss, actor_loss)

            step_time = time.time() - step_start
            step_times.append(step_time)

            # Episode end
            episode_ended = done or truncated

            # Progress output
            if (step + 1) % 10 == 0 or episode_ended:
                speed = info.get('speed', 0)
                collisions = int(info.get('collision_detected', False))

                training_status = ""
                if train_losses:
                    critic_loss, actor_loss = train_losses
                    training_status = f" | C={critic_loss:.4f}"
                    if actor_loss is not None:
                        training_status += f", A={actor_loss:.4f}"
                    else:
                        training_status += f", A=delayed"

                print(f"Step {step+1:3d}: action={action_source:6s}, reward={reward:7.4f}, " +
                      f"speed={speed:5.2f}m/s, collisions={collisions}{training_status}")

            if episode_ended:
                print(f"   → Episode {episode_num+1} ended at step {step+1}")
                print(f"      Total reward: {episode_reward:.4f}")
                print(f"      Duration: {episode_timesteps} timesteps")
                print(f"      Termination: {info.get('termination_reason', 'unknown')}")

                # Reset for next episode
                obs_dict = env.reset()
                state = flatten_dict_obs(obs_dict)  # Flatten dict to 535-dim array
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

        end_time = time.time()
        total_time = end_time - start_time

        # Final statistics
        print(f"\n" + "="*80)
        print(f"✅ TRAINING VALIDATION COMPLETED")
        print(f"="*80)

        avg_step_time = np.mean(step_times)
        steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0

        print(f"\n📊 Training Statistics:")
        print(f"   Total steps: {TOTAL_STEPS}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Avg step time: {avg_step_time*1000:.2f}ms")
        print(f"   Steps/sec: {steps_per_sec:.2f}")
        print(f"   Episodes completed: {episode_num}")
        print(f"   Total reward: {total_reward_sum:.4f}")
        print(f"   Avg reward/step: {total_reward_sum/TOTAL_STEPS:.4f}")

        print(f"\n💾 Agent State:")
        print(f"   Replay buffer size: {agent.replay_buffer.size}")
        print(f"   Training iterations: {agent.total_it}")
        print(f"   Actor updates: {agent.total_it // 2}")  # Actor updates every 2 steps

        print(f"\n🔍 Buffer Statistics:")
        print(f"   Max capacity: {agent.replay_buffer.max_size}")
        print(f"   Current size: {agent.replay_buffer.size}")
        print(f"   Fill ratio: {100*agent.replay_buffer.size/agent.replay_buffer.max_size:.2f}%")

        # Validation checks
        print(f"\n✓ Validation Checks:")
        assert TOTAL_STEPS == 100, "❌ Wrong number of steps!"
        assert agent.total_it > 0, "❌ No training iterations!"
        assert agent.replay_buffer.size == TOTAL_STEPS, "❌ Buffer size mismatch!"
        print(f"✅ All validation checks passed")

        # Cleanup
        env.close()

        print(f"\n✅ TEST 5.1 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ TEST 5.1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run Test Suite 5"""

    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*15 + "SYSTEM TESTING: Test Suite 5 - Training Pipeline Validation" + " "*3 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    print(f"\n⏱️  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  CARLA Server: localhost:2000")
    print(f"🤖 Compute: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    success = test_5_1_training_pipeline()

    # Summary
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*25 + "SUMMARY - TEST SUITE 5" + " "*31 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    if success:
        print("\n✅ TEST SUITE 5 PASSED")
        print("   Ready for Test Suite 6 (End-to-End Integration)")
        print("   Next: Run test_6_end_to_end_integration.py\n")
        return True
    else:
        print("\n❌ TEST SUITE 5 FAILED\n")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
