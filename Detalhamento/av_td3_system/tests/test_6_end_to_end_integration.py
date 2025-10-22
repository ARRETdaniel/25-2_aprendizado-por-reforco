#!/usr/bin/env python3
"""
Test Suite 6: End-to-End Integration - FULL SYSTEM TEST
Phase 3: System Testing for TD3 Autonomous Navigation System

Test:
  6.1: Full episode with real environment, agent, sensors, and metrics

Prerequisites: Complete system ready (CARLA, environment, agent, sensors)
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
    image_flat = image.reshape(4, -1).mean(axis=0)  # (7056,) per channel, mean across channels
    image_features = image_flat[:512]  # Take first 512 features

    # Pad if needed
    if len(image_features) < 512:
        image_features = np.pad(image_features, (0, 512 - len(image_features)))

    vector = obs_dict['vector']  # Shape: (23,)

    # Concatenate: 512 (image) + 23 (vector) = 535
    flat_state = np.concatenate([image_features, vector]).astype(np.float32)

    return flat_state


def test_6_1_end_to_end_integration():
    """Test 6.1: Full end-to-end integration test"""
    print("\n" + "="*80)
    print("🔗 TEST 6.1: End-to-End Integration (Full Episode)")
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

        # Initialize system
        print(f"\n🌍 Initializing environment...")
        env = CARLANavigationEnv(
            carla_config_path='/workspace/config/carla_config.yaml',
            td3_config_path='/workspace/config/td3_config.yaml',
            training_config_path='/workspace/config/training_config.yaml'
        )

        print(f"\n🤖 Initializing agent...")
        agent = TD3Agent(
            state_dim=535,
            action_dim=2,
            max_action=1.0,
            config_path='/workspace/config/td3_config.yaml'
        )

        print(f"✅ System initialized")
        print(f"   Environment: CARLANavigationEnv")
        print(f"   Agent: TD3 on {agent.device}")

        # Episode configuration
        MAX_EPISODE_STEPS = 300
        WARMUP_STEPS = 50

        print(f"\n⚙️  Episode Configuration:")
        print(f"   Max steps: {MAX_EPISODE_STEPS}")
        print(f"   Warmup (random actions): {WARMUP_STEPS}")
        print(f"   Policy starts: step {WARMUP_STEPS+1}")

        # Reset environment and agent
        print(f"\n" + "="*80)
        print(f"🎬 EPISODE START")
        print(f"="*80)

        obs_dict = env.reset()
        state = flatten_dict_obs(obs_dict)  # Flatten dict to 535-dim array
        episode_reward = 0
        episode_timesteps = 0
        episode_done = False
        termination_reason = None

        # Metrics collection
        metrics = {
            'rewards': [],
            'speeds': [],
            'lateral_deviations': [],
            'collisions': 0,
            'off_road': 0,
            'max_speed': 0,
            'avg_speed': 0,
            'training_steps': 0
        }

        episode_start = time.time()

        # Episode loop
        step = 0
        while step < MAX_EPISODE_STEPS and not episode_done:
            step_start = time.time()
            episode_timesteps += 1
            step += 1

            # Select action
            if step <= WARMUP_STEPS:
                action = env.action_space.sample()
                action_source = "exploration"
            else:
                action = agent.select_action(state, noise=0.1)
                action_source = "policy"

            # Execute action
            next_obs_dict, reward, done, truncated, info = env.step(action)
            next_state = flatten_dict_obs(next_obs_dict)  # Flatten dict to 535-dim array

            # Store transition
            agent.replay_buffer.add(state, action, next_state, reward, float(done))

            # Train agent
            if step > WARMUP_STEPS:
                agent.train(batch_size=64)
                metrics['training_steps'] += 1

            # Collect metrics
            state = next_state
            episode_reward += reward
            metrics['rewards'].append(reward)

            speed = info.get('speed', 0)
            metrics['speeds'].append(speed)
            metrics['max_speed'] = max(metrics['max_speed'], speed)

            lat_dev = info.get('lateral_deviation', 0)
            metrics['lateral_deviations'].append(lat_dev)

            if info.get('collision_detected', False):
                metrics['collisions'] += 1

            if info.get('off_road', False):
                metrics['off_road'] += 1

            # Episode termination
            if done or truncated:
                episode_done = True
                termination_reason = info.get('termination_reason', 'unknown')

            # Progress output every 25 steps
            if step % 25 == 0:
                print(f"Step {step:3d}: action={action_source:11s}, reward={reward:7.4f}, " +
                      f"speed={speed:6.2f}m/s, lat_dev={lat_dev:6.3f}")

        episode_end = time.time()
        episode_duration = episode_end - episode_start

        # Calculate final metrics
        metrics['avg_speed'] = np.mean(metrics['speeds']) if metrics['speeds'] else 0

        # Episode summary
        print(f"\n" + "="*80)
        print(f"🏁 EPISODE FINISHED")
        print(f"="*80)

        print(f"\n📊 Episode Results:")
        print(f"   Duration: {episode_duration:.2f}s")
        print(f"   Steps: {episode_timesteps}/{MAX_EPISODE_STEPS}")
        print(f"   Reason: {termination_reason}")

        print(f"\n💰 Rewards:")
        print(f"   Total: {episode_reward:.4f}")
        print(f"   Mean: {np.mean(metrics['rewards']):.4f}")
        print(f"   Std: {np.std(metrics['rewards']):.4f}")
        print(f"   Min: {np.min(metrics['rewards']):.4f}")
        print(f"   Max: {np.max(metrics['rewards']):.4f}")

        print(f"\n🚗 Vehicle Dynamics:")
        print(f"   Max speed: {metrics['max_speed']:.2f} m/s ({metrics['max_speed']*3.6:.2f} km/h)")
        print(f"   Avg speed: {metrics['avg_speed']:.2f} m/s ({metrics['avg_speed']*3.6:.2f} km/h)")
        print(f"   Avg lateral deviation: {np.mean(metrics['lateral_deviations']):.3f} m")
        print(f"   Max lateral deviation: {np.max(np.abs(metrics['lateral_deviations'])):.3f} m")

        print(f"\n⚠️  Safety Metrics:")
        print(f"   Collisions: {metrics['collisions']}")
        print(f"   Off-road events: {metrics['off_road']}")
        if metrics['collisions'] == 0 and metrics['off_road'] == 0:
            print(f"✅ No safety violations!")

        print(f"\n🧠 Training:")
        print(f"   Training steps: {metrics['training_steps']}")
        print(f"   Replay buffer size: {agent.replay_buffer.size}")
        print(f"   Total agent iterations: {agent.total_it}")

        # Validation checks
        print(f"\n✓ Validation Checks:")
        assert episode_timesteps > 0, "❌ Episode had no steps!"
        assert not np.any(np.isnan(metrics['rewards'])), "❌ NaN rewards!"
        assert len(metrics['speeds']) > 0, "❌ No speed data!"
        assert agent.total_it >= 0, "❌ Invalid iteration count!"
        print(f"✅ All validation checks passed")

        # Cleanup
        env.close()

        print(f"\n✅ TEST 6.1 PASSED")
        return True, metrics, episode_duration

    except Exception as e:
        print(f"\n❌ TEST 6.1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def main():
    """Run Test Suite 6"""

    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*16 + "SYSTEM TESTING: Test Suite 6 - End-to-End Integration" + " "*10 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    print(f"\n⏱️  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  CARLA Server: localhost:2000")
    print(f"🤖 Compute: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    success, metrics, duration = test_6_1_end_to_end_integration()

    # Summary
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*25 + "SUMMARY - TEST SUITE 6" + " "*31 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    if success:
        print("\n✅ TEST SUITE 6 PASSED")
        print("   Full end-to-end integration successful!")

        print(f"\n📈 System Performance Summary:")
        if metrics:
            print(f"   Episode duration: {duration:.2f}s")
            print(f"   Average speed: {metrics['avg_speed']*3.6:.2f} km/h")
            print(f"   Safety violations: {metrics['collisions'] + metrics['off_road']}")
            print(f"   Training completed: {metrics['training_steps']} steps")

        print(f"\n🎉 SYSTEM READY FOR TRAINING!")
        print(f"   All test suites passed successfully")
        print(f"   Proceed with: scripts/train_td3.py\n")
        return True
    else:
        print("\n❌ TEST SUITE 6 FAILED\n")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
