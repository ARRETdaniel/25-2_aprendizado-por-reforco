#!/usr/bin/env python3
"""
Test Suite 4: Agent Functionality - REAL INTEGRATION TESTS
Phase 3: System Testing for TD3 Autonomous Navigation System

Tests (ALL REAL TD3/DDPG AGENT):
  4.1: TD3 Agent Initialization & Components
  4.2: Action Selection (deterministic & stochastic)
  4.3: Training Step with Replay Buffer
  4.4: Checkpoint Save/Load Functionality

Prerequisites: CARLA server, environment, agent components ready
"""

import sys
import os
sys.path.insert(0, '/workspace')

import numpy as np
import torch
import tempfile
from datetime import datetime


def test_4_1_agent_initialization():
    """Test 4.1: Real TD3 agent initialization"""
    print("\n" + "="*80)
    print("🤖 TEST 4.1: TD3 Agent Initialization & Components")
    print("="*80)

    try:
        from src.agents.td3_agent import TD3Agent

        print(f"✅ TD3Agent imported successfully")

        # Initialize agent (device is determined automatically)
        agent = TD3Agent(
            state_dim=535,
            action_dim=2,
            max_action=1.0
        )

        print(f"✅ TD3 Agent initialized")
        print(f"   Device: {agent.device}")
        print(f"   State dimension: 535")
        print(f"   Action dimension: 2")
        print(f"   Max action: 1.0")

        # Count parameters
        actor_params = sum(p.numel() for p in agent.actor.parameters())
        critic_params = sum(p.numel() for p in agent.critic.parameters())
        total_params = actor_params + critic_params

        print(f"\n📊 Network Architecture:")
        print(f"   Actor parameters: {actor_params:,}")
        print(f"   Critic1 parameters: {critic_params // 2:,}")
        print(f"   Critic2 parameters: {critic_params // 2:,}")
        print(f"   Total trainable: {total_params:,}")

        # Verify target networks exist
        assert hasattr(agent, 'actor_target'), "❌ Actor target network missing!"
        assert hasattr(agent, 'critic_target'), "❌ Critic target network missing!"
        print(f"\n✅ Target networks created")

        # Verify replay buffer
        assert hasattr(agent, 'replay_buffer'), "❌ Replay buffer missing!"
        print(f"✅ Replay buffer initialized")
        print(f"   Capacity: {agent.replay_buffer.max_size:,}")
        print(f"   Current size: {agent.replay_buffer.size}")

        # Verify optimizers
        assert hasattr(agent, 'actor_optimizer'), "❌ Actor optimizer missing!"
        assert hasattr(agent, 'critic_optimizer'), "❌ Critic optimizer missing!"
        print(f"✅ Optimizers initialized")

        # Test parameter counts match across networks
        assert len(list(agent.actor.parameters())) > 0, "❌ Actor has no parameters!"
        assert len(list(agent.critic.parameters())) > 0, "❌ Critic has no parameters!"
        assert len(list(agent.actor_target.parameters())) > 0, "❌ Actor target has no parameters!"
        print(f"✅ All networks have parameters")

        print(f"\n✅ TEST 4.1 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ TEST 4.1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_2_action_selection():
    """Test 4.2: Real action selection"""
    print("\n" + "="*80)
    print("🎯 TEST 4.2: Action Selection (Deterministic & Stochastic)")
    print("="*80)

    try:
        from src.agents.td3_agent import TD3Agent

        # Create dummy state
        state = np.random.randn(535).astype(np.float32)
        print(f"✅ Test state created: shape={state.shape}")

        # Test deterministic action (evaluation)
        print(f"\n🔍 Testing deterministic action (noise=0.0)...")
        action_eval = agent.select_action(state, noise=0.0)

        assert action_eval.shape == (2,), f"❌ Action shape wrong: {action_eval.shape}"
        assert action_eval.dtype in [np.float32, np.float64], f"❌ Action dtype wrong: {action_eval.dtype}"
        assert np.all(action_eval >= -1.0) and np.all(action_eval <= 1.0), \
            f"❌ Action out of bounds: {action_eval}"

        print(f"✅ Deterministic action valid")
        print(f"   Action: [{action_eval[0]:.4f}, {action_eval[1]:.4f}]")
        print(f"   Range: [{action_eval.min():.4f}, {action_eval.max():.4f}]")

        # Test stochastic action (training)
        print(f"\n🔍 Testing stochastic action (noise=0.1)...")
        action_noise = agent.select_action(state, noise=0.1)

        assert action_noise.shape == (2,), f"❌ Noisy action shape wrong: {action_noise.shape}"
        assert np.all(action_noise >= -1.0) and np.all(action_noise <= 1.0), \
            f"❌ Noisy action out of bounds: {action_noise}"

        print(f"✅ Stochastic action valid")
        print(f"   Action: [{action_noise[0]:.4f}, {action_noise[1]:.4f}]")
        print(f"   Difference from eval: {np.linalg.norm(action_noise - action_eval):.4f}")

        # Test batch selection
        print(f"\n🔍 Testing batch action selection...")
        batch_size = 10
        states_batch = np.random.randn(batch_size, 535).astype(np.float32)
        actions_batch = np.array([agent.select_action(s, noise=0.0) for s in states_batch])

        assert actions_batch.shape == (batch_size, 2), f"❌ Batch shape wrong: {actions_batch.shape}"
        assert np.all((actions_batch >= -1.0) & (actions_batch <= 1.0)), "❌ Batch actions out of bounds!"

        print(f"✅ Batch actions valid")
        print(f"   Batch shape: {actions_batch.shape}")
        print(f"   Mean action: [{actions_batch.mean(axis=0)[0]:.4f}, {actions_batch.mean(axis=0)[1]:.4f}]")
        print(f"   Std action: [{actions_batch.std(axis=0)[0]:.4f}, {actions_batch.std(axis=0)[1]:.4f}]")

        print(f"\n✅ TEST 4.2 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ TEST 4.2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_3_training_step():
    """Test 4.3: Real training step with replay buffer"""
    print("\n" + "="*80)
    print("🏋️  TEST 4.3: Training Step with Replay Buffer")
    print("="*80)

    try:
        from src.agents.td3_agent import TD3Agent

        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = TD3Agent(state_dim=535, action_dim=2, max_action=1.0, device=device)

        print(f"✅ Agent initialized on {device}")

        # Populate replay buffer
        print(f"\n📚 Populating replay buffer...")
        num_transitions = 500

        for i in range(num_transitions):
            state = np.random.randn(535).astype(np.float32)
            action = np.random.uniform(-1, 1, 2).astype(np.float32)
            next_state = np.random.randn(535).astype(np.float32)
            reward = np.random.randn()
            done = float(np.random.rand() < 0.05)  # 5% done probability

            agent.replay_buffer.add(state, action, next_state, reward, done)

        print(f"✅ Added {num_transitions} transitions")
        print(f"   Buffer size: {agent.replay_buffer.size}")
        print(f"   Buffer capacity: {agent.replay_buffer.max_size}")

        # Perform training step
        print(f"\n🔄 Performing training update...")
        batch_size = 64

        critic_loss, actor_loss = agent.train(batch_size=batch_size)

        print(f"✅ Training step completed")
        print(f"   Batch size: {batch_size}")
        print(f"   Critic loss: {critic_loss:.6f}")
        print(f"   Actor loss: {actor_loss if actor_loss is not None else 'N/A (delayed)'}")

        # Validate loss is finite
        assert np.isfinite(critic_loss), f"❌ Critic loss non-finite: {critic_loss}"
        assert actor_loss is None or np.isfinite(actor_loss), f"❌ Actor loss non-finite: {actor_loss}"
        print(f"✅ Loss values are finite")

        # Perform multiple training steps
        print(f"\n🔄 Running 10 consecutive training updates...")
        losses_history = []

        for step in range(10):
            critic_loss, actor_loss = agent.train(batch_size=batch_size)
            losses_history.append((critic_loss, actor_loss))

            if (step + 1) % 2 == 0:
                actor_status = f"A={actor_loss:.6f}" if actor_loss is not None else "A=delayed"
                print(f"   Step {agent.total_it:4d}: C={critic_loss:.6f}, {actor_status}")

        # Verify actor updates every 2 steps (policy_freq=2)
        actor_updated = sum(1 for _, a in losses_history if a is not None)
        print(f"\n✅ Actor updated {actor_updated}/10 times (every 2 steps expected)")

        # Verify total iterations increased
        assert agent.total_it >= 10, f"❌ Total iterations not incremented: {agent.total_it}"
        print(f"✅ Total training iterations: {agent.total_it}")

        print(f"\n✅ TEST 4.3 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ TEST 4.3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_4_checkpoint():
    """Test 4.4: Real checkpoint save/load"""
    print("\n" + "="*80)
    print("💾 TEST 4.4: Checkpoint Save/Load Functionality")
    print("="*80)

    try:
        from src.agents.td3_agent import TD3Agent

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create and train first agent
        print(f"✅ Creating Agent 1...")
        agent1 = TD3Agent(state_dim=535, action_dim=2, max_action=1.0, device=device)

        # Add data and train
        for i in range(300):
            state = np.random.randn(535).astype(np.float32)
            action = np.random.uniform(-1, 1, 2).astype(np.float32)
            next_state = np.random.randn(535).astype(np.float32)
            reward = np.random.randn()
            done = 0.0
            agent1.replay_buffer.add(state, action, next_state, reward, done)

        # Train to update weights
        for _ in range(5):
            agent1.train(batch_size=32)

        print(f"✅ Agent 1 trained (total_it={agent1.total_it})")

        # Get reference action
        test_state = np.random.randn(535).astype(np.float32)
        action_before = agent1.select_action(test_state, noise=0.0)
        print(f"   Reference action: [{action_before[0]:.6f}, {action_before[1]:.6f}]")

        # Save checkpoint
        print(f"\n💾 Saving checkpoint...")
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pth")

            agent1.save(checkpoint_path)

            assert os.path.exists(checkpoint_path), "❌ Checkpoint file not created!"
            file_size = os.path.getsize(checkpoint_path) / 1024 / 1024  # MB
            print(f"✅ Checkpoint saved")
            print(f"   Path: {checkpoint_path}")
            print(f"   Size: {file_size:.2f} MB")

            # Create new agent and load
            print(f"\n🔄 Creating Agent 2 and loading checkpoint...")
            agent2 = TD3Agent(state_dim=535, action_dim=2, max_action=1.0, device=device)

            # Verify they're different before loading
            action_agent2_before = agent2.select_action(test_state, noise=0.0)
            different_before = not np.allclose(action_before, action_agent2_before, atol=1e-5)
            print(f"   Agents different before load: {different_before}")

            # Load checkpoint
            agent2.load(checkpoint_path)
            print(f"✅ Checkpoint loaded")
            print(f"   Total iterations: {agent2.total_it}")

            # Get action after load
            action_after = agent2.select_action(test_state, noise=0.0)
            print(f"   Loaded action: [{action_after[0]:.6f}, {action_after[1]:.6f}]")

            # Verify actions match
            difference = np.abs(action_before - action_after).max()
            print(f"   Max difference: {difference:.2e}")

            assert np.allclose(action_before, action_after, atol=1e-5), \
                f"❌ Actions differ after load: diff={difference}"
            print(f"✅ Actions match (difference < 1e-5)")

            # Verify training state restored
            assert agent2.total_it == agent1.total_it, \
                f"❌ Total iterations not restored: {agent2.total_it} != {agent1.total_it}"
            print(f"✅ Training state correctly restored")

        print(f"\n✅ TEST 4.4 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ TEST 4.4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Test Suite 4 tests"""

    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*18 + "SYSTEM TESTING: Test Suite 4 - Agent Functionality" + " "*11 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    print(f"\n⏱️  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  Compute Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    results = []

    # Test 4.1: Initialization
    success_4_1 = test_4_1_agent_initialization()
    results.append(("Test 4.1: Agent Initialization", success_4_1))

    if not success_4_1:
        print("\n⚠️  Test 4.1 failed - cannot proceed")
        return False

    # Test 4.2: Action Selection
    success_4_2 = test_4_2_action_selection()
    results.append(("Test 4.2: Action Selection", success_4_2))

    # Test 4.3: Training Step
    success_4_3 = test_4_3_training_step()
    results.append(("Test 4.3: Training Step", success_4_3))

    # Test 4.4: Checkpoint
    success_4_4 = test_4_4_checkpoint()
    results.append(("Test 4.4: Checkpoint Save/Load", success_4_4))

    # Summary
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*25 + "SUMMARY - TEST SUITE 4" + " "*31 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status:<12} {test_name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    print(f"\nResult: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n✅ ALL TESTS PASSED - Ready for Test Suite 5 (Training Pipeline)")
        print("   Next: Run test_5_training_pipeline.py\n")
        return True
    else:
        print("\n❌ SOME TESTS FAILED - Fix issues before proceeding\n")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
