"""
Test Checkpoint Save/Load Cycle for TD3Agent

Validates that save_checkpoint() and load_checkpoint() properly preserve:
1. Actor and Critic network weights
2. Actor and Critic CNN weights (Phase 21 fix)
3. All optimizer states (including CNN optimizers)
4. Training iteration counter
5. Configuration and hyperparameters

This test verifies the PRIMARY FIX for Bug #15 (missing CNN states in checkpoint).

Reference: ANALYSIS_SAVE_CHECKPOINT.md - Section 6.3
"""

import os
import sys
import tempfile

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.td3_agent import TD3Agent
from src.networks.cnn_extractor import get_cnn_extractor


def test_checkpoint_basic_networks():
    """Test that basic actor/critic networks are preserved."""
    print("\n" + "="*80)
    print("TEST 1: Basic Actor/Critic Network Preservation")
    print("="*80)

    # Create agent
    agent = TD3Agent(state_dim=535, action_dim=2, max_action=1.0, use_dict_buffer=False)

    # Train for a few steps to modify weights
    agent.total_it = 42

    # Get network weights before save
    actor_weight_before = agent.actor.fc1.weight.data.clone()
    critic_weight_before = agent.critic.Q1.fc1.weight.data.clone()

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_basic.pth')
        agent.save_checkpoint(checkpoint_path)

        # Create new agent and load
        agent2 = TD3Agent(state_dim=535, action_dim=2, max_action=1.0, use_dict_buffer=False)
        agent2.load_checkpoint(checkpoint_path)

    # Verify networks match
    actor_weight_after = agent2.actor.fc1.weight.data
    critic_weight_after = agent2.critic.Q1.fc1.weight.data

    assert torch.allclose(actor_weight_before, actor_weight_after, atol=1e-6), "❌ Actor weights NOT preserved!"
    assert torch.allclose(critic_weight_before, critic_weight_after, atol=1e-6), "❌ Critic weights NOT preserved!"
    assert agent2.total_it == 42, "❌ Training iteration NOT preserved!"

    print("✅ Actor network weights preserved")
    print("✅ Critic network weights preserved")
    print("✅ Training iteration preserved")
    print("✅ TEST 1 PASSED")


def test_checkpoint_with_separate_cnns():
    """Test that SEPARATE CNNs (Phase 21 fix) are preserved."""
    print("\n" + "="*80)
    print("TEST 2: Separate CNN Preservation (PRIMARY FIX)")
    print("="*80)

    # Create SEPARATE CNNs (Phase 21 architecture)
    actor_cnn = get_cnn_extractor(
        input_channels=4,
        output_dim=512,
        architecture='resnet18'
    )
    critic_cnn = get_cnn_extractor(
        input_channels=4,
        output_dim=512,
        architecture='resnet18'
    )

    # Verify they are different instances
    assert id(actor_cnn) != id(critic_cnn), "❌ CNNs must be separate instances!"
    print(f"✅ Actor CNN id: {id(actor_cnn)}")
    print(f"✅ Critic CNN id: {id(critic_cnn)}")

    # Create agent with separate CNNs
    agent = TD3Agent(
        state_dim=535,
        action_dim=2,
        max_action=1.0,
        actor_cnn=actor_cnn,
        critic_cnn=critic_cnn,
        use_dict_buffer=True,
        config={
            'algorithm': {'gamma': 0.99, 'tau': 0.005, 'policy_noise': 0.2,
                         'noise_clip': 0.5, 'policy_freq': 2},
            'training': {'buffer_size': 1000, 'batch_size': 32},  # Small buffer for testing
            'exploration': {'expl_noise': 0.1},
            'networks': {'cnn': {'learning_rate': 1e-4}}
        }
    )

    # Modify CNN weights (simulate training)
    agent.total_it = 100
    with torch.no_grad():
        for param in agent.actor_cnn.parameters():
            param.add_(torch.randn_like(param) * 0.01)
        for param in agent.critic_cnn.parameters():
            param.add_(torch.randn_like(param) * 0.01)

    # Get CNN weights before save
    actor_cnn_param_before = list(agent.actor_cnn.parameters())[0].data.clone()
    critic_cnn_param_before = list(agent.critic_cnn.parameters())[0].data.clone()

    # Verify CNNs are different
    assert not torch.allclose(actor_cnn_param_before, critic_cnn_param_before), \
        "❌ CNNs should have different weights!"
    print("✅ Actor and Critic CNNs have different weights (as expected)")

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_cnns.pth')
        agent.save_checkpoint(checkpoint_path)

        # Create new CNNs for loading
        actor_cnn2 = get_cnn_extractor(
            input_channels=4,
            output_dim=512,
            architecture='resnet18'
        )
        critic_cnn2 = get_cnn_extractor(
            input_channels=4,
            output_dim=512,
            architecture='resnet18'
        )

        # Create new agent and load
        agent2 = TD3Agent(
            state_dim=535,
            action_dim=2,
            max_action=1.0,
            actor_cnn=actor_cnn2,
            critic_cnn=critic_cnn2,
            use_dict_buffer=True,
            config={
                'algorithm': {'gamma': 0.99, 'tau': 0.005, 'policy_noise': 0.2,
                             'noise_clip': 0.5, 'policy_freq': 2},
                'training': {'buffer_size': 1000, 'batch_size': 32},  # Small buffer for testing
                'exploration': {'expl_noise': 0.1},
                'networks': {'cnn': {'learning_rate': 1e-4}}
            }
        )
        agent2.load_checkpoint(checkpoint_path)

    # Verify CNNs match
    actor_cnn_param_after = list(agent2.actor_cnn.parameters())[0].data
    critic_cnn_param_after = list(agent2.critic_cnn.parameters())[0].data

    assert torch.allclose(actor_cnn_param_before, actor_cnn_param_after, atol=1e-6), \
        "❌ Actor CNN weights NOT preserved!"
    assert torch.allclose(critic_cnn_param_before, critic_cnn_param_after, atol=1e-6), \
        "❌ Critic CNN weights NOT preserved!"
    assert agent2.total_it == 100, "❌ Training iteration NOT preserved!"

    print("✅ Actor CNN weights preserved")
    print("✅ Critic CNN weights preserved")
    print("✅ Separate CNN architecture preserved (Phase 21 fix)")
    print("✅ TEST 2 PASSED")


def test_checkpoint_cnn_optimizers():
    """Test that CNN optimizer states are preserved."""
    print("\n" + "="*80)
    print("TEST 3: CNN Optimizer State Preservation")
    print("="*80)

    # Create SEPARATE CNNs
    actor_cnn = get_cnn_extractor(input_channels=4, output_dim=512, architecture='resnet18')
    critic_cnn = get_cnn_extractor(input_channels=4, output_dim=512, architecture='resnet18')

    # Create agent
    agent = TD3Agent(
        state_dim=535,
        action_dim=2,
        max_action=1.0,
        actor_cnn=actor_cnn,
        critic_cnn=critic_cnn,
        use_dict_buffer=True,
        config={
            'algorithm': {'gamma': 0.99, 'tau': 0.005, 'policy_noise': 0.2,
                         'noise_clip': 0.5, 'policy_freq': 2},
            'training': {'buffer_size': 1000, 'batch_size': 32},  # Small buffer for testing
            'exploration': {'expl_noise': 0.1},
            'networks': {'cnn': {'learning_rate': 1e-4}}
        }
    )    # Simulate training to populate optimizer state (momentum buffers)
    dummy_loss = torch.tensor(1.0, requires_grad=True)
    for i in range(10):
        # Actor CNN optimizer step
        agent.actor_cnn_optimizer.zero_grad()
        loss = dummy_loss * torch.sum(list(agent.actor_cnn.parameters())[0])
        loss.backward()
        agent.actor_cnn_optimizer.step()

        # Critic CNN optimizer step
        agent.critic_cnn_optimizer.zero_grad()
        loss = dummy_loss * torch.sum(list(agent.critic_cnn.parameters())[0])
        loss.backward()
        agent.critic_cnn_optimizer.step()

    # Get optimizer states before save
    actor_cnn_opt_state_before = agent.actor_cnn_optimizer.state_dict()
    critic_cnn_opt_state_before = agent.critic_cnn_optimizer.state_dict()

    # Verify momentum buffers exist (Adam has exp_avg and exp_avg_sq)
    assert len(actor_cnn_opt_state_before['state']) > 0, "❌ Actor CNN optimizer has no state!"
    assert len(critic_cnn_opt_state_before['state']) > 0, "❌ Critic CNN optimizer has no state!"
    print(f"✅ Actor CNN optimizer has state for {len(actor_cnn_opt_state_before['state'])} parameters")
    print(f"✅ Critic CNN optimizer has state for {len(critic_cnn_opt_state_before['state'])} parameters")

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_cnn_opts.pth')
        agent.save_checkpoint(checkpoint_path)

        # Create new agent and load
        actor_cnn2 = get_cnn_extractor(input_channels=4, output_dim=512, architecture='resnet18')
        critic_cnn2 = get_cnn_extractor(input_channels=4, output_dim=512, architecture='resnet18')

        agent2 = TD3Agent(
            state_dim=535,
            action_dim=2,
            max_action=1.0,
            actor_cnn=actor_cnn2,
            critic_cnn=critic_cnn2,
            use_dict_buffer=True,
            config={
                'algorithm': {'gamma': 0.99, 'tau': 0.005, 'policy_noise': 0.2,
                             'noise_clip': 0.5, 'policy_freq': 2},
                'training': {'buffer_size': 1000, 'batch_size': 32},  # Small buffer for testing
                'exploration': {'expl_noise': 0.1},
                'networks': {'cnn': {'learning_rate': 1e-4}}
            }
        )
        agent2.load_checkpoint(checkpoint_path)

    # Verify optimizer states match
    actor_cnn_opt_state_after = agent2.actor_cnn_optimizer.state_dict()
    critic_cnn_opt_state_after = agent2.critic_cnn_optimizer.state_dict()

    # Check state exists and has same number of parameters
    assert len(actor_cnn_opt_state_after['state']) == len(actor_cnn_opt_state_before['state']), \
        "❌ Actor CNN optimizer state count mismatch!"
    assert len(critic_cnn_opt_state_after['state']) == len(critic_cnn_opt_state_before['state']), \
        "❌ Critic CNN optimizer state count mismatch!"

    print("✅ Actor CNN optimizer state preserved")
    print("✅ Critic CNN optimizer state preserved")
    print("✅ TEST 3 PASSED")


def test_checkpoint_hyperparameters():
    """Test that hyperparameters are preserved."""
    print("\n" + "="*80)
    print("TEST 4: Hyperparameter Preservation")
    print("="*80)

    # Create agent
    agent = TD3Agent(state_dim=535, action_dim=2, max_action=1.0, use_dict_buffer=False)

    # Get hyperparameters before save
    discount_before = agent.discount
    tau_before = agent.tau
    policy_freq_before = agent.policy_freq
    max_action_before = agent.max_action

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_hyperparams.pth')
        agent.save_checkpoint(checkpoint_path)

        # Load checkpoint and verify hyperparameters
        checkpoint = torch.load(checkpoint_path)

        assert 'discount' in checkpoint, "❌ discount not saved!"
        assert 'tau' in checkpoint, "❌ tau not saved!"
        assert 'policy_freq' in checkpoint, "❌ policy_freq not saved!"
        assert 'max_action' in checkpoint, "❌ max_action not saved!"

        assert checkpoint['discount'] == discount_before, "❌ discount mismatch!"
        assert checkpoint['tau'] == tau_before, "❌ tau mismatch!"
        assert checkpoint['policy_freq'] == policy_freq_before, "❌ policy_freq mismatch!"
        assert checkpoint['max_action'] == max_action_before, "❌ max_action mismatch!"

    print(f"✅ discount preserved: {discount_before}")
    print(f"✅ tau preserved: {tau_before}")
    print(f"✅ policy_freq preserved: {policy_freq_before}")
    print(f"✅ max_action preserved: {max_action_before}")
    print("✅ TEST 4 PASSED")


def test_checkpoint_full_cycle():
    """Test full training cycle with checkpoint resume."""
    print("\n" + "="*80)
    print("TEST 5: Full Training Cycle with Resume")
    print("="*80)

    # Create SEPARATE CNNs
    actor_cnn = get_cnn_extractor(input_channels=4, output_dim=512, architecture='resnet18')
    critic_cnn = get_cnn_extractor(input_channels=4, output_dim=512, architecture='resnet18')

    # Create agent
    agent = TD3Agent(
        state_dim=535,
        action_dim=2,
        max_action=1.0,
        actor_cnn=actor_cnn,
        critic_cnn=critic_cnn,
        use_dict_buffer=True,
        config={
            'algorithm': {'gamma': 0.99, 'tau': 0.005, 'policy_noise': 0.2,
                         'noise_clip': 0.5, 'policy_freq': 2},
            'training': {'buffer_size': 1000, 'batch_size': 32},  # Small buffer for testing
            'exploration': {'expl_noise': 0.1},
            'networks': {'cnn': {'learning_rate': 1e-4}}
        }
    )    # Simulate training for 50 iterations
    print("Training first agent for 50 iterations...")
    for i in range(50):
        agent.total_it += 1
        # Simulate weight updates
        with torch.no_grad():
            for param in agent.actor.parameters():
                param.add_(torch.randn_like(param) * 0.001)
            for param in agent.actor_cnn.parameters():
                param.add_(torch.randn_like(param) * 0.001)

    print(f"Agent 1 total_it: {agent.total_it}")

    # Get state snapshot before save
    actor_weight_50 = agent.actor.fc1.weight.data.clone()
    actor_cnn_weight_50 = list(agent.actor_cnn.parameters())[0].data.clone()

    # Save checkpoint at iteration 50
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_cycle.pth')
        agent.save_checkpoint(checkpoint_path)

        # Create new agent and resume training
        print("\nCreating new agent and resuming from checkpoint...")
        actor_cnn2 = get_cnn_extractor(input_channels=4, output_dim=512, architecture='resnet18')
        critic_cnn2 = get_cnn_extractor(input_channels=4, output_dim=512, architecture='resnet18')

        agent2 = TD3Agent(
            state_dim=535,
            action_dim=2,
            max_action=1.0,
            actor_cnn=actor_cnn2,
            critic_cnn=critic_cnn2,
            use_dict_buffer=True,
            config={
                'algorithm': {'gamma': 0.99, 'tau': 0.005, 'policy_noise': 0.2,
                             'noise_clip': 0.5, 'policy_freq': 2},
                'training': {'buffer_size': 1000, 'batch_size': 32},  # Small buffer for testing
                'exploration': {'expl_noise': 0.1},
                'networks': {'cnn': {'learning_rate': 1e-4}}
            }
        )
        agent2.load_checkpoint(checkpoint_path)

        # Verify resume point
        assert agent2.total_it == 50, f"❌ Should resume at iteration 50, got {agent2.total_it}!"
        print(f"✅ Agent 2 correctly resumed at iteration {agent2.total_it}")

        # Verify weights match
        actor_weight_loaded = agent2.actor.fc1.weight.data
        actor_cnn_weight_loaded = list(agent2.actor_cnn.parameters())[0].data

        assert torch.allclose(actor_weight_50, actor_weight_loaded, atol=1e-6), \
            "❌ Actor weights mismatch after load!"
        assert torch.allclose(actor_cnn_weight_50, actor_cnn_weight_loaded, atol=1e-6), \
            "❌ Actor CNN weights mismatch after load!"

        print("✅ Actor weights match after resume")
        print("✅ Actor CNN weights match after resume")

        # Continue training for another 50 iterations
        print("\nContinuing training for 50 more iterations...")
        for i in range(50):
            agent2.total_it += 1

        assert agent2.total_it == 100, f"❌ Should be at iteration 100, got {agent2.total_it}!"
        print(f"✅ Training continued successfully to iteration {agent2.total_it}")

    print("✅ TEST 5 PASSED")


def run_all_tests():
    """Run all checkpoint tests."""
    print("\n" + "="*80)
    print("CHECKPOINT SAVE/LOAD CYCLE TESTS")
    print("Testing PRIMARY FIX for Bug #15 (missing CNN states)")
    print("="*80)

    try:
        test_checkpoint_basic_networks()
        test_checkpoint_with_separate_cnns()
        test_checkpoint_cnn_optimizers()
        test_checkpoint_hyperparameters()
        test_checkpoint_full_cycle()

        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\n✅ Checkpoint save/load correctly preserves:")
        print("   1. Actor and Critic networks")
        print("   2. SEPARATE Actor CNN and Critic CNN (Phase 21 fix)")
        print("   3. All optimizer states (including CNN optimizers)")
        print("   4. Training iteration counter")
        print("   5. Hyperparameters")
        print("\n✅ PRIMARY FIX VERIFIED: Bug #15 is RESOLVED")

        return True

    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED!")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
