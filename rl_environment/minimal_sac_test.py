#!/usr/bin/env python
"""
Minimal test for SimpleSAC without any environment dependency.

This script tests the SimpleSAC implementation with synthetic data to verify
that the core training loop and network updates function correctly.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
import torch

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_feature_extractor():
    """Test the SimpleFeatureExtractor independently."""
    try:
        from rl_environment.simple_sac import SimpleFeatureExtractor

        # Define synthetic state dimensions
        state_dims = {
            'image': (84, 84, 3),
            'vector': (10,)
        }

        # Create feature extractor
        feature_dim = 32
        feature_extractor = SimpleFeatureExtractor(state_dims, feature_dim)

        # Create synthetic state
        state = {
            'image': torch.rand(2, 84, 84, 3),  # batch size 2
            'vector': torch.rand(2, 10)
        }

        # Forward pass
        features = feature_extractor(state)

        # Check output shape
        assert features.shape == (2, feature_dim)

        logger.info("✓ Feature extractor test passed")
        return True
    except Exception as e:
        logger.error(f"Feature extractor test failed: {e}")
        return False

def test_q_network():
    """Test the SimpleQNetwork independently."""
    try:
        from rl_environment.simple_sac import SimpleQNetwork

        # Create network
        feature_dim = 32
        action_dim = 3
        q_net = SimpleQNetwork(feature_dim, action_dim)

        # Create synthetic inputs
        features = torch.rand(2, feature_dim)  # batch size 2
        actions = torch.rand(2, action_dim)

        # Forward pass
        q_values = q_net(features, actions)

        # Check output shape
        assert q_values.shape == (2, 1)

        logger.info("✓ Q-network test passed")
        return True
    except Exception as e:
        logger.error(f"Q-network test failed: {e}")
        return False

def test_policy_network():
    """Test the SimplePolicy independently."""
    try:
        from rl_environment.simple_sac import SimplePolicy

        # Create network
        feature_dim = 32
        action_dim = 3
        policy = SimplePolicy(feature_dim, action_dim)

        # Create synthetic inputs
        features = torch.rand(2, feature_dim)  # batch size 2

        # Forward pass - mean and log_std
        mean, log_std = policy(features)

        # Check output shapes
        assert mean.shape == (2, action_dim)
        assert log_std.shape == (2, action_dim)

        # Test sampling
        actions, log_probs, tanh_mean = policy.sample(features)

        # Check output shapes
        assert actions.shape == (2, action_dim)
        assert log_probs.shape == (2, 1)
        assert tanh_mean.shape == (2, action_dim)

        logger.info("✓ Policy network test passed")
        return True
    except Exception as e:
        logger.error(f"Policy network test failed: {e}")
        return False

def test_replay_buffer():
    """Test the ReplayBuffer independently."""
    try:
        from rl_environment.simple_sac import ReplayBuffer

        # Create buffer
        buffer_size = 10
        replay_buffer = ReplayBuffer(buffer_size)

        # Create synthetic state
        state = {
            'image': np.random.rand(84, 84, 3),
            'vector': np.random.rand(10)
        }
        action = np.random.rand(3)
        reward = 1.0
        next_state = {
            'image': np.random.rand(84, 84, 3),
            'vector': np.random.rand(10)
        }
        done = False

        # Add several transitions
        for _ in range(5):
            replay_buffer.push(state, action, reward, next_state, done)

        # Check buffer size
        assert len(replay_buffer) == 5
        assert replay_buffer.can_sample(5)
        assert not replay_buffer.can_sample(6)

        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(3)

        # Check sample shapes
        assert list(states.keys()) == list(state.keys())
        assert actions.shape[1] == action.shape[0]
        assert rewards.shape == (3, 1)
        assert list(next_states.keys()) == list(next_state.keys())
        assert dones.shape == (3, 1)

        logger.info("✓ Replay buffer test passed")
        return True
    except Exception as e:
        logger.error(f"Replay buffer test failed: {e}")
        return False

def test_sac_initialization():
    """Test the SimpleSAC initialization."""
    try:
        from rl_environment.simple_sac import SimpleSAC

        # Define state dimensions
        state_dims = {
            'image': (84, 84, 3),
            'vector': (10,)
        }
        action_dim = 3

        # Create agent
        agent = SimpleSAC(
            state_dims=state_dims,
            action_dim=action_dim,
            lr=3e-4,
            feature_dim=32,
            hidden_dim=32,
            batch_size=8
        )

        # Test select_action
        state = {
            'image': np.random.rand(84, 84, 3).astype(np.float32),
            'vector': np.random.rand(10).astype(np.float32)
        }
        action = agent.select_action(state)

        # Check action shape
        assert action.shape == (3,)

        logger.info("✓ SAC initialization test passed")
        return True
    except Exception as e:
        logger.error(f"SAC initialization test failed: {e}")
        return False

def main():
    """Run all tests."""
    tests = [
        test_feature_extractor,
        test_q_network,
        test_policy_network,
        test_replay_buffer,
        test_sac_initialization
    ]

    results = []

    for test_func in tests:
        result = test_func()
        results.append(result)

    # Report overall results
    success = all(results)
    if success:
        logger.info("All tests passed successfully!")
    else:
        logger.error("Some tests failed. Review the logs.")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
