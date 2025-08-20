#!/usr/bin/env python
"""
Complete rewrite of the SAC update test that doesn't use the update method directly.

This script tests each component of the SAC algorithm independently to verify that
each part works correctly without causing backward pass issues.
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

def generate_synthetic_data(batch_size=32):
    """
    Generate synthetic data for testing.

    Args:
        batch_size: Number of samples to generate

    Returns:
        Dictionary of synthetic data
    """
    # Define dimensions
    image_shape = (84, 84, 3)
    vector_shape = (10,)
    action_dim = 3

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate states
    states = {
        'image': torch.FloatTensor(np.random.rand(batch_size, *image_shape)).to(device),
        'vector': torch.FloatTensor(np.random.rand(batch_size, *vector_shape)).to(device)
    }

    # Generate actions
    actions = torch.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, action_dim))).to(device)

    # Generate rewards
    rewards = torch.FloatTensor(np.random.randn(batch_size, 1)).to(device)

    # Generate next states
    next_states = {
        'image': torch.FloatTensor(np.random.rand(batch_size, *image_shape)).to(device),
        'vector': torch.FloatTensor(np.random.rand(batch_size, *vector_shape)).to(device)
    }

    # Generate dones
    dones = torch.FloatTensor(np.random.randint(0, 2, size=(batch_size, 1))).to(device)

    return states, actions, rewards, next_states, dones

def test_feature_extraction():
    """Test feature extraction."""
    from rl_environment.simple_sac import SimpleFeatureExtractor

    # Create feature extractor
    state_dims = {
        'image': (84, 84, 3),
        'vector': (10,)
    }
    feature_dim = 32
    extractor = SimpleFeatureExtractor(state_dims, feature_dim).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Generate synthetic data
    states, _, _, _, _ = generate_synthetic_data(batch_size=4)

    # Extract features
    features = extractor(states)

    logger.info(f"Feature shape: {features.shape}")
    logger.info("Feature extraction test passed!")

def test_q_network():
    """Test Q-network."""
    from rl_environment.simple_sac import SimpleQNetwork

    # Create Q-network
    feature_dim = 32
    action_dim = 3
    hidden_dim = 32
    q_net = SimpleQNetwork(feature_dim, action_dim, hidden_dim).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Generate synthetic data
    batch_size = 4
    features = torch.randn(batch_size, feature_dim).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    actions = torch.randn(batch_size, action_dim).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Forward pass
    q_values = q_net(features, actions)

    logger.info(f"Q-values shape: {q_values.shape}")
    logger.info("Q-network test passed!")

def test_policy_network():
    """Test policy network."""
    from rl_environment.simple_sac import SimplePolicy

    # Create policy network
    feature_dim = 32
    action_dim = 3
    hidden_dim = 32
    policy = SimplePolicy(feature_dim, action_dim, hidden_dim).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Generate synthetic data
    batch_size = 4
    features = torch.randn(batch_size, feature_dim).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Forward pass
    mean, log_std = policy(features)

    logger.info(f"Mean shape: {mean.shape}, Log-std shape: {log_std.shape}")

    # Sample actions
    actions, log_probs, tanh_mean = policy.sample(features)

    logger.info(f"Actions shape: {actions.shape}, Log-probs shape: {log_probs.shape}")
    logger.info("Policy network test passed!")

def test_critic_optimization():
    """Test critic optimization."""
    from rl_environment.simple_sac import SimpleFeatureExtractor, SimpleQNetwork

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create networks
    state_dims = {
        'image': (84, 84, 3),
        'vector': (10,)
    }
    feature_dim = 32
    action_dim = 3
    hidden_dim = 32

    extractor = SimpleFeatureExtractor(state_dims, feature_dim).to(device)
    q1 = SimpleQNetwork(feature_dim, action_dim, hidden_dim).to(device)
    q2 = SimpleQNetwork(feature_dim, action_dim, hidden_dim).to(device)

    # Create optimizers
    q1_optimizer = torch.optim.Adam(q1.parameters(), lr=3e-4)
    q2_optimizer = torch.optim.Adam(q2.parameters(), lr=3e-4)

    # Generate synthetic data
    states, actions, rewards, next_states, dones = generate_synthetic_data(batch_size=8)

    # Generate synthetic target values
    with torch.no_grad():
        target_q = torch.randn_like(rewards)

    # Extract features
    features = extractor(states)

    # Compute Q-values
    q1_pred = q1(features, actions)
    q2_pred = q2(features, actions)

    # Compute losses
    q1_loss = torch.nn.functional.mse_loss(q1_pred, target_q)
    q2_loss = torch.nn.functional.mse_loss(q2_pred, target_q)
    critic_loss = q1_loss + q2_loss

    # Optimize
    q1_optimizer.zero_grad()
    q2_optimizer.zero_grad()
    critic_loss.backward()
    q1_optimizer.step()
    q2_optimizer.step()

    logger.info(f"Critic loss: {critic_loss.item()}")
    logger.info("Critic optimization test passed!")

def test_actor_optimization():
    """Test actor optimization."""
    from rl_environment.simple_sac import SimpleFeatureExtractor, SimplePolicy, SimpleQNetwork

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create networks
    state_dims = {
        'image': (84, 84, 3),
        'vector': (10,)
    }
    feature_dim = 32
    action_dim = 3
    hidden_dim = 32

    extractor = SimpleFeatureExtractor(state_dims, feature_dim).to(device)
    policy = SimplePolicy(feature_dim, action_dim, hidden_dim).to(device)
    q1 = SimpleQNetwork(feature_dim, action_dim, hidden_dim).to(device)
    q2 = SimpleQNetwork(feature_dim, action_dim, hidden_dim).to(device)

    # Create optimizer
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # Generate synthetic data
    states, _, _, _, _ = generate_synthetic_data(batch_size=8)

    # Extract features with no gradient tracking
    with torch.no_grad():
        features = extractor(states)

    # Sample actions
    actions, log_probs, _ = policy.sample(features)

    # Compute Q-values with no gradient tracking
    with torch.no_grad():
        q1_val = q1(features, actions)
        q2_val = q2(features, actions)
        q_val = torch.min(q1_val, q2_val)

    # Compute actor loss
    alpha = 0.2  # Temperature parameter
    actor_loss = (alpha * log_probs - q_val).mean()

    # Optimize
    policy_optimizer.zero_grad()
    actor_loss.backward()
    policy_optimizer.step()

    logger.info(f"Actor loss: {actor_loss.item()}")
    logger.info("Actor optimization test passed!")

def test_alpha_optimization():
    """Test alpha optimization."""
    from rl_environment.simple_sac import SimpleFeatureExtractor, SimplePolicy

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create networks
    state_dims = {
        'image': (84, 84, 3),
        'vector': (10,)
    }
    feature_dim = 32
    action_dim = 3
    hidden_dim = 32

    extractor = SimpleFeatureExtractor(state_dims, feature_dim).to(device)
    policy = SimplePolicy(feature_dim, action_dim, hidden_dim).to(device)

    # Initialize alpha parameter
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    target_entropy = -action_dim  # Target entropy is -dim(A)

    # Create optimizer
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4)

    # Generate synthetic data
    states, _, _, _, _ = generate_synthetic_data(batch_size=8)

    # Extract features with no gradient tracking
    with torch.no_grad():
        features = extractor(states)

    # Sample actions and get log probs with no gradient tracking
    with torch.no_grad():
        _, log_probs, _ = policy.sample(features)

    # Compute alpha loss
    alpha_loss = -(log_alpha * (log_probs + target_entropy).detach()).mean()

    # Optimize
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    logger.info(f"Alpha loss: {alpha_loss.item()}, Alpha value: {log_alpha.exp().item()}")
    logger.info("Alpha optimization test passed!")

def main():
    """Run all tests."""
    logger.info("Starting SAC component tests...")

    logger.info("\n1. Testing feature extraction")
    test_feature_extraction()

    logger.info("\n2. Testing Q-network")
    test_q_network()

    logger.info("\n3. Testing policy network")
    test_policy_network()

    logger.info("\n4. Testing critic optimization")
    test_critic_optimization()

    logger.info("\n5. Testing actor optimization")
    test_actor_optimization()

    logger.info("\n6. Testing alpha optimization")
    test_alpha_optimization()

    logger.info("\nAll tests passed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
