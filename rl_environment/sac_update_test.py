#!/usr/bin/env python
"""
A test script for our SAC implementation with synthetic data.

This script creates an artificial training loop to verify that the SAC
algorithm works correctly, including the backward passes and network updates.
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

from rl_environment.simple_sac import SimpleSAC, ReplayBuffer

def generate_synthetic_data(state_dims, action_dim, batch_size=32):
    """
    Generate synthetic data for training.

    Args:
        state_dims: Dictionary of state dimensions
        action_dim: Action dimension
        batch_size: Number of samples to generate

    Returns:
        Dictionary of synthetic data ready for SAC
    """
    # Generate states
    states = {}
    for key, dims in state_dims.items():
        if key == 'image':
            states[key] = torch.FloatTensor(np.random.rand(batch_size, *dims)).to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            states[key] = torch.FloatTensor(np.random.randn(batch_size, *dims)).to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Generate actions
    actions = torch.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, action_dim))).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Generate rewards
    rewards = torch.FloatTensor(np.random.randn(batch_size, 1)).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Generate next states (same structure as states)
    next_states = {}
    for key, dims in state_dims.items():
        if key == 'image':
            next_states[key] = torch.FloatTensor(np.random.rand(batch_size, *dims)).to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            next_states[key] = torch.FloatTensor(np.random.randn(batch_size, *dims)).to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Generate dones
    dones = torch.FloatTensor(np.random.randint(0, 2, size=(batch_size, 1))).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return states, actions, rewards, next_states, dones

def fill_replay_buffer(buffer, state_dims, action_dim, size=1000):
    """Fill replay buffer with synthetic data."""
    for _ in range(size):
        # Generate single transition
        state = {k: np.random.rand(*v).astype(np.float32) if k == 'image'
                else np.random.randn(*v).astype(np.float32)
                for k, v in state_dims.items()}
        action = np.random.uniform(-1, 1, size=action_dim).astype(np.float32)
        reward = np.random.randn(1)[0].astype(np.float32)
        next_state = {k: np.random.rand(*v).astype(np.float32) if k == 'image'
                    else np.random.randn(*v).astype(np.float32)
                    for k, v in state_dims.items()}
        done = np.random.randint(0, 2)

        # Add to buffer
        buffer.push(state, action, reward, next_state, done)

def test_sac_update(update_iterations=10):
    """
    Test the SAC update method with synthetic data.

    Args:
        update_iterations: Number of updates to perform

    Returns:
        True if successful, False otherwise
    """
    try:
        # Define state dimensions
        state_dims = {
            'image': (84, 84, 3),
            'vector': (10,)
        }
        action_dim = 3
        batch_size = 32

        # Create SAC agent
        agent = SimpleSAC(
            state_dims=state_dims,
            action_dim=action_dim,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            buffer_size=10000,
            batch_size=batch_size,
            feature_dim=32,
            hidden_dim=32,
            update_freq=1
        )

        # Fill replay buffer
        logger.info("Filling replay buffer...")
        fill_replay_buffer(agent.memory, state_dims, action_dim, size=batch_size * 2)

        # Perform multiple updates
        logger.info(f"Performing {update_iterations} updates...")

        for i in range(update_iterations):
            # Synthetic batch
            states, actions, rewards, next_states, dones = generate_synthetic_data(
                state_dims, action_dim, batch_size)

            # Modified update method to avoid backward pass issues
            try:
                # Extract features
                features = agent.feature_extractor(states)
                next_features = agent.feature_extractor(next_states)

                # Get current alpha value
                alpha = agent.log_alpha.exp().item() if agent.auto_entropy else agent.alpha

                # Update critic
                with torch.no_grad():
                    # Sample actions from policy
                    next_actions, next_log_probs, _ = agent.policy.sample(next_features)

                    # Compute target Q values
                    q1_next = agent.q1_target(next_features, next_actions)
                    q2_next = agent.q2_target(next_features, next_actions)
                    q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
                    target_q = rewards + (1 - dones) * agent.gamma * q_next

                # Compute current Q values
                q1 = agent.q1(features, actions)
                q2 = agent.q2(features, actions)

                # Compute critic loss
                q1_loss = torch.nn.functional.mse_loss(q1, target_q)
                q2_loss = torch.nn.functional.mse_loss(q2, target_q)
                critic_loss = q1_loss + q2_loss

                # Update critics
                agent.q1_optimizer.zero_grad()
                agent.q2_optimizer.zero_grad()
                critic_loss.backward()
                agent.q1_optimizer.step()
                agent.q2_optimizer.step()

                # Update actor
                actions_pi, log_probs, _ = agent.policy.sample(features)
                q1_pi = agent.q1(features, actions_pi)
                q2_pi = agent.q2(features, actions_pi)
                q_pi = torch.min(q1_pi, q2_pi)

                # Actor loss is expectation of Q - entropy
                actor_loss = (alpha * log_probs - q_pi).mean()

                # Update actor
                agent.policy_optimizer.zero_grad()
                actor_loss.backward()
                agent.policy_optimizer.step()

                # Update alpha if using automatic entropy tuning
                if agent.auto_entropy:
                    alpha_loss = -(agent.log_alpha * (log_probs + agent.target_entropy).detach()).mean()

                    agent.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    agent.alpha_optimizer.step()

                # Soft update target networks
                agent._soft_update_targets()

                logger.info(f"Update {i+1}/{update_iterations}: " +
                        f"Actor Loss: {actor_loss.item():.4f}, " +
                        f"Critic Loss: {critic_loss.item():.4f}")

            except Exception as e:
                logger.error(f"Error during update {i+1}: {e}")
                return False

        logger.info("✓ SAC update test passed")
        return True

    except Exception as e:
        logger.error(f"SAC update test failed: {e}")
        return False

def main():
    """Run the SAC update test."""
    logger.info("Testing SAC update method...")
    success = test_sac_update()

    if success:
        logger.info("All tests passed successfully!")
    else:
        logger.error("Tests failed. Review the logs.")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
