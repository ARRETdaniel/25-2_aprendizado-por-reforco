#!/usr/bin/env python
"""
Test the SimpleSAC implementation with a dummy environment.

This script creates a minimal test environment to verify that our SAC
implementation works correctly without requiring CARLA.
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path
import torch
from collections import namedtuple

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import SimpleSAC after path setup
from rl_environment.simple_sac import SimpleSAC

# Define simple space classes for the dummy environment
class Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

class Dict:
    def __init__(self, spaces):
        self.spaces = spaces

class DummyEnv:
    """
    A simple dummy environment that mimics the structure of the CARLA environment.
    """

    def __init__(self):
        """Initialize the dummy environment."""
        # Define action space
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Define observation space with the same structure as CARLA
        self.observation_space = Dict({
            'image': Box(
                low=0,
                high=255,
                shape=(84, 84, 3),
                dtype=np.uint8
            ),
            'vehicle_state': Box(
                low=-np.inf,
                high=np.inf,
                shape=(9,),
                dtype=np.float32
            ),
            'navigation': Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float32
            ),
            'detections': Box(
                low=-np.inf,
                high=np.inf,
                shape=(10,),
                dtype=np.float32
            )
        })        # Internal state
        self.step_count = 0
        self.max_steps = 100

    def reset(self):
        """Reset the environment."""
        self.step_count = 0

        # Generate random initial state
        state = {
            'image': np.random.randint(0, 255, size=(84, 84, 3), dtype=np.uint8),
            'vehicle_state': np.random.randn(9).astype(np.float32),
            'navigation': np.random.randn(3).astype(np.float32),
            'detections': np.random.randn(10).astype(np.float32)
        }

        return state

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: NumPy array of shape (3,)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.step_count += 1

        # Generate next state
        next_state = {
            'image': np.random.randint(0, 255, size=(84, 84, 3), dtype=np.uint8),
            'vehicle_state': np.random.randn(9).astype(np.float32),
            'navigation': np.random.randn(3).astype(np.float32),
            'detections': np.random.randn(10).astype(np.float32)
        }

        # Calculate reward based on action (dummy calculation)
        reward = -np.mean(np.square(action)) + 0.1

        # Check if episode is done
        done = self.step_count >= self.max_steps

        # Additional info
        info = {
            'is_success': False,
            'timeout': False,
            'crashed': False
        }

        return next_state, reward, done, info

    def close(self):
        """Close the environment."""
        pass

def test_sac_with_dummy_env(episodes=5, max_steps=100):
    """
    Test the SimpleSAC implementation with a dummy environment.

    Args:
        episodes: Number of episodes to run
        max_steps: Maximum steps per episode

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create dummy environment
        env = DummyEnv()

        # Get state dimensions
        state = env.reset()
        state_dims = {key: value.shape for key, value in state.items()}
        action_dim = env.action_space.shape[0]

        logger.info(f"State dimensions: {state_dims}")
        logger.info(f"Action dimension: {action_dim}")

        # Initialize agent
        agent = SimpleSAC(
            state_dims=state_dims,
            action_dim=action_dim,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            buffer_size=10000,
            batch_size=32,
            feature_dim=64,
            hidden_dim=64,
            update_freq=10,
            seed=42
        )

        logger.info("Training for {} episodes".format(episodes))

        # Training loop
        total_rewards = []

        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                # Select action
                action = agent.select_action(state)

                # Take step in environment
                next_state, reward, done, _ = env.step(action)

                # Process step in agent
                agent.step(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward

                if done:
                    break

            total_rewards.append(episode_reward)
            logger.info(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

            # Update networks after each episode
            if len(agent.memory) > agent.batch_size:
                losses = agent.update(batch_size=min(256, len(agent.memory)))
                logger.info(f"  Actor loss: {losses['actor_loss']:.4f}, Critic loss: {losses['critic_loss']:.4f}")

        # Test agent in evaluation mode
        state = env.reset()
        total_eval_reward = 0

        for step in range(max_steps):
            # Select action without exploration
            action = agent.select_action(state, evaluate=True)

            # Take step
            next_state, reward, done, _ = env.step(action)

            state = next_state
            total_eval_reward += reward

            if done:
                break

        logger.info(f"Evaluation reward: {total_eval_reward:.2f}")

        # Save agent
        os.makedirs('./test_checkpoints', exist_ok=True)
        agent.save('./test_checkpoints/dummy_test')

        logger.info("Test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error during test: {e}")
        return False

if __name__ == "__main__":
    # Set a more stable torch multiprocessing method
    torch.multiprocessing.set_start_method('spawn', force=True)

    # Run test
    success = test_sac_with_dummy_env()
    sys.exit(0 if success else 1)
